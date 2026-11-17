"""验证圆周相位、缺失标签掩码、分区隔离及训练/独立测试完整链路。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from demodulation.joint_mlp import (JointMLP, check_no_leakage, circular_error,
                                    fit_opd_stats, joint_loss)
from scripts.train_joint_mlp import evaluate, prepare, train


def example(prefix, n=12):
    rng = np.random.default_rng(22)
    angle = np.linspace(-2.8, 2.8, n)
    return dict(X=rng.normal(size=(n, 16)).astype(np.float32),
                opd_pseudo_m=np.linspace(.002, .006, n),
                phase_sin=np.sin(angle), phase_cos=np.cos(angle), phase_pseudo_rad=angle,
                opd_valid=np.ones(n, bool), phase_valid=np.ones(n, bool),
                sample_ids=np.array([f'{prefix}s{i}' for i in range(n)]),
                position_id=np.array([f'{prefix}p{i // 2}' for i in range(n)]),
                duplicate_group=np.array([f'{prefix}h{i}' for i in range(n)]),
                k_rad_per_m=np.linspace(1e6, 2e6, 16),
                reference_wavelength_nm=np.array(1560.), scheme=np.array('C'))


def test_phase_error_wraps_at_pi():
    np.testing.assert_allclose(circular_error(np.deg2rad(-179), np.deg2rad(179)),
                               np.deg2rad(2), atol=1e-12)


def test_invalid_nan_targets_do_not_poison_loss_or_gradients():
    opd = torch.tensor([0.5, 0.2], requires_grad=True)
    phase = torch.tensor([[0.1, 0.8], [0.2, 0.5]], requires_grad=True)
    batch = (torch.zeros(2, 16), torch.tensor([float('nan'), 0.4]),
             torch.tensor([[0., 1.], [float('nan'), float('nan')]]),
             torch.tensor([False, True]), torch.tensor([True, False]))
    loss, _, _ = joint_loss((opd, phase), batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(opd.grad).all() and torch.isfinite(phase.grad).all()
    assert opd.grad[0] == 0 and torch.count_nonzero(phase.grad[1]) == 0


def test_batch_without_any_valid_labels_has_zero_finite_gradient():
    opd, phase = torch.randn(3, requires_grad=True), torch.randn(3, 2, requires_grad=True)
    batch = (torch.zeros(3, 16), torch.full((3,), float('nan')),
             torch.full((3, 2), float('nan')), torch.zeros(3, dtype=torch.bool),
             torch.zeros(3, dtype=torch.bool))
    loss, _, _ = joint_loss((opd, phase), batch)
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(opd.grad) == torch.count_nonzero(phase.grad) == 0


def test_opd_scaler_excludes_invalid_targets():
    data = example('a', 3)
    data['opd_pseudo_m'] = np.array([.002, .004, np.nan])
    data['opd_valid'] = np.array([True, True, False])
    stats = fit_opd_stats(data)
    assert stats['mean_mm'] == 3 and stats['std_mm'] == 1


def test_cross_position_duplicate_is_rejected():
    left, right = example('a'), example('b')
    check_no_leakage(left, right)
    right['duplicate_group'][0] = left['duplicate_group'][0]
    with pytest.raises(ValueError, match='duplicate_group'):
        check_no_leakage(left, right)


def test_mlp_has_two_heads_without_convolution():
    model = JointMLP(16, (12, 8), 0.0)
    opd, phase = model(torch.randn(5, 16))
    assert opd.shape == (5,) and phase.shape == (5, 2)
    assert not any(isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d))
                   for layer in model.modules())


def test_training_without_test_file_and_separate_evaluation(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    for split, prefix in [('train', 'a'), ('validation', 'b')]:
        np.savez(data_dir / f'{split}.npz', **example(prefix))
    # 故意不提供test.npz：训练仍必须能完成，证明训练过程不依赖测试集。
    args = argparse.Namespace(data_dir=data_dir, out=tmp_path / 'run', hidden=[12, 8],
                              dropout=0.0, lr=.001, batch_size=6, epochs=2,
                              phase_weight=1., patience=3, seed=4,
                              device='cpu', threads=1, command='train')
    device = prepare(4, 'cpu', 1)
    train(args, device)
    report = json.loads((args.out / 'validation_metrics.json').read_text(encoding='utf-8'))
    assert report['epochs_run'] == 2 and report['opd_um_count'] == 12
    np.savez(data_dir / 'test.npz', **example('c'))
    eval_args = argparse.Namespace(data_dir=data_dir, checkpoint=args.out / 'best.pt',
                                   out=tmp_path / 'test_run')
    evaluate(eval_args, device)
    test_report = json.loads((eval_args.out / 'test_metrics.json').read_text(encoding='utf-8'))
    assert test_report['phase_rad_count'] == 12
