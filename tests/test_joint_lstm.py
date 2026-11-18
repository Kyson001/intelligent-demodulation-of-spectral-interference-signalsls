"""检查光谱序列维度、参考点、梯度和LSTM检查点，避免仅能启动却不能正确评价。"""
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
from demodulation.joint_lstm import JointLSTM, reference_sample_index
from scripts.train_joint_mlp import train, evaluate, prepare


@pytest.mark.parametrize('readout', ['last', 'center_global'])
def test_shape_gradients_independent_samples_and_no_convolution(readout):
    torch.set_num_threads(1)
    model = JointLSTM(32, 4, 8, 1, 0, 15.3, readout)
    x = torch.randn(3, 32)
    opd, phase = model(x)
    assert opd.shape == (3,) and phase.shape == (3, 2)
    (opd.square().mean() + phase.square().mean()).backward()
    assert model.lstm.weight_ih_l0.grad.abs().sum() > 0
    assert not any(isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)) for m in model.modules())
    with torch.inference_mode():
        single = model(x[1:2])
    torch.testing.assert_close(single[0], opd[1:2])
    torch.testing.assert_close(single[1], phase[1:2])


def test_reshape_preserves_every_sample_in_order():
    model = JointLSTM(32, 4, 8, reference_index=15)
    seen = []
    hook = model.lstm.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    x = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    model(x)
    hook.remove()
    assert seen[0].shape == (2, 8, 4)
    torch.testing.assert_close(seen[0].reshape(2, 32), x)


def test_reference_coordinate_and_bad_patch_size():
    grid = np.linspace(3.9e6, 4.2e6, 32)
    index = reference_sample_index(dict(k_rad_per_m=grid, reference_wavelength_nm=np.array(1560.)))
    assert 0 < index < 31
    with pytest.raises(ValueError, match='整除'):
        JointLSTM(32, 3)


def test_lstm_can_train_without_test_and_load_own_checkpoint(tmp_path):
    device = prepare(4, 'cpu', 1)
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    rng = np.random.default_rng(9)

    def data(prefix):
        phase = np.linspace(-2, 2, 8)
        return dict(X=rng.normal(size=(8, 32)).astype(np.float32),
                    opd_pseudo_m=np.linspace(.002, .006, 8),
                    phase_sin=np.sin(phase), phase_cos=np.cos(phase), phase_pseudo_rad=phase,
                    opd_valid=np.ones(8, bool), phase_valid=np.ones(8, bool),
                    sample_ids=np.array([f'{prefix}s{i}' for i in range(8)]),
                    position_id=np.array([f'{prefix}p{i}' for i in range(8)]),
                    duplicate_group=np.array([f'{prefix}h{i}' for i in range(8)]),
                    k_rad_per_m=np.linspace(3.9e6, 4.2e6, 32),
                    reference_wavelength_nm=np.array(1560.), scheme=np.array('C'))

    for split in ('train', 'validation'):
        np.savez(data_dir / f'{split}.npz', **data(split))
    args = argparse.Namespace(data_dir=data_dir, out=tmp_path / 'run', dropout=0.,
                              lr=.001, batch_size=4, epochs=2, phase_weight=1.,
                              patience=3, seed=4, command='train', device='cpu', threads=1)
    model_args = dict(input_dim=32, patch_size=4, hidden_size=8,
                      num_layers=1, dropout=0., reference_index=15.)
    train(args, device, JointLSTM, model_args, 'lstm')
    np.savez(data_dir / 'test.npz', **data('test'))
    args.checkpoint, args.out = args.out / 'best.pt', tmp_path / 'eval'
    evaluate(args, device, JointLSTM, 'lstm')
    report = json.loads((args.out / 'test_metrics.json').read_text(encoding='utf-8'))
    assert report['phase_rad_count'] == 8 and report['opd_um_count'] == 8
