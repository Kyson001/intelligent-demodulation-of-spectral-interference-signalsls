"""手把手教学入口：训练/验证联合MLP；测试集必须显式执行evaluate命令。

训练：python scripts/train_joint_mlp.py train --epochs 100
评估：python scripts/train_joint_mlp.py evaluate --checkpoint analysis/joint_mlp/best.pt
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from demodulation.joint_mlp import (  # noqa: E402
    JointMLP, SpectralDataset, check_compatible, check_no_leakage, circular_error,
    fit_opd_stats, joint_loss, load_split, metrics, predict,
)


# 第六块：固定随机种子、选择设备。CPU也能跑，不要求安装CUDA。
def prepare(seed, device_name, threads):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('当前PyTorch不能使用CUDA，请选--device cpu或安装适合显卡的版本')
    # 同环境尽量可复现；不同硬件/版本仍可能有浮点差异。
    torch.use_deterministic_algorithms(True)
    return torch.device(device_name)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 第七块：保存逐条预测，便于以后绘图、检查异常和给网页读取。
def export_predictions(path, data, pred, stats):
    fields = ['sample_id', 'position_id', 'opd_valid', 'phase_valid',
              'opd_pseudo_m', 'opd_pred_m', 'opd_error_um',
              'phase_pseudo_rad', 'phase_pred_rad', 'phase_error_rad',
              'phase_vector_norm', 'phase_prediction_defined',
              'position_review', 'outside_training_opd_range']
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, sample_id in enumerate(data['sample_ids']):
            good_opd, good_phase = bool(data['opd_valid'][i]), bool(data['phase_valid'][i])
            defined = bool(pred['phase_norm'][i] > 1e-6)
            true_mm = float(data['opd_pseudo_m'][i] * 1e3)
            writer.writerow(dict(
                sample_id=str(sample_id), position_id=str(data['position_id'][i]),
                opd_valid=good_opd, phase_valid=good_phase,
                opd_pseudo_m=float(data['opd_pseudo_m'][i]) if good_opd else '',
                opd_pred_m=float(pred['opd_m'][i]),
                opd_error_um=float((pred['opd_m'][i] - data['opd_pseudo_m'][i]) * 1e6) if good_opd else '',
                phase_pseudo_rad=float(data['phase_pseudo_rad'][i]) if good_phase else '',
                phase_pred_rad=float(pred['phase_rad'][i]) if defined else '',
                phase_error_rad=float(circular_error(pred['phase_rad'][i], data['phase_pseudo_rad'][i])) if good_phase and defined else '',
                phase_vector_norm=float(pred['phase_norm'][i]),
                phase_prediction_defined=defined,
                position_review=bool(data['position_review'][i]) if 'position_review' in data else False,
                outside_training_opd_range=(true_mm < stats['min_mm'] or true_mm > stats['max_mm']) if good_opd else '',
            ))


# 第八块：读取训练/验证数据并建立网络。这里完全不读取test.npz。
def train(args, device, model_class=JointMLP, model_kwargs=None, model_type='mlp'):
    train_path, val_path = args.data_dir / 'train.npz', args.data_dir / 'validation.npz'
    train_data, val_data = load_split(train_path), load_split(val_path)
    check_no_leakage(train_data, val_data)
    for label, data in [('训练', train_data), ('验证', val_data)]:
        if not data['opd_valid'].any() or not data['phase_valid'].any():
            raise ValueError(f'{label}集必须有两个任务的有效标签')
    stats = fit_opd_stats(train_data)
    # 已有X做过逐谱RMS归一化，这里不再次拟合每个波长点的标准化。
    dataset = SpectralDataset(train_data, stats)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, generator=generator)
    # 默认仍是原来的MLP；LSTM入口传入另一模型类，共用完全相同的数据/损失/划分。
    model_args = (dict(input_dim=train_data['X'].shape[1], hidden=args.hidden, dropout=args.dropout)
                  if model_kwargs is None else model_kwargs)
    model = model_class(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    val_batch = tuple(value.to(device) for value in SpectralDataset(val_data, stats)[:])
    args.out.mkdir(parents=True, exist_ok=False)  # 避免意外覆盖之前的最佳模型。
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(opd_stats=stats, device_used=str(device), torch_version=str(torch.__version__),
                  numpy_version=np.__version__, python_version=sys.version,
                  train_sha256=file_sha(train_path), validation_sha256=file_sha(val_path),
                  model_args=model_args, model_type=model_type,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  train_samples=len(dataset), validation_samples=len(val_data['X']))
    write_json(args.out / 'config.json', config)
    print(f'设备={device}；参数量={sum(p.numel() for p in model.parameters()):,}；'
          f'训练={len(dataset)}；验证={len(val_data["X"])}', flush=True)
    history, best_loss, bad_epochs = [], float('inf'), 0
    start = time.perf_counter()

    # 第九块：训练循环。顺序是清梯度→前向→损失→反向→更新权重。
    for epoch in range(1, args.epochs + 1):
        model.train()  # 开启训练模式，Dropout开始工作。
        opd_sum, phase_sum, opd_count, phase_count = 0.0, 0.0, 0, 0
        for batch in loader:
            batch = tuple(value.to(device) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            total, opd_loss, phase_loss = joint_loss(model(batch[0]), batch, args.phase_weight)
            if not torch.isfinite(total):
                raise ValueError('训练损失出现NaN/Inf，请检查标签、掩码和学习率')
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            no, np_ = int(batch[3].sum()), int(batch[4].sum())
            opd_sum += opd_loss.item() * no
            phase_sum += phase_loss.item() * np_
            opd_count += no
            phase_count += np_
        train_opd = opd_sum / opd_count
        train_phase = phase_sum / phase_count

        # 第十块：验证不计算梯度、不更新参数。只按验证联合损失选最佳模型。
        model.eval()
        with torch.inference_mode():
            val_total, val_opd, val_phase = joint_loss(model(val_batch[0]), val_batch, args.phase_weight)
        val_loss = float(val_total)
        if not np.isfinite(val_loss):
            raise ValueError('验证损失不是有限数')
        row = dict(epoch=epoch, train_loss=train_opd + args.phase_weight * train_phase,
                   train_opd_loss=train_opd, train_phase_loss=train_phase,
                   validation_loss=val_loss, validation_opd_loss=float(val_opd),
                   validation_phase_loss=float(val_phase))
        history.append(row)
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            # 保存标准化参数、网格、位置和来源哈希，推理时不能重新拟合它们。
            torch.save(dict(
                model_state_dict=model.state_dict(), model_args=model_args, model_type=model_type, opd_stats=stats,
                epoch=epoch, validation_loss=best_loss, phase_weight=args.phase_weight,
                k_rad_per_m=torch.from_numpy(train_data['k_rad_per_m']),
                reference_wavelength_nm=float(train_data['reference_wavelength_nm']),
                train_positions=train_data['position_id'].tolist(),
                validation_positions=val_data['position_id'].tolist(),
                train_duplicates=train_data['duplicate_group'].tolist(),
                validation_duplicates=val_data['duplicate_group'].tolist(),
                train_sample_ids=train_data['sample_ids'].tolist(),
                validation_sample_ids=val_data['sample_ids'].tolist(),
                config=config,
            ), args.out / 'best.pt')
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0 or bad_epochs >= args.patience:
            print(f'Epoch {epoch:03d} | train={row["train_loss"]:.5f} | '
                  f'val={val_loss:.5f} | val_OPD={float(val_opd):.5f} | '
                  f'val_phase={float(val_phase):.5f}', flush=True)
        if bad_epochs >= args.patience:
            print(f'验证损失连续{args.patience}轮没有改善，早停。', flush=True)
            break

    # 第十一块：恢复最佳轮次，导出验证预测；默认仍不读取测试集。
    best = torch.load(args.out / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best['model_state_dict'])
    val_pred = predict(model, val_data, stats, device)
    report = metrics(val_data, val_pred, stats)
    # 训练均值基线用于判断网络是否学到了超过常量输出的信息。
    good = train_data['phase_valid']
    mean_phase = float(np.arctan2(np.mean(train_data['phase_sin'][good]),
                                 np.mean(train_data['phase_cos'][good])))
    baseline = dict(opd_m=np.full(len(val_data['X']), stats['mean_mm'] * 1e-3),
                    phase_rad=np.full(len(val_data['X']), mean_phase),
                    phase_norm=np.ones(len(val_data['X'])))
    report.update(best_epoch=int(best['epoch']), epochs_run=len(history),
                  training_seconds=time.perf_counter() - start,
                  constant_baseline=metrics(val_data, baseline, stats))
    write_json(args.out / 'validation_metrics.json', report)
    with (args.out / 'history.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    export_predictions(args.out / 'validation_predictions.csv', val_data, val_pred, stats)
    print(f'最佳轮次={best["epoch"]}；验证OPD MAE={report["opd_um_mae"]:.3f} μm；'
          f'相位圆周MAE={report["phase_rad_mae"]:.4f} rad', flush=True)
    print(f'结果保存到：{args.out.resolve()}；尚未读取测试集。', flush=True)


# 第十二块：模型和参数确定后，单独评价一次测试集，禁止用其选超参数。
def evaluate(args, device, model_class=JointMLP, expected_model_type='mlp'):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get('model_type', 'mlp') != expected_model_type:
        raise ValueError('检查点模型类型与当前入口不一致，请使用相应训练脚本')
    data = load_split(args.data_dir / 'test.npz')
    reference = dict(X=np.empty((0, checkpoint['model_args']['input_dim'])),
                     k_rad_per_m=checkpoint['k_rad_per_m'].cpu().numpy(),
                     reference_wavelength_nm=checkpoint['reference_wavelength_nm'])
    check_compatible(reference, data)
    for part in ('train', 'validation'):
        for key, stored in [('position_id', 'positions'), ('duplicate_group', 'duplicates'),
                            ('sample_ids', 'sample_ids')]:
            if set(data[key]) & set(checkpoint[f'{part}_{stored}']):
                raise ValueError(f'测试数据与{part}的{key}重叠，拒绝评价')
    model = model_class(**checkpoint['model_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    pred = predict(model, data, checkpoint['opd_stats'], device)
    report = metrics(data, pred, checkpoint['opd_stats'])
    report.update(checkpoint_sha256=file_sha(args.checkpoint),
                  test_sha256=file_sha(args.data_dir / 'test.npz'))
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'test_metrics.json', report)
    export_predictions(args.out / 'test_predictions.csv', data, pred, checkpoint['opd_stats'])
    print(f'测试OPD MAE={report["opd_um_mae"]:.3f} μm；'
          f'相位圆周MAE={report["phase_rad_mae"]:.4f} rad', flush=True)
    print(f'这只是伪标签一致性，结果：{args.out.resolve()}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'evaluate'):
        p = commands.add_parser(name)
        p.add_argument('--data-dir', type=Path, default=ROOT / 'datasets/processed')
        p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
        p.add_argument('--threads', type=int, default=4)
        p.add_argument('--seed', type=int, default=20261117)
        p.add_argument('--out', type=Path, default=ROOT / ('analysis/joint_mlp' if name == 'train' else 'analysis/joint_mlp_test'))
        if name == 'train':
            p.add_argument('--epochs', type=int, default=100)
            p.add_argument('--batch-size', type=int, default=64)
            p.add_argument('--lr', type=float, default=1e-3)
            p.add_argument('--hidden', type=int, nargs='+', default=[256, 128, 64])
            p.add_argument('--dropout', type=float, default=0.1)
            p.add_argument('--phase-weight', type=float, default=1.0)
            p.add_argument('--patience', type=int, default=20)
        else:
            p.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads必须为正整数')
    if args.command == 'train':
        if min(args.epochs, args.batch_size, args.patience, *args.hidden) < 1:
            parser.error('轮次、批量大小、早停轮数和隐藏层宽度必须为正整数')
        if not 0 <= args.dropout < 1 or args.lr <= 0 or args.phase_weight <= 0:
            parser.error('要求0≤dropout<1，lr>0，phase-weight>0')
    device = prepare(args.seed, args.device, args.threads)
    (train if args.command == 'train' else evaluate)(args, device)


if __name__ == '__main__':
    main()
