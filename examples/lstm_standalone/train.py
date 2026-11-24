r"""独立LSTM训练入口：你只需要运行这个文件，不需要打开原项目。

快速检查：.\.venv\Scripts\python.exe -X utf8 train.py --epochs 2
正式训练：.\.venv\Scripts\python.exe -X utf8 train.py --epochs 100
文件分工：model.py定义整个网络；本文件读取数据、训练、验证并保存权重。
现有数据已执行预处理；标签来自传统算法，误差表示与伪标签的一致性。
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# 仅导入当前文件夹里的model.py，不导入原工程的任何代码。
from model import JointLSTM, reference_sample_index

# 根据train.py自身的位置找数据，因此从其他目录启动也能找到data文件夹。
ROOT = Path(__file__).resolve().parent


# =================================================================
# 一、读数据和检查分区（数据已预处理，不再加窗）
# =================================================================

def load_split(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    required = {
        'X', 'opd_pseudo_m', 'phase_sin', 'phase_cos', 'phase_pseudo_rad',
        'opd_valid', 'phase_valid', 'sample_ids', 'position_id',
        'duplicate_group', 'k_rad_per_m', 'reference_wavelength_nm', 'scheme',
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f'{path}缺少字段：{sorted(missing)}')
    x = data['X']
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('X必须是非空、有限值的[样本数, 光谱点数]数组')
    n, points = x.shape
    for key in required - {'X', 'k_rad_per_m', 'reference_wavelength_nm', 'scheme'}:
        if data[key].shape != (n,):
            raise ValueError(f'{key}长度与X不匹配')
    if str(data['scheme'].item()) != 'C':
        raise ValueError('本教学模型只接受已经执行预处理输入')
    k = data['k_rad_per_m']
    if k.shape != (points,) or not np.isfinite(k).all() or not np.all(np.diff(k) > 0):
        raise ValueError('波数坐标必须有限、递增且与输入维度相同')
    for task in ('opd', 'phase'):
        if data[f'{task}_valid'].dtype != np.bool_:
            raise ValueError('有效性标记必须为bool，不能将缺失标签当作0')
    opd_mask, phase_mask = data['opd_valid'], data['phase_valid']
    if not np.isfinite(data['opd_pseudo_m'][opd_mask]).all():
        raise ValueError('有效OPD标签存在NaN/Inf')
    if np.any(data['opd_pseudo_m'][opd_mask] <= 0):
        raise ValueError('当前正OPD支路要求有效OPD为正值')
    phase = np.stack([data['phase_sin'], data['phase_cos']], axis=1)[phase_mask]
    if not np.isfinite(phase).all() or not np.allclose(np.sum(phase**2, axis=1), 1, atol=1e-5):
        raise ValueError('有效相位的sin/cos必须有限且在单位圆上')
    angles = data['phase_pseudo_rad'][phase_mask]
    if not np.isfinite(angles).all() or not np.allclose(
        phase, np.stack([np.sin(angles), np.cos(angles)], axis=1), atol=1e-5
    ):
        raise ValueError('相位角与sin/cos标签不一致')
    return data


def check_compatible(reference, other):
    """同一模型的所有输入必须使用同一波数网格和相位参考波长。"""
    if reference['X'].shape[1] != other['X'].shape[1]:
        raise ValueError('输入维度不同')
    if not np.allclose(reference['k_rad_per_m'], other['k_rad_per_m'], rtol=0, atol=1e-6):
        raise ValueError('波数网格不同，不能直接使用当前模型')
    if float(reference['reference_wavelength_nm']) != float(other['reference_wavelength_nm']):
        raise ValueError('相位参考波长不同')


def check_no_leakage(*splits):
    """除了样本编号，还检查位置和精确重复谱是否跨分区。"""
    for left, right in combinations(splits, 2):
        check_compatible(left, right)
        for key in ('sample_ids', 'position_id', 'duplicate_group'):
            if set(left[key]) & set(right[key]):
                raise ValueError(f'{key}跨分区泄漏，请使用已有的位置划分')


# =================================================================
# 二、准备标签和批次：OPD标准化只使用训练集
# =================================================================

def fit_opd_stats(train):
    values_mm = train['opd_pseudo_m'][train['opd_valid']] * 1e3
    if len(values_mm) < 2 or np.std(values_mm) < 1e-8:
        raise ValueError('训练集有效OPD不足或没有变化，无法标准化')
    return {
        'mean_mm': float(np.mean(values_mm)), 'std_mm': float(np.std(values_mm)),
        'min_mm': float(np.min(values_mm)), 'max_mm': float(np.max(values_mm)),
    }


class SpectralDataset(Dataset):
    """每次返回一条光谱、标准化OPD、[sinφ, cosφ]和两个有效标记。"""
    def __init__(self, data, stats):
        self.x = torch.from_numpy(data['X'].astype(np.float32))
        y = (data['opd_pseudo_m'] * 1e3 - stats['mean_mm']) / stats['std_mm']
        self.opd = torch.from_numpy(y.astype(np.float32))
        self.phase = torch.from_numpy(np.stack(
            [data['phase_sin'], data['phase_cos']], axis=1
        ).astype(np.float32))
        self.opd_mask = torch.from_numpy(data['opd_valid'])
        self.phase_mask = torch.from_numpy(data['phase_valid'])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (self.x[index], self.opd[index], self.phase[index],
                self.opd_mask[index], self.phase_mask[index])


# =================================================================
# 三、定义联合损失：两个任务分别筛选有效标签
# =================================================================

def joint_loss(predictions, batch, phase_weight=1.0):
    opd_pred, phase_pred = predictions
    _, opd_true, phase_true, opd_mask, phase_mask = batch
    # 无有效标签时保留可反传的0，但绝不读取那些无效目标。
    opd_loss = opd_pred.sum() * 0
    phase_loss = phase_pred.sum() * 0
    if opd_mask.any():
        opd_loss = F.smooth_l1_loss(opd_pred[opd_mask], opd_true[opd_mask])
    if phase_mask.any():
        # sin/cos向量的MSE连续；单位向量之间，它等价于1-cos(相位差)。
        # 此处不强制单位化，MSE同时监督方向与分量大小，避免零向量归一化。
        phase_loss = F.mse_loss(phase_pred[phase_mask], phase_true[phase_mask])
    return opd_loss + phase_weight * phase_loss, opd_loss, phase_loss


# =================================================================
# 四、模型推理和误差评价：恢复OPD单位、使用相位圆周距离
# =================================================================

@torch.inference_mode()
def predict(model, data, stats, device, batch_size=256):
    model.eval()  # 关闭Dropout，推理结果才稳定。
    outputs = []
    for start in range(0, len(data['X']), batch_size):
        x = torch.as_tensor(data['X'][start:start + batch_size],
                            dtype=torch.float32, device=device)
        opd_z, phase_sc = model(x)
        outputs.append(torch.cat([opd_z[:, None], phase_sc], dim=1).cpu().numpy())
    output = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(output).all():
        raise ValueError('模型输出NaN/Inf，训练失败，不能报告正常误差')
    opd_m = (output[:, 0] * stats['std_mm'] + stats['mean_mm']) * 1e-3
    phase_norm = np.linalg.norm(output[:, 1:], axis=1)
    phase_rad = np.arctan2(output[:, 1], output[:, 2])
    return {'opd_m': opd_m, 'phase_rad': phase_rad, 'phase_norm': phase_norm}


def circular_error(pred, target):
    delta = np.asarray(pred) - np.asarray(target)
    return np.arctan2(np.sin(delta), np.cos(delta))


def metrics(data, predictions, stats):
    opd_mask = data['opd_valid']
    phase_defined = predictions['phase_norm'] > 1e-6
    phase_mask = data['phase_valid'] & phase_defined
    opd_error_um = (predictions['opd_m'] - data['opd_pseudo_m']) * 1e6
    phase_error = circular_error(predictions['phase_rad'], data['phase_pseudo_rad'])

    def summarize(mask, error, prefix):
        values = error[mask]
        if not len(values):
            return {f'{prefix}_count': 0, f'{prefix}_mae': None, f'{prefix}_rmse': None}
        per_position = [float(np.mean(np.abs(error[mask & (data['position_id'] == p)])))
                        for p in np.unique(data['position_id'][mask])]
        return {
            f'{prefix}_count': int(len(values)),
            f'{prefix}_mae': float(np.mean(np.abs(values))),
            f'{prefix}_rmse': float(np.sqrt(np.mean(values**2))),
            f'{prefix}_bias': float(np.mean(values)),
            f'{prefix}_p95_abs': float(np.percentile(np.abs(values), 95)),
            f'{prefix}_position_macro_mae': float(np.mean(per_position)),
        }

    result = summarize(opd_mask, opd_error_um, 'opd_um')
    result.update(summarize(phase_mask, phase_error, 'phase_rad'))
    result['phase_undefined_predictions'] = int(np.sum(data['phase_valid'] & ~phase_defined))
    result['label_status'] = 'agreement_with_algorithmic_pseudo_labels_only'
    # 这是依照真实标签范围的事后评价分组，不能用于推理时选择模型。
    true_mm = data['opd_pseudo_m'] * 1e3
    in_range = (true_mm >= stats['min_mm']) & (true_mm <= stats['max_mm'])
    for name, group in [('within_training_opd_range', in_range & opd_mask),
                        ('outside_training_opd_range', ~in_range & opd_mask)]:
        result[name] = summarize(opd_mask & group, opd_error_um, 'opd_um')
        result[name].update(summarize(phase_mask & group, phase_error, 'phase_rad'))
    return result


# =================================================================
# 五、固定随机种子、选择CPU/GPU、保存结果的辅助函数
# =================================================================

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


# =================================================================
# 六、完整训练流程：初学时优先阅读这个train函数
# =================================================================

def train(args, device):
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
    # 建立光谱块双向LSTM：2048点 -> 256个8点块 -> 全局/参考点特征 -> 双输出头。
    # 这些参数会一起写进best.pt，后面加载权重时可以重建完全相同的网络。
    model_args = dict(input_dim=train_data['X'].shape[1], patch_size=args.patch_size,
                      hidden_size=args.hidden_size, num_layers=args.num_layers,
                      dropout=args.dropout, reference_index=reference_sample_index(train_data),
                      readout='center_global')
    model = JointLSTM(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    val_batch = tuple(value.to(device) for value in SpectralDataset(val_data, stats)[:])
    args.out.mkdir(parents=True, exist_ok=False)  # 避免意外覆盖之前的最佳模型。
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(opd_stats=stats, device_used=str(device), torch_version=str(torch.__version__),
                  numpy_version=np.__version__, python_version=sys.version,
                  train_sha256=file_sha(train_path), validation_sha256=file_sha(val_path),
                  model_args=model_args, model_type='lstm',
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
                model_state_dict=model.state_dict(), model_args=model_args, model_type='lstm', opd_stats=stats,
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


# =================================================================
# 七、独立测试：仅指定--evaluate时读取test.npz
# =================================================================

def evaluate(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get('model_type') != 'lstm':
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
    model = JointLSTM(**checkpoint['model_args']).to(device)
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


# 八、运行入口：在这里定义可修改的训练参数。
def main():
    parser = argparse.ArgumentParser(description='独立LSTM光程差与相位联合解调工程')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data', help='三份NPZ所在的文件夹')
    parser.add_argument('--epochs', type=int, default=100, help='最多训练多少轮')
    parser.add_argument('--batch-size', type=int, default=64, help='每批使用多少条光谱')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率，即权重更新的步长设置')
    parser.add_argument('--patch-size', type=int, default=8, help='每个序列步包含的连续采样点数')
    parser.add_argument('--hidden-size', type=int, default=32, help='LSTM每个方向的隐藏特征维数')
    parser.add_argument('--num-layers', type=int, default=1, help='堆叠的LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='读出层的Dropout概率')
    parser.add_argument('--phase-weight', type=float, default=1.0, help='相位损失的权重')
    parser.add_argument('--patience', type=int, default=20, help='验证损失多少轮不改善就停止')
    parser.add_argument('--seed', type=int, default=20261117, help='固定随机种子，便于复现')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu', help='当前环境使用cpu即可')
    parser.add_argument('--threads', type=int, default=4, help='CPU计算线程数')
    parser.add_argument('--out', type=Path, help='结果保存目录；省略时自动创建带时间的目录')
    parser.add_argument('--evaluate', action='store_true', help='模型确定后才单独评价测试集')
    parser.add_argument('--checkpoint', type=Path, help='测试时要加载的best.pt路径')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patch_size, args.hidden_size,
           args.num_layers, args.patience, args.threads) < 1:
        parser.error('轮次、批量、块长、特征维数、层数、耐心轮数和线程数必须为正')
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.phase_weight <= 0:
        parser.error('要求0≤dropout<1，lr>0，phase-weight>0')
    if args.evaluate and args.checkpoint is None:
        parser.error('测试时还需要--checkpoint指定已训练好的best.pt')
    args.command = 'evaluate' if args.evaluate else 'train'
    if args.out is None:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        args.out = ROOT / 'outputs' / (args.command + '_' + stamp)
    # 准备环境只固定随机数与计算设置，不读取测试数据。
    device = prepare(args.seed, args.device, args.threads)
    print('工程位置：', ROOT, flush=True)
    print('本次输出：', args.out.resolve(), flush=True)
    if args.evaluate:
        evaluate(args, device)
    else:
        train(args, device)


if __name__ == '__main__':
    main()
