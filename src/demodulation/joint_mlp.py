"""教学用全连接联合解调网络：数据读取、网络、掩码损失与物理单位评价。

本文件没有卷积层。监督目标均为算法伪标签，评价表示与伪标签的一致性。
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


# 第一块：读取已有NPZ，保留质量标记和位置编号，不重新随机划分。
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
        raise ValueError('本教学模型只接受已经执行C方案的输入')
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


# 第二块：只用训练集的有效OPD求均值/标准差，防止验证/测试信息泄漏。
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


# 第三块：共享的MLP先提取特征，然后分别接OPD头和相位头。
class JointMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden=(256, 128, 64), dropout=0.1):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden:
            # Linear为全连接层；LayerNorm不依赖一个批次里有哪些采集位置。
            layers.extend([nn.Linear(previous, width), nn.LayerNorm(width),
                           nn.GELU(), nn.Dropout(dropout)])
            previous = width
        self.backbone = nn.Sequential(*layers)
        self.opd_head = nn.Linear(previous, 1)
        self.phase_head = nn.Linear(previous, 2)

    def forward(self, x):
        features = self.backbone(x)                # [B, 2048] -> [B, 64]
        opd_z = self.opd_head(features).squeeze(-1) # [B]，标准化OPD
        phase_sc = self.phase_head(features)       # [B, 2]，顺序sin、cos
        return opd_z, phase_sc


# 第四块：两个任务分别选有效标签，再计算损失。不要先算NaN误差再乘0！
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


# 第五块：还原物理单位，并用圆周距离评价相位。
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
