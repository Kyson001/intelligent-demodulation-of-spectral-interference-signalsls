"""单文件交付：CNN训练、预测、C预处理、鲁棒性评价。
直接运行本文件：使用已有第二版模型完成验证预测和鲁棒性评价，不重新训练。
可选命令 train / predict / robustness；python 本文件.py --help 查看参数。
依赖numpy、scipy、torch；不依赖本项目的任何其他.py文件。
"""
from __future__ import annotations


# ==================== 1. 通用CNN骨干、数据读取与指标 ====================
"""迈克尔逊光谱联合解调：可独立使用的 PyTorch 1D-CNN。

train 只读 train/validation；predict 无需标签。历史测试指标随结果文件提供。
输入必须是已经完成 C 预处理的 NPZ，不可直接输入原始 CSV。
监督标签是算法伪标签，指标不等于独立仪器测量精度。
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def core_write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def core_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def core_setup(seed=20261118, device='auto'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

def core_load_npz(path, labeled=True):
    """读取时拒绝 pickle；检查输入、坐标、C方案及有效标签。"""
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k].copy() for k in f.files}
    x, k = (d['X'], d['k_rad_per_m'])
    if x.ndim != 2 or x.shape[1] != 2048 or (not len(x)) or (not np.isfinite(x).all()):
        raise ValueError('X必须为非空、有限的[N,2048]数组')
    if k.shape != (2048,) or not np.isfinite(k).all() or (not np.all(np.diff(k) > 0)):
        raise ValueError('波数坐标必须有限、递增且为2048点')
    if str(d['scheme'].item()) != 'C' or float(d['reference_wavelength_nm']) != 1560:
        raise ValueError('要求C方案及1560nm相位参考')
    if labeled:
        for task, target in [('opd', 'opd_pseudo_m'), ('phase', 'phase_pseudo_rad')]:
            mask = d[f'{task}_valid']
            if mask.shape != (len(x),) or mask.dtype != np.bool_ or (not mask.any()):
                raise ValueError(f'{task}有效掩码不正确或没有有效目标')
            if not np.isfinite(d[target][mask]).all():
                raise ValueError(f'{task}有效目标含NaN/Inf')
        good = d['phase_valid']
        sc = np.stack([d['phase_sin'], d['phase_cos']], -1)[good]
        if not np.allclose(np.sum(sc ** 2, -1), 1, atol=1e-06):
            raise ValueError('有效相位标签不在单位圆上')
        if not np.allclose(sc, np.stack([np.sin(d['phase_pseudo_rad'][good]), np.cos(d['phase_pseudo_rad'][good])], -1)):
            raise ValueError('角度与sin/cos不一致')
    return d

def core_compatible(a, b):
    np.testing.assert_allclose(a['k_rad_per_m'], b['k_rad_per_m'], rtol=0, atol=1e-06)
    if float(a['reference_wavelength_nm']) != float(b['reference_wavelength_nm']):
        raise ValueError('相位参考不同')

def core_no_leakage(a, b):
    core_compatible(a, b)
    for key in ['sample_ids', 'position_id', 'duplicate_group']:
        if set(a[key]) & set(b[key]):
            raise ValueError(f'{key}跨分区泄漏')

class core_JointCNN(nn.Module):
    """不作FFT、不输入标签/文件编号；仅从C方案光谱学习。

    三层不降采样的卷积感受野为63点。OPD头汇总全谱的频率特征；
    相位头保留1560nm附近128点的位置，以免全局池化抹掉相位。
    """

    def __init__(self, center_index=991, width=16, dropout=0.1):
        super().__init__()
        self.center_index = int(center_index)
        self.width = width
        self.dropout = dropout
        channels = [1, width, width * 2, width * 2]
        layers = []
        for i, (kernel, dilation) in enumerate([(15, 1), (9, 2), (9, 4)]):
            layers.extend([nn.Conv1d(channels[i], channels[i + 1], kernel, padding=dilation * (kernel // 2), dilation=dilation), nn.GroupNorm(4, channels[i + 1]), nn.SiLU()])
        self.features = nn.Sequential(*layers)
        self.opd_head = nn.Sequential(nn.Linear(width * 4, 64), nn.SiLU(), nn.Linear(64, 1))
        self.phase_head = nn.Sequential(nn.Linear(width * 2 * 128 + width * 4, 128), nn.SiLU(), nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, x):
        if x.ndim == 2:
            x = x[:, None, :]
        f = self.features(x)
        mean = f.mean(-1)
        rms = f.square().mean(-1).clamp_min(1e-10).sqrt()
        global_feature = torch.cat([mean, rms], -1)
        local = f[:, :, self.center_index - 64:self.center_index + 64].flatten(1)
        z = self.opd_head(global_feature).squeeze(-1)
        sc = self.phase_head(torch.cat([local, global_feature], -1))
        return (z, sc)

def core_tensors(d, stats, device):
    x = torch.as_tensor(d['X'], dtype=torch.float32, device=device)
    z = (d['opd_pseudo_m'] * 1000.0 - stats['mean_mm']) / stats['std_mm']
    sc = np.stack([d['phase_sin'], d['phase_cos']], -1)
    return (x, torch.as_tensor(z, dtype=torch.float32, device=device), torch.as_tensor(sc, dtype=torch.float32, device=device), torch.as_tensor(d['opd_valid'], device=device), torch.as_tensor(d['phase_valid'], device=device))

def core_loss_terms(output, batch):
    """先按各任务mask索引，再算损失；避免NaN乘0污染梯度。"""
    z, sc = output
    _, target, phase, om, pm = batch
    lo = F.smooth_l1_loss(z[om], target[om]) if om.any() else z.sum() * 0
    lp = F.mse_loss(sc[pm], phase[pm]) if pm.any() else sc.sum() * 0
    return (lo, lp)

@torch.inference_mode()
def core_infer(model, x, device, batch_size=128):
    model.eval()
    outputs = []
    for i in range(0, len(x), batch_size):
        z, sc = model(torch.as_tensor(x[i:i + batch_size], device=device, dtype=torch.float32))
        outputs.append(torch.cat([z[:, None], sc], -1).cpu().numpy())
    result = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError('模型产生非有限输出')
    return result

def core_physical(raw, stats):
    return dict(opd_m=(raw[:, 0] * stats['std_mm'] + stats['mean_mm']) * 0.001, phase_rad=np.arctan2(raw[:, 1], raw[:, 2]), phase_norm=np.linalg.norm(raw[:, 1:], axis=-1))

def core_circular_error(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))

def core_summarize_errors(error, mask, positions):
    values = error[mask]
    if not len(values):
        return {'count': 0, 'mae': None, 'rmse': None}
    return dict(count=int(len(values)), mae=float(np.abs(values).mean()), rmse=float(np.sqrt(np.mean(values ** 2))), bias=float(values.mean()), p95_abs=float(np.percentile(np.abs(values), 95)), max_abs=float(np.abs(values).max()), position_macro_mae=float(np.mean([np.abs(error[mask & (positions == p)]).mean() for p in np.unique(positions[mask])])))

def core_metrics(d, pred, stats):
    oe = (pred['opd_m'] - d['opd_pseudo_m']) * 1000000.0
    pe = core_circular_error(pred['phase_rad'], d['phase_pseudo_rad'])
    om = d['opd_valid']
    pm = d['phase_valid'] & (pred['phase_norm'] > 1e-06)
    inside = (d['opd_pseudo_m'] * 1000.0 >= stats['min_mm']) & (d['opd_pseudo_m'] * 1000.0 <= stats['max_mm'])
    out = dict(label_status='agreement_with_algorithmic_pseudo_labels_only', opd_um=core_summarize_errors(oe, om, d['position_id']), phase_rad=core_summarize_errors(pe, pm, d['position_id']), phase_undefined=int(np.sum(d['phase_valid'] & ~pm)))
    for name, group in [('within_training_range', inside), ('outside_training_range', ~inside)]:
        out[name] = dict(opd_um=core_summarize_errors(oe, om & group, d['position_id']), phase_rad=core_summarize_errors(pe, pm & group, d['position_id']))
    return out

def core_predictions_csv(path, d, p, labeled=True):
    fields = ['sample_id', 'position_id', 'opd_pred_m', 'phase_pred_rad', 'phase_vector_norm']
    if labeled:
        fields += ['opd_valid', 'phase_valid', 'opd_pseudo_m', 'phase_pseudo_rad', 'opd_error_um', 'phase_error_rad', 'position_review']
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for i in range(len(d['X'])):
            defined = p['phase_norm'][i] > 1e-06
            row = dict(sample_id=str(d['sample_ids'][i]) if 'sample_ids' in d else str(i), position_id=str(d['position_id'][i]) if 'position_id' in d else '', opd_pred_m=float(p['opd_m'][i]), phase_pred_rad=float(p['phase_rad'][i]) if defined else '', phase_vector_norm=float(p['phase_norm'][i]))
            if labeled:
                om, pm = (bool(d['opd_valid'][i]), bool(d['phase_valid'][i]))
                row.update(opd_valid=om, phase_valid=pm, opd_pseudo_m=float(d['opd_pseudo_m'][i]) if om else '', phase_pseudo_rad=float(d['phase_pseudo_rad'][i]) if pm else '', opd_error_um=float((p['opd_m'][i] - d['opd_pseudo_m'][i]) * 1000000.0) if om else '', phase_error_rad=float(core_circular_error(p['phase_rad'][i], d['phase_pseudo_rad'][i])) if pm and defined else '', position_review=bool(d.get('position_review', np.zeros(len(d['X']), bool))[i]))
            writer.writerow(row)

def core_load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('architecture') != 'JointCNN':
        raise ValueError('未知检查点结构')
    model = core_JointCNN(**ck['model_args']).to(device)
    model.load_state_dict(ck['state_dict'], strict=True)
    model.eval()
    return (model, ck)

def core_check_input(ck, data):
    np.testing.assert_allclose(ck['k_rad_per_m'].numpy(), data['k_rad_per_m'], rtol=0, atol=1e-06)
    if ck['scheme'] != str(data['scheme'].item()) or ck['reference_wavelength_nm'] != float(data['reference_wavelength_nm']):
        raise ValueError('检查点与输入预处理/参考波长不匹配')


# ==================== 2. C方案预处理 ====================
"""Reproducible, label-free preprocessing. No thermal compensation is inferred.

Intensity values are retained in their supplied units. Confirm linear units before
interpreting the background division physically. Diagnostics are not ground truth.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly, windows

@dataclass(frozen=True)
class pp_Config:
    reference_nm: float = 1560.0
    background_sigma_points: float = 100.0
    envelope_floor_fraction: float = 0.08
    fit_envelope_fraction: float = 0.15
    min_opd_mm: float = 0.5
    max_opd_mm: float = 15.0
    sideband_half_width_bins: int = 12
    model_points: int = 2048

    def to_dict(self):
        return asdict(self)

def pp_wrap_phase(x):
    return np.arctan2(np.sin(x), np.cos(x))

def pp_uniform_k(wavelength_nm, intensity):
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if wavelength_nm.ndim != 1 or wavelength_nm.shape != intensity.shape:
        raise ValueError('Expected matching 1-D wavelength and intensity arrays')
    if len(intensity) < 128 or not np.isfinite(wavelength_nm).all() or (not np.isfinite(intensity).all()):
        raise ValueError('Insufficient points or non-finite data')
    if np.any(wavelength_nm <= 0):
        raise ValueError('Wavelengths must be positive')
    k = 2 * np.pi / (wavelength_nm * 1e-09)
    order = np.argsort(k)
    k, intensity = (k[order], intensity[order])
    if np.any(np.diff(k) <= 0):
        raise ValueError('Duplicate wavelengths are not allowed')
    ku = np.linspace(k[0], k[-1], len(k))
    return (ku, np.interp(ku, k, intensity))

def pp_prepare(k, y, config=pp_Config()):
    """A: constant DC removal; B: guarded envelope division; C: B times Hann.

    The slowly varying intensity is an A(k) proxy, NOT an independently known
    visibility B(k). Dividing by it flattens the source envelope only under a
    slowly varying visibility model. All outputs retain the original k origin.
    """
    dk = np.diff(k)
    if not np.allclose(dk, dk[0], rtol=1e-08, atol=1e-09):
        raise ValueError('prepare requires an equispaced k grid')
    background = gaussian_filter1d(y, config.background_sigma_points, mode='reflect')
    peak = float(np.max(background))
    if peak <= 0:
        raise ValueError('Positive envelope required for this comparison')
    floor = config.envelope_floor_fraction * peak
    denominator = np.maximum(background, floor)
    corrected = (y - background) / denominator
    hann = windows.hann(len(y), sym=False)
    schemes = {'A': y - y.mean(), 'B': corrected, 'C': corrected * hann}
    support = (background >= config.fit_envelope_fraction * peak) & (hann >= 0.2)
    if support.sum() < 100:
        raise ValueError('Insufficient illuminated support')
    weights = (background / peak) ** 2
    return (schemes, dict(background=background, denominator=denominator, hann=hann, support=support, weights=weights))

def pp_fft_peak(k, signal, config=pp_Config()):
    n = len(k)
    amplitude = np.abs(np.fft.rfft(signal))
    opd_axis = 2 * np.pi * np.fft.rfftfreq(n, d=k[1] - k[0])
    valid = (opd_axis >= config.min_opd_mm * 0.001) & (opd_axis <= config.max_opd_mm * 0.001)
    if not np.any(valid):
        raise ValueError('OPD search range outside sampling grid')
    indices = np.flatnonzero(valid)
    j = int(indices[np.argmax(amplitude[valid])])
    shift = 0.0
    if 0 < j < len(amplitude) - 1:
        a, b, c = amplitude[j - 1:j + 2]
        denominator = a - 2 * b + c
        if denominator < 0:
            shift = float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))
    return ((j + shift) * (opd_axis[1] - opd_axis[0]), j, opd_axis, amplitude)

def pp_extract_phase(k, signal, support, weights, carrier_bin, config=pp_Config()):
    n = len(k)
    indices = np.arange(n)
    distance = np.abs(indices - carrier_bin)
    width = config.sideband_half_width_bins
    mask = np.zeros(n)
    mask[distance <= width / 2] = 1
    taper = (distance > width / 2) & (distance < width)
    mask[taper] = 0.5 * (1 + np.cos(np.pi * (distance[taper] - width / 2) / (width / 2)))
    mask[indices == 0] = 0
    mask[indices >= (n + 1) // 2] = 0
    analytic = 2 * np.fft.ifft(np.fft.fft(signal) * mask)
    carrier_opd = carrier_bin * 2 * np.pi / (n * (k[1] - k[0]))
    phase = np.unwrap(np.angle(analytic * np.exp(-1j * (k - k[0]) * carrier_opd)))
    kc = 2 * np.pi / (config.reference_nm * 1e-09)
    x = (k - kc) * 0.001
    design = np.column_stack([x, np.ones(n)])
    w = np.sqrt(weights[support])
    coef = np.linalg.lstsq(design[support] * w[:, None], phase[support] * w, rcond=None)[0]
    opd = carrier_opd + coef[0] * 0.001
    phi = pp_wrap_phase(coef[1] + carrier_opd * (kc - k[0]))
    residual = phase - design @ coef
    residual_rms = np.sqrt(np.average(residual[support] ** 2, weights=weights[support]))
    phase_centered = (k - kc) * opd + float(phi) + residual
    return dict(opd_m=float(opd), phase_rad=float(phi), residual_rms_rad=float(residual_rms), phase_curve=phase_centered, residual=residual, amplitude=np.abs(analytic))

def pp_compare(wavelength_nm, intensity, config=pp_Config()):
    k, y = pp_uniform_k(wavelength_nm, intensity)
    signals, aux = pp_prepare(k, y, config)
    _, carrier, _, _ = pp_fft_peak(k, signals['A'] * aux['hann'], config)
    outputs = {}
    for name, signal in signals.items():
        opd, peak, axis, amp = pp_fft_peak(k, signal, config)
        phase = pp_extract_phase(k, signal, aux['support'], aux['weights'], carrier, config)
        search = (axis >= config.min_opd_mm * 0.001) & (axis <= config.max_opd_mm * 0.001)
        near = np.abs(np.arange(len(amp)) - carrier) <= config.sideband_half_width_bins
        concentration = np.sum(amp[near] ** 2) / max(np.sum(amp[search] ** 2), 1e-30)
        half = amp[peak] / 2
        left, right = (peak, peak)
        while left > 0 and amp[left] > half:
            left -= 1
        while right < len(amp) - 1 and amp[right] > half:
            right += 1

        def cross(i, j):
            return i + (half - amp[i]) / (amp[j] - amp[i]) * (j - i)
        fwhm = (cross(right - 1, right) - cross(left, left + 1)) * (axis[1] - axis[0])
        outputs[name] = dict(fft_opd_m=float(opd), phase_opd_m=phase['opd_m'], phase_rad=phase['phase_rad'], phase_residual_mrad=phase['residual_rms_rad'] * 1000.0, fwhm_um=float(fwhm * 1000000.0), band_energy_fraction=float(concentration), signal=signal, fft_axis=axis, fft_amplitude=amp, **{'phase': phase})
    return dict(k=k, raw_k=y, aux=aux, methods=outputs)

def pp_model_input(wavelength_nm, intensity, scheme='A', config=pp_Config()):
    """训练/推理共用入口：A/B/C预处理、抗混叠重采样、逐谱RMS归一化。

    C在密波数网格上先补偿包络，再乘Hann窗。这里不生成距离或相位真值。
    返回真实重采样坐标；多相重采样一般不会同时保留原网格的两个端点。
    默认A保留旧接口兼容性；正式导出入口读取训练配置指定的方案。
    """
    if scheme not in ('A', 'B', 'C'):
        raise ValueError('Unknown preprocessing scheme')
    k, y = pp_uniform_k(wavelength_nm, intensity)
    if scheme == 'A':
        signal = y - y.mean()
    else:
        signals, _ = pp_prepare(k, y, config)
        signal = signals[scheme]
    nout = config.model_points
    out = resample_poly(signal, nout, len(k), padtype='line')
    ko = k[0] + np.arange(len(out)) * (k[1] - k[0]) * len(k) / nout
    centered = out - out.mean()
    scale = float(np.sqrt(np.mean(centered ** 2)))
    if scale < 1e-15:
        raise ValueError('Constant or nearly zero spectrum')
    return ((centered / scale).astype(np.float32), ko, dict(mean=float(out.mean()), rms=scale, scheme=scheme))


# ==================== 3. 第二版CNN与训练循环 ====================
"""CNN改进1-4：任务平衡、多尺度、训练内校准诊断、三种子稳定性。

本节包含第二版结构与训练循环，全部定义已合并在本文件中。
运行方式：python cnn/CNN完整程序.py --mode train；不修改或覆盖初版。
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import time
import shutil
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
HERE = Path(__file__).resolve().parent
SEEDS = [20261118, 20261119, 20261120]

def subset(d, mask):
    n = len(d['X'])
    static = {'k_rad_per_m', 'reference_wavelength_nm', 'scheme', 'label_status'}
    return {k: v[mask].copy() if k not in static and v.ndim > 0 and (len(v) == n) else v.copy() for k, v in d.items()}

def internal_split(d, seed=20261118):
    """按重复绑定组件留出约20%训练位置，不拆分位置和完全重复谱。"""
    components = d['leakage_component']
    groups = np.unique(components)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    chosen = []
    count = 0
    for group in groups:
        chosen.append(group)
        count += int(np.sum(components == group))
        if count >= 0.2 * len(components):
            break
    mask = np.isin(components, chosen)
    fit, hold = (subset(d, ~mask), subset(d, mask))
    core_no_leakage(fit, hold)
    return (fit, hold)

class WideResidual(nn.Module):
    """保留局部63点支路；两层深度卷积增加128点感受野，达到191。"""

    def __init__(self, channels, kind):
        super().__init__()
        kernel, dilation = (17, 4) if kind == 'dilated' else (65, 1)
        self.net = nn.Sequential(nn.Conv1d(channels, channels, kernel, padding=dilation * (kernel // 2), dilation=dilation, groups=channels), nn.SiLU(), nn.Conv1d(channels, channels, kernel, padding=dilation * (kernel // 2), dilation=dilation, groups=channels), nn.Conv1d(channels, channels, 1), nn.GroupNorm(4, channels), nn.SiLU())
        self.fuse = nn.Conv1d(channels * 2, channels, 1)

    def forward(self, f):
        wide = f + self.net(f)
        return self.fuse(torch.cat([f, wide], 1))

class ImprovedCNN(core_JointCNN):

    def __init__(self, center_index=991, width=16, dropout=0.1, variant='baseline', segments=1):
        super().__init__(center_index, width, dropout)
        self.variant = variant
        self.segments = segments
        self.wide = WideResidual(width * 2, variant) if variant != 'baseline' else nn.Identity()
        if segments != 1:
            self.opd_head = nn.Sequential(nn.Linear(width * 4 * segments, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        if x.ndim == 2:
            x = x[:, None, :]
        local_features = self.features(x)
        f = self.wide(local_features)
        g = torch.cat([local_features.mean(-1), local_features.square().mean(-1).clamp_min(1e-10).sqrt()], -1)
        chunks = f.reshape(len(f), f.shape[1], self.segments, -1)
        summary = torch.cat([chunks.mean(-1), chunks.square().mean(-1).clamp_min(1e-10).sqrt()], 1).flatten(1)
        z = self.opd_head(summary).squeeze(-1)
        local = local_features[:, :, self.center_index - 64:self.center_index + 64].flatten(1)
        sc = self.phase_head(torch.cat([local, g], -1))
        return (z, sc)

def score(m, task='joint'):
    """预先固定的工程折中尺度，非测量精度承诺。使用位置宏平均。"""
    o = m['opd_um']['position_macro_mae']
    p = m['phase_rad']['position_macro_mae']
    if task == 'opd':
        return o / 5.0
    if task == 'phase':
        return p / 0.03
    return o / 5.0 + p / 0.03

def make_stats(tr):
    values = tr['opd_pseudo_m'][tr['opd_valid']] * 1000.0
    return dict(mean_mm=float(values.mean()), std_mm=float(values.std()), min_mm=float(values.min()), max_mm=float(values.max()))

def diagnostic_gradients(model, batch, alpha):
    """同一固定批、eval模式，记录共享卷积层各任务梯度范数及夹角。"""
    model.eval()
    lo, lp = core_loss_terms(model(batch[0]), batch)
    parameters = list(model.features.parameters())
    a = torch.autograd.grad(lo, parameters, retain_graph=True)
    b = torch.autograd.grad(lp, parameters)
    a = torch.cat([x.flatten() for x in a])
    b = torch.cat([x.flatten() for x in b])
    na, nb = (a.norm(), b.norm())
    return dict(opd_norm=float(na), phase_norm=float(nb), weighted_opd_norm=float(alpha * na), cosine=float(torch.dot(a, b) / (na * nb).clamp_min(1e-20)))

def run(tr, va, out, *, seed=20261118, alpha=1.0, task='joint', variant='baseline', segments=1, epochs=120, patience=30, lr=0.001, batch_size=128):
    if (out / 'summary.json').exists():
        return json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=False)
    device = core_setup(seed, 'auto')
    core_no_leakage(tr, va)
    stats = make_stats(tr)
    center = int(np.argmin(np.abs(tr['k_rad_per_m'] - 2 * np.pi / 1.56e-06)))
    model_args = dict(center_index=center, width=16, dropout=0.1, variant=variant, segments=segments)
    model = ImprovedCNN(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
    tb = core_tensors(tr, stats, device)
    indices = torch.as_tensor(np.linspace(0, len(tr['X']) - 1, 128, dtype=int), device=device)
    diagnostic_batch = tuple((t[indices] for t in tb))
    config = dict(seed=seed, alpha=alpha, task=task, model_args=model_args, epochs=epochs, patience=patience, lr=lr, batch_size=batch_size, stats=stats, train_samples=len(tr['X']), validation_samples=len(va['X']), parameters=sum((p.numel() for p in model.parameters())), selection='position_macro_OPD_MAE/5um + position_macro_phase_MAE/0.03rad', torch=str(torch.__version__), device=str(device))
    core_write_json(out / 'config.json', config)
    core_write_json(out / 'sample_ids.json', dict(train=tr['sample_ids'].tolist(), validation=va['sample_ids'].tolist()))
    history = []
    best = float('inf')
    best_epoch = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        sums = np.zeros(4)
        for ids in torch.randperm(len(tr['X']), device=device).split(batch_size):
            batch = tuple((t[ids] for t in tb))
            optimizer.zero_grad(set_to_none=True)
            lo, lp = core_loss_terms(model(batch[0]), batch)
            loss = lo if task == 'opd' else lp if task == 'phase' else alpha * lo + lp
            if not torch.isfinite(loss):
                raise ValueError('non-finite loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            no, np_ = (int(batch[3].sum()), int(batch[4].sum()))
            sums += [lo.item() * no, no, lp.item() * np_, np_]
        raw = core_infer(model, va['X'], device)
        pred = core_physical(raw, stats)
        metric = core_metrics(va, pred, stats)
        selection = score(metric, task)
        if not np.isfinite(selection):
            raise ValueError('non-finite validation score')
        scheduler.step(selection)
        row = dict(epoch=epoch, train_opd=sums[0] / sums[1], train_phase=sums[2] / sums[3], opd_mae_um=metric['opd_um']['mae'], opd_p95_um=metric['opd_um']['p95_abs'], phase_mae_rad=metric['phase_rad']['mae'], score=selection, lr=optimizer.param_groups[0]['lr'], elapsed_s=time.perf_counter() - start)
        if epoch in [1, 10, 30, 60, 100, 120, 160]:
            row.update(diagnostic_gradients(model, diagnostic_batch, alpha))
        history.append(row)
        if selection < best:
            best, best_epoch = (selection, epoch)
            torch.save(dict(format_version=2, architecture='ImprovedCNN', model_args=model_args, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, stats=stats, k_rad_per_m=torch.from_numpy(tr['k_rad_per_m'].copy()), scheme='C', reference_wavelength_nm=1560.0, seed=seed, alpha=alpha, task=task, best_epoch=epoch, selection_score=best, label_status='algorithmic_pseudo_labels_not_independent_truth'), out / 'best.pt')
        if epoch == 1 or epoch % 10 == 0:
            print(f'{out.name} epoch={epoch} OPD={row['opd_mae_um']:.3f}um phase={row['phase_mae_rad']:.5f} score={selection:.4f} best={best_epoch} time={row['elapsed_s']:.0f}s', flush=True)
            core_write_json(out / 'history.json', history)
        if epoch - best_epoch >= patience:
            break
    core_write_json(out / 'history.json', history)
    keys = list(dict.fromkeys((k for r in history for k in r)))
    with (out / 'history.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(history)
    model, ck = load_model(out / 'best.pt', device)
    final_metrics = {}
    for name, d in [('train', tr), ('validation', va)]:
        pred = core_physical(core_infer(model, d['X'], device), stats)
        m = core_metrics(d, pred, stats)
        core_write_json(out / f'{name}_metrics.json', m)
        core_predictions_csv(out / f'{name}_predictions.csv', d, pred)
        final_metrics[name] = m
    grad = diagnostic_gradients(model, diagnostic_batch, alpha)
    core_write_json(out / 'best_gradients.json', grad)
    result = dict(config=config, best_epoch=best_epoch, epochs_run=epoch, selection_score=best, metrics=final_metrics, duration_s=time.perf_counter() - start, checkpoint_sha256=core_sha256(out / 'best.pt'), gradients=grad)
    core_write_json(out / 'summary.json', result)
    return result

def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('architecture') == 'JointCNN':
        return core_load_model(path, device)
    if ck.get('architecture') != 'ImprovedCNN':
        raise ValueError('unsupported model')
    model = ImprovedCNN(**ck['model_args']).to(device)
    model.load_state_dict(ck['state_dict'], strict=True)
    model.eval()
    return (model, ck)


# ==================== 4. 原始强度域鲁棒性评价 ====================
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
AUDIT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_OUT = Path(__file__).resolve().parent / 'results'
INITIAL_SHA256 = '61AE9F652EB5937614C76A0FA37A52FBE5D6EF29288B4CFA401A0A5D3CDA3F20'
SECOND_SHA256 = '331AF8583AC668BC718A439E9D45C88CBE339B8D99440E177458B022DB81E1E5'

def audit_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest().upper()

def audit_circular_error(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))

def audit_drift_metrics(clean, perturbed):
    opd_delta = (perturbed['opd_m'] - clean['opd_m']) * 1000000.0
    phase_delta = audit_circular_error(perturbed['phase_rad'], clean['phase_rad'])
    return {'opd_drift_mae_um': float(np.mean(np.abs(opd_delta))), 'opd_drift_p95_um': float(np.percentile(np.abs(opd_delta), 95)), 'opd_drift_bias_um': float(np.mean(opd_delta)), 'phase_drift_circular_mae_rad': float(np.mean(np.abs(phase_delta))), 'phase_drift_circular_p95_rad': float(np.percentile(np.abs(phase_delta), 95)), 'undefined_phase_vectors': int(np.sum(perturbed['phase_norm'] <= 1e-06))}

def audit_perturb(raw, kind, level, rng, sign=1):
    wavelength, intensity = raw
    y = intensity.astype(np.float64, copy=True)
    if kind == 'clean':
        return (wavelength, y)
    if kind == 'noise':
        background = gaussian_filter1d(y, 100.0, mode='reflect')
        sigma = level * float(background.max())
        y += rng.normal(0.0, sigma, size=y.shape)
    else:
        t = np.linspace(-0.5, 0.5, len(y), dtype=np.float64)
        if kind == 'tilt':
            factor = np.exp(sign * np.log1p(level) * t)
        elif kind == 'curvature':
            shape = 4.0 * t ** 2
            shape -= shape.mean()
            factor = np.exp(sign * np.log1p(level) * shape)
        else:
            raise ValueError(kind)
        y *= factor
    return (wavelength, y)

def audit_make_input(raw, expected=None):
    wavelength, intensity = raw
    x, k, _ = pp_model_input(wavelength, intensity, 'C', pp_Config())
    if expected is not None:
        np.testing.assert_array_equal(x, expected)
    return x

def audit_predict(model, ck, d, X, device):
    changed = dict(d)
    changed['X'] = X
    core_check_input(ck, changed)
    raw = core_infer(model, X, device)
    return (core_physical(raw, ck['stats']), core_metrics(changed, core_physical(raw, ck['stats']), ck['stats']))

def audit_run_robustness(project_root, out):
    global AUDIT_OUT, AUDIT_ROOT
    AUDIT_OUT = Path(out).resolve()
    AUDIT_ROOT = Path(project_root).resolve()
    data_dir = AUDIT_ROOT / 'datasets'
    validation_path = data_dir / 'processed' / 'validation.npz'
    d = core_load_npz(validation_path)
    ids = d['sample_ids'].astype(str)
    raw_dir = data_dir / 'raw'
    raw = []
    for sample_id in ids:
        arr = np.loadtxt(raw_dir / f'{sample_id}.csv', delimiter=',', skiprows=1, dtype=np.float64)
        if arr.shape != (10001, 2):
            raise ValueError(f'unexpected raw shape for {sample_id}: {arr.shape}')
        raw.append((arr[:, 0], arr[:, 1]))
    X_clean = np.stack([audit_make_input(r, d['X'][i]) for i, r in enumerate(raw)])
    device = core_setup(device='auto')
    checkpoint_paths = {'initial': AUDIT_ROOT / 'cnn' / 'models' / 'initial.pt', 'second': AUDIT_ROOT / 'cnn' / 'models' / 'second.pt'}
    expected_hashes = {'initial': INITIAL_SHA256, 'second': SECOND_SHA256}
    for name, path in checkpoint_paths.items():
        if audit_file_hash(path) != expected_hashes[name]:
            raise RuntimeError(f'checkpoint changed before robustness audit: {name}')
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    if any(AUDIT_OUT.iterdir()):
        raise FileExistsError(f'refusing to overwrite existing robustness results: {AUDIT_OUT}')
    loaded = {}
    clean_predictions = {}
    clean_metrics = {}
    for name, path in checkpoint_paths.items():
        model, ck = load_model(path, device)
        core_check_input(ck, d)
        pred, metrics = audit_predict(model, ck, d, X_clean, device)
        loaded[name] = (model, ck)
        clean_predictions[name] = pred
        clean_metrics[name] = metrics
    scenarios = []
    for replicate, seed in enumerate([2026111801, 2026111802, 2026111803], 1):
        for level in [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]:
            scenarios.append((f'noise_{level:g}_rep{replicate}', 'noise', level, 1, seed + int(level * 1000000.0)))
    for kind in ['tilt', 'curvature']:
        for level in [0.05, 0.1, 0.2]:
            for sign in [-1, 1]:
                scenarios.append((f'{kind}_{('minus' if sign < 0 else 'plus')}_{level:g}', kind, level, sign, 0))
    rows = []
    for idx, (scenario, kind, level, sign, seed) in enumerate(scenarios, 1):
        rng = np.random.default_rng(seed) if kind == 'noise' else np.random.default_rng(0)
        X = np.stack([audit_make_input(audit_perturb(r, kind, level, rng, sign)) for r in raw])
        for name, (model, ck) in loaded.items():
            pred, metrics = audit_predict(model, ck, d, X, device)
            drift = audit_drift_metrics(clean_predictions[name], pred)
            opd = metrics['opd_um']
            phase = metrics['phase_rad']
            rows.append({'scenario': scenario, 'kind': kind, 'level_fraction': level, 'sign': sign, 'replicate': int(scenario.rsplit('rep', 1)[1]) if 'rep' in scenario else 0, 'model': name, 'opd_mae_um': opd['mae'], 'opd_rmse_um': opd['rmse'], 'opd_bias_um': opd['bias'], 'opd_p95_um': opd['p95_abs'], 'opd_max_um': opd['max_abs'], 'opd_position_macro_mae_um': opd['position_macro_mae'], 'phase_mae_rad': phase['mae'], 'phase_rmse_rad': phase['rmse'], 'phase_p95_rad': phase['p95_abs'], **drift})
        print(f'scenario {idx}/{len(scenarios)}: {scenario}', flush=True)
    fields = list(rows[0])
    with (AUDIT_OUT / 'scenario_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {'validation_samples': len(d['X']), 'validation_positions': len(np.unique(d['position_id'])), 'raw_to_clean_c_preprocessing_max_abs_diff': float(np.max(np.abs(X_clean - d['X']))), 'device': str(device), 'validation_sha256': audit_file_hash(validation_path), 'checkpoint_sha256': {name: audit_file_hash(path) for name, path in checkpoint_paths.items()}, 'checkpoint_hashes_match_before_run': True, 'clean_metrics': clean_metrics, 'scenario_count': len(scenarios), 'measurements': 'agreement with existing algorithmic pseudo labels and drift from clean model prediction; not independent physical accuracy', 'raw_unit_caveat': 'recorded raw intensity unit is unconfirmed; Gaussian levels are fractions of estimated background peak, not instrument SNR', 'cross_batch': 'not evaluated; no new-date or changed-acquisition dataset is available', 'training_performed': False, 'test_accessed': False, 'original_or_second_checkpoint_modified': False}
    if result['checkpoint_sha256'] != expected_hashes:
        raise RuntimeError('a model checkpoint changed during read-only evaluation')
    (AUDIT_OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    aggregates = []
    numeric = ['opd_mae_um', 'opd_p95_um', 'opd_position_macro_mae_um', 'phase_mae_rad', 'opd_drift_mae_um', 'opd_drift_p95_um', 'phase_drift_circular_mae_rad']
    for model_name in ['initial', 'second']:
        for level in [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]:
            group = [r for r in rows if r['model'] == model_name and r['kind'] == 'noise' and (r['level_fraction'] == level)]
            row = {'model': model_name, 'level_fraction': level}
            for col in numeric:
                values = np.array([r[col] for r in group], dtype=np.float64)
                row[f'{col}_mean'] = float(values.mean())
                row[f'{col}_sd'] = float(values.std(ddof=1))
            aggregates.append(row)
    (AUDIT_OUT / 'noise_aggregate.json').write_text(json.dumps(aggregates, ensure_ascii=False, indent=2), encoding='utf-8')
    report = ['# 第六项：原始强度域鲁棒性压力测试', '', '## 结论边界', '', '本轮只评价，不训练、不改变检查点、不访问测试集。数据来自已有验证集的60个位置；该验证集已用于早停和第二版选择，因此本结果是开发阶段压力测试，不是独立盲测。标签为算法伪标签。原始强度的物理单位未确认，所以噪声比例不能称为真实仪器SNR。没有跨日期/跨批次数据，本轮不评价跨批次泛化。', '', '## 扰动协议', '', '验证集每条原始CSV先重跑C预处理，干净重建与已有网络输入逐点完全相同。噪声是在原始强度域注入零均值高斯噪声，标准差为该条谱100点高斯平滑背景峰值的0.1%、0.25%、0.5%、1%、2%、5%，每级3个固定随机种子。包络变化包含端到端倾斜比(1+5%)的±1次方、(1+10%)的±1次方、(1+20%)的±1次方，及边缘/中心弯曲比(1+5%)的±1次方、(1+10%)的±1次方、(1+20%)的±1次方。所有扰动后重新执行同一个C预处理和RMS归一化。', '', '每种条件下同时比较对原伪标签的OPD/相位误差、位置宏平均误差，以及相对本模型干净输入预测的漂移。模型间比较只用于观察敏感性，不在本实验中选出或替换模型。', '', '## 干净验证基线', '', '|模型|OPD MAE (μm)|OPD P95 (μm)|相位圆周 MAE (rad)|', '|---|---:|---:|---:|']
    for name in ['initial', 'second']:
        m = clean_metrics[name]
        report.append(f'|{name}|{m['opd_um']['mae']:.6f}|{m['opd_um']['p95_abs']:.6f}|{m['phase_rad']['mae']:.8f}|')
    report += ['', '## 噪声水平汇总', '', '数值为3次随机实现的均值±样本标准差。性能仍表示与伪标签的一致性。', '', '|模型|σ/背景峰|OPD MAE μm|OPD P95 μm|相位 MAE rad|OPD预测漂移 MAE μm|相位预测漂移 MAE rad|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:
        report.append(f'|{r['model']}|{r['level_fraction'] * 100:.2f}%|{r['opd_mae_um_mean']:.3f}±{r['opd_mae_um_sd']:.3f}|{r['opd_p95_um_mean']:.3f}±{r['opd_p95_um_sd']:.3f}|{r['phase_mae_rad_mean']:.5f}±{r['phase_mae_rad_sd']:.5f}|{r['opd_drift_mae_um_mean']:.3f}±{r['opd_drift_mae_um_sd']:.3f}|{r['phase_drift_circular_mae_rad_mean']:.5f}±{r['phase_drift_circular_mae_rad_sd']:.5f}|')
    report += ['', '## 包络变化', '', '完整倾斜和弯曲的正负方向结果见`scenario_metrics.csv`。', '', '## 是否进入增强训练', '', '仅当扰动使误差出现可重复、且相对于干净基线有实际幅度的恶化时，才值得另开独立训练分支做训练集加噪对照；验证与测试输入不得增强训练使用。若只有极端5%噪声或20%包络条件恶化，应先判断它是否与真实设备可能状态匹配，不能因为可合成就把它设成训练目标。当前输出是上述决策证据，不预设训练必要性。', '', '## 复现文件', '', '- 固定方案：`protocol.json`。', '- 每个扰动/模型的完整指标：`results/scenario_metrics.csv`。', '- 机器可读概要：`results/summary.json`、`results/noise_aggregate.json`。', '- 执行入口：`cnn/CNN完整程序.py --mode robustness`。', '- 第二版及初版模型SHA256记录在summary.json；两者均未修改。', '']
    (AUDIT_OUT / '鲁棒性实验报告.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'summary': result, 'noise_aggregate': aggregates}, ensure_ascii=False, indent=2), flush=True)

# ==================== 5. 唯一入口：直接运行这个文件 ====================
def main():
    """默认串行执行预测和鲁棒性；重训需显式选择train。"""
    from datetime import datetime
    import argparse
    script = Path(__file__).resolve()
    project_default = script.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['all','predict','train','robustness'], default='all',
                        help='默认all：加载已有模型做验证预测，再完成鲁棒性评价；train显式重训')
    parser.add_argument('--project-root', type=Path, default=project_default,
                        help='仓库根目录，包含 datasets/ 和 cnn/，默认自动定位')
    parser.add_argument('--data-dir', type=Path, help='训练processed目录，默认仓库 datasets/processed')
    parser.add_argument('--input', type=Path, help='预测用C方案NPZ，默认原验证集')
    parser.add_argument('--checkpoint', type=Path, default=script.parent/'models/second.pt')
    parser.add_argument('--out', type=Path, help='必须是新的输出目录；默认本包下运行结果/日期时间')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--seed', type=int, default=20261119)
    parser.add_argument('--epochs', type=int, default=160)
    args = parser.parse_args()
    data = args.data_dir or args.project_root/'datasets/processed'
    out = args.out or script.parent/'outputs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if out.exists(): raise FileExistsError(f'输出目录已存在，请换新名称：{out}')
    if args.mode=='train':
        tr, va = core_load_npz(data/'train.npz'), core_load_npz(data/'validation.npz')
        result = run(tr, va, out, seed=args.seed, alpha=100., variant='baseline', segments=1,
                     epochs=args.epochs, patience=30, lr=.001, batch_size=128)
        print('训练完成。新权重只保存在：', out/'best.pt', '最佳轮次：',result['best_epoch'])
        return
    out.mkdir(parents=True)
    if args.mode in ['all','predict']:
        print('步骤1：加载交付第二版模型，完成C方案输入预测。',flush=True)
        device = core_setup(device=args.device)
        model, ck = load_model(args.checkpoint, device)
        d = core_load_npz(args.input or data/'validation.npz',labeled=False)
        core_check_input(ck,d)
        pred = core_physical(core_infer(model,d['X'],device),ck['stats'])
        core_predictions_csv(out/'预测结果.csv',d,pred,labeled=False)
        print('预测结果：',out/'预测结果.csv',flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if args.mode in ['all','robustness']:
        print('步骤2：固定原项目初版和第二版，运行30场景鲁棒性评价；不训练。',flush=True)
        audit_run_robustness(args.project_root,out/'鲁棒性评价')
    print('执行完成。全部新结果位于：',out.resolve(),flush=True)

if __name__ == '__main__':
    main()
