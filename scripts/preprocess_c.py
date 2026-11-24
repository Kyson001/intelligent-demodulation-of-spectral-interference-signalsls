"""完整预处理脚本（中文逐块注释，可单独复制运行）。

依赖：numpy、scipy。默认输入：本项目analysis/data_4000中的file*.csv。
执行：python scripts/preprocess_c.py

主流程：检查数据 → 单位处理 → 波长转波数 → 等间隔插值 → 估计慢变背景
       → 背景扣除/包络代理补偿 → Hann加窗 → 抗混叠降采样 → 归一化 → 保存。

注意：包络补偿与加窗不是温度补偿；脚本不生成OPD或相位真值。
"""

# ==================== 第1块：导入工具、定义参数 ====================
# pathlib处理路径；csv/json保存说明；hashlib记录来源和识别完全重复谱。
import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly, windows


@dataclass(frozen=True)
class CConfig:
    """所有参数都随输出保存，保证以后训练和网页推理使用相同流程。"""
    background_sigma_points: float = 100.0  # 密网格上高斯背景估计的标准差，单位：采样点
    envelope_floor_fraction: float = 0.08  # 分母最小值为背景峰值的8%，限制弱光区增益
    model_points: int = 2048              # 神经网络每条光谱的输入长度
    intensity_unit: str = 'recorded'      # recorded：保持原值；dbm：明确按dBm转mW

    def validate(self):
        """参数错误直接报错，不悄悄使用不合理设置。"""
        if not np.isfinite(self.background_sigma_points) or self.background_sigma_points <= 0:
            raise ValueError('background_sigma_points必须为有限正数')
        if not 0 < self.envelope_floor_fraction <= 1:
            raise ValueError('envelope_floor_fraction必须在(0,1]内')
        if self.model_points < 128:
            raise ValueError('model_points不能小于128')
        if self.intensity_unit not in ('recorded', 'dbm'):
            raise ValueError('intensity_unit仅支持recorded或dbm')


# ==================== 第2块：读取、检查单条CSV ====================
def read_spectrum(path):
    """读取两列CSV：第一列波长nm，第二列仪器强度；跳过一行表头。"""
    data = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f'{path.name}必须有且只有两列：波长和强度')
    return data[:, 0], data[:, 1]


def prepare_dense_spectrum(wavelength_nm, intensity, config=CConfig()):
    """准备传统解调需要的高分辨率光谱，保留加窗前后的结果。"""
    config.validate()
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
    original = np.asarray(intensity, dtype=np.float64)
    if wavelength_nm.ndim != 1 or original.shape != wavelength_nm.shape:
        raise ValueError('波长与强度必须为长度相同的一维数组')
    if len(original) < max(128, config.model_points):
        raise ValueError('原始点数过少；此脚本只做保留点数或降采样，不通过上采样宣称增加信息')
    if not np.isfinite(wavelength_nm).all() or not np.isfinite(original).all():
        raise ValueError('发现NaN/Inf，请核查原始文件；本脚本不自动填补')
    if np.any(wavelength_nm <= 0):
        raise ValueError('波长必须为正，单位应为nm')

    # ==================== 第3块：强度单位处理 ====================
    # 列名写dBm不足以证明数据确实为dBm，因此默认recorded保持已有数值。
    # 只有老师确认原值为dBm后，才使用dbm选项：P(mW)=10**(P_dBm/10)。
    if config.intensity_unit == 'dbm':
        with np.errstate(over='raise', invalid='raise'):
            linear = np.power(10.0, original/10.0)
    else:
        linear = original.copy()
    if not np.isfinite(linear).all():
        raise ValueError('单位转换后出现非有限数')

    # ==================== 第4块：波长转波数、同步排序 ====================
    # k=2π/λ，λ使用米；波长增大时k减小，所以坐标和强度必须一起排序。
    k_raw = 2*np.pi/(wavelength_nm*1e-9)
    order = np.argsort(k_raw)
    k_sorted, intensity_sorted = k_raw[order], linear[order]
    if np.any(np.diff(k_sorted) <= 0):
        raise ValueError('存在重复波长，不能直接插值；请先核查采集记录')

    # ==================== 第5块：等波数密网格插值 ====================
    # 先保留原始点数，修正“等波长不等波数”的问题；不在实测范围外推。
    k_dense = np.linspace(k_sorted[0], k_sorted[-1], len(k_sorted))
    intensity_k = np.interp(k_dense, k_sorted, intensity_sorted)

    # ==================== 第6块：估计慢变背景/光源包络代理 ====================
    # 高斯核在这里估计慢变背景，不是用来直接平滑最终干涉条纹或消除温漂。
    # σ=100适用于当前波段、点数及已诊断OPD范围；换设备/范围后需要重新检验。
    background = gaussian_filter1d(
        intensity_k, sigma=config.background_sigma_points, mode='reflect')
    background_peak = float(background.max())
    if background_peak <= 0:
        raise ValueError('没有正的背景包络，不能执行当前补偿模型')

    # ==================== 第7块：背景扣除＋带分母下限的包络补偿 ====================
    # 理想模型I=A+B*cosφ。这里先扣估计背景，再除慢变包络代理。
    # 背景代理不等于独立测得的条纹幅度B，仍保留真实对比度变化。
    # max操作只保护分母；不会把原始负噪声或干涉谷值直接截断成零。
    denominator_floor = config.envelope_floor_fraction*background_peak
    denominator = np.maximum(background, denominator_floor)
    background_removed = intensity_k-background
    envelope_corrected = background_removed/denominator
    weak_mask = background < denominator_floor  # 标记弱光区，不静默删点

    # ==================== 第8块：Hann加窗，生成加窗谱 ====================
    # 逐点相乘；sym=False与此前FFT比较使用的周期Hann保持一致。
    # Hann降低光谱边缘权重，改变泄漏和有效带宽；不等于温度校正。
    hann = windows.hann(len(k_dense), sym=False)
    windowed_c = envelope_corrected*hann

    return dict(
        k_dense=k_dense, intensity_k=intensity_k, background=background, denominator=denominator,
        background_removed=background_removed, envelope_corrected=envelope_corrected,
        hann=hann, windowed_c=windowed_c, weak_mask=weak_mask,
        quality=dict(raw_min=float(original.min()), raw_max=float(original.max()),
                     background_peak=background_peak, weak_fraction=float(weak_mask.mean())),
    )


def preprocess_spectrum(wavelength_nm, intensity, config=CConfig()):
    """处理一条光谱，返回网络输入、坐标、高分辨率中间结果与质量记录。"""
    stages = prepare_dense_spectrum(wavelength_nm, intensity, config)
    k_dense, windowed_c = stages['k_dense'], stages['windowed_c']

    # ==================== 第9块：抗混叠降采样 ====================
    # 多相FIR先低通再改变采样率，不能替换成简单隔点抽样。
    # line延拓减少边界跳变。返回长度是model_points；输出坐标按实际采样率计算。
    n_in, n_out = len(k_dense), config.model_points
    resampled = resample_poly(windowed_c, n_out, n_in, padtype='line')
    k_model = k_dense[0]+np.arange(len(resampled))*(k_dense[1]-k_dense[0])*n_in/n_out
    if len(resampled) != n_out or k_model[-1] > k_dense[-1]+1e-7:
        raise ValueError('降采样长度或坐标越界')

    # ==================== 第10块：逐谱中心化和RMS归一化 ====================
    # 每条光谱单独计算，不利用其他样本或测试集统计量。
    # 中心化会使最终边缘不再严格为零，但中间的Hann窗数据仍保留供检查。
    mean_before = float(resampled.mean())
    centered = resampled-mean_before
    rms_before = float(np.sqrt(np.mean(centered**2)))
    if not np.isfinite(rms_before) or rms_before < 1e-15:
        raise ValueError('处理后为常数/近零信号，无法归一化')
    x_model = (centered/rms_before).astype(np.float32)
    if not np.isfinite(x_model).all():
        raise ValueError('网络输入存在非有限数')

    # ==================== 第11块：返回结果、保留全部中间步骤 ====================
    # quality用于判断问题来自哪一步；不是OPD/相位真值或真实SNR。
    quality = dict(stages['quality'], mean_before_normalization=mean_before, rms_before_normalization=rms_before)
    return dict(X=x_model, k_model=k_model, resampled_c=resampled, **dict(stages, quality=quality))


# ==================== 第12块：批量处理、保存并生成分组核对表 ====================
def main():
    """批量处理file*.csv；默认保存全体网络输入，以及首/中/末三条的中间数据。"""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='光谱预处理：包络补偿＋Hann加窗，中文注释版')
    parser.add_argument('--data-dir', type=Path, default=root/'analysis/data_4000', help='原始CSV目录')
    parser.add_argument('--out', type=Path, default=root/'analysis/preprocessed_C', help='输出目录')
    parser.add_argument('--points', type=int, default=2048, help='最终输入长度')
    parser.add_argument('--sigma', type=float, default=100, help='背景高斯核σ，单位为密网格点数')
    parser.add_argument('--floor', type=float, default=.08, help='包络分母下限相对峰值比例')
    parser.add_argument('--intensity-unit', choices=['recorded','dbm'], default='recorded', help='未核实单位时保持recorded')
    parser.add_argument('--save-all-stages', action='store_true', help='为所有样本保存中间数组，会占用较大空间')
    parser.add_argument('--overwrite', action='store_true', help='允许覆盖同名输出')
    args = parser.parse_args()
    config = CConfig(args.sigma, args.floor, args.points, args.intensity_unit)
    config.validate()
    paths = sorted(args.data_dir.glob('file*.csv'))
    if not paths:
        raise ValueError(f'没有找到file*.csv：{args.data_dir}')
    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out/f'model_inputs_C_{config.model_points}.npz'
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f'输出已存在：{output_path}；需要时添加--overwrite')
    stages_dir = args.out/'stages'
    stages_dir.mkdir(exist_ok=True)
    representative_indices = {0, (len(paths)-1)//2, len(paths)-1}
    signals, identifiers, duplicate_groups, rows, means, scales = [], [], [], [], [], []
    common_wavelength, common_k = None, None

    for index, path in enumerate(paths):
        wavelength, intensity = read_spectrum(path)
        # 全体样本须使用相同原始轴；不把不同轴的数组误堆叠成一套数据。
        if common_wavelength is None:
            common_wavelength = wavelength.copy()
        elif not np.array_equal(wavelength, common_wavelength):
            raise ValueError(f'{path.name}波长轴不同，请统一采集配置后再处理')
        try:
            result = preprocess_spectrum(wavelength, intensity, config)
        except (ValueError, FloatingPointError) as error:
            raise ValueError(f'{path.name}预处理失败：{error}；未静默删除该样本') from error
        if common_k is None:
            common_k = result['k_model']
        np.testing.assert_allclose(result['k_model'], common_k, rtol=0, atol=1e-8)
        signals.append(result['X'])
        identifiers.append(path.stem)
        # 完全重复谱使用同一个哈希组。它不能代替真实“采集位置”编号。
        duplicate_groups.append(hashlib.sha256(intensity.tobytes()).hexdigest())
        means.append(result['quality']['mean_before_normalization'])
        scales.append(result['quality']['rms_before_normalization'])
        rows.append(dict(sample_id=path.stem, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         order_block_candidate=index//10+1, position_id_confirmed='',
                         **result['quality']))
        if args.save_all_stages or index in representative_indices:
            arrays = {key:value for key,value in result.items() if key != 'quality'}
            np.savez_compressed(stages_dir/f'{path.stem}_C_stages.npz',
                                wavelength_nm=wavelength, intensity_recorded=intensity, **arrays)
        if (index+1)%500 == 0:
            print(f'预处理已处理 {index+1}/{len(paths)} 条', flush=True)

    # 不把“每10条一组”的推测自动当作真实位置；先输出候选块供采集者核对。
    # 新训练集必须按确认的位置分组。此处不沿用旧的随机样本划分。
    X = np.stack(signals)
    np.savez_compressed(output_path, X=X, k_rad_per_m=common_k,
                        sample_ids=np.array(identifiers), duplicate_group=np.array(duplicate_groups),
                        position_id=np.full(len(paths), ''), split=np.full(len(paths), 'unassigned'),
                        normalization_mean=np.array(means), normalization_rms=np.array(scales),
                        scheme=np.array('C'), label_status=np.array('no_truth_labels_generated'))
    with (args.out/'quality_and_position_review.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(scheme='C', config=asdict(config), shape=list(X.shape), dtype=str(X.dtype),
                    finite=bool(np.isfinite(X).all()), input_files=len(paths),
                    unique_exact_duplicate_groups=len(set(duplicate_groups)),
                    original_wavelength_nm=[float(common_wavelength.min()),float(common_wavelength.max())],
                    split_status='unassigned_pending_confirmed_acquisition_positions',
                    position_note='order_block_candidate仅为每10条的顺序候选块，非已确认位置',
                    intensity_note='recorded表示沿用原值；并未证明仪器强度单位',
                    temperature_compensation=False, label_status='没有生成或伪造OPD/相位真值',
                    output_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest())
    (args.out/'manifest_C.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


# ==================== 第13块：脚本执行入口 ====================
# 直接运行文件时批处理；被import时只提供函数，不自动读取整个数据集。
if __name__ == '__main__':
    main()
