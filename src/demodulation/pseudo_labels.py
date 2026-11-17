"""迈克尔逊光谱的光程差/中心相位伪标签，中文注释版。

光程差主标签来自多个干涉峰的频率间距，绝不使用10 μm名义步长或历史缓存替代。
相位主标签来自正侧带解析信号在固定参考波长处的局部拟合；它仍然是伪标签。
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.constants import c
from scipy.signal import find_peaks, savgol_filter


# ==================== 第1块：固定参数和质量阈值 ====================
@dataclass(frozen=True)
class LabelConfig:
    reference_nm: float = 1560.0        # 所有样本使用同一相位参考波长
    min_opd_mm: float = 0.5            # FFT仅用于估计条纹周期和定位侧带的搜索范围
    max_opd_mm: float = 15.0
    minimum_peaks: int = 8
    prominence_fraction: float = 0.25 # 相对有效区幅度的寻峰显著度
    minimum_envelope_fraction: float = 0.15
    minimum_hann_weight: float = 0.2
    maximum_spacing_cv: float = 0.02  # 相邻峰间隔变异系数上限，属于QC规则而非精度保证
    maximum_fft_disagreement: float = 0.02
    maximum_irregular_gap_fraction: float = 0.05
    maximum_missing_fraction: float = 0.10
    minimum_coverage: float = 0.35
    sideband_half_width_bins: int = 12
    phase_half_width_thz: float = 0.25 # 参考频率左右各0.25 THz，进行局部二次拟合
    maximum_local_phase_residual_rad: float = 0.10
    maximum_phase_method_difference_rad: float = 0.20

    def to_dict(self):
        return asdict(self)


def wrap_phase(value):
    """圆周相位统一到[-π,π]；不能直接对跨边界的角度做普通减法。"""
    return np.arctan2(np.sin(value), np.cos(value))


# ==================== 第2块：FFT粗定位（不是OPD最终标签） ====================
def coarse_carrier(k, windowed_c, config):
    """利用C方案的加窗谱，找干涉侧带中心和大致条纹周期。"""
    n = len(k)
    amplitude = np.abs(np.fft.rfft(windowed_c))
    opd_axis = 2*np.pi*np.fft.rfftfreq(n, d=k[1]-k[0])
    domain = (opd_axis >= config.min_opd_mm*1e-3) & (opd_axis <= config.max_opd_mm*1e-3)
    indices = np.flatnonzero(domain)
    if not len(indices) or np.max(amplitude[indices]) <= 1e-12:
        raise ValueError('没有可用的干涉侧带')
    j = int(indices[np.argmax(amplitude[indices])])
    offset = 0.0
    if 0 < j < len(amplitude)-1:
        a,b,d = amplitude[j-1:j+2]
        denominator = a-2*b+d
        if denominator < 0:
            offset = float(np.clip(.5*(a-d)/denominator,-.5,.5))
    return j, float((j+offset)*(opd_axis[1]-opd_axis[0]))


# ==================== 第3块：多峰寻峰及亚采样峰位细化 ====================
def locate_fringe_peaks(k, corrected, support, rough_opd, config):
    """先轻量平滑用于检测候选峰，再在未平滑/未乘Hann的补偿谱上细化峰位。

    不能把光频梳的梳齿当干涉峰；本算法适用于当前可分辨的慢变干涉条纹。
    Hann是随波数变化的乘子，会偏移局部最大值，因此不直接拿加窗谱的极大值制标签。
    """
    dk = k[1]-k[0]
    period_points = 2*np.pi/(rough_opd*dk)
    if period_points < 8:
        raise ValueError('每周期点数过少，不能可靠进行当前多点峰值拟合')
    smooth = savgol_filter(corrected, window_length=5, polyorder=3, mode='interp')
    level = np.quantile(np.abs(smooth[support]),0.90)
    if level <= 1e-10:
        raise ValueError('有效区没有足够的条纹对比度')
    candidates,_ = find_peaks(smooth,
                             distance=max(3,int(.55*period_points)),
                             prominence=config.prominence_fraction*level)
    # 峰中心及前后两个点都必须落在有效区域，避免边缘伪峰。
    candidates = np.array([j for j in candidates if j>=2 and j<len(k)-2 and support[j-2:j+3].all()],dtype=int)
    positions, refinement_ok = [],[]
    local_x = np.arange(-2,3,dtype=float)
    for j in candidates:
        a,b,_ = np.polyfit(local_x,corrected[j-2:j+3],2)
        shift = -b/(2*a) if a < 0 else np.nan
        ok = bool(np.isfinite(shift) and abs(shift)<=1.0)
        positions.append(j+float(shift) if ok else float(j))
        refinement_ok.append(ok)
    positions = np.asarray(positions)
    frequencies = c*(k[0]+positions*dk)/(2*np.pi)
    return candidates,positions,frequencies,np.asarray(refinement_ok)


# ==================== 第4块：峰序号-频率稳健直线拟合 ====================
def robust_peak_fit(frequencies, config):
    """f_j = f_c + 1e12*(a*m_j+b)，OPD=c/(a*1e12)。

    相邻间隔用于识别可能漏峰；接近整数倍的间隔才允许补计周期，并显式记录。
    严重杂峰/漏峰不会被静默解释为连续峰，而会触发QC。
    """
    if len(frequencies)<config.minimum_peaks:
        raise ValueError(f'有效干涉峰不足{config.minimum_peaks}个')
    gaps = np.diff(frequencies)
    median_gap = float(np.median(gaps))
    if median_gap <= 0:
        raise ValueError('频率峰位非递增')
    increments = np.maximum(1,np.rint(gaps/median_gap).astype(int))
    increments = np.minimum(increments,10)
    irregular = np.abs(gaps/median_gap-increments)>0.2
    orders = np.r_[0,np.cumsum(increments)]
    # 使用一个真实峰对应的整数序号作原点，保证相位截距只相差整数2π。
    orders = orders-orders[len(orders)//2]
    fc = c/(config.reference_nm*1e-9)
    y = (frequencies-fc)*1e-12
    design = np.column_stack([orders,np.ones(len(orders))])
    weights = np.ones(len(orders))
    # 迭代Huber样式权重，降低少数峰位偏差对整体峰间距的影响。
    for _ in range(6):
        sw = np.sqrt(weights)
        coef = np.linalg.lstsq(design*sw[:,None],y*sw,rcond=None)[0]
        residual = y-design@coef
        scale = max(1.4826*np.median(np.abs(residual-np.median(residual))),1e-8)
        weights = np.minimum(1,2.5*scale/np.maximum(np.abs(residual),1e-15))
    slope,intercept = coef
    if slope <= 0:
        raise ValueError('峰序号拟合斜率非正')
    spacing_hz = float(slope*1e12)
    opd = float(c/spacing_hz)
    # 由“干涉极大值处相位=2π整数”获得第二种相位估计，主要作交叉检查。
    peak_phase = float(wrap_phase(-2*np.pi*intercept/slope))
    per_period_gap = gaps/increments
    cv = float(np.std(per_period_gap)/np.mean(per_period_gap))
    missing = int(np.sum(increments-1))
    return dict(opd_m=opd, spacing_hz=spacing_hz, peak_phase_rad=peak_phase,
                spacing_cv=cv, missing_peaks=missing,
                missing_fraction=float(missing/(orders[-1]-orders[0]+1)),
                irregular_gap_fraction=float(irregular.mean()),
                fit_phase_residual_rad=float(np.sqrt(np.mean(residual**2))*2*np.pi/slope),
                orders=orders,fit_residual_hz=residual*1e12)


# ==================== 第5块：正侧带解析信号，提取中心相位伪标签 ====================
def local_center_phase(k,windowed_c,background,support,carrier_bin,config):
    """相位伪标签定义：1560 nm等固定参考波长处的包裹相位，固定正OPD支路。

    单独选择正侧带→IFFT→去整数载波→解包裹→中心附近局部二次拟合。
    局部拟合保留参考点相位，减轻宽带非线性相位导致的全带直线截距偏差。
    这不是独立真值，也没有恢复未知的整周期编号或OPD符号。
    """
    n = len(k)
    index = np.arange(n)
    distance = np.abs(index-carrier_bin)
    width = config.sideband_half_width_bins
    band = np.zeros(n)
    band[distance<=width/2] = 1
    taper = (distance>width/2)&(distance<width)
    band[taper] = .5*(1+np.cos(np.pi*(distance[taper]-width/2)/(width/2)))
    band[index==0] = 0
    band[index>=(n+1)//2] = 0
    analytic = 2*np.fft.ifft(np.fft.fft(windowed_c)*band)
    carrier_opd = carrier_bin*2*np.pi/(n*(k[1]-k[0]))
    demodulated = analytic*np.exp(-1j*(k-k[0])*carrier_opd)
    phase = np.unwrap(np.angle(demodulated))
    kc = 2*np.pi/(config.reference_nm*1e-9)
    f = c*k/(2*np.pi)
    fc = c/(config.reference_nm*1e-9)
    near = support & (np.abs(f-fc)<=config.phase_half_width_thz*1e12)
    if near.sum()<32 or not k[near].min()<kc<k[near].max():
        raise ValueError('相位参考点不在有效区，或局部点数不足')
    x = (k-kc)*1e-3
    design = np.column_stack([np.ones(n),x,x*x])
    weights = np.maximum(np.abs(analytic[near]),1e-15)**2
    sw = np.sqrt(weights/weights.max())
    coef = np.linalg.lstsq(design[near]*sw[:,None],phase[near]*sw,rcond=None)[0]
    phi = float(wrap_phase(coef[0]+carrier_opd*(kc-k[0])))
    residual = phase[near]-design[near]@coef
    rms = float(np.sqrt(np.average(residual**2,weights=weights)))
    return dict(phase_rad=phi,phase_sin=float(np.sin(phi)),phase_cos=float(np.cos(phi)),
                local_phase_residual_rad=rms,local_phase_points=int(near.sum()),
                local_opd_m=float(carrier_opd+coef[1]*1e-3),
                analytic_amplitude=np.abs(analytic),local_mask=near,
                demodulated_phase=phase)


# ==================== 第6块：合并标签、质量标志，保留不合格记录 ====================
def label_prepared(stages,config=LabelConfig()):
    """输入C预处理返回的字典；输出标量标签、QC及可用于核对的峰位数组。"""
    if not config.min_opd_mm>0 or config.max_opd_mm<=config.min_opd_mm or config.minimum_peaks<3:
        raise ValueError('非法标签参数')
    k = stages['k_dense']
    background = stages['background']
    if not np.isfinite(k).all() or not np.all(np.diff(k)>0):
        raise ValueError('标签要求有效的递增波数轴')
    if not np.allclose(np.diff(k),k[1]-k[0],rtol=1e-8,atol=1e-9):
        raise ValueError('标签要求等间隔波数轴')
    support = ((background>=config.minimum_envelope_fraction*background.max()) &
               (stages['hann']>=config.minimum_hann_weight))
    if support.sum()<100:
        raise ValueError('有效光谱范围过小')
    carrier,rough = coarse_carrier(k,stages['windowed_c'],config)
    candidates,positions,frequencies,refinement_ok = locate_fringe_peaks(
        k,stages['envelope_corrected'],support,rough,config)
    fit = robust_peak_fit(frequencies,config)
    coverage = float((frequencies[-1]-frequencies[0])/(c*(k[-1]-k[0])/(2*np.pi)))
    fft_difference = abs(fit['opd_m']-rough)/fit['opd_m']
    reasons=[]
    if fit['spacing_cv']>config.maximum_spacing_cv:
        reasons.append('spacing_not_uniform')
    if fit['irregular_gap_fraction']>config.maximum_irregular_gap_fraction:
        reasons.append('irregular_peak_gaps')
    if fit['missing_fraction']>config.maximum_missing_fraction:
        reasons.append('too_many_inferred_missing_peaks')
    if fft_difference>config.maximum_fft_disagreement:
        reasons.append('fft_peak_spacing_disagreement')
    if coverage<config.minimum_coverage:
        reasons.append('insufficient_peak_coverage')
    if float(refinement_ok.mean())<.8:
        reasons.append('peak_refinement_unstable')
    opd_valid = not reasons
    row = dict(opd_peak_m=fit['opd_m'],half_opd_m=fit['opd_m']/2,
               delta_f_hz=fit['spacing_hz'],peak_count=len(frequencies),
               peak_coverage=coverage,spacing_cv=fit['spacing_cv'],
               inferred_missing_peaks=fit['missing_peaks'],
               irregular_gap_fraction=fit['irregular_gap_fraction'],
               peak_refinement_fraction=float(refinement_ok.mean()),
               peak_fit_phase_residual_rad=fit['fit_phase_residual_rad'],
               fft_coarse_opd_m=rough,fft_relative_difference=fft_difference,
               phase_from_peak_fit_rad=fit['peak_phase_rad'],
               opd_valid=opd_valid,phase_valid=False,
               phase_pseudo_rad=np.nan,phase_sin=np.nan,phase_cos=np.nan,
               phase_local_residual_rad=np.nan,phase_method_difference_rad=np.nan,
               phase_local_opd_m=np.nan,phase_local_points=0)
    try:
        phase = local_center_phase(k,stages['windowed_c'],background,support,carrier,config)
        difference = float(abs(wrap_phase(phase['phase_rad']-fit['peak_phase_rad'])))
        row.update(phase_pseudo_rad=phase['phase_rad'],phase_sin=phase['phase_sin'],phase_cos=phase['phase_cos'],
                   phase_local_residual_rad=phase['local_phase_residual_rad'],
                   phase_method_difference_rad=difference,phase_local_opd_m=phase['local_opd_m'],
                   phase_local_points=phase['local_phase_points'])
        phase_reasons=[]
        if phase['local_phase_residual_rad']>config.maximum_local_phase_residual_rad:
            phase_reasons.append('local_phase_residual_large')
        if difference>config.maximum_phase_method_difference_rad:
            phase_reasons.append('phase_methods_disagree')
        row['phase_valid']=opd_valid and not phase_reasons
        reasons.extend(phase_reasons)
    except ValueError as error:
        reasons.append('phase_extraction_failed:'+str(error))
    row['status']='accepted' if row['opd_valid'] and row['phase_valid'] else 'review'
    row['quality_reasons']=';'.join(reasons)
    trace=dict(peak_indices=candidates,peak_positions_fractional=positions,peak_frequencies_hz=frequencies,
               orders=fit['orders'],peak_fit_residual_hz=fit['fit_residual_hz'],support=support)
    return row,trace
