"""Reproducible, label-free preprocessing. No thermal compensation is inferred.

Intensity values are retained in their supplied units. Confirm linear units before
interpreting the background division physically. Diagnostics are not ground truth.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly, windows


@dataclass(frozen=True)
class Config:
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


def wrap_phase(x):
    return np.arctan2(np.sin(x), np.cos(x))


def uniform_k(wavelength_nm, intensity):
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if wavelength_nm.ndim != 1 or wavelength_nm.shape != intensity.shape:
        raise ValueError('Expected matching 1-D wavelength and intensity arrays')
    if len(intensity) < 128 or not np.isfinite(wavelength_nm).all() or not np.isfinite(intensity).all():
        raise ValueError('Insufficient points or non-finite data')
    if np.any(wavelength_nm <= 0):
        raise ValueError('Wavelengths must be positive')
    k = 2*np.pi/(wavelength_nm*1e-9)
    order = np.argsort(k)
    k, intensity = k[order], intensity[order]
    if np.any(np.diff(k) <= 0):
        raise ValueError('Duplicate wavelengths are not allowed')
    ku = np.linspace(k[0], k[-1], len(k))
    return ku, np.interp(ku, k, intensity)


def prepare(k, y, config=Config()):
    """A: constant DC removal; B: guarded envelope division; C: B times Hann.

    The slowly varying intensity is an A(k) proxy, NOT an independently known
    visibility B(k). Dividing by it flattens the source envelope only under a
    slowly varying visibility model. All outputs retain the original k origin.
    """
    dk = np.diff(k)
    if not np.allclose(dk, dk[0], rtol=1e-8, atol=1e-9):
        raise ValueError('prepare requires an equispaced k grid')
    background = gaussian_filter1d(y, config.background_sigma_points, mode='reflect')
    peak = float(np.max(background))
    if peak <= 0:
        raise ValueError('Positive envelope required for this comparison')
    floor = config.envelope_floor_fraction*peak
    denominator = np.maximum(background, floor)
    corrected = (y-background)/denominator
    hann = windows.hann(len(y), sym=False)
    schemes = {'A': y-y.mean(), 'B': corrected, 'C': corrected*hann}
    # One common, label-free fit support for all three estimators.
    support = (background >= config.fit_envelope_fraction*peak) & (hann >= 0.2)
    if support.sum() < 100:
        raise ValueError('Insufficient illuminated support')
    weights = (background/peak)**2
    return schemes, dict(background=background, denominator=denominator,
                         hann=hann, support=support, weights=weights)


def fft_peak(k, signal, config=Config()):
    n = len(k)
    amplitude = np.abs(np.fft.rfft(signal))
    opd_axis = 2*np.pi*np.fft.rfftfreq(n, d=k[1]-k[0])
    valid = (opd_axis >= config.min_opd_mm*1e-3) & (opd_axis <= config.max_opd_mm*1e-3)
    if not np.any(valid):
        raise ValueError('OPD search range outside sampling grid')
    indices = np.flatnonzero(valid)
    j = int(indices[np.argmax(amplitude[valid])])
    shift = 0.0
    if 0 < j < len(amplitude)-1:
        a, b, c = amplitude[j-1:j+2]
        denominator = a-2*b+c
        if denominator < 0:
            shift = float(np.clip(.5*(a-c)/denominator, -.5, .5))
    return (j+shift)*(opd_axis[1]-opd_axis[0]), j, opd_axis, amplitude


def extract_phase(k, signal, support, weights, carrier_bin, config=Config()):
    n = len(k)
    # Identical carrier and cosine-tapered band for every scheme on this sample.
    indices = np.arange(n)
    distance = np.abs(indices-carrier_bin)
    width = config.sideband_half_width_bins
    mask = np.zeros(n)
    mask[distance <= width/2] = 1
    taper = (distance > width/2) & (distance < width)
    mask[taper] = .5*(1+np.cos(np.pi*(distance[taper]-width/2)/(width/2)))
    mask[indices == 0] = 0
    mask[indices >= (n+1)//2] = 0
    analytic = 2*np.fft.ifft(np.fft.fft(signal)*mask)
    # Demodulate the known integer carrier before unwrapping: stable phase origin.
    carrier_opd = carrier_bin*2*np.pi/(n*(k[1]-k[0]))
    phase = np.unwrap(np.angle(analytic*np.exp(-1j*(k-k[0])*carrier_opd)))
    kc = 2*np.pi/(config.reference_nm*1e-9)
    x = (k-kc)*1e-3  # coefficient now has mm units; well-conditioned least squares
    design = np.column_stack([x, np.ones(n)])
    w = np.sqrt(weights[support])
    coef = np.linalg.lstsq(design[support]*w[:,None], phase[support]*w, rcond=None)[0]
    opd = carrier_opd+coef[0]*1e-3
    phi = wrap_phase(coef[1]+carrier_opd*(kc-k[0]))
    residual = phase-design@coef
    residual_rms = np.sqrt(np.average(residual[support]**2, weights=weights[support]))
    # Phase curve for display includes carrier and is anchored to the same kc.
    phase_centered = (k-kc)*opd+float(phi)+residual
    return dict(opd_m=float(opd), phase_rad=float(phi), residual_rms_rad=float(residual_rms),
                phase_curve=phase_centered, residual=residual, amplitude=np.abs(analytic))


def compare(wavelength_nm, intensity, config=Config()):
    k, y = uniform_k(wavelength_nm, intensity)
    signals, aux = prepare(k, y, config)
    _, carrier, _, _ = fft_peak(k, signals['A']*aux['hann'], config)
    outputs = {}
    for name, signal in signals.items():
        opd, peak, axis, amp = fft_peak(k, signal, config)
        phase = extract_phase(k, signal, aux['support'], aux['weights'], carrier, config)
        search = (axis >= config.min_opd_mm*1e-3) & (axis <= config.max_opd_mm*1e-3)
        near = np.abs(np.arange(len(amp))-carrier) <= config.sideband_half_width_bins
        concentration = np.sum(amp[near]**2)/max(np.sum(amp[search]**2),1e-30)
        half = amp[peak]/2
        left, right = peak, peak
        while left > 0 and amp[left] > half:
            left -= 1
        while right < len(amp)-1 and amp[right] > half:
            right += 1
        # Linear crossings: this is amplitude FWHM, not power FWHM.
        def cross(i, j):
            return i+(half-amp[i])/(amp[j]-amp[i])*(j-i)
        fwhm = (cross(right-1,right)-cross(left,left+1))*(axis[1]-axis[0])
        outputs[name] = dict(fft_opd_m=float(opd), phase_opd_m=phase['opd_m'],
                             phase_rad=phase['phase_rad'], phase_residual_mrad=phase['residual_rms_rad']*1e3,
                             fwhm_um=float(fwhm*1e6), band_energy_fraction=float(concentration),
                             signal=signal, fft_axis=axis, fft_amplitude=amp, **{'phase':phase})
    return dict(k=k, raw_k=y, aux=aux, methods=outputs)


def model_input(wavelength_nm, intensity, scheme='A', config=Config()):
    """训练/推理共用入口：三种预处理配置、抗混叠重采样、逐谱RMS归一化。

    传统解调预处理在密波数网格上先补偿包络，再乘 Hann 窗。这里不生成距离或相位真值。
    返回真实重采样坐标；多相重采样一般不会同时保留原网格的两个端点。
    默认A保留旧接口兼容性；正式导出入口读取训练配置指定的方案。
    """
    if scheme not in ('A','B','C'):
        raise ValueError('Unknown preprocessing scheme')
    k, y = uniform_k(wavelength_nm,intensity)
    if scheme == 'A':
        signal = y-y.mean()
    else:
        signals, _ = prepare(k,y,config)
        signal = signals[scheme]
    nout = config.model_points
    # 抗混叠滤波后改变采样率；不能用简单隔点抽取代替。
    out = resample_poly(signal, nout, len(k), padtype='line')
    ko = k[0]+np.arange(len(out))*(k[1]-k[0])*len(k)/nout
    # 重采样后再次去均值，再统一尺度；C的边缘此时不再严格为零，这是中心化的结果。
    centered = out-out.mean()
    scale = float(np.sqrt(np.mean(centered**2)))
    if scale < 1e-15:
        raise ValueError('Constant or nearly zero spectrum')
    return (centered/scale).astype(np.float32), ko, dict(mean=float(out.mean()),rms=scale,scheme=scheme)
