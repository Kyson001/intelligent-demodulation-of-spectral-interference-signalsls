"""预处理独立正确性检查：单位、步骤对应、相位保持及错误输入。"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import resample_poly, windows

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('preprocess_c',ROOT/'scripts/preprocess_c.py')
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
sys.path.insert(0,str(ROOT/'src'))
from demodulation.preprocessing import model_input, fft_peak, extract_phase, wrap_phase


def spectrum(opd=.00624,phase=.7):
    w=np.linspace(1510,1610,10001)
    k=2*np.pi/(w*1e-9)
    envelope=np.exp(-.5*((w-1560)/16)**2)
    return w,envelope*(1+.8*np.cos((k-2*np.pi/(1560e-9))*opd+phase))


def test_c_matches_common_module_and_stage_definitions():
    w,y=spectrum()
    r=module.preprocess_spectrum(w,y)
    x,k,meta=model_input(w,y,'C')
    np.testing.assert_array_equal(r['X'],x)
    np.testing.assert_array_equal(r['k_model'],k)
    np.testing.assert_allclose(r['windowed_c'],r['envelope_corrected']*r['hann'])
    np.testing.assert_allclose(r['resampled_c'],resample_poly(r['windowed_c'],2048,10001,padtype='line'))
    assert r['denominator'].min()>0 and len(r['X'])==2048
    assert abs(r['X'].mean())<1e-6 and abs(np.mean(r['X']**2)-1)<1e-5


def test_dbm_conversion_is_explicit():
    w,y=spectrum()
    a=module.preprocess_spectrum(w,y)
    b=module.preprocess_spectrum(w,10*np.log10(y),module.CConfig(intensity_unit='dbm'))
    np.testing.assert_allclose(a['X'],b['X'],rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('opd',[.00231,.00624,.01028])
@pytest.mark.parametrize('phase',[-np.pi+.002,np.pi-.002])
def test_c_resampling_retains_known_opd_phase(opd,phase):
    w,y=spectrum(opd,phase)
    r=module.preprocess_spectrum(w,y)
    k=r['k_model']; x=r['X']
    _,carrier,_,_=fft_peak(k,x)
    lam=2*np.pi/k*1e9
    env=np.exp(-.5*((lam-1560)/16)**2)
    s=(env>.15)&(windows.hann(len(k),sym=False)>.2)
    fit=extract_phase(k,x,s,env**2,carrier)
    assert abs(fit['opd_m']-opd)<2e-8
    assert abs(wrap_phase(fit['phase_rad']-phase))<.005


def test_invalid_inputs_are_not_silently_repaired():
    w,y=spectrum()
    with pytest.raises(ValueError,match='重复波长'):
        module.preprocess_spectrum(np.ones_like(w),y)
    y[0]=np.nan
    with pytest.raises(ValueError,match='NaN'):
        module.preprocess_spectrum(w,y)
    with pytest.raises(ValueError):
        module.CConfig(envelope_floor_fraction=0).validate()
    with pytest.raises(ValueError):
        module.CConfig(background_sigma_points=0).validate()
    with pytest.raises(ValueError):
        module.preprocess_spectrum(w,np.ones_like(w))
