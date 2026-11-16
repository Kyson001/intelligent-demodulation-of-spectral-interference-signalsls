import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from demodulation.preprocessing import Config, compare, model_input, uniform_k, wrap_phase, fft_peak, extract_phase


def signal(opd, phase):
    wavelength = np.linspace(1510,1610,10001)
    k = 2*np.pi/(wavelength*1e-9)
    kc = 2*np.pi/(1560e-9)
    envelope = np.exp(-.5*((wavelength-1560)/16)**2)
    return wavelength, envelope*(1+.8*np.cos((k-kc)*opd+phase))


@pytest.mark.parametrize('opd',[.00231,.00624,.01028])
@pytest.mark.parametrize('phase',[-np.pi+.002,.37,np.pi-.002])
def test_known_opd_and_phase(opd,phase):
    w,y = signal(opd,phase)
    result = compare(w,y)
    for m in result['methods'].values():
        assert abs(m['phase_opd_m']-opd)<2e-8
        assert abs(wrap_phase(m['phase_rad']-phase))<.005


def test_sort_and_input_coordinates():
    w,y = signal(.01028,.3)
    k,s = uniform_k(w,y)
    kr,sr = uniform_k(w[::-1],y[::-1])
    np.testing.assert_array_equal(k,kr)
    np.testing.assert_array_equal(s,sr)
    x,ko,_ = model_input(w,y)
    assert len(x)==2048 and np.isfinite(x).all()
    assert ko[0]>=k[0] and ko[-1]<=k[-1]
    assert abs(x.mean())<1e-6
    assert abs(np.mean(x*x)-1)<1e-5
    with pytest.raises(ValueError):
        model_input(w,y,'C')
    with pytest.raises(ValueError):
        uniform_k(np.ones(10001),y)


def test_hann_and_envelope_definitions():
    w,y = signal(.00624,.3)
    result=compare(w,y)
    np.testing.assert_allclose(result['methods']['C']['signal'],
                               result['methods']['B']['signal']*result['aux']['hann'])
    assert result['aux']['denominator'].min()>0


@pytest.mark.parametrize('opd',[.00231,.00624,.01028])
@pytest.mark.parametrize('phase',[-np.pi+.002,np.pi-.002])
def test_resampled_network_input_retains_opd_and_phase(opd,phase):
    w,y=signal(opd,phase)
    x,k,_=model_input(w,y)
    from scipy.signal.windows import hann
    _,carrier,_,_=fft_peak(k,x*hann(len(k),sym=False))
    wavelength=2*np.pi/k*1e9
    envelope=np.exp(-.5*((wavelength-1560)/16)**2)
    support=(envelope>.15)&(hann(len(k),sym=False)>.2)
    estimate=extract_phase(k,x,support,envelope**2,carrier)
    assert abs(estimate['opd_m']-opd)<2e-8
    assert abs(wrap_phase(estimate['phase_rad']-phase))<.005
