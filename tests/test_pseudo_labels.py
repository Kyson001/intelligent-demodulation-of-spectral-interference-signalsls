"""已知光程差/相位的数值控制；不使用现有数据的未知真值作断言。"""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.constants import c

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from scripts.preprocess_c import preprocess_spectrum
from demodulation.pseudo_labels import LabelConfig,label_prepared,robust_peak_fit,wrap_phase
from scripts.generate_pseudo_labels import join_with_c_inputs
from scripts.preprocess_c import CConfig


def make_stages(opd,phase,noise=0):
    w=np.linspace(1510,1610,10001)
    k=2*np.pi/(w*1e-9); kc=2*np.pi/(1560e-9)
    envelope=np.exp(-.5*((w-1560)/16)**2)
    y=envelope*(1+.8*np.cos((k-kc)*opd+phase))
    y+=np.random.default_rng(31).normal(0,noise,len(y))
    return preprocess_spectrum(w,y)


@pytest.mark.parametrize('opd',[.00231,.00624,.01028])
@pytest.mark.parametrize('phase',[-np.pi+.01,.7,np.pi-.01])
def test_peak_spacing_and_center_phase_with_known_truth(opd,phase):
    row,trace=label_prepared(make_stages(opd,phase))
    assert row['opd_valid'] and row['phase_valid'], row
    assert abs(row['opd_peak_m']-opd)<5e-8
    assert abs(wrap_phase(row['phase_pseudo_rad']-phase))<.005
    assert abs(wrap_phase(row['phase_from_peak_fit_rad']-phase))<.01
    assert abs(row['delta_f_hz']*row['opd_peak_m']-c)<1e-5
    assert row['peak_count']>=8


def test_explicit_missing_peak_order_and_phase():
    config=LabelConfig()
    fc=c/(1560e-9); opd=.006; spacing=c/opd; phi=.9
    orders=np.arange(-30,31)
    frequencies=fc+spacing*(orders-phi/(2*np.pi))
    frequencies=np.delete(frequencies,20)
    result=robust_peak_fit(frequencies,config)
    assert result['missing_peaks']==1
    assert abs(result['opd_m']-opd)<1e-12
    assert abs(wrap_phase(result['peak_phase_rad']-phi))<1e-10


def test_noisy_labels_and_invalid_phase_reference():
    row,_=label_prepared(make_stages(.00624,.7,noise=.003))
    assert row['opd_valid'] and row['phase_valid']
    assert abs(row['opd_peak_m']-.00624)<2e-7
    row,_=label_prepared(make_stages(.00624,.7),LabelConfig(reference_nm=1800))
    assert row['opd_valid'] and not row['phase_valid']
    assert np.isnan(row['phase_pseudo_rad'])


def test_insufficient_peaks_and_zero_contrast_rejected():
    with pytest.raises(ValueError):
        robust_peak_fit(np.arange(4,dtype=float),LabelConfig())
    stages=make_stages(.00624,.7)
    stages['windowed_c'][:]=0
    with pytest.raises(ValueError,match='侧带'):
        label_prepared(stages)


def test_join_uses_id_hash_and_config(tmp_path):
    import json,csv
    ids=np.array(['sample_a','sample_b'])
    labels=dict(sample_ids=ids,opd_pseudo_m=np.array([.002,.003]),label_status=np.array('pseudo'))
    # 故意让输入顺序相反，检验按ID对齐而不是盲目拼接。
    inputs=tmp_path/'inputs.npz'
    np.savez(inputs,X=np.ones((2,4)),sample_ids=ids[::-1],split=np.array(['old','old']))
    (tmp_path/'manifest_C.json').write_text(json.dumps({'config':CConfig().__dict__}),encoding='utf-8')
    rows=[dict(sample_id='sample_a',source_sha256='a'),dict(sample_id='sample_b',source_sha256='b')]
    with (tmp_path/'quality_and_position_review.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['sample_id','source_sha256'])
        writer.writeheader(); writer.writerows(rows)
    output=tmp_path/'joined.npz'
    ok,_=join_with_c_inputs(inputs,labels,rows,output,CConfig())
    assert ok
    np.testing.assert_array_equal(np.load(output)['opd_pseudo_m'],[.003,.002])
    assert set(np.load(output)['split'])=={'unassigned'}
    rows[0]['source_sha256']='changed'
    ok,reason=join_with_c_inputs(inputs,labels,rows,tmp_path/'blocked.npz',CConfig())
    assert not ok and reason=='raw_source_hash_mismatch'
    assert not (tmp_path/'blocked.npz').exists()
