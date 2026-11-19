"""Run the paired full-dataset comparison and synthetic controls from raw CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from demodulation.preprocessing import Config, compare, model_input, wrap_phase

METRICS=['fft_opd_m','phase_opd_m','phase_rad','phase_residual_mrad','fwhm_um','band_energy_fraction']


def write_csv(path,rows):
    if not rows:
        return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trace_arrays(result):
    saved={'k':result['k'],'raw_k':result['raw_k']}
    saved.update(result['aux'])
    for name,m in result['methods'].items():
        for key in ['signal','fft_axis','fft_amplitude']:
            saved[f'{name}_{key}']=m[key]
        saved[f'{name}_phase_curve']=m['phase']['phase_curve']
        saved[f'{name}_residual']=m['phase']['residual']
        for key in METRICS:
            saved[f'{name}_{key}']=m[key]
    return saved


def synthesize(wavelength,opd,phi,case,seed):
    rng=np.random.default_rng(seed)
    k=2*np.pi/(wavelength*1e-9)
    kc=2*np.pi/(1560e-9)
    envelope=np.exp(-.5*((wavelength-1560)/16)**2)
    phase=(k-kc)*opd+phi
    fringe=.8*envelope*np.cos(phase)
    ideal=envelope+fringe
    # Fixed SNR referenced to mean squared FRINGE, not DC-inclusive total power.
    snr={'Clean':None,'White 40 dB':40,'White 20 dB':20,'Offset + 40 dB':40}[case]
    offset=np.zeros_like(ideal)
    if case=='Offset + 40 dB':
        offset=.12+.04*(wavelength-1560)/50
    sigma=0 if snr is None else np.sqrt(np.mean(fringe**2)/10**(snr/10))
    noise=rng.normal(0,sigma,len(ideal))
    return ideal+offset+noise, ideal+offset


def synthetic(config,out):
    w=np.linspace(1510,1610,10001)
    rows=[]
    # Independent synthetic controls only: no parameters tuned against these outcomes.
    for case in ['Clean','White 40 dB','White 20 dB','Offset + 40 dB']:
        for opd in [.00231,.00624,.01028]:
            for phi in [-np.pi+.01,-.6,.7,np.pi-.01]:
                for seed in range(5):
                    y,clean=synthesize(w,opd,phi,case,seed)
                    r=compare(w,y,config)
                    rclean=compare(w,clean,config)
                    clean_inputs={m:model_input(w,clean,m,config)[0] for m in ['A','B']}
                    for name,m in r['methods'].items():
                        row=dict(case=case,true_opd_mm=opd*1e3,true_phase_rad=phi,seed=seed,method=name,
                                 opd_error_nm=(m['phase_opd_m']-opd)*1e9,
                                 phase_error_mrad=float(wrap_phase(m['phase_rad']-phi)*1e3),
                                 fft_error_um=(m['fft_opd_m']-opd)*1e6,
                                 phase_residual_mrad=m['phase_residual_mrad'])
                        if name in ('A','B'):
                            xn,_,_=model_input(w,y,name,config)
                            row['normalized_input_rmse']=float(np.sqrt(np.mean((xn-clean_inputs[name])**2)))
                        else:
                            # Diagnostic only; C intentionally not a training input.
                            norm=lambda a:a/max(np.sqrt(np.mean(a*a)),1e-15)
                            row['normalized_input_rmse']=float(np.sqrt(np.mean((norm(m['signal'])-norm(rclean['methods'][name]['signal']))**2)))
                        rows.append(row)
    write_csv(out/'synthetic_metrics.csv',rows)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=ROOT/'datasets/raw')
    parser.add_argument('--out',type=Path,default=ROOT/'results/preprocessing')
    parser.add_argument('--limit',type=int,default=0,help='Debug only; official run uses all files')
    args=parser.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    config=Config(**json.loads((ROOT/'configs/preprocessing_comparison.json').read_text(encoding='utf-8')))
    files=sorted(args.data_dir.glob('file*.csv'))
    if args.limit:
        files=files[:args.limit]
    if not files:
        raise ValueError('No CSV spectra found')
    label_files=list((args.data_dir/'_cache').glob('d_mm*.npy'))
    labels=np.load(label_files[0]) if label_files else None
    rows,failures,hashes=[],[],[]
    start=time.perf_counter()
    axis=None
    for i,path in enumerate(files):
        raw=path.read_bytes()
        hashes.append((path.name,hashlib.sha256(raw).hexdigest()))
        a=np.loadtxt(path,delimiter=',',skiprows=1)
        if axis is None:
            axis=a[:,0].copy()
        if not np.array_equal(axis,a[:,0]):
            raise ValueError(f'Unexpected wavelength grid: {path.name}')
        try:
            r=compare(a[:,0],a[:,1],config)
            for name,m in r['methods'].items():
                row=dict(sample_id=path.stem,method=name,**{key:m[key] for key in METRICS})
                row['legacy_d_mm']=float(labels[int(path.stem[4:])-1]) if labels is not None else ''
                row['delta_opd_vs_A_nm']=(m['phase_opd_m']-r['methods']['A']['phase_opd_m'])*1e9
                row['delta_phase_vs_A_mrad']=float(wrap_phase(m['phase_rad']-r['methods']['A']['phase_rad'])*1e3)
                rows.append(row)
            if path.stem in ['file0001','file2000','file4000']:
                np.savez_compressed(out/f'{path.stem}_traces.npz',wavelength_nm=a[:,0],raw=a[:,1],**trace_arrays(r))
        except (ValueError,np.linalg.LinAlgError) as e:
            failures.append(dict(sample_id=path.stem,error=str(e)))
        if (i+1)%250==0:
            print(f'{i+1}/{len(files)} spectra; {time.perf_counter()-start:.1f}s',flush=True)
    write_csv(out/'real_metrics.csv',rows)
    (out/'failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    (out/'input_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    synthetic_rows=synthetic(config,out)
    summary={'config':config.to_dict(),'sample_count':len(files),'failures':len(failures),
             'real_label_status':'Legacy distance only; original workbook missing; no independent phase truth',
             'synthetic_design':'4 cases x 3 OPDs x 4 phases x 5 seeds = 240 paired spectra',
             'real':{},'synthetic':{}}
    for name in ['A','B','C']:
        selected=[r for r in rows if r['method']==name]
        summary['real'][name]={}
        for metric in METRICS+['delta_opd_vs_A_nm','delta_phase_vs_A_mrad']:
            values=np.array([r[metric] for r in selected])
            summary['real'][name][metric]={'median':float(np.median(values)),
                                           'p95_abs':float(np.quantile(abs(values),.95)),
                                           'max_abs':float(np.max(abs(values)))}
        for case in ['Clean','White 40 dB','White 20 dB','Offset + 40 dB']:
            subset=[r for r in synthetic_rows if r['method']==name and r['case']==case]
            summary['synthetic'][f'{case}/{name}']={key:{'median_abs':float(np.median(abs(np.array([r[key] for r in subset])))),
                                                     'p95_abs':float(np.quantile(abs(np.array([r[key] for r in subset])),.95))}
                                                   for key in ['opd_error_nm','phase_error_mrad','normalized_input_rmse']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Completed in',round(time.perf_counter()-start,1),'s. Failures:',len(failures),flush=True)


if __name__=='__main__':
    main()
