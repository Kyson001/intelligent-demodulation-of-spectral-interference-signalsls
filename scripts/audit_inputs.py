"""Audit supplied CSVs and compare legacy cache labels; never treat cache as truth."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'analysis/data_4000'
OUT = ROOT / 'analysis'
files = sorted(DATA.glob('file*.csv'))
cache = DATA / '_cache'
labels = np.load(next(cache.glob('d_mm*.npy')))
cached = np.load(next(cache.glob('spectra*.npy')), mmap_mode='r')
ids = np.load(next(cache.glob('file_ids*.npy')))
rows, examples, hashes = [], [], {}
axis = None
for i, path in enumerate(files):
    a = np.loadtxt(path, delimiter=',', skiprows=1)
    x, y = a.T
    if axis is None:
        axis = x.copy()
    digest = hashlib.sha256(y.tobytes()).hexdigest()
    hashes.setdefault(digest, []).append(path.name)
    row = dict(file=path.name, points=len(y), finite=bool(np.isfinite(a).all()),
               same_axis=bool(np.array_equal(x, axis)), minimum=float(y.min()),
               maximum=float(y.max()), zeros=int(np.count_nonzero(y == 0)),
               cache_matches=bool(np.array_equal(y.astype(np.float32), cached[i])),
               cache_max_abs_difference=float(np.max(np.abs(y-cached[i]))),
               cache_label_mm=float(labels[i]))
    rows.append(row)
    if i % 100 == 0 or i == len(files)-1:
        k = (2*np.pi/(x*1e-9))[::-1]
        ku = np.linspace(k[0], k[-1], len(k))
        yu = np.interp(ku, k, y[::-1])
        # Remove slowly varying DC using a broad Gaussian; selected examples only.
        from scipy.ndimage import gaussian_filter1d
        yp = (yu-gaussian_filter1d(yu, 100))*np.hanning(len(yu))
        power = abs(np.fft.rfft(yp))
        peaks, _ = find_peaks(power)
        peaks = peaks[peaks > 2]
        j = peaks[np.argmax(power[peaks])]
        delta = .5*(power[j-1]-power[j+1])/(power[j-1]-2*power[j]+power[j+1])
        opd = (j+delta)*2*np.pi/(len(ku)*(ku[1]-ku[0]))
        examples.append(dict(file=path.name, legacy_label_mm=float(labels[i]),
                             fft_opd_mm=float(opd*1e3), ratio=float(opd*1e3/labels[i])))
summary = dict(csv_count=len(files), points=sorted(set(r['points'] for r in rows)),
               wavelength_nm=[float(axis.min()),float(axis.max())], step_nm=float(np.median(np.diff(axis))),
               all_finite=all(r['finite'] for r in rows), all_same_axis=all(r['same_axis'] for r in rows),
               all_cache_match=all(r['cache_matches'] for r in rows),
               cache_max_abs_difference=max(r['cache_max_abs_difference'] for r in rows),
               cache_matches_at_csv_precision=all(r['cache_max_abs_difference'] < 5e-7 for r in rows),
               intensity_range=[min(r['minimum'] for r in rows),max(r['maximum'] for r in rows)],
               cache_label_range_mm=[float(labels.min()),float(labels.max())],
               ids_match=bool(np.array_equal(ids,np.arange(1,len(files)+1))),
               exact_duplicate_groups=[v for v in hashes.values() if len(v)>1],
               label_provenance='Legacy cache refers to datasets/label_4000.xlsx, not supplied; unverified',
               fft_examples=examples)
(OUT/'data_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'data_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['fft_examples','exact_duplicate_groups']},ensure_ascii=False,indent=2))
print('Duplicate groups:',len(summary['exact_duplicate_groups']))
print('FFT examples:',examples[:4],examples[-2:])
