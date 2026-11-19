from pathlib import Path
import argparse, hashlib, runpy
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(r'C:\Users\Kyson\Desktop\迈克尔逊光谱_C方案_分组数据集_20261117')
p = argparse.ArgumentParser(description='Show envelope compensation and Hann window effects')
p.add_argument('--index', type=int, default=0)
p.add_argument('--no-show', action='store_true')
a = p.parse_args()
with np.load(DATASET / 'train.npz') as d:
    if not 0 <= a.index < len(d['X']):
        p.error(f'index must be 0..{len(d["X"])-1}')
    name, digest, expected = str(d['sample_ids'][a.index]), str(d['source_sha256'][a.index]), d['X'][a.index].copy()
source = ROOT / 'analysis' / 'data_4000' / f'{name}.csv'
if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
    raise ValueError('Original file does not match training dataset hash')
f = runpy.run_path(str(DATASET / '预处理.py'))
wavelength, intensity = f['read_spectrum'](source)
r = f['preprocess_spectrum'](wavelength, intensity)
if not np.allclose(r['X'], expected, rtol=1e-6, atol=1e-7):
    raise ValueError('Reconstructed training signal differs; check preprocessing parameters')
k = r['k_dense'] / 1e6
removed = r['background_removed']
corrected = r['envelope_corrected']
windowed = r['windowed_c']
weak = r['weak_mask']
peak = r['background'].max()
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
ax = axes[0]
ax.plot(k, r['background']/peak, color='#cf7a20', lw=1.8, label='Estimated background / its peak')
ax.plot(k, r['denominator']/peak, color='#8a4da0', lw=1.2, ls='--', label='Compensation denominator / peak')
ax.fill_between(k, 0, 1.05, where=weak, color='grey', alpha=.15, label='Weak region: denominator floor active')
ax.set(ylabel='Relative value', title='1. Envelope proxy and denominator floor (8% of peak)', ylim=(0,1.08))
ax.legend(loc='upper center', ncol=2, fontsize=9)
ax = axes[1]
ax.plot(k, removed/peak, color='#2d72a2', lw=.65, label='Background subtracted / background peak')
ax.plot(k, corrected, color='#df812a', lw=.65, alpha=.85, label='After envelope compensation')
ax.set(ylabel='Relative amplitude', title='2. Envelope compensation changes amplitude across the band')
ax.legend(loc='upper right', fontsize=9)
ax = axes[2]
ax.plot(k, corrected, color='#df812a', lw=.6, alpha=.7, label='Before Hann')
ax.plot(k, windowed, color='#7b55a4', lw=.9, label='After Hann')
ax.set(ylabel='Compensated amplitude', title='3. Hann window suppresses both ends')
ax.legend(loc='upper left', fontsize=9)
ax2 = ax.twinx()
ax2.plot(k, r['hann'], color='black', lw=1.2, ls='--', label='Hann weight')
ax2.set_ylabel('Hann weight')
ax2.set_ylim(0, 1.1)
ax2.legend(loc='upper right', fontsize=9)
for ax in axes:
    ax.grid(alpha=.2)
axes[-1].set_xlabel('Wavenumber k (10^6 rad/m)')
fig.suptitle(f'Envelope compensation and Hann window | {name} (train index {a.index})', fontsize=14)
out = ROOT / 'analysis' / 'previews' / f'{name}_envelope_hann.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180)
print(f'Hash verified; max training difference {np.max(np.abs(r["X"]-expected)):.3g}; weak fraction {weak.mean():.1%}; output {out}')
if not a.no_show:
    plt.show()
