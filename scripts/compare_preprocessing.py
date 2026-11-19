"""绘制用户整理的单谱预处理步骤，并校验与训练集一致；批量比较见 compare_preprocessing_dataset.py。"""
from pathlib import Path
import argparse, hashlib, runpy
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'datasets' / 'processed'
p = argparse.ArgumentParser(description='Compare preprocessing stages for one training sample')
p.add_argument('--index', type=int, default=0, help='Training sample index, starting at 0')
p.add_argument('--no-show', action='store_true')
p.add_argument('--data-dir', type=Path, default=DATASET, help='处理后数据目录')
p.add_argument('--raw-dir', type=Path, default=ROOT / 'datasets' / 'raw', help='原始CSV目录')
p.add_argument('--out-dir', type=Path, default=ROOT / 'analysis' / 'previews', help='绘图输出目录')
a = p.parse_args()
DATASET = a.data_dir
with np.load(DATASET / 'train.npz') as train:
    if not 0 <= a.index < len(train['X']):
        p.error(f'index must be 0..{len(train["X"])-1}')
    name = str(train['sample_ids'][a.index])
    expected_hash = str(train['source_sha256'][a.index])
    expected_x = train['X'][a.index].copy()
source = a.raw_dir / f'{name}.csv'
if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
    raise ValueError(f'Raw file hash does not match training source: {source}')
funcs = runpy.run_path(str(ROOT / 'scripts' / 'preprocess_c.py'))
wavelength, intensity = funcs['read_spectrum'](source)
r = funcs['preprocess_spectrum'](wavelength, intensity)
difference = np.max(np.abs(r['X'] - expected_x))
if not np.allclose(r['X'], expected_x, rtol=1e-6, atol=1e-7):
    raise ValueError(f'Reprocessed signal differs from train.npz (max difference: {difference}); check configuration')
k = r['k_dense'] / 1e6
km = r['k_model'] / 1e6
mean = r['quality']['mean_before_normalization']
centered = r['resampled_c'] - mean
fig, axs = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
ax = axs[0, 0]
ax.plot(k, r['intensity_k'], lw=.65, label='Resampled raw spectrum')
ax.plot(k, r['background'], lw=2, color='darkorange', label='Estimated background')
ax.set(title='1. Original spectrum and background', ylabel='Recorded intensity')
ax.legend()
ax = axs[0, 1]
ax.plot(k, r['background_removed'], lw=.7)
ax.axhline(0, color='black', lw=.7)
ax.set(title='2. Background subtracted', ylabel='Intensity difference')
ax = axs[1, 0]
# Centering shift is small, so show a magnified vertical scale around the same segment.
s = slice(len(km)//2, len(km)//2 + 45)
ax.plot(km[s], r['resampled_c'][s], label='Before centering', lw=1.5)
ax.plot(km[s], centered[s], '--', label='After centering', lw=1)
ax.set(title=f'3. Centering (detail; mean = {mean:.6g})', ylabel='Amplitude before RMS scaling')
ax.legend()
ax = axs[1, 1]
ax.plot(km, expected_x, lw=.7)
ax.axhline(0, color='black', lw=.7)
ax.set(title='4. Final training input (RMS normalized)', ylabel='Normalized amplitude')
for ax in axs.flat:
    ax.set_xlabel('Wavenumber k (10^6 rad/m)')
    ax.grid(alpha=.25)
fig.suptitle(f'{name} | training index {a.index}', fontsize=15)
out = a.out_dir / f'{name}_preprocessing_comparison.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180)
print(f'Source hash verified; max training difference: {difference:g}; mean: {mean:g}; RMS: {r["quality"]["rms_before_normalization"]:g}; saved: {out}')
if not a.no_show:
    plt.show()
