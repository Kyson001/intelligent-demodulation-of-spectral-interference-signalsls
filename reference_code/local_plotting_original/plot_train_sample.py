from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description='Plot a sample from the processed training set')
p.add_argument('--index', type=int, default=0, help='Zero-based sample index (default: 0)')
p.add_argument('--save', type=Path, default=Path(__file__).resolve().parents[1] / 'analysis' / 'previews' / 'train_sample.png')
p.add_argument('--no-show', action='store_true')
a = p.parse_args()
source = Path(r'C:\Users\Kyson\Desktop\迈克尔逊光谱_C方案_分组数据集_20261117\train.npz')
with np.load(source) as data:
    if not 0 <= a.index < len(data['X']):
        p.error(f'index must be between 0 and {len(data["X"])-1}')
    x = data['X'][a.index]
    k = data['k_rad_per_m'] / 1e6
    name = str(data['sample_ids'][a.index])
fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
axes[0].plot(k, x, lw=0.8)
axes[0].set(title=f'Processed training spectrum: {name} (index {a.index})', xlabel='Wavenumber k (10^6 rad/m)', ylabel='Normalized amplitude')
center = len(k) // 2
s = slice(center - 150, center + 150)
axes[1].plot(k[s], x[s], lw=1)
axes[1].set(title='Central region (detail)', xlabel='Wavenumber k (10^6 rad/m)', ylabel='Normalized amplitude')
for ax in axes:
    ax.grid(alpha=0.3)
a.save.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.save, dpi=180)
print(f'{name}: {len(x)} points, k={k.min():.6f}..{k.max():.6f} x10^6 rad/m, amplitude={x.min():.4f}..{x.max():.4f}; saved {a.save}')
if not a.no_show:
    plt.show()
