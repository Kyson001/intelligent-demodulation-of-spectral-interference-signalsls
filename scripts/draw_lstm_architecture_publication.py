"""根据现有model.py重绘科研风LSTM架构图，统一布局、类型和图形层级。"""
# 图形契约：准确表达center_global双任务网络，仅调整视觉表现；无性能数值。
# 四个阶段属于同一结构链。示意矩阵/节点数量不代表真实维度，全部维度有标注。
# 输入曲线取现有训练集第一条已处理光谱，2048点全部用于绘制，仅作幅度缩放。
# 保留矩阵、递归单元和全连接神经元表现，不加入网络实现中不存在的层。
# 科研汇报宽图，PNG预览、SVG/PDF保留可编辑文字；非特定期刊的版心格式。
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description='绘制科研风LSTM网络结构图')
parser.add_argument('--data-dir', type=Path, default=ROOT / 'datasets' / 'processed', help='包含train.npz的数据目录')
parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'lstm_architecture', help='PNG/SVG/PDF输出目录')
args = parser.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                     'font.size': 12, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'axes.unicode_minus': False})
fig, ax = plt.subplots(figsize=(17.6, 6.9))
fig.subplots_adjust(left=.015, right=.99, bottom=.025, top=.98)
fig.patch.set_facecolor('white')
ax.set(xlim=(0, 235), ylim=(0, 89), aspect='equal')
ax.axis('off')
INK = '#263D4A'
MUTED = '#6C7F8C'
FLOW = '#667E8D'
STROKE = '#8198A8'
BLUE = '#527F9C'
READS = {'mean': '#88AEA5', 'std': '#91ACC1', 'center': '#C8B88F'}


def tint(color, white_fraction):
    return tuple((1-white_fraction)*c + white_fraction for c in to_rgb(color))


def label(x, y, value, size=12, ha='center', color=INK, weight='normal'):
    return ax.text(x, y, value, fontsize=size, ha=ha, va='center', color=color,
                   fontweight=weight, linespacing=1.4, zorder=10)


def wire(points, color=FLOW, width=.8, alpha=1):
    ax.plot([p[0] for p in points], [p[1] for p in points],
            lw=width, color=color, alpha=alpha, zorder=1, solid_capstyle='round')


def arrow(start, end, color=FLOW, width=.85, dashed=False, scale=12):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=scale,
                  linewidth=width, color=color, shrinkA=.5, shrinkB=.5,
                  linestyle=(0, (2.4, 2.5)) if dashed else '-', zorder=3))


def tensor(x, y, width, height, fill, columns=6, rows=10, stacked=True):
    if stacked:
        for dx, dy, fraction in [(1.7, 2.2, .55), (.85, 1.1, .30)]:
            ax.add_patch(Rectangle((x+dx, y+dy), width, height,
                         fc=tint(fill, fraction), ec='#91A5B3', lw=.65, zorder=1))
    ax.add_patch(Rectangle((x, y), width, height, fc=fill, ec=STROKE, lw=.8, zorder=2))
    for i in range(1, columns):
        ax.plot([x+width*i/columns]*2, [y, y+height], color=BLUE, lw=.35, alpha=.32, zorder=3)
    for j in range(1, rows):
        ax.plot([x, x+width], [y+height*j/rows]*2, color=BLUE, lw=.35, alpha=.32, zorder=3)


def recurrent_cell(x, y, fill):
    ax.add_patch(FancyBboxPatch((x-4.8, y-3.2), 9.6, 6.4,
                 boxstyle='round,pad=.05,rounding_size=.65',
                 fc=fill, ec='#9BB2C2', lw=.75, zorder=4))
    label(x, y, 'LSTM', size=10.5)


# 标题与阶段导航：不把每个小模块都做成粗重标题。
label(117.5, 85, 'BiLSTM光程差与相位联合解调框架', size=18, weight='medium')
for x, tag, title in [(5, 'a', '输入与序列组织'), (54, 'b', '双向序列编码'),
                       (129, 'c', '特征读出与融合'), (165, 'd', '联合回归')]:
    label(x, 77, tag, size=12.5, ha='left', color=BLUE, weight='bold')
    label(x+4.5, 77, title, size=12, ha='left', weight='medium')
for x in [49, 125, 161]:
    wire([(x, 12), (x, 72.5)], color='#E3EAF0', width=.55)

# A. 一条实际C光谱和光谱序列矩阵。
label(13, 66, '预处理光谱', size=11.5)
tensor(5, 36, 16, 20, '#F1F5F8', columns=1, rows=1)
with np.load(args.data_dir/'train.npz', allow_pickle=False) as archive:
    spectrum = archive['X'][0].astype(float)
assert spectrum.shape == (2048,) and np.isfinite(spectrum).all()
curve_x = 6 + np.linspace(0, 14, spectrum.size)
curve_y = 46 + 6.5 * spectrum / np.max(np.abs(spectrum))
ax.plot(curve_x, curve_y, color='#4E8199', lw=.48, zorder=4)
label(13, 29, '[B, 2048]', size=11)
label(13, 22.5, '预处理输入', size=10, color=MUTED)
arrow((23.3, 46), (30.5, 46))

label(38, 66, '连续8点分块', size=11.5)
tensor(31, 30, 13, 30, '#DDE8F0', columns=8, rows=10)
label(38, 23, '[B, 256, 8]', size=11)
label(38, 17.2, 'Reshape', size=10, color=MUTED)
arrow((46.2, 46), (52, 46))

# B. 以一个轻量外框组织双向LSTM，展开的3个位置只是代表性序列步。
ax.add_patch(FancyBboxPatch((52.5, 27.5), 49.5, 39,
             boxstyle='round,pad=.15,rounding_size=1.2',
             fc='#F5F8FB', ec='#D4E0E9', lw=.75, zorder=0))
label(77.3, 62.5, 'BiLSTM · 单层', size=12.5, weight='medium')
for cx, step in [(61, 1), (77, 2), (94, 256)]:
    label(cx, 55.5, f't={step}', size=9.5, color=MUTED)
    recurrent_cell(cx, 49, '#D9E6EF')
    recurrent_cell(cx, 37.5, '#E4EAF2')
arrow((66, 49), (72, 49), color=BLUE, scale=10)
arrow((82, 49), (89, 49), color=BLUE, dashed=True, scale=10)
arrow((89, 37.5), (82, 37.5), color=BLUE, dashed=True, scale=10)
arrow((72, 37.5), (66, 37.5), color=BLUE, scale=10)
label(77.3, 20.8, '正向32维 + 反向32维', size=10.5, color=MUTED)
arrow((102.5, 46), (108, 46))

label(114, 68.5, '编码特征', size=11.5)
tensor(108.5, 29.5, 11.8, 33, '#BED2E2', columns=7, rows=10)
label(114.4, 22, '[B, 256, 64]', size=11)

# C. 三路读出颜色与拼接向量三段严格对应。
wire([(122.8, 46), (127.5, 46), (127.5, 61), (133.5, 61)])
wire([(127.5, 46), (127.5, 27), (133.5, 27)])
arrow((127.5, 45), (133.5, 45))
arrow((131.2, 61), (133.5, 61))
arrow((131.2, 27), (133.5, 27))
for key, cy, title in [('mean', 61, '均值'), ('std', 45, '标准差'),
                        ('center', 27, '参考点')]:
    tensor(134, cy-4.5, 3.6, 9, tint(READS[key], .30), columns=1, rows=5, stacked=False)
    label(135.8, cy+7.5, title, size=10.5)
label(135.8, 14.5, '1560 nm', size=9.5, color=MUTED)
label(142.5, 8.8, '三路各 [B, 64]', size=10, color=MUTED)

for key, bottom in [('mean', 53.5), ('std', 39), ('center', 24.5)]:
    tensor(152, bottom, 3.9, 14.5, tint(READS[key], .18), columns=1, rows=5, stacked=False)
label(154, 71.5, 'Concat', size=11)
label(154, 19, '[B, 192]', size=11)
arrow((138, 61), (151.5, 60.75))
arrow((138, 45), (151.5, 46.25))
arrow((138, 27), (151.5, 31.75))

# D. 轻量化密集连接，7个节点示意共享64维向量。
label(178, 71.5, 'Linear 192 → 64', size=11)
hidden_y = np.linspace(30, 65, 7)
for sy in np.linspace(26, 66.5, 7):
    for hy in hidden_y:
        wire([(156.2, sy), (177, hy)], color='#93A4B0', width=.4, alpha=.48)
for hy in hidden_y:
    ax.add_patch(Circle((178, hy), .9, fc='#BDD7E3', ec='#618BA2', lw=.7, zorder=4))
label(178, 23, '[B, 64]', size=11)
label(178, 15.5, 'LayerNorm · GELU\nDropout p = 0.1', size=10, color=MUTED)

# 两个回归头：一个标准化OPD，两个相位分量。保留真实输出语义。
for hy in hidden_y:
    for target_y in [60.5, 39, 28]:
        wire([(179.2, hy), (207.5, target_y)], color='#93A4B0', width=.45, alpha=.52)
ax.add_patch(Circle((209, 60.5), 1.15, fc='#608F92', ec='#426F75', lw=.8, zorder=5))
for py in [39, 28]:
    ax.add_patch(Circle((209, py), 1.15, fc='#A995B7', ec='#806C91', lw=.8, zorder=5))
for py in [60.5, 39, 28]:
    arrow((210.5, py), (216, py), scale=10)
label(217.5, 69, 'OPD头 64 → 1\n[B]', size=10, ha='left', color=MUTED)
label(217.5, 60.5, 'OPD z', size=12.5, ha='left')
label(217.5, 53.5, '标准化光程差', size=9.5, ha='left', color=MUTED)
label(217.5, 46.5, '相位头 64 → 2\n[B, 2]', size=10, ha='left', color=MUTED)
label(217.5, 39, 'sinφ', size=12.5, ha='left')
label(217.5, 28, 'cosφ', size=12.5, ha='left')

label(117.5, 3, 'B为批量大小；矩阵网格及神经元数量仅作示意，真实维度以标注为准。',
      size=9.5, color=MUTED)

fig.savefig(OUT/'LSTM网络结构图_科研版.png', dpi=600)
fig.savefig(OUT/'LSTM网络结构图_科研版.svg', dpi=600)
fig.savefig(OUT/'LSTM网络结构图_科研版.pdf', dpi=600)
plt.close(fig)
print(OUT/'LSTM网络结构图_科研版.png')
