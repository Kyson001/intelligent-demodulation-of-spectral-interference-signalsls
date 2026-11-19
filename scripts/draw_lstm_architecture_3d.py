"""按实际LSTM网络新绘横向特征张量/神经元架构图，仅继承参考图表现风格。"""
# 图形约定：一条端到端结构链，标注真实张量维度，不展示虚构性能。
# 矩阵网格、batch叠片和神经元只作示意，数量不代表实际维度；不包含CNN或分类模块。
# 输入曲线来自现有训练集第一条C光谱，完整2048点仅在画面中缩放。
# Python/matplotlib；教学/PPT宽图；文字可编辑SVG/PDF，PNG用于直接查看。
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description='绘制LSTM立体张量和神经元结构图')
parser.add_argument('--data-dir', type=Path, default=ROOT / 'datasets' / 'processed', help='包含train.npz的数据目录')
parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'lstm_architecture', help='PNG/SVG/PDF输出目录')
args = parser.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                     'font.size': 14, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'axes.unicode_minus': False})
fig, ax = plt.subplots(figsize=(21, 8.8))
fig.subplots_adjust(left=.012, right=.992, bottom=.025, top=.99)
fig.patch.set_facecolor('white')
ax.set(xlim=(0, 214), ylim=(0, 90), aspect='equal')
ax.axis('off')
INK = '#21313D'
EDGE = '#526678'
LINK = '#59646D'
COLORS = {'sequence': '#DCEAF7', 'encoded': '#BFDCEF',
          'mean': '#B9D8BF', 'std': '#BFD7EC', 'center': '#F0D79C',
          'hidden': '#43A5C8', 'opd': '#117B9E', 'phase': '#9A74B6'}


def text(x, y, label, size=14, ha='center', color=INK, weight='normal'):
    return ax.text(x, y, label, ha=ha, va='center', fontsize=size,
                   color=color, fontweight=weight, linespacing=1.35, zorder=10)


def arrow(x1, y1, x2, y2, color=EDGE, lw=1.35, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=15, lw=lw, color=color,
                 linestyle=(0, (3, 3)) if dashed else '-',
                 shrinkA=1, shrinkB=1, zorder=3))


def wire(points, color=EDGE, lw=1.15):
    ax.plot([p[0] for p in points], [p[1] for p in points],
            color=color, lw=lw, zorder=1)


def matrix(x, y, w, h, fill, columns=8, rows=10, depth=True):
    """叠片透视张量，格子仅用于示意。"""
    if depth:
        for dx, dy in [(1.8, 2.1), (.9, 1.05)]:
            ax.add_patch(Rectangle((x+dx, y+dy), w, h, fc=fill,
                                    ec=EDGE, lw=.9, zorder=1))
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=EDGE, lw=1.1, zorder=2))
    for i in range(1, columns):
        ax.plot([x+w*i/columns]*2, [y, y+h], color=EDGE, lw=.45, alpha=.7, zorder=3)
    for j in range(1, rows):
        ax.plot([x, x+w], [y+h*j/rows]*2, color=EDGE, lw=.45, alpha=.7, zorder=3)


def cell(cx, cy, step, fill):
    ax.add_patch(FancyBboxPatch((cx-4, cy-4), 8, 8,
                 boxstyle='round,pad=0.1,rounding_size=.7', fc=fill, ec=EDGE, lw=1.1, zorder=4))
    text(cx, cy, f'LSTM\nt={step}', size=11)


text(107, 86, '光谱干涉信号的光程差与相位联合解调网络', size=23, weight='bold')

# 1. 光谱输入，立体平面中绘制实际2048点曲线，不隐含额外通道。
text(12, 76, '光谱输入', size=16, weight='bold')
matrix(3, 35, 18, 24, '#EFF3F7', columns=1, rows=1)
with np.load(args.data_dir/'train.npz', allow_pickle=False) as archive:
    spectrum = archive['X'][0].astype(float)
assert spectrum.shape == (2048,) and np.isfinite(spectrum).all()
curve_x = 4 + np.linspace(0, 16, spectrum.size)
curve_y = 47 + 7 * spectrum / np.max(np.abs(spectrum))
ax.plot(curve_x, curve_y, color='#397897', lw=.55, zorder=5)
text(12, 29, '[B, 2048]', size=14)
text(12, 22, 'C方案光谱', size=12, color=EDGE)
arrow(23.3, 47, 28, 47)

# 2. reshape得到256个8点块。
text(36, 76, '序列重排', size=16, weight='bold')
matrix(29, 29, 14, 36, COLORS['sequence'], columns=8, rows=12)
text(36, 22, '[B, 256, 8]', size=14)
text(36, 15, '连续8点 / 步', size=12, color=EDGE)
arrow(45.4, 47, 50.5, 47)

# 3. 双向LSTM展开代表性序列位置，隐藏维数每方向32。
text(69, 76, 'BiLSTM编码器', size=16, weight='bold')
for cx, step in [(55, 1), (67, 2), (81, 256)]:
    cell(cx, 56, step, '#E7E0F0')
    cell(cx, 39, step, '#DCE9F1')
arrow(59.2, 56, 62.8, 56)
arrow(71.2, 56, 76.8, 56, dashed=True)
arrow(76.8, 39, 71.2, 39, dashed=True)
arrow(62.8, 39, 59.2, 39)
text(69, 65, '正向 →', size=12, color=EDGE)
text(69, 30, '← 反向', size=12, color=EDGE)
text(69, 22, '1层 · 每方向32维', size=12, color=EDGE)
wire([(50.5, 47), (50.5, 56), (50.7, 56)])
wire([(50.5, 47), (50.5, 26), (87.2, 26), (87.2, 39)])
arrow(87.2, 39, 85.3, 39)
arrow(86, 47, 92, 47)

# 4. 输出序列特征张量；64来自两个方向各32维的拼接。
text(99, 76, '双向序列特征', size=15, weight='bold')
matrix(92, 29, 14, 36, COLORS['encoded'], columns=8, rows=12)
text(99, 22, '[B, 256, 64]', size=14)

# 5. 三种读出形成三条64维向量，颜色对应Concat中的三段。
wire([(108.4, 47), (112, 47), (112, 67), (119.4, 67)])
wire([(112, 47), (112, 25), (119.4, 25)])
arrow(112, 47, 119.4, 47)
arrow(116.5, 67, 119.4, 67)
arrow(116.5, 25, 119.4, 25)
for name, cy, label in [('mean', 67, '全局均值'), ('std', 47, '全局标准差'),
                        ('center', 25, '1560 nm参考点')]:
    matrix(120, cy-6, 4, 12, COLORS[name], columns=1, rows=6, depth=False)
    text(122, cy+10, label, size=12.5)
text(122, 13, '三路各 [B, 64]', size=12.5, color=EDGE)

# 6. 拼接为192维。顶到下的颜色对应mean、std、center的实际拼接顺序。
text(143, 76, '特征拼接', size=15, weight='bold')
for name, low in [('mean', 55), ('std', 40), ('center', 25)]:
    matrix(141, low, 4, 15, COLORS[name], columns=1, rows=5, depth=False)
arrow(124.5, 67, 140.5, 62.5)
arrow(124.5, 47, 140.5, 47.5)
arrow(124.5, 25, 140.5, 32.5)
text(143, 20, '[B, 192]', size=13.5)

# 7. 用神经元和全连接线表现192→64矩阵，画出的8个圆点仅作64维示意。
text(164, 76, '共享全连接层', size=15, weight='bold')
text(164, 70.5, 'Linear 192 → 64', size=12, color=EDGE)
hidden_y = np.linspace(29, 66, 8)
source_y = np.linspace(26.5, 68.5, 8)
for sy in source_y:
    for hy in hidden_y:
        ax.plot([145.3, 162], [sy, hy], color=LINK, lw=.5, alpha=.48, zorder=1)
for hy in hidden_y:
    ax.add_patch(Circle((163, hy), 1.05, fc=COLORS['hidden'], ec='#347893', lw=.85, zorder=5))
text(163, 23.5, '[B, 64]', size=13.5)
text(163, 15, 'LayerNorm · GELU\nDropout(0.1)', size=11.5, color=EDGE)

# 8. 两个回归输出头：1个OPD预测，2个相位分量，没有Softmax/分类标签。
text(193, 76, '双任务输出', size=15, weight='bold')
for hy in hidden_y:
    for oy in [63, 42, 30]:
        ax.plot([164.2, 189.5], [hy, oy], color=LINK, lw=.55, alpha=.55, zorder=1)
ax.add_patch(Circle((191, 63), 1.4, fc=COLORS['opd'], ec='#175D77', lw=1., zorder=5))
for oy in [42, 30]:
    ax.add_patch(Circle((191, oy), 1.4, fc=COLORS['phase'], ec='#745588', lw=1., zorder=5))
arrow(192.6, 63, 198, 63)
arrow(192.6, 42, 198, 42)
arrow(192.6, 30, 198, 30)
text(200, 63, 'OPD z', size=13, ha='left')
text(200, 42, 'sinφ', size=13, ha='left')
text(200, 30, 'cosφ', size=13, ha='left')
text(191, 69.5, '光程差头', size=11.5)
text(200, 55, '64 → 1\n标准化OPD\n[B]', size=11, ha='left')
text(193, 19, '相位头 64 → 2\n相位分量 [B, 2]', size=11.5)

text(107, 4, 'B为批量大小；矩阵网格与神经元数量为示意，实际维度以标注为准。',
     size=12, color=EDGE)

fig.savefig(OUT/'LSTM网络结构图_立体神经元版.png', dpi=300)
fig.savefig(OUT/'LSTM网络结构图_立体神经元版.svg', dpi=600)
fig.savefig(OUT/'LSTM网络结构图_立体神经元版.pdf', dpi=600)
plt.close(fig)
print(OUT/'LSTM网络结构图_立体神经元版.png')
