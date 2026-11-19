"""绘制与独立工程model.py一致的LSTM网络结构图，中文标签与张量维度。"""
# 绘图约定：说明center_global网络的数据流，不包含训练循环或性能结果。
# 单一结构示意图；LSTM只展开代表性的t=1/2/256，其余序列步由虚线表示。
# Python/matplotlib；PNG预览、SVG/PDF可编辑；全部文字至少11 pt。
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description='绘制LSTM网络的纵向教学框架图')
parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'lstm_architecture', help='PNG/SVG/PDF输出目录')
args = parser.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
    'font.size': 13, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.unicode_minus': False,
})
fig, ax = plt.subplots(figsize=(11, 12.2))
fig.subplots_adjust(left=.02, right=.98, top=.99, bottom=.01)
fig.patch.set_facecolor('white')
ax.set(xlim=(0, 120), ylim=(-15, 125))
ax.axis('off')
ink, line = '#183747', '#4C6271'
blue_fill, blue_edge = '#EDF4FA', '#4F7EA5'
green_fill, green_edge = '#EEF6F2', '#60917D'


def box(cx, cy, width, height, color, edge, text=None, fontsize=13, lw=1.3):
    patch = FancyBboxPatch((cx-width/2, cy-height/2), width, height,
                           boxstyle='round,pad=0.15,rounding_size=1.2',
                           facecolor=color, edgecolor=edge, linewidth=lw)
    ax.add_patch(patch)
    if text:
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
                color=ink, linespacing=1.38)
    return patch


def arrow(a, b, color=line, dashed=False):
    patch = FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=15,
                            linewidth=1.3, color=color,
                            linestyle=(0, (3, 3)) if dashed else '-',
                            shrinkA=0, shrinkB=0)
    ax.add_patch(patch)


def wire(points):
    ax.plot([p[0] for p in points], [p[1] for p in points], color=line, lw=1.3,
            solid_capstyle='round', zorder=0)


ax.text(60, 121, 'LSTM光程差与相位联合解调网络', fontsize=21, fontweight='bold',
        color=ink, ha='center', va='center')
ax.text(60, 116.5, 'B 为批量大小；每条光谱独立输入', fontsize=12,
        color=line, ha='center', va='center')

box(60, 108, 64, 8, '#F4F6F8', '#8799A5', 'C方案光谱输入\n[B, 2048]', fontsize=14)
arrow((60, 103.8), (60, 100))
box(60, 95, 64, 9, blue_fill, blue_edge,
    '连续8点分块（Reshape）\n[B, 256, 8]', fontsize=14)
arrow((60, 90.3), (60, 87))

# 展开双向LSTM：上行正向、下行反向，3个代表性序列位置。
box(60, 72, 100, 29, '#F8FBFE', blue_edge, lw=1.5)
ax.text(60, 82.5, '单层双向LSTM（每个方向 hidden_size = 32）',
        fontsize=14, ha='center', va='center', color=ink)
for y, label in [(75, '正向'), (65, '反向')]:
    ax.text(19, y, label, ha='center', va='center', color=line, fontsize=12)
    for x, step in [(36, 1), (62, 2), (91, 256)]:
        box(x, y, 18, 7, '#DEEBF6', blue_edge, f'LSTM\nt = {step}', fontsize=11.5)
arrow((45.2, 75), (52.8, 75), color=blue_edge)
arrow((71.2, 75), (81.8, 75), color=blue_edge, dashed=True)
arrow((81.8, 65), (71.2, 65), color=blue_edge, dashed=True)
arrow((52.8, 65), (45.2, 65), color=blue_edge)
ax.text(60, 59.5, '正反向特征拼接：encoded  [B, 256, 64]',
        fontsize=13, ha='center', va='center', color=ink)

# 全局统计与参考点读出均来自同一个encoded张量。
wire([(60, 57.3), (60, 53), (24, 53), (24, 49.2)])
wire([(60, 53), (96, 53), (96, 49.2)])
arrow((60, 57.3), (60, 49.2))
arrow((24, 51), (24, 49.2))
arrow((96, 51), (96, 49.2))
box(24, 44, 32, 10, green_fill, green_edge,
    '全局均值\nMean(dim=1)\n[B, 64]', fontsize=12)
box(60, 44, 32, 10, green_fill, green_edge,
    '全局标准差\nStd(dim=1)\n[B, 64]', fontsize=12)
box(96, 44, 32, 10, green_fill, green_edge,
    '1560 nm参考点\n相邻块特征插值\n[B, 64]', fontsize=12)

wire([(24, 38.8), (24, 35), (96, 35), (96, 38.8)])
wire([(60, 38.8), (60, 35)])
arrow((60, 35), (60, 34.2))
box(60, 30, 48, 8, '#F2F4F8', '#7F8BA2',
    '特征拼接（Concat）\n64 + 64 + 64 → [B, 192]', fontsize=13)
arrow((60, 25.8), (60, 23.2))
box(60, 18, 84, 10, '#EFF1FA', '#7B83AE',
    '共享特征层\nLinear(192→64) → LayerNorm → GELU → Dropout(0.1)\n[B, 64]', fontsize=12.5)

wire([(60, 12.8), (60, 9), (26, 9), (26, 5.7)])
wire([(60, 9), (94, 9), (94, 5.7)])
arrow((26, 7.2), (26, 5.7))
arrow((94, 7.2), (94, 5.7))
box(26, 0, 44, 11, '#EEF4FA', blue_edge,
    '光程差输出头\nLinear(64→1) → Squeeze\n标准化OPD  [B]', fontsize=12.5)
box(94, 0, 44, 11, '#F5F1F8', '#9A80AC',
    '相位输出头\nLinear(64→2)\nsinφ、cosφ预测  [B, 2]', fontsize=12.5)
ax.text(60, -11, '当前结构：center_global读出 · 无卷积层 · 23,427个可训练参数',
        fontsize=11.5, ha='center', va='center', color=line)

fig.savefig(OUT/'LSTM网络结构图.png', dpi=300)
fig.savefig(OUT/'LSTM网络结构图.svg', dpi=600)
fig.savefig(OUT/'LSTM网络结构图.pdf', dpi=600)
plt.close(fig)
print(OUT/'LSTM网络结构图.png')
