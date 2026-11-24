"""从仓库内实现导出带代码和三份数据的独立教学工程。"""
import argparse
import ast
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description='将代码和数据导出到新的独立LSTM工程目录')
parser.add_argument('--out', type=Path, default=ROOT / 'analysis' / 'lstm_standalone_export', help='尚不存在的导出目录')
args = parser.parse_args()
DEST = args.out.resolve()
# 拒绝覆盖已有目录，保护用户修改过的代码和图表。
DEST.mkdir(parents=True, exist_ok=False)
shutil.copy2(ROOT / 'src/demodulation/joint_lstm.py', DEST / 'model.py')


def extract(path, names):
    source = path.read_text(encoding='utf-8')
    nodes = {node.name: node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    result = {}
    lines = source.splitlines()
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        result[name] = '\n'.join(lines[start - 1:node.end_lineno])
    return result


data_parts = extract(ROOT / 'src/demodulation/joint_mlp.py', [
    'load_split', 'check_compatible', 'check_no_leakage', 'fit_opd_stats',
    'SpectralDataset', 'joint_loss', 'predict', 'circular_error', 'metrics',
])
train_parts = extract(ROOT / 'scripts/train_joint_mlp.py', [
    'prepare', 'write_json', 'file_sha', 'export_predictions', 'train', 'evaluate',
])

training = train_parts['train'].replace(
    "def train(args, device, model_class=JointMLP, model_kwargs=None, model_type='mlp'):",
    'def train(args, device):'
)
old = '''    # 默认仍是原来的MLP；LSTM入口传入另一模型类，共用完全相同的数据/损失/划分。
    model_args = (dict(input_dim=train_data['X'].shape[1], hidden=args.hidden, dropout=args.dropout)
                  if model_kwargs is None else model_kwargs)
    model = model_class(**model_args).to(device)'''
new = '''    # 建立光谱块双向LSTM：2048点 -> 256个8点块 -> 全局/参考点特征 -> 双输出头。
    # 这些参数会一起写进best.pt，后面加载权重时可以重建完全相同的网络。
    model_args = dict(input_dim=train_data['X'].shape[1], patch_size=args.patch_size,
                      hidden_size=args.hidden_size, num_layers=args.num_layers,
                      dropout=args.dropout, reference_index=reference_sample_index(train_data),
                      readout='center_global')
    model = JointLSTM(**model_args).to(device)'''
assert old in training
training = training.replace(old, new).replace('model_type=model_type', "model_type='lstm'")
evaluation = train_parts['evaluate'].replace(
    "def evaluate(args, device, model_class=JointMLP, expected_model_type='mlp'):",
    'def evaluate(args, device):'
).replace("checkpoint.get('model_type', 'mlp') != expected_model_type",
          "checkpoint.get('model_type') != 'lstm'").replace(
    'model = model_class(**checkpoint[\'model_args\']).to(device)',
    'model = JointLSTM(**checkpoint[\'model_args\']).to(device)'
)

header = '''r"""独立LSTM训练入口：你只需要运行这个文件，不需要打开原项目。

快速检查：.\\.venv\\Scripts\\python.exe -X utf8 train.py --epochs 2
正式训练：.\\.venv\\Scripts\\python.exe -X utf8 train.py --epochs 100
文件分工：model.py定义整个网络；本文件读取数据、训练、验证并保存权重。
现有数据已执行预处理；标签来自传统算法，误差表示与伪标签的一致性。
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# 仅导入当前文件夹里的model.py，不导入原工程的任何代码。
from model import JointLSTM, reference_sample_index

# 根据train.py自身的位置找数据，因此从其他目录启动也能找到data文件夹。
ROOT = Path(__file__).resolve().parent

'''

sections = [
    ('一、读数据和检查分区（数据已预处理，不再加窗）',
     [data_parts[name] for name in ('load_split', 'check_compatible', 'check_no_leakage')]),
    ('二、准备标签和批次：OPD标准化只使用训练集',
     [data_parts[name] for name in ('fit_opd_stats', 'SpectralDataset')]),
    ('三、定义联合损失：两个任务分别筛选有效标签', [data_parts['joint_loss']]),
    ('四、模型推理和误差评价：恢复OPD单位、使用相位圆周距离',
     [data_parts[name] for name in ('predict', 'circular_error', 'metrics')]),
    ('五、固定随机种子、选择CPU/GPU、保存结果的辅助函数',
     [train_parts[name] for name in ('prepare', 'write_json', 'file_sha', 'export_predictions')]),
    ('六、完整训练流程：初学时优先阅读这个train函数', [training]),
    ('七、独立测试：仅指定--evaluate时读取test.npz', [evaluation]),
]

main_code = '''# 八、运行入口：在这里定义可修改的训练参数。
def main():
    parser = argparse.ArgumentParser(description='独立LSTM光程差与相位联合解调工程')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data', help='三份NPZ所在的文件夹')
    parser.add_argument('--epochs', type=int, default=100, help='最多训练多少轮')
    parser.add_argument('--batch-size', type=int, default=64, help='每批使用多少条光谱')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率，即权重更新的步长设置')
    parser.add_argument('--patch-size', type=int, default=8, help='每个序列步包含的连续采样点数')
    parser.add_argument('--hidden-size', type=int, default=32, help='LSTM每个方向的隐藏特征维数')
    parser.add_argument('--num-layers', type=int, default=1, help='堆叠的LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='读出层的Dropout概率')
    parser.add_argument('--phase-weight', type=float, default=1.0, help='相位损失的权重')
    parser.add_argument('--patience', type=int, default=20, help='验证损失多少轮不改善就停止')
    parser.add_argument('--seed', type=int, default=20261117, help='固定随机种子，便于复现')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu', help='当前环境使用cpu即可')
    parser.add_argument('--threads', type=int, default=4, help='CPU计算线程数')
    parser.add_argument('--out', type=Path, help='结果保存目录；省略时自动创建带时间的目录')
    parser.add_argument('--evaluate', action='store_true', help='模型确定后才单独评价测试集')
    parser.add_argument('--checkpoint', type=Path, help='测试时要加载的best.pt路径')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patch_size, args.hidden_size,
           args.num_layers, args.patience, args.threads) < 1:
        parser.error('轮次、批量、块长、特征维数、层数、耐心轮数和线程数必须为正')
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.phase_weight <= 0:
        parser.error('要求0≤dropout<1，lr>0，phase-weight>0')
    if args.evaluate and args.checkpoint is None:
        parser.error('测试时还需要--checkpoint指定已训练好的best.pt')
    args.command = 'evaluate' if args.evaluate else 'train'
    if args.out is None:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        args.out = ROOT / 'outputs' / (args.command + '_' + stamp)
    # 准备环境只固定随机数与计算设置，不读取测试数据。
    device = prepare(args.seed, args.device, args.threads)
    print('工程位置：', ROOT, flush=True)
    print('本次输出：', args.out.resolve(), flush=True)
    if args.evaluate:
        evaluate(args, device)
    else:
        train(args, device)


if __name__ == '__main__':
    main()
'''

text = header
for title, functions in sections:
    text += '\n# ' + '=' * 65 + '\n# ' + title + '\n# ' + '=' * 65 + '\n\n'
    text += '\n\n\n'.join(functions) + '\n\n'
text += '\n' + main_code
ast.parse(text)
assert 'demodulation' not in text and 'JointMLP' not in text
(DEST / 'train.py').write_text(text, encoding='utf-8')
model = (DEST / 'model.py').read_text(encoding='utf-8').replace(
    '# 第六块：输出约定与MLP保持一致，复用原有掩码损失和圆周相位评价。',
    '# 第六块：输出标准化OPD和相位sin/cos，由train.py计算损失并更新权重。'
)
(DEST / 'model.py').write_text(model, encoding='utf-8')
(DEST / 'requirements.txt').write_text(
    '# 当前已验证：Windows、Python 3.13.5；只需要下面两个直接依赖。\n'
    'numpy==2.1.3\ntorch==2.15.0\n', encoding='utf-8'
)
(DEST / 'data').mkdir()
for name in ('train.npz', 'validation.npz', 'test.npz'):
    shutil.copy2(ROOT / 'datasets/processed' / name, DEST / 'data' / name)
shutil.copy2(ROOT / 'examples/lstm_standalone/先读我_运行步骤.md', DEST / '先读我_运行步骤.md')
(DEST / 'README.md').write_text(
    '# 独立LSTM工程\n\n在当前目录建立Python环境并安装requirements.txt，然后执行：\n\n'
    '```powershell\npython -X utf8 train.py --epochs 100\n```\n\n'
    'data内包含三份处理后数据；本导出包不包含本机虚拟环境。\n', encoding='utf-8')
print('Standalone files and data created:', DEST)
