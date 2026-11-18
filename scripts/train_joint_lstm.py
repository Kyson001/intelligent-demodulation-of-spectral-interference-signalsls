"""中文教学入口：光谱块LSTM联合解调；复用MLP训练器保证比较口径一致。

训练：python scripts/train_joint_lstm.py train --epochs 100 --out analysis/my_lstm
测试：python scripts/train_joint_lstm.py evaluate --checkpoint analysis/my_lstm/best.pt
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from demodulation.joint_lstm import JointLSTM, reference_sample_index
from demodulation.joint_mlp import load_split
from scripts.train_joint_mlp import evaluate, prepare, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'evaluate'):
        p = commands.add_parser(name)
        p.add_argument('--data-dir', type=Path, default=ROOT / 'datasets/processed')
        p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
        p.add_argument('--threads', type=int, default=4)
        p.add_argument('--seed', type=int, default=20261117)
        p.add_argument('--out', type=Path, default=ROOT / f'analysis/joint_lstm_{name}')
        if name == 'train':
            # 第七块：数据、损失、优化器、随机种子默认与原MLP一致。
            p.add_argument('--epochs', type=int, default=100)
            p.add_argument('--batch-size', type=int, default=64)
            p.add_argument('--lr', type=float, default=1e-3)
            p.add_argument('--dropout', type=float, default=0.1)
            p.add_argument('--phase-weight', type=float, default=1.)
            p.add_argument('--patience', type=int, default=20)
            p.add_argument('--patch-size', type=int, default=8)
            p.add_argument('--hidden-size', type=int, default=32)
            p.add_argument('--num-layers', type=int, default=1)
            p.add_argument('--readout', choices=['last', 'center_global'], default='center_global')
        else:
            p.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads必须为正整数')
    device = prepare(args.seed, args.device, args.threads)
    if args.command == 'evaluate':
        # 第八块：只有显式evaluate命令才读取test.npz，训练不接触真实测试集。
        evaluate(args, device, model_class=JointLSTM, expected_model_type='lstm')
        return
    if min(args.epochs, args.batch_size, args.patience, args.patch_size,
           args.hidden_size, args.num_layers) < 1:
        parser.error('轮次、批量、块长、隐藏维度、层数必须为正整数')
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.phase_weight <= 0:
        parser.error('要求0≤dropout<1，lr>0，phase-weight>0')
    data = load_split(args.data_dir / 'train.npz')
    model_args = dict(input_dim=data['X'].shape[1], patch_size=args.patch_size,
                      hidden_size=args.hidden_size, num_layers=args.num_layers,
                      dropout=args.dropout, reference_index=reference_sample_index(data),
                      readout=args.readout)
    train(args, device, model_class=JointLSTM, model_kwargs=model_args, model_type='lstm')


if __name__ == '__main__':
    main()
