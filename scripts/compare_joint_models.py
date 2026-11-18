"""按相同600条验证谱重新计算MLP/LSTM误差，并生成可追溯对比图。

不读取test.npz。光程差单位μm，相位使用圆周误差；不把伪标签当仪器真值。
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from demodulation.joint_mlp import JointMLP, circular_error, load_split, metrics, predict
from demodulation.joint_lstm import JointLSTM
from scripts.train_joint_mlp import file_sha, write_json


def read_run(path, data, train_sha, val_sha):
    """先按sample_id对齐，再由逐条预测重算指标，防止不同顺序/数据版本混比。"""
    config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
    if config['train_sha256'] != train_sha or config['validation_sha256'] != val_sha:
        raise ValueError(f'{path}不是同一版训练/验证数据')
    with (path / 'validation_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    mapping = {row['sample_id']: row for row in rows}
    if len(rows) != len(mapping) or set(mapping) != set(data['sample_ids']):
        raise ValueError('模型预测未完整覆盖相同验证样本，或存在重复编号')
    ordered = [mapping[str(sample_id)] for sample_id in data['sample_ids']]
    if any(row['position_id'] != str(position) for row, position in zip(ordered, data['position_id'])):
        raise ValueError('预测记录的位置编号与数据不一致')
    pred = dict(opd_m=np.array([float(r['opd_pred_m']) for r in ordered]),
                phase_rad=np.array([float(r['phase_pred_rad']) if r['phase_pred_rad'] else 0 for r in ordered]),
                phase_norm=np.array([float(r['phase_vector_norm']) for r in ordered]))
    if not all(np.isfinite(values).all() for values in pred.values()):
        raise ValueError('预测存在非有限数')
    report = metrics(data, pred, config['opd_stats'])
    stored = json.loads((path / 'validation_metrics.json').read_text(encoding='utf-8'))
    for key in ('opd_um_mae', 'opd_um_rmse', 'phase_rad_mae', 'phase_rad_rmse'):
        if not np.isclose(report[key], stored[key], rtol=1e-10, atol=1e-10):
            raise ValueError(f'重算{key}与训练记录不一致')
    checkpoint = torch.load(path / 'best.pt', map_location='cpu', weights_only=True)
    model_type = checkpoint.get('model_type', 'mlp')
    if model_type not in ('mlp', 'lstm'):
        raise ValueError('未知模型检查点类型')
    model = (JointMLP if model_type == 'mlp' else JointLSTM)(**checkpoint['model_args'])
    model.load_state_dict(checkpoint['model_state_dict'])
    # 实际加载权重重新推理全部验证谱，确认发布权重与记录的CSV预测一致。
    replay = predict(model, data, checkpoint['opd_stats'], torch.device('cpu'))
    np.testing.assert_allclose(replay['opd_m'], pred['opd_m'], rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(circular_error(replay['phase_rad'], pred['phase_rad']), 0, atol=1e-4)
    np.testing.assert_allclose(replay['phase_norm'], pred['phase_norm'], rtol=1e-4, atol=1e-5)
    params = sum(value.numel() for value in checkpoint['model_state_dict'].values())
    report.update(best_epoch=stored['best_epoch'], epochs_run=stored['epochs_run'],
                  training_seconds=stored['training_seconds'], parameter_count=int(params),
                  checkpoint_sha256=file_sha(path / 'best.pt'), checkpoint_replayed=True, config=config)
    return report, pred


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(data, runs, reports, mlp_path, lstm_path, out):
    """全600条点原样绘制；单种子无跨种子误差条，不人为筛选好看的位置。"""
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 7,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(7.0866, 5.3150))  # 180×135 mm
    grid = fig.add_gridspec(2, 2, height_ratios=[1, .85], left=.10, right=.97,
                            bottom=.13, top=.88, wspace=.36, hspace=.53)
    ax_opd, ax_phase, ax_loss = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    colors = {'MLP': '#999999', 'LSTM': '#12699B'}
    true_mm = data['opd_pseudo_m'] * 1e3
    for name in ('MLP', 'LSTM'):
        pred = runs[name]
        marker = 'o' if name == 'MLP' else '.'
        size = 7 if name == 'MLP' else 8
        opd_good = data['opd_valid']
        ax_opd.scatter(true_mm[opd_good], pred['opd_m'][opd_good] * 1e3, s=size, color=colors[name],
                       alpha=.4 if name == 'MLP' else .8, marker=marker, linewidths=0,
                       label=name, rasterized=True)
        good = data['phase_valid'] & opd_good & (pred['phase_norm'] > 1e-6)
        error = circular_error(pred['phase_rad'], data['phase_pseudo_rad'])
        ax_phase.scatter(true_mm[good], error[good], s=size, color=colors[name],
                         alpha=.4 if name == 'MLP' else .8, marker=marker,
                         linewidths=0, rasterized=True)
    lo, hi = float(true_mm.min()), float(true_mm.max())
    ax_opd.plot([lo, hi], [lo, hi], color='#333333', ls='--', lw=.7, zorder=0)
    ax_opd.set(xlabel='OPD pseudolabel (mm)', ylabel='Predicted OPD (mm)')
    ax_phase.axhline(0, color='#333333', ls='--', lw=.7, zorder=0)
    ax_phase.set(xlabel='OPD pseudolabel (mm)', ylabel='Circular phase error (rad)', ylim=(-3.4, 3.4))
    for name, path in [('MLP', mlp_path), ('LSTM', lstm_path)]:
        with (path / 'history.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        epochs = np.array([int(row['epoch']) for row in rows])
        for key, style, suffix in [('train_loss', '--', 'train'), ('validation_loss', '-', 'validation')]:
            loss = np.array([float(row[key]) for row in rows])
            if not np.all(loss > 0):
                raise ValueError('对数损失图要求所有损失严格为正')
            ax_loss.plot(epochs, loss, ls=style, lw=1.1, color=colors[name], label=f'{name} {suffix}')
    ax_loss.set(xlabel='Epoch', ylabel='Joint loss', yscale='log')
    # 用普通小数刻度，避免对数刻度的上标缩小到5 pt以下。
    ax_loss.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:g}'))
    ax_loss.legend(loc='center right', ncol=2, frameon=False)
    for label, ax in zip('abc', (ax_opd, ax_phase, ax_loss)):
        ax.text(-.16 if ax is not ax_loss else -.075, 1.06, label,
                transform=ax.transAxes, fontweight='bold', fontsize=9)
    handles, labels = ax_opd.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, .99))
    fig.text(.5, .025, 'Validation: 600 spectra / 60 positions; one seed; algorithmic pseudolabels; test set unused.',
             ha='center', fontsize=7)
    fig.savefig(out / 'mlp_vs_lstm.png', dpi=600)
    fig.savefig(out / 'mlp_vs_lstm.svg', dpi=600)
    fig.savefig(out / 'mlp_vs_lstm.pdf', dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mlp-run', type=Path, required=True)
    parser.add_argument('--lstm-run', type=Path, required=True)
    parser.add_argument('--control-run', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'datasets/processed')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/lstm_comparison')
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = load_split(args.data_dir / 'validation.npz')
    train_sha, val_sha = file_sha(args.data_dir / 'train.npz'), file_sha(args.data_dir / 'validation.npz')
    models = [('MLP', args.mlp_run), ('LSTM', args.lstm_run)]
    if args.control_run:
        models.append(('LSTM final-state control', args.control_run))
    reports, predictions = {}, {}
    for name, path in models:
        reports[name], predictions[name] = read_run(path, data, train_sha, val_sha)
    # 记录哪些训练条件匹配；参数量/训练耗时并不相等，不能称作等算力实验。
    conditions = ('seed', 'epochs', 'batch_size', 'lr', 'dropout', 'phase_weight', 'patience')
    for condition in conditions:
        if len({report['config'][condition] for report in reports.values()}) != 1:
            raise ValueError(f'比较条件{condition}不一致')
    args.out.mkdir(parents=True, exist_ok=True)
    metric_names = ['opd_um_mae', 'opd_um_rmse', 'opd_um_p95_abs', 'phase_rad_mae',
                    'phase_rad_rmse', 'phase_rad_p95_abs', 'parameter_count',
                    'best_epoch', 'epochs_run', 'training_seconds']
    write_csv(args.out / 'metrics.csv', [dict(model=name, **{key: reports[name][key] for key in metric_names})
                                        for name, _ in models])
    paired_rows, position_rows = [], []
    for i, sample_id in enumerate(data['sample_ids']):
        opd_valid, phase_valid = bool(data['opd_valid'][i]), bool(data['phase_valid'][i])
        row = dict(sample_id=str(sample_id), position_id=str(data['position_id'][i]),
                   opd_valid=opd_valid, phase_valid=phase_valid,
                   opd_pseudo_m=float(data['opd_pseudo_m'][i]) if opd_valid else None,
                   phase_pseudo_rad=float(data['phase_pseudo_rad'][i]) if phase_valid else None)
        for name, _ in models:
            pred = predictions[name]
            row[f'{name}_opd_pred_m'] = float(pred['opd_m'][i])
            row[f'{name}_opd_abs_error_um'] = float(abs(pred['opd_m'][i] - data['opd_pseudo_m'][i]) * 1e6) if opd_valid else None
            defined = pred['phase_norm'][i] > 1e-6
            row[f'{name}_phase_pred_rad'] = float(pred['phase_rad'][i]) if defined else None
            row[f'{name}_phase_abs_error_rad'] = float(abs(circular_error(pred['phase_rad'][i], data['phase_pseudo_rad'][i]))) if phase_valid and defined else None
        paired_rows.append(row)
    write_csv(args.out / 'paired_validation_predictions.csv', paired_rows)
    for position in np.unique(data['position_id']):
        selected = [row for row in paired_rows if row['position_id'] == position]
        row = dict(position_id=str(position), samples=len(selected))
        for name, _ in models:
            for task in ('opd_abs_error_um', 'phase_abs_error_rad'):
                values = [item[f'{name}_{task}'] for item in selected if item[f'{name}_{task}'] is not None]
                row[f'{name}_{task}'] = float(np.mean(values)) if values else None
        position_rows.append(row)
    write_csv(args.out / 'position_metrics.csv', position_rows)
    improvement = {key: 100 * (1 - reports['LSTM'][key] / reports['MLP'][key])
                   for key in metric_names[:6]}
    summary = dict(models=reports, improvement_percent=improvement,
                   validation_samples=len(data['X']), validation_positions=len(position_rows),
                   test_set_read=False, single_seed=True, matched_compute_budget=False,
                   comparison_conditions={key: reports['MLP']['config'][key] for key in conditions},
                   position_wins={task: sum(row[f'LSTM_{task}'] < row[f'MLP_{task}'] for row in position_rows
                                            if row[f'LSTM_{task}'] is not None and row[f'MLP_{task}'] is not None)
                                  for task in ('opd_abs_error_um', 'phase_abs_error_rad')})
    write_json(args.out / 'summary.json', summary)
    plot_comparison(data, predictions, reports, args.mlp_run, args.lstm_run, args.out)
    print(json.dumps({name: {key: report[key] for key in metric_names}
                      for name, report in reports.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
