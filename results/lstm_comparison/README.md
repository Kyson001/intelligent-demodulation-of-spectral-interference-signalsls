# MLP与LSTM：实际训练结果

2026-11-18完成。训练2800条、验证600条；位置分区与C方案NPZ不变。这里没有使用真实测试集，所有误差都是对算法伪标签的一致性。

| 模型 | OPD MAE / μm | OPD RMSE / μm | 相位圆周MAE / rad | 参数量 | 最佳轮/实际轮 |
|---|---:|---:|---:|---:|---:|
| 原MLP | 887.691 | 1552.578 | 0.638882 | 566787 | 99/100 |
| BiLSTM，仅终态读出 | 1829.367 | 2115.380 | 1.433779 | 15235 | 7/27 |
| BiLSTM，全局＋参考点读出 | 95.936 | 168.557 | 0.051840 | 23427 | 42/62 |

主LSTM相对于MLP的OPD/相位MAE下降89.19%/91.89%。60个验证位置中，OPD平均误差改善50个、相位改善57个；不是所有位置都改善。10条训练OPD范围外记录的OPD MAE仍为292.625 μm。

两种LSTM均为8点一块、256步、每方向32隐藏维、1层。复用同一训练器与有效标签掩码，上限100轮、早停20轮、种子20261117、批量64、学习率0.001、相位权重1、Dropout 0.1。参数量与实际计算时间不同，不是等算力比较。

## 文件

- `metrics.csv`、`summary.json`：完整指标、配置、来源/检查点哈希和改善比例。
- `paired_validation_predictions.csv`：按sample_id对齐的全部600条验证谱。
- `position_metrics.csv`：60个位置的误差汇总。
- `mlp_vs_lstm.png/.svg/.pdf`：主模型与MLP的预测/误差/损失图。
- `runs/mlp/`、`runs/lstm_center_global/`、`runs/lstm_last/`：三组最佳权重、配置、历史与原始预测。失败对照也完整保留。
- `figure_contract.md`、`figure_qa.md`和审计JSON：图表范围、数据完整性与排版检查。

## 复算图表

在仓库根目录安装训练依赖后运行：

```powershell
.\.venv-training\Scripts\python.exe -X utf8 scripts/compare_joint_models.py --mlp-run results/lstm_comparison/runs/mlp --lstm-run results/lstm_comparison/runs/lstm_center_global --control-run results/lstm_comparison/runs/lstm_last
```

脚本检查训练/验证数据哈希，重新加载三个检查点对全部验证谱推理，与发布CSV比较，再重算指标；真实测试集不参与。

## 重新训练

```powershell
.\.venv-training\Scripts\python.exe -X utf8 scripts/train_joint_lstm.py train --epochs 100 --device cpu --out analysis/my_lstm_100
.\.venv-training\Scripts\python.exe -X utf8 scripts/train_joint_lstm.py train --epochs 100 --device cpu --readout last --out analysis/my_lstm_last
```

逐块教学见`docs/10_LSTM联合解调手把手教程与MLP对比.md`。单种子、同次采集位置留出结果不能代替多种子、跨日期和独立测量真值验证；相位结果不能直接换算成已验证的纳米位移精度。
