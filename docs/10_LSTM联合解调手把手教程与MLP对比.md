# 用LSTM联合解调光程差与相位：代码、训练及MLP对比

2026-11-18已用现有训练集/验证集实际训练光谱块双向LSTM。主模型将共享LSTM的全局特征和1560 nm附近特征一起读出，再预测OPD与相位sin/cos。没有卷积层，没有重新划分数据，没有用文件编号或位移步长作为标签。

## 1. 先看真实结果，明确学习目标

使用与原MLP完全相同的训练2800条、验证600条，分别对应280和60个采集位置；C方案输入2048点，标签和掩码未变，真实测试集没有读取。

| 验证指标 | 原MLP | 光谱块BiLSTM＋全局/参考点读出 | 相对下降 |
|---|---:|---:|---:|
| OPD MAE / μm | 887.691 | 95.936 | 89.19% |
| OPD RMSE / μm | 1552.578 | 168.557 | 89.14% |
| OPD绝对误差P95 / μm | 3847.579 | 259.460 | 93.26% |
| 相位圆周MAE / rad | 0.638882 | 0.051840 | 91.89% |
| 相位圆周RMSE / rad | 1.124307 | 0.090217 | 91.98% |
| 参数量 | 566,787 | 23,427 | — |
| 最佳轮次 / 实际轮数 | 99 / 100 | 42 / 62 | — |

两者上限都是100轮，早停耐心20轮、批量64、AdamW学习率0.001、权重衰减0.0001、相位权重1、Dropout 0.1、种子20261117。LSTM第62轮因验证联合损失20轮未改善而停止，恢复第42轮权重。只比较一个种子和一次位置划分，不能称为统计显著改善或已完成论文验证。

本机4线程CPU训练约MLP 14秒、LSTM 81秒，仅为本次日志，不是严格速度基准。参数更少不意味着循环网络训练更快。这是相同训练上限/选模规则的比较，算力和实际训练时间没有匹配。

## 2. 文件分别做什么

| 文件 | 作用 |
|---|---|
| `src/demodulation/joint_lstm.py` | LSTM结构、光谱块组织、参考点与特征读出，中文注释第一至第六块 |
| `scripts/train_joint_lstm.py` | 命令行参数、数据/网络连接、独立测试入口，中文注释第七至第八块 |
| `scripts/train_joint_mlp.py` | 两种模型共用的训练/验证/导出逻辑，保留原MLP命令 |
| `src/demodulation/joint_mlp.py` | 两种模型共用的数据读取、训练集OPD标准化、掩码损失和圆周误差 |
| `scripts/compare_joint_models.py` | 核对数据哈希、按sample_id对齐、重算指标、位置级对比和绘图 |

首先阅读前两个文件；第三个文件的梯度更新、验证和检查点逻辑与MLP教程相同，不必重复写一套，以免两种模型的比较口径不同。

## 3. 第一步：把光谱组织成LSTM序列

先运行：

```python
import numpy as np
import torch

with np.load('datasets/processed/train.npz', allow_pickle=False) as data:
    x = torch.from_numpy(data['X'][:64])

sequence = x.reshape(x.shape[0], 256, 8)
print(x.shape)          # torch.Size([64, 2048])
print(sequence.shape)   # torch.Size([64, 256, 8])
assert torch.equal(sequence.reshape(64, 2048), x)
```

维度含义：第一维64是本批次光谱数；第二维256是按波数排列的光谱块数；第三维8是每块的连续采样点。`reshape`没有平均、抽点或滤波，全部2048点保持原顺序。

这条序列沿**波数轴**组织，不是真正的采集时间轴；不是将同位置10条光谱拼接成一个序列。每条光谱仍独立预测，LSTM隐藏状态每次重新初始化，不跨样本传递。

直接用`[64, 2048, 1]`也是合法LSTM输入，但需要2048次循环步。这里减少到256步，每步仍保留8个数，便于CPU训练。不同块长是否更好仍需验证，不声称8是最优值。

## 4. 第二步：定义双向LSTM

```python
self.lstm = nn.LSTM(
    input_size=8,       # 每一步输入8个连续光谱点
    hidden_size=32,     # 每个方向的隐藏特征为32维
    num_layers=1,
    batch_first=True,   # 输入顺序为[批量, 序列长度, 每步特征]
    bidirectional=True
)
```

LSTM的遗忘门、输入门和输出门决定哪些特征保留、更新和输出，细胞状态在序列步之间传递。PyTorch已实现这些门，不需要为了第一版训练手写门控公式。

双向表示一支按递增波数扫描，另一支反向扫描。当前任务在整条光谱采集后推理，因此可以同时使用参考点两侧信息。这不意味着网络可以读取其他采集位置或测试标签。

```python
encoded, (h_n, c_n) = self.lstm(sequence)
```

默认设置下：

| 张量 | 形状 | 含义 |
|---|---|---|
| `sequence` | `[64, 256, 8]` | 一批光谱块 |
| `encoded` | `[64, 256, 64]` | 每个序列步的正反向特征拼接 |
| `h_n` | `[2, 64, 32]` | 两个方向的最终隐藏状态 |
| `c_n` | `[2, 64, 32]` | 两个方向的最终细胞状态 |

`batch_first=True`不改变`h_n`/`c_n`的维度顺序。单层LSTM内部的`dropout`参数不起作用，因此代码单层时设为0，将0.1的Dropout放在读出层。

## 5. 第三步：正确读取光程差与局部相位所需特征

最简单的方法是：

```python
features = torch.cat([h_n[-2], h_n[-1]], dim=-1)
```

它拼接最后一层正、反向的终态。不能把`encoded[:, -1]`等同于它：反向分支在该位置的输出不代表已经反向扫描完整条光谱的终态。

本次主模型增加两种全局汇总和参考点附近特征：

```python
mean = encoded.mean(dim=1)                   # [B, 64]
std = encoded.std(dim=1, unbiased=False)      # [B, 64]
center = ((1 - self.fraction) * encoded[:, self.left]
          + self.fraction * encoded[:, self.right])
features = torch.cat([mean, std, center], dim=-1)  # [B, 192]
```

全局汇总用于提供整条条纹的信息；参考点特征用于保留局部相位信息。这是结构假设，需要实验对照，不能当作LSTM必然能改善性能的物理证明。

参考点从已有坐标计算：`k_ref=2π/(1560 nm)`，在2048点波数网格中的浮点索引约**991.0804**。它不是默认的数组中点1024。再将该索引映射到光谱块中心坐标，取两侧编码特征线性插值。

这一步只依赖输入坐标和已约定的参考波长，不读取OPD或相位标签，因此没有把答案加入输入。

## 6. 第四步：接回两个任务头

```python
self.shared = nn.Sequential(
    nn.Linear(192, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.1)
)
self.opd_head = nn.Linear(64, 1)
self.phase_head = nn.Linear(64, 2)
```

然后：

```python
shared = self.shared(features)
opd_z = self.opd_head(shared).squeeze(-1)
phase_sc = self.phase_head(shared)
return opd_z, phase_sc
```

与MLP的输出约定完全一致，因此直接复用：

```text
OPD：仅用训练集拟合标准化参数 → SmoothL1损失 → 分别用OPD mask
相位：[sinφ, cosφ] → 两分量MSE → 分别用phase mask
总损失：OPD损失 + phase_weight × 相位损失
```

预测OPD按检查点内参数反标准化；相位用`atan2(sin_pred, cos_pred)`恢复。相位误差用`atan2(sin(Δφ), cos(Δφ))`。不把无效标签当0、不通过NaN乘0屏蔽、不强迫两个伪标签满足无色散双参数模型。

## 7. 第五步：手动做一次前向传播

可在仓库根目录新建一个练习脚本运行：

```python
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path.cwd() / 'src'))
from demodulation.joint_mlp import load_split
from demodulation.joint_lstm import JointLSTM, reference_sample_index

data = load_split('datasets/processed/train.npz')
model = JointLSTM(reference_index=reference_sample_index(data))
x = torch.from_numpy(data['X'][:4])
opd_z, phase_sc = model(x)
print(opd_z.shape)      # torch.Size([4])
print(phase_sc.shape)   # torch.Size([4, 2])
```

此时权重还是随机的，输出仅用于检查维度，不能解释成可信解调结果。

## 8. 第六步：训练和验证

仍然按照`清梯度→前向→掩码损失→反向传播→梯度裁剪→AdamW更新`。训练打乱取样顺序但不改变位置分区；验证关闭Dropout和梯度。

在项目根目录，使用已准备好的训练环境：

```powershell
.\.venv-training\Scripts\python.exe -X utf8 scripts/train_joint_lstm.py train --epochs 2 --device cpu --out analysis/my_lstm_smoke
```

确认成功后，换一个新输出目录：

```powershell
.\.venv-training\Scripts\python.exe -X utf8 scripts/train_joint_lstm.py train --epochs 100 --device cpu --out analysis/my_lstm_100
```

默认就是本次主模型，CPU可以完成；输出目录若存在会报错，换`--out`防止覆盖原有最佳权重。现有环境为CPU版PyTorch，不需要安装新包。新环境安装方法见文档09。

训练命令不读取`test.npz`。使用`best.pt`选择的是验证联合损失最低轮次，可能不是某个单项MAE最低轮次，两个任务有权重折中。

## 9. 第七步：补做读出方式对照

只读取最终状态的版本使用：

```powershell
.\.venv-training\Scripts\python.exe -X utf8 scripts/train_joint_lstm.py train --epochs 100 --device cpu --readout last --out analysis/my_lstm_last
```

本次对照与主模型使用相同块长、隐藏维度、层数、种子及训练规则；只有读出方式及其后全连接输入维度随之变化。两个读出版本参数量不同，不能称作所有参数完全匹配的单变量实验。

实际终态对照在第7轮最佳，第27轮早停，验证OPD MAE为1829.367 μm，相位圆周MAE约1.4338 rad，接近训练集均值常量基线，明显差于MLP。这表明本次优化协议下终态读出没有学到足够信息；不能由此断定任何终态LSTM都不适用于光谱。Hann压低两端信息、参考相位位于波段内部，是设计参考点读出的动机，但这一组对照不足以单独证明根本原因。

主模型使用全局/参考点特征，不能把结果写成“只把MLP换成一个标准LSTM层即可得到相同改善”。对照的实际记录会一起保留在`results/lstm_comparison/metrics.csv`和`summary.json`。

## 10. 第八步：重算模型对比，而非手填结果

本机本次实验目录：MLP为`analysis/joint_mlp_tutorial_seed20261117`，主LSTM为`analysis/joint_lstm_p8_h32_seed20261117`，终态对照为`analysis/joint_lstm_last_p8_h32_seed20261117`。

```powershell
.\.venv-training\Scripts\python.exe -X utf8 scripts/compare_joint_models.py --mlp-run analysis/joint_mlp_tutorial_seed20261117 --lstm-run analysis/joint_lstm_p8_h32_seed20261117 --control-run analysis/joint_lstm_last_p8_h32_seed20261117
```

如果自己重训，换成对应的目录。比较脚本会核对两份NPZ的SHA256、种子及训练参数，按sample_id对齐全部600条验证预测，重新计算指标并检查是否与训练日志一致。它不读取测试集。

仓库同时归档三组实验的`best.pt`、配置、训练历史、验证指标与预测到`results/lstm_comparison/runs/`。新下载仓库后无需先重训，即可复算当前图表：

```powershell
.\.venv-training\Scripts\python.exe -X utf8 scripts/compare_joint_models.py --mlp-run results/lstm_comparison/runs/mlp --lstm-run results/lstm_comparison/runs/lstm_center_global --control-run results/lstm_comparison/runs/lstm_last
```

输出到`results/lstm_comparison/`：

| 文件 | 含义 |
|---|---|
| `summary.json` | 全部指标、配置、检查点哈希、相对改善、位置级胜负 |
| `metrics.csv` | 模型指标对照表，包括失败的对照版本 |
| `paired_validation_predictions.csv` | 同一条谱的模型预测和配对误差 |
| `position_metrics.csv` | 同一位置10条的误差均值，60行 |
| `mlp_vs_lstm.png/.svg/.pdf` | OPD预测、相位圆周误差和训练/验证损失对比 |

图中使用全部可评价验证谱，不筛选“好看”的位置。单种子不画虚构的跨种子误差条。每位置10条是技术重复，不当作10个独立采集位置做显著性检验。

## 11. 已经改善了什么，还没解决什么

主LSTM的OPD位置平均误差在60个验证位置中有50个低于MLP，相位有57个低于MLP，说明改善不是每个位置都发生。

95.936 μm的OPD MAE仍大于相邻位置约20 μm的OPD差，当前结果不能证明能可靠区分所有相邻位置。10条训练OPD范围外的验证记录，OPD MAE约292.625 μm；范围内590条约92.603 μm，边界问题仍存在。

0.05184 rad相位误差约2.97°，只是相对于当前相位算法伪标签。不能据此直接宣称纳米位移精度：绝对相位零点、未知整周期、独立仪器真值和环境稳定性仍未验证。

本次结论限于一种子、同次采集留出位置；还需多种子、训练范围边界、跨日期/设备以及独立参考测量的验证。下一步可先研究读出结构、任务权重和训练分布，不能只堆叠更多LSTM层或承诺一定更好。

## 12. 模型确定后才做最终测试

```powershell
.\.venv-training\Scripts\python.exe -X utf8 scripts/train_joint_lstm.py evaluate --checkpoint analysis/my_lstm_100/best.pt --device cpu --out analysis/my_lstm_test
```

该命令会检查测试位置、sample_id和完全重复谱不与检查点的训练/验证记录重叠，检查输入网格和相位参考一致，随后导出测试指标。禁止根据测试结果再选择超参数后继续称其为最终独立测试。

当前训练/比较阶段没有执行该命令。自动化测试只使用临时生成的小型数据测试加载链路，不涉及真实测试集。

检查代码：

```powershell
.\.venv-training\Scripts\python.exe -m pytest tests/test_joint_lstm.py tests/test_joint_mlp.py -q
.\.venv-training\Scripts\python.exe -m pytest tests -q
```

PyTorch输入/输出和双向隐藏状态的准确含义见[官方LSTM文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)。光程差与相位标签的约定仍以本仓库文档06为准。
