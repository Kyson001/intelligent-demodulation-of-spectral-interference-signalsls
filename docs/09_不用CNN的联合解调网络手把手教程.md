# 不用CNN：手把手写光程差与相位联合解调网络

第一版使用**多层感知机MLP（全连接网络）**。先学会数据读取、共享特征、双输出头、掩码损失、验证选模型与保存权重，后续才能有依据地换成GRU或Transformer。

这是一份可运行的教学基线。简单MLP的验证误差仍较大，不能直接称为高精度解调模型或论文创新。当前所有监督标签来自传统算法，指标只表示与伪标签的一致性。

## 1. 先认识本次要写的两个Python文件

| 文件 | 负责什么 | 阅读顺序 |
|---|---|---|
| `src/demodulation/joint_mlp.py` | 读数据、标准化、Dataset、网络、损失、预测和误差 | 中文注释第一至第五块 |
| `scripts/train_joint_mlp.py` | 选设备、训练、验证、保存、单独测试 | 中文注释第六至第十二块 |

两份文件共用同一个模型定义，避免训练和测试网络结构不一致。所有操作在仓库根目录执行。已有预处理数据无需再次加窗、插值或滤波。

## 2. 准备环境

本机已建立`.venv-training`，安装并验证PyTorch 2.15.0+cpu，可直接执行后面的命令。该环境通过`--system-site-packages`复用基础环境已有的NumPy等包，未修改基础环境的PyTorch。小型MLP可用CPU训练。

若从新下载的仓库开始，可以创建独立环境：

```powershell
cd 'C:\Users\Kyson\Desktop\基于深度学习的端到端光谱干涉信号智能解调'
python -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-training\Scripts\python.exe -m pip install -r requirements-training.txt
```

使用虚拟环境的完整Python路径，不必执行激活脚本，也无需修改PowerShell执行策略。CPU版`torch.cuda.is_available()`为False是正常情况；本机有RTX 4060，但当前教学环境安装的是CPU包。以后需要GPU时，根据[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)匹配驱动和CUDA构建，不盲目复制其他机器的安装命令。

## 3. 第一步自己写：读取数据，弄懂每个数组

```python
import numpy as np

with np.load('datasets/processed/train.npz', allow_pickle=False) as data:
    X = data['X']                         # (2800, 2048)，每行一条光谱
    opd_m = data['opd_pseudo_m']           # (2800,)，光程差，单位m
    phase_sin = data['phase_sin']          # (2800,)
    phase_cos = data['phase_cos']          # (2800,)
    opd_valid = data['opd_valid']          # (2800,)，OPD标签是否可用
    phase_valid = data['phase_valid']      # (2800,)，相位标签是否可用
    print(X.shape, opd_valid.sum(), phase_valid.sum())
```

期望输出：`(2800, 2048) 2799 2794`。验证集600条、两个任务均有效；测试集600条，其中598条相位有效。

这里一条输入对应一个OPD和**1560 nm处的一个包裹相位**，不是2048个相位输出。标签中的OPD已经是光程差，不再乘2；仅在确定空气、正入射等条件后，将OPD变化换算镜面位移时才考虑往返因子。

`load_split()`在正式代码中进一步检查数组、单位圆相位、波数网格和有效标记。`check_no_leakage()`确认训练和验证没有位置或完全重复光谱重叠。不要把三个数据集合并后重新按单条光谱随机划分。

## 4. 第二步自己写：标准化OPD

不同任务的标签量纲不同。把OPD从m换成mm，再只用训练集有效标签计算均值和标准差：

```python
valid_mm = opd_m[opd_valid] * 1e3
mean_mm = valid_mm.mean()
std_mm = valid_mm.std()
opd_z = (opd_m * 1e3 - mean_mm) / std_mm
```

本次训练均值约6.190679 mm，标准差约2.339062 mm。正式代码将它们保存到`best.pt`；验证、测试及将来网页推理都使用这同一组参数。

恢复单位：

```python
opd_pred_m = (opd_pred_z * std_mm + mean_mm) * 1e-3
```

输入X已经逐谱归一化，这一版不再次对每个波长点拟合标准化。相位sin/cos不做OPD那样的标准化。

## 5. 第三步自己写：Dataset和DataLoader

`SpectralDataset.__getitem__()`返回五项：

```python
(光谱, 标准化OPD, [sin相位, cos相位], OPD有效标记, 相位有效标记)
```

然后：

```python
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
```

`batch_size=64`表示一次用64条光谱计算梯度；`shuffle=True`只是每轮打乱训练集取样顺序，不改变原先的训练/验证/测试划分。Windows教学脚本先用`num_workers=0`，便于定位报错。

一个批次的X为`[B, 2048]`，OPD为`[B]`，相位为`[B, 2]`。最后一批不足64条也可以训练。

## 6. 第四步自己写：共享特征层和两个任务头

结构：

```text
2048点C方案光谱
        ↓
Linear(2048, 256) → LayerNorm → GELU → Dropout
        ↓
Linear(256, 128)  → LayerNorm → GELU → Dropout
        ↓
Linear(128, 64)   → LayerNorm → GELU → Dropout
        ├── Linear(64, 1) → 标准化OPD
        └── Linear(64, 2) → sinφ、cosφ
```

`Linear`让每个输出特征使用前一层的全部输入；`GELU`提供非线性；`LayerNorm`稳定隐藏层数值；`Dropout(0.1)`在训练中随机屏蔽部分特征，作为正则化。整个网络没有卷积层，总参数566,787个。

核心前向传播：

```python
def forward(self, x):
    features = self.backbone(x)
    opd_z = self.opd_head(features).squeeze(-1)
    phase_sc = self.phase_head(features)
    return opd_z, phase_sc
```

两个任务共用`features`，所以两个损失都会更新共享层。这就是本次的联合学习。它没有硬性约束相位必须等于某个常数OPD乘以波数；当前全带平均OPD和局部相位在存在色散时不宜直接套用这样的强约束。

## 7. 第五步自己写：分别掩码的联合损失

采用：`总损失 = OPD的SmoothL1损失 + λ × 相位sin/cos的MSE`，初始`λ=1.0`。OPD误差在标准化空间计算。

```python
opd_loss = F.smooth_l1_loss(opd_pred[opd_mask], opd_true[opd_mask])
phase_loss = F.mse_loss(phase_pred[phase_mask], phase_true[phase_mask])
total_loss = opd_loss + phase_weight * phase_loss
```

上面是有有效标签时的核心；完整`joint_loss()`还处理某一批没有有效标签的情况。两个mask分别使用：OPD无效的光谱仍可能给相位任务提供监督。

不要写`((prediction-target)**2 * mask).mean()`：NaN乘0还是NaN，会污染整个损失。也不要将无效标签替换成0后当作真标签训练。

为什么预测sin/cos？真实179°与预测−179°，圆周上只差2°，直接做角度MSE会当作358°。sin/cos表示连续，恢复时使用：

```python
phase_pred_rad = np.arctan2(sin_pred, cos_pred)  # 顺序不能写反
phase_error = np.arctan2(np.sin(phase_pred_rad - phase_true_rad),
                        np.cos(phase_pred_rad - phase_true_rad))
```

代码不在损失前强制单位化预测向量，MSE同时约束两个分量。对于单位预测/目标向量，两分量MSE恰好等于`1-cos(相位差)`；一般原始输出不严格在单位圆上，不能直接声称两式始终相等。`atan2`只依赖方向；近零向量的相位未定义，代码单独计数并导出标记。向量模长不是经过校准的置信度。

## 8. 第六步自己写：训练循环

```python
model.train()
for batch in loader:
    optimizer.zero_grad(set_to_none=True)     # 清除上一批梯度
    predictions = model(batch[0])            # 前向传播
    loss, _, _ = joint_loss(predictions, batch)
    loss.backward()                         # 自动微分，计算梯度
    optimizer.step()                        # AdamW更新参数
```

正式代码还做设备迁移、梯度裁剪、有限值检查和按有效标签数量汇总损失。`epoch`表示完整遍历一轮训练集，100轮不是100条光谱。

验证时：

```python
model.eval()
with torch.inference_mode():
    validation_predictions = model(validation_x)
```

`eval()`关闭Dropout；`inference_mode()`关闭梯度跟踪。验证集只用于选最佳轮次和超参数，不执行反向传播。训练默认完全不读取`test.npz`。

训练曲线记录的训练损失包含Dropout且参数在轮内更新，验证损失在轮末固定参数下计算，因此两者并非完全相同条件；持续很大的差距仍需要检查泛化。

## 9. 第七步：亲自运行

先用新目录跑2轮，确认环境、路径和输出都正确：

```powershell
.\.venv-training\Scripts\python.exe -X utf8 scripts/train_joint_mlp.py train --epochs 2 --device cpu --out analysis/my_mlp_smoke
```

然后另用新目录训练100轮：

```powershell
.\.venv-training\Scripts\python.exe -X utf8 scripts/train_joint_mlp.py train --epochs 100 --device cpu --out analysis/my_mlp_100
```

默认批量64、学习率0.001、Dropout 0.1、相位权重1.0。验证联合损失连续20轮没有改善时早停，最后恢复最佳模型。已经存在的输出目录会报错，换一个`--out`即可。

若使用桌面的简化数据包，可追加`--data-dir 'C:\Users\Kyson\Desktop\迈克尔逊光谱_C方案_分组数据集_20261117'`。目录中需有同名的三份NPZ；训练只需要其中train和validation。

以后一次改一个因素，例如学习率`--lr 0.0003`、相位权重`--phase-weight 0.5`，用验证结果比较。不得根据测试结果反复选参。最终应比较多个种子和更有针对性的非CNN结构。

## 10. 第八步：认识输出文件

| 文件 | 用途 |
|---|---|
| `best.pt` | 最佳模型权重、网络结构、训练集OPD标准化参数、波数网格和相位参考波长 |
| `config.json` | 设备、依赖版本、随机种子、超参数、数据文件SHA256 |
| `history.csv` | 每轮训练/验证损失及两个任务各自的损失 |
| `validation_metrics.json` | OPD的μm误差、相位的圆周rad误差及训练均值基线 |
| `validation_predictions.csv` | 每条光谱的预测、标签、误差、位置和质量标记 |

恢复模型时需先创建同一结构，再`load_state_dict()`。OPD反标准化必须使用检查点内的训练参数；新光谱必须执行同一C预处理并匹配波数网格、1560 nm相位参考。不要把原始CSV强度直接喂入本模型。

## 11. 本次已经实际跑过的教学实验

2026-11-17，Windows / Python 3.13.5 / NumPy 2.1.3 / PyTorch 2.15.0+cpu，4线程，种子20261117，100轮。输出在`analysis/joint_mlp_tutorial_seed20261117/`；此目录不自动纳入Git。

| 验证集指标（600条） | 训练集均值常量基线 | 教学MLP，最佳第99轮 |
|---|---:|---:|
| OPD MAE / μm | 1842.845 | 887.691 |
| OPD RMSE / μm | 2131.755 | 1552.578 |
| 相位圆周MAE / rad | 1.4350 | 0.6389 |
| 相位圆周RMSE / rad | 1.6579 | 1.1243 |

这些数值高于高精度解调的期望，只能证明训练链路可用并学到了一些信息。OPD的P95绝对误差约3847.6 μm，相位P95约2.9463 rad，说明部分位置误差很大；不能只报告平均值。

训练联合损失约0.01025，最后一轮验证约0.46857。严格留出位置后，简单全连接网络对未见条纹频率的泛化不足是一个需要检验的解释；不能凭这一轮断定根本原因，也不能认定已达到性能上限。可从逐位置预测、训练分布和更合适的非CNN结构继续排查。

这次验证中，590条位于训练OPD标签范围内，10条位于范围外；代码分别报告两类。不要把它称为专门设计的外推实验。训练仅跑一个种子，没有触碰真实测试集，尚不能作为最终论文性能报告。

后续改进顺序建议：先检查逐位置误差与训练/验证差距；再对比保持复数相位信息的频域输入MLP或局部光谱块Transformer；最后确定结构后做多种子、独立测试与抗干扰评价。只输入FFT幅值会丢失相位信息；加入FFT特征后应如实描述方法，不能继续声称完全由原光谱直接学习。

## 12. 最后才单独运行测试

模型结构和超参数选定后，执行一次：

```powershell
.\.venv-training\Scripts\python.exe -X utf8 scripts/train_joint_mlp.py evaluate --checkpoint analysis/my_mlp_100/best.pt --device cpu --out analysis/my_mlp_100_test
```

会生成`test_metrics.json`和`test_predictions.csv`。当前教学MLP仍有明显误差，建议先完成验证分析再做最终测试。不能将这些伪标签误差当作独立仪器测量精度；相位仅为约定支路和参考下的包裹相位。

针对性检查：

```powershell
.\.venv-training\Scripts\python.exe -m pytest tests/test_joint_mlp.py -q
```

完整项目检查：` .\.venv-training\Scripts\python.exe -m pytest tests -q`。基础环境未安装PyTorch时，神经网络测试会跳过；训练环境中目前48项检查全部通过。

PyTorch概念可对照官方[基础教程](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)、[参数优化](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)与[模型保存和加载](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)。物理标签、数据分组和预处理约定以本仓库文档05–08为准。
