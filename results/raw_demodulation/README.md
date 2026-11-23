# 10001 点原始光谱联合解调

为了提高预测速度，两种网络直接接收固定波长网格上的 **10001 点原始强度**，输出光程差和 **1560 nm 包裹相位**。预测入口仅检查 CSV 波长网格和强度，并转换为 `float32`。CNN 用步长卷积提取特征，BiLSTM 将光谱分成 32 点连续块，末块补零后共 313 步。网络图内有归一化层。

C 方案的高分辨率阶段为离线制标签提供等波数光谱、包络补偿谱和加窗谱。随后传统算法用多峰间距计算光程差，用局部傅里叶相位估计参考点相位。训练按样本编号和原始文件 SHA-256 对齐，使用同一个采集位置分区。新模型训练使用 NPZ 的标签和分区信息，光谱输入来自原始 CSV。

## 结果

| 模型 | OPD MAE / μm | OPD P95 / μm | 相位圆周 MAE / rad | 相位圆周 P95 / rad |
|---|---:|---:|---:|---:|
| CNN | 1.103 | 3.570 | 0.01410 | 0.04214 |
| BiLSTM | 10.203 | 45.539 | 0.02721 | 0.07447 |

两模型分别在 600 条有效 OPD、598 条有效相位标签上评价。训练用 2800 条，验证用 600 条，冻结最佳权重后使用 600 条测试光谱。通过检查确认三个分区没有共享采集位置、样本编号或精确重复谱。训练和选模只访问训练及验证数据。两模型使用同一训练种子 20261123，CNN 的最佳轮次为 125，BiLSTM 为 129。

| 方法 | P50 / ms | P95 / ms | 单帧计时次数 |
|---|---:|---:|---:|
| 传统算法 | 9.65 | 12.07 | 1800 |
| CNN | 1.31 | 1.77 | 1800 |
| BiLSTM | 0.97 | 1.34 | 1800 |

速度在 Intel Core i9-13980HX 上使用单线程，预热后对同一 600 条光谱逐条交错运行三遍。计时从内存中的原始波长和强度开始，至输出光程差和相位结束。CSV 读取和权重启动在计时范围之外。传统基线保留高分辨率 C 阶段、寻峰、相位提取及质量检查，省去仅用于历史网络输入的降采样。脚本核对了传统输出与有效伪标签的一致性。模型误差同样以伪标签为参考，实际物理精度尚待独立真值验证。

- [CNN 权重](models/cnn.pt) · [训练配置](cnn/config.json) · [训练曲线数据](cnn/history.json) · [验证指标](cnn/validation_metrics.json)
- [BiLSTM 权重](models/bilstm.pt) · [训练配置](bilstm/config.json) · [训练曲线数据](bilstm/history.json) · [验证指标](bilstm/validation_metrics.json)
- [测试指标与计时](summary.json) · [逐光谱预测](test_predictions.csv)

## 复现

在仓库根目录执行。预测输出字段 `opd_m` 和 `phase_rad` 分别为米和弧度。

```powershell
python scripts/predict_raw_demodulation.py --model cnn datasets/raw/file0001.csv
python scripts/predict_raw_demodulation.py --model bilstm datasets/raw/file0001.csv

$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python scripts/benchmark_raw_demodulation.py --out analysis/raw_benchmark_replay
```

从头训练使用 `python scripts/train_raw_demodulation.py --device cuda --out analysis/raw_train --epochs 140`。此次训练环境为 PyTorch 2.5.1+cu121 和 RTX 4060 Laptop GPU。计时使用 CPU 版 PyTorch 2.15.0。现有 WorkBuddy 网页继续使用历史权重，原始光谱权重可从本目录取得。
