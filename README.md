# 基于深度学习的端到端光谱干涉信号智能解调

重庆大学2024级智能感知工程专业《人工智能基础》课程设计。参考教材：Artificial Intelligence: A Modern Approach（第四版）。

研究对象：迈克尔逊干涉仪，目标为光程差（OPD）与参考波长处的相位联合估计。

当前已完成三种预处理的数值实现、全数据配对诊断、已知真值仿真控制和图表输出；尚未训练神经网络。没有独立真值的指标不表述为测量精度。

## 预处理比较

- A：等波数线性插值＋去常量背景。
- B：慢变背景扣除＋有下限的包络代理归一化。
- C：B＋Hann加窗。

首轮网络选择A，加抗混叠2048点重采样及逐谱RMS归一化；B保留为消融，C保留为FFT对照。详见[实验报告](docs/04_三种预处理方案实验报告.md)。

![光谱、FFT与相位比较](results/preprocessing/figures/01_spectrum_fft_phase.png)

## 复现

建议Python 3.10或以上，在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q
python scripts/compare_preprocessing.py --data-dir analysis/data_4000
python scripts/plot_preprocessing.py
python scripts/export_model_inputs.py --data-dir analysis/data_4000
```

CSV列为波长和强度，首行为标题。当前配置针对1510–1610 nm、10001点以及约2.3–10.3 mm的诊断OPD范围，不能直接作为其他仪器的通用标定。

原始数据和论文不在Git中分发；需要自行取得授权数据。缓存距离标签为未核实的历史派生数据，原始标签表和独立相位真值仍待补齐。导出程序不会将其冒充OPD/相位真值。

## 目录

- `src/demodulation/`：预处理、FFT与相位诊断估计器。
- `scripts/`：数据审计、配对比较、绘图及模型输入导出。
- `configs/`：比较和训练输入配置。
- `tests/`：已知OPD、相位边界、采样与归一化测试。
- `results/preprocessing/`：汇总、逐样本结果和可编辑图表。
- `docs/`：材料分析、方案、预处理协议及实验报告。

后续展示目标为PPT、网页、文档、代码及可复现结果。
