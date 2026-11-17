# 基于深度学习的端到端光谱干涉信号智能解调

重庆大学2024级智能感知工程专业《人工智能基础》课程设计。参考教材：Artificial Intelligence: A Modern Approach（第四版）。

研究对象：迈克尔逊干涉仪，目标为光程差（OPD）与参考波长处的相位联合估计。

当前已完成三种预处理的数值实现、全数据配对诊断、已知真值仿真控制和图表输出；尚未训练神经网络。没有独立真值的指标不表述为测量精度。

## 预处理比较

- A：等波数线性插值＋去常量背景。
- B：慢变背景扣除＋有下限的包络代理归一化。
- C：B＋Hann加窗。

当前按用户选择使用**C方案：包络补偿＋Hann窗**，再抗混叠至2048点并逐谱RMS归一化。完整中文注释脚本为[scripts/preprocess_c.py](scripts/preprocess_c.py)，分块讲解见[C方案使用说明](docs/05_C方案完整代码与分块讲解.md)。此前A/B/C比较结果保留于[实验报告](docs/04_三种预处理方案实验报告.md)，不将用户选择表述为已证明的精度最优。

![光谱、FFT与相位比较](results/preprocessing/figures/01_spectrum_fft_phase.png)

## 复现

建议Python 3.10或以上，在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q
python scripts/preprocess_c.py --data-dir analysis/data_4000
python scripts/compare_preprocessing.py --data-dir analysis/data_4000
python scripts/plot_preprocessing.py
python scripts/export_model_inputs.py --data-dir analysis/data_4000
python -X utf8 scripts/generate_pseudo_labels.py --data-dir analysis/data_4000
```

CSV列为波长和强度，首行为标题。当前配置针对1510–1610 nm、10001点以及约2.3–10.3 mm的诊断OPD范围，不能直接作为其他仪器的通用标定。

C方案结果默认保存到`analysis/preprocessed_C/`，预处理/制标签阶段不自动划分。新增顺序核查强烈支持从file0001起每位置连续10条；`scripts/split_dataset_by_position.py`按位置生成2800/600/600条训练、验证和测试数据，跨位置完全重复谱绑定分区。桌面包包含原始CSV、C输入、伪标签、分组清单和完整代码，详见[位置划分与交付说明](docs/07_按采集位置分组与数据包说明.md)。

```powershell
python -X utf8 scripts/split_dataset_by_position.py
python -X utf8 scripts/verify_dataset_package.py "桌面数据包路径" --verify-raw-hashes
```

已实现多干涉峰频率间距OPD伪标签和1560 nm中心相位伪标签。全部4000条已计算，3999条OPD和3992条相位通过当前QC，8条保留待复核。中心相位使用正侧带局部二次拟合，另有峰位截距相位用于交叉检查。详见[伪标签算法与相位说明](docs/06_光程差与相位伪标签生成说明.md)。

伪标签及合并后的C输入默认保存在`analysis/pseudo_labels/`，合并前检查sample_id、原始文件哈希和预处理配置。原始数据和论文不在Git中分发；缓存距离标签仍未核实，新增算法标签也不是独立测量真值。

## 目录

- `src/demodulation/`：预处理、FFT与相位诊断估计器。
- `scripts/`：数据审计、配对比较、绘图及模型输入导出。
- `configs/`：比较和训练输入配置。
- `tests/`：已知OPD、相位边界、采样与归一化测试。
- `results/preprocessing/`：汇总、逐样本结果和可编辑图表。
- `docs/`：材料分析、方案、预处理协议及实验报告。

后续展示目标为PPT、网页、文档、代码及可复现结果。
