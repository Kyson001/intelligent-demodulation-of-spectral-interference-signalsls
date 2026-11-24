# 独立预处理代码

`预处理.py`收录先前交付的独立预处理脚本，包含中文分块注释。它与仓库主入口 [`../../scripts/preprocess_c.py`](../../scripts/preprocess_c.py)实现同一流程：波长转波数、等间隔插值、背景扣除、带下限的包络补偿、Hann窗、抗混叠降采样和逐谱归一化。

在仓库根目录执行：

```powershell
python -X utf8 examples/preprocessing_c/预处理.py --data-dir datasets/raw --out analysis/standalone_preprocessing_C
```

此步骤输出预处理输入和质量记录；伪标签生成及按位置分区继续使用 `scripts/generate_pseudo_labels.py` 和 `scripts/split_dataset_by_position.py`。已整理的数据在 `datasets/processed/`。
