# 独立 LSTM 教学工程

本目录保存此前桌面工程的 `model.py`、`train.py`、依赖清单及中文运行说明。代码保留中文分块注释；输入为2048点预处理光谱，联合输出OPD和相位sin/cos。

三份数据已经位于仓库的 `datasets/processed/`。在**仓库根目录**使用已安装训练依赖的Python执行：

```powershell
python -X utf8 examples/lstm_standalone/prepare_data.py
python -X utf8 examples/lstm_standalone/train.py --epochs 100
```

第一条命令只复制并核验三份NPZ，供默认的 `data/` 路径使用。也可以不复制数据，直接运行：

```powershell
python -X utf8 examples/lstm_standalone/train.py --data-dir datasets/processed --epochs 100
```

训练结果默认写入本目录的 `outputs/`，新训练从随机初始化开始。已经训练好的参考权重位于 [`../../results/lstm_comparison/runs/lstm_center_global/best.pt`](../../results/lstm_comparison/runs/lstm_center_global/best.pt)，其验证指标与训练日志在同一目录。

若要导出可单独放在桌面的完整工程，在仓库根目录运行：

```powershell
python -X utf8 scripts/build_standalone_lstm.py --out analysis/my_standalone_lstm
```

导出目录必须尚不存在。导出包括三份数据和代码，不包含本机虚拟环境。`先读我_运行步骤.md`保留此前桌面版本的详细讲解，其中桌面路径是当时的示例，实际使用时替换为自己的目录。
