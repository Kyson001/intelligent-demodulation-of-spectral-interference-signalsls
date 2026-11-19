# CNN 光谱干涉联合解调：阅读目录与运行说明

本目录独立保存 CNN 成果：输入迈克尔逊光谱的 C 方案预处理结果，预测光程差 OPD 与 1560 nm 中心包裹相位。包含初版、第二版和固定两代模型的鲁棒性评价；鲁棒性部分没有新增训练。

## 先看哪些文件

|阅读顺序|文件|用途|
|---|---|---|
|1|[项目说明：从入门到汇报](docs/项目说明_从入门到汇报.md)|理解初版不足、优化实验、第二版取舍、逐层网络参数、训练过程和结果|
|2|[CNN真实计算图](figures/01_CNN真实计算图.png)|看三层卷积、实际特征图、OPD/相位两个输出头|
|3|[关键算法展开](figures/02_关键算法展开.png)|看预处理、卷积计算、标签、损失和训练反馈|
|4|[鲁棒性结果图](figures/03_鲁棒性结果.png)|看原始域加噪与包络变化的影响|
|5|[CNN完整程序.py](CNN完整程序.py)|唯一的项目代码文件，包含训练、预测、预处理与鲁棒性评价|

`figures/` 同时提供 PNG 和可编辑 SVG。`requirements.txt` 是第三方依赖列表。

## 两个模型怎么用

|文件|含义|
|---|---|
|[models/initial.pt](models/initial.pt)|保留的第一版权重，供回退和对照|
|[models/second.pt](models/second.pt)|第二版最终权重，默认预测使用它|

模型结构仍为三层 1D 卷积：1→16（核15、膨胀1）、16→32（核9、膨胀2）、32→32（核9、膨胀4），各层 GroupNorm+SiLU；共享特征后分成 OPD 和相位预测头。最终第二版主要把 OPD 损失权重从1调整到100，并按双任务位置宏平均指标选模。多尺度和仿射校准做了对照，没有纳入最终推理。

## 下载后如何运行

下载**整个仓库**，数据直接使用上级 `datasets/processed/` 和 `datasets/raw/`。无需另放本机历史项目文件夹。以下命令均在仓库根目录执行：

```powershell
python -m pip install -r cnn/requirements.txt
python cnn/CNN完整程序.py
```

直接运行会加载第二版预测验证集，再用两代模型执行30种扰动条件的鲁棒性评价。默认不训练；鲁棒性重跑要处理600条原始光谱，耗时长于单次预测。

只预测或只做鲁棒性评价：

```powershell
python cnn/CNN完整程序.py --mode predict
python cnn/CNN完整程序.py --mode robustness
```

用初版预测，或输入自己的同格式 C 方案 NPZ：

```powershell
python cnn/CNN完整程序.py --mode predict --checkpoint cnn/models/initial.pt
python cnn/CNN完整程序.py --mode predict --input new_C.npz
```

预测输入不是原始 CSV，必须符合仓库既有 C 预处理、2048点波数网格及参考相位约定。程序检查 NPZ 与检查点的输入约定。

显式重新训练第二版最终配置：

```powershell
python cnn/CNN完整程序.py --mode train --seed 20261119 --epochs 160
```

新训练只读取 train/validation，并将新权重写到 `cnn/outputs/日期时间/best.pt`；不会覆盖 `models/`。所有模式默认创建新的时间戳目录，也可用 `--out` 指定一个尚不存在的目录。`--project-root` 可指定另一份完整仓库的根目录；鲁棒性数据及两代模型都从该根目录读取。`--data-dir` 用于训练或普通预测的数据目录，鲁棒性仍按仓库固定的原始/验证集进行。

本单文件复现的是最终配置训练、推理与30场景评价；历史筛选、三种子结构对照和校准诊断的结果已归档，直接运行不会自动重做整个模型选择过程。GPU 完全重现训练数值受硬件和运行环境影响。`--device` 控制普通预测，训练与鲁棒性当前自动选择可用 GPU，否则使用 CPU。

## 得到了什么结果

|历史测试指标|初版|第二版|
|---|---:|---:|
|OPD MAE / μm|2.929|0.913|
|OPD P95 / μm|6.764|2.650|
|相位圆周 MAE / rad|0.01354|0.01796|

第二版改善 OPD，相位误差有所增加，应同时汇报。历史测试集已在迭代中查看，不能称为独立盲测。标签是传统算法生成的伪标签，这些指标表示与伪标签的一致性，不代表独立标定的物理精度。

鲁棒性使用已有验证集600条/60个位置：第二版干净 OPD MAE 为0.780 μm；1%、2%、5%合成噪声下分别约0.813、1.066、4.761 μm。这里的百分比是高斯噪声标准差相对于平滑背景峰值的比例，不是仪器 SNR。评价没有访问测试集，也没有进行增强训练；跨批次泛化尚未验证。

## 结果证据在哪里

|目录/文件|内容|
|---|---|
|[results/improvements/comparison_summary.json](results/improvements/comparison_summary.json)|同预算三种子配置比较|
|[results/improvements/confirmation_summary.json](results/improvements/confirmation_summary.json)|各次训练配置与确认结果|
|`results/improvements/*test_metrics.json`|两代历史测试指标|
|[results/improvements/second_test_predictions.csv](results/improvements/second_test_predictions.csv)|第二版历史测试逐样本预测|
|[results/robustness/protocol.json](results/robustness/protocol.json)|固定扰动协议|
|[results/robustness/scenario_metrics.csv](results/robustness/scenario_metrics.csv)|30场景×2模型的完整指标|
|[results/robustness/noise_aggregate.json](results/robustness/noise_aggregate.json)|各噪声级别三次随机实现的汇总|
|[results/robustness/summary.json](results/robustness/summary.json)|干净基线和评价范围|

结果 JSON 是既有实验档案，可能包含当时运行目录或旧模块名，仅用于追溯；当前执行入口以本页为准。第1–4项优化实验和第6项评价已完成，第5项注意力、第7项仿真辅助训练未实施。模型、图和精选结果直接放在本目录，未重复上传原始数据、文献、环境、中间模型或压缩包。
