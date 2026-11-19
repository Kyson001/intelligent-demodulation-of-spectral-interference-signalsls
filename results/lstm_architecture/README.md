# LSTM网络结构图历史导出

本目录归档桌面工程里的三套图，每套包含PNG、SVG、PDF：

| 文件名前缀 | 内容 | 绘图入口 |
|---|---|---|
| `LSTM网络结构图` | 纵向教学框架与维度 | `scripts/draw_lstm_network.py` |
| `LSTM网络结构图_立体神经元版` | 立体张量、LSTM展开及全连接神经元 | `scripts/draw_lstm_architecture_3d.py` |
| `LSTM网络结构图_科研版` | 科研风横向结构图 | `scripts/draw_lstm_architecture_publication.py` |

三套图均对应`center_global`双向LSTM。网格及神经元数量是示意，实际维度以文字标注和网络定义为准。

用户随后修改的可编辑PPT单独保存在 [`../../presentations/LSTM网络结构图_可编辑版.pptx`](../../presentations/LSTM网络结构图_可编辑版.pptx)。重新运行这三个绘图脚本仅生成PNG/SVG/PDF，不会覆盖该PPT。
