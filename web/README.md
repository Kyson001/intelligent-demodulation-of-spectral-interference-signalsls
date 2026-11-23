# 光谱干涉智能解调网页

为了提高预测速度，网页提供原始光谱 CNN 和双向 LSTM。两者直接读取 **10001 点原始强度**，联合输出光程差和 **1560 nm 包裹相位**。C 方案的高分辨率阶段用于离线制标签，新版网页预测跳过 C 方案。

## 在线演示

[10001 点原始光谱版](https://kyson001.github.io/intelligent-demodulation-of-spectral-interference-signalsls/)

GitHub Pages 版本使用浏览器内的 ONNX Runtime 解调，文件留在本机。页面支持单文件、多文件、文件夹、逐条结果、顺序汇总和 CSV 导出。网页上的对照耗时来自仓库单线程 PyTorch CPU 实验。

## 光谱格式

- 波长从 **1510 nm 到 1610 nm**，按 **0.01 nm** 递增，每条 **10001 点**。点数或网格不符时，页面会指出对应文件。
- CSV、TXT、DAT、TSV、PRN 使用波长和强度两列，可包含表头。XLSX/XLSM 的首列为波长，随后每列是一条强度记录。旧版 XLS 可先另存为 XLSX。
- NPY、NPZ 和 JSON 可提供波长与数值强度数组。含 `wavelength_nm` 或 `wavelength` 的结构化文件使用显式波长。仅提供强度的数组须由采集端保证满足同一波长网格。
- 页面以微米显示 OPD，以弧度显示包裹相位。OPD/2 列是光程差的一半。

## 本地或 WorkBuddy 部署

Flask 入口使用仓库中的 PyTorch 权重。保留仓库内的目录关系，从仓库根目录启动。

```powershell
python -m pip install -r web/requirements.txt
python web/app.py
```

访问 `http://127.0.0.1:5000`。可用 `PORT` 指定端口。部署到原 WorkBuddy 链接时，在原发布项目中同步 `web/`、`src/demodulation/raw_models.py` 与 `results/raw_demodulation/models/` 后更新发布。

服务端上传接口为 `POST /api/cnn/upload`，form-data 中使用 `files` 和 `model=cnn` 或 `model=bilstm`。历史参数 `cnn_best` 和 `lstm` 分别映射到新版 CNN 和 BiLSTM。当前会话的结果仍可通过 `GET /api/cnn/result` 和 `GET /api/cnn/export` 获取。

## 构建静态站

```powershell
cd web
npm ci
npm run build
python -m http.server 8000 --directory dist
```

浏览器访问 `http://127.0.0.1:8000`。GitHub Actions 发布同一份 `dist/`。浏览器模型与 PyTorch 检查点的 SHA-256 对应关系见 [模型清单](models/manifest.json)。如需更新权重，在安装了 `torch`、`onnx` 和 `onnxruntime` 的 Python 环境下从仓库根目录运行 `python web/export_models.py`。
