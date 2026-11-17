# 数据集

| 目录 | 内容 |
|---|---|
| `raw/` | 4000个处理前CSV，每条10001点，1510–1610 nm |
| `processed/train.npz` | 训练集：280个位置、2800条C方案输入与伪标签 |
| `processed/validation.npz` | 验证集：60个位置、600条 |
| `processed/test.npz` | 测试集：60个位置、600条 |
| `manifests/` | 逐条分组、位置表、伪标签CSV、顺序核查和划分说明 |
| `manifest.json` | 三份NPZ的SHA256、数量、格式和预处理约定 |

NPZ是NumPy压缩数组文件，不是目录。读取示例：

```python
import numpy as np
d = np.load('datasets/processed/train.npz')
X = d['X']                         # [2800, 2048]，float32
y_opd = d['opd_pseudo_m']           # 单位m
y_sin, y_cos = d['phase_sin'], d['phase_cos']
opd_mask, phase_mask = d['opd_valid'], d['phase_valid']
positions = d['position_id']
```

同位置10条不跨分区，跨位置完全重复谱绑定分区，随机种子20261117。5条位置一致性复核标记和8条伪标签QC复核记录均保留。任务损失应先按各自掩码选有效目标，不将复核记录改成零标签。

当前标签由多峰间距和局部傅里叶相位算法生成，不是独立仪器真值。相位参考波长1560 nm、包裹表示、正OPD支路；强度单位保持采集数值，仍需核实物理单位。

校验所有原始文件及分组：

```powershell
python -X utf8 scripts/verify_repository_data.py
```
