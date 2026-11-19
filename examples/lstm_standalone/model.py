"""不使用卷积的光谱块双向LSTM：共享编码器联合估计OPD和参考相位。"""
import math

import numpy as np
import torch
from torch import nn


# 第一块：用已有波数坐标计算1560 nm参考点的位置，不读取任何标签。
def reference_sample_index(data):
    k_ref = 2 * np.pi / (float(data['reference_wavelength_nm']) * 1e-9)
    grid = data['k_rad_per_m']
    if not grid[0] <= k_ref <= grid[-1]:
        raise ValueError('相位参考波长不在输入波段内')
    return float(np.interp(k_ref, grid, np.arange(len(grid))))


class JointLSTM(nn.Module):
    """序列步沿递增波数排列，不是将同位置的10次采集拼成时间序列。

    [B, 2048] -> [B, 256, 8] -> BiLSTM -> 全局/参考点特征 -> 双任务头。
    reshape仅重新组织8个连续采样点，不平均、不降采样，不丢弃输入点。
    """
    def __init__(self, input_dim=2048, patch_size=8, hidden_size=32,
                 num_layers=1, dropout=0.1, reference_index=991.0,
                 readout='center_global'):
        super().__init__()
        if input_dim < 1 or patch_size < 1 or input_dim % patch_size:
            raise ValueError('输入点数必须为正且能被光谱块长度整除')
        if hidden_size < 1 or num_layers < 1 or not 0 <= dropout < 1:
            raise ValueError('隐藏维度/层数必须为正，且0≤dropout<1')
        if not 0 <= reference_index <= input_dim - 1:
            raise ValueError('参考点超出光谱范围')
        if readout not in ('center_global', 'last'):
            raise ValueError('readout必须为center_global或last')
        self.input_dim, self.patch_size = input_dim, patch_size
        self.readout = readout

        # 第二块：标准双向LSTM；每一步输入相邻8点，隐藏状态每个方向32维。
        # 单层LSTM内部dropout不生效，因此这里设为0，正则化放在后面的读出层。
        self.lstm = nn.LSTM(input_size=patch_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)

        # 第三块：将实际参考点映射到光谱块中心坐标，准备线性插值索引。
        sequence_length = input_dim // patch_size
        token_index = (reference_index - (patch_size - 1) / 2) / patch_size
        token_index = min(max(token_index, 0), sequence_length - 1)
        self.left = math.floor(token_index)
        self.right = min(self.left + 1, sequence_length - 1)
        self.fraction = float(token_index - self.left)

        # 第四块：全局均值/标准差帮助表征条纹频率，参考点特征帮助表征局部相位。
        # 这是待验证的结构选择，不是由物理推导保证正确的估计器。
        width = hidden_size * 2
        feature_dim = width * (3 if readout == 'center_global' else 1)
        self.shared = nn.Sequential(nn.Linear(feature_dim, 64), nn.LayerNorm(64),
                                    nn.GELU(), nn.Dropout(dropout))
        self.opd_head = nn.Linear(64, 1)
        self.phase_head = nn.Linear(64, 2)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f'输入必须为[B, {self.input_dim}]')
        # 第五块：每条光谱独立编码；不传入上一条光谱的隐藏状态，防止样本间串联。
        sequence = x.reshape(x.shape[0], -1, self.patch_size)
        encoded, (h_n, _) = self.lstm(sequence)
        if self.readout == 'last':
            # 双向LSTM最后一层的正/反向终态。不能用encoded[:, -1]冒充两方向终态。
            features = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            mean = encoded.mean(dim=1)
            std = encoded.std(dim=1, unbiased=False)
            center = ((1 - self.fraction) * encoded[:, self.left]
                      + self.fraction * encoded[:, self.right])
            features = torch.cat([mean, std, center], dim=-1)
        shared = self.shared(features)
        # 第六块：输出标准化OPD和相位sin/cos，由train.py计算损失并更新权重。
        return self.opd_head(shared).squeeze(-1), self.phase_head(shared)
