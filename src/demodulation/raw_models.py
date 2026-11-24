"""Joint demodulation from the recorded 10,001-point wavelength grid."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


RAW_POINTS = 10001
REFERENCE_INDEX = 5000
WAVELENGTH_NM = 1510.0 + np.arange(RAW_POINTS) * 0.01


def recorded_intensity(wavelength_nm, intensity):
    """Check the instrument grid, then cast the recorded intensities for PyTorch."""
    if (wavelength_nm.shape != (RAW_POINTS,) or intensity.shape != (RAW_POINTS,)
            or not np.allclose(wavelength_nm, WAVELENGTH_NM, rtol=0, atol=1e-6)):
        raise ValueError('Expected 10,001 wavelengths from 1510 to 1610 nm at 0.01 nm intervals')
    if not np.isfinite(intensity).all() or np.max(np.abs(intensity)) > np.finfo(np.float32).max:
        raise ValueError('Recorded intensity is outside the finite float32 range')
    return np.ascontiguousarray(intensity, dtype=np.float32)


class JointReadout(nn.Module):
    """Global statistics estimate OPD. Ordered reference features estimate phase."""

    def __init__(self, channels, center, radius):
        super().__init__()
        self.center, self.radius = center, radius
        self.opd = nn.Sequential(
            nn.Linear(2 * channels, 96), nn.SiLU(), nn.Linear(96, 1),
        )
        self.phase = nn.Sequential(
            nn.Linear((2 * radius + 2) * channels, 96), nn.SiLU(), nn.Linear(96, 2),
        )

    def forward(self, features):
        global_features = torch.cat(
            (features.mean(dim=1), features.square().mean(dim=1).clamp_min(1e-10).sqrt()),
            dim=-1,
        )
        local = features[:, self.center - self.radius:self.center + self.radius]
        return (
            self.opd(global_features).squeeze(-1),
            self.phase(torch.cat((local.flatten(1), global_features), dim=-1)),
        )


class RawCNN(nn.Module):
    """Learned strided features on the original wavelength axis."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 24, 65, stride=8, padding=32, bias=False),
            nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv1d(24, 32, 9, stride=2, padding=4),
            nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv1d(32, 48, 7, dilation=2, padding=6),
            nn.GroupNorm(4, 48), nn.SiLU(),
        )
        self.readout = JointReadout(48, REFERENCE_INDEX // 16, radius=12)

    def forward(self, x):
        # This inexpensive tensor normalization belongs to the network graph.
        # The caller supplies the original recorded intensities without the traditional preprocessing pipeline.
        x = F.layer_norm(x, (RAW_POINTS,))
        return self.readout(self.features(x.unsqueeze(1)).transpose(1, 2))


class RawBiLSTM(nn.Module):
    """Encode contiguous patches with a learned projection and a bidirectional LSTM."""

    patch_size = 32

    def __init__(self):
        super().__init__()
        self.patch_projection = nn.Sequential(
            nn.Linear(self.patch_size, 32), nn.LayerNorm(32), nn.SiLU(),
        )
        self.lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
        self.readout = JointReadout(64, REFERENCE_INDEX // self.patch_size, radius=8)

    def forward(self, x):
        x = F.layer_norm(x, (RAW_POINTS,))
        padding = (-RAW_POINTS) % self.patch_size
        patches = F.pad(x, (0, padding)).reshape(x.shape[0], -1, self.patch_size)
        # Remove the patch mean inside the model so projection weights see local fringes.
        features = self.patch_projection(patches - patches.mean(dim=-1, keepdim=True))
        encoded, _ = self.lstm(features)
        return self.readout(encoded)


MODELS = {'cnn': RawCNN, 'bilstm': RawBiLSTM}


def decode_predictions(output, stats):
    z, sc = output
    opd_m = (z * stats['std_mm'] + stats['mean_mm']) * 1e-3
    return opd_m, torch.atan2(sc[:, 0], sc[:, 1])


def load_raw_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['input_points'] != RAW_POINTS or payload['reference_nm'] != 1560.0:
        raise ValueError('Checkpoint uses a different raw spectrum grid')
    model = MODELS[payload['model']]().to(device)
    model.load_state_dict(payload['state_dict'])
    return model.eval(), payload
