"""Export the trained 10001-point models and verify browser-runtime outputs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from demodulation.raw_models import MODELS, RAW_POINTS, load_raw_model, recorded_intensity


class WebModel(torch.nn.Module):
    def __init__(self, model, stats):
        super().__init__()
        self.model = model
        self.mean_mm = stats['mean_mm']
        self.std_mm = stats['std_mm']

    def forward(self, intensity):
        z, sine_cosine = self.model(intensity)
        opd_m = (z * self.std_mm + self.mean_mm) * 0.001
        return torch.cat([opd_m[:, None], sine_cosine], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'models')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    inputs = []
    for sample in ('file0001', 'file0041', 'file2000', 'file4000'):
        spectrum = np.loadtxt(ROOT / 'datasets' / 'raw' / f'{sample}.csv', delimiter=',', skiprows=1)
        inputs.append(recorded_intensity(spectrum[:, 0], spectrum[:, 1])[None, :])
    checkpoints = ROOT / 'results' / 'raw_demodulation' / 'models'
    manifest = {'input_points': RAW_POINTS, 'reference_wavelength_nm': 1560,
                'outputs': ['opd_m', 'phase_sin', 'phase_cos'], 'models': {}}
    for name in MODELS:
        source = checkpoints / f'{name}.pt'
        model, metadata = load_raw_model(source, torch.device('cpu'))
        wrapper = WebModel(model, metadata['stats']).eval()
        target = args.out / f'{name}.onnx'
        with torch.no_grad():
            torch.onnx.export(wrapper, torch.from_numpy(inputs[0]), target, opset_version=17,
                              input_names=['intensity'], output_names=['prediction'])
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        session = onnxruntime.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
        for values in inputs:
            with torch.inference_mode():
                expected = wrapper(torch.from_numpy(values)).numpy()[0]
            actual = session.run(['prediction'], {'intensity': values})[0][0]
            np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=2e-9)
            np.testing.assert_allclose(actual[1:], expected[1:], rtol=2e-5, atol=2e-6)
        manifest['models'][name] = {
            'checkpoint_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'onnx_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        print(f'{name}: checked {len(inputs)} spectra')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
