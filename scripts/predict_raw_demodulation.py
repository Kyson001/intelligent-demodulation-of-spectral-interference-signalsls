"""Predict OPD in meters and 1560 nm wrapped phase in radians from raw CSVs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from demodulation.raw_models import MODELS, decode_predictions, load_raw_model, recorded_intensity
from scripts.preprocess_c import read_spectrum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, nargs='+')
    parser.add_argument('--model', choices=MODELS, default='cnn')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint or ROOT / 'results' / 'raw_demodulation' / 'models' / f'{args.model}.pt'
    torch.set_num_threads(1)
    device = torch.device('cpu')
    model, metadata = load_raw_model(checkpoint, device)
    samples = np.stack([recorded_intensity(*read_spectrum(path)) for path in args.csv])
    with torch.inference_mode():
        opd, phase = decode_predictions(model(torch.from_numpy(samples)), metadata['stats'])
    predictions = [{'file': str(path), 'model': metadata['model'], 'opd_m': float(m),
                    'phase_rad': float(p), 'reference_nm': metadata['reference_nm']}
                   for path, m, p in zip(args.csv, opd, phase, strict=True)]
    print(json.dumps(predictions, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
