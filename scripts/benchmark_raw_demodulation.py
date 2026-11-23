"""Compare the frozen raw models on the same held-out positions and one CPU thread."""

import argparse
import csv
import json
import platform
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from demodulation.joint_mlp import circular_error, metrics, predict
from demodulation.pseudo_labels import LabelConfig, label_prepared
from demodulation.raw_models import MODELS, decode_predictions, load_raw_model, recorded_intensity
from scripts.preprocess_c import CConfig, prepare_dense_spectrum, read_spectrum
from scripts.train_raw_demodulation import load_raw_split, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Frozen training directory with cnn/ and bilstm/')
    parser.add_argument('--out', type=Path, required=True, help='New result directory')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.out.exists() or args.repeats < 1:
        parser.error('--out must be new and --repeats must be positive')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cpu')
    data_dir = ROOT / 'datasets'
    data = load_raw_split(data_dir, 'test')
    for split in ('train', 'validation'):
        with np.load(data_dir / 'processed' / f'{split}.npz', allow_pickle=False) as metadata:
            for field in ('sample_ids', 'position_id', 'duplicate_group'):
                if set(metadata[field]) & set(data[field]):
                    raise ValueError(f'{field} overlaps test and {split}')
    checkpoints = {
        name: args.run / name / 'best.pt' if args.run is not None
        else ROOT / 'results' / 'raw_demodulation' / 'models' / f'{name}.pt'
        for name in MODELS
    }
    loaded = {name: load_raw_model(path, device) for name, path in checkpoints.items()}
    accuracy, predictions = {}, {}
    for name, (model, metadata) in loaded.items():
        if metadata['model'] != name:
            raise ValueError(f'Checkpoint label does not match model: {name}')
        for split, expected in metadata['split_sha256'].items():
            if sha256(data_dir / 'processed' / f'{split}.npz') != expected:
                raise ValueError(f'Checkpoint was trained with a different {split} archive')
        predictions[name] = predict(model, data, metadata['stats'], device, batch_size=64)
        accuracy[name] = metrics(data, predictions[name], metadata['stats'])
        accuracy[name].update(epoch=metadata['epoch'], parameters=metadata['parameters'],
                              checkpoint_sha256=sha256(checkpoints[name]))

    raw = [read_spectrum(data_dir / 'raw' / f'{sid}.csv') for sid in data['sample_ids']]
    label_config = LabelConfig(**json.loads((ROOT / 'configs' / 'pseudo_labels.json').read_text(encoding='utf-8')))
    preprocess_config = json.loads((ROOT / 'configs' / 'training_preprocessing.json').read_text(encoding='utf-8'))
    c_config = CConfig(background_sigma_points=preprocess_config['background_sigma_points'],
                       envelope_floor_fraction=preprocess_config['envelope_floor_fraction'],
                       model_points=preprocess_config['model_points'],
                       intensity_unit='dbm' if preprocess_config['intensity_unit'] == 'dbm' else 'recorded')

    def traditional(spectrum):
        row, _ = label_prepared(prepare_dense_spectrum(*spectrum, c_config), label_config)
        return row

    @torch.inference_mode()
    def neural(spectrum, model, metadata):
        x = torch.from_numpy(recorded_intensity(*spectrum)).unsqueeze(0)
        opd, phase = decode_predictions(model(x), metadata['stats'])
        return opd.item(), phase.item()

    methods = {'traditional': traditional}
    for name, (model, metadata) in loaded.items():
        methods[name] = lambda spectrum, model=model, metadata=metadata: neural(spectrum, model, metadata)
    order = list(methods)
    for spectrum in raw[:24]:
        for run in methods.values():
            run(spectrum)
    elapsed = {name: [] for name in order}
    direct = [None] * len(raw)
    rng = np.random.default_rng(413)
    for repeat in range(args.repeats):
        for j, ix in enumerate(rng.permutation(len(raw))):
            offset = (j + repeat) % len(order)
            for name in order[offset:] + order[:offset]:
                start = perf_counter_ns()
                result = methods[name](raw[ix])
                elapsed[name].append((perf_counter_ns() - start) / 1e6)
                if repeat == 0 and name == 'traditional':
                    direct[ix] = result
    for task, field in [('opd', 'opd_peak_m'), ('phase', 'phase_pseudo_rad')]:
        mask = data[f'{task}_valid']
        values = np.array([row[field] for row in direct])
        target = data['opd_pseudo_m'] if task == 'opd' else data['phase_pseudo_rad']
        np.testing.assert_array_equal([row[f'{task}_valid'] for row in direct], mask)
        np.testing.assert_allclose(values[mask], target[mask], rtol=1e-7, atol=1e-8)

    summary = {
        'input_points': 10001, 'reference_nm': 1560.0,
        'test_spectra': len(raw), 'test_positions': len(np.unique(data['position_id'])),
        'test_sha256': sha256(data_dir / 'processed' / 'test.npz'),
        'accuracy_reference': 'traditional pseudo labels, not independent instrument truth',
        'timing_scope': 'warmed batch=1 from raw arrays in memory to OPD/phase; excludes CSV loading and model startup',
        'traditional_path': 'high-resolution C stages, peak spacing, Fourier phase, and QC; skips unused model resampling',
        'threads': 1, 'repeats': args.repeats, 'platform': platform.platform(),
        'processor': platform.processor(), 'torch': str(torch.__version__),
        'numpy': np.__version__, 'scipy': scipy.__version__,
        'timing_ms': {name: {'p50': float(np.median(xs)), 'p95': float(np.percentile(xs, 95)), 'n': len(xs)}
                      for name, xs in elapsed.items()},
        'test_metrics': accuracy,
    }
    args.out.mkdir(parents=True)
    write_json(args.out / 'summary.json', summary)
    fields = ['sample_id', 'position_id', 'opd_valid', 'phase_valid', 'opd_pseudo_m', 'phase_pseudo_rad']
    with (args.out / 'test_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + [f'{name}_{metric}' for name in MODELS
                                for metric in ('opd_m', 'phase_rad', 'opd_abs_error_um', 'phase_abs_error_rad')])
        writer.writeheader()
        for i in range(len(raw)):
            row = {key: data['sample_ids' if key == 'sample_id' else key][i] for key in fields}
            for name, p in predictions.items():
                row.update({f'{name}_opd_m': p['opd_m'][i], f'{name}_phase_rad': p['phase_rad'][i]})
                if row['opd_valid']:
                    row[f'{name}_opd_abs_error_um'] = abs(p['opd_m'][i] - row['opd_pseudo_m']) * 1e6
                if row['phase_valid'] and p['phase_norm'][i] > 1e-6:
                    row[f'{name}_phase_abs_error_rad'] = abs(circular_error(p['phase_rad'][i], row['phase_pseudo_rad']))
            writer.writerow(row)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
