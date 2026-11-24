"""Train both raw-spectrum models against the existing position-split pseudo labels."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from demodulation.joint_mlp import (
    SpectralDataset, check_no_leakage, fit_opd_stats, joint_loss, load_split, metrics, predict,
)
from demodulation.raw_models import MODELS, RAW_POINTS, recorded_intensity


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw_split(data_dir, split):
    """Use the NPZ metadata and labels, and replace its preprocessed input with hash-checked CSV intensity."""
    source = data_dir / 'processed' / f'{split}.npz'
    data = load_split(source)
    x = np.empty((len(data['sample_ids']), RAW_POINTS), dtype=np.float32)
    for i, (sample_id, expected_hash) in enumerate(zip(data['sample_ids'], data['source_sha256'], strict=True)):
        raw_file = data_dir / 'raw' / f'{sample_id}.csv'
        payload = raw_file.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f'Raw CSV and pseudo label have different source hashes: {sample_id}')
        raw = np.loadtxt(io.BytesIO(payload), delimiter=',', skiprows=1, dtype=np.float64)
        if raw.shape != (RAW_POINTS, 2):
            raise ValueError(f'Expected a two-column raw spectrum: {raw_file}')
        x[i] = recorded_intensity(raw[:, 0], raw[:, 1])
    data['X'] = x
    return data


def score(report):
    return report['opd_um_position_macro_mae'] / 5 + report['phase_rad_position_macro_mae'] / 0.03


def train_one(name, train, validation, args, device, split_hashes):
    directory = args.out / name
    directory.mkdir()
    torch.manual_seed(args.seed)
    stats = fit_opd_stats(train)
    spectra = SpectralDataset(train, stats)
    tensors = tuple(t.to(device) for t in (
        spectra.x, spectra.opd, spectra.phase, spectra.opd_mask, spectra.phase_mask,
    ))
    model = MODELS[name]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    parameters = sum(w.numel() for w in model.parameters())
    config = {
        'model': name, 'input_points': RAW_POINTS, 'reference_nm': 1560.0,
        'stats': stats, 'seed': args.seed, 'batch_size': args.batch_size,
        'learning_rate': args.lr, 'opd_loss_weight': args.opd_weight,
        'epoch_limit': args.epochs, 'patience': args.patience, 'parameters': parameters,
        'torch': str(torch.__version__), 'numpy': np.__version__,
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'split_sha256': split_hashes,
        'selection': 'validation position-macro OPD MAE / 5 um + phase circular MAE / 0.03 rad',
    }
    write_json(directory / 'config.json', config)
    history = []
    best_epoch, best_score = 0, float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = torch.zeros((), device=device)
        for indices in torch.randperm(len(spectra), device=device).split(args.batch_size):
            batch = tuple(t[indices] for t in tensors)
            optimizer.zero_grad(set_to_none=True)
            _, opd_loss, phase_loss = joint_loss(model(batch[0]), batch)
            loss = args.opd_weight * opd_loss + phase_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses += loss.detach() * len(indices)

        prediction = predict(model, validation, stats, device, batch_size=args.batch_size)
        report = metrics(validation, prediction, stats)
        val_score = score(report)
        train_loss = float(losses / len(spectra))
        if not np.isfinite([train_loss, val_score]).all():
            raise RuntimeError(f'{name} produced a non-finite training or validation loss')
        scheduler.step(val_score)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'score': val_score,
                        'opd_um_mae': report['opd_um_mae'], 'phase_rad_mae': report['phase_rad_mae']})
        if val_score < best_score:
            best_epoch, best_score = epoch, val_score
            torch.save({
                'format_version': 1, **config, 'epoch': epoch, 'validation_score': val_score,
                'state_dict': {k: w.detach().cpu() for k, w in model.state_dict().items()},
            }, directory / 'best.pt')
            write_json(directory / 'validation_metrics.json', report)
        if epoch == 1 or epoch % 5 == 0:
            print(f'{name} {epoch:3d} raw OPD MAE={report["opd_um_mae"]:.3f} um '
                  f'phase MAE={report["phase_rad_mae"]:.5f} rad best={best_epoch}', flush=True)
        if epoch - best_epoch >= args.patience:
            break
    write_json(directory / 'history.json', history)
    print(f'{name} finished at epoch {epoch}, best epoch {best_epoch}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=(*MODELS, 'both'), default='both')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--seed', type=int, default=20261123)
    parser.add_argument('--epochs', type=int, default=140)
    parser.add_argument('--patience', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--opd-weight', type=float, default=40)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must be a new directory')
    if min(args.epochs, args.patience, args.batch_size, args.lr, args.opd_weight) <= 0:
        parser.error('training arguments must be positive')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    torch.set_num_threads(2)
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    print(f'Loading source-matched raw spectra for training and validation on {device}', flush=True)
    train, validation = (load_raw_split(args.data_dir, split) for split in ('train', 'validation'))
    check_no_leakage(train, validation)
    split_hashes = {split: sha256(args.data_dir / 'processed' / f'{split}.npz')
                    for split in ('train', 'validation')}
    args.out.mkdir(parents=True)
    for name in MODELS if args.model == 'both' else (args.model,):
        train_one(name, train, validation, args, device, split_hashes)


if __name__ == '__main__':
    main()
