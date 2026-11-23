"""Read the uploaded file formats supported by the original Flask page."""

import csv
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

from demodulation.raw_models import RAW_POINTS, WAVELENGTH_NM


MAX_UPLOAD_BYTES = 200 * 1024 * 1024
TEXT_EXTENSIONS = {'.csv', '.txt', '.dat', '.tsv', '.prn'}
EXCEL_EXTENSIONS = {'.xlsx', '.xlsm'}
EXTENSIONS = TEXT_EXTENSIONS | EXCEL_EXTENSIONS | {'.npy', '.npz', '.json'}
WAVELENGTH_KEYS = ('wavelength_nm', 'wavelength', 'wl', 'k_rad_per_m', 'k', 'x_wave')
SIGNAL_KEYS = ('intensity', 'spectra', 'spectrum', 'x', 'i', 'data', 'y', 'signal', 'value')


def numeric(value):
    if isinstance(value, bool):
        return float('nan')
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')


def wave_columns(rows, name, single=False):
    first = next((i for i, row in enumerate(rows) if row and np.isfinite(numeric(row[0]))), None)
    if first is None:
        raise ValueError('未找到有效波长列。')
    body = [row for row in rows[first:] if any(v is not None and str(v).strip() for v in row)]
    columns = 2 if single else max(map(len, body))
    if columns < 2:
        raise ValueError('至少需要波长和强度两列。')
    wave = np.array([numeric(row[0]) if row else np.nan for row in body])
    for col in range(1, columns):
        cells = [row[col] if len(row) > col else None for row in body]
        if not single and all(value is None or not str(value).strip() for value in cells):
            continue
        yield name if single else f'{name}::col{col}', wave, np.array([numeric(v) for v in cells])


def array_spectra(value, name, wavelength=None):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiub' or not array.size:
        raise ValueError('需要普通的数值光谱数组。')
    if wavelength is None and array.ndim == 2 and array.shape[0] > 2 and array.shape[1] == 2:
        yield name, array[:, 0], array[:, 1]
        return
    if (wavelength is None and array.shape == (2, RAW_POINTS)
            and array[0, 0] == 1510 and array[0, -1] == 1610):
        yield name, array[0], array[1]
        return
    wavelength = WAVELENGTH_NM if wavelength is None else np.asarray(wavelength, dtype=np.float64)
    if array.ndim == 1:
        yield name, wavelength, array
    elif array.ndim == 2:
        for i, row in enumerate(array, 1):
            yield f'{name}::{i}', wavelength, row
    else:
        raise ValueError('光谱数组需要一维或二维。')


def structured(source, name):
    keys = {key.lower(): key for key in source.keys()}
    wave_key = next((key for key in WAVELENGTH_KEYS if key in keys), None)
    signals = [key for key in SIGNAL_KEYS if key in keys]
    if not signals:
        if len(keys) != 1:
            raise ValueError('结构化文件需提供 intensity、spectra 或 X 等光谱字段。')
        yield from array_spectra(source[next(iter(keys.values()))], name)
        return
    wavelength = source[keys[wave_key]] if wave_key is not None else None
    if wave_key in ('k', 'k_rad_per_m'):
        with np.errstate(divide='ignore', invalid='ignore'):
            wavelength = (2 * np.pi * 1e9) / np.asarray(wavelength, dtype=np.float64)
    for key in signals:
        yield from array_spectra(source[keys[key]], f'{name}::{key}', wavelength)


def read_upload(file, name):
    extension = Path(name).suffix.lower()
    if extension in TEXT_EXTENSIONS:
        lines = [line for line in file.read().decode('utf-8-sig').splitlines() if line.strip()]
        first = next((line for line in lines if re.match(r'^\s*"?[+\-]?[\d.]', line)), '')
        separator = next((char for char in (',', '\t', ';') if char in first), None)
        rows = list(csv.reader(lines, delimiter=separator)) if separator else [line.split() for line in lines]
        yield from wave_columns(rows, name, single=True)
    elif extension in EXCEL_EXTENSIONS:
        from openpyxl import load_workbook

        workbook = load_workbook(io.BytesIO(file.read()), read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                rows = list(sheet.iter_rows(values_only=True))
                if any(any(cell is not None for cell in row) for row in rows):
                    yield from wave_columns(rows, f'{name}::{sheet.title}')
        finally:
            workbook.close()
    elif extension == '.npy':
        yield from array_spectra(np.load(io.BytesIO(file.read()), allow_pickle=False), name)
    elif extension == '.npz':
        payload = file.read()
        with ZipFile(io.BytesIO(payload)) as entries:
            if sum(entry.file_size for entry in entries.infolist()) > MAX_UPLOAD_BYTES:
                raise ValueError('NPZ 解压后超过 200 MB。')
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            yield from structured(data, name)
    elif extension == '.json':
        value = json.loads(file.read().decode('utf-8-sig'))
        if isinstance(value, dict):
            yield from structured(value, name)
        else:
            yield from array_spectra(value, name)
