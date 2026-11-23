"""Flask entry point for the raw-spectrum web demo."""

from collections import OrderedDict
import csv
from functools import lru_cache
import io
import os
from pathlib import Path
import re
import secrets
import sys
from threading import Lock
from time import monotonic
from urllib.parse import urlencode

import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_from_directory


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from demodulation.raw_models import MODELS, RAW_POINTS, decode_predictions, load_raw_model, recorded_intensity
from web.spectra import EXTENSIONS, MAX_UPLOAD_BYTES, read_upload


app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES
model_dir = ROOT / 'results' / 'raw_demodulation' / 'models'
device = torch.device('cpu')
torch.set_num_threads(1)

GRID_ERROR = '需要 10001 点，波长 1510–1610 nm（间隔 0.01 nm，按升序排列）。'
ALIASES = {'cnn_best': 'cnn', 'best': 'cnn', 'lstm': 'bilstm'}
COOKIE_NAME = 'spectral_result'
recent = OrderedDict()
recent_lock = Lock()


@lru_cache(maxsize=2)
def model_for(name):
    return load_raw_model(model_dir / f'{name}.pt', device)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error='上传文件总大小超过 200 MB，请分批解调。'), 413


def page(filename):
    html = (BASE / filename).read_text(encoding='utf-8')
    html = html.replace('data-runtime="browser"', 'data-runtime="flask"')
    if sharecode := request.args.get('sharecode'):
        query = urlencode({'sharecode': sharecode})
        for asset in ('', 'style.css', 'app.js', 'source.html'):
            html = html.replace(f'"./{asset}"', f'"./{asset}?{query}"')
    return Response(html, mimetype='text/html')


@app.get('/')
def index():
    return page('index.html')


@app.get('/source')
@app.get('/source.html')
def source():
    return page('source.html')


@app.get('/<any(app.js,style.css):asset>')
def asset_file(asset):
    return send_from_directory(BASE, asset)


def model_status(name):
    path = model_dir / f'{name}.pt'
    return dict(available=path.is_file(), model=path.name if path.is_file() else None,
                input_points=RAW_POINTS, reference_nm=1560)


@app.get('/api/models/status')
def status():
    states = {name: model_status(name) for name in MODELS}
    return jsonify(**states, lstm=states['bilstm'])


@app.get('/api/cnn/status')
def cnn_status():
    return jsonify(model_status('cnn'))


def natural_name(file):
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r'(\d+)', file.filename or '')]


@torch.inference_mode()
def predict_spectrum(model, metadata, order, name, wavelength, intensity):
    try:
        values = recorded_intensity(wavelength, intensity)
    except ValueError as error:
        if str(error).startswith('Expected'):
            raise ValueError(GRID_ERROR) from error
        raise ValueError('强度缺失或超出有效数值范围。') from error
    output = model(torch.from_numpy(values).unsqueeze(0))
    opd, phase = decode_predictions(output, metadata['stats'])
    sine, cosine = output[1][0].tolist()
    phase_norm = float(np.hypot(sine, cosine))
    opd_um = opd.item() * 1e6
    if not np.isfinite([opd_um, sine, cosine, phase_norm]).all():
        raise ValueError('模型输出超出有效数值范围。')
    defined = phase_norm > 1e-6
    return dict(order=order, name=name, n_raw_points=RAW_POINTS, opd_um=opd_um,
                distance_um=opd_um / 2, phase_sin=sine, phase_cos=cosine,
                phase_rad=phase.item() if defined else None,
                phase_defined=defined, phase_norm=phase_norm)


def position_rows(results):
    rows = []
    for offset in range(0, len(results), 10):
        group = results[offset:offset + 10]
        opd = np.array([row['opd_um'] for row in group])
        rows.append(dict(position_index=offset // 10 + 1, n_repeats=len(group),
                         opd_mean_um=float(opd.mean()), opd_std_um=float(opd.std()),
                         distance_um=float(opd.mean() / 2), first_order=group[0]['order'],
                         last_order=group[-1]['order']))
    return rows


@app.post('/api/cnn/upload')
def upload():
    requested = request.form.get('model', 'cnn')
    name = ALIASES.get(requested, requested)
    if name not in MODELS:
        return jsonify(error='请选择原始光谱 CNN 或双向 LSTM。'), 400
    files = sorted((file for file in request.files.getlist('files') if file.filename), key=natural_name)
    if not files:
        return jsonify(error='未接收到光谱文件。'), 400
    try:
        model, metadata = model_for(name)
    except (OSError, RuntimeError, ValueError):
        app.logger.exception('Model load failed for %s', name)
        return jsonify(error='模型权重暂不可用。'), 503

    results, failed, errors = [], [], []
    order = skipped = 0
    for file in files:
        filename = file.filename.replace('\\', '/')
        ext = Path(filename).suffix.lower()
        if ext not in EXTENSIONS:
            skipped += 1
            continue
        before = order
        try:
            for label, wavelength, intensity in read_upload(file, filename):
                order += 1
                try:
                    results.append(predict_spectrum(model, metadata, order, label, wavelength, intensity))
                except ValueError as error:
                    failed.append(dict(order=order, name=label, error=str(error)))
            if before == order:
                errors.append(dict(file=filename, error='文件中没有可读光谱。'))
        except (ValueError, OSError, TypeError, ArithmeticError) as error:
            errors.append(dict(file=filename, error=str(error)))
        except Exception:
            app.logger.exception('Could not parse uploaded spectrum')
            errors.append(dict(file=filename, error='无法解析文件，请检查数据格式。'))

    positions = position_rows(results)
    steps = np.diff([row['opd_mean_um'] for row in positions])
    response = jsonify(model=name, n_files=len(files), n_skipped=skipped, n_spectra=order,
                       n_success=len(results), n_failed=len(failed), results=results, positions=positions,
                       step_mean_um=float(steps.mean()) if steps.size else None,
                       step_std_um=float(steps.std()) if steps.size else None, errors=errors, failed=failed)
    if order == 0:
        return response, 400
    token = secrets.token_urlsafe(24)
    with recent_lock:
        recent[token] = (monotonic() + 300, name, results)
        if len(recent) > 8:
            recent.popitem(last=False)
    response.set_cookie(COOKIE_NAME, token, max_age=300, httponly=True, samesite='Lax',
                        secure=request.is_secure)
    return response


def latest():
    with recent_lock:
        token = request.cookies.get(COOKIE_NAME)
        item = recent.get(token)
        if item is not None and item[0] > monotonic():
            return item[1:]
        recent.pop(token, None)
    return None


@app.get('/api/cnn/result')
def result():
    item = latest()
    if item is None:
        return jsonify(error='当前会话没有解调结果。'), 404
    return jsonify(model=item[0], results=item[1])


@app.get('/api/cnn/export')
def export():
    item = latest()
    if item is None:
        return jsonify(error='当前会话没有解调结果。'), 404
    stream = io.StringIO()
    fields = ('order', 'name', 'opd_um', 'distance_um', 'phase_sin', 'phase_cos', 'phase_rad', 'phase_norm')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in item[1]:
        name = row['name']
        writer.writerow(dict(row, name=("'" + name) if re.match(r'^\s*[=+\-@]', name) else name))
    return Response('\ufeff' + stream.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=demodulation.csv'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')), debug=False)
