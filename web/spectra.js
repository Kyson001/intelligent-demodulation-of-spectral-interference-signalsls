import { unzipSync } from 'fflate';
import readExcelFile from 'read-excel-file/browser';

export const POINTS = 10001;
const GRID_ERROR = '需要 10001 点，波长 1510–1610 nm（间隔 0.01 nm，按升序排列）。';
const wavelengthKeys = ['wavelength_nm', 'wavelength', 'wl', 'k_rad_per_m', 'k', 'x_wave'];
const intensityKeys = ['intensity', 'spectra', 'spectrum', 'x', 'i', 'data', 'y', 'signal', 'value'];

function numeric(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : NaN;
  if (typeof value !== 'string') return NaN;
  const text = value.trim().replace(/^"(.*)"$/, '$1').trim();
  return text === '' ? NaN : Number(text);
}

const isRow = (value) => Array.isArray(value) || ArrayBuffer.isView(value);

function spectrum(name, wavelengths, values) {
  if (!isRow(values) || values.length !== POINTS || (wavelengths != null && wavelengths.length !== POINTS)) {
    throw new Error(GRID_ERROR);
  }
  const intensity = new Float32Array(POINTS);
  for (let i = 0; i < POINTS; i++) {
    if (wavelengths != null) {
      const wavelength = numeric(wavelengths[i]);
      if (!Number.isFinite(wavelength) || Math.abs(wavelength - (1510 + 0.01 * i)) > 1e-6) {
        throw new Error(GRID_ERROR);
      }
    }
    intensity[i] = numeric(values[i]);
    if (!Number.isFinite(intensity[i])) {
      throw new Error(`第 ${i + 1} 个强度缺失或超出有效数值范围。`);
    }
  }
  return { name, intensity };
}

function waveRows(rows, name, columnsFromStart) {
  const first = rows.findIndex(row => row?.length && Number.isFinite(numeric(row[0])));
  if (first === -1) throw new Error('未找到波长和强度列。');
  const body = rows.slice(first).filter(row => row.some(v => v != null && String(v).trim()));
  const wavelengths = body.map(row => row[0]);
  const columns = columnsFromStart ? 2 : body.reduce((count, row) => Math.max(count, row.length), 0);
  if (columns < 2) throw new Error('至少需要波长和强度两列。');
  const inputs = [];
  const errors = [];
  for (let col = 1; col < columns; col++) {
    const values = body.map(row => row[col]);
    if (!columnsFromStart && values.every(v => v == null || String(v).trim() === '')) continue;
    const id = columnsFromStart ? name : `${name}::col${col}`;
    try {
      inputs.push(spectrum(id, wavelengths, values));
    } catch (error) {
      errors.push({ file: id, error: error.message });
    }
  }
  return { inputs, errors };
}

function arraySpectra(values, name, wavelengths = null) {
  if (!isRow(values) || !values.length) throw new Error('未找到光谱数组。');
  if (!isRow(values[0])) return [spectrum(name, wavelengths, values)];
  if (wavelengths == null && values.length > 2 && values[0].length === 2) {
    return [spectrum(name, values.map(row => row[0]), values.map(row => row[1]))];
  }
  if (wavelengths == null && values.length === 2 && values[0].length === POINTS &&
      Number(values[0][0]) === 1510 && Number(values[0].at(-1)) === 1610) {
    return [spectrum(name, values[0], values[1])];
  }
  return values.map((row, index) => spectrum(`${name}::${index + 1}`, wavelengths, row));
}

function readNpy(bytes) {
  if (bytes.length < 10 || bytes[0] !== 147 || new TextDecoder().decode(bytes.subarray(1, 6)) !== 'NUMPY') {
    throw new Error('NPY 文件头无效。');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = bytes[6];
  if (version < 1 || version > 3) throw new Error('不支持此 NPY 版本。');
  const start = version === 1 ? 10 : 12;
  const headerLength = version === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const end = start + headerLength;
  if (end > bytes.length) throw new Error('NPY 文件头不完整。');
  const header = new TextDecoder().decode(bytes.subarray(start, end));
  const dtype = header.match(/['"]descr['"]\s*:\s*['"]([<>|=])([fiub])(\d+)['"]/);
  const dimensions = header.match(/['"]shape['"]\s*:\s*\(([\d\s,]+)\)/);
  if (!dtype || !dimensions) throw new Error('只支持普通数值 NPY 数组。');
  const [, order, kind, width] = dtype;
  const readers = { f4: 'getFloat32', f8: 'getFloat64', i1: 'getInt8', i2: 'getInt16',
                    i4: 'getInt32', i8: 'getBigInt64', u1: 'getUint8', u2: 'getUint16',
                    u4: 'getUint32', u8: 'getBigUint64', b1: 'getUint8' };
  const reader = readers[kind + width];
  const shape = dimensions[1].split(',').map(x => x.trim()).filter(Boolean).map(Number);
  const count = shape.reduce((total, n) => total * n, 1);
  const size = Number(width);
  if (!reader || !shape.length || shape.length > 2 || count < 1 || count > 25000000 ||
      end + count * size !== bytes.length) {
    throw new Error('NPY 数值类型、数组维度或数据长度无效。');
  }
  const read = (index) => Number(view[reader](end + index * size, order !== '>'));
  if (shape.length === 1) return Float64Array.from({ length: count }, (_, i) => read(i));
  const [rows, cols] = shape;
  const byColumn = /['"]fortran_order['"]\s*:\s*True/.test(header);
  return Array.from({ length: rows }, (_, row) => Float64Array.from(
    { length: cols }, (_, col) => read(byColumn ? row + col * rows : row * cols + col),
  ));
}

function structured(keys, get, name) {
  const byLowerCase = new Map(keys.map(key => [key.toLowerCase(), key]));
  const wavelengthKey = wavelengthKeys.find(key => byLowerCase.has(key));
  const signalKeys = intensityKeys.filter(key => byLowerCase.has(key));
  if (!signalKeys.length) {
    if (keys.length === 1) return arraySpectra(get(keys[0]), name);
    throw new Error('结构化文件需提供 intensity、spectra 或 X 等光谱字段。');
  }
  let wavelengths = wavelengthKey ? get(byLowerCase.get(wavelengthKey)) : null;
  if (wavelengthKey === 'k' || wavelengthKey === 'k_rad_per_m') {
    wavelengths = Array.from(wavelengths, k => (2 * Math.PI * 1e9) / k);
  }
  return signalKeys.flatMap(key => arraySpectra(get(byLowerCase.get(key)), `${name}::${key}`, wavelengths));
}

export async function readSpectra(file, name) {
  const extension = file.name.split('.').at(-1).toLowerCase();
  if (extension === 'xlsx' || extension === 'xlsm') {
    const sheets = await readExcelFile(file);
    const inputs = [];
    const errors = [];
    for (const { sheet, data } of sheets) {
      if (!data.length) continue;
      try {
        const output = waveRows(data, `${name}::${sheet}`, false);
        inputs.push(...output.inputs);
        errors.push(...output.errors);
      } catch (error) {
        errors.push({ file: `${name}::${sheet}`, error: error.message });
      }
    }
    if (!inputs.length && !errors.length) throw new Error('工作簿没有光谱记录。');
    return { inputs, errors };
  }
  let inputs;
  if (extension === 'npy') {
    inputs = arraySpectra(readNpy(new Uint8Array(await file.arrayBuffer())), name);
  } else if (extension === 'npz') {
    let total = 0;
    const entries = unzipSync(new Uint8Array(await file.arrayBuffer()), { filter: ({ name, originalSize }) => {
      if (!name.endsWith('.npy')) return false;
      total += originalSize;
      if (total > 200 * 1024 * 1024) throw new Error('NPZ 解压后超过 200 MB。');
      return true;
    } });
    inputs = structured(Object.keys(entries).map(key => key.slice(0, -4)),
                        key => readNpy(entries[`${key}.npy`]), name);
  } else if (extension === 'json') {
    const data = JSON.parse((await file.text()).replace(/^\uFEFF/, ''));
    inputs = Array.isArray(data) ? arraySpectra(data, name)
      : structured(Object.keys(data ?? {}), key => data[key], name);
  } else {
    const lines = (await file.text()).trim().split(/\r?\n/).filter(line => line.trim());
    const delimiter = lines.find(line => /^[\s"]*[+\-]?[\d.]/.test(line))?.match(/[,;\t]/)?.[0] ?? /\s+/;
    return waveRows(lines.map(line => line.trim().split(delimiter)), name, true);
  }
  return { inputs, errors: [] };
}
