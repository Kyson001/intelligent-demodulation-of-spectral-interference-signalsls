const element = (id) => document.getElementById(id);
const form = element('demodulate-form');
const selector = element('cnn-model');
const sources = [element('cnn-files'), element('cnn-dir')];
const runButton = element('run-model');
const status = element('cnn-status');
const usingServer = document.documentElement.dataset.runtime === 'flask';
const models = { cnn: '原始光谱 CNN', bilstm: '原始光谱双向 LSTM' };
const fileExtension = /\.(csv|txt|dat|tsv|prn|xlsx|xlsm|npy|npz|json)$/i;
const collator = new Intl.Collator('zh-CN', { numeric: true });
const maxSize = 200 * 1024 * 1024;
const sharecode = new URL(location.href).searchParams.get('sharecode');
let results = [];
let resultModel = '';

element('processing-location').textContent = usingServer
  ? '所选文件由当前服务解调。'
  : '此页面在浏览器内完成解调，光谱文件留在本机。';

function fileName(file) {
  return file.webkitRelativePath || file.name;
}

function selectedFiles() {
  const files = new Map();
  for (const source of sources) {
    for (const file of source.files) {
      if (fileExtension.test(file.name)) {
        files.set(JSON.stringify([fileName(file), file.size, file.lastModified]), file);
      }
    }
  }
  return [...files.values()].sort((a, b) => collator.compare(fileName(a), fileName(b)));
}

for (const source of sources) {
  source.addEventListener('change', () => {
    element('cnn-file-count').textContent = `已选择 ${selectedFiles().length} 个支持的文件。`;
  });
}

function displayErrors(errors) {
  const list = element('input-errors');
  list.replaceChildren(...errors.map(({ file, error }) => {
    const item = document.createElement('li');
    item.textContent = `${file}: ${error}`;
    return item;
  }));
  list.hidden = errors.length === 0;
}

function addRow(table, values) {
  const row = table.insertRow();
  for (const value of values) {
    row.insertCell().textContent = value;
  }
}

function showResults(model, values) {
  results = values;
  resultModel = model;
  element('results-panel').hidden = values.length === 0;
  const table = document.querySelector('#cnn-table tbody');
  table.replaceChildren();
  element('cnn-summary').textContent = `${models[model]}，成功解调 ${values.length} 条光谱。距离为光程差的一半。`;
  for (const row of values) {
    addRow(table, [row.order, row.name, row.opd_um.toFixed(4), (row.opd_um / 2).toFixed(4)]);
  }
}

async function runServer(model, files) {
  const url = new URL('./api/cnn/upload', location.href);
  if (sharecode) url.searchParams.set('sharecode', sharecode);
  const payload = new FormData();
  payload.set('model', model);
  for (const file of files) payload.append('files', file, fileName(file));
  const response = await fetch(url, { method: 'POST', body: payload });
  const data = await response.json();
  if (!response.ok && !data.errors?.length) {
    throw new Error(data.error || '解调服务请求失败');
  }
  return {
    results: data.results ?? [],
    errors: [...(data.errors ?? []), ...(data.failed ?? []).map(row => ({ file: row.name, error: row.error }))],
  };
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const model = selector.value;
  const files = selectedFiles();
  displayErrors([]);
  if (!files.length) {
    status.textContent = '请选择至少一个支持的光谱文件。';
    return;
  }
  if (files.reduce((sum, file) => sum + file.size, 0) > maxSize) {
    status.textContent = '所选文件总大小超过 200 MB，请分批解调。';
    return;
  }
  runButton.disabled = true;
  selector.disabled = true;
  for (const source of sources) source.disabled = true;
  showResults(model, []);
  status.textContent = `正在加载 ${models[model]} 并解调…`;
  try {
    const response = usingServer
      ? await runServer(model, files)
      : await (await import('./browser-inference.js')).runBrowser(model, files);
    showResults(model, response.results);
    displayErrors(response.errors);
    status.textContent = `成功 ${response.results.length} 条，发现 ${response.errors.length} 项输入错误。`;
  } catch (error) {
    status.textContent = `解调失败。 ${error.message}`;
  } finally {
    runButton.disabled = false;
    selector.disabled = false;
    for (const source of sources) source.disabled = false;
  }
});

function quoteCsv(value) {
  const safe = /^[\s]*[=+\-@]/.test(value) ? `'${value}` : value;
  return `"${safe.replaceAll('"', '""')}"`;
}

element('export-results').addEventListener('click', () => {
  if (!results.length) return;
  const rows = ['order,name,opd_um,opd_half_um,phase_sin,phase_cos,phase_rad,phase_norm'];
  for (const row of results) {
    rows.push([row.order, quoteCsv(row.name), row.opd_um, row.opd_um / 2,
               row.phase_sin, row.phase_cos, row.phase_rad ?? '', row.phase_norm].join(','));
  }
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob(['\uFEFF', rows.join('\r\n') + '\r\n'], { type: 'text/csv;charset=utf-8' }));
  link.download = `${resultModel}_demodulation.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
