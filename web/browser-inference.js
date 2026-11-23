import { env, InferenceSession, Tensor } from 'onnxruntime-web/wasm';
import { POINTS, readSpectra } from './spectra.js';

env.wasm.numThreads = 1;
env.wasm.wasmPaths = {
  mjs: new URL('./runtime/ort-wasm.js', import.meta.url),
  wasm: new URL('./runtime/ort-wasm.wasm', import.meta.url),
};
const sessions = new Map();

function getSession(model) {
  if (!sessions.has(model)) {
    const url = new URL(`./models/${model}.onnx`, import.meta.url);
    sessions.set(model, InferenceSession.create(url.href, { executionProviders: ['wasm'] }).catch(error => {
      sessions.delete(model);
      throw error;
    }));
  }
  return sessions.get(model);
}

export async function runBrowser(model, files) {
  if (model !== 'cnn' && model !== 'bilstm') throw new Error('请选择有效模型。');
  const results = [];
  const errors = [];
  let order = 0;
  let session;
  for (const file of files) {
    const name = file.webkitRelativePath || file.name;
    let spectra;
    try {
      spectra = await readSpectra(file, name);
      errors.push(...spectra.errors);
    } catch (error) {
      errors.push({ file: name, error: error.message });
      continue;
    }
    for (const input of spectra.inputs) {
      session ??= await getSession(model);
      order++;
      try {
        const values = await session.run({ intensity: new Tensor('float32', input.intensity, [1, POINTS]) });
        const [opd, sine, cosine] = values.prediction.data;
        if (![opd, sine, cosine].every(Number.isFinite)) throw new Error('模型输出超出有效数值范围。');
        const norm = Math.hypot(sine, cosine);
        results.push({ order, name: input.name, opd_um: opd * 1e6, phase_sin: sine, phase_cos: cosine,
                       phase_rad: norm > 1e-6 ? Math.atan2(sine, cosine) : null, phase_norm: norm });
      } catch (error) {
        errors.push({ file: input.name, error: error.message });
      }
    }
  }
  return { results, errors };
}
