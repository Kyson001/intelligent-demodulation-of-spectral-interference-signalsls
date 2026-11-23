import { copyFile, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { build } from 'esbuild';

const root = path.dirname(fileURLToPath(import.meta.url));
const out = path.join(root, 'dist');
await mkdir(path.join(out, 'models'), { recursive: true });
await mkdir(path.join(out, 'runtime'), { recursive: true });
for (const file of ['index.html', 'source.html', 'app.js', 'style.css']) {
  await copyFile(path.join(root, file), path.join(out, file));
}
for (const file of ['cnn.onnx', 'bilstm.onnx', 'manifest.json']) {
  await copyFile(path.join(root, 'models', file), path.join(out, 'models', file));
}
for (const [source, target] of [
  ['ort-wasm-simd-threaded.mjs', 'ort-wasm.js'],
  ['ort-wasm-simd-threaded.wasm', 'ort-wasm.wasm'],
]) {
  await copyFile(path.join(root, 'node_modules', 'onnxruntime-web', 'dist', source),
                 path.join(out, 'runtime', target));
}
await build({
  entryPoints: [path.join(root, 'browser-inference.js')],
  outfile: path.join(out, 'browser-inference.js'),
  bundle: true,
  minify: true,
  format: 'esm',
  platform: 'browser',
  target: 'es2022',
});
await writeFile(path.join(out, '.nojekyll'), '');
