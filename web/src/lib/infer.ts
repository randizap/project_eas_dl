// Inferensi di browser dengan ONNX Runtime Web (WebAssembly). Tidak ada citra yang dikirim ke server.
import type { InferenceSession } from "onnxruntime-web";

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
export const SIZE = 224;

type Ort = typeof import("onnxruntime-web/wasm");
let ortPromise: Promise<Ort> | null = null;
function getOrt(): Promise<Ort> {
  if (!ortPromise) {
    ortPromise = import("onnxruntime-web/wasm").then((ort) => {
      // berkas .wasm di-bundle Vite sebagai aset (dist/assets), tidak perlu CDN
      // tanpa header cross-origin isolation, multi-thread tidak tersedia
      ort.env.wasm.numThreads = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      return ort;
    });
  }
  return ortPromise;
}

const sessions = new Map<string, Promise<InferenceSession>>();

/** Unduh model (dengan progres) lalu buat sesi. Hasil di-cache per URL. */
export function loadModel(url: string, onProgress?: (f: number) => void): Promise<InferenceSession> {
  if (!sessions.has(url)) {
    sessions.set(url, (async () => {
      const ort = await getOrt();
      const res = await fetch(url);
      if (!res.ok || !res.body) throw new Error(`Gagal mengunduh model (${res.status})`);
      const total = Number(res.headers.get("content-length")) || 0;
      const reader = res.body.getReader();
      const chunks: Uint8Array[] = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.length;
        if (total) onProgress?.(got / total);
      }
      const buf = new Uint8Array(got);
      let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.length; }
      onProgress?.(1);
      return ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    })().catch((e) => { sessions.delete(url); throw e; }));
  }
  return sessions.get(url)!;
}

/** Canvas 224x224 -> tensor NCHW ternormalisasi ImageNet (sama dengan eval_transform). */
export async function toTensor(canvas: HTMLCanvasElement) {
  const ort = await getOrt();
  const { data } = canvas.getContext("2d")!.getImageData(0, 0, SIZE, SIZE);
  const n = SIZE * SIZE;
  const f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) f[c * n + i] = (data[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
  return new ort.Tensor("float32", f, [1, 3, SIZE, SIZE]);
}

export interface Prediction {
  probs: number[];
  cams: Float32Array[]; // satu per kelas, 7x7 mentah (sebelum ReLU)
  camSize: number;
  ms: number;
}

export async function predict(session: InferenceSession, canvas: HTMLCanvasElement): Promise<Prediction> {
  const t0 = performance.now();
  const x = await toTensor(canvas);
  const out = await session.run({ input: x });
  const probs = Array.from(out.probs.data as Float32Array);
  const cam = out.cam.data as Float32Array;
  const [, k, h, w] = out.cam.dims as number[];
  const cams = Array.from({ length: k }, (_, i) => cam.slice(i * h * w, (i + 1) * h * w));
  return { probs, cams, camSize: h, ms: performance.now() - t0 };
}
