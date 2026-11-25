// Rendering peta Grad-CAM di atas citra (canvas). Colormap mirip "inferno":
// sekuensial gelap -> terang, sehingga tetap terbaca oleh pembaca buta warna.

const STOPS: [number, number, number, number][] = [
  [0.0, 0, 0, 4],
  [0.25, 87, 16, 110],
  [0.5, 188, 55, 84],
  [0.75, 249, 142, 9],
  [1.0, 252, 255, 164],
];

export function inferno(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    if (v <= STOPS[i][0]) {
      const [t0, r0, g0, b0] = STOPS[i - 1];
      const [t1, r1, g1, b1] = STOPS[i];
      const f = (v - t0) / (t1 - t0);
      return [r0 + f * (r1 - r0), g0 + f * (g1 - g0), b0 + f * (b1 - b0)];
    }
  }
  return [252, 255, 164];
}

export function decodeB64Map(b64: string): { data: Float32Array; size: number } {
  const bin = atob(b64);
  const out = new Float32Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i) / 255;
  return { data: out, size: Math.round(Math.sqrt(bin.length)) };
}

/** Normalisasi CAM: ReLU lalu bagi maksimum (sama dengan src/gradcam.py). */
export function normalizeCam(raw: Float32Array): Float32Array {
  let mx = 0;
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    out[i] = Math.max(0, raw[i]);
    if (out[i] > mx) mx = out[i];
  }
  if (mx > 0) for (let i = 0; i < out.length; i++) out[i] /= mx;
  return out;
}

/** Gambar citra + overlay CAM (diperhalus bilinear oleh canvas) ke canvas target. */
export function drawCamOverlay(
  canvas: HTMLCanvasElement,
  img: CanvasImageSource | null,
  cam: Float32Array | null,
  size: number,
  alpha: number,
) {
  const ctx = canvas.getContext("2d")!;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (img) ctx.drawImage(img, 0, 0, W, H);
  if (!cam || alpha <= 0) return;
  const small = document.createElement("canvas");
  small.width = size;
  small.height = size;
  const sctx = small.getContext("2d")!;
  const id = sctx.createImageData(size, size);
  for (let i = 0; i < cam.length; i++) {
    const [r, g, b] = inferno(cam[i]);
    id.data[i * 4] = r;
    id.data[i * 4 + 1] = g;
    id.data[i * 4 + 2] = b;
    id.data[i * 4 + 3] = 255 * (0.25 + 0.75 * cam[i]);
  }
  sctx.putImageData(id, 0, 0);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(small, 0, 0, W, H);
  ctx.restore();
}

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((res, rej) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = src;
  });
}
