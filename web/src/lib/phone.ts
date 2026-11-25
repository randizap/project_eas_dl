// Simulasi karakteristik foto ponsel di browser. Urutan dan rentang mengikuti
// src/augment.py (intervensi): zoom-out -> color jitter -> white balance -> blur -> JPEG.
// Semua operasi piksel ditulis manual (tanpa ctx.filter) supaya hasilnya sama di semua browser.
import { SIZE } from "./infer";

export interface PhoneParams {
  zoom: number;        // 1 = tanpa zoom-out; 0,4 = lesi tampak 40% ukuran
  brightness: number;  // faktor, 1 = asli
  contrast: number;    // faktor
  saturation: number;  // faktor
  warmth: number;      // -0,15..0,15 -> gain R = 1+w, B = 1-w (white balance)
  blur: number;        // sigma piksel, 0 = mati
  jpeg: number;        // kualitas 1-100, 100 = mati
}

export const DEFAULT_PARAMS: PhoneParams = { zoom: 1, brightness: 1, contrast: 1, saturation: 1, warmth: 0, blur: 0, jpeg: 100 };

/** Sampel acak dari distribusi augmentasi ponsel di configs/default.yaml. */
export function randomParams(cfg: any): PhoneParams {
  const ph = cfg?.augment?.phone ?? {};
  const u = (a: number, b: number) => a + Math.random() * (b - a);
  const [bj, cj, sj] = ph.color_jitter ?? [0.3, 0.3, 0.3];
  const [g0, g1] = ph.wb_gain ?? [0.85, 1.15];
  return {
    zoom: Math.random() < (ph.zoom_out_p ?? 0.5) ? +u(...((ph.zoom_out_scale ?? [0.4, 0.9]) as [number, number])).toFixed(2) : 1,
    brightness: +u(1 - bj, 1 + bj).toFixed(2),
    contrast: +u(1 - cj, 1 + cj).toFixed(2),
    saturation: +u(1 - sj, 1 + sj).toFixed(2),
    warmth: +u(g0 - 1, g1 - 1).toFixed(2),
    blur: Math.random() < (ph.blur_p ?? 0.3) ? +u(...((ph.blur_sigma ?? [0.1, 2]) as [number, number])).toFixed(1) : 0,
    jpeg: Math.random() < (ph.jpeg_p ?? 0.5) ? Math.round(u(...((ph.jpeg_quality ?? [30, 90]) as [number, number]))) : 100,
  };
}

/** Resize sisi pendek ke 256 lalu center crop 224 (sama dengan eval_transform). */
export function centerCrop(src: CanvasImageSource & { width: number; height: number }): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = SIZE;
  const ctx = c.getContext("2d")!;
  const sw = (src as HTMLImageElement).naturalWidth || src.width;
  const sh = (src as HTMLImageElement).naturalHeight || src.height;
  const r = Math.round(SIZE / 0.875) / Math.min(sw, sh);
  const w = sw * r, h = sh * r;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, (SIZE - w) / 2, (SIZE - h) / 2, w, h);
  return c;
}

function skinColor(d: Uint8ClampedArray): [number, number, number] {
  // median piksel pita tepi 10%, buang piksel sangat gelap (vignette) - seperti ZoomOut di augment.py
  const b = Math.max(1, Math.floor(0.1 * SIZE));
  const rs: number[] = [], gs: number[] = [], bs: number[] = [];
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    if (x >= b && x < SIZE - b && y >= b && y < SIZE - b) continue;
    const i = (y * SIZE + x) * 4;
    if (d[i] + d[i + 1] + d[i + 2] > 90) { rs.push(d[i]); gs.push(d[i + 1]); bs.push(d[i + 2]); }
  }
  const med = (a: number[]) => (a.length ? a.sort((p, q) => p - q)[a.length >> 1] : 180);
  return [med(rs), med(gs), med(bs)];
}

function gaussianBlur(d: Uint8ClampedArray, sigma: number) {
  const rad = Math.max(1, Math.ceil(sigma * 3));
  const k: number[] = [];
  let s = 0;
  for (let i = -rad; i <= rad; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k.push(v); s += v; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const tmp = new Float32Array(d.length);
  const pass = (src: ArrayLike<number>, dst: Float32Array | Uint8ClampedArray, horiz: boolean) => {
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) for (let c = 0; c < 3; c++) {
      let acc = 0;
      for (let j = -rad; j <= rad; j++) {
        const xx = horiz ? Math.min(SIZE - 1, Math.max(0, x + j)) : x;
        const yy = horiz ? y : Math.min(SIZE - 1, Math.max(0, y + j));
        acc += src[(yy * SIZE + xx) * 4 + c] * k[j + rad];
      }
      dst[(y * SIZE + x) * 4 + c] = acc;
    }
  };
  pass(d, tmp, true);
  pass(tmp, d, false);
}

function loadUrl(url: string): Promise<HTMLImageElement> {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = url; });
}

/** Terapkan simulasi pada canvas 224x224, kembalikan canvas baru. */
export async function simulate(base: HTMLCanvasElement, p: PhoneParams): Promise<HTMLCanvasElement> {
  const out = document.createElement("canvas");
  out.width = out.height = SIZE;
  const ctx = out.getContext("2d", { willReadFrequently: true })!;
  const src = base.getContext("2d")!.getImageData(0, 0, SIZE, SIZE);

  // 1) zoom-out: kecilkan citra di atas kanvas berwarna kulit (+ sedikit tekstur)
  if (p.zoom < 0.999) {
    const [r, g, b] = skinColor(src.data);
    const bg = ctx.createImageData(SIZE, SIZE);
    for (let i = 0; i < bg.data.length; i += 4) {
      const n = () => (Math.random() + Math.random() + Math.random() - 1.5) * 6.9; // ~N(0,4)
      bg.data[i] = r + n(); bg.data[i + 1] = g + n(); bg.data[i + 2] = b + n(); bg.data[i + 3] = 255;
    }
    ctx.putImageData(bg, 0, 0);
    const s = SIZE * p.zoom;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(base, (SIZE - s) / 2, (SIZE - s) / 2, s, s);
  } else {
    ctx.drawImage(base, 0, 0);
  }

  // 2) kecerahan, kontras, saturasi (urutan seperti torchvision ColorJitter)  3) white balance
  const img = ctx.getImageData(0, 0, SIZE, SIZE);
  const d = img.data;
  let mean = 0;
  for (let i = 0; i < d.length; i += 4) mean += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  mean = (mean / (d.length / 4)) * p.brightness;
  const gr = 1 + p.warmth, gb = 1 - p.warmth;
  for (let i = 0; i < d.length; i += 4) {
    let r = d[i] * p.brightness, g = d[i + 1] * p.brightness, b = d[i + 2] * p.brightness;
    r = (r - mean) * p.contrast + mean; g = (g - mean) * p.contrast + mean; b = (b - mean) * p.contrast + mean;
    const l = 0.299 * r + 0.587 * g + 0.114 * b;
    r = (r - l) * p.saturation + l; g = (g - l) * p.saturation + l; b = (b - l) * p.saturation + l;
    d[i] = r * gr; d[i + 1] = g; d[i + 2] = b * gb;
  }
  // 4) blur gaussian
  if (p.blur > 0.05) gaussianBlur(d, p.blur);
  ctx.putImageData(img, 0, 0);

  // 5) kompresi JPEG sungguhan lewat encoder browser
  if (p.jpeg < 100) {
    const im = await loadUrl(out.toDataURL("image/jpeg", p.jpeg / 100));
    ctx.drawImage(im, 0, 0);
  }
  return out;
}
