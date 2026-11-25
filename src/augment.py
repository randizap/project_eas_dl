"""Transformasi: evaluasi, augmentasi standar, dan augmentasi karakteristik ponsel (intervensi)."""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms as T

from .models import IMAGENET_MEAN, IMAGENET_STD


def eval_transform(img_size: int = 224):
    # Cache sudah bersisi pendek 256; center crop 224 (rasio 0,875 standar ImageNet).
    return T.Compose([T.Resize(int(round(img_size / 0.875))), T.CenterCrop(img_size),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class RandomRot90:
    def __call__(self, im: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return im if k == 0 else im.rotate(90 * k, expand=True)


class ZoomOut:
    """Perkecil citra lalu tempel di kanvas berwarna kulit: meniru foto dari jarak lebih jauh.

    Warna kanvas = median piksel pita tepi citra, piksel sangat gelap dibuang
    (sudut vignette dermatoskop tidak boleh ikut menjadi warna kulit). Ini
    pendekatan kasar; tekstur kulit asli tidak direproduksi.
    """

    def __init__(self, p: float, scale: tuple[float, float]):
        self.p, self.scale = p, scale

    @staticmethod
    def _skin_color(a: np.ndarray) -> np.ndarray:
        h, w, _ = a.shape
        b = max(1, int(0.1 * min(h, w)))
        band = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                               a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
        bright = band[band.sum(1) > 90]
        src = bright if len(bright) > 50 else a.reshape(-1, 3)
        return np.median(src, axis=0)

    def __call__(self, im: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return im
        a = np.asarray(im)
        h, w, _ = a.shape
        s = random.uniform(*self.scale)
        nw, nh = max(8, int(w * s)), max(8, int(h * s))
        small = im.resize((nw, nh), Image.BICUBIC)
        col = self._skin_color(a)
        canvas = np.empty_like(a, dtype=np.float32)
        canvas[:] = col
        canvas += np.random.normal(0, 4.0, canvas.shape)  # sedikit tekstur agar tidak rata sempurna
        canvas = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
        x0, y0 = random.randint(0, w - nw), random.randint(0, h - nh)
        canvas.paste(small, (x0, y0))
        return canvas


class WhiteBalance:
    def __init__(self, gain: tuple[float, float]):
        self.gain = gain

    def __call__(self, im: Image.Image) -> Image.Image:
        g = np.random.uniform(*self.gain, size=3).astype(np.float32)
        a = np.asarray(im).astype(np.float32) * g
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


class RandomGaussianBlur:
    def __init__(self, p: float, sigma: tuple[float, float]):
        self.p, self.sigma = p, sigma

    def __call__(self, im: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return im
        return im.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma)))


class RandomJPEG:
    def __init__(self, p: float, quality: tuple[int, int]):
        self.p, self.quality = p, quality

    def __call__(self, im: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return im
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        with Image.open(buf) as out:
            return out.convert("RGB")


def train_transform(cfg: dict, phone: bool):
    a, size = cfg["augment"], cfg["data"]["img_size"]
    st = a["standard"]
    pre, post = [], []
    if phone:
        ph = a["phone"]
        pre.append(ZoomOut(ph["zoom_out_p"], tuple(ph["zoom_out_scale"])))
        cj = ph["color_jitter"]
        post += [T.ColorJitter(*cj), WhiteBalance(tuple(ph["wb_gain"])),
                 RandomGaussianBlur(ph["blur_p"], tuple(ph["blur_sigma"])),
                 RandomJPEG(ph["jpeg_p"], tuple(ph["jpeg_quality"]))]
    else:
        post.append(T.ColorJitter(*st["color_jitter"]))
    geo = [T.RandomResizedCrop(size, scale=tuple(st["rrc_scale"])),
           T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    if st.get("rot90", True):
        geo.append(RandomRot90())
    return T.Compose(pre + geo + post + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
