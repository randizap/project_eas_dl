"""Korupsi sintetis bergaya ImageNet-C (Hendrycks & Dietterich, ICLR 2019).

Parameter severity 1-5 mengikuti kode ImageNet-C; implementasi ditulis ulang
dengan numpy/scipy/PIL (tanpa Wand/OpenCV), jadi TIDAK identik bit-per-bit dan
angkanya tidak boleh dibandingkan langsung dengan tabel ImageNet-C.
motion_blur adalah aproksimasi (kernel garis berbobot Gaussian).

OVERLAP menandai korupsi yang mirip augmentasi intervensi (blur, JPEG,
kecerahan/kontras). Ketahanan intervensi terhadap korupsi ini sebagian besar
"dilatih langsung", jadi dilaporkan terpisah dari korupsi non-overlap.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage

OVERLAP = {"defocus_blur", "motion_blur", "brightness", "contrast", "jpeg_compression"}


def _f(x):
    return np.asarray(x, dtype=np.float32) / 255.0


def _u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def gaussian_noise(x, s, rng):
    c = [.08, .12, .18, .26, .38][s - 1]
    x = _f(x)
    return _u8(x + rng.normal(scale=c, size=x.shape))


def shot_noise(x, s, rng):
    c = [60, 25, 12, 5, 3][s - 1]
    return _u8(rng.poisson(_f(x) * c) / c)


def impulse_noise(x, s, rng):
    c = [.03, .06, .09, .17, .27][s - 1]
    x = _f(x).copy()
    m = rng.random(x.shape)
    x[m < c / 2] = 0.0
    x[(m >= c / 2) & (m < c)] = 1.0
    return _u8(x)


def _disk(radius, alias_blur):
    L = np.arange(-8, 9) if radius <= 8 else np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    k = ((X ** 2 + Y ** 2) <= radius ** 2).astype(np.float32)
    k /= k.sum()
    return ndimage.gaussian_filter(k, alias_blur)


def _conv(x, k):
    x = _f(x)
    return _u8(np.stack([ndimage.convolve(x[..., c], k, mode="reflect") for c in range(3)], -1))


def defocus_blur(x, s, rng):
    r, a = [(3, .1), (4, .5), (6, .5), (8, .5), (10, .5)][s - 1]
    return _conv(x, _disk(r, a))


def motion_blur(x, s, rng):
    radius, sigma = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][s - 1]
    ang = np.deg2rad(rng.uniform(-45, 45))
    size = 2 * radius + 1
    k = np.zeros((size, size), np.float32)
    for t in range(radius + 1):                      # satu arah, seperti motion blur ImageMagick
        w = np.exp(-(t ** 2) / (2 * sigma ** 2))
        xi = int(round(radius + t * np.cos(ang)))
        yi = int(round(radius + t * np.sin(ang)))
        k[yi, xi] += w
    k /= k.sum()
    return _conv(x, k)


def _rgb2hsv_v_shift(x, c):
    from skimage.color import hsv2rgb, rgb2hsv
    h = rgb2hsv(_f(x))
    h[..., 2] = np.clip(h[..., 2] + c, 0, 1)
    return _u8(hsv2rgb(h))


def brightness(x, s, rng):
    return _rgb2hsv_v_shift(x, [.1, .2, .3, .4, .5][s - 1])


def contrast(x, s, rng):
    c = [.4, .3, .2, .1, .05][s - 1]
    x = _f(x)
    m = x.mean(axis=(0, 1), keepdims=True)
    return _u8((x - m) * c + m)


def jpeg_compression(x, s, rng):
    q = [25, 18, 15, 10, 7][s - 1]
    buf = io.BytesIO()
    Image.fromarray(np.asarray(x)).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"))


def pixelate(x, s, rng):
    c = [.6, .5, .4, .3, .25][s - 1]
    im = Image.fromarray(np.asarray(x))
    w, h = im.size
    im = im.resize((max(1, int(w * c)), max(1, int(h * c))), Image.BOX).resize((w, h), Image.BOX)
    return np.asarray(im)


CORRUPTIONS = {f.__name__: f for f in [gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                                       motion_blur, brightness, contrast, jpeg_compression, pixelate]}


def apply(name: str, img_u8: np.ndarray, severity: int, seed: int) -> np.ndarray:
    if name == "clean":   # acuan tanpa korupsi lewat jalur praproses yang sama persis
        return np.asarray(img_u8)
    rng = np.random.default_rng(seed)
    out = CORRUPTIONS[name](img_u8, severity, rng)
    assert out.shape == img_u8.shape and out.dtype == np.uint8
    return out
