"""Data sintetis KECIL untuk uji asap (--smoke). Bukan data penelitian.

Meniru struktur file & metadata HAM10000 dan PAD-UFES-20 (nama kolom, beberapa
citra per lesi, satu pasang duplikat lintas lesi, satu duplikat lintas dataset)
supaya setiap tahap pipeline teruji tanpa internet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .utils import resolve

HAM_DX = ["mel", "nv", "bcc", "akiec", "bkl"]          # bkl harus dibuang oleh pipeline
PAD_DX = ["MEL", "NEV", "BCC", "ACK", "SCC"]           # SCC harus dibuang
COLORS = {0: (90, 40, 30), 1: (140, 90, 60), 2: (190, 120, 140), 3: (170, 60, 50), 4: (100, 100, 100)}


def _lesion(rng, cls, size, dermo: bool):
    w, h = size
    bg = (225, 180, 160) if dermo else tuple(int(v) for v in rng.integers(150, 230, 3))
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    base = min(w, h) * (0.3 if dermo else 0.15)
    rx, ry = int(base * rng.uniform(0.6, 1.3)), int(base * rng.uniform(0.6, 1.3))
    cx, cy = w // 2 + int(rng.integers(-15, 15)), h // 2 + int(rng.integers(-15, 15))
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=COLORS[cls])
    for _ in range(int(rng.integers(3, 9))):   # bintik acak supaya tiap citra unik bagi pHash
        x, y, s = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(3, 12))
        d.ellipse([x, y, x + s, y + s], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    if dermo:  # sudut vignette hitam khas dermatoskop
        d.rectangle([0, 0, w // 8, h // 8], fill=(0, 0, 0))
    a = np.asarray(im).astype(np.float32) + rng.normal(0, 8, (h, w, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def make(cfg: dict, n_lesions: int = 70, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    ham_dir, pad_dir = resolve(cfg["paths"]["ham_dir"]), resolve(cfg["paths"]["pad_dir"])
    if (ham_dir / "HAM10000_metadata.csv").exists():
        return
    (ham_dir / "HAM10000_images_part_1").mkdir(parents=True, exist_ok=True)
    (pad_dir / "images").mkdir(parents=True, exist_ok=True)
    rows, k = [], 0
    for les in range(n_lesions):
        cls = les % len(HAM_DX)
        for _ in range(int(rng.integers(1, 4))):
            iid = f"ISIC_{k:07d}"; k += 1
            _lesion(rng, cls, (120, 90), True).save(ham_dir / "HAM10000_images_part_1" / f"{iid}.jpg", quality=95)
            rows.append({"lesion_id": f"HAM_{les:07d}", "image_id": iid, "dx": HAM_DX[cls], "dx_type": "histo",
                         "age": 50, "sex": "male", "localization": "back", "dataset": "vidir_modern"})
    # duplikat lintas lesi (harus terdeteksi dan digabung jadi satu grup)
    src = rows[0]
    Image.open(ham_dir / "HAM10000_images_part_1" / f"{src['image_id']}.jpg").save(
        ham_dir / "HAM10000_images_part_1" / "ISIC_9999999.jpg", quality=95)
    rows.append({**src, "lesion_id": "HAM_9999999", "image_id": "ISIC_9999999"})
    pd.DataFrame(rows).to_csv(ham_dir / "HAM10000_metadata.csv", index=False)

    prow = []
    for les in range(40):
        cls = les % len(PAD_DX)
        for j in range(int(rng.integers(1, 3))):
            iid = f"PAT_{les}_{les + 100}_{j}"
            _lesion(rng, cls, (160, 120), False).save(pad_dir / "images" / f"{iid}.png")
            prow.append({"patient_id": f"PAT_{les}", "lesion_id": les + 100, "img_id": f"{iid}.png",
                         "diagnostic": PAD_DX[cls], "biopsed": True})
    # duplikat lintas dataset: citra HAM disalin ke PAD (harus dikeluarkan dari uji OOD)
    Image.open(ham_dir / "HAM10000_images_part_1" / f"{rows[1]['image_id']}.jpg").save(
        pad_dir / "images" / "PAT_999_999_0.png")
    prow.append({"patient_id": "PAT_999", "lesion_id": 999, "img_id": "PAT_999_999_0.png",
                 "diagnostic": "MEL", "biopsed": True})
    pd.DataFrame(prow).to_csv(pad_dir / "metadata.csv", index=False)
    print(f"[synthetic] {len(rows)} citra HAM, {len(prow)} citra PAD")
