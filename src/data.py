"""Membaca metadata HAM10000 & PAD-UFES-20, menyatukan label, dan cache citra ber-resize.

Keluaran utama: outputs/index.csv, satu baris per citra yang dipakai (4 kelas).
"""
from __future__ import annotations

# Bisa dijalankan sebagai file (python src/xxx.py, tombol Run VS Code) maupun modul
# (python -m src.xxx). Saat sebagai file, folder proyek ditambahkan ke sys.path.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "src"
    import src  # noqa: F401

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .utils import ROOT, load_config, resolve, work_dir

IMG_EXT = {".jpg", ".jpeg", ".png"}


def _find_images(root: Path) -> dict[str, Path]:
    """Petakan nama file tanpa ekstensi -> path, mencari rekursif (struktur folder bebas)."""
    out: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            stem = p.stem
            if stem in out and out[stem] != p:
                # Salinan ganda (mis. folder HAM10000_images_part_1 dan ham10000_images_part_1
                # di mirror Kaggle). Ambil yang pertama; isinya identik.
                continue
            out[stem] = p
    return out


SKIP_ZIP = ("isic2018_task3_test", "segmentation")   # tidak dipakai; hemat ruang disk


def extract_archives(root: Path) -> None:
    """Ekstrak semua .zip di root (termasuk zip di dalam zip) ke folder bernama sama.

    Penanda '.extracted' mencegah ekstraksi ulang. Zip test ISIC2018 dan masker
    segmentasi dilewati. Jalur di dalam zip divalidasi (cegah zip-slip).
    """
    import zipfile
    while True:
        todo = [z for z in root.rglob("*.zip")
                if not any(k in z.name.lower() for k in SKIP_ZIP)
                and not (z.parent / (z.stem + ".extracted")).exists()]
        if not todo:
            return
        for z in todo:
            dest = z.parent / z.stem
            print(f"[data] mengekstrak {z.name} -> {dest}")
            with zipfile.ZipFile(z) as zf:
                base = dest.resolve()
                for m in zf.infolist():
                    target = (dest / m.filename).resolve()
                    if base != target and base not in target.parents:
                        raise ValueError(f"Jalur mencurigakan di {z.name}: {m.filename}")
                zf.extractall(dest)
            (z.parent / (z.stem + ".extracted")).write_text("ok", encoding="utf-8")


def _find_meta(root: Path, stems: list[str]) -> Path:
    """Cari file metadata: .csv, .tab (Harvard Dataverse), atau tanpa ekstensi."""
    for stem in stems:
        for pat in (f"{stem}.csv", f"{stem}.tab", stem):
            hits = [h for h in root.rglob(pat) if h.is_file() and not h.name.startswith("._")]
            if hits:
                return hits[0]
    raise FileNotFoundError(
        f"Metadata {stems} tidak ditemukan di {root}.\n"
        "  Unduh dataset dan letakkan sesuai README bagian 2 (zip boleh tidak diekstrak).")


def _read_meta(path: Path) -> pd.DataFrame:
    # sep=None: deteksi otomatis koma (Kaggle/Mendeley) atau tab (Dataverse .tab)
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def _require_dir(root: Path, name: str) -> None:
    if not root.exists() or not any(root.iterdir()):
        raise FileNotFoundError(
            f"Folder data {name} kosong/tidak ada: {root}\n"
            "  Unduh dataset dan taruh di folder itu (README bagian 2). "
            "Untuk sekadar menguji instalasi tanpa data: python -m src.run_all --smoke")


def load_ham(cfg: dict) -> pd.DataFrame:
    root = resolve(cfg["paths"]["ham_dir"])
    _require_dir(root, "HAM10000")
    extract_archives(root)
    meta = _read_meta(_find_meta(root, ["HAM10000_metadata"]))
    need = {"lesion_id", "image_id", "dx"}
    if not need.issubset(meta.columns):
        raise ValueError(f"Kolom metadata HAM kurang: {need - set(meta.columns)}")
    n_all = len(meta)
    mp = cfg["data"]["ham_map"]
    meta = meta[meta["dx"].isin(mp)].copy()
    imgs = _find_images(root)
    meta["src_path"] = meta["image_id"].map(lambda s: str(imgs.get(s, "")))
    missing = (meta["src_path"] == "").sum()
    if missing:
        raise FileNotFoundError(f"{missing} citra HAM tercantum di metadata tetapi tidak ada di disk.")
    df = pd.DataFrame({
        "image_id": "ham_" + meta["image_id"].astype(str),
        "domain": "ham",
        "label": meta["dx"].map(mp),
        "orig_label": meta["dx"],
        "lesion_id": "ham_" + meta["lesion_id"].astype(str),
        # HAM tidak punya patient_id; lesi adalah unit independen terkecil yang tersedia.
        "patient_id": "ham_" + meta["lesion_id"].astype(str),
        "confirm": meta.get("dx_type", pd.Series("", index=meta.index)).astype(str),
        "site": meta.get("dataset", pd.Series("", index=meta.index)).astype(str),
        "src_path": meta["src_path"],
    })
    print(f"[HAM] {n_all} baris metadata -> {len(df)} citra di 4 kelas")
    return df


def load_pad(cfg: dict) -> pd.DataFrame:
    root = resolve(cfg["paths"]["pad_dir"])
    _require_dir(root, "PAD-UFES-20")
    extract_archives(root)
    meta = _read_meta(_find_meta(root, ["metadata"]))
    need = {"patient_id", "lesion_id", "img_id", "diagnostic"}
    if not need.issubset(meta.columns):
        raise ValueError(f"Kolom metadata PAD kurang: {need - set(meta.columns)}")
    n_all = len(meta)
    mp = cfg["data"]["pad_map"]
    meta = meta[meta["diagnostic"].isin(mp)].copy()
    imgs = _find_images(root)
    stems = meta["img_id"].astype(str).map(lambda s: Path(s).stem)
    meta["src_path"] = stems.map(lambda s: str(imgs.get(s, "")))
    missing = (meta["src_path"] == "").sum()
    if missing:
        raise FileNotFoundError(f"{missing} citra PAD tercantum di metadata tetapi tidak ada di disk.")
    df = pd.DataFrame({
        "image_id": "pad_" + stems,
        "domain": "pad",
        "label": meta["diagnostic"].map(mp),
        "orig_label": meta["diagnostic"],
        "lesion_id": "pad_" + meta["lesion_id"].astype(str),
        "patient_id": "pad_" + meta["patient_id"].astype(str),
        "confirm": meta.get("biopsed", pd.Series("", index=meta.index)).astype(str),
        "site": "pad_ufes",
        "src_path": meta["src_path"],
    })
    print(f"[PAD] {n_all} baris metadata -> {len(df)} citra di 4 kelas")
    return df


def _resize_one(args):
    src, dst, short = args
    dst = Path(dst)
    if dst.exists():
        return
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = short / min(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.suffix.lower() == ".png":
            im.save(dst, optimize=False)
        else:
            # HAM sumbernya sudah JPEG; simpan ulang kualitas tinggi.
            im.save(dst, quality=95)


def build_index(cfg: dict) -> pd.DataFrame:
    """Gabungkan dua dataset, buat cache resize, simpan outputs/index.csv."""
    ham, pad = load_ham(cfg), load_pad(cfg)
    df = pd.concat([ham, pad], ignore_index=True)
    if df["image_id"].duplicated().any():
        raise ValueError("image_id ganda setelah penggabungan.")
    classes = cfg["data"]["classes"]
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)}).astype(int)
    cache = work_dir(cfg) / "cache"
    # Format sumber dipertahankan: PAD tetap PNG agar tidak menambah artefak JPEG
    # pada data OOD (JPEG juga salah satu augmentasi intervensi).
    df["cache_path"] = [str(cache / d / (i + Path(p).suffix.lower()))
                        for d, i, p in zip(df["domain"], df["image_id"], df["src_path"])]
    short = cfg["data"]["cache_short_side"]
    jobs = [(s, d, short) for s, d in zip(df["src_path"], df["cache_path"])]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_resize_one, jobs))
    out = work_dir(cfg) / "index.csv"
    # Simpan path relatif terhadap folder proyek (format '/'), supaya outputs/ bisa
    # dipindah antar komputer (mis. tahap CPU di Windows, training di Colab).
    disk = df.copy()
    for col in ("src_path", "cache_path"):
        disk[col] = disk[col].map(_portable)
    disk.to_csv(out, index=False)
    print(f"[index] {len(df)} citra -> {out}")
    print(pd.crosstab(df["label"], df["domain"]).to_string())
    return df


def _portable(p: str) -> str:
    import os
    ap = Path(os.path.abspath(p))           # abspath: TIDAK mengikuti symlink (cache Colab)
    try:
        return ap.relative_to(Path(os.path.abspath(ROOT))).as_posix()
    except ValueError:                      # di luar folder proyek (atau beda drive): absolut
        return ap.as_posix()


def load_index(cfg: dict) -> pd.DataFrame:
    p = work_dir(cfg) / "index.csv"
    if not p.exists():
        raise FileNotFoundError("index.csv belum ada. Jalankan tahap 'prepare' dulu.")
    df = pd.read_csv(p, dtype={"lesion_id": str, "patient_id": str})
    for col in ("src_path", "cache_path"):
        df[col] = df[col].map(lambda x: str(resolve(x)))
    return df


# ---------------------------------------------------------------- Dataset PyTorch
class ImageDataset:
    """Dataset sederhana dari DataFrame index. Mengembalikan (tensor, y, posisi)."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["cache_path"].tolist()
        self.y = df["y"].to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        return self.transform(im), int(self.y[i]), i


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    build_index(load_config(a.config, a.smoke))
