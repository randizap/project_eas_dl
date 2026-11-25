"""Periksa kesiapan sebelum menjalankan pipeline.

  python src/check_env.py        (atau: python -m src.check_env)

Mencetak: versi Python & paket, ketersediaan GPU CUDA, dan apakah data sudah ada.
Tidak mengubah apa pun.
"""
from __future__ import annotations

# Bisa dijalankan sebagai file maupun modul.
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "src"
    import src  # noqa: F401

import importlib
import platform
import sys

PKGS = ["torch", "torchvision", "timm", "numpy", "pandas", "sklearn", "scipy", "skimage",
        "PIL", "imagehash", "matplotlib", "yaml"]


def gpu_status() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch belum terpasang"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return True, f"{p.name}, {p.total_memory / 1e9:.1f} GB, CUDA {torch.version.cuda}"
    build = "build CPU" if torch.version.cuda is None else f"build CUDA {torch.version.cuda} tapi GPU tidak terdeteksi"
    return False, f"tidak ada GPU CUDA ({build})"


def main() -> int:
    ok = True
    print(f"Python  : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    if sys.version_info < (3, 10):
        print("  [X] butuh Python >= 3.10"); ok = False
    print("Paket   :")
    for m in PKGS:
        try:
            mod = importlib.import_module(m)
            print(f"  [ok] {m:12s} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"  [X]  {m:12s} TIDAK ADA ({type(e).__name__})"); ok = False
    has_gpu, msg = gpu_status()
    print(f"GPU     : {msg}")
    if not has_gpu:
        print("  [!] Training 12 run EfficientNet di CPU tidak realistis (bisa berhari-hari).")
        print("      Pilihan: (a) GPU NVIDIA lokal: pasang torch versi CUDA, lihat README bagian 3;")
        print("               (b) Google Colab / Kaggle (GPU T4 gratis).")
        print("      Tahap prepare/leakage/split tetap bisa jalan di CPU.")

    from .utils import load_config, resolve
    cfg = load_config(resolve("configs/default.yaml"))
    print("Data    :")
    for key, name in (("ham_dir", "HAM10000"), ("pad_dir", "PAD-UFES-20")):
        d = resolve(cfg["paths"][key])
        n = sum(1 for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".zip"}) if d.exists() else 0
        tag = "[ok]" if n else "[X] "
        print(f"  {tag} {name:12s} {d}  ({n} file citra/zip)")
        ok &= n > 0
    print("\nHasil   :", "SIAP" if ok else "BELUM SIAP (lihat [X] di atas)")
    print("Uji instalasi tanpa data : python src/run_all.py --smoke")
    print("Pipeline asli            : python src/run_all.py")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
