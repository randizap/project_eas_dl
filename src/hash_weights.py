"""Tulis weights.txt: tautan + SHA-256 setiap checkpoint final (slide 8).

  python -m src.hash_weights --link-prefix "https://drive.google.com/drive/folders/<ID>"

Unggah folder outputs/runs/<run_id>/ ke Google Drive / Hugging Face dulu,
lalu jalankan ini. Penilai bisa memverifikasi file dengan `sha256sum`.
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

from .evaluate import _ckpt, expected_runs
from .utils import ROOT, load_config, now_utc, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--link-prefix", required=True, help="tautan folder bobot (Drive/HF)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.smoke)
    lines = ["# weights.txt - dibuat otomatis oleh src/hash_weights.py",
             f"# dibuat: {now_utc()}",
             f"# lokasi: {a.link_prefix}",
             "# Bobot pretrained awal (bukan hasil kami):",
             "#   EfficientNet-B0: torchvision EfficientNet_B0_Weights.IMAGENET1K_V1",
             "#   DINOv2 ViT-S/14: timm vit_small_patch14_dinov2.lvd142m (Oquab dkk. 2024)",
             "# format: run_id<TAB>file<TAB>sha256<TAB>tautan", ""]
    for r in expected_runs(cfg):
        p = _ckpt(cfg, r)
        lines.append(f"{r['id']}\t{p.name}\t{sha256_file(p)}\t{a.link_prefix.rstrip('/')}/{r['id']}/{p.name}")
    out = ROOT / ("weights.txt" if not a.smoke else "outputs_smoke/weights_smoke.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{out} ditulis")


if __name__ == "__main__":
    main()
