"""Satu perintah untuk seluruh pipeline (slide 4: 'jalan dengan satu perintah').

  python -m src.run_all                      # semua tahap, berurutan, idempoten
  python -m src.run_all --stages train       # hanya tahap tertentu
  python -m src.run_all --smoke              # uji asap pada data sintetis (CPU, ~menit)

Tahap: prepare -> leakage -> split -> tune -> train -> evaluate -> report -> export
('export' menulis results.json + model ONNX ke web/public untuk website Vercel.)
Run yang sudah selesai (done.json) dilewati, jadi aman diulang setelah Colab putus.
Tahap 'evaluate' membuka test set dan hanya boleh sekali.
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

from . import data, evaluate, export_web, leakage, make_report, split, train_cnn, train_linear_probe
from .utils import load_config, read_json, resolve, save_environment, work_dir, write_json

STAGES = ["prepare", "leakage", "split", "tune", "train", "evaluate", "report", "export"]


def tune(cfg):
    """Tuning learning rate model utama: 1 seed, dipilih dari val (slide 7)."""
    out = work_dir(cfg) / "tuned.json"
    if out.exists():
        print(f"[tune] sudah ada: {read_json(out)}")
        return
    s = cfg["tune_seed"]
    grid = cfg.get("tune_lr_grid", [1e-4, 3e-4, 1e-3]) if not cfg["_smoke"] else [3e-4]
    scores = {}
    for lr in grid:
        r = train_cnn.train(cfg, "main", s, "lesion", tag=f"lr{lr:g}", overrides={"lr": lr})
        scores[lr] = r["val_bal_acc"]
    best = max(scores, key=scores.get)
    write_json(out, {"lr": best, "val_bal_acc_per_lr": {str(k): v for k, v in scores.items()},
                     "seed": s, "selected_on": "val balanced accuracy HAM10000"})
    print(f"[tune] lr terpilih {best}")


def train_all(cfg):
    tj = work_dir(cfg) / "tuned.json"
    ov = {"lr": read_json(tj)["lr"]} if tj.exists() else {}
    if not ov:
        print("[train] PERINGATAN: tuned.json tidak ada, memakai lr dari config")
    for s in cfg["seeds"]:
        train_linear_probe.train(cfg, s, "lesion")
        train_linear_probe.train(cfg, s, "image")
    for s in cfg["seeds"]:
        train_cnn.train(cfg, "main", s, "lesion", overrides=ov)
        train_cnn.train(cfg, "phone", s, "lesion", overrides=ov)   # lr sama: hanya augmentasi yang beda
    for s in cfg["rq3_cnn_seeds"]:
        train_cnn.train(cfg, "main", s, "image", overrides=ov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--force-reopen-test", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="izinkan tune/train/evaluate tanpa GPU (sangat lambat)")
    a = ap.parse_args()
    cfg = load_config(resolve(a.config), a.smoke)
    gpu_stages = {"tune", "train", "evaluate"} & set(a.stages)
    if not a.smoke and gpu_stages:
        from .check_env import gpu_status
        has_gpu, msg = gpu_status()
        print(f"[run_all] GPU: {msg}")
        if not has_gpu and not a.allow_cpu:
            raise SystemExit(
                "[run_all] Tidak ada GPU CUDA. Tahap " + ", ".join(sorted(gpu_stages)) + " di CPU bisa makan "
                "berhari-hari.\n  - Jalankan dulu tahap CPU: python src/run_all.py --stages prepare leakage split\n"
                "  - Lalu training di Colab/Kaggle (README bagian 3), atau pasang torch CUDA bila punya GPU NVIDIA.\n"
                "  - Tetap mau di CPU? tambahkan --allow-cpu")
    if a.smoke:
        from . import make_synthetic
        make_synthetic.make(cfg)
    save_environment(cfg)
    for st in STAGES:
        if st not in a.stages:
            continue
        print(f"\n===== tahap: {st} =====")
        if st == "prepare":
            data.build_index(cfg)
        elif st == "leakage":
            leakage.run(cfg)
        elif st == "split":
            for m in ("lesion", "image"):
                for s in sorted(set(cfg["seeds"]) | set(cfg["rq3_cnn_seeds"])):
                    split.make_split(cfg, m, s)
        elif st == "tune":
            tune(cfg)
        elif st == "train":
            train_all(cfg)
        elif st == "evaluate":
            evaluate.run(cfg, force=a.force_reopen_test)
        elif st == "report":
            make_report.run(cfg)
        elif st == "export":
            export_web.run(cfg)


if __name__ == "__main__":
    main()
