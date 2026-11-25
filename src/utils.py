"""Utilitas bersama: konfigurasi, seed, log run, info lingkungan, hash file."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

LOG_FIELDS = [
    "run_id", "timestamp_utc", "stage", "config_name", "seed", "split_mode",
    "status", "gpu", "train_minutes", "best_epoch", "val_bal_acc", "val_macro_f1",
    "hparams", "notes",
]


def load_config(path: str | os.PathLike, smoke: bool = False) -> dict:
    path = resolve(path)   # relatif terhadap folder proyek, tidak bergantung cwd
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_smoke"] = smoke
    # Colab: data mentah & cache citra di disk lokal /content (cepat, tidak memenuhi kuota
    # Google Drive), sementara checkpoint & hasil tetap di Drive (outputs/).
    if not smoke:
        if os.environ.get("EAS_DATA_DIR"):
            d = Path(os.environ["EAS_DATA_DIR"])
            cfg["paths"]["ham_dir"] = str(d / "ham10000")
            cfg["paths"]["pad_dir"] = str(d / "pad_ufes_20")
        if os.environ.get("EAS_CACHE_DIR"):
            cfg["paths"]["cache_dir"] = os.environ["EAS_CACHE_DIR"]
    if smoke:
        # Mode uji asap: data sintetis, tanpa bobot pretrained, epoch sangat sedikit.
        # Output dan log dipisah supaya tidak mencemari log eksperimen asli.
        cfg["paths"]["ham_dir"] = "outputs_smoke/synthetic/ham10000"
        cfg["paths"]["pad_dir"] = "outputs_smoke/synthetic/pad_ufes_20"
        cfg["paths"]["work_dir"] = "outputs_smoke"
        cfg["paths"]["run_log"] = "outputs_smoke/log_smoke.csv"
        # EAS_SMOKE_WORKERS: hanya untuk menguji DataLoader multi-proses (spawn, seperti Windows)
        cfg["cnn"].update(epochs=2, batch_size=16, patience=2,
                          num_workers=int(os.environ.get("EAS_SMOKE_WORKERS", "0")))
        cfg["linear_probe"]["C_grid"] = [0.1, 1.0]
        # Bobot acak (tanpa internet) -> embedding DINOv2 hampir identik untuk semua citra;
        # deteksi kosinus tidak bermakna di mode ini, jadi hanya pHash yang diuji.
        cfg["leakage"]["cosine_min"] = 1.01
        cfg["eval"].update(bootstrap_B=200, corruption_severities=[1, 3, 5],
                           gradcam_n_per_domain=4)
    return cfg


def resolve(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def work_dir(cfg: dict) -> Path:
    d = resolve(cfg["paths"]["work_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cudnn.benchmark dimatikan: sedikit lebih lambat, hasil lebih bisa diulang.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "cpu"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_run_log(cfg: dict, row: dict) -> None:
    """Tambahkan satu baris ke runs/log.csv. Kolom yang tidak diisi dibiarkan kosong."""
    path = resolve(cfg["paths"]["run_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        out = {k: row.get(k, "") for k in LOG_FIELDS}
        for k, v in out.items():
            if isinstance(v, (dict, list)):
                out[k] = json.dumps(v, sort_keys=True)
            elif isinstance(v, float):
                out[k] = f"{v:.6g}"
        w.writerow(out)


@contextmanager
def logged_run(cfg: dict, base: dict):
    """Bungkus satu run: catat 'ok' atau 'failed' (beserta pesan galat) ke log.

    Run yang gagal tetap tercatat (slide 8: 'semua run, termasuk yang gagal').
    """
    info = dict(base)
    info.setdefault("timestamp_utc", now_utc())
    info.setdefault("gpu", gpu_name())
    t0 = time.time()
    try:
        yield info
    except BaseException as e:  # termasuk KeyboardInterrupt: tetap dicatat
        info["status"] = "failed"
        msg = f"{type(e).__name__}: {e}".replace("\n", " ")[:300]
        info["notes"] = (info.get("notes", "") + " | " + msg).strip(" |")
        info["train_minutes"] = (time.time() - t0) / 60
        append_run_log(cfg, info)
        traceback.print_exc()
        raise
    else:
        info.setdefault("status", "ok")
        info.setdefault("train_minutes", (time.time() - t0) / 60)
        append_run_log(cfg, info)


def save_environment(cfg: dict) -> Path:
    """Simpan versi Python/paket/GPU yang benar-benar dipakai (untuk reproducibility)."""
    out = work_dir(cfg) / "environment.txt"
    lines = [f"timestamp_utc: {now_utc()}", f"python: {sys.version.split()[0]}",
             f"platform: {platform.platform()}", f"gpu: {gpu_name()}"]
    try:
        import torch, torchvision
        lines += [f"torch: {torch.__version__}", f"torchvision: {torchvision.__version__}",
                  f"cuda: {torch.version.cuda}", f"cudnn: {torch.backends.cudnn.version()}"]
    except ImportError:
        pass
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, timeout=120).stdout
        lines += ["", "# pip freeze", freeze]
    except Exception as e:  # pip freeze bukan hal kritis
        lines.append(f"pip freeze gagal: {e}")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path: str | os.PathLike, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=float)


def read_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
