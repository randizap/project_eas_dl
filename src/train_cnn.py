"""Latih EfficientNet-B0 (model utama) atau EfficientNet-B0 + augmentasi ponsel (intervensi).

Pemilihan checkpoint HANYA dari balanced accuracy validasi HAM10000.
PAD-UFES-20 dan test set tidak disentuh di sini.

Contoh:
  python -m src.train_cnn --name main  --seed 0
  python -m src.train_cnn --name phone --seed 0
  python -m src.train_cnn --name main  --seed 0 --split-mode image   # RQ3
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
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .augment import eval_transform, train_transform
from .data import ImageDataset, load_index
from .metrics import balanced_accuracy, macro_f1
from .models import build_efficientnet
from .split import load_split
from .utils import (get_device, load_config, logged_run, read_json, set_seed, work_dir,
                    write_json)

CONFIGS = {"main": False, "phone": True}  # nama -> pakai augmentasi ponsel?


def class_weights(y: np.ndarray, k: int) -> torch.Tensor:
    n = np.bincount(y, minlength=k).astype(float)
    if (n == 0).any():
        raise ValueError(f"Ada kelas tanpa sampel di train: {n}")
    w = n.sum() / (k * n)   # bobot invers frekuensi; rerata berbobot = 1
    return torch.tensor(w, dtype=torch.float32)


def merged_frame(cfg, mode, seed) -> pd.DataFrame:
    idx = load_index(cfg)
    sp = load_split(cfg, mode, seed)
    return idx.merge(sp[["image_id", "split", "group_id"]], on="image_id", how="inner")


@torch.no_grad()
def predict_probs(model, loader, device, amp: bool) -> np.ndarray:
    model.eval()
    out = []
    for x, _, _ in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        out.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out)


def atomic_torch_save(obj, path) -> None:
    """Simpan ke file sementara lalu os.replace: sesi Colab yang putus di tengah
    penyimpanan tidak meninggalkan checkpoint korup (yang membuat resume gagal terus)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load_resume(last: Path, device):
    if not last.exists():
        return None
    try:
        return torch.load(last, map_location=device, weights_only=False)
    except Exception as e:  # checkpoint korup dari versi lama / disk penuh
        bad = last.with_suffix(".corrupt")
        os.replace(last, bad)
        print(f"[resume] PERINGATAN: {last.name} korup ({type(e).__name__}); dipindah ke {bad.name}, "
              "training diulang dari awal")
        return None


def lr_lambda(warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


def train(cfg: dict, name: str, seed: int, mode: str = "lesion", tag: str = "",
          overrides: dict | None = None) -> dict:
    if name not in CONFIGS:
        raise ValueError(f"name harus salah satu {list(CONFIGS)}")
    hp = dict(cfg["cnn"])
    hp.update(overrides or {})
    run_id = f"cnn_{name}_{mode}_s{seed}{('_' + tag) if tag else ''}"
    rdir = work_dir(cfg) / "runs" / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    if (rdir / "done.json").exists():
        print(f"[skip] {run_id} sudah selesai")
        return read_json(rdir / "done.json")

    stage = "tune" if tag else "train"
    base = {"run_id": run_id, "stage": stage, "config_name": name, "seed": seed,
            "split_mode": mode, "hparams": {k: hp[k] for k in
                                            ("epochs", "batch_size", "lr", "weight_decay", "warmup_epochs")}}
    with logged_run(cfg, base) as info:
        set_seed(seed)
        device = get_device()
        classes = cfg["data"]["classes"]
        k = len(classes)
        df = merged_frame(cfg, mode, seed)
        tr, va = df[df.split == "train"], df[df.split == "val"]
        size = cfg["data"]["img_size"]
        g = torch.Generator().manual_seed(seed)
        dl_tr = DataLoader(ImageDataset(tr, train_transform(cfg, CONFIGS[name])),
                           batch_size=hp["batch_size"], shuffle=True, drop_last=True,
                           num_workers=hp["num_workers"], pin_memory=device.type == "cuda",
                           generator=g, persistent_workers=hp["num_workers"] > 0)
        dl_va = DataLoader(ImageDataset(va, eval_transform(size)), batch_size=hp["batch_size"] * 2,
                           shuffle=False, num_workers=hp["num_workers"])

        weights = None if cfg["_smoke"] else cfg["cnn"]["weights"]
        model = build_efficientnet(k, weights).to(device)
        crit = nn.CrossEntropyLoss(weight=class_weights(tr.y.to_numpy(), k).to(device))
        opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        steps_per_epoch = max(1, len(dl_tr))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda(hp["warmup_epochs"] * steps_per_epoch, hp["epochs"] * steps_per_epoch))
        use_amp = bool(hp["amp"]) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") and \
            hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=use_amp)

        start_epoch, best, best_epoch, bad, hist = 0, -1.0, -1, 0, []
        last = rdir / "last.pt"
        ck = _load_resume(last, device)
        if ck is not None:  # lanjut dari sesi Colab yang putus
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
            start_epoch, best, best_epoch, bad, hist = (ck["epoch"] + 1, ck["best"], ck["best_epoch"],
                                                        ck["bad"], ck["hist"])
            torch.set_rng_state(ck["rng_cpu"])
            print(f"[resume] {run_id} dari epoch {start_epoch}")

        t0 = time.time()
        for ep in range(start_epoch, hp["epochs"]):
            model.train()
            tl, nb = 0.0, 0
            for x, y, _ in dl_tr:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = crit(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"loss tidak hingga di epoch {ep}")
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                tl += loss.item(); nb += 1
            pv = predict_probs(model, dl_va, device, use_amp)
            yv = va.y.to_numpy()
            ba, mf = balanced_accuracy(yv, pv.argmax(1), k), macro_f1(yv, pv.argmax(1), k)
            hist.append({"epoch": ep, "train_loss": tl / max(1, nb), "val_bal_acc": ba,
                         "val_macro_f1": mf, "lr": opt.param_groups[0]["lr"]})
            print(f"[{run_id}] ep {ep} loss {tl / max(1, nb):.4f} val_bal_acc {ba:.4f} macro_f1 {mf:.4f}")
            if ba > best:
                best, best_epoch, bad = ba, ep, 0
                atomic_torch_save({"model": model.state_dict(), "epoch": ep, "val_bal_acc": ba},
                                  rdir / "best.pt")
                np.savez_compressed(rdir / "val_preds.npz", prob=pv, y=yv,
                                    image_id=va.image_id.to_numpy())
            else:
                bad += 1
            atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": ep,
                        "best": best, "best_epoch": best_epoch, "bad": bad, "hist": hist,
                        "rng_cpu": torch.get_rng_state()}, last)
            if bad >= hp["patience"]:
                print(f"[{run_id}] early stopping di epoch {ep}")
                break

        pd.DataFrame(hist).to_csv(rdir / "history.csv", index=False)
        best_row = next(h for h in hist if h["epoch"] == best_epoch)
        result = {"run_id": run_id, "best_epoch": best_epoch, "val_bal_acc": best_row["val_bal_acc"],
                  "val_macro_f1": best_row["val_macro_f1"], "minutes_this_session": (time.time() - t0) / 60}
        if torch.cuda.is_available():
            result["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        write_json(rdir / "done.json", result)
        last.unlink(missing_ok=True)   # hemat ruang Drive; best.pt yang dipakai
        info.update(best_epoch=best_epoch, val_bal_acc=result["val_bal_acc"],
                    val_macro_f1=result["val_macro_f1"],
                    notes=f"peak_gpu_mem_gb={result.get('peak_gpu_mem_gb', 0):.2f}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", choices=list(CONFIGS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-mode", choices=["lesion", "image"], default="lesion")
    a = ap.parse_args()
    train(load_config(a.config, a.smoke), a.name, a.seed, a.split_mode)
