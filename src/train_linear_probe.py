"""Baseline: fitur DINOv2 ViT-S/14 beku + regresi logistik berbobot kelas.

C dipilih dari balanced accuracy validasi (per seed). Fitur diambil dari tahap
'leakage' (transformasi evaluasi, tanpa augmentasi), jadi baseline ini murah.
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
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import balanced_accuracy, macro_f1
from .train_cnn import merged_frame
from .utils import load_config, logged_run, read_json, set_seed, work_dir, write_json


def load_features(cfg) -> dict[str, np.ndarray]:
    p = work_dir(cfg) / "features" / "dinov2_vits14.npz"
    if not p.exists():
        raise FileNotFoundError("Fitur DINOv2 belum ada. Jalankan tahap 'leakage' dulu.")
    z = np.load(p, allow_pickle=True)
    return dict(zip(z["image_id"], z["feat"]))


def train(cfg: dict, seed: int, mode: str = "lesion") -> dict:
    run_id = f"lp_dinov2_{mode}_s{seed}"
    rdir = work_dir(cfg) / "runs" / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    if (rdir / "done.json").exists():
        print(f"[skip] {run_id} sudah selesai")
        return read_json(rdir / "done.json")
    lp = cfg["linear_probe"]
    base = {"run_id": run_id, "stage": "train", "config_name": "baseline_lp", "seed": seed,
            "split_mode": mode, "hparams": {"C_grid": lp["C_grid"], "backbone": lp["backbone"]}}
    with logged_run(cfg, base) as info:
        set_seed(seed)
        k = len(cfg["data"]["classes"])
        feats = load_features(cfg)
        df = merged_frame(cfg, mode, seed)
        tr, va = df[df.split == "train"], df[df.split == "val"]
        Xtr = np.stack([feats[i] for i in tr.image_id]); ytr = tr.y.to_numpy()
        Xva = np.stack([feats[i] for i in va.image_id]); yva = va.y.to_numpy()
        scores = {}
        best = (-1.0, None, None, None)
        for C in lp["C_grid"]:
            clf = make_pipeline(StandardScaler(), LogisticRegression(
                C=C, class_weight="balanced", max_iter=lp["max_iter"], random_state=seed))
            clf.fit(Xtr, ytr)
            pv = clf.predict_proba(Xva)
            ba = balanced_accuracy(yva, pv.argmax(1), k)
            scores[str(C)] = ba
            if ba > best[0]:
                best = (ba, C, clf, pv)
        ba, C, clf, pv = best
        with open(rdir / "model.pkl", "wb") as f:
            pickle.dump(clf, f)
        np.savez_compressed(rdir / "val_preds.npz", prob=pv, y=yva, image_id=va.image_id.to_numpy())
        res = {"run_id": run_id, "best_C": C, "val_bal_acc": ba,
               "val_macro_f1": macro_f1(yva, pv.argmax(1), k), "val_scores_per_C": scores}
        write_json(rdir / "done.json", res)
        info.update(val_bal_acc=ba, val_macro_f1=res["val_macro_f1"], notes=f"best_C={C}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-mode", choices=["lesion", "image"], default="lesion")
    a = ap.parse_args()
    train(load_config(a.config, a.smoke), a.seed, a.split_mode)
