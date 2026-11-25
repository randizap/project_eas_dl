"""Split HAM10000 70/15/15 per grup (lesi + gabungan duplikat), atau per citra untuk RQ3.

- Test set ditentukan SEKALI dengan split.test_seed dan sama untuk semua run.
- Train/val dipartisi ulang per seed run (varians dari pengambilan sampel data).
- PAD-UFES-20 seluruhnya 'ood' (tidak pernah dipakai melatih atau memilih model).
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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import load_config, work_dir, write_json


def _group_split(groups: pd.DataFrame, frac: float, seed: int):
    """Pisahkan grup secara terstratifikasi berdasarkan label mayoritas grup."""
    g = groups.groupby("g")["label"].agg(lambda s: s.value_counts().index[0])
    counts = g.value_counts()
    strat = g if counts.min() >= 2 else None
    a, b = train_test_split(g.index.to_numpy(), test_size=frac, random_state=seed, stratify=strat)
    return set(a), set(b)


def make_split(cfg: dict, mode: str, seed: int) -> pd.DataFrame:
    if mode not in ("lesion", "image"):
        raise ValueError("mode harus 'lesion' atau 'image'")
    wd = work_dir(cfg)
    gp = wd / "groups.csv"
    if not gp.exists():
        raise FileNotFoundError("groups.csv belum ada. Jalankan tahap 'leakage' dulu.")
    groups = pd.read_csv(gp)
    rep = wd / "leakage" / "report.json"
    if rep.exists():
        import json
        r = json.loads(rep.read_text(encoding="utf-8"))
        if "WARNING" in r and not cfg["leakage"].get("allow_large_groups", False):
            raise RuntimeError("Split dihentikan: " + r["WARNING"])
    sp = cfg["split"]
    ham = groups[groups.domain == "ham"].copy()
    ham["g"] = ham["group_id"] if mode == "lesion" else ham["image_id"]

    rest, test = _group_split(ham, sp["test_frac"], sp["test_seed"])
    rest_df = ham[ham.g.isin(rest)]
    val_frac = sp["val_frac"] / (1.0 - sp["test_frac"])
    train, val = _group_split(rest_df, val_frac, seed)

    ham["split"] = np.select([ham.g.isin(train), ham.g.isin(val), ham.g.isin(test)],
                             ["train", "val", "test"], default="?")
    pad = groups[groups.domain == "pad"].copy()
    pad["g"] = pad["patient_id"]
    pad["split"] = np.where(pad["ood_exclude"], "excluded", "ood")
    out = pd.concat([ham, pad], ignore_index=True)[
        ["image_id", "domain", "lesion_id", "patient_id", "group_id", "label", "split"]]

    # ---- Verifikasi: tidak boleh ada grup di dua sisi split
    assert (out["split"] != "?").all()
    hs = out[out.domain == "ham"]
    key = "group_id" if mode == "lesion" else "image_id"
    per = hs.groupby(key)["split"].nunique()
    assert (per == 1).all(), "Grup bocor ke lebih dari satu split!"
    info = {"mode": mode, "seed": seed,
            "counts": {s: hs[hs.split == s]["label"].value_counts().to_dict()
                       for s in ("train", "val", "test")},
            "ood_counts": out[out.split == "ood"]["label"].value_counts().to_dict(),
            "ood_excluded": int((out.split == "excluded").sum())}
    if mode == "lesion":
        lesion_sides = hs.groupby("lesion_id")["split"].nunique()
        assert (lesion_sides == 1).all()
        info["lesion_overlap_train_test"] = 0
    else:
        tr_les = set(hs[hs.split == "train"].lesion_id)
        te = hs[hs.split == "test"]
        info["test_images_whose_lesion_in_train"] = int(te.lesion_id.isin(tr_les).sum())
        info["test_images_total"] = int(len(te))
    d = wd / "splits"
    d.mkdir(exist_ok=True)
    f = d / f"{mode}_seed{seed}.csv"
    if f.exists():
        old = pd.read_csv(f, dtype={"lesion_id": str, "patient_id": str})
        same = (len(old) == len(out) and (old["image_id"].to_numpy() == out["image_id"].to_numpy()).all()
                and (old["split"].to_numpy() == out["split"].to_numpy()).all())
        if not same and (wd / "runs").exists() and any((wd / "runs").iterdir()):
            raise RuntimeError(f"Split {f.name} berubah padahal sudah ada run terlatih. Kemungkinan ambang "
                               "kebocoran/config diubah. Kembalikan config, atau hapus outputs/runs dan latih ulang.")
    out.to_csv(f, index=False)
    write_json(d / f"{mode}_seed{seed}.json", info)
    return out


def load_split(cfg: dict, mode: str, seed: int) -> pd.DataFrame:
    p = work_dir(cfg) / "splits" / f"{mode}_seed{seed}.csv"
    if not p.exists():
        return make_split(cfg, mode, seed)
    return pd.read_csv(p, dtype={"lesion_id": str, "patient_id": str})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.smoke)
    for m in ("lesion", "image"):
        for s in cfg["seeds"]:
            make_split(cfg, m, s)
    print("split selesai")
