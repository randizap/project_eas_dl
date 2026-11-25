"""Evaluasi akhir: membuka test set SATU KALI (slide 7).

Syarat sebelum jalan: semua run yang direncanakan sudah selesai. Kunci
outputs/TEST_OPENED.json ditulis SEBELUM prediksi pertama; menjalankan ulang
ditolak kecuali --force (dan penggunaan --force tercatat di log).

Yang dihasilkan (semuanya prediksi mentah; metrik dihitung di make_report):
  predictions/<run_id>__test.npz, __ood.npz       in-distribution & out-of-distribution
  predictions/corruption/<run_id>.csv             balanced acc per korupsi x severity
  gradcam/                                        uji pengacakan + energi tepi/lesi
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
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import corruptions as C
from .augment import eval_transform
from .data import ImageDataset, load_index
from .gradcam import GradCAM, border_energy, mask_energy, randomization_test
from .metrics import balanced_accuracy
from .models import IMAGENET_MEAN, IMAGENET_STD, build_dinov2, build_efficientnet, gradcam_target_layer
from .split import load_split
from .utils import (append_run_log, get_device, gpu_name, load_config, now_utc, sha256_file,
                    work_dir, write_json)


def expected_runs(cfg) -> list[dict]:
    runs = []
    for s in cfg["seeds"]:
        runs += [{"id": f"lp_dinov2_lesion_s{s}", "kind": "lp", "mode": "lesion", "seed": s},
                 {"id": f"cnn_main_lesion_s{s}", "kind": "cnn", "mode": "lesion", "seed": s},
                 {"id": f"cnn_phone_lesion_s{s}", "kind": "cnn", "mode": "lesion", "seed": s},
                 {"id": f"lp_dinov2_image_s{s}", "kind": "lp", "mode": "image", "seed": s}]
    for s in cfg["rq3_cnn_seeds"]:
        runs.append({"id": f"cnn_main_image_s{s}", "kind": "cnn", "mode": "image", "seed": s})
    return runs


def _ckpt(cfg, r) -> Path:
    d = work_dir(cfg) / "runs" / r["id"]
    return d / ("model.pkl" if r["kind"] == "lp" else "best.pt")


def _frame(cfg, mode, seed):
    idx = load_index(cfg)
    sp = load_split(cfg, mode, seed)
    return idx.merge(sp[["image_id", "split"]], on="image_id")


class CorruptDataset(Dataset):
    """Citra test -> resize+crop 224 (uint8) -> korupsi -> normalisasi."""

    def __init__(self, df, size, name, sev):
        self.paths, self.y = df["cache_path"].tolist(), df["y"].to_numpy()
        self.size, self.name, self.sev = size, name, sev
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        w, h = im.size
        r = round(self.size / 0.875) / min(w, h)
        im = im.resize((round(w * r), round(h * r)), Image.BILINEAR)
        w, h = im.size
        l, t = (w - self.size) // 2, (h - self.size) // 2
        a = np.asarray(im.crop((l, t, l + self.size, t + self.size)))
        # seed deterministik per (korupsi, severity, citra); hash() Python diacak per proses
        seed = (zlib.crc32(f"{self.name}:{self.sev}".encode()) + 7919 * i) % (2 ** 32)
        a = C.apply(self.name, a, self.sev, seed)
        x = torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 255.0
        return (x - self.mean) / self.std, int(self.y[i]), i


@torch.no_grad()
def _probs_cnn(model, x):
    return torch.softmax(model(x).float(), 1).cpu().numpy()


def run(cfg: dict, force: bool = False) -> None:
    wd = work_dir(cfg)
    lock = wd / "TEST_OPENED.json"
    runs = expected_runs(cfg)
    missing = [r["id"] for r in runs if not (wd / "runs" / r["id"] / "done.json").exists()]
    if missing:
        raise RuntimeError(f"Test BELUM boleh dibuka; run belum selesai: {missing}")
    if lock.exists() and not force:
        raise RuntimeError(f"Test set sudah pernah dibuka ({lock}). Hasil ada di outputs/predictions. "
                           "Jangan buka ulang untuk memilih model. --force hanya untuk galat teknis "
                           "dan akan tercatat di log.")
    write_json(lock, {"opened_utc": now_utc(), "forced_reopen": bool(lock.exists() and force),
                      "checkpoints": {r["id"]: sha256_file(_ckpt(cfg, r)) for r in runs}})
    append_run_log(cfg, {"run_id": "final_eval", "timestamp_utc": now_utc(), "stage": "test_open",
                         "status": "ok", "gpu": gpu_name(),
                         "notes": "FORCED REOPEN" if force else "test dibuka pertama kali"})

    device = get_device()
    size = cfg["data"]["img_size"]
    k = len(cfg["data"]["classes"])
    bs = cfg["cnn"]["batch_size"] * 2
    nw = cfg["cnn"]["num_workers"]
    pdir = wd / "predictions"
    (pdir / "corruption").mkdir(parents=True, exist_ok=True)
    feats_cache = np.load(wd / "features" / "dinov2_vits14.npz", allow_pickle=True)
    feats = dict(zip(feats_cache["image_id"], feats_cache["feat"]))

    models = {}
    for r in runs:
        df = _frame(cfg, r["mode"], r["seed"])
        if r["kind"] == "lp":
            with open(_ckpt(cfg, r), "rb") as f:
                clf = pickle.load(f)
            models[r["id"]] = ("lp", clf)
        else:
            m = build_efficientnet(k, None)
            m.load_state_dict(torch.load(_ckpt(cfg, r), map_location="cpu", weights_only=False)["model"])
            m.to(device).eval()
            models[r["id"]] = ("cnn", m)
        for part in ("test", "ood"):
            sub = df[df.split == part]
            if r["kind"] == "lp":
                prob = models[r["id"]][1].predict_proba(np.stack([feats[i] for i in sub.image_id]))
            else:
                dl = DataLoader(ImageDataset(sub, eval_transform(size)), batch_size=bs, num_workers=nw)
                prob = np.concatenate([_probs_cnn(models[r["id"]][1], x.to(device)) for x, _, _ in dl])
            np.savez_compressed(pdir / f"{r['id']}__{part}.npz", prob=prob, y=sub.y.to_numpy(),
                                image_id=sub.image_id.to_numpy(), lesion_id=sub.lesion_id.to_numpy(),
                                patient_id=sub.patient_id.to_numpy())
            print(f"[eval] {r['id']} {part}: bal_acc={balanced_accuracy(sub.y.to_numpy(), prob.argmax(1), k):.4f}")

    # ---------------- korupsi pada test in-distribution (split per lesi)
    lesion_runs = [r for r in runs if r["mode"] == "lesion"]
    test_df = _frame(cfg, "lesion", cfg["seeds"][0])
    test_df = test_df[test_df.split == "test"]
    dino = build_dinov2(pretrained=not cfg["_smoke"], img_size=size).to(device)
    rows = []
    plan = [("clean", 0)] + [(c, s) for c in cfg["eval"]["corruptions"]
                             for s in cfg["eval"]["corruption_severities"]]
    for name, sev in plan:
        dl = DataLoader(CorruptDataset(test_df, size, name, sev), batch_size=bs, num_workers=nw)
        probs = {r["id"]: [] for r in lesion_runs}
        ys = []
        for x, y, _ in dl:
            x = x.to(device)
            f = None
            for r in lesion_runs:
                kind, m = models[r["id"]]
                if kind == "cnn":
                    probs[r["id"]].append(_probs_cnn(m, x))
                else:
                    if f is None:
                        with torch.no_grad():
                            f = dino(x).float().cpu().numpy()
                    probs[r["id"]].append(m.predict_proba(f))
            ys.append(y.numpy())
        yy = np.concatenate(ys)
        for rid, pl in probs.items():
            p = np.concatenate(pl)
            rows.append({"run_id": rid, "corruption": name, "severity": sev,
                         "overlap_with_phone_aug": name in C.OVERLAP,
                         "bal_acc": balanced_accuracy(yy, p.argmax(1), k)})
        print(f"[corruption] {name} s{sev} selesai")
    pd.DataFrame(rows).to_csv(pdir / "corruption" / "corruption_results.csv", index=False)

    # ---------------- Grad-CAM + uji pengacakan (model seed tuning, utama & intervensi)
    gdir = wd / "gradcam"
    gdir.mkdir(exist_ok=True)
    n = cfg["eval"]["gradcam_n_per_domain"]
    s0 = cfg["tune_seed"]
    full = _frame(cfg, "lesion", s0)
    rng = np.random.default_rng(0)
    pick = {}
    for part in ("test", "ood"):
        sub = full[full.split == part]
        # sampel seimbang per kelas sejauh tersedia
        per = max(1, n // k)
        idxs = np.concatenate([rng.choice(g.index.to_numpy(), min(per, len(g)), replace=False)
                               for _, g in sub.groupby("y")])
        pick[part] = sub.loc[idxs]
    mask_dir = cfg["paths"].get("ham_mask_dir")
    sims, energy = [], []
    for name in ("main", "phone"):
        rid = f"cnn_{name}_lesion_s{s0}"
        m = models[rid][1]
        for part, sub in pick.items():
            ds = ImageDataset(sub, eval_transform(size))
            x = torch.stack([ds[i][0] for i in range(len(ds))])
            y = torch.tensor(sub.y.to_numpy())
            orig, df_sim, maps = randomization_test(m, x, y, seed=0)
            df_sim["run_id"], df_sim["domain"] = rid, part
            sims.append(df_sim)
            np.savez_compressed(gdir / f"{rid}__{part}_maps.npz", image_id=sub.image_id.to_numpy(),
                                **{k_.replace(".", "_"): v for k_, v in maps.items()})
            for i, iid in enumerate(sub.image_id):
                e = {"run_id": rid, "domain": part, "image_id": iid,
                     "border_energy": border_energy(orig[i]), "label": int(sub.y.iloc[i])}
                if mask_dir and part == "test":
                    hits = list(Path(mask_dir).rglob(f"*{iid.replace('ham_', '')}*"))
                    if hits:
                        with Image.open(hits[0]) as mi:
                            mk = np.asarray(eval_transform_mask(mi, size))
                        e["lesion_energy"] = mask_energy(orig[i], mk)
                energy.append(e)
    pd.concat(sims).to_csv(gdir / "randomization_similarity.csv", index=False)
    pd.DataFrame(energy).to_csv(gdir / "cam_energy.csv", index=False)
    print("[eval] selesai. Lanjut: python -m src.make_report")


def eval_transform_mask(im: Image.Image, size: int) -> Image.Image:
    """Transformasi geometri yang sama dengan eval_transform, untuk masker."""
    im = im.convert("L")
    w, h = im.size
    r = round(size / 0.875) / min(w, h)
    im = im.resize((round(w * r), round(h * r)), Image.NEAREST)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return im.crop((l, t, l + size, t + size))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(load_config(a.config, a.smoke), a.force)
