"""Pemeriksaan kebocoran (slide 4 & 5): duplikat / hampir-kembar sebelum split.

1. Perceptual hash (pHash 64-bit) semua citra.
2. Embedding DINOv2 (juga dipakai ulang oleh linear probe) dan jarak kosinus.
3. Pasangan di atas ambang -> union-find: lesi HAM yang punya citra hampir-kembar
   digabung menjadi satu grup, sehingga tidak bisa terpisah ke train dan test.
4. Pasangan HAM<->PAD -> citra PAD dikeluarkan dari uji OOD.
5. Lembar kontak + CSV untuk pemeriksaan manual (kolom manual_verdict).
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
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .augment import eval_transform
from .data import ImageDataset, load_index
from .models import build_dinov2, extract_features
from .utils import get_device, load_config, set_seed, work_dir, write_json

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash_all(paths: list[str]) -> np.ndarray:
    import imagehash
    out = np.empty(len(paths), dtype=np.uint64)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            h = imagehash.phash(im.convert("RGB"), hash_size=8).hash.flatten()
        out[i] = np.packbits(h.astype(np.uint8)).view(">u8")[0]
    return out


def hamming_pairs(h: np.ndarray, max_d: int, chunk: int = 256):
    """Semua pasangan (i<j) dengan jarak Hamming <= max_d."""
    n = len(h)
    I, J, D = [], [], []
    for s in range(0, n, chunk):
        x = np.bitwise_xor(h[s:s + chunk, None], h[None, :])          # (c, n) uint64
        d = _POP[x.view(np.uint8)].reshape(x.shape[0], n, 8).sum(-1)  # popcount
        ii, jj = np.nonzero(d <= max_d)
        ii = ii + s
        keep = jj > ii
        I.append(ii[keep]); J.append(jj[keep]); D.append(d[ii[keep] - s, jj[keep]])
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def cosine_pairs(f: np.ndarray, min_c: float, chunk: int = 512):
    f = f / np.linalg.norm(f, axis=1, keepdims=True).clip(1e-12)
    n = len(f)
    I, J, C = [], [], []
    for s in range(0, n, chunk):
        c = f[s:s + chunk] @ f.T
        ii, jj = np.nonzero(c >= min_c)
        ii = ii + s
        keep = jj > ii
        I.append(ii[keep]); J.append(jj[keep]); C.append(c[ii[keep] - s, jj[keep]])
    return np.concatenate(I), np.concatenate(J), np.concatenate(C)


class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def compute_dinov2_features(cfg: dict, df: pd.DataFrame) -> np.ndarray:
    """Fitur CLS DINOv2 untuk semua citra (transformasi evaluasi, tanpa augmentasi)."""
    out = work_dir(cfg) / "features" / "dinov2_vits14.npz"
    if out.exists():
        z = np.load(out, allow_pickle=True)
        if list(z["image_id"]) == df["image_id"].tolist():
            return z["feat"]
    from torch.utils.data import DataLoader
    ds = ImageDataset(df, eval_transform(cfg["data"]["img_size"]))
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=cfg["cnn"]["num_workers"])
    model = build_dinov2(pretrained=not cfg["_smoke"], img_size=cfg["data"]["img_size"])
    feat = extract_features(model, dl, get_device()).numpy().astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feat=feat, image_id=df["image_id"].to_numpy())
    return feat


def contact_sheet(df, pairs: pd.DataFrame, path, n: int, thumb: int = 128):
    pairs = pairs.head(n)
    if pairs.empty:
        return
    rows = len(pairs)
    sheet = Image.new("RGB", (thumb * 2 + 8, rows * (thumb + 4)), "white")
    for r, (_, p) in enumerate(pairs.iterrows()):
        for c, col in enumerate(["path_a", "path_b"]):
            with Image.open(p[col]) as im:
                im = im.convert("RGB").resize((thumb, thumb))
            sheet.paste(im, (c * (thumb + 8), r * (thumb + 4)))
    sheet.save(path)


def run(cfg: dict) -> None:
    set_seed(0)
    lk = cfg["leakage"]
    df = load_index(cfg)
    wd = work_dir(cfg)
    print("[leakage] menghitung pHash ...")
    h = phash_all(df["cache_path"].tolist())
    print("[leakage] menghitung embedding DINOv2 ...")
    feat = compute_dinov2_features(cfg, df)

    (wd / "leakage").mkdir(exist_ok=True)
    hi, hj, hd = hamming_pairs(h, lk["phash_max_hamming"])
    ci, cj, cc = cosine_pairs(feat, lk["cosine_min"])
    fn = feat / np.linalg.norm(feat, axis=1, keepdims=True).clip(1e-12)
    pairs = pd.concat([
        pd.DataFrame({"i": hi, "j": hj, "method": "phash", "phash_dist": hd}),
        pd.DataFrame({"i": ci, "j": cj, "method": "cosine", "cosine": cc}),
    ], ignore_index=True)
    if not pairs.empty:
        pairs = pairs.groupby(["i", "j"], as_index=False).agg(
            method=("method", lambda s: "+".join(sorted(set(s)))))
        ii, jj = pairs["i"].to_numpy(), pairs["j"].to_numpy()
        x = np.ascontiguousarray(h[ii] ^ h[jj])
        pairs["phash_dist"] = _POP[x.view(np.uint8)].reshape(-1, 8).sum(1).astype(int)
        pairs["cosine"] = np.einsum("ij,ij->i", fn[ii], fn[jj])
    else:
        pairs = pd.DataFrame(columns=["i", "j", "method", "phash_dist", "cosine"])
    for side, col in (("a", "i"), ("b", "j")):
        for k in ("image_id", "domain", "lesion_id", "label", "cache_path"):
            pairs[f"{k}_{side}"] = df[k].to_numpy()[pairs[col].astype(int)] if len(pairs) else []
    pairs = pairs.rename(columns={"cache_path_a": "path_a", "cache_path_b": "path_b"})
    pairs["kind"] = np.where(pairs["domain_a"] != pairs["domain_b"], "cross_dataset",
                    np.where(pairs["lesion_id_a"] == pairs["lesion_id_b"], "same_lesion",
                             "diff_lesion_" + pairs["domain_a"].astype(str)))
    pairs["label_conflict"] = pairs["label_a"] != pairs["label_b"]
    pairs["manual_verdict"] = ""   # isi: dup / bukan  (pemeriksaan manual, proposal 3.1)
    pairs = pairs.sort_values(["kind", "cosine"], ascending=[True, False])
    # Verdict manual (proposal 3.1): salin pairs.csv -> pairs_reviewed.csv, isi kolom
    # manual_verdict dengan 'bukan' untuk pasangan yang ternyata bukan duplikat,
    # lalu jalankan ulang tahap leakage. Pasangan 'bukan' tidak digabung/dikeluarkan.
    reviewed = wd / "leakage" / "pairs_reviewed.csv"
    rejected = set()
    if reviewed.exists():
        rv = pd.read_csv(reviewed, dtype=str).fillna("")
        rej = rv[rv["manual_verdict"].str.strip().str.lower() == "bukan"]
        rejected = set(zip(rej["image_id_a"], rej["image_id_b"]))
        pairs["manual_verdict"] = [("bukan" if (a, b) in rejected else "")
                                   for a, b in zip(pairs.image_id_a, pairs.image_id_b)]
        print(f"[leakage] {len(rejected)} pasangan ditolak lewat pemeriksaan manual")
    active = pairs[pairs["manual_verdict"] != "bukan"]

    # --- Grup split: lesi HAM + gabungan karena hampir-kembar lintas lesi
    ham = df[df.domain == "ham"]
    uf = UnionFind(ham["lesion_id"].unique())
    sus = active[active.kind == "diff_lesion_ham"]
    for a, b in zip(sus["lesion_id_a"], sus["lesion_id_b"]):
        uf.union(a, b)
    groups = df[["image_id", "domain", "lesion_id", "patient_id", "label"]].copy()
    groups["group_id"] = [uf.find(l) if d == "ham" else l for d, l in zip(df.domain, df.lesion_id)]
    cross = active[active.kind == "cross_dataset"]
    pad_ex = set(np.where(cross["domain_a"] == "pad", cross["image_id_a"], cross["image_id_b"]))
    groups["ood_exclude"] = groups["image_id"].isin(pad_ex)

    gsize = groups[groups.domain == "ham"].groupby("group_id").size()
    mixed = groups[groups.domain == "ham"].groupby("group_id")["label"].nunique()
    report = {
        "thresholds": {"phash_max_hamming": lk["phash_max_hamming"], "cosine_min": lk["cosine_min"]},
        "n_images": int(len(df)),
        "pairs_by_kind": {k: int(v) for k, v in pairs["kind"].value_counts().items()},
        "pairs_label_conflict": int(pairs["label_conflict"].sum()),
        "pairs_rejected_manually": int(len(rejected)),
        "ham_lesions": int(ham["lesion_id"].nunique()),
        "ham_groups_after_merge": int(gsize.size),
        "largest_group_images": int(gsize.max()) if len(gsize) else 0,
        "groups_with_mixed_labels": int((mixed > 1).sum()),
        "pad_excluded_from_ood": int(len(pad_ex)),
    }
    # Satu lesi HAM wajar punya beberapa citra; peringatan bila grup jauh lebih besar.
    if report["largest_group_images"] > max(lk["max_group_warn_frac"] * len(ham), 20):
        report["WARNING"] = ("Ada grup duplikat sangat besar: ambang kemungkinan terlalu longgar. "
                             "Periksa outputs/leakage/contact_sheet.png & pairs.csv, lalu perketat ambang di config atau tandai 'bukan' di pairs_reviewed.csv. Tahap split akan menolak jalan sampai ini beres.")
    pairs.drop(columns=["i", "j"]).to_csv(wd / "leakage" / "pairs.csv", index=False)
    groups.to_csv(wd / "groups.csv", index=False)
    write_json(wd / "leakage" / "report.json", report)
    review = pairs[pairs.kind != "same_lesion"]
    contact_sheet(df, review, wd / "leakage" / "contact_sheet.png", lk["contact_sheet_pairs"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(load_config(a.config, a.smoke))
