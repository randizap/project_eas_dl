"""Ekspor hasil pipeline ke website (folder web/public), dijalankan di akhir notebook Colab.

  python -m src.export_web            # hasil eksperimen asli (butuh tahap evaluate + report)
  python -m src.export_web --smoke    # data sintetis, website menampilkan banner DEMO

Yang ditulis:
  web/public/data/results.json        satu file berisi SEMUA angka yang ditampilkan website
  web/public/models/<run>.onnx        EfficientNet-B0 utama & intervensi (seed tuning) untuk demo di browser
  web/public/media/...                citra contoh Grad-CAM & contact sheet kebocoran

Tidak ada metrik baru yang dihitung di sini: semua angka diambil dari keluaran
make_report (summary.json) dan file log. Website hanya menampilkan ulang.
Model ONNX punya dua keluaran: probabilitas dan CAM per kelas. Untuk arsitektur
GAP + linear, Grad-CAM persis sebanding dengan CAM klasik (bobot = W[c] / HW), jadi
peta di browser sama dengan peta Grad-CAM di laporan (hingga normalisasi).
"""
from __future__ import annotations

# Bisa dijalankan sebagai file maupun modul (lihat run_all.py).
if not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "src"
    import src  # noqa: F401

import argparse
import base64
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .models import IMAGENET_MEAN, IMAGENET_STD, build_efficientnet
from .utils import ROOT, load_config, now_utc, read_json, resolve, sha256_file, work_dir, write_json

WEB = ROOT / "web" / "public"
CAM_RES = 28          # resolusi peta Grad-CAM yang dikirim ke web (uint8, base64)
N_EXAMPLES = 6        # contoh Grad-CAM per domain per model


class CamHead(nn.Module):
    """EfficientNet-B0 -> (probabilitas [N,K], CAM per kelas [N,K,7,7] sebelum ReLU)."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.features, self.avgpool = m.features, m.avgpool
        self.fc = m.classifier[1]          # classifier[0] = Dropout, no-op saat eval

    def forward(self, x):
        a = self.features(x)                                   # [N,1280,7,7]
        logits = self.fc(torch.flatten(self.avgpool(a), 1))
        cam = torch.einsum("kc,nchw->nkhw", self.fc.weight, a)
        return torch.softmax(logits, 1), cam


def _clean(o):
    """NaN/inf -> None supaya JSON valid untuk browser."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _b64_map(cam: np.ndarray) -> str:
    c = np.asarray(cam, dtype=np.float32)
    t = torch.from_numpy(c)[None, None]
    c = F.interpolate(t, size=(CAM_RES, CAM_RES), mode="area")[0, 0].numpy()
    mx = c.max()
    c = c / mx if mx > 0 else c
    return base64.b64encode((np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()).decode()


def _crop224(path: str, size: int = 224) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
    w, h = im.size
    r = round(size / 0.875) / min(w, h)
    im = im.resize((round(w * r), round(h * r)), Image.BILINEAR)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return im.crop((l, t, l + size, t + size))


def export_models(cfg, wd: Path) -> list[dict]:
    k = len(cfg["data"]["classes"])
    out = []
    (WEB / "models").mkdir(parents=True, exist_ok=True)
    s0 = cfg["tune_seed"]
    for name, label in (("main", "EfficientNet-B0"), ("phone", "EfficientNet-B0 + aug. ponsel")):
        rid = f"cnn_{name}_lesion_s{s0}"
        ck = wd / "runs" / rid / "best.pt"
        if not ck.exists():
            print(f"[export] lewati model {rid}: {ck} tidak ada")
            continue
        m = build_efficientnet(k, None)
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"])
        head = CamHead(m).eval()
        x = torch.randn(1, 3, 224, 224)
        dst = WEB / "models" / f"{name}.onnx"
        torch.onnx.export(head, (x,), str(dst), input_names=["input"], output_names=["probs", "cam"],
                          dynamic_axes={"input": {0: "n"}, "probs": {0: "n"}, "cam": {0: "n"}},
                          opset_version=17, dynamo=False)
        # verifikasi: keluaran ONNX == keluaran PyTorch
        import onnxruntime as ort
        sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            p_t, c_t = head(x)
            p_ref = torch.softmax(m.eval()(x), 1)
        p_o, c_o = sess.run(None, {"input": x.numpy()})
        diff = float(max(np.abs(p_o - p_t.numpy()).max(), np.abs(p_o - p_ref.numpy()).max()))
        assert diff < 1e-4, f"ONNX berbeda dari PyTorch: {diff}"
        out.append({"id": name, "run_id": rid, "label": label, "file": f"models/{name}.onnx",
                    "size_mb": round(dst.stat().st_size / 1e6, 2), "sha256": sha256_file(dst),
                    "checkpoint_sha256": sha256_file(ck), "max_abs_diff_vs_torch": diff})
        print(f"[export] {dst.name}: {out[-1]['size_mb']} MB, selisih vs PyTorch {diff:.2e}")
    return out


def export_gradcam(cfg, wd: Path) -> list[dict]:
    gdir = wd / "gradcam"
    if not gdir.exists():
        return []
    from .data import load_index
    idx = load_index(cfg).set_index("image_id")
    classes = cfg["data"]["classes"]
    media = WEB / "media" / "gradcam"
    media.mkdir(parents=True, exist_ok=True)
    pred_cache = {}
    out = []
    for f in sorted(gdir.glob("*_maps.npz")):
        rid, dom = f.name.replace("_maps.npz", "").split("__")
        z = np.load(f, allow_pickle=True)
        steps = ["original"] + [k for k in z.files if k not in ("image_id", "original")]
        # prediksi model (untuk label 'prediksi' di galeri) dari file prediksi evaluate
        pf = wd / "predictions" / f"{rid}__{dom}.npz"
        if pf.exists() and pf not in pred_cache:
            pz = np.load(pf, allow_pickle=True)
            pred_cache[pf] = dict(zip(pz["image_id"], pz["prob"]))
        preds = pred_cache.get(pf, {})
        ids = z["image_id"]
        # pilih contoh seimbang per kelas
        lab = [int(idx.loc[i, "y"]) for i in ids]
        order = sorted(range(len(ids)), key=lambda i: (sum(1 for j in range(i) if lab[j] == lab[i]), lab[i]))
        for i in order[:N_EXAMPLES]:
            iid = str(ids[i])
            fn = f"{iid}.jpg"
            if not (media / fn).exists():
                _crop224(idx.loc[iid, "cache_path"]).save(media / fn, quality=88)
            p = preds.get(iid)
            out.append({"run_id": rid, "domain": dom, "image_id": iid, "image": f"media/gradcam/{fn}",
                        "label": classes[lab[i]],
                        "pred": classes[int(np.argmax(p))] if p is not None else None,
                        "probs": [round(float(v), 4) for v in p] if p is not None else None,
                        "steps": [s.replace("_", ".") for s in steps],
                        "maps": [_b64_map(z[s][i]) for s in steps]})
    return out


def _read_csv(p: Path) -> list[dict]:
    if not p.exists() or p.stat().st_size == 0:
        return []
    return pd.read_csv(p).replace({np.nan: None}).to_dict("records")


def run(cfg: dict, skip_models: bool = False) -> Path:
    wd = work_dir(cfg)
    summ = wd / "results" / "summary.json"
    if not summ.exists():
        raise SystemExit("[export] outputs/results/summary.json belum ada. Jalankan dulu tahap "
                         "evaluate + report: python -m src.run_all --stages evaluate report")
    res = read_json(summ)
    (WEB / "data").mkdir(parents=True, exist_ok=True)

    # ---------- korupsi rinci (per jenis x severity x konfigurasi, rerata seed)
    corr_detail = []
    cp = wd / "predictions" / "corruption" / "corruption_results.csv"
    if cp.exists():
        cr = pd.read_csv(cp)
        cr["config"] = cr["run_id"].str.extract(r"^(lp_dinov2|cnn_main|cnn_phone)")[0].map(
            {"lp_dinov2": "baseline_lp", "cnn_main": "main", "cnn_phone": "phone"})
        corr_detail = (cr.groupby(["config", "corruption", "severity", "overlap_with_phone_aug"])["bal_acc"]
                         .agg(["mean", "std"]).reset_index().replace({np.nan: None}).to_dict("records"))

    # ---------- riwayat training per run
    histories = {}
    for d in sorted((wd / "runs").glob("*")):
        h = d / "history.csv"
        if h.exists():
            histories[d.name] = _read_csv(h)
    done = {d.name: read_json(d / "done.json") for d in sorted((wd / "runs").glob("*")) if (d / "done.json").exists()}

    # ---------- data
    idx = pd.read_csv(wd / "index.csv")
    data_counts = pd.crosstab(idx["label"], idx["domain"]).to_dict("index")
    orig_counts = idx.groupby(["domain", "orig_label"]).size().reset_index(name="n").to_dict("records")
    splits = {}
    for f in sorted((wd / "splits").glob("*.json")):
        splits[f.stem] = read_json(f)
    # jalur file lokal (path_a/path_b) tidak ikut dipublikasikan
    pairs = [{k_: v for k_, v in p.items() if not k_.startswith("path")}
             for p in _read_csv(wd / "leakage" / "pairs.csv")[:200]]
    media = WEB / "media"
    media.mkdir(parents=True, exist_ok=True)
    cs = wd / "leakage" / "contact_sheet.png"
    if cs.exists():
        shutil.copy(cs, media / "contact_sheet.png")

    env = {}
    ef = wd / "environment.txt"
    if ef.exists():
        for line in ef.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                break
            if ": " in line:
                k_, v = line.split(": ", 1)
                env[k_] = v

    models = [] if skip_models else export_models(cfg, wd)
    gradcam = export_gradcam(cfg, wd)
    to = wd / "TEST_OPENED.json"
    payload = {
        "meta": {
            "is_demo": bool(cfg["_smoke"]),
            "generated_utc": now_utc(),
            "title": "Generalisasi Klasifikasi Lesi Kulit dari Citra Dermatoskopi ke Citra Ponsel",
            "track": "Track D - Pergeseran Data (Distribution Shift)",
            "authors": [{"name": "Raditya Andiza Putro", "nrp": "23083010058"},
                        {"name": "Rosyidatul Kamila", "nrp": "23083010076"}],
            "classes": cfg["data"]["classes"],
            "class_names": {"MEL": "Melanoma", "NEV": "Nevus", "BCC": "Karsinoma sel basal",
                            "AK": "Keratosis aktinik"},
            "environment": env,
            "test_opened": read_json(to) if to.exists() else None,
        },
        "config": {k: cfg[k] for k in ("data", "leakage", "split", "seeds", "tune_seed", "tune_lr_grid",
                                       "rq3_cnn_seeds", "linear_probe", "cnn", "augment", "eval")},
        "summary": res,
        "corruption_detail": corr_detail,
        "runs_log": _read_csv(resolve(cfg["paths"]["run_log"])),
        "histories": histories,
        "done": done,
        "tuned": read_json(wd / "tuned.json") if (wd / "tuned.json").exists() else None,
        "data_counts": data_counts,
        "orig_label_counts": orig_counts,
        "splits": splits,
        "leakage_pairs": pairs,
        "has_contact_sheet": cs.exists(),
        "models": models,
        "gradcam_examples": gradcam,
    }
    dst = WEB / "data" / "results.json"
    dst.write_text(json.dumps(_clean(payload), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[export] {dst} ({dst.stat().st_size / 1e3:.0f} kB), {len(models)} model ONNX, "
          f"{len(gradcam)} contoh Grad-CAM. Demo: {cfg['_smoke']}")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-models", action="store_true", help="jangan ekspor ONNX (lebih cepat)")
    a = ap.parse_args()
    run(load_config(a.config, a.smoke), a.skip_models)
