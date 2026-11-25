"""Hitung semua metrik + interval kepercayaan dari prediksi tersimpan, lalu tulis
tabel (Markdown & LaTeX) dan gambar untuk laporan.

Tidak membuka test set lagi: hanya membaca outputs/predictions/*.npz yang
ditulis evaluate.py. Aman dijalankan ulang.
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
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import ClusterBootstrap, all_metrics, bootstrap, ci, metric_fn, seed_avg
from .utils import ROOT, load_config, read_json, work_dir, write_json

# Palet kategorikal tervalidasi (3 slot pertama lolos CVD all-pairs), teks abu-abu.
COL = {"baseline_lp": "#257de0", "main": "#f47230", "phone": "#00b894"}
NAME = {"baseline_lp": "Baseline (DINOv2 + LR)", "main": "EfficientNet-B0",
        "phone": "EfficientNet-B0 + aug. ponsel"}
INK, INK2, GRID = "#191919", "#666666", "#e5e5e5"
METRICS = ["bal_acc", "macro_f1"]


def _load(pdir: Path, run_id: str, part: str):
    z = np.load(pdir / f"{run_id}__{part}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def _cfg_runs(cfg, cname: str, mode: str, seeds) -> list[str]:
    pre = {"baseline_lp": "lp_dinov2", "main": "cnn_main", "phone": "cnn_phone"}[cname]
    return [f"{pre}_{mode}_s{s}" for s in seeds]


def _stack(pdir, runs, part):
    ds = [_load(pdir, r, part) for r in runs]
    ids = ds[0]["image_id"]
    for d in ds[1:]:
        assert (d["image_id"] == ids).all(), "urutan citra antar seed berbeda"
    cluster = ds[0]["lesion_id"] if part == "test" else ds[0]["patient_id"]
    return [d["y"] for d in ds], [d["prob"] for d in ds], cluster, ids


def _fmt(m, lo, hi, sd=None, pct=True):
    f = 100 if pct else 1
    s = f"{m * f:.1f} [{lo * f:.1f}, {hi * f:.1f}]"
    return s + (f" (sd {sd * f:.1f})" if sd is not None else "")


def summarize(cfg) -> dict:
    wd = work_dir(cfg)
    pdir = wd / "predictions"
    classes = cfg["data"]["classes"]
    k = len(classes)
    B, lvl = cfg["eval"]["bootstrap_B"], cfg["eval"]["ci"]
    seeds = cfg["seeds"]
    res = {"classes": classes, "B": B, "ci_level": lvl, "configs": {}}
    boots = {}

    # ---------- RQ1 & RQ2: ID vs OOD, split per lesi
    for cname in ("baseline_lp", "main", "phone"):
        runs = _cfg_runs(cfg, cname, "lesion", seeds)
        res["configs"][cname] = {}
        for part in ("test", "ood"):
            ys, ps, cl, _ = _stack(pdir, runs, part)
            cb = ClusterBootstrap(cl, B, seed=1 if part == "test" else 2)  # resample sama untuk semua konfigurasi
            entry = {"per_seed": [all_metrics(y, p, classes, cfg["eval"]["ece_bins"]) for y, p in zip(ys, ps)],
                     "n_images": int(len(ys[0])), "n_clusters": int(len(np.unique(cl)))}
            for mname in METRICS + [f"recall_{c}" for c in range(k)] + ["ece"]:
                fn = metric_fn(mname, k)
                reps = bootstrap(fn, ys, ps, cb)
                boots[(cname, part, mname)] = reps
                per = [fn(y, p) for y, p in zip(ys, ps)]
                lo, hi = ci(reps, lvl)
                label = mname if not mname.startswith("recall_") else f"recall_{classes[int(mname[7:])]}"
                entry[label] = {"mean": float(np.mean(per)), "sd_seeds": float(np.std(per, ddof=1)) if len(per) > 1 else 0.0,
                                "ci": [lo, hi]}
            res["configs"][cname][part] = entry
        # celah ID - OOD (dua test set independen -> selisih replikasi independen)
        res["configs"][cname]["gap"] = {}
        for mname in METRICS:
            d = boots[(cname, "test", mname)] - boots[(cname, "ood", mname)]
            pt = res["configs"][cname]["test"][mname]["mean"] - res["configs"][cname]["ood"][mname]["mean"]
            res["configs"][cname]["gap"][mname] = {"mean": pt, "ci": list(ci(d, lvl))}

    # ---------- RQ2: intervensi - utama (berpasangan: resample & seed sama)
    res["paired_phone_minus_main"] = {}
    for part in ("test", "ood"):
        res["paired_phone_minus_main"][part] = {}
        for mname in METRICS + [f"recall_{c}" for c in range(k)]:
            d = boots[("phone", part, mname)] - boots[("main", part, mname)]
            key = mname if not mname.startswith("recall_") else f"recall_{classes[int(mname[7:])]}"
            pt = (res["configs"]["phone"][part][key]["mean"] - res["configs"]["main"][part][key]["mean"])
            lo, hi = ci(d, lvl)
            res["paired_phone_minus_main"][part][key] = {"mean": pt, "ci": [lo, hi],
                                                         "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    # selisih celah (apakah intervensi mempersempit celah?)
    for mname in METRICS:
        gp = boots[("phone", "test", mname)] - boots[("phone", "ood", mname)]
        gm = boots[("main", "test", mname)] - boots[("main", "ood", mname)]
        d = gp - gm
        res["paired_phone_minus_main"][f"gap_change_{mname}"] = {
            "mean": res["configs"]["phone"]["gap"][mname]["mean"] - res["configs"]["main"]["gap"][mname]["mean"],
            "ci": list(ci(d, lvl))}

    # ---------- RQ3: split per citra vs per lesi
    res["rq3"] = {}
    for cname, sd in (("baseline_lp", seeds), ("main", cfg["rq3_cnn_seeds"])):
        out = {}
        for mode in ("image", "lesion"):
            ys, ps, cl, ids = _stack(pdir, _cfg_runs(cfg, cname, mode, sd), "test")
            cb = ClusterBootstrap(cl, B, seed=3)   # klaster = lesi, juga untuk test split per citra
            for mname in METRICS:
                reps = bootstrap(metric_fn(mname, k), ys, ps, cb)
                out[(mode, mname)] = reps
                res["rq3"].setdefault(cname, {}).setdefault(mode, {})[mname] = {
                    "mean": seed_avg(metric_fn(mname, k), ys, ps), "ci": list(ci(reps, lvl))}
            if mode == "image":
                # mekanisme: citra test yang lesinya juga ada di train vs yang tidak
                sp = pd.read_csv(wd / "splits" / f"image_seed{sd[0]}.csv")
                tr_les = set(sp[(sp.split == "train") & (sp.domain == "ham")].lesion_id)
                les = pd.Series(sp.set_index("image_id").loc[ids, "lesion_id"].to_numpy())
                seen = les.isin(tr_les).to_numpy()
                fn = metric_fn("bal_acc", k)
                res["rq3"][cname]["image"]["bal_acc_lesion_seen_in_train"] = {
                    "n": int(seen.sum()), "value": float(fn(ys[0][seen], ps[0][seen])) if seen.any() else None}
                res["rq3"][cname]["image"]["bal_acc_lesion_unseen"] = {
                    "n": int((~seen).sum()), "value": float(fn(ys[0][~seen], ps[0][~seen])) if (~seen).any() else None}
        for mname in METRICS:
            d = out[("image", mname)] - out[("lesion", mname)]
            res["rq3"][cname][f"inflation_{mname}"] = {
                "mean": res["rq3"][cname]["image"][mname]["mean"] - res["rq3"][cname]["lesion"][mname]["mean"],
                "ci": list(ci(d, lvl)),
                "note": "dua test set berbeda; bootstrap independen per lesi"}
        # model split-per-citra pada OOD (apakah skor 'bocor' membantu di lapangan?)
        ys, ps, cl, _ = _stack(pdir, _cfg_runs(cfg, cname, "image", sd), "ood")
        res["rq3"][cname]["image_ood_bal_acc"] = seed_avg(metric_fn("bal_acc", k), ys, ps)

    # ---------- Korupsi
    cp = pdir / "corruption" / "corruption_results.csv"
    if cp.exists():
        cr = pd.read_csv(cp)
        cr["config"] = cr["run_id"].str.extract(r"^(lp_dinov2|cnn_main|cnn_phone)")[0].map(
            {"lp_dinov2": "baseline_lp", "cnn_main": "main", "cnn_phone": "phone"})
        clean = cr[cr.corruption == "clean"].set_index("run_id")["bal_acc"]
        cr = cr[cr.corruption != "clean"].copy()
        cr["drop_vs_clean"] = cr["bal_acc"] - cr["run_id"].map(clean)
        agg = (cr.groupby(["config", "overlap_with_phone_aug", "severity"])[["bal_acc", "drop_vs_clean"]]
                 .mean().reset_index())
        res["corruption"] = {
            "clean_bal_acc": cr.assign(c=cr.run_id.map(clean)).groupby("config")["c"].mean().to_dict(),
            "mean_by_group": cr.groupby(["config", "overlap_with_phone_aug"])["bal_acc"].mean()
                               .unstack().rename(columns={True: "overlap", False: "non_overlap"}).to_dict("index"),
            "note": "rerata atas korupsi x severity x seed; tanpa CI (lihat Keterbatasan)"}
        (wd / "results").mkdir(exist_ok=True)
        agg.to_csv(wd / "results" / "corruption_by_severity.csv", index=False)
        res["_corruption_agg"] = agg.to_dict("records")

    # ---------- Grad-CAM
    gdir = wd / "gradcam"
    if (gdir / "randomization_similarity.csv").exists():
        sim = pd.read_csv(gdir / "randomization_similarity.csv")
        res["gradcam_randomization"] = (sim.groupby(["run_id", "domain", "step", "randomized_upto"])
                                          [["spearman", "ssim"]].mean().reset_index().to_dict("records"))
        en = pd.read_csv(gdir / "cam_energy.csv")
        cols = [c for c in ("border_energy", "lesion_energy") if c in en.columns]
        res["gradcam_energy"] = en.groupby(["run_id", "domain"])[cols].mean().reset_index().to_dict("records")

    # ---------- Data & kebocoran
    lr = wd / "leakage" / "report.json"
    if lr.exists():
        res["leakage"] = read_json(lr)
    res["split_counts"] = read_json(wd / "splits" / f"lesion_seed{seeds[0]}.json")
    return res


# ====================================================================== keluaran
def write_tables(cfg, res, outdir: Path, texdir: Path):
    classes = res["classes"]
    md = ["# Ringkasan hasil (otomatis dari outputs/predictions)\n",
          f"Format: rerata 3 seed [CI {int(res['ci_level'] * 100)}% bootstrap klaster, B={res['B']}] (sd antar seed). Satuan: %.\n",
          "## RQ1/RQ2 - in-distribution (HAM test) vs out-of-distribution (PAD-UFES-20)\n",
          "| Konfigurasi | BalAcc ID | BalAcc OOD | Celah BalAcc | Macro-F1 ID | Macro-F1 OOD | Celah Macro-F1 |",
          "|---|---|---|---|---|---|---|"]
    tex = []
    for c, d in res["configs"].items():
        t, o, g = d["test"], d["ood"], d["gap"]
        md.append(f"| {NAME[c]} | {_fmt(t['bal_acc']['mean'], *t['bal_acc']['ci'], t['bal_acc']['sd_seeds'])} | "
                  f"{_fmt(o['bal_acc']['mean'], *o['bal_acc']['ci'], o['bal_acc']['sd_seeds'])} | "
                  f"{_fmt(g['bal_acc']['mean'], *g['bal_acc']['ci'])} | "
                  f"{_fmt(t['macro_f1']['mean'], *t['macro_f1']['ci'])} | {_fmt(o['macro_f1']['mean'], *o['macro_f1']['ci'])} | "
                  f"{_fmt(g['macro_f1']['mean'], *g['macro_f1']['ci'])} |")
        tex.append(f"{NAME[c].replace('+', '+')} & {t['bal_acc']['mean']*100:.1f} [{t['bal_acc']['ci'][0]*100:.1f}, {t['bal_acc']['ci'][1]*100:.1f}] & "
                   f"{o['bal_acc']['mean']*100:.1f} [{o['bal_acc']['ci'][0]*100:.1f}, {o['bal_acc']['ci'][1]*100:.1f}] & "
                   f"{g['bal_acc']['mean']*100:.1f} [{g['bal_acc']['ci'][0]*100:.1f}, {g['bal_acc']['ci'][1]*100:.1f}] & "
                   f"{t['macro_f1']['mean']*100:.1f} & {o['macro_f1']['mean']*100:.1f} \\\\")
    (texdir / "tab_id_ood.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    md += ["\n## Recall per kelas (rerata seed [CI])\n",
           "| Konfigurasi | Domain | " + " | ".join(classes) + " | ECE |", "|---|---|" + "---|" * (len(classes) + 1)]
    tex = []
    for c, d in res["configs"].items():
        for part, lab in (("test", "ID"), ("ood", "OOD")):
            e = d[part]
            cells = [_fmt(e[f"recall_{cl}"]["mean"], *e[f"recall_{cl}"]["ci"]) for cl in classes]
            md.append(f"| {NAME[c]} | {lab} | " + " | ".join(cells) + f" | {e['ece']['mean']:.3f} |")
            tex.append(f"{NAME[c]} & {lab} & " + " & ".join(f"{e[f'recall_{cl}']['mean']*100:.1f}" for cl in classes)
                       + f" & {e['ece']['mean']:.3f} \\\\")
    (texdir / "tab_recall.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    md += ["\n## RQ2 - intervensi dikurangi model utama (berpasangan)\n", "| Metrik | Domain | Selisih [CI] | CI tidak memuat 0? |", "|---|---|---|---|"]
    tex = []
    for part, lab in (("test", "ID"), ("ood", "OOD")):
        for key, v in res["paired_phone_minus_main"][part].items():
            md.append(f"| {key} | {lab} | {_fmt(v['mean'], *v['ci'])} | {'ya' if v['ci_excludes_zero'] else 'tidak'} |")
            if key in METRICS or key == "recall_MEL":
                tex.append(f"{key.replace('_', '-')} & {lab} & {v['mean']*100:+.1f} [{v['ci'][0]*100:+.1f}, {v['ci'][1]*100:+.1f}] \\\\")
    for m in METRICS:
        v = res["paired_phone_minus_main"][f"gap_change_{m}"]
        md.append(f"| perubahan celah {m} | ID-OOD | {_fmt(v['mean'], *v['ci'])} | - |")
        tex.append(f"perubahan celah {m.replace('_', '-')} & ID$-$OOD & {v['mean']*100:+.1f} [{v['ci'][0]*100:+.1f}, {v['ci'][1]*100:+.1f}] \\\\")
    (texdir / "tab_paired.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    md += ["\n## RQ3 - split acak per citra vs per lesi (test in-distribution)\n",
           "| Konfigurasi | BalAcc split citra | BalAcc split lesi | Inflasi [CI] | BalAcc OOD (model split citra) |", "|---|---|---|---|---|"]
    tex = []
    for c, d in res["rq3"].items():
        im, le, inf = d["image"]["bal_acc"], d["lesion"]["bal_acc"], d["inflation_bal_acc"]
        md.append(f"| {NAME[c]} | {_fmt(im['mean'], *im['ci'])} | {_fmt(le['mean'], *le['ci'])} | "
                  f"{_fmt(inf['mean'], *inf['ci'])} | {d['image_ood_bal_acc']*100:.1f} |")
        s, u = d["image"]["bal_acc_lesion_seen_in_train"], d["image"]["bal_acc_lesion_unseen"]
        md.append(f"|  - rincian seed pertama | lesi ada di train: {s['value'] if s['value'] is None else round(s['value']*100, 1)} (n={s['n']}) | "
                  f"lesi tidak ada di train: {u['value'] if u['value'] is None else round(u['value']*100, 1)} (n={u['n']}) | | |")
        tex.append(f"{NAME[c]} & {im['mean']*100:.1f} & {le['mean']*100:.1f} & {inf['mean']*100:+.1f} "
                   f"[{inf['ci'][0]*100:+.1f}, {inf['ci'][1]*100:+.1f}] & {d['image_ood_bal_acc']*100:.1f} \\\\")
    (texdir / "tab_rq3.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    if "corruption" in res:
        md += ["\n## Uji korupsi (HAM test, balanced accuracy %, rerata korupsi x severity x seed)\n",
               "| Konfigurasi | Bersih | Korupsi non-overlap | Korupsi overlap dgn aug. ponsel |", "|---|---|---|---|"]
        tex = []
        for c, g in res["corruption"]["mean_by_group"].items():
            cl = res["corruption"]["clean_bal_acc"][c]
            md.append(f"| {NAME[c]} | {cl*100:.1f} | {g.get('non_overlap', np.nan)*100:.1f} | {g.get('overlap', np.nan)*100:.1f} |")
            tex.append(f"{NAME[c]} & {cl*100:.1f} & {g.get('non_overlap', np.nan)*100:.1f} & {g.get('overlap', np.nan)*100:.1f} \\\\")
        (texdir / "tab_corruption.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    if "gradcam_randomization" in res:
        g = pd.DataFrame(res["gradcam_randomization"])
        last = g[g.step == g.step.max()]
        first = g[g.step == 0]
        md += ["\n## Grad-CAM - uji pengacakan parameter (Spearman rerata vs peta asli)\n",
               "| Run | Domain | Kepala diacak | Semua lapisan diacak |", "|---|---|---|---|"]
        tex = []
        for (rid, dom), _ in g.groupby(["run_id", "domain"]):
            a = first[(first.run_id == rid) & (first.domain == dom)].spearman.iloc[0]
            b = last[(last.run_id == rid) & (last.domain == dom)].spearman.iloc[0]
            md.append(f"| {rid} | {dom} | {a:.3f} | {b:.3f} |")
            rid_tex = rid.replace("_", r"\_")
            dom_tex = "OOD" if dom == "ood" else "ID"
            tex.append(f"{rid_tex} & {dom_tex} & {a:.3f} & {b:.3f} \\\\")
        (texdir / "tab_gradcam.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        e = pd.DataFrame(res["gradcam_energy"])
        md += ["\n### Energi Grad-CAM (heuristik)\n", "```", e.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "```"]

    if "leakage" in res:
        md += ["\n## Pemeriksaan kebocoran\n", "```json", json.dumps(res["leakage"], indent=2), "```"]
    md += ["\n## Jumlah citra per split (split per lesi, seed pertama)\n", "```json",
           json.dumps(res["split_counts"], indent=2), "```"]
    (outdir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def make_figures(cfg, res, figdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
    cfgs = list(res["configs"])

    # Gambar 1: ID vs OOD, balanced accuracy + CI
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    for ax, m, title in zip(axes, METRICS, ["Balanced accuracy", "Macro-F1"]):
        w = 0.26
        for i, c in enumerate(cfgs):
            for j, part in enumerate(("test", "ood")):
                v = res["configs"][c][part][m]
                x = j + (i - 1) * (w + 0.02)
                ax.bar(x, v["mean"] * 100, w, color=COL[c], label=NAME[c] if j == 0 else None)
                ax.errorbar(x, v["mean"] * 100, yerr=[[100 * (v["mean"] - v["ci"][0])], [100 * (v["ci"][1] - v["mean"])]],
                            color=INK, lw=1, capsize=2)
        ax.set_xticks([0, 1], ["HAM10000 test (ID)", "PAD-UFES-20 (OOD)"])
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("%")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(figdir / "fig_id_vs_ood.png", dpi=200)
    plt.close(fig)

    # Gambar 2: confusion matrix seed pertama (model utama & intervensi, OOD)
    classes = res["classes"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    panels = [(c, p) for c in ("main", "phone") for p in ("test", "ood")]
    for ax, (c, p) in zip(axes, panels):
        cm = np.array(res["configs"][c][p]["per_seed"][0]["confusion"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True).clip(1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for (a, b), v in np.ndenumerate(cmn):
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else INK)
        ax.set_xticks(range(len(classes)), classes, fontsize=6); ax.set_yticks(range(len(classes)), classes, fontsize=6)
        ax.set_title(f"{'Utama' if c == 'main' else 'Intervensi'} - {'ID' if p == 'test' else 'OOD'}", fontsize=7)
        ax.grid(False)
    axes[0].set_ylabel("label benar", fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "fig_confusion.png", dpi=200)
    plt.close(fig)

    # Gambar 3: korupsi per severity
    if "_corruption_agg" in res:
        agg = pd.DataFrame(res["_corruption_agg"])
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
        for ax, ov, title in zip(axes, (False, True), ("Korupsi non-overlap", "Korupsi overlap dgn aug. ponsel")):
            for c in cfgs:
                s = agg[(agg.config == c) & (agg.overlap_with_phone_aug == ov)].sort_values("severity")
                ax.plot(s.severity, s.bal_acc * 100, "-o", color=COL[c], lw=2, ms=4, label=NAME[c])
            ax.set_title(title, fontsize=9, color=INK); ax.set_xlabel("severity")
        axes[0].set_ylabel("balanced acc. (%)")
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / "fig_corruption.png", dpi=200)
        plt.close(fig)

    # Gambar 4: uji pengacakan Grad-CAM
    if "gradcam_randomization" in res:
        g = pd.DataFrame(res["gradcam_randomization"])
        fig, ax = plt.subplots(figsize=(6.8, 2.4))
        styles = {"test": "-", "ood": "--"}
        for (rid, dom), s in g.groupby(["run_id", "domain"]):
            c = "phone" if "phone" in rid else "main"
            ax.plot(s.step, s.spearman, styles[dom], marker="o", ms=3, lw=2, color=COL[c],
                    label=f"{NAME[c]} - {'ID' if dom == 'test' else 'OOD'}")
        names = g.drop_duplicates("step").sort_values("step").randomized_upto.tolist()
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=6)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel("Spearman vs peta asli"); ax.set_xlabel("diacak bertingkat sampai lapisan")
        ax.legend(frameon=False, fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(figdir / "fig_gradcam_randomization.png", dpi=200)
        plt.close(fig)
        _gradcam_examples(cfg, figdir, plt)


def _gradcam_examples(cfg, figdir, plt):
    wd = work_dir(cfg)
    from .data import load_index
    idx = load_index(cfg).set_index("image_id")
    rid = f"cnn_main_lesion_s{cfg['tune_seed']}"
    from PIL import Image
    rows = []
    for dom in ("test", "ood"):
        z = np.load(wd / "gradcam" / f"{rid}__{dom}_maps.npz", allow_pickle=True)
        keys = [k for k in z.files if k != "image_id"]
        last = [k for k in keys if k.startswith("features_")][-1] if any(k.startswith("features_") for k in keys) else keys[-1]
        for i in range(min(3, len(z["image_id"]))):
            rows.append((z["image_id"][i], z["original"][i], z[last][i], dom))
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(4.2, 1.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (iid, cam, rnd, dom) in enumerate(rows):
        with Image.open(idx.loc[iid, "cache_path"]) as im:
            im = im.convert("RGB")
            w, h = im.size; s = 224 / min(w, h) * (256 / 224)
            im = im.resize((round(w * s), round(h * s)))
            w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2
            a = np.asarray(im.crop((l, t, l + 224, t + 224)))
        for c, (title, ov) in enumerate([("citra", None), ("Grad-CAM", cam), ("bobot diacak", rnd)]):
            ax = axes[r, c]; ax.imshow(a); ax.axis("off")
            if ov is not None:
                ax.imshow(ov, cmap="magma", alpha=0.5, vmin=0, vmax=1)
            if r == 0:
                ax.set_title(title, fontsize=7)
        axes[r, 0].text(-8, 112, "ID" if dom == "test" else "OOD", fontsize=7, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(figdir / "fig_gradcam_examples.png", dpi=200)
    plt.close(fig)


def run(cfg):
    wd = work_dir(cfg)
    outdir = wd / "results"; outdir.mkdir(exist_ok=True)
    texdir = (ROOT / "laporan" / "tables") if not cfg["_smoke"] else (wd / "laporan_tables")
    figdir = (ROOT / "laporan" / "figures") if not cfg["_smoke"] else (wd / "laporan_figures")
    texdir.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    res = summarize(cfg)
    write_tables(cfg, res, outdir, texdir)
    make_figures(cfg, res, figdir)
    res.pop("_corruption_agg", None)
    write_json(outdir / "summary.json", res)
    print(f"[report] {outdir / 'results.md'}\n[report] tabel LaTeX -> {texdir}\n[report] gambar -> {figdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(load_config(a.config, a.smoke))
