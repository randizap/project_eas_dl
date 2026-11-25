"""Metrik dan bootstrap klaster.

Bootstrap dilakukan per KLASTER (lesi untuk HAM, pasien untuk PAD), bukan per
citra, karena citra dari lesi/pasien yang sama tidak independen. Bootstrap per
citra akan menghasilkan interval yang terlalu sempit.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def per_class_recall(y: np.ndarray, yp: np.ndarray, k: int) -> np.ndarray:
    r = np.full(k, np.nan)
    for c in range(k):
        m = y == c
        if m.any():
            r[c] = (yp[m] == c).mean()
    return r


def balanced_accuracy(y, yp, k):
    # Rerata recall per kelas; kelas yang tidak muncul di sampel diabaikan.
    return float(np.nanmean(per_class_recall(y, yp, k)))


def macro_f1(y, yp, k):
    present = np.unique(y)
    return float(f1_score(y, yp, labels=present, average="macro", zero_division=0))


def ece(y, prob, n_bins=15):
    conf = prob.max(1)
    pred = prob.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def all_metrics(y, prob, classes: Sequence[str], n_bins=15) -> dict:
    k = len(classes)
    yp = prob.argmax(1)
    rec = per_class_recall(y, yp, k)
    out = {"bal_acc": balanced_accuracy(y, yp, k), "macro_f1": macro_f1(y, yp, k),
           "acc": float((yp == y).mean()), "ece": ece(y, prob, n_bins), "n": int(len(y))}
    for c, r in zip(classes, rec):
        out[f"recall_{c}"] = float(r)
    out["confusion"] = confusion_matrix(y, yp, labels=list(range(k))).tolist()
    return out


# ------------------------------------------------------------------ bootstrap
class ClusterBootstrap:
    """Menghasilkan indeks resample berbasis klaster, dipakai bersama untuk perbandingan berpasangan."""

    def __init__(self, clusters: np.ndarray, B: int, seed: int = 12345):
        clusters = np.asarray(clusters)
        uniq, inv = np.unique(clusters, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
        self.rows = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
        rng = np.random.default_rng(seed)
        self.draws = rng.integers(0, len(uniq), size=(B, len(uniq)))
        self.B = B

    def indices(self, b: int) -> np.ndarray:
        return np.concatenate([self.rows[c] for c in self.draws[b]])


def seed_avg(fn: Callable, ys: list, probs: list, idx=None) -> float:
    """Rerata metrik atas seed. ys/probs: list per seed, sampel sejajar (citra yang sama)."""
    vals = []
    for y, p in zip(ys, probs):
        if idx is not None:
            y, p = y[idx], p[idx]
        vals.append(fn(y, p))
    return float(np.mean(vals))


def bootstrap(fn: Callable, ys: list, probs: list, cb: ClusterBootstrap) -> np.ndarray:
    return np.array([seed_avg(fn, ys, probs, cb.indices(b)) for b in range(cb.B)])


def ci(reps: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    reps = reps[~np.isnan(reps)]
    return float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))


def metric_fn(name: str, k: int) -> Callable:
    if name == "bal_acc":
        return lambda y, p: balanced_accuracy(y, p.argmax(1), k)
    if name == "macro_f1":
        return lambda y, p: macro_f1(y, p.argmax(1), k)
    if name.startswith("recall_"):
        c = int(name.split("_", 1)[1])
        return lambda y, p: per_class_recall(y, p.argmax(1), k)[c]
    if name == "ece":
        return lambda y, p: ece(y, p)
    raise ValueError(name)
