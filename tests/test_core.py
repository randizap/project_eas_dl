"""Uji unit cepat: python -m pytest -q tests  (tanpa GPU, tanpa dataset)."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from src import corruptions as C
from src.metrics import ClusterBootstrap, balanced_accuracy, ece, macro_f1


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y, yp = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    assert np.isclose(balanced_accuracy(y, yp, 4), balanced_accuracy_score(y, yp))
    assert np.isclose(macro_f1(y, yp, 4), f1_score(y, yp, average="macro"))


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    assert ece(y, np.eye(2)[y]) == 0.0                       # benar & yakin penuh
    assert np.isclose(ece(y, np.eye(2)[1 - y]), 1.0)          # salah & yakin penuh


def test_cluster_bootstrap_keeps_clusters_whole():
    cl = np.array(["a", "a", "b", "c", "c", "c"])
    cb = ClusterBootstrap(cl, B=50, seed=0)
    for b in range(50):
        idx = cb.indices(b)
        for c in np.unique(cl[idx]):
            n_in = (cl[idx] == c).sum()
            assert n_in % (cl == c).sum() == 0                 # klaster ikut utuh, kelipatan ukuran aslinya


def test_corruptions_shape_dtype_and_monotone():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    for name in C.CORRUPTIONS:
        prev = 0.0
        for s in (1, 3, 5):
            out = C.apply(name, img, s, seed=1)
            assert out.shape == img.shape and out.dtype == np.uint8
            diff = np.abs(out.astype(float) - img).mean()
            if name not in ("jpeg_compression",):            # JPEG tidak dijamin monoton per piksel
                assert diff >= prev - 1.0, name
            prev = diff
    assert (C.apply("gaussian_noise", img, 3, 7) == C.apply("gaussian_noise", img, 3, 7)).all()  # deterministik
