"""Grad-CAM + uji pengacakan parameter bertingkat (Adebayo dkk., NeurIPS 2018).

Aturan slide 9: Grad-CAM hanya dihitung sebagai interpretasi bila petanya
BERUBAH saat bobot diacak. Modul ini menghitung kemiripan (Spearman & SSIM)
antara peta asli dan peta setelah lapisan diacak bertahap dari kepala ke
lapisan awal. Kemiripan yang tetap tinggi = peta tidak bergantung pada bobot
terlatih = tidak layak ditafsirkan.

Metrik tambahan (heuristik): fraksi energi peta di pita tepi 15% citra. Sudut
vignette dermatoskop, penggaris, dan latar ada di tepi; energi tepi tinggi
mengisyaratkan model melihat artefak, bukan lesi. Jika masker lesi HAM
tersedia (paths.ham_mask_dir), dihitung juga fraksi energi di dalam lesi.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

from .models import gradcam_target_layer, randomization_order


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.layer = model, layer
        self.act = self.grad = None
        self.h = layer.register_forward_hook(self._fwd)

    def _fwd(self, m, i, o):
        self.act = o
        o.register_hook(lambda g: setattr(self, "grad", g))

    def remove(self):
        self.h.remove()

    def __call__(self, x: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            logits.gather(1, target[:, None]).sum().backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam.detach().cpu().numpy()
        mx = cam.reshape(len(cam), -1).max(1).reshape(-1, 1, 1)
        return np.where(mx > 0, cam / np.maximum(mx, 1e-12), 0.0)


def _reinit(module: torch.nn.Module, gen: torch.Generator):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen)))
                m.reset_parameters()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()


def _sim(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.std() == 0:
        return float("nan"), float("nan")      # peta asli datar: tidak bisa dinilai
    if b.std() == 0:
        # peta acak datar sementara peta asli tidak: peta jelas berubah -> korelasi 0
        return 0.0, float(structural_similarity(a, b, data_range=1.0))
    rho = spearmanr(a.ravel(), b.ravel()).statistic
    ss = structural_similarity(a, b, data_range=1.0)
    return float(rho), float(ss)


def border_energy(cam: np.ndarray, frac: float = 0.15) -> float:
    h, w = cam.shape
    b = max(1, int(frac * min(h, w)))
    inner = cam[b:h - b, b:w - b].sum()
    tot = cam.sum()
    return float((tot - inner) / tot) if tot > 0 else float("nan")


def randomization_test(model, x: torch.Tensor, target: torch.Tensor, seed: int = 0):
    """Kembalikan (peta asli, DataFrame kemiripan per langkah, dict peta per langkah)."""
    device = next(model.parameters()).device
    x, target = x.to(device), target.to(device)
    gc = GradCAM(model, gradcam_target_layer(model))
    orig = gc(x, target)
    gc.remove()
    rows, maps = [], {"original": orig}
    m = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed)
    for step, (name, _) in enumerate(randomization_order(m)):
        mod = dict(m.named_modules())[name]
        _reinit(mod, gen)
        gc = GradCAM(m, gradcam_target_layer(m))
        r = gc(x, target)
        gc.remove()
        maps[name] = r
        for i in range(len(x)):
            rho, ss = _sim(orig[i], r[i])
            rows.append({"step": step, "randomized_upto": name, "img": i, "spearman": rho, "ssim": ss})
    return orig, pd.DataFrame(rows), maps


def mask_energy(cam: np.ndarray, mask: np.ndarray) -> float:
    tot = cam.sum()
    return float((cam * (mask > 0)).sum() / tot) if tot > 0 else float("nan")
