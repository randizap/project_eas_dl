"""Definisi model: EfficientNet-B0 (torchvision) dan ekstraktor fitur DINOv2 ViT-S/14 (timm)."""
from __future__ import annotations

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_efficientnet(n_classes: int, weights: str | None) -> nn.Module:
    """EfficientNet-B0 pretrained ImageNet, kepala diganti ke n_classes.

    weights=None hanya untuk uji asap (tanpa internet); eksperimen asli wajib
    memakai 'IMAGENET1K_V1' dan versi itu dicatat di weights.txt.
    """
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    w = getattr(EfficientNet_B0_Weights, weights) if weights else None
    m = efficientnet_b0(weights=w)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, n_classes)
    return m


def gradcam_target_layer(m: nn.Module) -> nn.Module:
    # Blok konvolusi terakhir (1280 kanal, 7x7 pada input 224).
    return m.features[-1]


def randomization_order(m: nn.Module) -> list[tuple[str, nn.Module]]:
    """Urutan pengacakan bertingkat (Adebayo dkk. 2018): dari kepala ke lapisan awal."""
    order = [("classifier", m.classifier)]
    for i in reversed(range(len(m.features))):
        order.append((f"features.{i}", m.features[i]))
    return order


def build_dinov2(pretrained: bool, img_size: int = 224) -> nn.Module:
    """DINOv2 ViT-S/14 dari timm, beku, mengeluarkan token CLS (384 dimensi)."""
    import timm
    # Tanpa pretrained (uji asap), inisialisasi dibuat deterministik agar fitur yang
    # di-cache dan fitur yang dihitung ulang saat uji korupsi berasal dari bobot sama.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=pretrained,
                              num_classes=0, img_size=img_size)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def extract_features(model: nn.Module, loader, device) -> torch.Tensor:
    model.to(device).eval()
    feats = []
    for x, _, _ in loader:
        f = model(x.to(device, non_blocking=True))
        feats.append(f.float().cpu())
    return torch.cat(feats)
