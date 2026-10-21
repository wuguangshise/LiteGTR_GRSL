"""Backbone registry. LiteGTR uses the RepViT backbone (models/backbone/repvit.py)."""
from __future__ import annotations

import torch.nn as nn


def build_backbone(cfg: dict, use_p2: bool) -> nn.Module:
    kind = cfg.get("type", "repvit").lower()
    if kind != "repvit":
        raise ValueError(f"unknown backbone type: {kind!r} (LiteGTR uses 'repvit')")
    if cfg.get("stem", "conv") != "conv":
        raise ValueError("the RepViT backbone has the conv stem only (backbone.stem: conv)")
    from models.backbone.repvit import RepViTBackbone

    out_indices = tuple(range(4)) if use_p2 else (1, 2, 3)
    return RepViTBackbone(channels=tuple(cfg["channels"]), depths=tuple(cfg["depths"]),
                          mlp_ratio=cfg.get("mlp_ratio", 2),
                          drop_path_rate=cfg.get("drop_path_rate", 0.0), out_indices=out_indices)
