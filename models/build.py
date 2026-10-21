"""Config loading and model construction."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_update(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths: str | Path) -> dict:
    """Merge YAML files left to right; ``_base_`` inside a file is resolved first."""
    cfg: dict = {}
    for p in paths:
        p = Path(p)
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for base in data.pop("_base_", []) or []:
            cfg = _deep_update(cfg, load_config((p.parent / base).resolve()))
        cfg = _deep_update(cfg, data)
    return cfg


def count_deploy_params(model) -> int:
    """Learnable parameters at inference: ``nn.Parameter``s without the training-only EMA
    teacher, with the RepViT backbone counted after re-parameterisation (``fuse()``)."""
    if hasattr(model, "deploy_param_count"):
        return model.deploy_param_count()
    n = sum(p.numel() for n_, p in model.named_parameters()
            if not n_.startswith("ema_router.") and not n_.startswith("backbone."))
    bb = model.backbone
    if hasattr(bb, "fuse") and not getattr(bb, "fused", False):
        bb = copy.deepcopy(bb).eval().fuse()
    return n + sum(p.numel() for p in bb.parameters())


def build_model(cfg: dict):
    # imported lazily so load_config() works without torch installed
    if cfg["model"].get("type", "litegtr") == "yolo_transfer":
        from models.transfer.yolo_transfer import YOLOTransfer

        return YOLOTransfer(cfg)
    from models.detector import LiteGTR

    return LiteGTR(cfg)
