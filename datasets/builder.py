"""Dataset factory."""
from __future__ import annotations

from datasets.base import DetectionDataset


def build_dataset(cfg: dict, split: str, train: bool) -> DetectionDataset:
    d = cfg["data"]
    if d["name"].lower() != "visdrone":
        raise ValueError(f"unknown dataset: {d['name']!r} (this release supports 'visdrone')")
    from datasets.visdrone import VisDroneDataset

    # training paints ignored regions out; evaluation uses the untouched image
    mode = d.get("ignore_mode", "mask") if train else d.get("eval_ignore_mode", "drop")
    return VisDroneDataset(root=d["root"], split=d[f"{split}_split"], img_size=d.get("img_size", 640),
                           train=train, mosaic_prob=d.get("mosaic_prob", 0.5) if train else 0.0,
                           scale_aug=d.get("scale_aug", 0.0) if train else 0.0, ignore_mode=mode)
