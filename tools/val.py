"""Evaluate a checkpoint on VisDrone2019-DET val (COCO metrics in original-image pixels).

    python tools/val.py --config configs/litegtr.yaml --weights weights/litegtr.pt --data-root data/VisDrone

``--save-dir`` also writes the confusion matrix, PR curves and example predictions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.base import collate_fn  # noqa: E402
from datasets.builder import build_dataset  # noqa: E402
from engine.checkpoint import CheckpointManager  # noqa: E402
from engine.evaluator import evaluate, postprocess_cfg, vis_cfg  # noqa: E402
from models.build import build_model, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default=None, help="override data.root of the config")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-dir", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    ds = build_dataset(cfg, a.split, train=False)
    cfg["model"]["num_classes"] = len(ds.classes)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate_fn)
    device = torch.device(a.device)
    model = build_model(cfg)
    CheckpointManager.load(a.weights, model, map_location=device)
    model.to(device)

    metrics = evaluate(model, loader, device, ds.classes, desc=a.split, save_dir=a.save_dir,
                       vis=vis_cfg(cfg), **postprocess_cfg(cfg))
    for k, v in metrics.items():
        print(f"  {k:12s} {v:.4f}" if isinstance(v, float) else f"  {k:12s} {v}")


if __name__ == "__main__":
    main()
