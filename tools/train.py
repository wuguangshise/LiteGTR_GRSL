"""Train a model on VisDrone2019-DET (train split), validating on val.

    python tools/train.py --config configs/litegtr.yaml --data-root data/VisDrone --name litegtr

Outputs go to runs/train/<name>/ (weights/best.pt, weights/last.pt, results.csv, training.log).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.base import collate_fn  # noqa: E402
from datasets.builder import build_dataset  # noqa: E402
from engine.trainer import Trainer  # noqa: E402
from models.build import build_model, count_deploy_params, load_config  # noqa: E402


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default=None, help="run name under runs/train/")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None, help="path to a last.pt")
    ap.add_argument("--data-root", default=None, help="override data.root of the config")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    seed = a.seed if a.seed is not None else cfg.get("seed", 0)
    set_seed(seed, cfg.get("deterministic", False))

    name = a.name or f"{Path(a.config).stem}_seed{seed}"
    out_dir = Path(cfg.get("output_root", "runs/train")) / name
    if not a.resume and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {out_dir}; use a new --name or --resume")
    device = torch.device(a.device)

    train_ds = build_dataset(cfg, "train", train=True)
    val_ds = build_dataset(cfg, "val", train=False)
    nw = cfg.get("loader", {}).get("num_workers", 4)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=nw, collate_fn=collate_fn, pin_memory=pin, drop_last=True,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"].get("val_batch_size", cfg["train"]["batch_size"]),
                            shuffle=False, num_workers=nw, collate_fn=collate_fn, pin_memory=pin,
                            persistent_workers=nw > 0)

    cfg["model"]["num_classes"] = len(train_ds.classes)
    model = build_model(cfg)

    trainer = Trainer(model, train_loader, val_loader, cfg, device, out_dir, train_ds.classes)
    trainer.recorder.save_json("args.yaml", {"config": a.config, "seed": seed,
                                             "device": str(device), "resolved_config": cfg})
    if a.resume:
        trainer.resume(a.resume)
    trainer.recorder.logger.info(
        f"params: deploy={count_deploy_params(model) / 1e6:.2f}M | "
        f"train={len(train_ds)} val={len(val_ds)} images")
    best = trainer.fit()
    trainer.recorder.logger.info(f"best: {best}")


if __name__ == "__main__":
    main()
