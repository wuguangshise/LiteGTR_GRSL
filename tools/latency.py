"""GPU latency with the protocol used in the paper.

PyTorch, FP16 weights and input, batch 1, 640 x 640, one GPU. The forward pass up to
decoded boxes is timed; NMS / top-k post-processing is not. RepViT (and the YOLO's
Conv-BN blocks) are fused as deployed. cudnn.benchmark on, 50 warm-up runs, then the
median of 300 runs, each closed by cuda.synchronize(). Latency does not depend on the
weights, so ``--weights`` is optional.

    python tools/latency.py --config configs/litegtr.yaml
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.checkpoint import CheckpointManager  # noqa: E402
from models.build import build_model, load_config  # noqa: E402


def deploy_fn(cfg: dict, device: torch.device, weights: str | None = None):
    """The deployed network as a function x -> decoded boxes, FP16 on ``device``."""
    m = build_model(cfg).eval()
    if weights:
        CheckpointManager.load(weights, m)
    if cfg["model"].get("type") == "yolo_transfer":
        m.yolo.fuse(verbose=False)
        return m.to(device).half().eval()                  # Detect in eval returns decoded boxes
    m.backbone = m.backbone.fuse()
    m.ema_router = None                                    # training-only teacher
    m = m.to(device).half().eval()
    return lambda x: m.head.decode(*m(x))


def measure(fn, device: torch.device, imgsz: int = 640, warmup: int = 50, iters: int = 300) -> dict:
    torch.backends.cudnn.benchmark = True
    x = torch.rand(1, 3, imgsz, imgsz, device=device, dtype=torch.half)
    ts = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize(device)
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(x)
            torch.cuda.synchronize(device)
            ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": statistics.median(ts), "fps": 1000.0 / statistics.median(ts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    device = torch.device(a.device)
    if device.type != "cuda":
        raise SystemExit("latency is measured on a GPU")
    cfg = load_config(a.config)
    cfg["model"]["num_classes"] = 10
    r = measure(deploy_fn(cfg, device, a.weights), device)
    print(f"{a.config}: {r['median_ms']:.2f} ms ({r['fps']:.0f} FPS) on {torch.cuda.get_device_name(device)}, "
          f"torch {torch.__version__}")


if __name__ == "__main__":
    main()
