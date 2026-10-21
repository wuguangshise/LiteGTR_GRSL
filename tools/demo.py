"""Run a released model on images and save the drawn detections.

    python tools/demo.py --config configs/litegtr.yaml --weights weights/litegtr.pt
    python tools/demo.py --config configs/litegtr_s.yaml --weights weights/litegtr_s.pt --source my_images/
    python tools/demo.py --config configs/yolo26n_p2_litegtr.yaml --weights weights/yolo26n_p2_litegtr.pt

``--source`` is an image or a folder (default: assets/demo). Results are written to
``--out`` (default: runs/demo/<config name>/), one image per input plus ``detections.txt``
with ``file x1 y1 x2 y2 score class`` in original-image pixels. Boxes are drawn after the
same score threshold, class-agnostic NMS and containment filter the paper's figures use
(the ``vis:`` block of the config); evaluation uses the ``test:`` block instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.transforms import LetterBox  # noqa: E402
from datasets.visdrone import CLASSES  # noqa: E402
from engine.evaluator import display_filter, postprocess_cfg, vis_cfg  # noqa: E402
from models.build import build_model, load_config  # noqa: E402
from utils.boxes import unletterbox  # noqa: E402
from utils.plots import draw_predictions  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_model(cfg: dict, weights: str, device: torch.device):
    """Build the model and load the released (EMA) weights strictly."""
    cfg["model"]["num_classes"] = len(CLASSES)
    model = build_model(cfg)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # the EMA teacher of the routing map is training-only and is not shipped
    missing = [k for k in missing if not k.startswith("ema_router.")]
    if missing or unexpected:
        raise RuntimeError(f"weights do not match the config: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.to(device).eval()


@torch.no_grad()
def detect(model, img_bgr: np.ndarray, imgsz: int, device: torch.device, pp: dict):
    """One image -> (boxes in original pixels, scores, labels) with the evaluation post-processing."""
    canvas, _ = LetterBox(imgsz)(img_bgr, np.zeros((0, 4), np.float32))
    x = torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))).float().div_(255.0)
    p = model.predict(x[None].to(device), **pp)[0]
    boxes = unletterbox(p["boxes"].float().cpu().numpy(), img_bgr.shape[:2], imgsz)
    return boxes, p["scores"].float().cpu().numpy(), p["labels"].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="assets/demo", help="an image or a folder of images")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device(a.device)
    model = load_model(cfg, a.weights, device)
    imgsz = cfg.get("data", {}).get("img_size", 640)
    pp, vis = postprocess_cfg(cfg), vis_cfg(cfg)

    src = Path(a.source)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"no images found in {src}")
    out = Path(a.out or Path("runs/demo") / Path(a.config).stem)
    out.mkdir(parents=True, exist_ok=True)

    lines = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"skipped (unreadable): {f}")
            continue
        boxes, scores, labels = detect(model, img, imgsz, device, pp)
        boxes, scores, labels = display_filter(boxes, scores, labels, **vis)
        cv2.imwrite(str(out / f.name), draw_predictions(img, boxes, scores, labels, None, CLASSES,
                                                         conf=0.0, label=vis["label"]))
        lines += [f"{f.name} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f} {s:.3f} {CLASSES[int(l)]}"
                  for b, s, l in zip(boxes, scores, labels)]
        print(f"{f.name}: {len(scores)} objects")
    (out / "detections.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
