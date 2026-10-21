"""Receiver coverage eta(s) and region coverage kappa. No model and no GPU needed.

* eta(s): share of small objects (COCO small, area < 32^2 original px) whose box, mapped into the
  letterboxed input, contains at least one receiving position of a level with stride s -- the cell
  centres ((k_x + 1/2) s, (k_y + 1/2) s); also the share of boxes with at least two positions and
  the median number of positions per box (boxes with none included). Ignored regions and the
  "others" class are excluded; a centre counts as inside when x0 <= c < x1 (half-open box).
* kappa: share of the cells of an 8 x 8 grid over the input that hold at least one region-anchor
  token, with the token budgets and grids of the given config.

    python tools/receiver_coverage.py --data-root data/VisDrone
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.build import load_config  # noqa: E402
from models.token.token_selector import RegionAnchors  # noqa: E402

STRIDES = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}


def small_boxes(root: Path, split: str, img_size: int) -> np.ndarray:
    """Boxes (x0, y0, x1, y1) of the small objects of a VisDrone split, in letterboxed input pixels."""
    from PIL import Image

    base = root / split
    boxes = []
    for ann in sorted((base / "annotations").glob("*.txt")):
        w_img, h_img = Image.open(base / "images" / f"{ann.stem}.jpg").size
        r = img_size / max(w_img, h_img)
        px, py = (img_size - round(w_img * r)) // 2, (img_size - round(h_img * r)) // 2   # as LetterBox
        for line in ann.read_text().splitlines():
            v = [x for x in line.strip().split(",") if x]
            if len(v) < 6:
                continue
            x, y, w, h, c = float(v[0]), float(v[1]), float(v[2]), float(v[3]), int(v[5])
            if c in (0, 11) or w <= 0 or h <= 0 or w * h >= 32 ** 2:     # ignored / others / not small
                continue
            boxes.append((x * r + px, y * r + py, (x + w) * r + px, (y + h) * r + py))
    return np.asarray(boxes)


def positions_in_box(boxes: np.ndarray, s: int) -> np.ndarray:
    """Number of cell centres (k + 1/2) s of a stride-s level inside each box (half-open [x0, x1))."""
    def per_axis(lo, hi):
        return np.maximum(np.ceil(hi / s - 0.5) - np.ceil(lo / s - 0.5), 0)
    return per_axis(boxes[:, 0], boxes[:, 2]) * per_axis(boxes[:, 1], boxes[:, 3])


def region_coverage(token_cfg: dict, img_size: int, cells: int = 8) -> float:
    """Share of the cells x cells grid over the input holding at least one region-anchor token."""
    hit = set()
    for lv in token_cfg["levels"]:
        hw = img_size // STRIDES[lv]
        grid = token_cfg["grids"][lv]
        k, r = token_cfg["budget"][lv] // (grid * grid), hw // grid
        for gy in range(grid):
            for gx in range(grid):
                for o in RegionAnchors.uniform_offsets(k, r, r, "cpu").tolist():
                    y, x = gy * r + o // r, gx * r + o % r
                    hit.add((int((y + 0.5) / hw * cells), int((x + 0.5) / hw * cells)))
    return len(hit) / cells ** 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/litegtr.yaml")
    ap.add_argument("--data-root", default=None, help="override data.root of the config")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()

    cfg = load_config(a.config)
    data = cfg["data"]
    root = Path(a.data_root or data["root"])
    size = data.get("img_size", 640)
    b = small_boxes(root, data[f"{a.split}_split"], size)
    print(f"{len(b)} small objects in {data[f'{a.split}_split']} at {size} input")
    for lv in ("P2", "P3", "P4"):
        s = STRIDES[lv]
        n = positions_in_box(b, s)
        print(f"  eta({s:2d}) = {np.mean(n >= 1) * 100:5.1f}%   >=2 positions {np.mean(n >= 2) * 100:5.1f}%"
              f"   median positions per box {np.median(n):.0f}   (finest receiving level {lv})")
    print(f"region-anchor coverage kappa on the 8x8 grid: {region_coverage(cfg['model']['token'], size) * 100:.1f}%")


if __name__ == "__main__":
    main()
