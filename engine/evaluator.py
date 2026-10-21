"""Evaluation loop: COCO metrics in original-image pixels (``datasets/metrics.py``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets.metrics import COCOMeanAP
from utils.boxes import unletterbox
from utils.plots import ConfusionMatrix, draw_predictions, plot_pr_curves


@torch.no_grad()
def postprocess_cfg(cfg: dict) -> dict:
    """``test:`` block of a config -> keyword arguments for ``LiteGTR.predict``.

    One place, so training-time validation and tools/val.py post-process identically.
    """
    t = cfg.get("test", {}) or {}
    return {"score_thr": t.get("score_thr", 0.02), "nms_iou": t.get("nms_iou", 0.6),
            "max_det": t.get("max_det", 500), "agnostic": t.get("agnostic", False),
            "containment": t.get("containment"), "multi_label": t.get("multi_label", False),
            "pre_nms": t.get("pre_nms", 30000 if t.get("multi_label") else 3000)}


def vis_cfg(cfg: dict) -> dict:
    """``vis:`` block of a config -> keyword arguments for :func:`display_filter`.

    Drawing only; metrics never see it.
    """
    v = cfg.get("vis", {}) or {}
    return {"score_thr": v.get("score_thr", 0.3), "nms_iou": v.get("nms_iou", 0.6),
            "agnostic": v.get("agnostic", True), "containment": v.get("containment", 0.8),
            "label": v.get("label", "class"), "show_gt": v.get("show_gt", False)}


def display_filter(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                   score_thr: float = 0.3, nms_iou: float = 0.6, agnostic: bool = True,
                   containment: float | None = 0.8, **_) -> tuple[np.ndarray, ...]:
    """Reduce evaluation detections to what a figure should show: a score threshold,
    then NMS (class-agnostic by default: one box per person, not a pedestrian box
    AND a people box) and containment. Metrics never see the result."""
    from torchvision.ops import batched_nms

    from models.detector import _not_contained

    keep = scores >= score_thr
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if len(scores) < 2:
        return boxes, scores, labels
    b = torch.as_tensor(boxes, dtype=torch.float32)
    s = torch.as_tensor(scores, dtype=torch.float32)
    g = torch.zeros(len(s), dtype=torch.long) if agnostic else torch.as_tensor(labels).long()
    k = batched_nms(b, s, g, nms_iou)                     # sorted by score, descending
    if containment is not None and k.numel() > 1:
        k = k[_not_contained(b[k], g[k], containment)]
    k = k.numpy()
    return boxes[k], scores[k], labels[k]


def to_original(meta: dict, input_hw: tuple[int, int], gt_boxes: np.ndarray,
                gt_labels: np.ndarray, dt_boxes: np.ndarray):
    """``(height, width, gt_boxes, gt_labels, dt_boxes)`` in original-image pixels.

    Predictions go back through the letterbox; GT is the dataset's own annotation
    before any resizing (``meta['ori_boxes']``), so boxes that shrink below a pixel
    at the input size are still counted -- as they are in a COCO json. Samples
    without that metadata (e.g. synthetic tests) stay in input space.
    """
    if "ori_shape" not in meta or "ori_boxes" not in meta:
        return input_hw[0], input_hw[1], gt_boxes, gt_labels, dt_boxes
    ori_hw = tuple(int(v) for v in meta["ori_shape"][:2])
    assert input_hw[0] == input_hw[1], "letterbox input is square"
    return (ori_hw[0], ori_hw[1], np.asarray(meta["ori_boxes"]).reshape(-1, 4),
            np.asarray(meta["ori_labels"]).reshape(-1), unletterbox(dt_boxes, ori_hw, input_hw[0]))


def evaluate(model, loader, device, classes: list[str], score_thr: float = 0.02,
             nms_iou: float = 0.6, max_det: int = 500, amp: bool = False,
             agnostic: bool = False, containment: float | None = None,
             multi_label: bool = False, pre_nms: int = 3000, desc: str = "val",
             save_dir: str | Path | None = None, num_vis: int = 16,
             vis: dict | None = None) -> dict:
    """Returns the overall metrics.

    When ``save_dir`` is given, also writes ``confusion_matrix.png``, ``pr_curve.png``
    and ``val_predictions/``. ``vis`` (see :func:`vis_cfg`) filters what is drawn only.
    """
    vis = vis if vis is not None else vis_cfg({})
    # Metrics are computed in ORIGINAL-image pixels against the original annotations
    # (see to_original), like mmdet/mmyolo's CocoMetric; drawing and the confusion
    # matrix stay in the letterboxed input the model saw.
    model.eval()
    metric = COCOMeanAP(classes)
    cm = ConfusionMatrix(len(classes)) if save_dir else None
    vis_dir = None
    if save_dir:
        vis_dir = Path(save_dir) / "val_predictions"
        vis_dir.mkdir(parents=True, exist_ok=True)
    img_id = 0
    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            preds = model.predict(images, score_thr=score_thr, nms_iou=nms_iou, max_det=max_det,
                                  agnostic=agnostic, containment=containment,
                                  multi_label=multi_label, pre_nms=pre_nms)
        h, w = images.shape[-2:]
        for bi, (t, p) in enumerate(zip(targets, preds)):
            gt_b = t["boxes"].cpu().numpy()
            gt_l = t["labels"].cpu().numpy()
            dt_b = p["boxes"].float().cpu().numpy()
            dt_s = p["scores"].float().cpu().numpy()
            dt_l = p["labels"].cpu().numpy()
            meta = t.get("meta", {})
            mh, mw, m_gt_b, m_gt_l, m_dt_b = to_original(meta, (h, w), gt_b, gt_l, dt_b)
            metric.add(image_id=img_id, height=mh, width=mw,
                       gt_boxes=m_gt_b, gt_labels=m_gt_l,
                       dt_boxes=m_dt_b, dt_scores=dt_s, dt_labels=dt_l)
            if cm is not None:
                cm.update(dt_b, dt_s, dt_l, gt_b, gt_l)
            if vis_dir is not None and img_id < num_vis:
                img = (images[bi].float().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255)
                name = t.get("meta", {}).get("file_name", f"{img_id:06d}.jpg")
                try:
                    import cv2
                    vb, vs, vl = display_filter(dt_b, dt_s, dt_l, **vis)
                    cv2.imwrite(str(vis_dir / f"{img_id:04d}_{name}"),
                                draw_predictions(img.astype(np.uint8), vb, vs, vl, gt_b, classes,
                                                 conf=0.0, label=vis["label"],
                                                 show_gt=vis["show_gt"]))
                except Exception:
                    pass
            img_id += 1

    overall = metric.evaluate()
    if save_dir:
        cm.plot(Path(save_dir) / "confusion_matrix.png", classes)
        curves = metric.pr_curves()
        if curves:
            plot_pr_curves(curves, Path(save_dir) / "pr_curve.png")
    return overall
