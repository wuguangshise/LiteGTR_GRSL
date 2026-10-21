"""Task-Aligned Assigner (TOOD).

Alignment metric ``t = s^alpha * IoU^beta`` picks positives that are good at
BOTH classification and localisation, which matters more than usual on VisDrone
where neighbouring small objects otherwise steal each other's anchors.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from losses.giou import bbox_iou


class TaskAlignedAssigner(nn.Module):
    # Above this many (point, GT) pairs in one image the assignment runs in GT chunks
    # (_forward_chunked): the same result with a bounded peak memory.
    MAX_PAIRS = 1 << 23

    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9,
                 enforce_reg_range: bool = True, stal_size: float = 0.0):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.enforce_reg_range = enforce_reg_range
        self.stal_size = stal_size

    @torch.no_grad()
    def forward(self, pred_scores: torch.Tensor, pred_boxes: torch.Tensor,
                points: torch.Tensor, gt_boxes: torch.Tensor,
                gt_labels: torch.Tensor, point_strides: torch.Tensor | None = None,
                reg_max: int | None = None) -> dict:
        """Single image.

        pred_scores (L, C) sigmoid probabilities; pred_boxes (L, 4) xyxy;
        points (L, 2); gt_boxes (G, 4) xyxy; gt_labels (G,).

        ``point_strides`` and ``reg_max`` enable the regression-range constraint:
        DFL can only express a distance up to ``reg_max`` stride units, so a
        stride-4 point can reach just ``reg_max * 4`` px. Without this, a large
        object can be assigned to P2, its DFL target gets clamped to something it
        never asked for, and GIoU then pulls against DFL for the rest of training.
        """
        num_pts = points.shape[0]
        num_gt = gt_boxes.shape[0]
        device = points.device
        if num_gt == 0:
            return {
                "fg_mask": torch.zeros(num_pts, dtype=torch.bool, device=device),
                "assigned_gt": torch.zeros(num_pts, dtype=torch.long, device=device),
                "assigned_labels": torch.full((num_pts,), -1, dtype=torch.long, device=device),
                "assigned_ious": torch.zeros(num_pts, device=device),
            }
        if num_pts * num_gt > self.MAX_PAIRS:
            return self._forward_chunked(pred_scores, pred_boxes, points, gt_boxes, gt_labels,
                                         point_strides, reg_max)

        ious, align, mask = self._columns(pred_scores, pred_boxes, points, gt_boxes, gt_labels,
                                          point_strides, reg_max)

        # a point claimed by several GTs goes to the highest IoU among the claiming GTs
        multi = mask.sum(1) > 1
        best = (ious * mask).argmax(1)
        fix = torch.zeros_like(mask)
        fix[torch.arange(num_pts, device=device), best] = True
        mask = torch.where(multi[:, None], fix & mask, mask)

        fg = mask.any(1)
        assigned_gt = mask.float().argmax(1)
        labels = torch.full((num_pts,), -1, dtype=torch.long, device=device)
        labels[fg] = gt_labels[assigned_gt[fg]]

        # normalised alignment -> soft QFL target (TOOD eq. 8)
        align_pos = align * mask
        max_align = align_pos.max(0, keepdim=True).values             # (1, G)
        max_iou = (ious * mask).max(0, keepdim=True).values
        norm_align = align_pos / (max_align + self.eps) * max_iou
        assigned_ious = norm_align.max(1).values

        return {
            "fg_mask": fg,
            "assigned_gt": assigned_gt,
            "assigned_labels": labels,
            "assigned_ious": assigned_ious,
        }

    def _columns(self, pred_scores, pred_boxes, points, gt_boxes, gt_labels, point_strides, reg_max):
        """The per-GT part of the assignment: every column depends on its own GT only.
        Returns (ious, align, candidate mask), each (L, G)."""
        num_pts = points.shape[0]
        # candidates must lie inside the GT box
        lt = points[:, None, :] - gt_boxes[None, :, :2]
        rb = gt_boxes[None, :, 2:] - points[:, None, :]
        ltrb = torch.cat([lt, rb], -1)                               # (L, G, 4) -- regression geometry
        if self.stal_size > 0:
            # Small-Target-Aware Label Assignment (STAL, Ultralytics YOLO26): for candidate
            # selection only, a GT side shorter than ``stal_size`` is widened to it about the
            # centre, so a box narrower than the finest stride still has candidates. The
            # regression target stays the original box (``ltrb`` above).
            ctr = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
            half = ((gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp_min(self.stal_size)) * 0.5
            sel = torch.cat([points[:, None, :] - (ctr - half)[None], (ctr + half)[None] - points[:, None, :]], -1)
            inside = sel.min(-1).values > 0                          # (L, G)
        else:
            inside = ltrb.min(-1).values > 0                         # (L, G)

        if self.enforce_reg_range and point_strides is not None and reg_max is not None:
            in_range = (ltrb / point_strides[:, None, None]).max(-1).values <= reg_max
            inside = inside & in_range

        ious = bbox_iou(pred_boxes, gt_boxes)                        # (L, G)
        scores = pred_scores[:, gt_labels]                           # (L, G)
        align = scores.clamp_min(self.eps).pow(self.alpha) * ious.clamp_min(0).pow(self.beta)
        align = align * inside

        topk = min(self.topk, num_pts)
        _, cand = align.topk(topk, dim=0)                            # (topk, G)
        cand_mask = torch.zeros_like(align, dtype=torch.bool)
        cand_mask.scatter_(0, cand, True)
        mask = cand_mask & inside & (align > 0)

        return ious, align, mask

    @torch.no_grad()
    def _forward_chunked(self, pred_scores, pred_boxes, points, gt_boxes, gt_labels,
                         point_strides, reg_max) -> dict:
        """forward() for an image with many GTs, a chunk of GT columns at a time.

        The per-GT part (_columns) is computed per chunk. The cross-GT steps need only
        the (L, G) bool candidate mask, which is kept whole (1 byte a pair), plus running
        reductions: for each point the GT with the highest masked IoU (first index on a
        tie, like argmax), and, in a second pass, each GT's largest positive alignment
        and IoU. Every (L, G) float tensor exists for one chunk at a time. The outputs
        equal forward()'s (tests/test_assigner_chunked.py)."""
        num_pts, num_gt = points.shape[0], gt_boxes.shape[0]
        device = points.device
        step = max(1, self.MAX_PAIRS // num_pts)
        chunks = [slice(s, min(s + step, num_gt)) for s in range(0, num_gt, step)]
        cols = lambda c: self._columns(pred_scores, pred_boxes, points, gt_boxes[c], gt_labels[c],
                                       point_strides, reg_max)

        mask = torch.zeros(num_pts, num_gt, dtype=torch.bool, device=device)
        best_val = torch.full((num_pts,), -1.0, device=device)
        best = torch.zeros(num_pts, dtype=torch.long, device=device)
        for c in chunks:
            ious, _, m = cols(c)
            mask[:, c] = m
            v, j = (ious * m).max(1)
            upd = v > best_val                       # strictly greater: the first index wins a tie
            best_val = torch.where(upd, v.to(best_val.dtype), best_val)
            best = torch.where(upd, j + c.start, best)
        multi = mask.sum(1) > 1
        fix = torch.zeros_like(mask)
        fix[torch.arange(num_pts, device=device), best] = True
        mask = torch.where(multi[:, None], fix & mask, mask)
        del fix

        fg = mask.any(1)
        # every row now holds at most one True, so argmax over all GTs is that GT (0 if none)
        assigned_gt = torch.zeros(num_pts, dtype=torch.long, device=device)
        assigned_ious = None
        for c in chunks:
            ious, align, _ = cols(c)
            m = mask[:, c]
            assigned_gt = torch.where(m.any(1), m.float().argmax(1) + c.start, assigned_gt)
            align_pos = align * m
            max_align = align_pos.max(0, keepdim=True).values
            max_iou = (ious * m).max(0, keepdim=True).values
            row = (align_pos / (max_align + self.eps) * max_iou).max(1).values
            assigned_ious = row if assigned_ious is None else torch.maximum(assigned_ious, row)
        labels = torch.full((num_pts,), -1, dtype=torch.long, device=device)
        labels[fg] = gt_labels[assigned_gt[fg]]
        return {
            "fg_mask": fg,
            "assigned_gt": assigned_gt,
            "assigned_labels": labels,
            "assigned_ious": assigned_ious,
        }
