"""LiteGTR: sparse participants, dense receivers.

    image -> RepViT (C2-C5) -> weighted FPN -> P2-P5 -> local path -> GFL head
                                                |
             region anchors: 56 single-point tokens from P3-P5 (fixed positions)
                                                |
                                           token mixer
                                                |
             geometry-aware writeback into every position of P2-P5 (P2 receives only)

The stride-2 stem activation is injected into P2, gated by the routing score map of the
region anchors (``models/token/detail_inject.py``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import batched_nms

from assigners.task_aligned_assigner import TaskAlignedAssigner
from losses.dfl import DistributionFocalLoss
from losses.giou import GIoULoss
from losses.qfl import QualityFocalLoss
from losses.token_consistency import TokenConsistencyLoss
from losses.token_routing import TokenRoutingLoss
from models.backbone.builder import build_backbone
from models.head.gfl_head import GFLHead
from models.neck.pyramid_projection import LocalCNNPath, PyramidProjection
from models.token.detail_inject import RoutedDetailInject
from models.token.ema_token_router import EMATokenRouter, bn_batch_stats_only, photometric_view
from models.token.geometric_writeback import MultiLevelWriteback
from models.token.token_mixer import TokenMixer
from models.token.token_selector import MultiLevelTokenSelector

LEVELS = ["P2", "P3", "P4", "P5"]
STRIDES = (4, 8, 16, 32)


class LiteGTR(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        mc = cfg["model"]
        self.num_classes = mc["num_classes"]
        self.levels = LEVELS
        self.strides = STRIDES

        # local path
        self.backbone = build_backbone(mc["backbone"], use_p2=True)
        dim = mc["neck"]["channels"]
        self.neck = PyramidProjection(self.backbone.out_channels, dim, levels=self.levels,
                                      weighted_fusion=mc["neck"].get("weighted_fusion", []))
        self.local_path = LocalCNNPath(dim, len(self.levels), strides=self.strides)

        # global path: region anchors -> token mixer -> geometry-aware writeback
        tk = mc["token"]
        self.token_levels = list(tk["levels"])
        self.selector = MultiLevelTokenSelector(dim, tk["budget"], tk["grids"], self.token_levels)
        self.mixer = TokenMixer(dim, num_heads=tk.get("num_heads", 1), mlp_ratio=tk.get("mlp_ratio", 2.0),
                                num_layers=tk.get("mixer_layers", 1), num_levels=len(self.token_levels))
        self.writeback = MultiLevelWriteback(dim, list(tk["writeback_levels"]), num_heads=tk.get("num_heads", 1))

        # routing map training (not used at inference): EMA teacher on a photometric view
        # and GT-centre supervision
        ema = tk["ema"]
        self.ema_router = EMATokenRouter(self.selector, ema["momentum"], ema.get("warmup_iters", 1000))
        pv = ema.get("photometric", {})
        self.ema_view_cfg = dict(brightness=pv.get("brightness", 0.4), contrast=pv.get("contrast", 0.4),
                                 gamma=tuple(pv.get("gamma", (0.7, 1.5))), noise=pv.get("noise", 0.03))
        rs = tk["routing_sup"]
        self.routing_loss = TokenRoutingLoss(loss_weight=rs.get("weight", 0.5),
                                             sigma_ratio=rs.get("sigma_ratio", 1 / 6),
                                             sigma_min=rs.get("sigma_min", 0.5))
        self.scorer_no_decay = tk.get("scorer_no_decay", False)

        # routed detail injection into P2 (before P2's local path)
        di = mc["detail_inject"]
        self.detail_inject = RoutedDetailInject(self.backbone.stem_mid_channels, dim,
                                                source=di.get("source", "P3"), dilate=di.get("dilate", 3))

        hd = mc["head"]
        self.head = GFLHead(self.num_classes, dim, strides=self.strides,
                            stacked_convs=hd.get("stacked_convs", 2),
                            p2_stacked_convs=hd.get("p2_stacked_convs", 1),
                            reg_max=hd.get("reg_max", 16))

        lc = cfg.get("loss", {})
        self.qfl = QualityFocalLoss(loss_weight=lc.get("qfl_weight", 1.0))
        self.box_iou = GIoULoss(loss_weight=lc.get("iou_weight", 2.0))
        self.dfl = DistributionFocalLoss(loss_weight=lc.get("dfl_weight", 0.25))
        self.token_consistency = TokenConsistencyLoss(
            temperature=lc.get("token_temperature", 1.0),
            hard_ratio=lc.get("token_hard_ratio", 1.0),
            loss_weight=lc.get("token_weight", 0.5),
        )
        self.assigner = TaskAlignedAssigner(**cfg.get("assigner", {}))
        self._last_student_maps: dict | None = None
        self._last_teacher_maps: dict | None = None

    # ------------------------------------------------------------------ core
    def extract_feats(self, images: torch.Tensor) -> list[torch.Tensor]:
        c, stem_s2 = self.backbone(images, return_stem=True)
        fmap = dict(zip(self.levels, self.neck(c)))
        # P2's local path waits for the routing map, which the detail injection needs
        for i, lv in enumerate(self.levels):
            if lv != "P2":
                fmap[lv] = self.local_path.paths[i](fmap[lv])

        tok_in = {lv: fmap[lv] for lv in self.token_levels}
        sel = self.selector(tok_in)
        self._last_student_maps = sel["score_maps"]
        self._last_teacher_maps = None
        if self.training:
            self._last_teacher_maps = self.ema_router.teacher_score_maps(self._teacher_inputs(images))
        mixed = self.mixer(sel["tokens"], sel["coords"], sel["level_ids"])

        fmap["P2"] = self.detail_inject(fmap["P2"], stem_s2, sel["score_maps"])
        fmap["P2"] = self.local_path.paths[0](fmap["P2"])
        fmap = self.writeback(fmap, mixed, sel["coords"])
        return [fmap[lv] for lv in self.levels]

    @torch.no_grad()
    def _teacher_inputs(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Token-level features of a photometrically perturbed copy of the batch, for the
        EMA teacher. One extra forward without backward; BatchNorm uses batch statistics
        and does not update its running buffers."""
        xp = photometric_view(images, **self.ema_view_cfg)
        with bn_batch_stats_only(self.backbone, self.neck, self.local_path):
            feats = self.local_path(self.neck(self.backbone(xp)))
        fmap = dict(zip(self.levels, feats))
        return {lv: fmap[lv] for lv in self.token_levels}

    def forward(self, images: torch.Tensor):
        feats = self.extract_feats(images)
        cls_scores, bbox_preds = self.head(feats)
        return cls_scores, bbox_preds, feats

    # ------------------------------------------------------------------ loss
    def loss(self, images: torch.Tensor, targets: list[dict]) -> dict:
        cls_scores, bbox_preds, feats = self(images)
        cls, reg, boxes, points, strides = self.head.decode(cls_scores, bbox_preds, feats)

        # the loss is computed in fp32 under AMP
        cls, reg, boxes = cls.float(), reg.float(), boxes.float()
        points, strides = points.float(), strides.float()

        b = cls.shape[0]
        device = cls.device

        cls_targets = torch.zeros_like(cls)
        gts: list[torch.Tensor] = []
        pos_boxes, pos_tgt_boxes, pos_reg, pos_points, pos_strides, pos_w = [], [], [], [], [], []
        num_pos = 0
        for i in range(b):
            gt_boxes = targets[i]["boxes"].to(device, non_blocking=True)
            gt_labels = targets[i]["labels"].to(device, non_blocking=True)
            gts.append(gt_boxes)
            res = self.assigner(cls[i].detach().sigmoid(), boxes[i].detach(), points,
                                gt_boxes, gt_labels,
                                point_strides=strides, reg_max=self.head.reg_max)
            idx = res["fg_mask"].nonzero(as_tuple=True)[0]
            if idx.numel():
                lab = res["assigned_labels"][idx]
                cls_targets[i, idx, lab] = res["assigned_ious"][idx]
                g = res["assigned_gt"][idx]
                pos_boxes.append(boxes[i][idx])
                pos_tgt_boxes.append(gt_boxes[g])
                pos_reg.append(reg[i][idx])
                pos_points.append(points[idx])
                pos_strides.append(strides[idx])
                pos_w.append(res["assigned_ious"][idx])
                num_pos += idx.numel()

        avg = cls_targets.sum().clamp_min(1.0)          # stays on device: no host sync
        losses = {"loss_qfl": self.qfl(cls.reshape(-1, self.num_classes),
                                       cls_targets.reshape(-1, self.num_classes), avg_factor=avg)}

        if num_pos > 0:
            pb = torch.cat(pos_boxes)
            tb = torch.cat(pos_tgt_boxes)
            pr = torch.cat(pos_reg)
            pp = torch.cat(pos_points)
            ps = torch.cat(pos_strides)
            w = torch.cat(pos_w)
            wsum = w.sum().clamp_min(1.0)
            losses["loss_giou"] = self.box_iou(pb, tb, weight=w, avg_factor=wsum)

            reg_max = self.head.reg_max
            tgt_dist = torch.stack([
                pp[:, 0] - tb[:, 0], pp[:, 1] - tb[:, 1],
                tb[:, 2] - pp[:, 0], tb[:, 3] - pp[:, 1],
            ], -1) / ps[:, None]
            tgt_dist = tgt_dist.clamp(0, reg_max - 0.01)
            losses["loss_dfl"] = self.dfl(
                pr.reshape(-1, reg_max + 1), tgt_dist.reshape(-1),
                weight=w[:, None].expand(-1, 4).reshape(-1), avg_factor=wsum * 4)
        else:
            losses["loss_giou"] = boxes.sum() * 0.0
            losses["loss_dfl"] = reg.sum() * 0.0

        if self._last_teacher_maps is not None:
            losses["loss_token"] = self.token_consistency(self._last_student_maps, self._last_teacher_maps)
        if self._last_student_maps is not None:
            stride_of = dict(zip(self.levels, self.strides))
            losses["loss_route"] = self.routing_loss(self._last_student_maps, gts, stride_of)
        return losses

    # --------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, images: torch.Tensor, score_thr: float = 0.02, nms_iou: float = 0.6,
                max_det: int = 500, pre_nms: int = 3000, agnostic: bool = False,
                containment: float | None = None, multi_label: bool = False) -> list[dict]:
        """Decoded, NMS-filtered detections per image.

        ``agnostic`` runs one NMS over all classes: on VisDrone a person is often
        predicted as both ``pedestrian`` and ``people``, and class-wise NMS keeps both.
        ``containment`` additionally drops a box when a higher-scoring kept box of the
        same class (any class if ``agnostic``) covers at least that fraction of its
        area: nested boxes on one tall object have IoU = small/large area, which falls
        below ``nms_iou`` and survives plain NMS. ``multi_label`` lets one location
        emit every class above ``score_thr`` instead of only its best one (mmyolo /
        Ultralytics evaluation). The evaluation protocol lives in the ``test:`` config
        block.
        """
        cls_scores, bbox_preds, feats = self(images)
        cls, _, boxes, _, _ = self.head.decode(cls_scores, bbox_preds, feats)
        return [postprocess(s, bx, images.shape[-2:], score_thr, nms_iou, max_det, pre_nms,
                            agnostic, containment, multi_label)
                for s, bx in zip(cls.sigmoid(), boxes)]

    # ----------------------------------------------------------------- utils
    def ema_step(self) -> None:
        self.ema_router.update(self.selector)

    def no_weight_decay(self) -> set[str]:
        """Parameter names the optimizer must not decay (``build_optimizer`` reads this).

        With ``scorer_no_decay`` the scorers' final 1x1 conv is exempt: its output is
        the routing score, and decaying it pulls every score toward the same value.
        """
        if not self.scorer_no_decay:
            return set()
        last = {id(sel.scorer.conv[-1].weight) for sel in self.selector.selectors.values()}
        return {n for n, p in self.named_parameters() if id(p) in last}

    def deploy_state_dict(self) -> dict:
        """State dict without the training-only EMA teacher."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("ema_router.")}


def postprocess(scores: torch.Tensor, boxes: torch.Tensor, hw: tuple[int, int],
                score_thr: float = 0.02, nms_iou: float = 0.6, max_det: int = 500,
                pre_nms: int = 3000, agnostic: bool = False,
                containment: float | None = None, multi_label: bool = False) -> dict:
    """One image: ``scores`` (L, C) probabilities, ``boxes`` (L, 4) xyxy -> detections."""
    h, w = hw
    if multi_label:
        loc, labels = (scores > score_thr).nonzero(as_tuple=True)
        s_max, bx = scores[loc, labels], boxes[loc].clone()
    else:
        s_max, labels = scores.max(-1)
        keep = s_max > score_thr
        s_max, labels, bx = s_max[keep], labels[keep], boxes[keep].clone()
    if s_max.numel() > pre_nms:
        topv, topi = s_max.topk(pre_nms)
        s_max, labels, bx = topv, labels[topi], bx[topi]
    bx[:, 0::2] = bx[:, 0::2].clamp(0, w)
    bx[:, 1::2] = bx[:, 1::2].clamp(0, h)
    groups = torch.zeros_like(labels) if agnostic else labels
    k = batched_nms(bx, s_max, groups, nms_iou)          # sorted by score, descending
    if containment is not None and k.numel() > 1:
        # the first ``max_det`` survivors depend only on a score-sorted prefix: grow
        # that prefix instead of building the N x N matrix over every NMS survivor
        m = min(k.numel(), 2 * max_det)
        while True:
            keep = _not_contained(bx[k[:m]], groups[k[:m]], containment)
            if int(keep.sum()) >= max_det or m == k.numel():
                break
            m = min(k.numel(), 2 * m)
        k = k[:m][keep]
    k = k[:max_det]
    return {"boxes": bx[k], "scores": s_max[k], "labels": labels[k]}


def _not_contained(bx: torch.Tensor, groups: torch.Tensor, thr: float) -> torch.Tensor:
    """Keep-mask over score-sorted boxes: drop j if an earlier box of its group covers
    >= ``thr`` of j's area. Earlier boxes that were themselves dropped still count --
    after NMS such chains are rare, and it keeps this a single vectorised pass."""
    area = ((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])).clamp_min(1e-6)
    lt = torch.max(bx[:, None, :2], bx[None, :, :2])
    rb = torch.min(bx[:, None, 2:], bx[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)                  # (N, N)
    covered = inter / area[None, :]                          # [i, j]: share of j inside i
    earlier = torch.ones_like(covered, dtype=torch.bool).triu(1)
    same = groups[:, None] == groups[None, :]
    return ~((covered >= thr) & earlier & same).any(0)
