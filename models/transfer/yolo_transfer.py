"""LiteGTR plugged into a native Ultralytics YOLO (YOLO26n-P2).

The Ultralytics network is used unchanged: backbone, PAN neck and the P2-P5 ``Detect``
head of ``yolo26n-p2.yaml`` (``ultralytics==8.4.171``), with Ultralytics' own assigner and
loss. LiteGTR's modules are inserted between the neck and ``Detect``:

    neck P3/P4/P5 --(1x1 to 64 ch)--> region anchors (56 tokens) --> token mixer
                                         |  routing map S_P3                    |
    stem (stride 2) --> routed detail injection into P2          geometry-aware writeback
                                         +------------------> P2/P3/P4/P5 --> Detect

Tokens come from P3-P5, the writeback reaches every position of P2-P5 (P2 receives
only), and the stride-2 stem activation is injected into P2 -- the same layout as the
LiteGTR detector. Every added residual starts at zero, so at initialisation the model
computes exactly the native YOLO. Data, training recipe and evaluation are the same as
for LiteGTR. Ultralytics multiplies its loss by the batch size; here it is averaged over
the batch. Evaluation uses the one-to-many head with NMS.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from losses.token_consistency import TokenConsistencyLoss
from losses.token_routing import TokenRoutingLoss
from models.token.detail_inject import RoutedDetailInject
from models.token.ema_token_router import EMATokenRouter, bn_batch_stats_only, photometric_view
from models.token.geometric_writeback import MultiLevelWriteback
from models.token.token_mixer import TokenMixer
from models.token.token_selector import MultiLevelTokenSelector

ULTRALYTICS_VERSION = "8.4.171"


def _check_ultralytics():
    try:
        import ultralytics
    except ImportError as e:  # pragma: no cover - environment
        raise ImportError("the YOLO model needs Ultralytics: "
                          f"pip install ultralytics=={ULTRALYTICS_VERSION}") from e
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        import warnings

        warnings.warn(f"this model was trained with ultralytics=={ULTRALYTICS_VERSION}, "
                      f"found {ultralytics.__version__}: the 'native' models may differ")


class YOLOTransfer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        _check_ultralytics()
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel

        self.cfg = cfg
        mc = cfg["model"]
        self.num_classes = mc["num_classes"]
        self.yolo = DetectionModel(mc["yolo"]["cfg"], nc=self.num_classes, verbose=False)
        # Ultralytics' loss reads its gains and, for YOLO26's E2ELoss, the number of epochs
        # from ``model.args``: the library defaults with our epoch count
        self.yolo.args = get_cfg(overrides={"epochs": int(cfg.get("train", {}).get("epochs", 300))})
        det = self.yolo.model[-1]
        det.end2end = False                       # one-to-many head + NMS at evaluation
        self.strides = tuple(int(s) for s in det.stride.tolist())
        self.levels = [f"P{s.bit_length() - 1}" for s in self.strides]      # P2-P5
        ch = {lv: det.cv2[i][0].conv.in_channels for i, lv in enumerate(self.levels)}
        stem_ch = self.yolo.model[0].conv.out_channels

        tk = mc["token"]
        dim = tk.get("dim", 64)
        self.token_levels = list(tk["levels"])
        # the neck is wider than 64 on P4/P5; the token path runs at 64 channels
        self.in_proj = nn.ModuleDict({lv: (nn.Identity() if ch[lv] == dim else nn.Conv2d(ch[lv], dim, 1))
                                      for lv in self.token_levels})
        self.selector = MultiLevelTokenSelector(dim, tk["budget"], tk["grids"], self.token_levels)
        self.mixer = TokenMixer(dim, num_heads=tk.get("num_heads", 1), mlp_ratio=tk.get("mlp_ratio", 2.0),
                                num_layers=tk.get("mixer_layers", 1), num_levels=len(self.token_levels))
        wb = list(tk["writeback_levels"])
        self.writeback = MultiLevelWriteback(dim, wb, num_heads=tk.get("num_heads", 1),
                                             in_channels={lv: ch[lv] for lv in wb})
        ema = tk["ema"]
        self.ema_router = EMATokenRouter(self.selector, ema["momentum"], ema.get("warmup_iters", 1000))
        pv = ema.get("photometric", {}) or {}
        self.ema_view_cfg = dict(brightness=pv.get("brightness", 0.4), contrast=pv.get("contrast", 0.4),
                                 gamma=tuple(pv.get("gamma", (0.7, 1.5))), noise=pv.get("noise", 0.03))
        rs = tk["routing_sup"]
        self.routing_loss = TokenRoutingLoss(loss_weight=rs.get("weight", 0.5),
                                             sigma_ratio=rs.get("sigma_ratio", 1 / 6),
                                             sigma_min=rs.get("sigma_min", 0.5))
        self.scorer_no_decay = tk.get("scorer_no_decay", False)

        di = mc["detail_inject"]
        self.inject_target = di.get("target", "P2")
        factor = self.strides[self.levels.index(self.inject_target)] // 2   # the stem is stride 2
        self.detail_inject = RoutedDetailInject(stem_ch, ch[self.inject_target], source=di.get("source", "P3"),
                                                dilate=di.get("dilate", 3), factor=factor)

        lc = cfg.get("loss", {})
        self.token_consistency = TokenConsistencyLoss(temperature=lc.get("token_temperature", 1.0),
                                                      hard_ratio=lc.get("token_hard_ratio", 1.0),
                                                      loss_weight=lc.get("token_weight", 0.5))
        self._last_student_maps: dict | None = None
        self._last_teacher_maps: dict | None = None
        self._epoch = 1
        self.__dict__["_criterion"] = None        # not a module: no state, rebuilt on the device

    # ------------------------------------------------------------------ core
    def _backbone_neck(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Ultralytics' own layer loop up to ``Detect``: the stride-2 stem output and the
        tensors ``Detect`` reads (P2-P5)."""
        y, x, stem = [], images, None
        for m in self.yolo.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if m.i == 0:
                stem = x
            y.append(x if m.i in self.yolo.save else None)
        return stem, [y[j] for j in self.yolo.model[-1].f]

    def token_inputs(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """What the region anchors see (``engine/evaluator.py``)."""
        _, feats = self._backbone_neck(images)
        fmap = dict(zip(self.levels, feats))
        return {lv: self.in_proj[lv](fmap[lv]) for lv in self.token_levels}

    def feature_maps(self, images: torch.Tensor) -> list[torch.Tensor]:
        """The levels as ``Detect`` receives them (after the global path)."""
        stem, feats = self._backbone_neck(images)
        fmap = dict(zip(self.levels, feats))
        tok_in = {lv: self.in_proj[lv](fmap[lv]) for lv in self.token_levels}
        sel = self.selector(tok_in)
        self._last_student_maps = sel["score_maps"]
        self._last_teacher_maps = None
        if self.training:
            self._last_teacher_maps = self.ema_router.teacher_score_maps(self._teacher_inputs(images))
        mixed = self.mixer(sel["tokens"], sel["coords"], sel["level_ids"])
        # as in LiteGTR: detail first, then the global context is written on top
        lv = self.inject_target
        fmap[lv] = self.detail_inject(fmap[lv], stem, sel["score_maps"])
        fmap = self.writeback(fmap, mixed, sel["coords"])
        return [fmap[lv] for lv in self.levels]

    @torch.no_grad()
    def _teacher_inputs(self, images):
        """The EMA teacher's view: one extra forward on a photometrically perturbed batch,
        BatchNorm on batch statistics without touching its running buffers."""
        xp = photometric_view(images, **self.ema_view_cfg)
        with bn_batch_stats_only(self.yolo):
            _, feats = self._backbone_neck(xp)
        fmap = dict(zip(self.levels, feats))
        return {lv: self.in_proj[lv](fmap[lv]) for lv in self.token_levels}

    def forward(self, images: torch.Tensor):
        """Training: Ultralytics' raw prediction dict. Eval: ``Detect``'s ``(y, preds)``."""
        return self.yolo.model[-1](self.feature_maps(images))

    # ------------------------------------------------------------------ loss
    def _criterion_fn(self):
        if self._criterion is None:
            self.__dict__["_criterion"] = self.yolo.init_criterion()
            self.set_epoch(self._epoch)
        return self._criterion

    def set_epoch(self, epoch: int) -> None:
        """Called by the trainer at the start of every (1-based) epoch. YOLO26's E2ELoss
        decays its one-to-many weight once per epoch (Ultralytics calls ``update()`` at
        each epoch end); setting it from the epoch keeps resumed runs on schedule."""
        self._epoch = int(epoch)
        c = self._criterion
        if c is not None and hasattr(c, "decay"):
            c.updates = self._epoch - 1
            c.o2m = c.decay(c.updates)
            c.o2o = max(c.total - c.o2m, 0)

    @staticmethod
    def _batch(targets: list[dict], hw: tuple[int, int], device) -> dict:
        """LiteGTR targets (xyxy input pixels) -> Ultralytics' batch dict (normalised xywh)."""
        h, w = hw
        idx, cls, box = [], [], []
        for i, t in enumerate(targets):
            b = t["boxes"].to(device, non_blocking=True).float().reshape(-1, 4)
            if b.numel() == 0:
                continue
            xywh = torch.stack([(b[:, 0] + b[:, 2]) / 2 / w, (b[:, 1] + b[:, 3]) / 2 / h,
                                (b[:, 2] - b[:, 0]) / w, (b[:, 3] - b[:, 1]) / h], -1)
            idx.append(torch.full((len(b),), float(i), device=device))
            cls.append(t["labels"].to(device, non_blocking=True).float().reshape(-1, 1))
            box.append(xywh)
        if not idx:
            return {"batch_idx": torch.zeros(0, device=device), "cls": torch.zeros(0, 1, device=device),
                    "bboxes": torch.zeros(0, 4, device=device)}
        return {"batch_idx": torch.cat(idx), "cls": torch.cat(cls), "bboxes": torch.cat(box)}

    def loss(self, images: torch.Tensor, targets: list[dict]) -> dict:
        preds = self(images)
        bs = images.shape[0]
        crit = self._criterion_fn()
        vec, _ = crit(preds, self._batch(targets, tuple(images.shape[-2:]), images.device))
        vec = vec / bs                                    # Ultralytics returns loss * batch size
        names = (crit.one2many if hasattr(crit, "one2many") else crit).loss_names
        losses = {"loss_" + n.replace("_loss", ""): vec[i] for i, n in enumerate(names)}
        if self._last_teacher_maps is not None:
            losses["loss_token"] = self.token_consistency(self._last_student_maps, self._last_teacher_maps)
        if self._last_student_maps is not None:
            gts = [t["boxes"].to(images.device, non_blocking=True) for t in targets]
            losses["loss_route"] = self.routing_loss(self._last_student_maps, gts,
                                                     dict(zip(self.levels, self.strides)))
        return losses

    # --------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, images: torch.Tensor, score_thr: float = 0.02, nms_iou: float = 0.6,
                max_det: int = 500, pre_nms: int = 3000, agnostic: bool = False,
                containment: float | None = None, multi_label: bool = False) -> list[dict]:
        """Same contract and post-processing as ``LiteGTR.predict`` (the ``test:`` block)."""
        from models.detector import postprocess

        det = self.yolo.model[-1]
        out = self(images)
        y = out[0] if isinstance(out, tuple) else out
        hw = tuple(images.shape[-2:])
        y = y.float()                                     # (B, 4 + nc, A): xywh, class probabilities
        xywh, scores = y[:, :4].transpose(1, 2), y[:, 4:4 + det.nc].transpose(1, 2)
        boxes = torch.cat([xywh[..., :2] - xywh[..., 2:] / 2, xywh[..., :2] + xywh[..., 2:] / 2], -1)
        return [postprocess(s, bx, hw, score_thr, nms_iou, max_det, pre_nms, agnostic, containment, multi_label)
                for s, bx in zip(scores, boxes)]

    # ----------------------------------------------------------------- utils
    def ema_step(self) -> None:
        self.ema_router.update(self.selector)

    def no_weight_decay(self) -> set[str]:
        if not self.scorer_no_decay:
            return set()
        last = {id(sel.scorer.conv[-1].weight) for sel in self.selector.selectors.values()}
        return {n for n, p in self.named_parameters() if id(p) in last}

    def deploy_state_dict(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("ema_router.")}

    def fuse_for_deploy(self) -> "YOLOTransfer":
        """Drop the Detect branch inference does not use (YOLO26 trains two), as Ultralytics'
        own ``Detect.fuse`` does before export. Params/FLOPs are reported on this form."""
        self.yolo.model[-1].fuse()
        return self

    def deploy_param_count(self) -> int:
        m = copy.deepcopy(self).fuse_for_deploy()
        return sum(p.numel() for n, p in m.named_parameters() if not n.startswith("ema_router."))
