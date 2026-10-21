"""Training loop: AMP, cosine schedule with warmup, EMA routing step, checkpoints."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from engine.checkpoint import CheckpointManager
from engine.ema import ModelEMA
from engine.evaluator import evaluate, postprocess_cfg, vis_cfg
from engine.recorder import Recorder


def _sig(v: float, digits: int = 5) -> float:
    """Round to significant figures (small loss terms stay visible in the log)."""
    return float(f"{v:.{digits}g}")


def build_optimizer(model, cfg: dict):
    t = cfg.get("optimizer", "adamw").lower()
    lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 0.05)
    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        exempt = p.ndim <= 1 or n.endswith(".gamma") or "level_embed" in n or n in skip
        (no_decay if exempt else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    if t == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=cfg.get("betas", (0.9, 0.999)))
    if t == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=cfg.get("momentum", 0.937), nesterov=True)
    raise ValueError(f"unknown optimizer: {t}")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Linear warmup, then either plain cosine or flat-then-cosine (``flat_cosine``,
    as in DEIM): the peak learning rate is held for ``flat_epochs`` before the cosine
    decay to ``final_lr_ratio`` of the base rate."""
    epochs = cfg["epochs"]
    warmup = cfg.get("warmup_epochs", 3) * steps_per_epoch
    total = epochs * steps_per_epoch
    final = cfg.get("final_lr_ratio", 0.01)
    kind = cfg.get("scheduler", "cosine")
    if kind not in ("cosine", "flat_cosine"):
        raise ValueError(f"unknown scheduler: {kind!r} (expected 'cosine' or 'flat_cosine')")
    flat_end = max(cfg.get("flat_epochs", 0) * steps_per_epoch, warmup) if kind == "flat_cosine" else warmup

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        if step < flat_end:
            return 1.0
        p = (step - flat_end) / max(total - flat_end, 1)
        return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg: dict, device, out_dir: str | Path,
                 classes: list[str]):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.classes = classes
        tc = cfg["train"]
        self.epochs = tc["epochs"]
        self.amp = tc.get("amp", True) and device.type == "cuda"
        self.clip = tc.get("grad_clip", 10.0)
        self.optimizer = build_optimizer(self.model, tc)
        self.scheduler = build_scheduler(self.optimizer, tc, max(len(train_loader), 1))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        # weight EMA, distinct from the EMA teacher of the routing map
        ec = tc.get("model_ema", {})
        self.model_ema = (ModelEMA(self.model, decay=ec.get("decay", 0.9998),
                                   tau=ec.get("tau", 2000)) if ec.get("enabled", True) else None)
        self.recorder = Recorder(out_dir)
        self.ckpt = CheckpointManager(out_dir, monitor=tc.get("monitor", "mAP50_95"),
                                      save_period=tc.get("save_period", 0))
        self.start_epoch = 1
        # BatchNorm running statistics are written during the forward pass, before a
        # non-finite loss can be seen; train_one_epoch restores them on a skipped step.
        self._bn_buffers = [b for m in self.model.modules()
                            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                            and m.track_running_stats
                            for b in (m.running_mean, m.running_var)]

    def train_one_epoch(self, epoch: int) -> dict:
        """One pass over the training set.

        Loss statistics stay on the device and are read back every ``log_every`` steps.
        A step whose loss is not finite is skipped whole (no backward, no update, no
        EMA) and the BatchNorm statistics its forward wrote are restored; such steps
        are counted in ``nonfinite_steps``. Also returns ``_data_wait``, the seconds
        spent waiting for the DataLoader.
        """
        self.model.train()
        agg: dict[str, torch.Tensor] = {}
        nb = 0
        skipped = 0
        log_every = self.cfg["train"].get("log_every", 20)
        data_wait = 0.0
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch}/{self.epochs}", leave=False)
        it = iter(pbar)
        while True:
            t_fetch = time.perf_counter()
            try:
                images, targets = next(it)
            except StopIteration:
                break
            data_wait += time.perf_counter() - t_fetch

            images = images.to(self.device, non_blocking=True)
            bn_state = [b.clone() for b in self._bn_buffers]
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                losses = self.model.loss(images, targets)
                total = sum(losses.values())
            if not torch.isfinite(total):
                with torch.no_grad():
                    for b, saved in zip(self._bn_buffers, bn_state):
                        b.copy_(saved)
                skipped += 1
                self.scheduler.step()      # the schedule is defined in steps; keep it aligned
                continue
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total).backward()
            if self.clip:
                self.scaler.unscale_(self.optimizer)
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                agg["grad_norm"] = agg.get("grad_norm", 0.0) + gn.detach()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.model.ema_step()          # EMA teacher of the routing map
            if self.model_ema is not None:
                self.model_ema.update(self.model)

            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + v.detach()
            agg["loss"] = agg.get("loss", 0.0) + total.detach()
            nb += 1
            if nb % log_every == 0:        # the only per-step host sync
                pbar.set_postfix(loss=f"{float(total):.3f}",
                                 lr=f"{self.scheduler.get_last_lr()[0]:.2e}")
        out = {k: float(v) / max(nb, 1) for k, v in agg.items()}
        out["nonfinite_steps"] = skipped
        if skipped:
            self.recorder.logger.warning(
                f"epoch {epoch}: skipped {skipped} step(s) with a non-finite loss")
        out["_data_wait"] = data_wait
        return out

    def fit(self) -> dict:
        self.recorder.logger.info(f"training for {self.epochs} epochs on {self.device}")
        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            if hasattr(self.model, "set_epoch"):     # per-epoch loss schedule of YOLO26's E2ELoss
                self.model.set_epoch(epoch)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            tr = self.train_one_epoch(epoch)
            data_wait = tr.pop("_data_wait", 0.0)
            t_train = time.time() - t0
            row = {"epoch": epoch, **{f"train/{k}": _sig(v) for k, v in tr.items()},
                   "lr": self.scheduler.get_last_lr()[0]}
            if self.device.type == "cuda":
                # peak memory per epoch; cached blocks are released once per epoch
                gib = 1024 ** 3
                row["mem/peak_reserved_GiB"] = round(torch.cuda.max_memory_reserved(self.device) / gib, 2)
                row["mem/peak_allocated_GiB"] = round(torch.cuda.max_memory_allocated(self.device) / gib, 2)
                torch.cuda.empty_cache()

            t_val0 = time.time()
            if self._should_validate(epoch):
                eval_model = self._eval_model()
                # confusion matrix, PR curves and example predictions: final epoch only
                save_dir = self.recorder.dir if epoch == self.epochs else None
                overall = evaluate(eval_model, self.val_loader, self.device,
                                   self.classes, amp=self.amp, save_dir=save_dir,
                                   vis=vis_cfg(self.cfg), **postprocess_cfg(self.cfg))
                row.update({f"val/{k}": _sig(v) for k, v in overall.items()})
                self.ckpt.save(self.model, self.optimizer, self.scheduler,
                               epoch, overall, self.cfg, model_ema=self.model_ema,
                               scaler=self.scaler)
            else:
                self.ckpt.save(self.model, self.optimizer, self.scheduler, epoch, {}, self.cfg,
                               model_ema=self.model_ema, scaler=self.scaler)

            t_val = time.time() - t_val0
            row["time"] = round(time.time() - t0, 1)
            row["time/data"] = round(data_wait, 1)
            row["time/train"] = round(t_train - data_wait, 1)
            row["time/val"] = round(t_val, 1)
            self.recorder.log_epoch(row)
            self.recorder.logger.info(" | ".join(f"{k}={v}" for k, v in row.items()))
            self.recorder.plot_curves()

        best = self.ckpt.best_metrics
        self.recorder.save_json("best_metrics.json", best)
        return best

    def resume(self, path: str | Path) -> int:
        """Restore a run from ``last.pt`` (raw weights, weight EMA, optimiser, scheduler,
        scaler and best-so-far) and return the epoch to continue from. Loading is strict."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"], strict=True)
        epoch = int(ck.get("epoch", 0))
        if self.model_ema is not None:
            self.model_ema.load_state_dict(ck["model_ema"])
            self.model_ema.updates = int(ck.get("ema_updates", epoch * len(self.train_loader)))
        if ck.get("optimizer"):
            self.optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"):
            self.scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(ck["scaler"])
        self.ckpt.best = ck["best"]
        self.ckpt.best_metrics = ck.get("best_metrics") or {}
        self.start_epoch = epoch + 1
        self.recorder.logger.info(
            f"resumed from {path}: continuing at epoch {self.start_epoch}, "
            f"best {self.ckpt.monitor}={self.ckpt.best}, ema_updates="
            f"{self.model_ema.updates if self.model_ema is not None else '-'}")
        return self.start_epoch

    def _should_validate(self, epoch: int) -> bool:
        """Every ``val_interval`` epochs, and every epoch in the final ``val_dense_last``.
        Validation never changes the weights, only how finely best.pt is chosen."""
        tc = self.cfg["train"]
        interval = max(int(tc.get("val_interval", 1)), 1)
        dense = int(tc.get("val_dense_last", 0))
        return epoch % interval == 0 or epoch > self.epochs - dense or epoch == self.epochs

    def _eval_model(self):
        """Evaluate the EMA weights when weight-EMA is on, else the raw model."""
        return self.model if self.model_ema is None else self.model_ema.ema
