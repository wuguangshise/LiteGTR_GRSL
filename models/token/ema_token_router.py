"""EMA teacher for the routing score maps (training only).

The teacher is an EMA copy of the scorers. It sees a photometric perturbation of the
batch (brightness, contrast, gamma, sensor noise) while the student sees the normal
input; a spatial-softmax KL between the two score maps (``losses/token_consistency.py``)
asks the routing map not to depend on illumination. The perturbation leaves geometry
untouched, so the maps stay pixel-aligned. The teacher is not part of the deployed model.
"""
from __future__ import annotations

import contextlib
import copy

import torch
import torch.nn as nn


class EMATokenRouter(nn.Module):
    def __init__(self, selector: nn.Module, momentum: float = 0.999, warmup_iters: int = 1000):
        super().__init__()
        self.momentum = momentum
        self.warmup_iters = warmup_iters
        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))
        self.teacher = copy.deepcopy(selector)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, selector: nn.Module) -> None:
        """Call once per optimiser step. Momentum ramps in over ``warmup_iters``."""
        self.num_updates += 1
        n = int(self.num_updates.item())
        m = min(self.momentum, (1 + n) / (10 + n)) if n < self.warmup_iters else self.momentum
        for tp, sp in zip(self.teacher.parameters(), selector.parameters()):
            tp.mul_(m).add_(sp.detach(), alpha=1 - m)
        for tb, sb in zip(self.teacher.buffers(), selector.buffers()):
            tb.copy_(sb)

    @torch.no_grad()
    def teacher_score_maps(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.teacher.eval()
        return self.teacher(feats)["score_maps"]

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()   # teacher is never in train mode
        return self


@torch.no_grad()
def photometric_view(x: torch.Tensor, brightness: float = 0.4, contrast: float = 0.4,
                     gamma: tuple[float, float] = (0.7, 1.5), noise: float = 0.03) -> torch.Tensor:
    """Per-sample illumination / sensor perturbation of a [0,1] image batch.

    Geometry is untouched, so score maps stay pixel-aligned with the input.

    * brightness  multiply by U(1-b, 1+b)
    * contrast    scale around the mean by U(1-c, 1+c)
    * gamma       x ** U(lo, hi)
    * noise       additive Gaussian, sigma ~ U(0, noise)
    """
    b = x.shape[0]
    dev, dt = x.device, x.dtype

    def u(lo: float, hi: float) -> torch.Tensor:
        return torch.empty(b, 1, 1, 1, device=dev, dtype=dt).uniform_(lo, hi)

    y = x * u(1 - brightness, 1 + brightness)
    mean = y.mean(dim=(1, 2, 3), keepdim=True)
    y = (y - mean) * u(1 - contrast, 1 + contrast) + mean
    y = y.clamp(0, 1).pow(u(gamma[0], gamma[1]))
    if noise > 0:
        y = y + torch.randn_like(y) * u(0.0, noise)
    return y.clamp(0, 1)


@contextlib.contextmanager
def bn_batch_stats_only(*modules: nn.Module):
    """Train-mode forward that normalises with batch statistics but never writes the
    BatchNorm running statistics (an in-place write there would break the student's
    pending backward). With ``track_running_stats`` off, BatchNorm uses batch statistics
    and performs no update."""
    toggled = []
    for m in modules:
        if m is None:
            continue
        for bn in m.modules():
            if isinstance(bn, nn.modules.batchnorm._BatchNorm) and bn.track_running_stats:
                bn.track_running_stats = False
                toggled.append(bn)
    try:
        yield
    finally:
        for bn in toggled:
            bn.track_running_stats = True
