"""Consistency between the EMA teacher's and the student's routing score maps (training only).

See ``models/token/ema_token_router.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenConsistencyLoss(nn.Module):
    """Spatial-softmax KL(teacher || student) per level, averaged over levels.

    ``hard_ratio`` < 1 restricts the loss to the positions the teacher is least certain
    about (1.0 = all positions).
    """

    def __init__(self, temperature: float = 1.0, hard_ratio: float = 1.0, loss_weight: float = 1.0):
        super().__init__()
        assert 0.0 < hard_ratio <= 1.0
        self.t = temperature
        self.hard_ratio = hard_ratio
        self.loss_weight = loss_weight

    def forward(self, student_maps: dict[str, torch.Tensor],
                teacher_maps: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []
        for lv, s in student_maps.items():
            # fp32: a spatial softmax over H*W positions underflows badly in half
            s = s.float()
            t = teacher_maps[lv].detach().float()
            b = s.shape[0]
            s_flat = s.reshape(b, -1) / self.t
            t_flat = t.reshape(b, -1) / self.t
            log_p_s = F.log_softmax(s_flat, dim=-1)
            p_t = F.softmax(t_flat, dim=-1)
            kl = p_t * (torch.log(p_t.clamp_min(1e-12)) - log_p_s)     # (B, HW)
            if self.hard_ratio < 1.0:
                k = max(int(kl.shape[-1] * self.hard_ratio), 1)
                kl, _ = kl.topk(k, dim=-1)
            losses.append(kl.sum(-1).mean())
        if not losses:
            return torch.zeros((), device=next(iter(student_maps.values())).device)
        return self.loss_weight * torch.stack(losses).mean()
