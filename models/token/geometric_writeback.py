"""Geometry-aware writeback: the dense receivers of the global path.

The mixed tokens are written back into every position of every target level (P2-P5
in LiteGTR; P2 receives but provides no tokens). For a target position ``p`` with
normalised coordinate ``q_p`` and token ``i`` with source centre ``c_i``::

    C(p, i) = <W_q f_p, W_k t_i> / sqrt(D)
    G(p, i) = -0.5 * [ (qx_p - cx_i)^2 / sx_i^2 + (qy_p - cy_i)^2 / sy_i^2 ]
    A(p, .) = softmax_i( C(p, i) + G(p, i) )
    f'_p    = f_p + gamma * W_o sum_i A(p, i) W_v t_i

``sigma_i = softplus(W_s t_i) + sigma_min`` is predicted per token and per axis by each
target level, so the same token can act locally on one level and broadly on another.
Coordinates and sigma share one unit (1 = the long side of the letterboxed input), and
sigma has no upper bound. ``gamma`` starts at 0, so the detector starts from its purely
local solution.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GeometricWriteback(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1, sigma_min: float = 0.02,
                 in_channels: int | None = None):
        """``in_channels``: width of the target map when it differs from the token width
        ``dim`` (a YOLO neck); queries are projected to ``dim`` and the output back."""
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.dh = dim // num_heads
        self.scale = self.dh ** -0.5
        self.sigma_min = sigma_min

        c = in_channels or dim
        self.q = nn.Conv2d(c, dim, 1, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.sigma = nn.Linear(dim, 2)             # per-token, per-axis scale
        self.proj = nn.Conv2d(dim, c, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))  # residual gate, starts closed
        nn.init.zeros_(self.sigma.weight)
        nn.init.constant_(self.sigma.bias, 0.0)

    @staticmethod
    def _grid(h: int, w: int, device, dtype) -> torch.Tensor:
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1).reshape(h * w, 2)   # (HW, 2) as (x, y)

    def forward(self, feat: torch.Tensor, tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """feat: (B,C,H,W) target level; tokens: (B,N,D); coords: (B,N,2) normalised (x,y)."""
        b, c, h, w = feat.shape
        n = tokens.shape[1]
        hw = h * w

        q = self.q(feat).reshape(b, self.h, self.dh, hw).permute(0, 1, 3, 2)   # (B,h,HW,dh)
        k = self.k(tokens).reshape(b, n, self.h, self.dh).permute(0, 2, 1, 3)  # (B,h,N,dh)
        v = self.v(tokens).reshape(b, n, self.h, self.dh).permute(0, 2, 1, 3)

        # Logits and the Gaussian term in fp32, scaled before the product: under AMP a
        # half matmul of un-normalised features can exceed fp16's range.
        with torch.autocast(device_type=feat.device.type, enabled=False):
            logits = (q.float() * self.scale) @ k.float().transpose(-2, -1)    # (B,h,HW,N)
        grid = self._grid(h, w, feat.device, torch.float32)                    # (HW,2)
        sigma = nn.functional.softplus(self.sigma(tokens)).float() + self.sigma_min
        d = grid.view(1, hw, 1, 2) - coords.float().view(b, 1, n, 2)           # (B,HW,N,2)
        g = -0.5 * ((d / sigma.view(b, 1, n, 2)) ** 2).sum(-1)                 # (B,HW,N)
        attn = (logits + g.unsqueeze(1)).softmax(dim=-1)

        out = attn.to(v.dtype) @ v                                             # (B,h,HW,dh)
        out = out.permute(0, 1, 3, 2).reshape(b, self.h * self.dh, h, w)
        return feat + self.gamma * self.proj(out)


class MultiLevelWriteback(nn.Module):
    """Writes the shared token set back into each target level (separate weights per level)."""

    def __init__(self, dim: int, levels: list[str], num_heads: int = 1,
                 in_channels: dict[str, int] | None = None):
        super().__init__()
        ch = in_channels or {}
        self.levels = levels
        self.blocks = nn.ModuleDict({lv: GeometricWriteback(dim, num_heads, in_channels=ch.get(lv))
                                     for lv in levels})

    def forward(self, feats: dict[str, torch.Tensor], tokens: torch.Tensor,
                coords: torch.Tensor) -> dict[str, torch.Tensor]:
        out = dict(feats)
        for lv in self.levels:
            out[lv] = self.blocks[lv](feats[lv], tokens, coords)
        return out
