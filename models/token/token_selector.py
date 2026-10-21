"""Region anchors: the sparse participants of the global path.

Each token level is cut into a G x G grid of regions, and every region contributes the
same fixed number of tokens taken at evenly spaced positions. A token is the single
feature at its position (no pooling, no ranking), so the tokens cover the whole image
at the resolution of their source level. Positions and counts are constants, which
keeps every shape static at export time.

A light scorer still runs on each level. Its score map does not choose tokens; it is
the routing map that gates the detail injection into P2 (``models/token/detail_inject.py``)
and is trained with GT-centre supervision and an EMA consistency term.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenScorer(nn.Module):
    """Per-level score map (B, 1, H, W): a 3x3 conv, BN, SiLU and a 1x1 conv."""

    def __init__(self, channels: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(channels // 2, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, groups=hidden if channels == hidden else 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RegionAnchors(nn.Module):
    """``num_tokens`` single-point tokens from one level: ``num_tokens / grid**2`` per region."""

    def __init__(self, channels: int, num_tokens: int, grid: int, level_id: int):
        super().__init__()
        assert num_tokens % (grid * grid) == 0, (
            f"num_tokens={num_tokens} must be divisible by grid^2={grid * grid}"
        )
        self.num_tokens = num_tokens
        self.grid = grid
        self.k = num_tokens // (grid * grid)
        self.level_id = level_id
        self.scorer = TokenScorer(channels)

    @staticmethod
    def uniform_offsets(k: int, rh: int, rw: int, device) -> torch.Tensor:
        """``k`` evenly spaced positions in an ``rh x rw`` region, as flat indices:
        an a x (k/a) lattice (a = largest divisor of k <= sqrt(k)) at cell centres,
        laid along the longer side of the region."""
        a = max(d for d in range(1, int(k ** 0.5) + 1) if k % d == 0)
        rows, cols = (a, k // a) if rh <= rw else (k // a, a)
        iy = [int((j + 0.5) * rh / rows) for j in range(rows) for _ in range(cols)]
        ix = [int((m + 0.5) * rw / cols) for _ in range(rows) for m in range(cols)]
        return torch.tensor([y * rw + x for y, x in zip(iy, ix)], device=device, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> dict:
        b, c, h, w = x.shape
        g = self.grid
        ph, pw = (-h) % g, (-w) % g
        if ph or pw:                                               # equal region sizes
            x = F.pad(x, (0, pw, 0, ph))
        hh, ww = x.shape[-2:]
        rh, rw = hh // g, ww // g

        score = self.scorer(x)                                     # (B, 1, hh, ww)

        # (B, C, hh, ww) -> (B, G*G, rh*rw, C)
        def to_regions(t: torch.Tensor) -> torch.Tensor:
            bb, cc = t.shape[0], t.shape[1]
            t = t.view(bb, cc, g, rh, g, rw).permute(0, 2, 4, 3, 5, 1)
            return t.reshape(bb, g * g, rh * rw, cc)

        pos = self.uniform_offsets(self.k, rh, rw, x.device).view(1, 1, -1).expand(b, g * g, -1)
        idx = pos.unsqueeze(-1).expand(-1, -1, -1, c)
        tokens = to_regions(x).gather(2, idx).reshape(b, self.num_tokens, c)

        # normalised (x, y) centre of every token, in [0, 1]
        iy, ix = pos // rw, pos % rw
        ridx = torch.arange(g * g, device=x.device)
        ry, rx = (ridx // g).view(1, -1, 1), (ridx % g).view(1, -1, 1)
        abs_y = (ry * rh + iy).float() + 0.5
        abs_x = (rx * rw + ix).float() + 0.5
        coords = torch.stack([abs_x / ww, abs_y / hh], dim=-1).reshape(b, self.num_tokens, 2)

        return {"tokens": tokens,                 # (B, N, C)
                "coords": coords,                 # (B, N, 2)
                "score_map": score[..., :h, :w],  # (B, 1, H, W)
                "level": self.level_id}


class MultiLevelTokenSelector(nn.Module):
    """One :class:`RegionAnchors` per token level; tokens are concatenated across levels."""

    def __init__(self, channels: int, token_budget: dict[str, int], grids: dict[str, int],
                 levels: list[str]):
        super().__init__()
        self.levels = levels
        self.selectors = nn.ModuleDict(
            {lv: RegionAnchors(channels, token_budget[lv], grids[lv], i) for i, lv in enumerate(levels)})
        self.num_tokens = sum(token_budget[lv] for lv in levels)

    def forward(self, feats: dict[str, torch.Tensor]) -> dict:
        outs = [self.selectors[lv](feats[lv]) for lv in self.levels]
        return {
            "tokens": torch.cat([o["tokens"] for o in outs], dim=1),
            "coords": torch.cat([o["coords"] for o in outs], dim=1),
            "level_ids": torch.cat(
                [torch.full((o["tokens"].shape[1],), float(o["level"]), device=o["tokens"].device)
                 for o in outs]),
            "score_maps": {lv: o["score_map"] for lv, o in zip(self.levels, outs)},
        }
