"""Project backbone C2-C5 to a common width and fuse them top-down (P2-P5)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import LayerNorm2d


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, groups: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWSepConv(nn.Module):
    """Depthwise separable 3x3 -- the workhorse of the local CNN path."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.dw = ConvBNAct(cin, cin, 3, groups=cin)
        self.pw = ConvBNAct(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class PyramidProjection(nn.Module):
    """``weighted_fusion`` lists the levels whose top-down merge is weighted instead of
    the plain FPN sum ``lat(C_l) + up(P_{l+1})``, which lays the coarser map on the
    finer one's detail with the same weight everywhere:

        P_l = (w1 * lat(C_l) + w2 * up(P_{l+1})) / (w1 + w2 + eps),   w = relu(.)

    fast normalised fusion (EfficientDet / BiFPN, CVPR 2020), one pair of weights per
    channel, both initialised to 1. Each channel learns how much of the coarser level
    it takes. Levels not listed (and P5, which has nothing above it) are unchanged.
    """

    def __init__(self, in_channels: list[int], out_channels: int = 64, use_fpn: bool = True,
                 levels: list[str] | None = None, weighted_fusion: list[str] | tuple = ()):
        super().__init__()
        self.levels = list(levels) if levels is not None else [f"L{i}" for i in range(len(in_channels))]
        assert len(self.levels) == len(in_channels)
        weighted = list(weighted_fusion)
        mergeable = self.levels[:-1]                 # the coarsest level has nothing to merge
        if not set(weighted) <= set(mergeable):
            raise ValueError(f"weighted_fusion takes levels {mergeable}, got {weighted}")
        if weighted and not use_fpn:
            raise ValueError("weighted_fusion needs the top-down path (use_fpn: true)")
        self.out_channels = out_channels
        self.use_fpn = use_fpn
        self.weighted = weighted
        self.lateral = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, out_channels, 1, bias=False), LayerNorm2d(out_channels))
             for c in in_channels]
        )
        if use_fpn:
            # one depthwise-separable smoothing conv per level after top-down add
            self.smooth = nn.ModuleList([DWSepConv(out_channels, out_channels) for _ in in_channels])
        # 1-D vectors, not a (2, C) tensor: 1-D parameters are exempt from weight decay
        # (engine/trainer.py), which would otherwise pull every weight towards eps
        self.fuse_w = nn.ParameterDict({f"{lv}_{k}": nn.Parameter(torch.ones(out_channels))
                                        for lv in weighted for k in ("lat", "up")})

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        outs = [lat(f) for lat, f in zip(self.lateral, feats)]
        if not self.use_fpn:
            return outs
        for i in range(len(outs) - 2, -1, -1):  # top-down: P5 -> P2
            up = F.interpolate(outs[i + 1], size=outs[i].shape[-2:], mode="nearest")
            lv = self.levels[i]
            if lv in self.weighted:
                wl = F.relu(self.fuse_w[f"{lv}_lat"]).view(1, -1, 1, 1)
                wu = F.relu(self.fuse_w[f"{lv}_up"]).view(1, -1, 1, 1)
                outs[i] = (wl * outs[i] + wu * up) / (wl + wu + 1e-4)
            else:
                outs[i] = outs[i] + up
        return [s(o) for s, o in zip(self.smooth, outs)]


class LocalCNNPath(nn.Module):
    """Per-level local modelling: edges, texture, dense-object separation.

    Kept cheap on P2 (stride 4): at 640 input P2 is 160x160 and dominates the cost.
    """

    def __init__(self, channels: int, num_levels: int, strides: tuple[int, ...] | None = None,
                 blocks_per_level: tuple[int, ...] | None = None):
        super().__init__()
        if blocks_per_level is None:
            # one block on the stride-4 level, two elsewhere
            strides = strides or tuple(4 * (2 ** i) for i in range(num_levels))
            blocks_per_level = tuple(1 if s == 4 else 2 for s in strides[:num_levels])
        assert len(blocks_per_level) == num_levels
        self.paths = nn.ModuleList(
            [nn.Sequential(*[DWSepConv(channels, channels) for _ in range(n)]) for n in blocks_per_level]
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return [p(f) for p, f in zip(self.paths, feats)]

