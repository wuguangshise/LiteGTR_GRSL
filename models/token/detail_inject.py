"""Routed detail injection: stride-2 stem detail added to the finest level where objects are.

    S  = stride-2 stem activation
    D  = Conv1x1(SpaceToDepth(S))            lossless space-to-depth, then to the level width
    M  = MaxPool_d(Up(sigmoid(S_P3)))        the routing score map of the region anchors
    P  = P + gamma * M * D                   gamma per channel, initialised to 0

The routing map is detached, so it stays what its GT-centre supervision makes it.
``factor`` is the space-to-depth factor from the stem to the target level: 2 for P2,
4 for P3 on a detector without P2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutedDetailInject(nn.Module):
    def __init__(self, src_channels: int, dim: int, source: str = "P3", dilate: int = 3,
                 factor: int = 2):
        super().__init__()
        assert dilate % 2 == 1, "dilate must be odd so the mask stays centred"
        self.source = source
        self.dilate = dilate
        self.factor = factor
        self.proj = nn.Conv2d(factor * factor * src_channels, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def routing_mask(self, score_maps: dict[str, torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
        m = torch.sigmoid(score_maps[self.source].detach())
        if tuple(m.shape[-2:]) != tuple(size):
            m = F.interpolate(m, size=size, mode="nearest")
        if self.dilate > 1:
            m = F.max_pool2d(m, self.dilate, stride=1, padding=self.dilate // 2)
        return m

    def forward(self, feat: torch.Tensor, stem_s2: torch.Tensor,
                score_maps: dict[str, torch.Tensor]) -> torch.Tensor:
        d = self.proj(F.pixel_unshuffle(stem_s2, self.factor))
        d = d * self.routing_mask(score_maps, tuple(feat.shape[-2:])).to(d.dtype)
        return feat + self.gamma.view(1, -1, 1, 1).to(feat.dtype) * d.to(feat.dtype)
