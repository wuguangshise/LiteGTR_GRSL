"""RepViT backbone, re-balanced for tiny objects trained from scratch.

Blocks from RepViT (Wang et al., "RepViT: Revisiting Mobile CNN From ViT Perspective",
CVPR 2024; github.com/THU-MIG/RepViT, Apache-2.0), rewritten in plain PyTorch -- no
timm, mmcv or mmdet. The downsampling layout is timm's RepViT (a block at the old
resolution, 3x3 depthwise s2, 1x1 to the new width, an FFN).

What changes against ConvNeXt-style blocks, and why it matters for 4-8 px objects:

  * 3x3 depthwise instead of 7x7. At stride 4 a 7x7 kernel spans ~28 px, so every
    block averages a 4-8 px object with a background several times its size.
  * BatchNorm instead of LayerNorm, and no layer scale. ConvNeXt's layer scale starts
    at 1e-6, so a from-scratch ConvNeXt begins with every block an identity and the
    shallow stages learn slowly. Here each residual branch ends in a BN whose weight
    starts at 0 -- the same identity start, but a BN weight moves at the pace of any
    other weight.
  * Training-time branches (3x3 dw + 1x1 dw + identity) folded into one 3x3 dw conv by
    :meth:`RepViTBackbone.fuse` -- inference is a plain conv net.

Allocation after LTDNet (Liu et al., "Exploring Lightweight Structures for Tiny Object
Detection in Remote Sensing Images", IEEE TGRS 2025), which re-balances RepViT for
AI-TOD: depths [4, 4, 2, 1] instead of RepViT's stride-16-heavy layout, and C2-C4 read
after one more block at their own resolution (LTDNet reads the pre-block of the next
downsampling; here that block closes the stage, which is the same network). Two things
of LTDNet are not taken: it freezes every BatchNorm to keep its ImageNet statistics,
which does not work from scratch, and its [64, 128, 256, 512] widths put 65% of the
parameters at stride 32. LiteGTR uses [64, 128, 192, 256], LiteGTR-S [48, 96, 144, 192].

The stem is RepViT's (3x3 s2 -> GELU -> 3x3 s2, BN), which is also LiteGTR's
overlapping conv stem; its stride-2 activation feeds routed detail injection.

Outputs strides 4/8/16/32 -> C2/C3/C4/C5.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import DropPath


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBN(nn.Sequential):
    """Conv (no bias) + BN; :meth:`fuse` returns the equivalent biased conv."""

    def __init__(self, cin: int, cout: int, k: int = 1, stride: int = 1, groups: int = 1,
                 bn_weight_init: float = 1.0):
        super().__init__()
        self.add_module("c", nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, groups=groups, bias=False))
        self.add_module("bn", nn.BatchNorm2d(cout))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.zeros_(self.bn.bias)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        c, bn = self.c, self.bn
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        m = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding,
                      groups=c.groups, bias=True, device=c.weight.device, dtype=c.weight.dtype)
        m.weight.copy_(c.weight * scale[:, None, None, None])
        m.bias.copy_(bn.bias - bn.running_mean * scale)
        return m


class RepDW(nn.Module):
    """BN(3x3 dw-BN + 1x1 dw + identity) in training; one 3x3 dw conv after :meth:`fuse`."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = ConvBN(dim, dim, 3, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x) + self.conv1(x) + x)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        conv = self.conv.fuse()
        w = conv.weight + F.pad(self.conv1.weight, [1, 1, 1, 1])
        w = w + F.pad(torch.ones_like(self.conv1.weight), [1, 1, 1, 1])     # identity
        b = conv.bias + self.conv1.bias
        bn = self.bn
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        conv.weight.copy_(w * scale[:, None, None, None])
        conv.bias.copy_(bn.bias + (b - bn.running_mean) * scale)
        return conv


class SqueezeExcite(nn.Module):
    def __init__(self, dim: int, ratio: float = 0.25):
        super().__init__()
        rd = _make_divisible(dim * ratio)
        self.fc1 = nn.Conv2d(dim, rd, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rd, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.fc2(self.act(self.fc1(x.mean((2, 3), keepdim=True)))))


class FFN(nn.Module):
    """x + 1x1 expand -> GELU -> 1x1 project; the last BN starts at 0 (identity at init)."""

    def __init__(self, dim: int, ratio: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.pw1 = ConvBN(dim, dim * ratio)
        self.act = nn.GELU()
        self.pw2 = ConvBN(dim * ratio, dim, bn_weight_init=0.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.pw2(self.act(self.pw1(x))))


class RepViTBlock(nn.Module):
    def __init__(self, dim: int, use_se: bool, mlp_ratio: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.mixer = RepDW(dim)
        self.se = SqueezeExcite(dim) if use_se else nn.Identity()
        self.ffn = FFN(dim, mlp_ratio, drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.se(self.mixer(x)))


class Downsample(nn.Module):
    """3x3 depthwise s2 (overlapping windows) -> 1x1 to the new width -> FFN."""

    def __init__(self, cin: int, cout: int, mlp_ratio: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.spatial = ConvBN(cin, cin, 3, 2, groups=cin)
        self.channel = ConvBN(cin, cout, 1)
        self.ffn = FFN(cout, mlp_ratio, drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.channel(self.spatial(x)))


class RepViTBackbone(nn.Module):
    """Stage ``i``: [Downsample if i > 0] + ``depths[i]`` blocks (SE on every other one,
    starting with the first) + [one block without SE if i < 3, the block LTDNet reads C2-C4
    from]."""

    stem_type = "conv"

    def __init__(
        self,
        channels: tuple[int, ...] = (56, 112, 192, 288),
        depths: tuple[int, ...] = (4, 4, 2, 1),
        in_ch: int = 3,
        mlp_ratio: int = 2,
        drop_path_rate: float = 0.0,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
    ):
        super().__init__()
        if len(channels) != 4 or len(depths) != 4:
            raise ValueError(f"need 4 stages, got channels={list(channels)} depths={list(depths)}")
        if channels[0] % 2:
            raise ValueError(f"channels[0] must be even (the stem's hidden width is half), got {channels[0]}")
        self.channels = list(channels)
        self.out_indices = tuple(out_indices)
        self.out_channels = [channels[i] for i in self.out_indices]
        self.out_strides = [4 * (2 ** i) for i in self.out_indices]
        # width of the stride-2 activation inside the stem (models/token/detail_inject.py)
        self.stem_mid_channels = channels[0] // 2
        self.fused = False

        self.stem1 = nn.Sequential(ConvBN(in_ch, channels[0] // 2, 3, 2), nn.GELU())
        self.stem2 = ConvBN(channels[0] // 2, channels[0], 3, 2)

        # stochastic depth over every residual FFN, shallow to deep
        n_ffn = sum(depths) + 3 + 3            # blocks + closing blocks (stages 1-3) + downsamples
        dpr = iter(torch.linspace(0, drop_path_rate, max(n_ffn, 1)).tolist())
        self.stages = nn.ModuleList()
        for i in range(4):
            layers: list[nn.Module] = []
            if i > 0:
                layers.append(Downsample(channels[i - 1], channels[i], mlp_ratio, next(dpr)))
            use_se = True
            for _ in range(depths[i]):
                layers.append(RepViTBlock(channels[i], use_se, mlp_ratio, next(dpr)))
                use_se = not use_se
            if i < 3:
                layers.append(RepViTBlock(channels[i], False, mlp_ratio, next(dpr)))
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor, return_stem: bool = False):
        """C2..C5; with ``return_stem`` also the stride-2 stem activation."""
        s2 = self.stem1(x)
        x = self.stem2(s2)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return (outs, s2) if return_stem else outs

    @torch.no_grad()
    def fuse(self) -> "RepViTBackbone":
        """Inference only: fold every BN and the RepDW branches into plain convs, in place.
        Outputs are unchanged (up to float rounding); the result can no longer be trained."""
        if self.training:
            raise RuntimeError("fuse() uses the BN running statistics: call .eval() first")

        def swap(parent: nn.Module) -> None:
            for name, child in parent.named_children():
                if isinstance(child, (ConvBN, RepDW)):
                    setattr(parent, name, child.fuse())
                else:
                    swap(child)

        swap(self)
        self.fused = True
        return self
