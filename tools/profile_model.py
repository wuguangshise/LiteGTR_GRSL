"""Parameters and GFLOPs of a LiteGTR config, as reported in the paper.

Parameters are counted as deployed (RepViT's training branches fused, the EMA teacher of
the routing map excluded). MACs come from thop plus the attention matmuls, which thop
cannot see; GFLOPs = 2 x MACs.

    python tools/profile_model.py --config configs/litegtr.yaml
    python tools/profile_model.py --config configs/litegtr_s.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def attention_macs(model, imgsz: int) -> int:
    """MACs of the attention matmuls (q@k^T and attn@v) that thop cannot see:
    token mixer N^2*C per matmul, writeback H*W*N*C per matmul and level."""
    hw = {lv: (-(-imgsz // s), -(-imgsz // s)) for lv, s in zip(model.levels, model.strides)}
    blk = model.writeback.blocks[next(iter(model.writeback.blocks))]
    n, c = model.selector.num_tokens, blk.dh * blk.h
    total = 2 * n * n * c * len(model.mixer.layers)
    for lv in model.writeback.levels:
        total += 2 * hw[lv][0] * hw[lv][1] * n * c
    return total


def human(n: float) -> str:
    return f"{n / 1e6:.2f}M" if n < 1e9 else f"{n / 1e9:.2f}G"


def group_params(model) -> dict[str, int]:
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad and name.startswith("ema_router"):
            key = "ema_router (train-only)"
        else:
            key = name.split(".")[0]
        groups[key] = groups.get(key, 0) + p.numel()
    return groups


def run_profile(cfg_paths: list[str], imgsz: int, out: str | None, batch: int) -> None:
    import torch
    from models.build import build_model, count_deploy_params, load_config

    cfg = load_config(*cfg_paths)
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).eval()

    groups = group_params(model)
    deploy = count_deploy_params(model)
    total = sum(p.numel() for p in model.parameters())
    model.backbone.fuse()            # MACs as deployed: RepViT's training branches folded

    x = torch.randn(batch, 3, imgsz, imgsz)
    with torch.no_grad():
        cls, reg, feats = model(x)

    macs = None
    try:
        from thop import profile as thop_profile

        class _Fwd(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, t):
                return self.m(t)

        macs, _ = thop_profile(_Fwd(model), inputs=(x,), verbose=False)
        macs /= batch
    except ImportError:
        print("[warn] thop not installed -- MACs not measured (pip install thop)")

    lines = []
    lines.append(f"config          : {', '.join(cfg_paths)}")
    lines.append(f"input           : {batch}x3x{imgsz}x{imgsz}")
    lines.append(f"levels          : {model.levels}  strides={model.strides}")
    lines.append(f"tokens          : {model.selector.num_tokens} (levels {model.token_levels})")
    lines.append("")
    lines.append(f"{'module':28s}{'params':>12s}{'share':>9s}")
    lines.append("-" * 49)
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        lines.append(f"{k:28s}{v:12,d}{100.0 * v / total:8.1f}%")
    lines.append("-" * 49)
    lines.append(f"{'TOTAL (train)':28s}{total:12,d}")
    lines.append(f"{'TOTAL (deploy, no EMA)':28s}{deploy:12,d}   -> {human(deploy)}")
    if macs:
        lines.append(f"{'MACs':28s}{macs:12,.0f}   -> {human(macs)}  (~{human(2 * macs)} FLOPs)")
    lines.append("")
    lines.append("feature map shapes:")
    for lv, f in zip(model.levels, feats):
        lines.append(f"  {lv}: {tuple(f.shape)}")

    attn = attention_macs(model, imgsz)
    if macs:
        lines.append(f"{'attention matmuls (q@k, a@v)':28s}{attn:12,.0f}   -> {human(attn)}")
        lines.append(f"{'MACs incl. attention':28s}{macs + attn:12,.0f}   -> {human(macs + attn)}  "
                     f"(= {2 * (macs + attn) / 1e9:.2f} GFLOPs)")

    lines.append("")
    if macs:
        lines.append(f"MACs @{imgsz}: {human(macs)} from thop, which does not count the attention "
                     f"matmuls q@k and attn@v; they are added analytically above")

    text = "\n".join(lines)
    print(text)
    if out:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "flops_params.txt").write_text(text, encoding="utf-8")
        (d / "model_summary.txt").write_text(str(model), encoding="utf-8")
        print(f"\nwritten to {d}/flops_params.txt and model_summary.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["configs/litegtr.yaml"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_profile(a.config, a.imgsz, a.out, a.batch)


if __name__ == "__main__":
    main()
