"""
Compare Gemini Flash vs Pro on the Knapsack benchmark.

Reads two report.json files (flash + pro) and produces an overlay plot:
  Success@B vs Budget for each agent × model.

Usage
-----
    python scripts/compare_models.py \
        --flash results/final/report.json \
        --pro   results/knapsack_pro/report.json \
        --out   results/figures/knapsack_flash_vs_pro.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


_AGENT_COLORS = {
    "SingleShot":  "#1f77b4",
    "LoopAgent":   "#ff7f0e",
    "BudgetAware": "#2ca02c",
}


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _series(report: dict, agent: str) -> tuple[list[int], list[float]]:
    if agent not in report:
        return [], []
    items = sorted(report[agent].items(), key=lambda kv: int(kv[0]))
    bs = [int(b) for b, _ in items]
    ss = [v["success_at_b_max"] for _, v in items]
    return bs, ss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flash", required=True)
    p.add_argument("--pro",   required=True)
    p.add_argument("--out",   required=True)
    p.add_argument("--title", default="Knapsack: Gemini Flash vs Pro (50 seeds)")
    args = p.parse_args()

    flash = _load(args.flash)
    pro = _load(args.pro)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for agent, color in _AGENT_COLORS.items():
        bf, sf = _series(flash, agent)
        bp, sp = _series(pro, agent)
        if bf:
            ax.plot(bf, sf, marker="o", linestyle="--", color=color,
                    label=f"{agent} (flash)", alpha=0.7)
        if bp:
            ax.plot(bp, sp, marker="s", linestyle="-", color=color,
                    label=f"{agent} (pro)", linewidth=2)

    ax.set_xlabel("Budget B (tool calls)")
    ax.set_ylabel("Success@B")
    ax.set_title(args.title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved: {args.out}")

    # Also dump a small text summary table
    txt_out = Path(args.out).with_suffix(".txt")
    lines = ["agent,budget,flash_success,pro_success,delta"]
    for agent in _AGENT_COLORS:
        bf, sf = _series(flash, agent)
        bp, sp = _series(pro, agent)
        for b in sorted(set(bf) | set(bp)):
            f_val = sf[bf.index(b)] if b in bf else None
            p_val = sp[bp.index(b)] if b in bp else None
            d = (p_val - f_val) if (f_val is not None and p_val is not None) else None
            lines.append(f"{agent},{b},{f_val},{p_val},{d}")
    txt_out.write_text("\n".join(lines))
    print(f"saved: {txt_out}")


if __name__ == "__main__":
    main()
