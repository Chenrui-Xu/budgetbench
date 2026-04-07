"""
BudgetBench — Evaluation Visualizations
=========================================
Generates all paper figures from a list of episodes.

Figures
-------
1. success_curve.png   — Success@B vs budget B (one line per agent)
2. status_bars.png     — Status breakdown stacked bar chart
3. auc_comparison.png  — AUC comparison across agents / models
4. ttfv_histogram.png  — Time-to-First-Valid distribution

Usage
-----
::

    from budgetbench.visualize import plot_all
    plot_all(results_by_agent, output_dir="results/")

Implemented by: Chenrui Xu (Project Lead)
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from .interfaces import Episode
from .metrics import (
    auc_success_vs_budget,
    bootstrap_ci,
    mean_time_to_first_valid,
    success_at_b,
    time_to_first_valid,
)

# Consistent colour palette per agent
_PALETTE = [
    "#2563eb",  # blue   — SingleShotAgent
    "#16a34a",  # green  — LoopAgent
    "#7c3aed",  # purple — BudgetAwareAgent
    "#ea580c",  # orange — extra
    "#0891b2",  # cyan   — extra
]

_STATUS_COLOURS = {
    "OK_SOLVED":        "#16a34a",
    "TEST_FAIL":        "#ea580c",
    "INVALID_FORMAT":   "#dc2626",
    "PATCH_APPLY_FAIL": "#f97316",
    "BUDGET_EXCEEDED":  "#64748b",
}


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_success_curve(
    results: dict[str, list[Episode]],
    b_max:   int | None = None,
    output:  str = "success_curve.png",
) -> None:
    """Success@B vs budget B, one line per agent (with 95% CI band)."""
    if b_max is None:
        all_eps = [ep for eps in results.values() for ep in eps]
        b_max   = max((ep.budget_total for ep in all_eps), default=20)

    budgets = list(range(1, b_max + 1))
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (agent_name, episodes) in enumerate(results.items()):
        colour  = _PALETTE[idx % len(_PALETTE)]
        success = [success_at_b(episodes, b) for b in budgets]

        # Bootstrap CI band
        def _success_at(b):
            return [1.0 if ep.budget_used <= b and ep.solved else 0.0 for ep in episodes]

        lo_list, hi_list = [], []
        for b in budgets:
            vals    = _success_at(b)
            lo, hi  = bootstrap_ci(vals)
            lo_list.append(lo)
            hi_list.append(hi)

        ax.plot(budgets, success, label=agent_name, color=colour, linewidth=2)
        ax.fill_between(budgets, lo_list, hi_list, color=colour, alpha=0.15)

    ax.set_xlabel("Budget (tool calls)", fontsize=12)
    ax.set_ylabel("Success@B", fontsize=12)
    ax.set_title("Success Rate vs Budget", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


def plot_status_bars(
    results: dict[str, list[Episode]],
    output:  str = "status_bars.png",
) -> None:
    """Stacked bar chart showing status breakdown per agent."""
    statuses = ["OK_SOLVED", "TEST_FAIL", "PATCH_APPLY_FAIL",
                "INVALID_FORMAT", "BUDGET_EXCEEDED"]
    agent_names = list(results.keys())
    x = np.arange(len(agent_names))
    bar_width = 0.55

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(agent_names))

    for status in statuses:
        fracs = np.array([
            sum(1 for ep in results[a] if ep.final_status == status) / max(len(results[a]), 1)
            for a in agent_names
        ])
        ax.bar(x, fracs, bar_width, bottom=bottom,
               label=status, color=_STATUS_COLOURS[status])
        bottom += fracs

    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, fontsize=11)
    ax.set_ylabel("Fraction of episodes", fontsize=12)
    ax.set_title("Episode Outcome Breakdown", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


def plot_auc_comparison(
    results: dict[str, list[Episode]],
    b_max:   int | None = None,
    output:  str = "auc_comparison.png",
) -> None:
    """Bar chart of AUC (Success-vs-Budget) per agent."""
    if b_max is None:
        all_eps = [ep for eps in results.values() for ep in eps]
        b_max   = max((ep.budget_total for ep in all_eps), default=20)

    agent_names = list(results.keys())
    aucs = [auc_success_vs_budget(results[a], b_max) for a in agent_names]
    cis  = [
        bootstrap_ci([1.0 if ep.solved else 0.0 for ep in results[a]])
        for a in agent_names
    ]
    yerr_lo = [max(0.0, auc - lo) for auc, (lo, _) in zip(aucs, cis)]
    yerr_hi = [max(0.0, hi - auc) for auc, (_, hi) in zip(aucs, cis)]

    x      = np.arange(len(agent_names))
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(agent_names))]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, aucs, 0.5, color=colours,
           yerr=[yerr_lo, yerr_hi], capsize=6, error_kw={"linewidth": 2})
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, fontsize=11)
    ax.set_ylabel("AUC (Success-vs-Budget)", fontsize=12)
    ax.set_title("AUC Comparison (with 95% CI)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


def plot_ttfv_histogram(
    results: dict[str, list[Episode]],
    output:  str = "ttfv_histogram.png",
) -> None:
    """Distribution of Time-to-First-Valid per agent."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (agent_name, episodes) in enumerate(results.items()):
        ttfvs = [t for ep in episodes if (t := time_to_first_valid(ep)) is not None]
        if not ttfvs:
            continue
        colour = _PALETTE[idx % len(_PALETTE)]
        ax.hist(ttfvs, bins=15, alpha=0.6, label=agent_name, color=colour, edgecolor="white")

    ax.set_xlabel("Tool calls to first valid patch", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Time-to-First-Valid Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def plot_all(
    results:    dict[str, list[Episode]],
    output_dir: str = "results",
    b_max:      int | None = None,
) -> None:
    """Generate all four paper figures into output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    def path(name):
        return os.path.join(output_dir, name)

    plot_success_curve(results,  b_max=b_max, output=path("success_curve.png"))
    plot_status_bars(results,                 output=path("status_bars.png"))
    plot_auc_comparison(results, b_max=b_max, output=path("auc_comparison.png"))
    plot_ttfv_histogram(results,              output=path("ttfv_histogram.png"))

    print(f"\nAll figures saved to: {output_dir}/")
