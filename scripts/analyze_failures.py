"""
BudgetBench — Failure-Mode Taxonomy Analyzer
=============================================
Reads a `episodes.csv` produced by `batch_experiment.py` and breaks down
episode outcomes by status code, per (agent x budget).

This is the deliverable promised in the proposal as a "failure-mode taxonomy":
we categorise every non-solved episode into one of the reproducible status
codes (INVALID_FORMAT, PATCH_APPLY_FAIL, TEST_FAIL, BUDGET_EXCEEDED) and
produce a CSV + stacked bar chart + text summary.

Usage
-----
    python scripts/analyze_failures.py --input results/final/episodes.csv
    python scripts/analyze_failures.py --input results/swe_final/episodes.csv \\
        --output-dir results/swe_final/failures

Outputs
-------
    <output-dir>/failure_breakdown.csv  # counts + percentages per (agent, budget)
    <output-dir>/failure_modes.png      # stacked bar plot
    <output-dir>/failure_summary.txt    # top-2 failure modes per agent
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

STATUS_CODES = [
    "OK_SOLVED",
    "INVALID_FORMAT",
    "PATCH_APPLY_FAIL",
    "TEST_FAIL",
    "BUDGET_EXCEEDED",
]

STATUS_COLORS = {
    "OK_SOLVED":        "#2ca02c",
    "INVALID_FORMAT":   "#9467bd",
    "PATCH_APPLY_FAIL": "#ff7f0e",
    "TEST_FAIL":        "#d62728",
    "BUDGET_EXCEEDED":  "#7f7f7f",
}


def load_episodes(path: str) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"agent", "budget_total", "final_status"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return rows


def build_breakdown(rows: list[dict]) -> dict:
    """
    breakdown[(agent, budget)][status] = count
    """
    counts: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = (r["agent"], int(r["budget_total"]))
        status = r["final_status"]
        counts[key][status] += 1
    return counts


def write_csv(counts: dict, path: str) -> None:
    header = ["agent", "budget_total", "n_episodes"]
    for s in STATUS_CODES:
        header += [f"n_{s.lower()}", f"pct_{s.lower()}"]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (agent, budget) in sorted(counts.keys()):
            per_status = counts[(agent, budget)]
            total = sum(per_status.values())
            row = [agent, budget, total]
            for s in STATUS_CODES:
                n = per_status.get(s, 0)
                pct = 100.0 * n / total if total else 0.0
                row += [n, f"{pct:.1f}"]
            w.writerow(row)
    print(f"Saved breakdown -> {path}")


def write_plot(counts: dict, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    keys = sorted(counts.keys())  # [(agent, budget), ...]
    labels = [f"{a}\nB={b}" for (a, b) in keys]
    totals = [sum(counts[k].values()) for k in keys]

    bottoms = [0.0] * len(keys)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(keys)), 5))
    for s in STATUS_CODES:
        heights = []
        for i, k in enumerate(keys):
            n = counts[k].get(s, 0)
            pct = 100.0 * n / totals[i] if totals[i] else 0.0
            heights.append(pct)
        ax.bar(labels, heights, bottom=bottoms,
               label=s, color=STATUS_COLORS[s], edgecolor="white", linewidth=0.5)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_ylabel("Episodes (%)")
    ax.set_title("Failure-Mode Taxonomy by Agent x Budget")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved plot -> {path}")


def write_summary(counts: dict, path: str) -> None:
    # Aggregate across budgets for each agent
    by_agent: dict = defaultdict(lambda: defaultdict(int))
    for (agent, _budget), per_status in counts.items():
        for s, n in per_status.items():
            by_agent[agent][s] += n

    lines = ["BudgetBench — Failure-Mode Summary", "=" * 40, ""]
    for agent in sorted(by_agent.keys()):
        per_status = by_agent[agent]
        total = sum(per_status.values())
        lines.append(f"{agent}  (n={total})")
        solved = per_status.get("OK_SOLVED", 0)
        lines.append(f"  OK_SOLVED     : {solved:4d}  ({100.0*solved/total:.1f}%)")
        failures = [(s, n) for s, n in per_status.items() if s != "OK_SOLVED"]
        failures.sort(key=lambda x: x[1], reverse=True)
        top = failures[:2]
        lines.append("  Top failure modes:")
        if not top:
            lines.append("    (none)")
        else:
            for s, n in top:
                pct = 100.0 * n / total
                lines.append(f"    {s:18s} {n:4d}  ({pct:.1f}%)")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved summary -> {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BudgetBench failure-mode analyzer")
    p.add_argument("--input", type=str, default="results/final/episodes.csv",
                   help="Path to episodes.csv (default: results/final/episodes.csv)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory for outputs (default: same dir as --input)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    rows = load_episodes(args.input)
    counts = build_breakdown(rows)

    write_csv(counts,     os.path.join(out_dir, "failure_breakdown.csv"))
    write_plot(counts,    os.path.join(out_dir, "failure_modes.png"))
    write_summary(counts, os.path.join(out_dir, "failure_summary.txt"))

    print(f"\nAnalyzed {len(rows)} episodes across "
          f"{len(set((r['agent'], r['budget_total']) for r in rows))} (agent, budget) cells.")


if __name__ == "__main__":
    main()
