"""
BudgetBench — Batch Experiment Script
======================================
Runs all three agents across multiple issues and multiple budget levels.
Saves per-episode records to CSV + JSON, then prints a summary table.

Usage
-----
    # Knapsack (no external APIs needed):
    python scripts/batch_experiment.py --env knapsack

    # SWE-bench (requires GEMINI_API_KEY and Docker):
    python scripts/batch_experiment.py --env swe --num-issues 5 --budget 10 20 30

    # Knapsack, custom budgets and seeds:
    python scripts/batch_experiment.py --env knapsack \\
        --budget 5 10 20 --num-issues 10 --output-dir results/batch_run1

Output
------
    results/
        episodes.csv   — one row per (agent × issue × budget) episode
        report.json    — aggregate metrics per agent × budget
        *.png          — success-vs-budget plots (if matplotlib installed)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from budgetbench import BudgetConfig, BudgetRunner
from budgetbench.interfaces import Episode
from budgetbench.metrics import compute_report, auc_success_vs_budget


# ---------------------------------------------------------------------------
# Agent factories (re-instantiate per budget to reset internal state)
# ---------------------------------------------------------------------------

_ALL_AGENTS = ["SingleShot", "LoopAgent", "BudgetAware"]

def _make_agents(budget: int, names: list[str] | None = None) -> dict:
    from agents.single_shot import SingleShotAgent
    from agents.loop_agent import LoopAgent
    from agents.budget_aware import BudgetAwareAgent
    all_agents = {
        "SingleShot":  SingleShotAgent(),
        "LoopAgent":   LoopAgent(),
        "BudgetAware": BudgetAwareAgent(budget_total=budget),
    }
    if names:
        return {k: v for k, v in all_agents.items() if k in names}
    return all_agents


def _make_knapsack_agents() -> dict:
    """Pure heuristic agents for Knapsack (no LLM, no API key needed)."""
    from budgetbench.interfaces import Agent, ToolCall

    class _GreedyAgent(Agent):
        def act(self, obs, budget_remaining, history):
            if not history:
                self._cap = obs.get("capacity", 50)
                return {"tool": "list_items"}
            items = history[-1].result.get("items", [])
            items = sorted(items, key=lambda x: x["value"] / max(x["weight"], 1), reverse=True)
            sel, w = [], 0
            for it in items:
                if w + it["weight"] <= self._cap:
                    sel.append(it["id"]); w += it["weight"]
            return {"tool": "submit_patch", "patch": json.dumps({"selected": sel})}

    class _LoopAgent(Agent):
        def act(self, obs, budget_remaining, history):
            if not history:
                self._cap = obs.get("capacity", 50)
                self._items: dict = {}
                self._to_add: list = []
                return {"tool": "list_items"}
            last = history[-1]
            if last.name == "list_items":
                self._items = {it["id"]: it for it in last.result.get("items", [])}
                self._to_add = list(self._items.keys())
            if self._to_add:
                return {"tool": "add_item", "id": self._to_add.pop(0)}
            if last.name in ("add_item", "remove_item"):
                return {"tool": "check_feasibility"}
            if last.name == "check_feasibility":
                if last.result.get("feasible", True) or budget_remaining < 3:
                    return {"tool": "submit_patch", "patch": ""}
                sel = last.result.get("selected_ids", [])
                if sel:
                    worst = min(sel, key=lambda i: self._items[i]["value"] / max(self._items[i]["weight"], 1))
                    return {"tool": "remove_item", "id": worst}
            return {"tool": "submit_patch", "patch": ""}

    class _BudgetAwareAgent(Agent):
        def _greedy_submit(self):
            items = sorted(self._items.values(),
                           key=lambda x: x["value"] / max(x["weight"], 1), reverse=True)
            sel, w = [], 0
            for it in items:
                if w + it["weight"] <= self._cap:
                    sel.append(it["id"]); w += it["weight"]
            return {"tool": "submit_patch", "patch": json.dumps({"selected": sel})}

        def act(self, obs, budget_remaining, history):
            if not history:
                self._cap = obs.get("capacity", 50)
                self._budget_total = budget_remaining
                self._items: dict = {}
                self._to_add: list = []
                return {"tool": "list_items"}
            last = history[-1]
            # Always capture items when received
            if last.name == "list_items":
                self._items = {it["id"]: it for it in last.result.get("items", [])}
                self._to_add = list(self._items.keys())
            frac = budget_remaining / max(self._budget_total, 1)
            # MEDIUM/LOW: greedy submit immediately (if we have items)
            if frac <= 0.6 and self._items:
                return self._greedy_submit()
            # HIGH regime: thorough loop strategy
            if self._to_add:
                return {"tool": "add_item", "id": self._to_add.pop(0)}
            if last.name in ("add_item", "remove_item"):
                return {"tool": "check_feasibility"}
            if last.name == "check_feasibility":
                if last.result.get("feasible", True) or budget_remaining < 3:
                    return {"tool": "submit_patch", "patch": ""}
                sel = last.result.get("selected_ids", [])
                if sel:
                    worst = min(sel, key=lambda i: self._items[i]["value"] / max(self._items[i]["weight"], 1))
                    return {"tool": "remove_item", "id": worst}
            return self._greedy_submit() if self._items else {"tool": "submit_patch", "patch": ""}

    return {
        "SingleShot":  _GreedyAgent(),
        "LoopAgent":   _LoopAgent(),
        "BudgetAware": _BudgetAwareAgent(),
    }


# ---------------------------------------------------------------------------
# Core batch runner
# ---------------------------------------------------------------------------

def run_batch(
    env_name: str,
    issues: list[str],
    budgets: list[int],
    output_dir: str,
    dataset_path: str = "data/swe-bench-lite",
    verbose: bool = False,
    use_eval_image: bool = False,
    agent_names: list[str] | None = None,
) -> tuple[list[dict], dict]:
    """
    Run all agents × all issues × all budgets.

    Returns
    -------
    rows : list[dict]
        One dict per episode (written to CSV).
    agg  : dict
        Nested dict: agg[agent_name][budget] = metrics report.
    """
    os.makedirs(output_dir, exist_ok=True)
    runner = BudgetRunner()
    rows: list[dict] = []
    # agg[agent][budget] = list of episodes
    from collections import defaultdict
    episodes_by: dict = defaultdict(lambda: defaultdict(list))

    for budget in budgets:
        config = BudgetConfig(
            total=budget,
            cost_weights={"run_tests": 3, "submit_patch": 0},
        )
        for issue_id in issues:
            if env_name == "knapsack":
                agents = _make_knapsack_agents()
            else:
                agents = _make_agents(budget, names=agent_names)

            for agent_name, agent in agents.items():
                env = _make_env(env_name, issue_id, dataset_path, verbose=verbose, use_eval_image=use_eval_image)
                try:
                    episode = runner.run_episode(
                        agent=agent,
                        env=env,
                        config=config,
                        metadata={"agent": agent_name, "issue_id": issue_id, "budget": budget},
                    )
                finally:
                    if hasattr(env, "close"):
                        env.close()

                episodes_by[agent_name][budget].append(episode)
                row = _episode_to_row(episode, agent_name, issue_id, budget)
                rows.append(row)
                print(
                    f"  budget={budget:>3}  {agent_name:14s}  issue={issue_id}  "
                    f"status={episode.final_status:18s}  used={episode.budget_used}"
                )

    # Build aggregate reports
    agg: dict = {}
    for agent_name, budget_eps in episodes_by.items():
        agg[agent_name] = {}
        for budget, eps in budget_eps.items():
            report = compute_report(eps, b_max=budget)
            lo, hi = report["success_ci_95"]
            agg[agent_name][budget] = {
                "n_episodes":            report["n_episodes"],
                "success_at_b_max":      round(report["success_at_b_max"], 4),
                "invalid_at_b_max":      round(report["invalid_at_b_max"], 4),
                "violation_rate":        round(report["violation_rate"], 4),
                "auc_success_vs_budget": round(report["auc_success_vs_budget"], 4),
                "success_ci_95_lo":      round(lo, 4),
                "success_ci_95_hi":      round(hi, 4),
                "n_ok_solved":           report["n_ok_solved"],
                "n_invalid_format":      report["n_invalid_format"],
                "n_test_fail":           report["n_test_fail"],
                "n_budget_exceeded":     report["n_budget_exceeded"],
            }

    return rows, agg


def _make_env(env_name: str, issue_id: str, dataset_path: str, verbose: bool = False, use_eval_image: bool = False):
    if env_name == "knapsack":
        from environments.knapsack import KnapsackEnvironment
        seed = hash(issue_id) % (2**31)
        return KnapsackEnvironment(n_items=10, capacity=50, seed=seed)
    else:
        from environments.swe_bench import SWEBenchEnvironment
        return SWEBenchEnvironment(
            issue_id,
            docker_image="python:3.11",
            use_eval_image=use_eval_image,
            verbose=verbose,
        )


def _episode_to_row(ep: Episode, agent_name: str, issue_id: str, budget: int) -> dict:
    return {
        "agent":        agent_name,
        "issue_id":     issue_id,
        "budget_total": budget,
        "budget_used":  ep.budget_used,
        "final_status": ep.final_status,
        "solved":       int(ep.solved),
        "n_tool_calls": len(ep.tool_calls),
        "timestamp":    datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} episode rows → {path}")


def save_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved aggregate report → {path}")


def print_summary(agg: dict, budgets: list[int]) -> None:
    print("\n" + "=" * 72)
    print("BATCH EXPERIMENT SUMMARY")
    print("=" * 72)
    header = f"{'Agent':14s}  {'Budget':>6}  {'Success':>7}  {'AUC':>6}  {'Invalid':>7}  {'Status counts'}"
    print(header)
    print("-" * 72)
    for agent_name, budget_data in agg.items():
        for budget in budgets:
            if budget not in budget_data:
                continue
            r = budget_data[budget]
            statuses = (
                f"OK={r['n_ok_solved']} FAIL={r['n_test_fail']} "
                f"INV={r['n_invalid_format']} OVER={r['n_budget_exceeded']}"
            )
            print(
                f"{agent_name:14s}  {budget:>6}  "
                f"{r['success_at_b_max']:>6.1%}  "
                f"{r['auc_success_vs_budget']:>6.3f}  "
                f"{r['invalid_at_b_max']:>6.1%}  "
                f"{statuses}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BudgetBench batch experiment runner")
    p.add_argument("--env", choices=["knapsack", "swe"], default="knapsack",
                   help="Environment to use (default: knapsack)")
    p.add_argument("--budget", type=int, nargs="+", default=[5, 10, 20],
                   help="One or more budget levels (default: 5 10 20)")
    p.add_argument("--num-issues", type=int, default=10,
                   help="Number of issues/seeds to evaluate (default: 10)")
    p.add_argument("--issue", type=str, nargs="+", default=None,
                   help="Explicit SWE-bench issue ID(s) (overrides --num-issues)")
    p.add_argument("--dataset-path", type=str, default="data/swe-bench-lite",
                   help="Path to SWE-bench Lite dataset (only used with --env swe)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable sandbox debug logging for SWE-bench env")
    p.add_argument("--agent", type=str, nargs="+", default=None,
                   choices=_ALL_AGENTS,
                   help="Agent(s) to run (default: all). e.g. --agent LoopAgent BudgetAware")
    p.add_argument("--use-eval-image", action="store_true",
                   help="Use the official swebench eval Docker image for the agent sandbox "
                        "(enables meaningful edit/run_tests; image must already be cached "
                        "by a prior harness run)")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Directory to write CSV/JSON/plots (default: results)")
    return p.parse_args()


def _resolve_issues(args: argparse.Namespace) -> list[str]:
    if args.issue:
        return args.issue
    if args.env == "knapsack":
        # Use integer seeds as issue IDs for knapsack
        return [str(i) for i in range(args.num_issues)]
    else:
        from environments.swe_bench import SWEBenchEnvironment
        issues = SWEBenchEnvironment.list_issues(
            dataset_path=args.dataset_path, limit=args.num_issues
        )
        if not issues:
            print("No SWE-bench issues found. Check --dataset-path.")
            sys.exit(1)
        return issues


if __name__ == "__main__":
    args = parse_args()
    issues  = _resolve_issues(args)
    budgets = sorted(set(args.budget))

    print(f"BudgetBench batch experiment")
    print(f"  env={args.env}  issues={len(issues)}  budgets={budgets}")
    print(f"  output → {args.output_dir}/\n")

    rows, agg = run_batch(
        env_name=args.env,
        issues=issues,
        budgets=budgets,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        verbose=args.verbose,
        use_eval_image=args.use_eval_image,
        agent_names=args.agent,
    )

    save_csv(rows, os.path.join(args.output_dir, "episodes.csv"))
    save_json(agg,  os.path.join(args.output_dir, "report.json"))
    print_summary(agg, budgets)

    # Plots (optional)
    try:
        import matplotlib  # noqa: F401
        from budgetbench.visualize import plot_all
        from collections import defaultdict
        from budgetbench.interfaces import Episode as Ep
        # Rebuild episode lists for plot_all (uses max budget)
        max_budget = max(budgets)
        from collections import defaultdict
        _eps_for_plot: dict = defaultdict(list)
        for row in rows:
            if row["budget_total"] == max_budget:
                # Reconstruct minimal Episode for plotting
                ep = Ep(
                    budget_total=row["budget_total"],
                    budget_used=row["budget_used"],
                    final_status=row["final_status"],
                )
                _eps_for_plot[row["agent"]].append(ep)
        if _eps_for_plot:
            plot_all(dict(_eps_for_plot), output_dir=args.output_dir, b_max=max_budget)
    except ImportError:
        pass
