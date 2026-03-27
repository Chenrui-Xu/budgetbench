"""
BudgetBench — End-to-End Demo (Knapsack)
=========================================
Runs three heuristic agents against the Knapsack toy environment
under different budgets and prints a summary report + saves plots.

Usage
-----
    python examples/demo_knapsack.py

No external API keys required — agents use deterministic heuristics
to produce meaningful evaluation curves.

Agent strategies
----------------
KnapsackSingleShot  (budget ≈ 1)
    One list_items call → greedy by value/weight ratio → submit.
    Cheapest, but greedy is not always optimal for 0/1 knapsack.

KnapsackLoop        (budget ≈ 12–18)
    Add all items → iteratively remove worst until feasible → submit.
    Uses more budget; explores the full item set before pruning.

KnapsackBudgetAware (budget ≈ 9)
    HIGH regime  (>60% budget remaining): loop strategy (thorough).
    MEDIUM/LOW regime: greedy submit immediately.
    Adapts mid-episode when budget crosses the 60% threshold.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from budgetbench import BudgetConfig, BudgetRunner
from budgetbench.interfaces import Agent, ToolCall
from budgetbench.metrics import compute_report
from budgetbench.visualize import plot_all
from environments.knapsack import KnapsackEnvironment


# ---------------------------------------------------------------------------
# Heuristic agents (no LLM — for framework demonstration)
# ---------------------------------------------------------------------------

class KnapsackSingleShot(Agent):
    """Greedy by value/weight ratio. Budget ≈ 1."""

    def act(self, obs: dict, budget_remaining: int, history: list[ToolCall]) -> dict:
        if not history:
            self._capacity = obs.get("capacity", 50)
            return {"tool": "list_items"}
        items = history[-1].result.get("items", [])
        items_sorted = sorted(items, key=lambda x: x["value"] / max(x["weight"], 1), reverse=True)
        selected, weight = [], 0
        for item in items_sorted:
            if weight + item["weight"] <= self._capacity:
                selected.append(item["id"])
                weight += item["weight"]
        return {"tool": "submit_patch", "patch": json.dumps({"selected": selected})}


class KnapsackLoop(Agent):
    """
    Add all items, then remove the worst (by value/weight ratio)
    until the selection is feasible.  Budget ≈ 12–18.
    """

    def act(self, obs: dict, budget_remaining: int, history: list[ToolCall]) -> dict:
        if not history:
            self._capacity = obs.get("capacity", 50)
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
            selected_ids = last.result.get("selected_ids", [])
            if selected_ids:
                worst = min(
                    selected_ids,
                    key=lambda i: self._items[i]["value"] / max(self._items[i]["weight"], 1),
                )
                return {"tool": "remove_item", "id": worst}

        return {"tool": "submit_patch", "patch": ""}


class KnapsackBudgetAware(Agent):
    """
    Adapts strategy based on remaining budget fraction:
      HIGH   (>60%): loop strategy — add all, remove worst until feasible
      MEDIUM/LOW    : greedy submit from the item list
    """

    def __init__(self, budget_total: int = 20) -> None:
        self.budget_total = budget_total

    def _regime(self, budget_remaining: int) -> str:
        frac = budget_remaining / max(self.budget_total, 1)
        if frac > 0.6:
            return "HIGH"
        return "MEDIUM" if frac > 0.3 else "LOW"

    def _greedy_submit(self) -> dict:
        items = sorted(
            self._items.values(),
            key=lambda x: x["value"] / max(x["weight"], 1),
            reverse=True,
        )
        selected, weight = [], 0
        for item in items:
            if weight + item["weight"] <= self._capacity:
                selected.append(item["id"])
                weight += item["weight"]
        return {"tool": "submit_patch", "patch": json.dumps({"selected": selected})}

    def act(self, obs: dict, budget_remaining: int, history: list[ToolCall]) -> dict:
        if not history:
            self._capacity = obs.get("capacity", 50)
            self._items: dict = {}
            self._to_add: list = []
            return {"tool": "list_items"}

        last = history[-1]
        regime = self._regime(budget_remaining)

        if last.name == "list_items":
            self._items = {it["id"]: it for it in last.result.get("items", [])}
            self._to_add = list(self._items.keys())

        # Switch to fast path when budget shrinks below HIGH threshold
        if regime in ("MEDIUM", "LOW") and self._items:
            return self._greedy_submit()

        # HIGH regime: thorough loop
        if self._to_add:
            return {"tool": "add_item", "id": self._to_add.pop(0)}

        if last.name in ("add_item", "remove_item"):
            return {"tool": "check_feasibility"}

        if last.name == "check_feasibility":
            if last.result.get("feasible", True) or budget_remaining < 3:
                return {"tool": "submit_patch", "patch": ""}
            selected_ids = last.result.get("selected_ids", [])
            if selected_ids:
                worst = min(
                    selected_ids,
                    key=lambda i: self._items[i]["value"] / max(self._items[i]["weight"], 1),
                )
                return {"tool": "remove_item", "id": worst}

        return self._greedy_submit()


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiments(n_seeds: int = 10, budget: int = 20) -> dict:
    runner = BudgetRunner()
    config = BudgetConfig(
        total=budget,
        cost_weights={"run_tests": 3, "submit_patch": 0},
    )
    agents = {
        "SingleShot":  KnapsackSingleShot(),
        "LoopAgent":   KnapsackLoop(),
        "BudgetAware": KnapsackBudgetAware(budget_total=budget),
    }

    results: dict[str, list] = {name: [] for name in agents}

    for seed in range(n_seeds):
        env = KnapsackEnvironment(n_items=10, capacity=50, seed=seed)
        for agent_name, agent in agents.items():
            episode = runner.run_episode(
                agent=agent,
                env=env,
                config=config,
                metadata={"agent": agent_name, "seed": seed},
            )
            results[agent_name].append(episode)
            print(
                f"[seed={seed}] {agent_name:14s} "
                f"status={episode.final_status:18s} "
                f"used={episode.budget_used:>2}/{budget}"
            )

    return results


def print_report(results: dict) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    for agent_name, episodes in results.items():
        report = compute_report(episodes)
        lo, hi = report["success_ci_95"]
        print(f"\n{agent_name}")
        print(f"  Success@B_max : {report['success_at_b_max']:.2%}")
        print(f"  Invalid@B_max : {report['invalid_at_b_max']:.2%}")
        print(f"  Violation rate: {report['violation_rate']:.2%}")
        print(f"  AUC           : {report['auc_success_vs_budget']:.3f}")
        print(f"  95% CI        : [{lo:.2%}, {hi:.2%}]")


if __name__ == "__main__":
    print("Running BudgetBench demo on Knapsack environment...\n")
    results = run_experiments(n_seeds=10, budget=20)
    print_report(results)

    try:
        import matplotlib  # noqa: F401
        plot_all(results, output_dir="results", b_max=20)
    except ImportError:
        print("\n(matplotlib not installed — skipping plots)")
