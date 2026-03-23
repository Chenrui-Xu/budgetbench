"""
BudgetBench — End-to-End Demo (Knapsack)
=========================================
Runs all three agent stubs against the Knapsack toy environment
under different budgets and prints a summary report + saves plots.

Usage
-----
    python examples/demo_knapsack.py

This script requires no external API keys — it uses stub agents that
immediately submit a patch (not yet connected to an LLM).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from budgetbench import BudgetConfig, BudgetRunner
from budgetbench.metrics import compute_report
from budgetbench.visualize import plot_all
from environments.knapsack import KnapsackEnvironment
from agents.single_shot import SingleShotAgent
from agents.loop_agent import LoopAgent
from agents.budget_aware import BudgetAwareAgent


def run_experiments(n_seeds: int = 5, budget: int = 20) -> dict:
    runner  = BudgetRunner()
    config  = BudgetConfig(
        total=budget,
        cost_weights={"run_tests": 3, "submit_patch": 0},
    )
    agents = {
        "SingleShot":   SingleShotAgent(),
        "LoopAgent":    LoopAgent(),
        "BudgetAware":  BudgetAwareAgent(budget_total=budget),
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
                f"used={episode.budget_used}/{budget}"
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
        import matplotlib
        plot_all(results, output_dir="results", b_max=20)
    except ImportError:
        print("\n(matplotlib not installed — skipping plots)")
