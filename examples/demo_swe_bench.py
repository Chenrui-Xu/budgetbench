"""
BudgetBench — End-to-End Demo (SWE-bench)
==========================================
Runs all three agent stubs against the SWE-bench Lite environment
across one or more issues and prints a summary report + saves plots.

Usage
-----
    # single default issue
    python examples/demo_swe_bench.py

    # specific issue
    python examples/demo_swe_bench.py --issue django__django-11099

    # first N issues from the dataset
    python examples/demo_swe_bench.py --num-issues 5

Requirements
------------
- Docker must be running without sudo (add user to docker group)
- Per-instance images must be built:
    pip install swebench
    python -m swebench.harness.docker_build \\
        --instance_ids django__django-11099 \\
        --dataset_name princeton-nlp/SWE-bench_Lite
- datasets package must be installed:
    pip install datasets
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from budgetbench import BudgetConfig, BudgetRunner
from budgetbench.metrics import compute_report
from budgetbench.visualize import plot_all
from environments.swe_bench import SWEBenchEnvironment
from agents.single_shot import SingleShotAgent
from agents.loop_agent import LoopAgent
from agents.budget_aware import BudgetAwareAgent

DEFAULT_ISSUE = "django__django-11099"


def _image_for(instance_id: str) -> str:
    return f"python:3.11-slim"


def run_experiments(issues: list[str], budget: int = 20) -> dict:
    runner = BudgetRunner()
    config = BudgetConfig(
        total=budget,
        cost_weights={"run_tests": 3, "submit_patch": 0},
    )
    agents = {
        "SingleShot":  SingleShotAgent(),
        "LoopAgent":   LoopAgent(),
        "BudgetAware": BudgetAwareAgent(budget_total=budget),
    }

    results: dict[str, list] = {name: [] for name in agents}

    for issue_id in issues:
        for agent_name, agent in agents.items():
            env = SWEBenchEnvironment(issue_id, docker_image=_image_for(issue_id))
            episode = runner.run_episode(
                agent=agent,
                env=env,
                config=config,
                metadata={"agent": agent_name, "issue_id": issue_id},
            )
            results[agent_name].append(episode)
            print(
                f"[issue={issue_id}] {agent_name:14s} "
                f"status={episode.final_status:18s} "
                f"used={episode.budget_used}/{budget}"
            )
            env.close()

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BudgetBench SWE-bench demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--issue", type=str, default=None,
                       help="Single instance ID to evaluate")
    group.add_argument("--num-issues", type=int, default=None,
                       help="Number of issues to load from the dataset")
    parser.add_argument("--dataset-path", type=str, default="data/swe-bench-lite",
                        help="Path to local SWE-bench Lite JSONL file or directory")
    parser.add_argument("--budget", type=int, default=20,
                        help="Total tool-call budget per episode (default: 20)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.issue:
        issues = [args.issue]
    elif args.num_issues:
        issues = SWEBenchEnvironment.list_issues(
            dataset_path=args.dataset_path, limit=args.num_issues
        )
        if not issues:
            print("No issues found. Check --dataset-path.")
            sys.exit(1)
    else:
        issues = [DEFAULT_ISSUE]

    print(f"Running BudgetBench demo on SWE-bench ({len(issues)} issue(s))...\n")
    results = run_experiments(issues, budget=args.budget)
    print_report(results)

    try:
        import matplotlib  # noqa: F401
        plot_all(results, output_dir="results", b_max=args.budget)
    except ImportError:
        print("\n(matplotlib not installed — skipping plots)")
