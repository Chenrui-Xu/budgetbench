"""
BudgetBench — Evaluation Metrics
==================================
Computes all reported metrics over a collection of episodes.

Implemented by: Chenrui Xu (Project Lead)
Scheduled:      Weeks 7–9
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .interfaces import Episode, StatusCode


# ---------------------------------------------------------------------------
# Per-episode helpers
# ---------------------------------------------------------------------------

def is_solved(ep: Episode) -> bool:
    return ep.final_status == "OK_SOLVED"


def is_invalid(ep: Episode) -> bool:
    return ep.final_status in ("INVALID_FORMAT", "PATCH_APPLY_FAIL")


def is_budget_exceeded(ep: Episode) -> bool:
    return ep.final_status == "BUDGET_EXCEEDED"


def time_to_first_valid(ep: Episode) -> int | None:
    """
    Number of tool calls consumed before the first valid patch attempt.
    Returns None if no valid patch was ever submitted.
    """
    cumulative = 0
    for call in ep.tool_calls:
        cumulative += call.cost
        if call.name == "submit_patch":
            return cumulative
    return None


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def success_at_b(episodes: Sequence[Episode], b: int) -> float:
    """Fraction of episodes solved within budget b."""
    if not episodes:
        return 0.0
    solved = sum(
        1 for ep in episodes
        if ep.budget_used <= b and is_solved(ep)
    )
    return solved / len(episodes)


def invalid_at_b(episodes: Sequence[Episode], b: int) -> float:
    """Fraction of episodes with INVALID_FORMAT or PATCH_APPLY_FAIL within budget b."""
    if not episodes:
        return 0.0
    invalid = sum(
        1 for ep in episodes
        if ep.budget_used <= b and is_invalid(ep)
    )
    return invalid / len(episodes)


def violation_rate(episodes: Sequence[Episode]) -> float:
    """Fraction of episodes that exceeded the declared budget."""
    if not episodes:
        return 0.0
    violated = sum(1 for ep in episodes if ep.budget_used > ep.budget_total)
    return violated / len(episodes)


def auc_success_vs_budget(
    episodes: Sequence[Episode],
    b_max: int | None = None,
) -> float:
    """
    Area under the Success@B curve as B sweeps from 1 to b_max,
    normalised to [0, 1].
    """
    if not episodes:
        return 0.0
    if b_max is None:
        b_max = max(ep.budget_total for ep in episodes)
    budgets = range(1, b_max + 1)
    values  = [success_at_b(episodes, b) for b in budgets]
    # trapezoidal integration, normalised
    auc = float(np.trapezoid(values, budgets)) / b_max
    return auc


def mean_time_to_first_valid(episodes: Sequence[Episode]) -> float | None:
    """Average tool calls to first valid patch; None if never submitted."""
    ttfvs = [t for ep in episodes if (t := time_to_first_valid(ep)) is not None]
    if not ttfvs:
        return None
    return float(np.mean(ttfvs))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    n_samples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the mean of ``values``.

    Returns (lower, upper) bounds.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    samples = rng.choice(arr, size=(n_samples, len(arr)), replace=True)
    means   = samples.mean(axis=1)
    alpha   = (1 - ci) / 2
    lower   = float(np.quantile(means, alpha))
    upper   = float(np.quantile(means, 1 - alpha))
    return lower, upper


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def compute_report(
    episodes: Sequence[Episode],
    b_max: int | None = None,
    bootstrap_n: int = 10_000,
) -> dict:
    """
    Compute and return all reported metrics as a dict.

    Keys
    ----
    success_at_b_max, invalid_at_b_max, violation_rate,
    auc_success_vs_budget, mean_time_to_first_valid,
    success_ci_95 (tuple[lower, upper])
    """
    if b_max is None and episodes:
        b_max = max(ep.budget_total for ep in episodes)
    elif b_max is None:
        b_max = 1

    successes = [1.0 if is_solved(ep) else 0.0 for ep in episodes]

    report = {
        "n_episodes":              len(episodes),
        "success_at_b_max":        success_at_b(episodes, b_max),
        "invalid_at_b_max":        invalid_at_b(episodes, b_max),
        "violation_rate":          violation_rate(episodes),
        "auc_success_vs_budget":   auc_success_vs_budget(episodes, b_max),
        "mean_time_to_first_valid": mean_time_to_first_valid(episodes),
        "success_ci_95":           bootstrap_ci(successes, n_samples=bootstrap_n),
    }

    # Status breakdown
    for status in ("OK_SOLVED", "INVALID_FORMAT", "PATCH_APPLY_FAIL",
                   "TEST_FAIL", "BUDGET_EXCEEDED"):
        report[f"n_{status.lower()}"] = sum(
            1 for ep in episodes if ep.final_status == status
        )

    return report
