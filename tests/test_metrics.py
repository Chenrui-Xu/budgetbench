"""Tests for evaluation metrics."""

import pytest
from budgetbench.interfaces import Episode, ToolCall
from budgetbench.metrics import (
    auc_success_vs_budget,
    bootstrap_ci,
    compute_report,
    invalid_at_b,
    is_solved,
    success_at_b,
    violation_rate,
)


def make_episode(status, used, total=10):
    return Episode(budget_total=total, budget_used=used, final_status=status)


def test_success_at_b_all_solved():
    episodes = [make_episode("OK_SOLVED", 5) for _ in range(4)]
    assert success_at_b(episodes, b=10) == 1.0


def test_success_at_b_none_solved():
    episodes = [make_episode("TEST_FAIL", 10) for _ in range(4)]
    assert success_at_b(episodes, b=10) == 0.0


def test_success_at_b_partial():
    episodes = [
        make_episode("OK_SOLVED", 5),
        make_episode("TEST_FAIL", 10),
        make_episode("OK_SOLVED", 8),
        make_episode("BUDGET_EXCEEDED", 10),
    ]
    assert success_at_b(episodes, b=10) == 0.5


def test_invalid_at_b():
    episodes = [
        make_episode("INVALID_FORMAT",   5),
        make_episode("PATCH_APPLY_FAIL", 5),
        make_episode("OK_SOLVED",        5),
        make_episode("TEST_FAIL",        5),
    ]
    assert invalid_at_b(episodes, b=10) == 0.5


def test_violation_rate_no_violations():
    episodes = [make_episode("OK_SOLVED", 5, total=10) for _ in range(3)]
    assert violation_rate(episodes) == 0.0


def test_violation_rate_with_violations():
    episodes = [
        make_episode("OK_SOLVED", 10, total=10),
        make_episode("TEST_FAIL", 11, total=10),  # violated
    ]
    assert violation_rate(episodes) == 0.5


def test_auc_success_vs_budget_perfect():
    # All episodes solved in 1 tool call → AUC ≈ 1
    episodes = [make_episode("OK_SOLVED", 1) for _ in range(5)]
    auc = auc_success_vs_budget(episodes, b_max=10)
    assert auc >= 0.9


def test_auc_success_vs_budget_zero():
    episodes = [make_episode("TEST_FAIL", 10) for _ in range(5)]
    auc = auc_success_vs_budget(episodes, b_max=10)
    assert auc == 0.0


def test_bootstrap_ci_deterministic():
    values = [0.5, 0.6, 0.4, 0.5, 0.7]
    lo1, hi1 = bootstrap_ci(values, seed=42)
    lo2, hi2 = bootstrap_ci(values, seed=42)
    assert lo1 == lo2 and hi1 == hi2


def test_bootstrap_ci_bounds():
    values = [0.0, 1.0] * 50
    lo, hi = bootstrap_ci(values)
    assert lo < 0.5 < hi


def test_compute_report_keys():
    episodes = [
        make_episode("OK_SOLVED", 5),
        make_episode("TEST_FAIL", 10),
    ]
    report = compute_report(episodes, b_max=10)
    for key in ("success_at_b_max", "invalid_at_b_max", "violation_rate",
                "auc_success_vs_budget", "success_ci_95", "n_episodes"):
        assert key in report, f"Missing key: {key}"


def test_empty_episodes():
    assert success_at_b([], b=10) == 0.0
    assert violation_rate([]) == 0.0
    assert auc_success_vs_budget([]) == 0.0
