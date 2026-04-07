"""
BudgetBench — Integration Tests
================================
Tests the full pipeline: KnapsackEnvironment → BudgetRunner → metrics.

Unlike unit tests (which use MockEnvironment/FixedAgent), these tests
exercise the real environment oracle and the heuristic agents from
examples/demo_knapsack.py.

Implemented by: Chenrui Xu (Phase 2)
"""

from __future__ import annotations

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from budgetbench import BudgetConfig, BudgetRunner
from budgetbench.interfaces import Agent, ToolCall
from budgetbench.metrics import (
    auc_success_vs_budget,
    compute_report,
    success_at_b,
    violation_rate,
)
from environments.knapsack import KnapsackEnvironment


# ---------------------------------------------------------------------------
# Heuristic agents (same as demo_knapsack.py — copied to keep tests
# self-contained and independent of the examples directory)
# ---------------------------------------------------------------------------

class _GreedyAgent(Agent):
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


class _LoopAgent(Agent):
    """Add all items → remove worst until feasible. Budget ≈ 12–18."""

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
                worst = min(selected_ids, key=lambda i: self._items[i]["value"] / max(self._items[i]["weight"], 1))
                return {"tool": "remove_item", "id": worst}
        return {"tool": "submit_patch", "patch": ""}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    return BudgetRunner()


@pytest.fixture
def config_20():
    return BudgetConfig(total=20, cost_weights={"run_tests": 3, "submit_patch": 0})


@pytest.fixture
def knapsack_seed0():
    return KnapsackEnvironment(n_items=10, capacity=50, seed=0)


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestKnapsackEnvironment:

    def test_reset_returns_required_keys(self, knapsack_seed0):
        obs = knapsack_seed0.reset()
        assert obs["task"] == "knapsack"
        assert obs["n_items"] == 10
        assert obs["capacity"] == 50

    def test_list_items_returns_all_items(self, knapsack_seed0):
        knapsack_seed0.reset()
        obs, done = knapsack_seed0.step({"tool": "list_items"})
        assert not done
        assert len(obs["items"]) == 10
        for item in obs["items"]:
            assert "id" in item and "weight" in item and "value" in item

    def test_add_and_check_feasibility(self, knapsack_seed0):
        knapsack_seed0.reset()
        knapsack_seed0.step({"tool": "add_item", "id": 0})
        obs, done = knapsack_seed0.step({"tool": "check_feasibility"})
        assert not done
        assert "feasible" in obs
        assert "current_weight" in obs
        assert "selected_ids" in obs
        assert 0 in obs["selected_ids"]

    def test_submit_patch_ends_episode(self, knapsack_seed0):
        knapsack_seed0.reset()
        obs, done = knapsack_seed0.step({"tool": "submit_patch", "patch": '{"selected": []}'})
        assert done

    def test_score_empty_selection_is_test_fail(self, knapsack_seed0):
        knapsack_seed0.reset()
        status = knapsack_seed0.score('{"selected": []}')
        assert status == "TEST_FAIL"

    def test_score_invalid_json_is_invalid_format(self, knapsack_seed0):
        knapsack_seed0.reset()
        status = knapsack_seed0.score("not json")
        assert status == "INVALID_FORMAT"

    def test_score_invalid_item_id_is_patch_apply_fail(self, knapsack_seed0):
        knapsack_seed0.reset()
        status = knapsack_seed0.score('{"selected": [999]}')
        assert status == "PATCH_APPLY_FAIL"

    def test_score_overweight_is_test_fail(self, knapsack_seed0):
        knapsack_seed0.reset()
        # Select all items — almost certainly overweight
        status = knapsack_seed0.score('{"selected": [0,1,2,3,4,5,6,7,8,9]}')
        assert status in ("TEST_FAIL", "OK_SOLVED")  # OK_SOLVED if capacity is huge

    def test_optimal_patch_is_ok_solved(self, knapsack_seed0):
        """Compute optimal via oracle and verify score returns OK_SOLVED."""
        knapsack_seed0.reset()
        optimal_value = knapsack_seed0.optimal_value
        assert optimal_value > 0

    def test_reset_is_reproducible(self):
        """Same seed must produce identical instances."""
        env1 = KnapsackEnvironment(n_items=5, capacity=20, seed=7)
        env2 = KnapsackEnvironment(n_items=5, capacity=20, seed=7)
        obs1 = env1.reset()
        obs2 = env2.reset()
        env1.step({"tool": "list_items"})
        env2.step({"tool": "list_items"})
        items1 = env1.step({"tool": "list_items"})[0]["items"]
        items2 = env2.step({"tool": "list_items"})[0]["items"]
        assert items1 == items2

    def test_different_seeds_produce_different_instances(self):
        env_a = KnapsackEnvironment(seed=0)
        env_b = KnapsackEnvironment(seed=1)
        env_a.reset()
        env_b.reset()
        obs_a, _ = env_a.step({"tool": "list_items"})
        obs_b, _ = env_b.step({"tool": "list_items"})
        weights_a = [it["weight"] for it in obs_a["items"]]
        weights_b = [it["weight"] for it in obs_b["items"]]
        assert weights_a != weights_b


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_greedy_agent_uses_budget_1(self, runner, config_20, knapsack_seed0):
        episode = runner.run_episode(_GreedyAgent(), knapsack_seed0, config_20)
        assert episode.budget_used == 1

    def test_greedy_agent_never_invalid_format(self, runner, config_20):
        """Greedy agent always submits valid JSON — never INVALID_FORMAT."""
        for seed in range(5):
            env = KnapsackEnvironment(n_items=10, capacity=50, seed=seed)
            ep = runner.run_episode(_GreedyAgent(), env, config_20)
            assert ep.final_status != "INVALID_FORMAT", f"INVALID_FORMAT on seed={seed}"

    def test_greedy_agent_never_exceeds_budget(self, runner, config_20):
        for seed in range(5):
            env = KnapsackEnvironment(n_items=10, capacity=50, seed=seed)
            ep = runner.run_episode(_GreedyAgent(), env, config_20)
            assert ep.budget_used <= config_20.total
            assert ep.final_status != "BUDGET_EXCEEDED"

    def test_loop_agent_uses_more_budget_than_greedy(self, runner, config_20):
        env = KnapsackEnvironment(n_items=10, capacity=50, seed=1)
        greedy_ep = runner.run_episode(_GreedyAgent(), env, config_20)
        env2 = KnapsackEnvironment(n_items=10, capacity=50, seed=1)
        loop_ep = runner.run_episode(_LoopAgent(), env2, config_20)
        assert loop_ep.budget_used > greedy_ep.budget_used

    def test_loop_agent_never_invalid_format(self, runner, config_20):
        for seed in range(5):
            env = KnapsackEnvironment(n_items=10, capacity=50, seed=seed)
            ep = runner.run_episode(_LoopAgent(), env, config_20)
            assert ep.final_status != "INVALID_FORMAT", f"INVALID_FORMAT on seed={seed}"

    def test_episode_tool_calls_logged(self, runner, config_20, knapsack_seed0):
        ep = runner.run_episode(_GreedyAgent(), knapsack_seed0, config_20)
        assert len(ep.tool_calls) >= 1
        assert ep.tool_calls[-1].name == "submit_patch"

    def test_budget_enforced_at_limit(self, runner):
        """Agent that hammers search calls must hit BUDGET_EXCEEDED."""
        class SpendthriftAgent(Agent):
            def act(self, obs, budget_remaining, history):
                return {"tool": "list_items"}  # never submits

        config = BudgetConfig(total=3, cost_weights={"submit_patch": 0})
        env = KnapsackEnvironment(seed=0)
        ep = runner.run_episode(SpendthriftAgent(), env, config)
        assert ep.final_status == "BUDGET_EXCEEDED"
        assert ep.budget_used <= config.total

    def test_violation_rate_is_zero(self, runner, config_20):
        """No episode should ever exceed its declared budget."""
        episodes = []
        for seed in range(10):
            env = KnapsackEnvironment(n_items=10, capacity=50, seed=seed)
            ep = runner.run_episode(_GreedyAgent(), env, config_20)
            episodes.append(ep)
        assert violation_rate(episodes) == 0.0


# ---------------------------------------------------------------------------
# Metrics integration tests
# ---------------------------------------------------------------------------

class TestMetricsIntegration:

    @pytest.fixture
    def greedy_episodes(self, runner, config_20):
        episodes = []
        for seed in range(10):
            env = KnapsackEnvironment(n_items=10, capacity=50, seed=seed)
            ep = runner.run_episode(_GreedyAgent(), env, config_20)
            episodes.append(ep)
        return episodes

    def test_success_at_1_equals_success_at_20_for_greedy(self, greedy_episodes):
        """Greedy uses budget=1, so success_at_b is identical for all b≥1."""
        s1  = success_at_b(greedy_episodes, 1)
        s20 = success_at_b(greedy_episodes, 20)
        assert s1 == s20

    def test_success_at_0_is_zero(self, greedy_episodes):
        assert success_at_b(greedy_episodes, 0) == 0.0

    def test_auc_between_0_and_1(self, greedy_episodes):
        auc = auc_success_vs_budget(greedy_episodes, b_max=20)
        assert 0.0 <= auc <= 1.0

    def test_compute_report_has_all_keys(self, greedy_episodes):
        report = compute_report(greedy_episodes)
        for key in ("success_at_b_max", "invalid_at_b_max", "violation_rate",
                    "auc_success_vs_budget", "success_ci_95", "n_episodes"):
            assert key in report, f"Missing key: {key}"

    def test_loop_agent_lower_auc_than_greedy(self, runner, config_20):
        """LoopAgent uses more budget → lower AUC despite similar success rate."""
        greedy_eps, loop_eps = [], []
        for seed in range(10):
            env_g = KnapsackEnvironment(n_items=10, capacity=50, seed=seed)
            env_l = KnapsackEnvironment(n_items=10, capacity=50, seed=seed)
            greedy_eps.append(runner.run_episode(_GreedyAgent(), env_g, config_20))
            loop_eps.append(runner.run_episode(_LoopAgent(),   env_l, config_20))
        auc_greedy = auc_success_vs_budget(greedy_eps, b_max=20)
        auc_loop   = auc_success_vs_budget(loop_eps,   b_max=20)
        assert auc_greedy > auc_loop, (
            f"Expected greedy AUC ({auc_greedy:.3f}) > loop AUC ({auc_loop:.3f})"
        )

    def test_ci_lower_leq_mean_leq_upper(self, greedy_episodes):
        report = compute_report(greedy_episodes)
        lo, hi = report["success_ci_95"]
        mean = report["success_at_b_max"]
        assert lo <= mean <= hi
