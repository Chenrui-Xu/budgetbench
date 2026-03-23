"""Tests for BudgetRunner."""

import pytest
from budgetbench import BudgetConfig, BudgetRunner
from .conftest import FixedAgent, MockEnvironment


def make_runner():
    return BudgetRunner()


# ---------------------------------------------------------------------------
# Basic episode
# ---------------------------------------------------------------------------

def test_solve_within_budget(ok_env, default_config):
    agent = FixedAgent([
        {"tool": "search",       "query": "def calculate"},
        {"tool": "inspect",      "file": "calc.py"},
        {"tool": "submit_patch", "patch": "--- a\n+++ b\n"},
    ])
    episode = make_runner().run_episode(agent, ok_env, default_config)
    assert episode.final_status == "OK_SOLVED"
    assert episode.budget_used <= default_config.total
    assert episode.solved is True


def test_budget_exceeded_before_submit(default_config):
    env = MockEnvironment()
    # Never submit; keep searching until budget runs out
    agent = FixedAgent([{"tool": "search", "query": f"q{i}"} for i in range(20)])
    episode = make_runner().run_episode(agent, env, default_config)
    assert episode.final_status == "BUDGET_EXCEEDED"
    assert episode.budget_used <= default_config.total


def test_expensive_tool_triggers_budget_exceeded():
    config = BudgetConfig(total=5, cost_weights={"run_tests": 10})
    env = MockEnvironment()
    agent = FixedAgent([{"tool": "run_tests"}])
    episode = make_runner().run_episode(agent, env, config)
    assert episode.final_status == "BUDGET_EXCEEDED"
    # run_tests costs 10 > budget 5; no tool calls should be recorded
    assert len(episode.tool_calls) == 0


def test_failing_env_returns_test_fail(failing_env, default_config):
    agent = FixedAgent([{"tool": "submit_patch", "patch": "--- a\n+++ b\n"}])
    episode = make_runner().run_episode(agent, failing_env, default_config)
    assert episode.final_status == "TEST_FAIL"


# ---------------------------------------------------------------------------
# Tool call logging
# ---------------------------------------------------------------------------

def test_tool_calls_logged(ok_env, default_config):
    agent = FixedAgent([
        {"tool": "search",       "query": "bug"},
        {"tool": "submit_patch", "patch": "--- a\n+++ b\n"},
    ])
    episode = make_runner().run_episode(agent, ok_env, default_config)
    assert len(episode.tool_calls) == 2
    assert episode.tool_calls[0].name == "search"
    assert episode.tool_calls[0].cost == 1
    assert episode.tool_calls[0].ts_ms > 0


def test_run_tests_costs_3x(ok_env):
    config = BudgetConfig(total=20, cost_weights={"run_tests": 3, "submit_patch": 0})
    agent = FixedAgent([
        {"tool": "run_tests"},
        {"tool": "submit_patch", "patch": "--- a\n+++ b\n"},
    ])
    episode = make_runner().run_episode(agent, ok_env, config)
    run_test_call = episode.tool_calls[0]
    assert run_test_call.cost == 3


# ---------------------------------------------------------------------------
# BudgetConfig helpers
# ---------------------------------------------------------------------------

def test_budget_config_cost_of_unknown_defaults_to_1():
    config = BudgetConfig(total=10, cost_weights={"run_tests": 3})
    assert config.cost_of("some_random_tool") == 1
    assert config.cost_of("run_tests") == 3


def test_episode_budget_remaining():
    from budgetbench.interfaces import Episode
    ep = Episode(budget_total=10, budget_used=3)
    assert ep.budget_remaining == 7


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------

def test_metadata_stored_in_episode(ok_env, default_config):
    agent = FixedAgent([{"tool": "submit_patch", "patch": ""}])
    meta = {"issue_id": "SWE-42", "model": "gpt-4o"}
    episode = make_runner().run_episode(agent, ok_env, default_config, metadata=meta)
    assert episode.metadata["issue_id"] == "SWE-42"
