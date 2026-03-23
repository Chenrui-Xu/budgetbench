"""Shared fixtures for BudgetBench tests."""

import pytest
from budgetbench.interfaces import Agent, BudgetConfig, Environment, StatusCode, ToolCall


class MockEnvironment(Environment):
    """Minimal environment that accepts any patch and returns OK_SOLVED."""

    def __init__(self, oracle_status: StatusCode = "OK_SOLVED"):
        self._oracle_status = oracle_status
        self._step_count = 0

    def reset(self) -> dict:
        self._step_count = 0
        return {"issue": "mock issue", "repo": "mock_repo"}

    def step(self, action: dict) -> tuple[dict, bool]:
        self._step_count += 1
        done = action.get("tool") == "submit_patch"
        obs = {"step": self._step_count, "patch_result": None}
        return obs, done

    def score(self, patch: str) -> StatusCode:
        return self._oracle_status


class FixedAgent(Agent):
    """Agent that always plays a fixed sequence of actions."""

    def __init__(self, actions: list[dict]):
        self._actions = actions
        self._idx = 0

    def act(self, obs: dict, budget_remaining: int, history: list[ToolCall]) -> dict:
        if self._idx >= len(self._actions):
            return {"tool": "submit_patch", "patch": "--- a\n+++ b\n"}
        action = self._actions[self._idx]
        self._idx += 1
        return action


@pytest.fixture
def default_config():
    return BudgetConfig(total=10, cost_weights={"run_tests": 3, "submit_patch": 0})


@pytest.fixture
def ok_env():
    return MockEnvironment(oracle_status="OK_SOLVED")


@pytest.fixture
def failing_env():
    return MockEnvironment(oracle_status="TEST_FAIL")
