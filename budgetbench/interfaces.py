"""
BudgetBench — Core Interfaces
==============================
All three tracks develop against these contracts.
Rule: never break an interface without notifying the team and opening a
GitHub issue tagged [interface-change].

Defined by: Chenrui Xu (Project Lead)
Delivered:  2026-03-22  (Phase 1, Week 1)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Status codes — deterministic, reproducible
# ---------------------------------------------------------------------------

StatusCode = Literal[
    "OK_SOLVED",        # patch applied and all tests pass
    "INVALID_FORMAT",   # patch is not valid unified-diff
    "PATCH_APPLY_FAIL", # patch is valid diff but fails to apply
    "TEST_FAIL",        # patch applied but ≥1 test fails
    "BUDGET_EXCEEDED",  # agent ran out of tool-call budget
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Record of a single tool invocation."""
    name:   str   # "search" | "inspect" | "edit" | "run_tests" | "submit_patch" …
    args:   dict  # tool arguments as passed by the agent
    result: Any   # raw output returned by the environment
    cost:   int   # budget units charged for this call
    ts_ms:  int   # wall-clock timestamp in milliseconds


@dataclass
class Episode:
    """Complete record of one agent–environment interaction."""
    budget_total: int
    budget_used:  int
    tool_calls:   list[ToolCall] = field(default_factory=list)
    final_status: StatusCode     = "BUDGET_EXCEEDED"
    metadata:     dict           = field(default_factory=dict)

    @property
    def budget_remaining(self) -> int:
        return self.budget_total - self.budget_used

    @property
    def solved(self) -> bool:
        return self.final_status == "OK_SOLVED"


@dataclass
class BudgetConfig:
    """Controls how budget is allocated and charged."""
    total:        int            # total budget for one episode
    cost_weights: dict[str, int] = field(default_factory=lambda: {"run_tests": 3})
    # e.g. {"run_tests": 3, "search": 1, "inspect": 1, "edit": 1, "submit_patch": 0}

    def cost_of(self, tool_name: str) -> int:
        """Return the budget cost for the given tool (defaults to 1)."""
        return self.cost_weights.get(tool_name, 1)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class Environment(ABC):
    """
    Abstract environment — implemented by Hanyu Jin (Track A).

    Contract
    --------
    - reset()  →  initial observation dict (always call before stepping)
    - step()   →  (new observation dict, done: bool)
    - score()  →  deterministic StatusCode for a candidate patch string
    """

    @abstractmethod
    def reset(self) -> dict:
        """Return the initial observation for a fresh episode."""
        ...

    @abstractmethod
    def step(self, action: dict) -> tuple[dict, bool]:
        """
        Apply action and return (new_obs, done).

        Parameters
        ----------
        action : dict
            Must contain at least ``{"tool": "<name>", ...}`` keys.

        Returns
        -------
        obs : dict
            Updated observation for the agent.
        done : bool
            True if the episode is over (agent submitted a patch).
        """
        ...

    @abstractmethod
    def score(self, patch: str) -> StatusCode:
        """
        Deterministically evaluate a patch string.

        This is the oracle — it must be side-effect-free and reproducible.
        Called by BudgetRunner after the episode loop exits with done=True.
        """
        ...


class Agent(ABC):
    """
    Abstract agent — implemented by Sylvia Li (Track B).

    Contract
    --------
    - act() receives obs, budget_remaining, and the full call history,
      and returns the next action dict.
    """

    @abstractmethod
    def act(
        self,
        obs:              dict,
        budget_remaining: int,
        history:          list[ToolCall],
    ) -> dict:
        """
        Choose the next action.

        Parameters
        ----------
        obs : dict
            Current environment observation.
        budget_remaining : int
            Units left; agents should adapt strategy as this decreases.
        history : list[ToolCall]
            Immutable record of all prior tool calls in this episode.

        Returns
        -------
        action : dict
            Must contain ``{"tool": "<name>", ...}``.
        """
        ...
