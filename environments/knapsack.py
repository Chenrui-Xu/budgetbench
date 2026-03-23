"""
BudgetBench — Knapsack / ILP Toy Environment  (Track A · Task A2)
===================================================================
A self-contained oracle-controlled environment for de-risking early
development and integration testing.

The task: given a list of items with weights and values, find the subset
that maximises total value without exceeding capacity W.

The agent interacts via tool calls:
  - list_items         → returns item catalogue
  - inspect_item id    → returns details for one item
  - add_item id        → adds item to current selection
  - remove_item id     → removes item from current selection
  - check_feasibility  → reports current weight vs capacity
  - submit_patch       → finalises the selection (ends episode)

"Patch" here is a JSON string like '{"selected": [0, 2, 5]}'.

Oracle: ground-truth computed via dynamic programming (exact optimal).

Implemented by: Hanyu Jin (Track A)
Reference stub provided by: Chenrui Xu
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Optional

from budgetbench.interfaces import Environment, StatusCode


@dataclass
class Item:
    id:     int
    name:   str
    weight: int
    value:  int


class KnapsackEnvironment(Environment):
    """
    Knapsack toy environment.

    Parameters
    ----------
    n_items   : number of items to generate (default 10)
    capacity  : knapsack weight capacity (default 50)
    seed      : random seed for reproducibility
    """

    def __init__(
        self,
        n_items:  int = 10,
        capacity: int = 50,
        seed:     int = 42,
    ) -> None:
        self.n_items  = n_items
        self.capacity = capacity
        self.seed     = seed
        self._items:    list[Item]    = []
        self._selected: set[int]      = set()
        self._optimal:  Optional[int] = None  # cached optimal value

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Generate a fresh instance and return the initial observation."""
        rng = random.Random(self.seed)
        self._items = [
            Item(
                id=i,
                name=f"item_{i}",
                weight=rng.randint(1, 20),
                value=rng.randint(1, 30),
            )
            for i in range(self.n_items)
        ]
        self._selected = set()
        self._optimal  = None

        return {
            "task": "knapsack",
            "n_items":  self.n_items,
            "capacity": self.capacity,
            "message":  "New knapsack instance. Use list_items to see the catalogue.",
        }

    def step(self, action: dict) -> tuple[dict, bool]:
        """Execute one tool call and return (observation, done)."""
        tool = action.get("tool", "unknown")

        if tool == "list_items":
            obs = {
                "items": [
                    {"id": it.id, "name": it.name, "weight": it.weight, "value": it.value}
                    for it in self._items
                ]
            }
            return obs, False

        if tool == "inspect_item":
            item_id = int(action.get("id", -1))
            item    = self._find(item_id)
            if item is None:
                return {"error": f"Item {item_id} not found"}, False
            return {
                "id":       item.id,
                "name":     item.name,
                "weight":   item.weight,
                "value":    item.value,
                "selected": item.id in self._selected,
            }, False

        if tool == "add_item":
            item_id = int(action.get("id", -1))
            item    = self._find(item_id)
            if item is None:
                return {"error": f"Item {item_id} not found"}, False
            self._selected.add(item_id)
            return {"added": item_id, "current_weight": self._current_weight()}, False

        if tool == "remove_item":
            item_id = int(action.get("id", -1))
            self._selected.discard(item_id)
            return {"removed": item_id, "current_weight": self._current_weight()}, False

        if tool == "check_feasibility":
            w = self._current_weight()
            return {
                "current_weight": w,
                "capacity":       self.capacity,
                "feasible":       w <= self.capacity,
                "current_value":  self._current_value(),
                "selected_ids":   sorted(self._selected),
            }, False

        if tool == "submit_patch":
            # Accept either a patch string from action or the current selection
            patch_str = action.get("patch", "")
            if patch_str:
                try:
                    parsed = json.loads(patch_str)
                    self._selected = set(parsed.get("selected", []))
                except (json.JSONDecodeError, TypeError):
                    pass  # score() will catch the infeasibility
            return {
                "submitted": True,
                "selected_ids":  sorted(self._selected),
                "total_weight":  self._current_weight(),
                "total_value":   self._current_value(),
            }, True  # done = True

        return {"error": f"Unknown tool: {tool}"}, False

    def score(self, patch: str) -> StatusCode:
        """
        Score the submitted selection deterministically.

        Returns
        -------
        OK_SOLVED          if selection is feasible and value ≥ optimal value
        TEST_FAIL          if feasible but sub-optimal
        INVALID_FORMAT     if patch JSON is malformed
        PATCH_APPLY_FAIL   if any selected item id is invalid
        """
        # --- parse patch -----------------------------------------------
        try:
            data = json.loads(patch) if patch else {"selected": sorted(self._selected)}
            selected_ids = set(data.get("selected", []))
        except (json.JSONDecodeError, TypeError):
            return "INVALID_FORMAT"

        # --- validate ids -----------------------------------------------
        valid_ids = {it.id for it in self._items}
        if not selected_ids.issubset(valid_ids):
            return "PATCH_APPLY_FAIL"

        # --- check feasibility ------------------------------------------
        total_weight = sum(self._items[i].weight for i in selected_ids)
        if total_weight > self.capacity:
            return "TEST_FAIL"

        # --- check optimality -------------------------------------------
        total_value  = sum(self._items[i].value for i in selected_ids)
        optimal      = self._compute_optimal()
        if total_value >= optimal:
            return "OK_SOLVED"
        return "TEST_FAIL"

    # ------------------------------------------------------------------
    # Oracle
    # ------------------------------------------------------------------

    def _compute_optimal(self) -> int:
        """Exact 0/1 knapsack via dynamic programming."""
        if self._optimal is not None:
            return self._optimal

        W = self.capacity
        n = len(self._items)
        dp = [0] * (W + 1)
        for item in self._items:
            for w in range(W, item.weight - 1, -1):
                dp[w] = max(dp[w], dp[w - item.weight] + item.value)
        self._optimal = dp[W]
        return self._optimal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find(self, item_id: int) -> Optional[Item]:
        for it in self._items:
            if it.id == item_id:
                return it
        return None

    def _current_weight(self) -> int:
        return sum(self._items[i].weight for i in self._selected)

    def _current_value(self) -> int:
        return sum(self._items[i].value for i in self._selected)

    @property
    def optimal_value(self) -> int:
        """Expose oracle value (useful for evaluation)."""
        return self._compute_optimal()
