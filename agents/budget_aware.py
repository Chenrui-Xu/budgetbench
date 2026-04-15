"""
BudgetBench — BudgetAwareAgent  (Track B · Task B3)  ★ Key Novelty
====================================================================
The budget-aware agent monitors ``budget_remaining`` in real time and
switches strategy as the budget shrinks.

Budget regimes
--------------
HIGH   (remaining > 0.6 × total):  thorough exploration
        - broad search, multiple file inspections, run_tests allowed
MEDIUM (0.3 < remaining ≤ 0.6):   focused mode
        - targeted search, minimal inspections, no run_tests
LOW    (remaining ≤ 0.3):          emergency mode
        - skip exploration, submit best patch constructed so far

This is the primary technical contribution of BudgetBench.
"""

from __future__ import annotations

import json
import os

from budgetbench.interfaces import Agent, ToolCall
from agents._gemini import generate_with_retry


_DEFAULT_MODEL    = "gemini-2.5-flash"
_HIGH_THRESHOLD   = 0.6
_MEDIUM_THRESHOLD = 0.3

_REGIME_INSTRUCTIONS = {
    "HIGH": (
        "You have ample budget. Explore thoroughly:\n"
        "- Use search + inspect to understand the codebase\n"
        "- run_tests is allowed to verify fixes\n"
        "- Take your time to find the root cause"
    ),
    "MEDIUM": (
        "Budget is limited. Stay focused:\n"
        "- Do targeted search only — no broad exploration\n"
        "- Avoid run_tests (costs 3 budget)\n"
        "- Move toward submitting a patch"
    ),
    "LOW": (
        "Budget is critical. Submit immediately:\n"
        "- Do NOT call any exploration tools\n"
        "- Submit the best patch you can based on what you know\n"
        "- Use submit_patch NOW"
    ),
}

_SYSTEM_PROMPT = """\
You are an expert software engineer working inside a code repository.
Fix the given GitHub issue by exploring the codebase and producing a patch.

At each step output a JSON object (and ONLY that JSON object):

  {"tool": "search",   "query": "<keyword>", "path": "<optional subdir>"}
  {"tool": "inspect",  "file": "<path>", "start": <int>, "end": <int>}
  {"tool": "hierarchy", "depth": <int>}
  {"tool": "edit",     "patch": "<unified diff hunk>"}
  {"tool": "run_tests"}
  {"tool": "submit_patch", "patch": "<full unified diff>"}"""


def _summarise_history(history: list[ToolCall], max_chars: int = 6000) -> str:
    lines = []
    for tc in history:
        limit = 4000 if tc.name in ("inspect", "edit") else 500
        result_str = str(tc.result)[:limit]
        lines.append(f"[{tc.name}] → {result_str}")
    full = "\n".join(lines)
    return full[-max_chars:] if len(full) > max_chars else full


def _extract_current_diff(history: list[ToolCall]) -> str:
    """Return the latest current_diff from an edit call, or empty string."""
    for tc in reversed(history):
        if tc.name == "edit" and isinstance(tc.result, dict):
            diff = tc.result.get("current_diff", "")
            if diff:
                return diff
    return ""


class BudgetAwareAgent(Agent):
    """
    Budget-adaptive agent — the key novelty of this project.

    Parameters
    ----------
    model : str
        Gemini model identifier.
    budget_total : int
        Total budget declared upfront. Inferred from first act() call if 0.
    """

    def __init__(
        self,
        model:        str = _DEFAULT_MODEL,
        budget_total: int = 0,
    ) -> None:
        self.model         = model
        self._budget_total = budget_total
        self._issue_ctx:   str = ""

    def _regime(self, budget_remaining: int) -> str:
        total = self._budget_total or 1
        frac  = budget_remaining / total
        if frac > _HIGH_THRESHOLD:
            return "HIGH"
        if frac > _MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    def act(
        self,
        obs:              dict,
        budget_remaining: int,
        history:          list[ToolCall],
    ) -> dict:
        # Infer total budget on first call
        if self._budget_total == 0:
            self._budget_total = budget_remaining

        # Capture issue context on first call
        if not history:
            self._issue_ctx = (
                obs.get("problem_statement")
                or obs.get("issue", "No issue description provided.")
            )

        regime = self._regime(budget_remaining)

        # LOW: skip LLM call, submit best available patch
        if regime == "LOW":
            saved_diff = _extract_current_diff(history)
            return {"tool": "submit_patch", "patch": saved_diff}

        regime_instruction = _REGIME_INSTRUCTIONS[regime]
        history_str        = _summarise_history(history)

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"## Current budget regime: {regime}\n{regime_instruction}\n\n"
            f"## Issue\n{self._issue_ctx}\n\n"
            f"## Tool call history\n{history_str or '(none yet)'}\n\n"
            f"## Latest observation\n{json.dumps(obs, indent=2)[:800]}\n\n"
            f"Budget remaining: {budget_remaining} / {self._budget_total}\n\n"
            f"## Next action (JSON only)"
        )

        raw = generate_with_retry(self.model, prompt).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            action = json.loads(raw)
            if "tool" not in action:
                raise ValueError("Missing 'tool' key")
            # In MEDIUM regime, block run_tests to save budget
            if regime == "MEDIUM" and action.get("tool") == "run_tests":
                action = {"tool": "submit_patch", "patch": ""}
            # Strip markdown fences from patch content if LLM wrapped it
            if action.get("tool") == "submit_patch" and action.get("patch"):
                patch = action["patch"].strip()
                if patch.startswith("```"):
                    parts = patch.split("```")
                    patch = parts[1] if len(parts) > 1 else ""
                    if "\n" in patch:
                        first_line = patch[:patch.index("\n")]
                        if first_line.strip() in ("", "diff", "patch", "python", "py"):
                            patch = patch[patch.index("\n")+1:]
                    action["patch"] = patch.strip()
            # If patch is still empty, fall back to saved diff from prior edit
            if action.get("tool") == "submit_patch" and not action.get("patch"):
                saved_diff = _extract_current_diff(history)
                if saved_diff:
                    action["patch"] = saved_diff
            return action
        except Exception:
            saved_diff = _extract_current_diff(history)
            return {"tool": "submit_patch", "patch": saved_diff}
