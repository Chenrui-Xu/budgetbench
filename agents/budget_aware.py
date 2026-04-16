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
        limit = 4000 if tc.name in ("inspect", "edit", "run_tests") else 500
        result_str = str(tc.result)[:limit]
        lines.append(f"[{tc.name}] → {result_str}")
    full = "\n".join(lines)
    return full[-max_chars:] if len(full) > max_chars else full


def _has_tested_after_last_edit(history: list[ToolCall]) -> bool:
    last_edit_idx = -1
    for i, tc in enumerate(history):
        if tc.name == "edit":
            last_edit_idx = i
    if last_edit_idx == -1:
        return False
    return any(tc.name == "run_tests" for tc in history[last_edit_idx + 1:])


def _extract_current_diff(history: list[ToolCall]) -> str:
    """Return the latest current_diff from an edit call, or empty string."""
    for tc in reversed(history):
        if tc.name == "edit" and isinstance(tc.result, dict):
            diff = tc.result.get("current_diff", "")
            if diff:
                return diff
    return ""


_FORCE_PATCH_PROMPT = """\
You are an expert software engineer. Based on the code you have read, write a \
unified diff patch that fixes the issue below.

Output ONLY the unified diff. No JSON, no explanation, no markdown fences.
The diff must start with '--- ' and use standard unified diff format.

## Issue
{issue}

## Code you have read
{code_seen}

## Patch"""


def _force_generate_patch(model: str, issue: str, history: list[ToolCall]) -> str:
    """Make a focused LLM call to generate a unified diff when normal path fails."""
    code_parts = []
    for tc in history:
        if tc.name in ("inspect", "search", "edit"):
            result_str = str(tc.result)[:3000]
            code_parts.append(f"[{tc.name}]\n{result_str}")
    code_seen = "\n\n".join(code_parts[-6:])

    prompt = _FORCE_PATCH_PROMPT.format(issue=issue, code_seen=code_seen)
    raw = generate_with_retry(model, prompt).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if "\n" in raw:
            raw = raw[raw.index("\n")+1:]
        raw = raw.strip()

    if raw.startswith("---") or raw.startswith("diff --git"):
        return raw
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
        has_edited        = any(tc.name == "edit" for tc in history)
        has_inspected     = any(tc.name == "inspect" for tc in history)
        tested_after_edit = _has_tested_after_last_edit(history)

        # LOW: skip normal LLM loop, submit best available patch
        if regime == "LOW":
            saved_diff = _extract_current_diff(history)
            if saved_diff:
                return {"tool": "submit_patch", "patch": saved_diff}
            forced = _force_generate_patch(self.model, self._issue_ctx, history)
            return {"tool": "submit_patch", "patch": forced}

        regime_instruction = _REGIME_INSTRUCTIONS[regime]
        history_str        = _summarise_history(history)

        force_note = ""
        if has_inspected and not has_edited:
            force_note = (
                "\n\nYou have inspected the code but have NOT called edit yet. "
                "You MUST call edit with a unified diff to make the actual change. "
                "Do NOT call submit_patch before calling edit."
            )
        elif has_edited and not tested_after_edit and regime == "HIGH":
            force_note = (
                "\n\nYou have made an edit but have NOT run tests yet. "
                "Call run_tests to verify your fix."
            )

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"## Current budget regime: {regime}\n{regime_instruction}\n\n"
            f"## Issue\n{self._issue_ctx}\n\n"
            f"## Tool call history\n{history_str or '(none yet)'}\n\n"
            f"## Latest observation\n{json.dumps(obs, indent=2)[:4000]}\n\n"
            f"Budget remaining: {budget_remaining} / {self._budget_total}"
            f"{force_note}\n\n"
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
            if action.get("tool") == "submit_patch":
                # Step 1: must have edited
                if not has_edited and budget_remaining > 2:
                    diff = action.get("patch", "") or _force_generate_patch(
                        self.model, self._issue_ctx, history
                    )
                    if diff:
                        return {"tool": "edit", "patch": diff}

                # Step 2: must run tests after edit (HIGH regime only — MEDIUM saves budget)
                if has_edited and not tested_after_edit and regime == "HIGH" and budget_remaining > 3:
                    return {"tool": "run_tests"}

                # Step 3: submit using git diff as ground truth
                saved_diff = _extract_current_diff(history)
                if saved_diff:
                    return {"tool": "submit_patch", "patch": saved_diff}

                if not action.get("patch") and budget_remaining > 2:
                    forced = _force_generate_patch(self.model, self._issue_ctx, history)
                    if forced:
                        return {"tool": "submit_patch", "patch": forced}
            return action
        except Exception:
            saved_diff = _extract_current_diff(history)
            if saved_diff:
                return {"tool": "submit_patch", "patch": saved_diff}
            if budget_remaining > 2:
                forced = _force_generate_patch(self.model, self._issue_ctx, history)
                return {"tool": "submit_patch", "patch": forced}
            return {"tool": "submit_patch", "patch": ""}
