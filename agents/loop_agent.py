"""
BudgetBench — LoopAgent  (Track B · Task B2)
=============================================
Baseline 2: standard agentic coding loop — search → inspect → edit → test.

The agent keeps calling tools until it decides to submit or runs out of budget.
This mirrors the behaviour of systems like SWE-agent and OpenHands.
"""

from __future__ import annotations

import json
import os

from budgetbench.interfaces import Agent, ToolCall
from agents._gemini import generate_with_retry
from agents._shared import (
    has_tested_after_last_edit as _has_tested_after_last_edit,
    extract_current_diff as _extract_current_diff,
    force_generate_patch as _force_generate_patch,
)


_DEFAULT_MODEL = "gemini-2.5-flash"

_SYSTEM_PROMPT = """\
You are an expert software engineer working inside a GitHub repository.

Your task is to fix the given GitHub issue by exploring the codebase and
producing a unified-diff patch that makes the tests pass.

You operate as an autonomous coding agent that can call tools to explore,
modify code, and validate fixes.

At each step you MUST output a JSON object (and ONLY that JSON object)
describing the next tool call.

Available tools:

{"tool": "search", "query": "<keyword>", "path": "<optional subdir>"}

Search for files, symbols, or keywords related to the issue.


{"tool": "inspect", "file": "<path>", "start": <int>, "end": <int>}

Read the contents of a specific file and line range.


{"tool": "hierarchy", "depth": <int>}

Inspect the repository structure to understand where code lives.


{"tool": "edit", "patch": "<unified diff hunk>"}

Apply a code modification using a unified diff.


{"tool": "run_tests"}

Run the repository's test suite to verify whether the bug is fixed.


{"tool": "submit_patch", "patch": "<full unified diff>"}

Submit the final patch once you believe the fix is correct.
After calling edit, the observation includes "current_diff" showing all changes
made so far. Copy that value directly into the "patch" field of submit_patch.



Guidelines for tool usage:

1. Start by understanding the issue description and identifying where in the
   codebase the bug may be located.

2. Use **search** to locate relevant files, functions, or error messages.

3. Use **inspect** to read the relevant code sections and understand how the
   system works.

4. Once you have enough evidence about the likely bug location, prefer **edit**
   instead of repeatedly searching or inspecting.

5. After making a meaningful code change, use **run_tests** to check whether
   the fix works.

6. Avoid repeating search or inspect unnecessarily once you already understand
   the relevant code.

7. Avoid long exploration loops. Do not perform more than a few consecutive
   search or inspect calls.

8. If the budget is limited, prioritize attempting a fix rather than exploring
   indefinitely.

9. You MUST call **edit** before calling **submit_patch**. Never submit without
   having made at least one edit. After edit succeeds, the observation contains
   "current_diff" — copy that value exactly into submit_patch's "patch" field.

10. Call **submit_patch** when:
    - you have called edit and believe the patch fixes the issue, OR
    - the remaining budget is very small (but still only after calling edit).

Rules:

- Always return a valid JSON object.
- Never output explanations or text outside the JSON object.
- The JSON must contain the key "tool".
- Prefer making progress toward a fix instead of endless exploration.
- NEVER call submit_patch before calling edit at least once.
"""

def _summarise_history(history: list[ToolCall]) -> str:
    lines = []
    for tc in history:
        limit = 4000 if tc.name in ("inspect", "edit", "run_tests") else 500
        result_str = str(tc.result)[:limit]
        lines.append(f"[{tc.name}] → {result_str}")
    return "\n".join(lines)


class LoopAgent(Agent):
    """
    Standard tool-using loop agent.

    Uses Gemini to decide the next tool call at each step, given the
    current observation and full history.

    Parameters
    ----------
    model : str
        Gemini model identifier.
    max_search_calls : int
        Upper bound on search calls before forcing a submit.
    """

    def __init__(
        self,
        model:            str = _DEFAULT_MODEL,
        max_search_calls: int = 5,
    ) -> None:
        self.model            = model
        self.max_search_calls = max_search_calls
        self._issue_ctx: str  = ""

    def act(
        self,
        obs:              dict,
        budget_remaining: int,
        history:          list[ToolCall],
    ) -> dict:
        # Capture issue context on first call
        if not history:
            self._issue_ctx = (
                obs.get("problem_statement")
                or obs.get("issue", "No issue description provided.")
            )

        # Count how many times we've already retried an empty patch
        empty_retries = sum(
            1 for tc in history
            if tc.name == "search" and str(tc.args.get("query", "")).startswith("__empty_retry__")
        )
        has_edited    = any(tc.name == "edit" for tc in history)
        has_inspected = any(tc.name == "inspect" for tc in history)
        tested_after_edit = _has_tested_after_last_edit(history)

        history_str = _summarise_history(history)

        force_note = ""
        if budget_remaining <= 3:
            force_note = "\n\nBudget is almost gone. Call submit_patch NOW with your best patch."
        elif empty_retries > 0:
            force_note = (
                f"\n\nYou have submitted an empty patch {empty_retries} time(s). "
                "You MUST write actual code changes in the 'patch' field. "
                "Based on the code you have read, write the fix as a unified diff now."
            )
        elif has_inspected and not has_edited:
            force_note = (
                "\n\nYou have inspected the code but have NOT called edit yet. "
                "You MUST call edit with a unified diff to make the actual change. "
                "Do NOT call submit_patch before calling edit."
            )
        elif has_edited and not tested_after_edit:
            force_note = (
                "\n\nYou have made an edit but have NOT run tests yet. "
                "You MUST call run_tests to verify your fix before submitting."
            )

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"## Issue\n{self._issue_ctx}\n\n"
            f"## Tool call history\n{history_str or '(none yet)'}\n\n"
            f"## Latest observation\n{json.dumps(obs, indent=2)[:4000]}\n\n"
            f"Budget remaining: {budget_remaining}"
            f"{force_note}\n\n"
            f"## Next action (JSON only)"
        )

        raw = generate_with_retry(self.model, prompt).strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            action = json.loads(raw)
            if "tool" not in action:
                raise ValueError("Missing 'tool' key")
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
            if action.get("tool") == "submit_patch":
                # Step 1: must have edited
                if not has_edited and budget_remaining > 2:
                    diff = action.get("patch", "") or _force_generate_patch(
                        self.model, self._issue_ctx, history
                    )
                    if diff:
                        return {"tool": "edit", "patch": diff}
                    return {"tool": "search", "query": f"__empty_retry__{empty_retries}__", "path": "."}

                # Step 2: must have run tests after last edit
                if has_edited and not tested_after_edit and budget_remaining > 3:
                    return {"tool": "run_tests"}

                # Step 3: submit using git diff as ground truth
                saved_diff = _extract_current_diff(history)
                if saved_diff:
                    return {"tool": "submit_patch", "patch": saved_diff}

                if not action.get("patch") and budget_remaining > 2 and empty_retries < 2:
                    forced = _force_generate_patch(self.model, self._issue_ctx, history)
                    if forced:
                        return {"tool": "submit_patch", "patch": forced}
                    return {"tool": "search", "query": f"__empty_retry__{empty_retries}__", "path": "."}
            return action
        except Exception:
            saved_diff = _extract_current_diff(history)
            if saved_diff:
                return {"tool": "submit_patch", "patch": saved_diff}
            if budget_remaining > 2 and empty_retries < 2:
                forced = _force_generate_patch(self.model, self._issue_ctx, history)
                if forced:
                    return {"tool": "submit_patch", "patch": forced}
                return {"tool": "search", "query": f"__empty_retry__{empty_retries}__", "path": "."}
            return {"tool": "submit_patch", "patch": ""}
