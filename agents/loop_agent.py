"""
BudgetBench — LoopAgent + shared base  (Track B)
=================================================
_BaseLoopAgent implements the full inspect→edit→test loop.
LoopAgent is a thin subclass with no budget-regime awareness.
BudgetAwareAgent (in budget_aware.py) subclasses _BaseLoopAgent and
overrides three hook methods to add regime-based strategy switching.
"""

from __future__ import annotations

import json

from budgetbench.interfaces import Agent, ToolCall
from agents._gemini import generate_with_retry
from agents._shared import (
    has_tested_after_last_edit as _has_tested_after_last_edit,
    extract_current_diff as _extract_current_diff,
    force_generate_patch as _force_generate_patch,
)


_DEFAULT_MODEL = "gemini-2.5-flash"

_SYSTEM_PROMPT = """\
You are an expert software engineer fixing a real GitHub issue inside a live repository.

At each step output a JSON object (and ONLY that JSON object) for the next tool call.

━━━ AVAILABLE TOOLS ━━━

{"tool": "search",   "query": "<keyword>", "path": "<optional subdir>", "include": "<glob>"}
  Grep for files, symbols, or error strings across the repo.
  "include" defaults to "*.py". Use "*.cfg,*.toml,*.ini" for config files,
  "*.py,*.rst" for docs, or "*" for all files.

{"tool": "inspect",  "file": "<path>", "start": <int>, "end": <int>}
  Read exact lines of a file (1-indexed, inclusive).

{"tool": "hierarchy", "depth": <int>}
  Print the directory tree to understand repo layout.

{"tool": "edit",     "patch": "<unified diff>"}
  Apply a unified-diff patch to the working tree.
  The observation will contain:
    "success": true/false
    "message": error details if false (includes actual file lines near the failure)
    "current_diff": the full git diff of ALL changes made so far
    "post_edit_content": (on success) the actual file lines around the edited area
      — use this instead of calling inspect again to know the current file state

{"tool": "run_tests"}
  Run the issue's FAIL_TO_PASS test suite.
  The observation will contain:
    "run_status": "passed" | "failed" | "not_run"
      "not_run" means the environment couldn't run the tests (import error,
      test IDs don't exist yet at base_commit, etc.) — this is NOT a signal
      that your fix is wrong. Use run_script to verify instead.
    "passed": true/false
    "output": test output

{"tool": "run_script", "script": "<python code>"}
  Run a Python script you write to verify your fix.
  Use this when run_tests returns "not_run", or to do a quick targeted check.
  Example:
    {"tool": "run_script", "script": "from astropy.modeling import ...\nresult = ...\nassert result == expected, f'got {result}'\nprint('OK')"}
  The observation will contain "run_status", "passed", and "output".

{"tool": "submit_patch", "patch": "<full unified diff>"}
  Submit the final patch. Use the "current_diff" value from the last successful
  edit — do NOT write your own diff here.

━━━ WORKFLOW ━━━

  STEP 1 — UNDERSTAND
    • Search for the relevant file(s).
    • Inspect the EXACT lines you plan to change (at least 20 lines of context).
    • Do not edit until you know: "the bug is on line N of file F because X."

  STEP 2 — EDIT
    • Write a unified diff. Context lines (space-prefixed) must be copied
      EXACTLY from your inspect output — character for character.
    • If the edit succeeds (success=true), the observation includes
      "post_edit_content" showing the current file lines around the edit.
      Use this to plan your next patch — do NOT inspect the same location again.
    • If the edit fails (success=false), the message shows the actual file
      content. Re-inspect that location, then write a corrected patch.

  STEP 3 — VERIFY BEFORE SUBMITTING
    • Before submit_patch, always run run_tests (or run_script) once to verify.
    • run_status == "passed"  → submit_patch immediately.
    • run_status == "failed"  → read traceback, re-inspect, edit, then test again.
    • run_status == "not_run" → environment issue, not your fault. Use run_script
      to write a targeted Python check, or submit if confident in your fix.
    • Do NOT run tests multiple times in a row without an edit in between.

  STEP 4 — SUBMIT
    • Use the exact "current_diff" value from your last successful edit.

━━━ PATCH FORMAT ━━━

  --- a/path/to/file.py
  +++ b/path/to/file.py
  @@ -N,M +N,M @@
   exact context line (space prefix, copied from inspect)
  -removed line (exact copy from file)
  +added line (your fix)
   exact context line (space prefix, copied from inspect)

Rules: ≥3 context lines above/below each change. Never invent context.

━━━ OUTPUT RULES ━━━
- Valid JSON only. No text outside the JSON.
- submit_patch requires a successful edit first.
- After edit fails: inspect → edit (do NOT skip inspect).
- After tests fail: inspect → edit → run_tests (do NOT skip any step).
"""


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def _summarise_history(history: list[ToolCall]) -> str:
    """
    Render the tool-call history for the prompt.

    Edit results are formatted specially so that success/message/current_diff
    are always readable (not buried inside a truncated dict repr).
    Inspect and run_tests results use recency-aware limits.
    """
    lines = []
    n = len(history)
    for i, tc in enumerate(history):
        age = n - 1 - i  # 0 = most recent

        if tc.name == "edit" and isinstance(tc.result, dict):
            # Always show success + message so the agent knows whether each
            # edit succeeded.  Show current_diff with recency-aware length —
            # the most recent edit's diff is the ground truth for submit_patch.
            # Also show post_edit_content so the agent knows the current file
            # state after the edit without needing a separate inspect call.
            success = tc.result.get("success", False)
            message = tc.result.get("message", "")[:300]
            diff    = tc.result.get("current_diff", "")
            post    = tc.result.get("post_edit_content", "")
            diff_limit = 8000 if age == 0 else 4000 if age <= 2 else 1500 if age <= 5 else 600
            diff_display = diff[:diff_limit] + ("…" if len(diff) > diff_limit else "")
            result_str = (
                f"success={success}, message={message!r},\n"
                f"current_diff=\n{diff_display}"
            )
            if post and age <= 2:
                post_limit = 3000 if age == 0 else 1500
                post_display = post[:post_limit] + ("…" if len(post) > post_limit else "")
                result_str += f"\npost_edit_content (current file state):\n{post_display}"
        elif tc.name == "inspect":
            limit = 12000 if age == 0 else 8000 if age <= 2 else 2000 if age <= 5 else 600
            result_str = str(tc.result)[:limit]
        elif tc.name in ("run_tests", "run_script"):
            limit = 6000 if age == 0 else 3000 if age <= 2 else 800
            result_str = str(tc.result)[:limit]
        else:
            result_str = str(tc.result)[:600]

        lines.append(f"[{tc.name}] → {result_str}")
    return "\n".join(lines)


def _has_tested_after_last_edit(history: list[ToolCall]) -> bool:
    """Return True if run_tests was called after the most recent edit."""
    last_edit_idx = -1
    for i, tc in enumerate(history):
        if tc.name == "edit":
            last_edit_idx = i
    if last_edit_idx == -1:
        return False
    return any(tc.name == "run_tests" for tc in history[last_edit_idx + 1:])


def _last_edit_status(history: list[ToolCall]) -> tuple[bool | None, str]:
    """Return (success, message) for the most recent edit, or (None, "")."""
    for tc in reversed(history):
        if tc.name == "edit":
            if isinstance(tc.result, dict):
                return tc.result.get("success", False), tc.result.get("message", "")
            return False, str(tc.result)
    return None, ""


def _last_test_status(history: list[ToolCall]) -> tuple[bool | None, str]:
    """Return (passed, output) for the most recent run_tests, or (None, "").

    Returns None (not False) when run_status == "not_run" so force-note logic
    does not treat environment failures as real test failures.
    """
    for tc in reversed(history):
        if tc.name in ("run_tests", "run_script"):
            if isinstance(tc.result, dict):
                run_status = tc.result.get("run_status", "")
                if run_status == "not_run":
                    return None, tc.result.get("output", "")
                return tc.result.get("passed", False), tc.result.get("output", "")
            return False, str(tc.result)
    return None, ""


def _run_tests_since_last_edit(history: list[ToolCall]) -> int:
    """Count consecutive run_tests/run_script calls since the last edit (or from start)."""
    count = 0
    for tc in reversed(history):
        if tc.name == "edit":
            break
        if tc.name in ("run_tests", "run_script"):
            count += 1
    return count


def _last_inspected_file(history: list[ToolCall]) -> tuple[str, int]:
    """Return (file_path, start_line) from the most recent inspect call."""
    for tc in reversed(history):
        if tc.name == "inspect" and isinstance(tc.args, dict):
            return tc.args.get("file", ""), int(tc.args.get("start", 1))
    return "", 1


def _extract_current_diff(history: list[ToolCall]) -> str:
    """Return the latest non-empty current_diff from any edit call."""
    for tc in reversed(history):
        if tc.name == "edit" and isinstance(tc.result, dict):
            diff = tc.result.get("current_diff", "")
            if diff:
                return diff
    return ""


def _is_patch_complete(patch: str) -> bool:
    """Return False if the patch looks truncated (does not end with newline)."""
    if not patch or not patch.strip():
        return False
    return patch.endswith("\n")


def _count_edit_redirects(history: list[ToolCall]) -> int:
    """Count how many times a submit_patch was silently redirected to edit."""
    return sum(
        1 for tc in history
        if tc.name == "edit" and tc.args.get("_redirect")
    )


_FORCE_PATCH_PROMPT = """\
You are an expert software engineer. Based on the code you have read, write a \
unified diff patch that fixes the issue below.

CRITICAL RULES:
- Output ONLY the unified diff. No JSON, no explanation, no markdown fences.
- The diff must start with '--- ' and use standard unified diff format.
- Context lines (lines starting with a space) must EXACTLY match the file content.
- Every hunk must be complete — do NOT stop mid-hunk.
- The patch must end with a newline character.

## Issue
{issue}

## Code you have read
{code_seen}

## Patch (complete unified diff, ending with newline)"""


def _force_generate_patch(model: str, issue: str, history: list[ToolCall]) -> str:
    """Focused LLM call to produce a unified diff as a last-resort fallback."""
    code_parts = []
    for tc in history:
        if tc.name in ("inspect", "search", "edit"):
            code_parts.append(f"[{tc.name}]\n{str(tc.result)[:3000]}")
    code_seen = "\n\n".join(code_parts[-6:])

    raw = generate_with_retry(model, _FORCE_PATCH_PROMPT.format(
        issue=issue, code_seen=code_seen
    )).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith(("diff", "patch", "python", "py", "\n")):
            raw = raw[raw.index("\n") + 1:] if "\n" in raw else raw
        raw = raw.strip()

    if not (raw.startswith("---") or raw.startswith("diff --git")):
        return ""
    if not raw.endswith("\n"):
        raw = raw + "\n"
    return raw


# ---------------------------------------------------------------------------
# Base agent — all loop logic lives here
# ---------------------------------------------------------------------------

class _BaseLoopAgent(Agent):
    """
    Shared implementation of the inspect → edit → run_tests loop.

    Subclasses override three lightweight hooks to inject budget-regime
    behaviour without duplicating any loop logic:

      _regime_section(budget_remaining) → str
          Extra block appended to the prompt (e.g. "## Regime: MEDIUM …").

      _pre_act(budget_remaining, history) → dict | None
          If not None, return this action immediately (used for LOW regime).

      _allow_run_tests(budget_remaining) → bool
          Return False to block run_tests calls (e.g. MEDIUM regime).
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self.model = model
        self._issue_ctx: str = ""

    # ── hook points ───────────────────────────────────────────────────────

    def _regime_section(self, budget_remaining: int) -> str:
        return ""

    def _pre_act(
        self, budget_remaining: int, history: list[ToolCall]
    ) -> dict | None:
        return None

    def _allow_run_tests(self, budget_remaining: int) -> bool:
        return True

    # ── main loop ─────────────────────────────────────────────────────────

    def act(
        self,
        obs:              dict,
        budget_remaining: int,
        history:          list[ToolCall],
    ) -> dict:
        # Always set issue context before any hook so _pre_act can use it.
        if not history:
            self._issue_ctx = (
                obs.get("problem_statement")
                or obs.get("issue", "No issue description provided.")
            )

        early = self._pre_act(budget_remaining, history)
        if early is not None:
            return early

        # ── state flags ───────────────────────────────────────────────────
        empty_retries       = sum(
            1 for tc in history
            if tc.name == "search"
            and str(tc.args.get("query", "")).startswith("__empty_retry__")
        )
        edit_redirects      = _count_edit_redirects(history)
        has_edited          = any(tc.name == "edit" for tc in history)
        has_inspected       = any(tc.name == "inspect" for tc in history)
        inspect_count       = sum(1 for tc in history if tc.name == "inspect")
        runtests_since_edit = _run_tests_since_last_edit(history)

        last_edit_ok,     last_edit_msg = _last_edit_status(history)
        last_test_passed, last_test_out = _last_test_status(history)

        # ── force note (injected into prompt) ─────────────────────────────
        force_note = self._build_force_note(
            budget_remaining, has_edited, has_inspected,
            runtests_since_edit, empty_retries,
            last_edit_ok, last_edit_msg, last_test_passed, last_test_out,
            inspect_count=inspect_count,
        )

        # ── build prompt ──────────────────────────────────────────────────
        history_str  = _summarise_history(history)
        regime_block = self._regime_section(budget_remaining)
        last_tool    = history[-1].name if history else ""
        obs_limit    = (
            12000 if last_tool in ("inspect", "run_tests")
            else 6000 if last_tool == "edit"
            else 4000
        )

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            + (f"{regime_block}\n\n" if regime_block else "")
            + f"## Issue\n{self._issue_ctx}\n\n"
            f"## Tool call history\n{history_str or '(none yet)'}\n\n"
            f"## Latest observation\n{json.dumps(obs, indent=2)[:obs_limit]}\n\n"
            f"Budget remaining: {budget_remaining}"
            f"{force_note}\n\n"
            f"## Next action (JSON only)"
        )

        raw = generate_with_retry(self.model, prompt).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # ── parse + intercept ─────────────────────────────────────────────
        try:
            action = json.loads(raw)
            if "tool" not in action:
                raise ValueError("Missing 'tool' key")

            # Clean markdown fences from patch field
            if action.get("tool") == "submit_patch" and action.get("patch"):
                patch = action["patch"].strip()
                if patch.startswith("```"):
                    parts = patch.split("```")
                    patch = parts[1] if len(parts) > 1 else ""
                    if "\n" in patch:
                        first_line = patch[:patch.index("\n")]
                        if first_line.strip() in ("", "diff", "patch", "python", "py"):
                            patch = patch[patch.index("\n") + 1:]
                action["patch"] = patch.strip()

            action = self._intercept(
                action, budget_remaining, history,
                has_edited, edit_redirects,
                runtests_since_edit, empty_retries,
                last_test_passed, last_test_out,
            )
            return action

        except Exception:
            saved_diff = _extract_current_diff(history)
            if saved_diff:
                return {"tool": "submit_patch", "patch": saved_diff}
            if budget_remaining > 2 and empty_retries < 2:
                forced = _force_generate_patch(self.model, self._issue_ctx, history)
                if forced and _is_patch_complete(forced):
                    return {"tool": "submit_patch", "patch": forced}
                return {
                    "tool": "search",
                    "query": f"__empty_retry__{empty_retries}__",
                    "path": ".",
                }
            return {"tool": "submit_patch", "patch": ""}

    # ── force note builder ────────────────────────────────────────────────

    def _build_force_note(
        self,
        budget_remaining:    int,
        has_edited:          bool,
        has_inspected:       bool,
        runtests_since_edit: int,
        empty_retries:       int,
        last_edit_ok:        bool | None,
        last_edit_msg:       str,
        last_test_passed:    bool | None,
        last_test_out:       str,
        inspect_count:       int = 0,
    ) -> str:
        if budget_remaining <= 3:
            return "\n\nBudget is almost gone. Call submit_patch NOW with your best patch."

        if last_edit_ok is False and has_edited:
            return (
                "\n\n⚠ YOUR LAST EDIT FAILED TO APPLY.\n"
                f"Reason: {last_edit_msg[:600]}\n"
                "Context lines do not match the actual file.\n"
                "You MUST: (1) inspect the file at the lines in the error, "
                "(2) copy exact content into the patch context, (3) retry edit."
            )

        if last_test_passed is False and runtests_since_edit >= 2:
            snippet = last_test_out[:600] if last_test_out else "(no output)"
            return (
                f"\n\n🚫 STOP. You have run tests {runtests_since_edit} times without "
                "any code change. Running tests again will NOT fix the failure.\n"
                f"Last failure:\n{snippet}\n\n"
                "You MUST inspect and edit before running tests again."
            )

        if last_test_passed is False and has_edited:
            snippet = last_test_out[:800] if last_test_out else "(no output)"
            return (
                "\n\n🚫 TESTS FAILED. You cannot submit yet.\n"
                f"Test output:\n{snippet}\n"
                "Inspect the relevant code, make a corrective edit, then test again."
            )

        if last_test_passed is None and runtests_since_edit >= 1:
            return (
                "\n\n⚠ Tests could not run (run_status=not_run) — environment issue, "
                "not a signal your fix is wrong.\n"
                "Use run_script to verify directly, or submit_patch if confident."
            )

        if empty_retries > 0:
            return (
                f"\n\nYou have submitted an empty patch {empty_retries} time(s). "
                "Write actual code changes as a unified diff."
            )

        # Only nudge toward edit when the agent has been inspecting for a long
        # time without making any change — not after every single inspect call.
        if inspect_count >= 4 and not has_edited:
            return (
                f"\n\nYou have called inspect {inspect_count} times without making "
                "any edit. If you have identified the bug location, call edit now. "
                "If you still need to look at more code, keep inspecting — but "
                "do not inspect indefinitely without attempting a fix."
            )

        return ""

    # ── action interceptor ────────────────────────────────────────────────

    def _intercept(
        self,
        action:              dict,
        budget_remaining:    int,
        history:             list[ToolCall],
        has_edited:          bool,
        edit_redirects:      int,
        runtests_since_edit: int,
        empty_retries:       int,
        last_test_passed:    bool | None,
        last_test_out:       str,
    ) -> dict:
        tool = action.get("tool")

        # ── submit_patch gates ────────────────────────────────────────────
        if tool == "submit_patch":
            # Gate 1: must have at least one edit
            if not has_edited and budget_remaining > 2:
                diff = action.get("patch", "") or _force_generate_patch(
                    self.model, self._issue_ctx, history
                )
                if diff:
                    return {"tool": "edit", "patch": diff}
                return {
                    "tool": "search",
                    "query": f"__empty_retry__{empty_retries}__",
                    "path": ".",
                }

            # Gate 1.5: pre-submit test check
            # — no test run since last edit → run tests first
            if runtests_since_edit == 0 and has_edited and budget_remaining > 4 and self._allow_run_tests(budget_remaining):
                return {"tool": "run_tests"}
            # — tests explicitly failed → must fix, not submit
            # (last_test_passed is None for not_run, so only fires on real failures)
            if last_test_passed is False and budget_remaining > 4:
                last_file, last_start = _last_inspected_file(history)
                if last_file:
                    return {
                        "tool":  "inspect",
                        "file":  last_file,
                        "start": last_start,
                        "end":   last_start + 60,
                    }
                return {"tool": "search", "query": "__recheck_after_fail__", "path": "."}

            # Gate 2: use git diff as ground truth when available
            saved_diff = _extract_current_diff(history)
            if saved_diff:
                return {"tool": "submit_patch", "patch": saved_diff}

            # Gate 3: all edits failed — try applying agent's patch via edit
            agent_patch = (action.get("patch") or "").strip()
            if agent_patch and edit_redirects < 1 and budget_remaining > 1:
                return {"tool": "edit", "patch": agent_patch, "_redirect": True}

            # Gate 4: force-generate a patch as last resort
            if budget_remaining > 2 and empty_retries < 2:
                forced = _force_generate_patch(self.model, self._issue_ctx, history)
                if forced and _is_patch_complete(forced):
                    return {"tool": "submit_patch", "patch": forced}
                if not agent_patch:
                    return {
                        "tool": "search",
                        "query": f"__empty_retry__{empty_retries}__",
                        "path": ".",
                    }

            if agent_patch and _is_patch_complete(agent_patch):
                return {"tool": "submit_patch", "patch": agent_patch}
            return {"tool": "submit_patch", "patch": ""}

        # ── run_tests gates ───────────────────────────────────────────────
        if tool == "run_tests":
            # Subclass hook: block in MEDIUM regime
            if not self._allow_run_tests(budget_remaining):
                saved_diff = _extract_current_diff(history)
                if saved_diff:
                    return {"tool": "submit_patch", "patch": saved_diff}
                # Fall through — no diff yet, let the agent keep working

            # Block redundant test calls (tests already failed, no edit since)
            if runtests_since_edit >= 1 and last_test_passed is False:
                last_file, last_start = _last_inspected_file(history)
                if last_file:
                    return {
                        "tool": "inspect",
                        "file": last_file,
                        "start": last_start,
                        "end": last_start + 60,
                    }
                return {"tool": "search", "query": "__re_investigate__", "path": "."}

        return action


# ---------------------------------------------------------------------------
# LoopAgent — no budget-regime awareness (all defaults from base)
# ---------------------------------------------------------------------------

class LoopAgent(_BaseLoopAgent):
    """
    Standard agentic coding loop (no budget-regime awareness).
    Mirrors the behaviour of systems like SWE-agent and OpenHands.
    """

    def __init__(
        self,
        model:            str = _DEFAULT_MODEL,
        max_search_calls: int = 5,
    ) -> None:
        super().__init__(model)
        self.max_search_calls = max_search_calls
