"""Shared agent helpers — deduplicates code between LoopAgent and BudgetAwareAgent."""
from __future__ import annotations

from budgetbench.interfaces import ToolCall
from agents._gemini import generate_with_retry


def has_tested_after_last_edit(history: list[ToolCall]) -> bool:
    """Return True if run_tests was called after the most recent edit."""
    last_edit_idx = -1
    for i, tc in enumerate(history):
        if tc.name == "edit":
            last_edit_idx = i
    if last_edit_idx == -1:
        return False
    return any(tc.name == "run_tests" for tc in history[last_edit_idx + 1:])


def extract_current_diff(history: list[ToolCall]) -> str:
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


def force_generate_patch(model: str, issue: str, history: list[ToolCall]) -> str:
    """Make a focused LLM call to generate a unified diff when normal path fails."""
    code_parts = []
    for tc in history:
        if tc.name in ("inspect", "search", "edit"):
            result_str = str(tc.result)[:3000]
            code_parts.append(f"[{tc.name}]\n{result_str}")
    code_seen = "\n\n".join(code_parts[-6:])

    prompt = _FORCE_PATCH_PROMPT.format(issue=issue, code_seen=code_seen)
    raw = generate_with_retry(model, prompt).strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith(("diff", "patch", "python", "py", "\n")):
            raw = raw[raw.index("\n") + 1:] if "\n" in raw else raw
        raw = raw.strip()

    if raw.startswith("---") or raw.startswith("diff --git"):
        return raw
    return ""
