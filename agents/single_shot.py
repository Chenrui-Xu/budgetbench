"""
BudgetBench — SingleShotAgent  (Track B · Task B1)
====================================================
Baseline 1: given the issue description + initial repo context,
produce a patch in ONE LLM call with no iteration.

Budget usage: exactly 1 tool call (submit_patch, cost = 0).
This is the cheapest possible agent and serves as a lower-bound baseline.
"""

from __future__ import annotations

import os

from budgetbench.interfaces import Agent, ToolCall
from agents._gemini import generate_with_retry

_DEFAULT_MODEL = "gemini-2.5-flash"

_SYSTEM_PROMPT = """\
You are an expert software engineer.
Given a GitHub issue and repository context, produce a minimal unified-diff
patch that resolves the issue.
Output ONLY the unified diff. No explanation, no markdown fences."""


class SingleShotAgent(Agent):
    """
    Single-shot patch generation agent.

    Makes exactly one LLM call with the full issue + repo context,
    then submits the generated unified-diff patch.

    Parameters
    ----------
    model : str
        Gemini model identifier (default: gemini-2.0-flash).
    system_prompt : str, optional
        Override the default system prompt.
    """

    def __init__(
        self,
        model:         str = _DEFAULT_MODEL,
        system_prompt: str | None = None,
    ) -> None:
        self.model         = model
        self.system_prompt = system_prompt or _SYSTEM_PROMPT
        self._acted        = False

    def act(
        self,
        obs:              dict,
        budget_remaining: int,
        history:          list[ToolCall],
    ) -> dict:
        if self._acted:
            return {"tool": "submit_patch", "patch": ""}

        self._acted = True

        issue    = obs.get("problem_statement") or obs.get("issue", "")
        repo_ctx = obs.get("repo_context", "")
        hints    = obs.get("hints_text", "")

        prompt = (
            f"{self.system_prompt}\n\n"
            f"## Issue\n{issue}\n\n"
            + (f"## Hints\n{hints}\n\n" if hints else "")
            + (f"## Repository context\n{repo_ctx}\n\n" if repo_ctx else "")
            + "## Patch"
        )

        patch = generate_with_retry(self.model, prompt).strip()

        return {"tool": "submit_patch", "patch": patch}
