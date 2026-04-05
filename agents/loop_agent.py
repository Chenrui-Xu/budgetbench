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

from google import genai

from budgetbench.interfaces import Agent, ToolCall

_DEFAULT_MODEL = "gemini-2.5-flash"

_SYSTEM_PROMPT = """\
You are an expert software engineer working inside a code repository.
Your goal is to fix the given GitHub issue by exploring the codebase and
producing a unified-diff patch.

At each step you must output a JSON object (and ONLY that JSON object) choosing
the next tool call:

  {"tool": "search",   "query": "<keyword>", "path": "<optional subdir>"}
  {"tool": "inspect",  "file": "<path>", "start": <int>, "end": <int>}
  {"tool": "hierarchy", "depth": <int>}
  {"tool": "edit",     "patch": "<unified diff hunk>"}
  {"tool": "run_tests"}
  {"tool": "submit_patch", "patch": "<full unified diff>"}

Rules:
- Call submit_patch when you are confident the fix is correct.
- If budget_remaining <= 3, call submit_patch immediately with your best patch.
- Never output anything outside the JSON object."""


def _gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "ai-agent-for-engineers"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


def _summarise_history(history: list[ToolCall]) -> str:
    lines = []
    for tc in history:
        result_str = str(tc.result)[:300]
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

        # Force submit when budget is critical
        if budget_remaining <= 3:
            return {"tool": "submit_patch", "patch": ""}

        history_str = _summarise_history(history)
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"## Issue\n{self._issue_ctx}\n\n"
            f"## Tool call history\n{history_str or '(none yet)'}\n\n"
            f"## Latest observation\n{json.dumps(obs, indent=2)[:1000]}\n\n"
            f"Budget remaining: {budget_remaining}\n\n"
            f"## Next action (JSON only)"
        )

        client = _gemini_client()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        raw = (response.text or "").strip()
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
            return action
        except Exception:
            # Fallback: submit with whatever we have
            return {"tool": "submit_patch", "patch": ""}
