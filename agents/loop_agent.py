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

9. Call **submit_patch** when:
   - you believe the patch fixes the issue, OR
   - the remaining budget is very small.

Rules:

- Always return a valid JSON object.
- Never output explanations or text outside the JSON object.
- The JSON must contain the key "tool".
- Prefer making progress toward a fix instead of endless exploration.
"""

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
            return action
        except Exception:
            # Fallback: submit with whatever we have
            return {"tool": "submit_patch", "patch": ""}
