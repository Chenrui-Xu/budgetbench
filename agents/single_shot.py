"""
BudgetBench — SingleShotAgent  (Track B · Task B1)
====================================================
Baseline 1: given the issue description + initial repo context,
produce a patch in ONE LLM call with no iteration.

Budget usage: exactly 1 tool call (submit_patch, cost = 0 by default).
This is the cheapest possible agent and serves as a lower-bound baseline.

To be implemented by: Sylvia Li (sylvia/agents branch)
Stub provided by: Chenrui Xu
"""

from __future__ import annotations

from budgetbench.interfaces import Agent, ToolCall


class SingleShotAgent(Agent):
    """
    Single-shot patch generation agent.

    The agent reads the initial observation (which contains the issue
    description and any repo context provided by reset()), makes exactly
    one LLM call to generate a unified-diff patch, and submits it.

    Parameters
    ----------
    model : str
        LLM model identifier (e.g. "claude-haiku-4-5-20251001").
    system_prompt : str, optional
        Custom system prompt. Defaults to a concise patch-generation prompt.
    """

    def __init__(
        self,
        model:         str = "claude-haiku-4-5-20251001",
        system_prompt: str | None = None,
    ) -> None:
        self.model         = model
        self.system_prompt = system_prompt or (
            "You are an expert software engineer. "
            "Given a GitHub issue and repository context, produce a minimal "
            "unified-diff patch that resolves the issue. "
            "Output ONLY the diff, no explanation."
        )
        self._acted = False

    def act(
        self,
        obs:              dict,
        budget_remaining: int,
        history:          list[ToolCall],
    ) -> dict:
        """
        On the first call: generate a patch via LLM and submit it.
        On any subsequent call (should not happen): submit an empty patch.

        TODO (Sylvia):
          1. Build a prompt from obs["issue"] and obs["repo_context"].
          2. Call the LLM API (use the Anthropic SDK or openai).
          3. Parse the diff from the response.
          4. Return {"tool": "submit_patch", "patch": <diff>}.
        """
        if self._acted:
            # Safety: should not be called twice in one episode
            return {"tool": "submit_patch", "patch": ""}

        self._acted = True

        # TODO: replace this stub with a real LLM call
        issue       = obs.get("issue", "")
        repo_ctx    = obs.get("repo_context", "")
        stub_patch  = (
            "--- a/stub.py\n"
            "+++ b/stub.py\n"
            "@@ -1 +1 @@\n"
            "-# TODO\n"
            "+# Fixed\n"
        )
        return {"tool": "submit_patch", "patch": stub_patch}
