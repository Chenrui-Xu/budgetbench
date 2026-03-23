"""
BudgetBench — LoopAgent  (Track B · Task B2)
=============================================
Baseline 2: standard agentic coding loop — search → inspect → edit → test.

The agent keeps calling tools until it decides to submit or runs out of budget.
This mirrors the behaviour of systems like SWE-agent and OpenHands.

To be implemented by: Sylvia Li (sylvia/agents branch)
Stub provided by: Chenrui Xu
"""

from __future__ import annotations

from budgetbench.interfaces import Agent, ToolCall


class LoopAgent(Agent):
    """
    Standard tool-using loop agent.

    Strategy
    --------
    1. search  — locate relevant code using the issue description
    2. inspect — read the identified file(s)
    3. edit    — draft a fix
    4. run_tests — validate (costs 3× budget)
    5. submit_patch — finalise

    Parameters
    ----------
    model : str
        LLM model identifier.
    max_search_calls : int
        Upper bound on search calls before forcing a submit.
    """

    def __init__(
        self,
        model:            str = "claude-haiku-4-5-20251001",
        max_search_calls: int = 5,
    ) -> None:
        self.model            = model
        self.max_search_calls = max_search_calls

    def act(
        self,
        obs:              dict,
        budget_remaining: int,
        history:          list[ToolCall],
    ) -> dict:
        """
        TODO (Sylvia):
          - Use history to track what has been searched/inspected.
          - Use an LLM to decide the next tool call given obs + history.
          - When confident, submit the patch.
          - If budget_remaining is very low, skip to submit_patch.

        For now this stub always submits immediately (same as SingleShotAgent).
        """
        # Stub: immediately submit
        return {
            "tool":  "submit_patch",
            "patch": (
                "--- a/stub.py\n"
                "+++ b/stub.py\n"
                "@@ -1 +1 @@\n"
                "-# TODO\n"
                "+# Fixed by LoopAgent\n"
            ),
        }
