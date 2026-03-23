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
        - targeted search, minimal inspections, limit run_tests calls
LOW    (remaining ≤ 0.3):          emergency mode
        - skip further search/test, summarise state, submit best patch

This is the primary technical contribution of BudgetBench.

To be implemented by: Sylvia Li (sylvia/agents branch)
Stub provided by: Chenrui Xu
"""

from __future__ import annotations

from budgetbench.interfaces import Agent, ToolCall


_HIGH_THRESHOLD   = 0.6
_MEDIUM_THRESHOLD = 0.3


class BudgetAwareAgent(Agent):
    """
    Budget-adaptive agent — the key novelty of this project.

    Parameters
    ----------
    model : str
        LLM model identifier.
    budget_total : int
        Total budget declared upfront (needed to compute fractions).
        If 0, the agent infers it from the first act() call.
    """

    def __init__(
        self,
        model:        str = "claude-sonnet-4-6",
        budget_total: int = 0,
    ) -> None:
        self.model        = model
        self._budget_total = budget_total

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
        """
        TODO (Sylvia):
          HIGH regime:
            - Call LLM with full context: issue + repo + history
            - Prefer: search → inspect → edit → run_tests
          MEDIUM regime:
            - Narrow search scope, avoid run_tests unless critical
            - Summarise history to compress context
          LOW regime:
            - Skip exploration entirely
            - Submit the best patch constructed so far (or an LLM-generated one
              based on compressed state summary)

        The ``budget_remaining`` arg is injected by BudgetRunner on every call.
        """
        if self._budget_total == 0:
            # Infer total on first call (budget_remaining equals total at step 0)
            self._budget_total = budget_remaining

        regime = self._regime(budget_remaining)

        # Stub: submit regardless of regime
        return {
            "tool":   "submit_patch",
            "patch":  (
                "--- a/stub.py\n"
                "+++ b/stub.py\n"
                "@@ -1 +1 @@\n"
                "-# TODO\n"
                f"+# Fixed by BudgetAwareAgent [{regime}]\n"
            ),
            "_regime": regime,  # metadata for analysis
        }
