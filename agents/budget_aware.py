"""
BudgetBench — BudgetAwareAgent  (Track B · Task B3)  ★ Key Novelty
====================================================================
Subclasses _BaseLoopAgent and overrides three hooks to add budget-regime
strategy switching.  All loop logic (inspect→edit→test, force notes,
action interception) is inherited unchanged from the base class.

Budget regimes
--------------
HIGH   (remaining > 0.6 × total):  thorough — run_tests allowed
MEDIUM (0.3 < remaining ≤ 0.6):   focused  — run_tests blocked (cost = 3×)
LOW    (remaining ≤ 0.3):          emergency — skip loop, submit immediately

Difference from LoopAgent
--------------------------
- _regime_section()  : adds a "## Current regime" block to the prompt
- _pre_act()         : short-circuits to submit in LOW regime
- _allow_run_tests() : returns False in MEDIUM regime to save budget
"""

from __future__ import annotations

from budgetbench.interfaces import ToolCall
from agents.loop_agent import (
    _BaseLoopAgent,
    _DEFAULT_MODEL,
    _extract_current_diff,
    _force_generate_patch,
    _is_patch_complete,
)


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
        "- Do targeted search/inspect only\n"
        "- run_tests is BLOCKED (costs 3 budget units) — skip it\n"
        "- Move directly toward submitting a patch after editing"
    ),
    "LOW": (
        "Budget is critical. Submit immediately:\n"
        "- Do NOT call any exploration or test tools\n"
        "- Submit the best patch you can based on what you know"
    ),
}


class BudgetAwareAgent(_BaseLoopAgent):
    """
    Budget-adaptive agent — the key novelty of this project.

    Parameters
    ----------
    model : str
        Gemini model identifier.
    budget_total : int
        Total budget declared upfront. Inferred from the first act() call if 0.
    """

    def __init__(
        self,
        model:        str = _DEFAULT_MODEL,
        budget_total: int = 0,
    ) -> None:
        super().__init__(model)
        self._budget_total = budget_total

    # ── regime calculation ────────────────────────────────────────────────

    def _regime(self, budget_remaining: int) -> str:
        total = self._budget_total or 1
        frac  = budget_remaining / total
        if frac > _HIGH_THRESHOLD:
            return "HIGH"
        if frac > _MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    # ── hook overrides ────────────────────────────────────────────────────

    def _regime_section(self, budget_remaining: int) -> str:
        regime = self._regime(budget_remaining)
        return (
            f"## Current budget regime: {regime}\n"
            f"{_REGIME_INSTRUCTIONS[regime]}"
        )

    def _pre_act(
        self, budget_remaining: int, history: list[ToolCall]
    ) -> dict | None:
        # Infer total budget on the very first real call
        if self._budget_total == 0:
            self._budget_total = budget_remaining

        if self._regime(budget_remaining) == "LOW":
            # Emergency: submit whatever we have, no further LLM call
            saved_diff = _extract_current_diff(history)
            if saved_diff:
                return {"tool": "submit_patch", "patch": saved_diff}
            forced = _force_generate_patch(self.model, self._issue_ctx, history)
            if forced and _is_patch_complete(forced):
                return {"tool": "submit_patch", "patch": forced}
            return {"tool": "submit_patch", "patch": ""}

        return None  # proceed with normal loop

    def _allow_run_tests(self, budget_remaining: int) -> bool:
        # run_tests costs 3× — skip it in MEDIUM and LOW regimes
        return self._regime(budget_remaining) == "HIGH"
