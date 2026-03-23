"""
BudgetBench — BudgetRunner
===========================
Deterministic execution harness that:
  1. Enforces per-episode tool-call budgets (with cost weighting).
  2. Logs every tool call with a millisecond timestamp.
  3. Assigns a reproducible StatusCode at episode end.

Implemented by: Chenrui Xu (Project Lead)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .interfaces import Agent, BudgetConfig, Episode, Environment, StatusCode, ToolCall

logger = logging.getLogger(__name__)


class BudgetRunner:
    """
    Runs a single agent–environment episode under a fixed budget.

    Usage
    -----
    ::

        runner = BudgetRunner()
        episode = runner.run_episode(agent, env, config)
        print(episode.final_status, episode.budget_used)

    Notes
    -----
    - The runner is stateless; it is safe to reuse across episodes.
    - If the agent's chosen action would push ``budget_used`` past
      ``config.total``, the episode terminates immediately with
      ``BUDGET_EXCEEDED`` *before* the action is executed.
    - ``env.score()`` is called only when the episode exits via
      ``done=True`` (i.e., the agent submitted a patch).
    """

    def run_episode(
        self,
        agent:    Agent,
        env:      Environment,
        config:   BudgetConfig,
        metadata: Optional[dict] = None,
    ) -> Episode:
        """
        Execute one episode and return the complete record.

        Parameters
        ----------
        agent    : Agent           — the agent under evaluation
        env      : Environment     — the task environment
        config   : BudgetConfig    — budget total and cost weights
        metadata : dict, optional  — arbitrary tags stored in Episode.metadata
        """
        episode = Episode(
            budget_total=config.total,
            budget_used=0,
            metadata=metadata or {},
        )

        logger.info(
            "Episode start | budget=%d | weights=%s",
            config.total,
            config.cost_weights,
        )

        # ── initialise environment ──────────────────────────────────────────
        obs = env.reset()
        done = False
        submitted_patch: str = ""

        # ── main loop ───────────────────────────────────────────────────────
        while not done:

            # Check budget *before* asking the agent to act
            if episode.budget_used >= config.total:
                logger.info(
                    "Budget exhausted before agent acted | used=%d/%d",
                    episode.budget_used, config.total,
                )
                episode.final_status = "BUDGET_EXCEEDED"
                return episode

            budget_remaining = config.total - episode.budget_used

            # ── agent decides ──────────────────────────────────────────────
            action = agent.act(
                obs=obs,
                budget_remaining=budget_remaining,
                history=list(episode.tool_calls),  # pass a copy
            )

            tool_name = action.get("tool", "unknown")
            cost      = config.cost_of(tool_name)

            # Check whether this action *would* exceed the budget
            if episode.budget_used + cost > config.total:
                logger.info(
                    "Action '%s' (cost=%d) would exceed budget | used=%d/%d",
                    tool_name, cost, episode.budget_used, config.total,
                )
                episode.final_status = "BUDGET_EXCEEDED"
                return episode

            # ── execute action ─────────────────────────────────────────────
            ts_ms = int(time.time() * 1000)
            obs, done = env.step(action)

            # Charge the budget
            episode.budget_used += cost

            # Record the call
            tool_call = ToolCall(
                name=tool_name,
                args=action,
                result=obs,
                cost=cost,
                ts_ms=ts_ms,
            )
            episode.tool_calls.append(tool_call)

            logger.debug(
                "Tool call | name=%s | cost=%d | budget_used=%d/%d | done=%s",
                tool_name, cost, episode.budget_used, config.total, done,
            )

            # Track the latest patch (for scoring after the loop)
            if "patch" in action:
                submitted_patch = action["patch"]

        # ── episode ended (agent submitted) ─────────────────────────────────
        episode.final_status = env.score(submitted_patch)

        logger.info(
            "Episode end | status=%s | used=%d/%d | calls=%d",
            episode.final_status,
            episode.budget_used,
            config.total,
            len(episode.tool_calls),
        )
        return episode
