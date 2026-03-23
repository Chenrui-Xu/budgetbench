# BudgetBench

**Budget-Constrained Evaluation Harness for Tool-Using Code Agents**

> CMU 24-880 AI Agents for Engineers · Spring 2026
> Team: Chenrui Xu · Hanyu Jin · Sylvia Li

---

## Overview

BudgetBench studies how well tool-using code agents perform under strict **tool-call budgets**.
We evaluate agents on SWE-bench Lite issues and measure:

| Metric | Description |
|--------|-------------|
| `Success@B` | Fraction of episodes solved within budget B |
| `Invalid@B` | Format / patch-apply failure rate |
| `Violation@B` | Budget over-spend rate |
| `Time-to-First-Valid` | Tool calls before first patch attempt |
| `AUC (Success-vs-B)` | Area under success curve across all budgets |

---

## Project Structure

```
budgetbench/
├── budgetbench/
│   ├── interfaces.py   # Core data structures + abstract classes (Chenrui)
│   ├── runner.py       # BudgetRunner — deterministic harness (Chenrui)
│   └── metrics.py      # Evaluation pipeline (Chenrui)
├── environments/       # Track A — Hanyu Jin
│   ├── swe_bench.py    # SWEBenchEnvironment
│   └── knapsack.py     # KnapsackEnvironment
├── agents/             # Track B — Sylvia Li
│   ├── single_shot.py  # SingleShotAgent
│   ├── loop_agent.py   # LoopAgent
│   └── budget_aware.py # BudgetAwareAgent (key novelty)
└── tests/
```

---

## Shared Interfaces (Week 1 deliverable)

All modules implement against `budgetbench/interfaces.py`.
**Rule: never break an interface without opening a `[interface-change]` GitHub issue.**

```python
from budgetbench import Agent, Environment, BudgetConfig, BudgetRunner

# Hanyu implements Environment; Sylvia implements Agent
episode = BudgetRunner().run_episode(agent, env, BudgetConfig(total=20))
print(episode.final_status)  # "OK_SOLVED" | "TEST_FAIL" | "BUDGET_EXCEEDED" | …
```

---

## Quick Start

```bash
git clone https://github.com/Eric-XC/budgetbench.git
cd budgetbench
pip install -e ".[dev]"
pytest
```

---

## Branch Policy

| Branch | Owner |
|--------|-------|
| `chenrui/runner` | Chenrui Xu |
| `hanyu/environments` | Hanyu Jin |
| `sylvia/agents` | Sylvia Li |

All merges to `main` require a PR + approval from Chenrui.
Never push directly to `main`.

---

## Timeline

| Phase | Dates | Milestone |
|-------|-------|-----------|
| 1 — Foundation | Mar 15 – Mar 29 | interfaces.py, BudgetRunner skeleton, repo setup |
| 2 — Core Impl. | Mar 30 – Apr 19 | Full runner, both envs, two agents |
| 3 — Experiments | Apr 20 – May 4  | Integration, BudgetAwareAgent, RAG |
| 4 — Evaluation | May 5 – May 25  | Metrics, plots, report |

Repo freeze: **May 22** · Submission: **May 25**

---

*Generated 2026-03-22*
