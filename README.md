# BudgetBench

**Budget-Constrained Evaluation Harness for Tool-Using Code Agents**

> CMU 24-880 AI Agents for Engineers · Spring 2026
> Team: Chenrui Xu · Hanyu Jin · Sylvia Li

---

## Overview

BudgetBench studies how well tool-using code agents perform under strict **tool-call budgets**.
Agents must fix GitHub issues (SWE-bench Lite) while spending as few tool calls as possible.

The key insight: a cheap agent that solves a problem in 1 call dominates an expensive agent
that solves the same problem in 15 calls, because it succeeds at every budget level ≥ 1.

| Metric | Description |
|--------|-------------|
| `Success@B` | Fraction of episodes solved within budget B |
| `Invalid@B` | Format / patch-apply failure rate |
| `Violation@B` | Budget over-spend rate |
| `AUC (Success-vs-B)` | Area under the success curve — the primary ranking metric |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Eric-XC/budgetbench.git
cd budgetbench
pip install -e ".[dev]"

# 2. Run all tests
pytest

# 3. Knapsack demo (no API key needed — instant results)
python examples/demo_knapsack.py

# 4. Batch experiment (multi-issue × multi-budget, saves CSV + JSON + plots)
python scripts/batch_experiment.py --env knapsack --budget 5 10 20 --num-issues 10

# 5. SWE-bench demo (requires GEMINI_API_KEY and Docker)
export GEMINI_API_KEY=<your_key>
python examples/demo_swe_bench.py --issue astropy__astropy-12907
```

---

## Reproducing the Paper Results

### Knapsack benchmark (fully self-contained)

```bash
python scripts/batch_experiment.py \
    --env knapsack \
    --budget 1 5 10 20 \
    --num-issues 50 \
    --output-dir results/knapsack_paper
```

Output: `results/knapsack_paper/episodes.csv`, `report.json`, `*.png`

### SWE-bench Lite (requires setup)

```bash
# Step 1: install dependencies
pip install swebench datasets

# Step 2: build Docker images for target issues
python -m swebench.harness.docker_build \
    --instance_ids astropy__astropy-12907 \
    --dataset_name princeton-nlp/SWE-bench_Lite

# Step 3: run batch experiment
export GEMINI_API_KEY=<your_key>
python scripts/batch_experiment.py \
    --env swe \
    --budget 10 20 \
    --num-issues 5 \
    --output-dir results/swe_paper
```

---

## Agents (Track B)

Three agents are implemented in `agents/`, all using Gemini 2.5-flash.

| Agent | Strategy | Typical budget used |
|-------|----------|---------------------|
| `SingleShotAgent` | One LLM call → submit patch | 0 |
| `LoopAgent` | search → inspect → edit → test loop | 6–18 |
| `BudgetAwareAgent` | Regime-adaptive (HIGH / MEDIUM / LOW) | 4–10 |

**Key novelty**: `BudgetAwareAgent` switches strategy in real time as budget shrinks:
- **HIGH** (>60% remaining): thorough exploration with `run_tests`
- **MEDIUM** (30–60%): targeted search only, no `run_tests`
- **LOW** (≤30%): skips LLM, submits best patch immediately

```python
from agents.budget_aware import BudgetAwareAgent
from agents.loop_agent import LoopAgent
from agents.single_shot import SingleShotAgent
```

---

## Project Structure

```
budgetbench/
├── budgetbench/
│   ├── interfaces.py    # Core data structures + abstract classes
│   ├── runner.py        # BudgetRunner — deterministic episode harness
│   ├── metrics.py       # success_at_b, AUC, violation_rate, bootstrap CI
│   └── visualize.py     # Matplotlib plots
├── environments/
│   ├── swe_bench.py     # SWEBenchEnvironment (Docker-based)
│   └── knapsack.py      # KnapsackEnvironment (toy, DP oracle)
├── agents/
│   ├── single_shot.py   # SingleShotAgent
│   ├── loop_agent.py    # LoopAgent
│   └── budget_aware.py  # BudgetAwareAgent (key novelty)
├── scripts/
│   └── batch_experiment.py  # Multi-issue × multi-budget runner
├── examples/
│   ├── demo_knapsack.py     # Heuristic agents on KnapsackEnvironment
│   └── demo_swe_bench.py    # LLM agents on SWE-bench
└── tests/
    ├── test_runner.py        # BudgetRunner unit tests
    ├── test_metrics.py       # Metrics unit tests
    └── test_integration.py   # Full pipeline integration tests (46 tests)
```

---

## Interface Contract

All components implement against `budgetbench/interfaces.py`.

```python
from budgetbench import BudgetConfig, BudgetRunner
from budgetbench.interfaces import Agent, Environment, Episode

# Run one episode
episode: Episode = BudgetRunner().run_episode(
    agent=my_agent,
    env=my_env,
    config=BudgetConfig(total=20, cost_weights={"run_tests": 3, "submit_patch": 0}),
)

print(episode.final_status)  # "OK_SOLVED" | "TEST_FAIL" | "BUDGET_EXCEEDED" | ...
print(episode.budget_used)
print(episode.solved)        # True if final_status == "OK_SOLVED"
```

**Rule**: never break an interface without opening a `[interface-change]` GitHub issue.

---

## Running Tests

```bash
pytest                          # all tests
pytest tests/test_integration.py  # integration tests only
pytest -v --tb=short            # verbose output
```

Expected: **46 tests passing** (9 runner + 12 metrics + 25 integration).

---

## API Keys

The LLM agents (`SingleShotAgent`, `LoopAgent`, `BudgetAwareAgent`) require a Gemini API key:

```bash
export GEMINI_API_KEY=<your_key>
```

Get a key at [Google AI Studio](https://aistudio.google.com/). Free tier has rate limits;
link a billing account for higher throughput.

Alternatively, use Vertex AI by setting:
```bash
export GOOGLE_CLOUD_PROJECT=<your_project>
export GOOGLE_CLOUD_LOCATION=us-central1
# (omit GEMINI_API_KEY)
```

---

## Team

| Person | Role | Branch |
|--------|------|--------|
| Chenrui Xu | Project Lead · Harness · Metrics | `main` |
| Hanyu Jin | Track A — Environments | `hanyu/environments` |
| Sylvia Li | Track B — Agents | `sylvia/agents` |

All merges to `main` require a PR + approval from Chenrui.

---

*Last updated: 2026-04-05*
