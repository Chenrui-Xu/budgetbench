# Track B — Agent Implementation Guide
**Assigned to: Sylvia Li**
**Branch: `sylvia/agents`**

---

## What you need to do

Implement three agents that use a real LLM (Claude) to solve coding tasks.
Each agent file already has a stub with TODOs — you just need to fill them in.

The agents will eventually run on **SWEBench** (real GitHub issues), but you
can develop and test everything right now using **KnapsackEnvironment** —
no Docker, no external setup required.

---

## Step 1 — Set up your branch

```bash
git checkout main
git pull origin main
git checkout -b sylvia/agents
```

---

## Step 2 — Install dependencies

```bash
pip install anthropic          # Anthropic SDK for Claude
pip install budgetbench        # or: pip install -e . from repo root
```

Get an Anthropic API key from https://console.anthropic.com and set it:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Step 3 — Understand the interface

Every agent must implement one method:

```python
def act(self, obs: dict, budget_remaining: int, history: list[ToolCall]) -> dict:
    ...
    return {"tool": "submit_patch", "patch": "<unified diff string>"}
```

| Parameter | What it is |
|---|---|
| `obs` | Latest observation from the environment (dict) |
| `budget_remaining` | How many tool calls you have left |
| `history` | All previous tool calls this episode (`tc.name`, `tc.args`, `tc.result`) |

Your agent returns an action dict: `{"tool": "<name>", ...extra args...}`.

Available tools (for SWEBench):

| Tool | Cost | What it does |
|---|---|---|
| `search` | 1 | Search codebase for a keyword |
| `inspect` | 1 | Read a file |
| `edit` | 1 | Write a file change |
| `run_tests` | **3** | Run the test suite |
| `submit_patch` | **0** | Submit final unified diff (ends episode) |

---

## Step 4 — Implement the three agents

### B1 · `single_shot.py` — Start here (easiest)

One LLM call, then submit. Budget used: ~1.

```python
import anthropic

client = anthropic.Anthropic()

def act(self, obs, budget_remaining, history):
    if self._acted:
        return {"tool": "submit_patch", "patch": ""}
    self._acted = True

    prompt = f"""You are an expert software engineer.
Issue: {obs.get('issue', '')}

Repository context:
{obs.get('repo_context', '')}

Output ONLY a unified diff patch that fixes the issue. No explanation."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    patch = response.content[0].text.strip()
    return {"tool": "submit_patch", "patch": patch}
```

### B2 · `loop_agent.py` — Standard agentic loop

Keep calling tools until confident, then submit.

```python
def act(self, obs, budget_remaining, history):
    # Build context from history
    tool_results = "\n".join(
        f"[{tc.name}] {tc.result}" for tc in history
    )

    # If budget is low, force submit
    if budget_remaining <= 2:
        return {"tool": "submit_patch", "patch": self._best_patch}

    prompt = f"""Issue: {obs.get('issue')}
Previous tool calls:
{tool_results}
Current observation: {obs}

Choose the next action. Options: search, inspect, edit, run_tests, submit_patch.
If ready to submit, return the patch. Otherwise return the next tool call.
Respond in JSON: {{"tool": "...", "args": {{...}}}}"""

    response = client.messages.create(...)
    action = json.loads(response.content[0].text)
    return action
```

### B3 · `budget_aware.py` — Key novelty ★

Same as LoopAgent but **changes strategy based on `budget_remaining`**.

```python
def act(self, obs, budget_remaining, history):
    regime = self._regime(budget_remaining)  # HIGH / MEDIUM / LOW

    if regime == "HIGH":
        # Full exploration: search → inspect → edit → run_tests
        system = "Be thorough. Search widely, run tests before submitting."
    elif regime == "MEDIUM":
        # Focused: skip broad search, limit run_tests
        system = "Be focused. Avoid run_tests unless critical."
    else:  # LOW
        # Emergency: submit best patch now
        system = "Budget is almost gone. Submit the best patch you have immediately."

    # Pass regime as context to the LLM
    prompt = f"[Budget regime: {regime}] ..."
    ...
```

The `_regime()` method is already implemented in the stub — just use it.

---

## Step 5 — Test on KnapsackEnvironment first

You don't need SWEBench to test your agent. The Knapsack env works right now:

```python
# quick_test.py
from budgetbench import BudgetConfig, BudgetRunner
from budgetbench.interfaces import Episode
from environments.knapsack import KnapsackEnvironment
from agents.single_shot import SingleShotAgent

runner = BudgetRunner()
config = BudgetConfig(total=20, cost_weights={"run_tests": 3, "submit_patch": 0})
env    = KnapsackEnvironment(n_items=10, capacity=50, seed=0)
agent  = SingleShotAgent()

episode = runner.run_episode(agent, env, config)
print(episode.final_status, episode.budget_used)
```

For the Knapsack environment, `obs` looks like:
```python
# Initial obs (from reset):
{"task": "knapsack", "n_items": 10, "capacity": 50, "message": "..."}

# After list_items:
{"items": [{"id": 0, "name": "item_0", "weight": 5, "value": 12}, ...]}
```

The LLM can use `list_items`, `add_item`, `remove_item`, `check_feasibility`, `submit_patch`.
The patch format for knapsack is: `'{"selected": [0, 2, 5]}'`

---

## Step 6 — Run the full demo

```bash
python examples/demo_knapsack.py
```

Check `results/` for the 4 evaluation plots.

---

## Step 7 — Open a PR

```bash
git add agents/
git commit -m "feat(B1): implement SingleShotAgent with Claude Haiku"
git push origin sylvia/agents
# then open PR on GitHub targeting main
```

---

## Key rules

- **Never break the `act()` signature** — the runner depends on it
- **Always handle `budget_remaining <= 2`** — force a submit to avoid BUDGET_EXCEEDED
- **`submit_patch` costs 0** — call it without worrying about budget
- **`run_tests` costs 3** — use sparingly, especially in MEDIUM/LOW regime
- Ask Chenrui if the interfaces are unclear before changing them

---

## Questions?

Ping Chenrui on WeChat or open a GitHub issue tagged `[track-b]`.
