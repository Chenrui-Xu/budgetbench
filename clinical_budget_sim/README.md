# ClinicalBudget-Sim

A single Python file with no dependencies beyond the standard
library: synthetic medication-ordering patients, priced clinical tools, the four agents
(SingleShot, Loop, BudgetAware, ClinicalBudgetAware with verification floor and handoff
fallback), and the metrics with bootstrap CIs. 1,000 seeded patients x 4 agents x
7 budgets = 28,000 episodes, deterministic (patient seeds 0-999, bootstrap seed 0).
    python3 clinical_budget_sim.py    # writes results.json (identical to the one here)
    python3 plot.py                   # writes fig_clinical_sim.png (needs matplotlib)
No LLM is used, so no API key is needed.

