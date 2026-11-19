# BioBudget

Budget-constrained tool-using agents on LAB-Bench DNA sequence questions (SeqQA,
360 questions) and CloningScenarios (33 questions).

- biobudget/tasks.py    LAB-Bench SeqQA / CloningScenarios loader, option construction
- biobudget/tools.py    seven priced Biopython tools
- biobudget/agents.py   runner semantics and the four agents
                        (SingleShot, Loop, BudgetAware, BudgetAware+Read)
- run_pilot.py          resumable runner with cost cap and kill switch
- validate_tools.py     tool-only oracle check (358/360 SeqQA questions)
- rescore.py            reads SingleShot answers given as \boxed{X} instead of tags
- analyze.py, gen_tables.py   metrics, bootstrap CIs, figures, LaTeX tables
- biobudget/protocol.py ProtocolQA extension; the run was stopped early and is NOT
                        reported in the thesis

results/raw/       episode logs exactly as produced (one JSON line per episode)
results/analysis/  scored episodes, metrics, tables and figures used in the report

Reproduce the report numbers (no API calls needed):
    pip install -r requirements.txt
    python analyze.py results/analysis/seqqa_episodes_scored.jsonl out_seqqa
    python gen_tables.py out_seqqa/metrics.json out_seqqa/tables.tex
    python analyze.py results/analysis/cloning_episodes_scored.jsonl out_cloning

Re-run experiments: download the LAB-Bench public files
(https://github.com/Future-House/LAB-Bench) into data/SeqQA and data/CloningScenarios,
set GEMINI_API_KEY, then `python run_pilot.py --yes --per-subtask 40 --agents SingleShot
Loop BudgetAware BudgetAwareRead` (and `--cloning`). Model: Gemini 2.5 Flash,
temperature 0, thinking budget 1024. LAB-Bench data are not redistributed here.

