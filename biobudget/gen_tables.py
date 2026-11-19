"""metrics.json -> LaTeX tables for Chapter 5 (numbers copied mechanically)."""
import json
import sys

m = json.load(open(sys.argv[1]))
B = m["budgets"]
cells = m["cells"]


def pct(x):
    return f"{100 * x:.1f}"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


rows = []
AG = [a for a in ("Loop", "BudgetAware", "BudgetAwareRead") if f"{a}@{B[0]}" in cells]
NAME = {"Loop": "Loop", "BudgetAware": "BudgetAware", "BudgetAwareRead": "BudgetAware+Read"}
order = [("SingleShot", B[0])] + [(a, b) for a in AG for b in B]
for ag, b in order:
    c = cells[f"{ag}@{b}"]
    btxt = "--" if ag == "SingleShot" else str(b)
    cpc = c["cost_per_correct"]
    rows.append(
        f"{NAME.get(ag, ag)} & {btxt} & {pct(c['success'])} {ci(c['success_ci'])} & {pct(c['coverage'])} & "
        f"{pct(c['precision'])} & {pct(c['abstain'])} & {pct(c['violation'])} & "
        f"{c['auc']:.3f} & {c['mean_units']:.2f} & {c['mean_calls']:.2f} & "
        f"{c['mean_tokens'] / 1000:.1f} & {100 * c['cost_per_ep']:.2f} & "
        f"{(100 * cpc) if cpc else float('nan'):.2f} \\\\")
    if ag != AG[-1] and (ag == "SingleShot" or b == B[-1]):
        rows.append("\\addlinespace")
main = r"""\begin{table}[t]
\centering
\caption[BioBudget pilot results]{BioBudget results on LAB-Bench SeqQA (Gemini 2.5 Flash, 360 questions per row). Success, coverage, precision, abstention and violation in \%; Success@B with 95\% bootstrap CI. Units = mean budget units spent; calls = mean LLM calls; tokens = mean input+output+thinking tokens (thousands); cost in US cents at list price, per episode and per correct answer. SingleShot uses no tools and does not depend on $B$.}
\label{tab:bioresults}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccccccccc}
\toprule
Agent & $B$ & Success@B & Cov. & Prec. & Abst. & Viol. & AUC & Units & Calls & kTok & \textcent/ep & \textcent/correct \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

subs = list(m["per_subtask"].keys())
ab = {"Loop": "L", "BudgetAware": "BA", "BudgetAwareRead": "R"}
cols = ["SingleShot@" + str(B[0])] + [f"{a}@{b}" for a in AG for b in B]
head = "Subtask & SS & " + " & ".join(f"{ab[a]}{b}" for a in AG for b in B) + " \\\\"
srows = []
for s in subs:
    r = m["per_subtask"][s]
    srows.append(s + " & " + " & ".join(f"{100 * r[c]:.0f}" for c in cols) + " \\\\")
per = r"""\begin{table}[t]
\centering
\caption[BioBudget success by subtask]{Success@B by subtask (\%, 40 questions per cell). SS = SingleShot; L$B$, BA$B$, R$B$ = LoopAgent, BudgetAwareAgent, and BudgetAware+Read at budget $B$.}
\label{tab:biosub}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l""" + "c" * len(cols) + r"""}
\toprule
""" + head + "\n\\midrule\n" + "\n".join(srows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(sys.argv[2], "w").write(main + "\n" + per)
print(main)
print(per)
