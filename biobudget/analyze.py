"""Analyse BioBudget pilot episodes -> metrics JSON, LaTeX tables, figure.

Usage: python3 -I analyze.py episodes.jsonl outdir
"""
import collections
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AGENTS = ["SingleShot", "Loop", "BudgetAware", "BudgetAwareRead"]
LABEL = {"SingleShot": "SingleShot", "Loop": "Loop", "BudgetAware": "BudgetAware", "BudgetAwareRead": "BudgetAware+Read"}
COLOR = {"SingleShot": "#2a78d6", "Loop": "#eb6834", "BudgetAware": "#1baf7a", "BudgetAwareRead": "#eda100"}
MARK = {"SingleShot": "o", "Loop": "s", "BudgetAware": "^", "BudgetAwareRead": "D"}
PRICE_IN, PRICE_OUT = 0.30e-6, 2.50e-6


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    # keep the last record per (agent, qid, B) in case of reruns
    last = {}
    for r in rows:
        last[(r["agent"], r["qid"], r["B"])] = r
    return list(last.values()), rows


def metrics(eps, B):
    n = len(eps)
    st = collections.Counter(e["status"] for e in eps)
    ok, wrong = st["OK_SOLVED"], st["WRONG_ANSWER"]
    answered = [e for e in eps if e["status"] in ("OK_SOLVED", "WRONG_ANSWER", "ABSTAINED")]
    solved = [e for e in eps if e["status"] == "OK_SOLVED"]
    tok = [e["tokens_in"] + e["tokens_out"] + e["tokens_think"] for e in eps]
    cost = [e["cost_usd"] for e in eps]
    auc = (sum((B - (e["answered_at"] or 0)) for e in solved) / (n * B)) if (B and n) else (ok / n if n else 0)
    return {
        "n": n, "success": ok / n, "coverage": (ok + wrong) / n,
        "precision": ok / (ok + wrong) if ok + wrong else 0.0,
        "abstain": st["ABSTAINED"] / n, "violation": st["BUDGET_EXCEEDED"] / n,
        "invalid": st["INVALID_FORMAT"] / n, "auc": auc,
        "ttfv": (sum(e["answered_at"] or 0 for e in answered) / len(answered)) if answered else None,
        "mean_units": sum(e.get("spent", 0) or 0 for e in eps) / n,
        "mean_calls": sum(e["n_llm_calls"] for e in eps) / n,
        "mean_tokens": sum(tok) / n, "mean_in": sum(e["tokens_in"] for e in eps) / n,
        "mean_out_think": sum(e["tokens_out"] + e["tokens_think"] for e in eps) / n,
        "cost_total": sum(cost), "cost_per_ep": sum(cost) / n,
        "cost_per_correct": (sum(cost) / ok) if ok else None,
        "status": dict(st),
    }


def boot_ci(eps, B, key, R=1000, seed=0):
    rng = random.Random(seed)
    qids = sorted({e["qid"] for e in eps})
    by = collections.defaultdict(list)
    for e in eps:
        by[e["qid"]].append(e)
    vals = []
    for _ in range(R):
        sample = [x for q in (rng.choice(qids) for _ in qids) for x in by[q]]
        vals.append(metrics(sample, B)[key])
    vals.sort()
    return [vals[int(0.025 * R)], vals[int(0.975 * R) - 1]]


def paired_diff(a_eps, b_eps, B, key="success", R=1000, seed=0):
    """Bootstrap CI of metric(a) - metric(b) resampling questions jointly."""
    rng = random.Random(seed)
    A = {e["qid"]: e for e in a_eps}
    Bm = {e["qid"]: e for e in b_eps}
    qids = sorted(set(A) & set(Bm))
    pt = metrics([A[q] for q in qids], B)[key] - metrics([Bm[q] for q in qids], B)[key]
    vals = []
    for _ in range(R):
        s = [rng.choice(qids) for _ in qids]
        vals.append(metrics([A[q] for q in s], B)[key] - metrics([Bm[q] for q in s], B)[key])
    vals.sort()
    return pt, [vals[int(0.025 * R)], vals[int(0.975 * R) - 1]]


def main(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    eps, raw = load(path)
    api_err = [e for e in eps if e["status"] == "API_ERROR"]
    eps = [e for e in eps if e["status"] != "API_ERROR"]
    budgets = sorted({e["B"] for e in eps if e["agent"] != "SingleShot"})
    out = {"n_records": len(raw), "n_episodes": len(eps), "n_api_error": len(api_err),
           "budgets": budgets, "cost_total_usd": sum(e["cost_usd"] for e in raw), "cells": {}}
    ss = [e for e in eps if e["agent"] == "SingleShot"]
    for ag in AGENTS:
        for B in budgets:
            cell = ss if ag == "SingleShot" else [e for e in eps if e["agent"] == ag and e["B"] == B]
            if not cell:
                continue
            m = metrics(cell, B)
            m["success_ci"] = boot_ci(cell, B, "success")
            m["precision_ci"] = boot_ci(cell, B, "precision")
            out["cells"][f"{ag}@{B}"] = m
    # paired BudgetAware - Loop per budget
    out["paired"] = {}
    for B in budgets:
        la = [e for e in eps if e["agent"] == "Loop" and e["B"] == B]
        ba = [e for e in eps if e["agent"] == "BudgetAware" and e["B"] == B]
        if la and ba:
            for key in ("success", "auc", "violation"):
                pt, ci = paired_diff(ba, la, B, key)
                out["paired"][f"BA-Loop@{B}:{key}"] = {"diff": pt, "ci": ci}
        br = [e for e in eps if e["agent"] == "BudgetAwareRead" and e["B"] == B]
        if br and ba:
            for key in ("success", "auc"):
                pt, ci = paired_diff(br, ba, B, key)
                out["paired"][f"BAR-BA@{B}:{key}"] = {"diff": pt, "ci": ci}
        if br and la:
            for key in ("success", "auc", "violation"):
                pt, ci = paired_diff(br, la, B, key)
                out["paired"][f"BAR-Loop@{B}:{key}"] = {"diff": pt, "ci": ci}
    # per-subtask success
    subs = sorted({e["subtask"] for e in eps})
    out["per_subtask"] = {}
    for s in subs:
        row = {}
        for ag in AGENTS:
            for B in budgets:
                cell = [e for e in (ss if ag == "SingleShot" else eps)
                        if e["subtask"] == s and (ag == "SingleShot" or (e["agent"] == ag and e["B"] == B))]
                if cell:
                    row[f"{ag}@{B}"] = sum(e["status"] == "OK_SOLVED" for e in cell) / len(cell)
        out["per_subtask"][s] = row
    # tool usage
    use = collections.Counter()
    for e in eps:
        for st in e.get("steps", []):
            if st.get("tool") and st.get("tool") != "answer":
                use[(e["agent"], st["tool"])] += 1
    out["tool_use"] = {f"{a}:{t}": c for (a, t), c in sorted(use.items())}
    json.dump(out, open(os.path.join(outdir, "metrics.json"), "w"), indent=1)

    # figure: 2x2
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    ax = ax.ravel()
    panels = [("success", "Success@B"), ("violation", "Budget-violation rate"),
              ("precision", "Precision (correct / answered)"), ("cost_per_ep", "Mean API cost per episode (US cents)")]
    for a_, (k, t) in zip(ax, panels):
        for ag in AGENTS:
            ys = []
            for B in budgets:
                m = out["cells"].get(f"{ag}@{B}")
                v = m[k] if m else None
                ys.append(v * 100 if (v is not None and k == "cost_per_ep") else v)
            a_.plot(budgets, ys, color=COLOR[ag], marker=MARK[ag], ms=5, lw=2, label=LABEL[ag],
                    mec="#fcfcfb", mew=1, ls="--" if ag == "SingleShot" else "-")
        a_.set_title(t, fontsize=10, loc="left", color="#0b0b0b")
        a_.set_xticks(budgets)
        a_.grid(axis="y", color="#e4e3df", lw=0.6)
        a_.tick_params(colors="#52514e")
        if k != "cost_per_ep":
            a_.set_ylim(-0.03, 1.03)
    ax[2].set_xlabel("Budget B (units)")
    ax[3].set_xlabel("Budget B (units)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(outdir, "fig_biobudget.png"), dpi=200, facecolor="white")
    print(json.dumps({k: v for k, v in out.items() if k in ("n_records", "n_episodes", "n_api_error", "cost_total_usd")}))
    for key, m in out["cells"].items():
        print(f"{key:16s} succ={m['success']:.3f} {m['success_ci']} cov={m['coverage']:.3f} prec={m['precision']:.3f} "
              f"abst={m['abstain']:.3f} viol={m['violation']:.3f} inv={m['invalid']:.3f} auc={m['auc']:.3f} "
              f"units={m['mean_units']:.2f} calls={m['mean_calls']:.2f} tok={m['mean_tokens']:.0f} cents/ep={100*m['cost_per_ep']:.3f}")
    for k, v in out["paired"].items():
        print(k, round(v["diff"], 3), [round(x, 3) for x in v["ci"]])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
