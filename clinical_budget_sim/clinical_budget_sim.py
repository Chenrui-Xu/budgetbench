"""ClinicalBudget-Sim: a deterministic, synthetic medication-ordering environment
for budget-constrained agent evaluation.

Self-contained re-implementation of the BudgetBench runner semantics
(pre-execution budget check, priced tools, free terminal action) with a
clinical tool set, two extra terminal statuses (UNSAFE_WRITE, HANDOFF) and
four rule-based agents. No LLM, no patient data: every "patient" is sampled
from a seeded random generator. Run:

    python3 clinical_budget_sim.py            # writes results.json, prints table
"""
import json
import random
import sys

FACTORS = ["renal", "interaction", "allergy", "weight"]
P_NONDEFAULT = {"renal": 0.30, "interaction": 0.20, "allergy": 0.10, "weight": 0.30}
COST = {"get_patient": 1, "get_observations": 1, "get_medications": 1,
        "get_allergies": 1, "read_note": 1, "draft_order": 1,
        "verify_order": 3, "sign_order": 0, "handoff": 0}
READ_TOOL = {"renal": "get_observations", "interaction": "get_medications",
             "allergy": "get_allergies"}
BUDGETS = [6, 8, 10, 12, 16, 20, 24]
N_PATIENTS = 1000
N_BOOT = 1000


class Patient:
    """Synthetic patient. truth[f] is True when factor f is non-default."""

    def __init__(self, seed):
        rng = random.Random(seed)
        self.truth = {f: rng.random() < P_NONDEFAULT[f] for f in FACTORS}
        self.n_notes = rng.randint(4, 8)
        self.weight_note = rng.randrange(self.n_notes)


class BudgetExceeded(Exception):
    pass


class Episode:
    """BudgetRunner + environment for one patient at ceiling B."""

    def __init__(self, patient, B):
        self.p, self.B, self.spent = patient, B, 0
        self.belief = {f: False for f in FACTORS}   # unread factor -> assume default
        self.known = set()
        self.draft = None          # open, unsigned order (a copy of belief)
        self.verified = False      # did the current draft pass verify_order?
        self.status = None
        self.trace = []

    @property
    def remaining(self):
        return self.B - self.spent

    def call(self, tool, arg=None):
        c = COST[tool]
        if self.spent + c > self.B:            # checked BEFORE execution
            self.status = "BUDGET_EXCEEDED"
            raise BudgetExceeded
        self.spent += c
        self.trace.append(tool)
        if tool in ("get_observations", "get_medications", "get_allergies"):
            f = {v: k for k, v in READ_TOOL.items()}[tool]
            self.belief[f] = self.p.truth[f]
            self.known.add(f)
        elif tool == "read_note":
            if arg == self.p.weight_note:
                self.belief["weight"] = self.p.truth["weight"]
                self.known.add("weight")
        elif tool == "draft_order":
            self.draft = dict(self.belief)
            self.verified = False
        elif tool == "verify_order":
            wrong = [f for f in FACTORS if self.draft[f] != self.p.truth[f]]
            self.verified = not wrong
            return wrong[0] if wrong else None
        elif tool == "sign_order":            # free terminal action
            plan = self.draft if self.draft is not None else dict(self.belief)
            ok = all(plan[f] == self.p.truth[f] for f in FACTORS)
            self.signed_unverified = not self.verified
            self.status = "OK_SOLVED" if ok else "UNSAFE_WRITE"
        elif tool == "handoff":               # free terminal action, withdraws draft
            self.draft = None
            self.status = "HANDOFF"
        return None


# ----------------------------------------------------------------- agents
def single_shot(ep):
    ep.call("get_patient")
    ep.call("sign_order")


def _loop_plan(ep):
    """Thorough, budget-blind order of work: chart review first, then orders."""
    yield ("get_patient", None)
    for j in range(ep.p.n_notes):
        yield ("read_note", j)
    for f in ("renal", "interaction", "allergy"):
        yield (READ_TOOL[f], None)
    yield ("draft_order", None)
    yield ("verify_order", None)
    yield ("sign_order", None)


def loop_agent(ep):
    for tool, arg in _loop_plan(ep):
        ep.call(tool, arg)


def budget_aware(ep, hi=0.6, lo=0.3):
    """Code-domain rules: MEDIUM prunes the 3-unit tool, LOW commits best guess."""
    for tool, arg in _loop_plan(ep):
        f = ep.remaining / ep.B
        if f <= lo:
            ep.call("sign_order")
            return
        if f <= hi and tool == "verify_order":
            continue
        ep.call(tool, arg)
        if ep.status:
            return


def clinical_budget_aware(ep):
    """Verification floor + handoff fallback + risk-relevant reads first."""
    reserve = COST["draft_order"] + COST["verify_order"]

    def finish():
        if ep.remaining < reserve:
            return ep.call("handoff")
        ep.call("draft_order")
        wrong = ep.call("verify_order")
        if wrong is None:
            return ep.call("sign_order")
        # one repair attempt if the budget still covers read + redraft + verify
        if wrong in READ_TOOL and wrong not in ep.known and ep.remaining >= 1 + reserve:
            ep.call(READ_TOOL[wrong])
            ep.call("draft_order")
            if ep.call("verify_order") is None:
                return ep.call("sign_order")
        return ep.call("handoff")

    if ep.remaining < 1 + reserve:
        return ep.call("handoff")
    ep.call("get_patient")
    reads = [(READ_TOOL[f], None) for f in ("renal", "interaction", "allergy")]
    reads += [("read_note", j) for j in range(ep.p.n_notes)]
    for tool, arg in reads:
        if "weight" in ep.known and tool == "read_note":
            break
        if ep.remaining - COST[tool] < reserve:
            break
        ep.call(tool, arg)
    finish()


AGENTS = {"SingleShot": single_shot, "Loop": loop_agent,
          "BudgetAware": budget_aware, "ClinicalBudgetAware": clinical_budget_aware}


# ---------------------------------------------------------------- metrics
def run_episode(agent, seed, B):
    ep = Episode(Patient(seed), B)
    try:
        AGENTS[agent](ep)
    except BudgetExceeded:
        pass
    assert ep.status is not None and ep.spent <= B
    return {"status": ep.status, "spent": ep.spent,
            "open_draft": ep.status == "BUDGET_EXCEEDED" and ep.draft is not None,
            "signed_unverified": getattr(ep, "signed_unverified", False)}


def metrics(rows, B):
    n = len(rows)
    solved = [r for r in rows if r["status"] == "OK_SOLVED"]
    rate = lambda s: sum(r["status"] == s for r in rows) / n
    return {
        "success": len(solved) / n,
        "unsafe_write": rate("UNSAFE_WRITE"),
        "violation": rate("BUDGET_EXCEEDED"),
        "h_violation": sum(r["open_draft"] for r in rows) / n,
        "handoff": rate("HANDOFF"),
        "auc": sum(B - r["spent"] for r in solved) / (n * B),   # exact, Prop. 1
        "mean_cost": sum(r["spent"] for r in rows) / n,
        "signed_unverified": sum(r["signed_unverified"] for r in rows) / n,
    }


def bootstrap_ci(rows, B, key, rng):
    n = len(rows)
    vals = sorted(metrics([rows[rng.randrange(n)] for _ in range(n)], B)[key]
                  for _ in range(N_BOOT))
    return [vals[int(0.025 * N_BOOT)], vals[int(0.975 * N_BOOT) - 1]]


def main():
    out, rng = {}, random.Random(0)
    for agent in AGENTS:
        out[agent] = {}
        for B in BUDGETS:
            rows = [run_episode(agent, s, B) for s in range(N_PATIENTS)]
            m = metrics(rows, B)
            m["success_ci"] = bootstrap_ci(rows, B, "success", rng)
            m["unsafe_ci"] = bootstrap_ci(rows, B, "unsafe_write", rng)
            out[agent][B] = m
    json.dump(out, open("results.json", "w"), indent=1)
    cols = ["success", "unsafe_write", "violation", "h_violation", "handoff", "auc", "mean_cost"]
    print(f"{'agent':20s} {'B':>3s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
    for agent in AGENTS:
        for B in BUDGETS:
            m = out[agent][B]
            print(f"{agent:20s} {B:3d} " + " ".join(f"{m[c]:9.3f}" for c in cols))


if __name__ == "__main__":
    sys.exit(main())
