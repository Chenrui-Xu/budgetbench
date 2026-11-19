"""BioBudget pilot: budget-constrained LLM agents on LAB-Bench SeqQA.

Usage (from ~/budgetbench/biobudget):
    python run_pilot.py --smoke            # 3 questions x 3 agents, prints cost
    python run_pilot.py                    # full pilot (resumable)

Credentials are read exactly like the original BudgetBench agents:
GEMINI_API_KEY if set, otherwise Vertex AI with GOOGLE_CLOUD_PROJECT
(default "ai-agent-for-engineers"). Secrets are never printed.

Kill switch: create the file results/STOP to stop launching new episodes.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from biobudget.agents import AGENTS, LLM  # noqa: E402
from biobudget.tasks import load_questions  # noqa: E402

PRICE_IN, PRICE_OUT = 0.30e-6, 2.50e-6  # USD per token, Gemini 2.5 Flash (output incl. thinking)
LOG_PATH = None


def log(msg):
    print(msg, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def make_client():
    from google import genai
    from google.genai import types
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        # v1beta: the thinking-budget setting is not accepted on the v1 endpoint
        return (genai.Client(api_key=key, http_options=types.HttpOptions(api_version="v1beta")),
                "Gemini API key (GEMINI_API_KEY)")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "ai-agent-for-engineers")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    return (genai.Client(vertexai=True, project=project, location=location),
            f"Vertex AI, Google Cloud project '{project}' ({location})")


def cost_of(calls):
    return sum(c["in"] * PRICE_IN + (c["out"] + c["think"]) * PRICE_OUT for c in calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "SeqQA"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "episodes.jsonl"))
    ap.add_argument("--per-subtask", type=int, default=20)
    ap.add_argument("--budgets", type=int, nargs="+", default=[3, 6, 12])
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cost-cap", type=float, default=20.0)
    ap.add_argument("--thinking-budget", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cloning", action="store_true", help="run LAB-Bench CloningScenarios instead of SeqQA")
    ap.add_argument("--protocol", action="store_true", help="run LAB-Bench ProtocolQA under a reading budget")
    ap.add_argument("--agents", nargs="+", default=["SingleShot", "Loop", "BudgetAware"])
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    global LOG_PATH
    LOG_PATH = os.path.join(os.path.dirname(a.out), "run.log")
    stop_file = os.path.join(os.path.dirname(a.out), "STOP")
    agents_map = AGENTS
    if a.protocol:
        from biobudget import protocol as P
        agents_map = P.AGENTS
        qs = P.load_protocol_questions(os.path.join(HERE, "data", "ProtocolQA"))
        a.out = a.out.replace("episodes.jsonl", "protocol_episodes.jsonl")
        if a.smoke:
            qs, a.budgets = qs[:3], [6]
            a.out = a.out.replace("protocol_episodes.jsonl", "protocol_smoke.jsonl")
    elif a.cloning:
        qs = load_questions(os.path.join(HERE, "data", "CloningScenarios"), per_subtask=1000,
                            subtasks=["CloningScenarios"], file_pattern="cloningscenarios-v1-public.jsonl")
        a.out = a.out.replace("episodes.jsonl", "cloning_episodes.jsonl")
    else:
        qs = load_questions(a.data, per_subtask=a.per_subtask)
    if a.smoke and not a.protocol:
        picked, seen = [], set()
        for q in qs:
            if q["subtask"] in ("ORF-seq-AAid", "PCR-primers-len", "RE-seq-numfrags") and q["subtask"] not in seen:
                picked.append(q)
                seen.add(q["subtask"])
        qs, a.budgets = picked, [6]
        a.out = a.out.replace("episodes.jsonl", "smoke.jsonl")

    _, auth = make_client()
    log(f"[auth] {auth}")
    log(f"[plan] {len(qs)} questions, budgets {a.budgets}, model {a.model}, "
        f"thinking_budget {a.thinking_budget}, cost cap ${a.cost_cap:.2f}")

    jobs = []
    for q in qs:
        if "SingleShot" in a.agents:
            jobs.append(("SingleShot", q, 0))
        for B in a.budgets:
            for ag in a.agents:
                if ag != "SingleShot":
                    jobs.append((ag, q, B))

    done = set()
    spent_usd = 0.0
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip():
                r = json.loads(line)
                if r["status"] != "API_ERROR":
                    done.add((r["agent"], r["qid"], r["B"]))
                spent_usd += r.get("cost_usd", 0.0)
    todo = [j for j in jobs if (j[0], j[1]["id"], j[2]) not in done]
    log(f"[resume] {len(done)} episodes already done, {len(todo)} to run, "
        f"${spent_usd:.2f} spent so far")

    # cost projection: from the smoke test if it exists, otherwise a rough range
    smoke_path = os.path.join(os.path.dirname(a.out), "protocol_smoke.jsonl" if a.protocol else ("episodes.jsonl" if a.cloning else "smoke.jsonl"))
    per_agent = collections.defaultdict(list)
    if not a.smoke and os.path.exists(smoke_path):
        for line in open(smoke_path):
            if line.strip():
                r = json.loads(line)
                if r["status"] != "API_ERROR":
                    per_agent[r["agent"]].append(r["cost_usd"])
    if per_agent:
        proj = sum((sum(per_agent[j[0]]) / len(per_agent[j[0]])) if per_agent.get(j[0]) else 0.01
                   for j in todo)
        log(f"[estimate] projected cost for {len(todo)} episodes: about ${proj:.2f} "
            f"(from the smoke test; real runs vary), hard cap ${a.cost_cap:.2f}")
    else:
        log(f"[estimate] {len(todo)} episodes; rough cost "
            f"{'a few cents' if a.smoke else '$5-15'}, hard cap ${a.cost_cap:.2f}")
    if todo and not a.yes:
        log(f">>> This will call {a.model} and bill: {auth}")
        log(">>> Press Enter to start, or Ctrl+C to cancel.")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            log("[cancelled] nothing was run, nothing was billed.")
            sys.exit(3)

    lock = threading.Lock()
    local = threading.local()
    state = {"cost": spent_usd, "n": 0, "stop": None}
    counts = collections.Counter()

    def llm():
        if not hasattr(local, "llm"):
            client, _ = make_client()
            local.llm = LLM(client, a.model, a.thinking_budget, a.temperature)
        return local.llm

    def work(job):
        agent, q, B = job
        if state["stop"]:
            return None
        if os.path.exists(stop_file):
            state["stop"] = "STOP file found"
            return None
        if state["cost"] >= a.cost_cap:
            state["stop"] = f"cost cap ${a.cost_cap:.2f} reached"
            return None
        t0 = time.time()
        try:
            ep = agents_map[agent](llm(), q, B)
            status, calls = ep.status, ep.calls
            rec = {"choice": ep.choice, "spent": ep.spent, "answered_at": ep.answered_at,
                   "best_guess": ep.best_guess, "steps": ep.steps, "raw_tail": ep.raw_tail}
        except Exception as e:
            status, calls, rec = "API_ERROR", [], {"error": f"{type(e).__name__}: {str(e)[:1500]}"}
        c = cost_of(calls)
        row = {"agent": agent, "B": B, "qid": q["id"], "subtask": q["subtask"],
               "status": status, "answer": q["answer"], "refuse": q["refuse"],
               "n_llm_calls": len(calls),
               "tokens_in": sum(x["in"] for x in calls),
               "tokens_out": sum(x["out"] for x in calls),
               "tokens_think": sum(x["think"] for x in calls),
               "cost_usd": round(c, 6), "seconds": round(time.time() - t0, 2),
               "model": a.model, **rec}
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            state["cost"] += c
            state["n"] += 1
            counts[(agent, status)] += 1
            state["consec_err"] = state.get("consec_err", 0) + 1 if status == "API_ERROR" else 0
            if state["consec_err"] >= 5 and not state["stop"]:
                state["stop"] = "5 API errors in a row (last: " + rec.get("error", "")[:300] + ")"
            if state["n"] % 25 == 0 or a.smoke:
                log(f"[progress] {state['n']}/{len(todo)}  cost ${state['cost']:.2f}  "
                f"last: {agent} B={B} {status}")
        if status == "API_ERROR" and state["n"] <= 3:
            state["stop"] = "API error on first calls: " + rec["error"][:200]
        return status

    with ThreadPoolExecutor(max_workers=1 if a.smoke else a.workers) as ex:
        for _ in as_completed([ex.submit(work, j) for j in todo]):
            pass

    log("[summary] " + str(dict(sorted((f"{k[0]}:{k[1]}", v) for k, v in counts.items()))))
    log(f"[cost] estimated total ${state['cost']:.2f} (list price, Gemini 2.5 Flash)")
    if state["stop"]:
        log(f"[stopped early] {state['stop']}")
    with open(os.path.join(os.path.dirname(a.out), ("DONE_protocol_smoke" if a.smoke else "DONE_protocol") if a.protocol else ("DONE_smoke" if a.smoke else ("DONE_cloning" if a.cloning else "DONE"))), "w") as f:
        json.dump({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "episodes_this_run": state["n"],
                   "estimated_cost_usd": round(state["cost"], 4), "stopped_early": state["stop"]}, f)
    log("[exit] all workers finished; the process will now exit.")
    if state["stop"] and (state["stop"].startswith("API error") or "API errors in a row" in state["stop"]):
        sys.exit(2)


if __name__ == "__main__":
    main()
