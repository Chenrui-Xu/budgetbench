"""ProtocolQA under a reading budget (BioBudget, third task family).

LAB-Bench ProtocolQA gives a wet-lab protocol into which one error has been
introduced, describes the failed outcome, and asks which change would fix it.
Here the protocol text is NOT shown to the tool-using agents. They must pay to
read it: the protocol is cut into sections of at most SECTION_CHARS characters
and every section read costs one budget unit. Answering is free.

Tools (prices in budget units)
  outline()            1   first line and length of every section
  search(keyword)      1   lines that contain the keyword, with section numbers
  read_section(n)      1   full text of section n
  read_protocol()      N   full text (N = number of sections); pruned in MEDIUM

Agents: SingleShot (full protocol in the prompt, no budget; the LAB-Bench
setting), Loop, BudgetAware, BudgetAwareRead, with the same runner semantics
and regimes as biobudget.agents.
"""
from __future__ import annotations

import json
import os
import random
import re

from .agents import HIGH, MEDIUM, MCQ_TEMPLATE, _parse_json
from .tasks import ALPHABET, REFUSE_CHOICE, _seed

SECTION_CHARS = 1200
EXPENSIVE_TOOL = "read_protocol"
TOOL_DOCS = {
    "outline": '{"tool": "outline", "args": {}} -> first line and length of every section of the protocol',
    "search": '{"tool": "search", "args": {"keyword": "<word or phrase>"}} -> protocol lines containing the keyword, with section numbers',
    "read_section": '{"tool": "read_section", "args": {"n": <section number>}} -> full text of one section',
    "read_protocol": '{"tool": "read_protocol", "args": {}} -> full text of the whole protocol',
}

SYSTEM = """You are an expert experimental biologist troubleshooting a laboratory protocol.
One step of the protocol contains an error that caused the problem described in the question. The protocol "{title}" has {n} sections, numbered 1 to {n}. Its text is not shown; you can only see it through tools.

At each step output ONLY one JSON object, either a tool call:
  {{"tool": "<name>", "args": {{...}}, "best_guess": "<option letter you currently think is correct, or null>"}}
or your final answer, which ends the episode:
  {{"tool": "answer", "choice": "<option letter>"}}

Available tools:
{tool_docs}
  {{"tool": "answer", "choice": "<letter>"}} -> final answer

If you cannot determine the answer, choose the option "Insufficient information to answer the question".
"""

REGIME_TEXT = {
    "HIGH": "Budget regime HIGH: you have ample budget. Read what you need to find the faulty step.",
    "MEDIUM": "Budget regime MEDIUM: budget is limited. Read only the sections most likely to contain the faulty step and move toward answering. read_protocol is not available.",
}


def split_sections(text: str) -> list[str]:
    """Greedy packing of whole lines into sections of at most SECTION_CHARS characters."""
    out, cur = [], ""
    for line in text.split("\n"):
        while len(line) > SECTION_CHARS:           # a single very long line
            if cur:
                out.append(cur); cur = ""
            out.append(line[:SECTION_CHARS]); line = line[SECTION_CHARS:]
        if cur and len(cur) + 1 + len(line) > SECTION_CHARS:
            out.append(cur); cur = line
        else:
            cur = line if not cur else cur + "\n" + line
    if cur.strip():
        out.append(cur)
    return [s for s in out if s.strip()]


def load_protocol_questions(data_dir: str, seed: int = 0) -> list[dict]:
    path = os.path.join(data_dir, "protocolqa-v1-public.jsonl")
    rows = sorted((json.loads(l) for l in open(path) if l.strip()), key=lambda r: r["id"])
    out = []
    for r in rows:
        choices = [str(r["ideal"]), REFUSE_CHOICE, *map(str, r["distractors"])]
        perm = list(range(len(choices)))
        random.Random(_seed(seed, r["id"])).shuffle(perm)
        letters = ALPHABET[: len(choices)]
        shown = {letters[k]: choices[perm[k]] for k in range(len(choices))}
        secs = split_sections(r["protocol"])
        title = next((l.strip() for l in r["protocol"].split("\n") if l.strip()), "protocol")[:120]
        out.append({"id": r["id"], "subtask": "ProtocolQA", "question": r["question"],
                    "choices": shown, "answer": letters[perm.index(0)],
                    "refuse": letters[perm.index(1)], "ideal": str(r["ideal"]), "refs": {},
                    "protocol": r["protocol"], "sections": secs, "title": title})
    return out


def _render(q: dict, with_protocol: bool) -> str:
    lines = []
    if with_protocol:
        lines += ["Protocol:", q["protocol"], ""]
    lines += [f"Question: {q['question']}", "", "Options:"]
    lines += [f"({L}) {t}" for L, t in q["choices"].items()]
    return "\n".join(lines)


def price(name: str, q: dict) -> int:
    return len(q["sections"]) if name == "read_protocol" else 1


def run_tool(name: str, args: dict, q: dict):
    secs = q["sections"]
    if name == "outline":
        return [{"section": i, "chars": len(s), "first_line": s.strip().split("\n")[0][:90]}
                for i, s in enumerate(secs, 1)]
    if name == "search":
        kw = str(args.get("keyword", "")).strip().lower()
        if not kw:
            return {"error": "empty keyword"}
        hits = [{"section": i, "line": l.strip()[:200]}
                for i, s in enumerate(secs, 1) for l in s.split("\n") if kw in l.lower()]
        return {"n_matches": len(hits), "matches": hits[:8]}
    if name == "read_section":
        n = int(args.get("n", 0))
        if not 1 <= n <= len(secs):
            return {"error": f"section must be between 1 and {len(secs)}"}
        return {"section": n, "text": secs[n - 1]}
    if name == "read_protocol":
        return {"text": q["protocol"]}
    raise KeyError(name)


class Episode:
    def __init__(self, q, B):
        self.q, self.B, self.spent = q, B, 0
        self.steps, self.calls = [], []
        self.status = self.choice = self.best_guess = self.answered_at = self.raw_tail = None

    @property
    def remaining(self):
        return self.B - self.spent

    def tool(self, name, args):
        p = price(name, self.q)
        if self.spent + p > self.B:                       # checked before execution
            self.status = "BUDGET_EXCEEDED"
            self.steps.append({"tool": name, "price": p, "executed": False})
            return {}
        self.spent += p
        try:
            res = run_tool(name, args or {}, self.q)
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        self.steps.append({"tool": name, "args": {k: str(v)[:40] for k, v in (args or {}).items()},
                           "price": p, "result_chars": len(json.dumps(res))})
        return res

    def answer(self, letter):
        letter = (letter or "").strip().upper()[:1]
        self.choice, self.answered_at = letter, self.spent
        self.status = ("OK_SOLVED" if letter == self.q["answer"] else
                       "ABSTAINED" if letter == self.q["refuse"] else
                       "WRONG_ANSWER" if letter in self.q["choices"] else "INVALID_FORMAT")


def _history(results):
    if not results:
        return "(no tool calls yet)"
    return "\n".join(f"Step {k}: {json.dumps(c)[:300]}\n  -> {json.dumps(r, ensure_ascii=False)}"
                     for k, (c, r) in enumerate(results, 1))


def run_single_shot(llm, q, B):
    ep = Episode(q, B)
    text, use = llm(MCQ_TEMPLATE.format(question=_render(q, True)), json_mode=False)
    ep.calls.append(use)
    ep.raw_tail = (text or "")[-300:]
    m = re.search(r"\[ANSWER\]\s*\**\(?\s*([A-Za-z])\b", text or "", re.I)
    if not m:  # the model sometimes ignores the tags and ends with \boxed{X} or \boxed{\text{X}}
        b = re.findall(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-Za-z])\)?\s*\}", text or "")
        m = re.match(r"(.)", b[-1]) if b else None
    ep.answer(m.group(1) if m else None)
    return ep


def _system(q, docs):
    return SYSTEM.format(title=q["title"], n=len(q["sections"]), tool_docs=docs)


def _final_read(llm, ep, q, B, results, max_calls):
    prompt = (_system(q, "  (no tools left)") + "\n" + _render(q, False) + "\n\n"
              + f"Total budget: {B} units. Remaining budget: {ep.remaining} units. "
              + "Budget regime LOW: no tools are available. Read the results below and "
              + "answer now with {\"tool\": \"answer\", \"choice\": \"<letter>\"}.\n\n"
              + "Tool call history:\n" + _history(results) + "\n\nAnswer (JSON only):")
    choice = None
    if len(ep.calls) < max_calls + 1:
        text, use = llm(prompt, json_mode=True)
        ep.calls.append(use)
        try:
            a = _parse_json(text)
            if a.get("tool") == "answer":
                choice = str(a.get("choice", ""))
        except Exception:
            pass
    ep.answer(choice or ep.best_guess or q["refuse"])
    ep.steps.append({"tool": "answer", "regime": "LOW", "final_read": True, "choice": ep.choice})


def _tool_loop(llm, q, B, budget_aware, final_read=False):
    ep, results = Episode(q, B), []
    max_calls = B + 4
    while ep.status is None:
        regime, offered = None, list(TOOL_DOCS)
        if budget_aware:
            f = ep.remaining / B
            regime = "HIGH" if f > HIGH else ("MEDIUM" if f > MEDIUM else "LOW")
            if regime == "LOW":
                if final_read:
                    _final_read(llm, ep, q, B, results, max_calls)
                else:
                    ep.answer(ep.best_guess or q["refuse"])
                    ep.steps.append({"tool": "answer", "regime": "LOW", "fallback": True, "choice": ep.choice})
                break
            offered = [t for t in offered if price(t, q) <= ep.remaining
                       and not (regime == "MEDIUM" and t == EXPENSIVE_TOOL)]
        docs = "\n".join("  " + TOOL_DOCS[t] + (f"  [price {price(t, q)}]" if budget_aware else "")
                         for t in offered)
        prompt = _system(q, docs) + "\n" + _render(q, False) + "\n\n"
        if budget_aware:
            prompt += (f"Total budget: {B} units. Remaining budget: {ep.remaining} units. "
                       f"The final answer is free.\n{REGIME_TEXT[regime]}\n\n")
        prompt += "Tool call history:\n" + _history(results) + "\n\nNext step (JSON only):"
        action, err = None, None
        for attempt in range(2):
            if len(ep.calls) >= max_calls:
                break
            text, use = llm(prompt if attempt == 0 else prompt + f"\n\nYour previous output was rejected: {err}. Output one valid JSON object.", json_mode=True)
            ep.calls.append(use)
            try:
                a = _parse_json(text)
                if a.get("tool") != "answer" and a.get("tool") not in offered:
                    raise ValueError(f"tool '{a.get('tool')}' is not available")
                action = a
                break
            except Exception as e:
                err = str(e)[:200]
        if action is None:
            if budget_aware and final_read:
                _final_read(llm, ep, q, B, results, max_calls)
            elif budget_aware:
                ep.answer(ep.best_guess or q["refuse"])
                ep.steps.append({"tool": "answer", "fallback": True, "choice": ep.choice})
            else:
                ep.status = "INVALID_FORMAT"
            break
        if action.get("best_guess"):
            g = str(action["best_guess"]).strip().upper()[:1]
            if g in q["choices"]:
                ep.best_guess = g
        if action["tool"] == "answer":
            ep.answer(str(action.get("choice", "")))
            ep.steps.append({"tool": "answer", "choice": ep.choice, "regime": regime})
            break
        res = ep.tool(action["tool"], action.get("args") or {})
        if ep.status == "BUDGET_EXCEEDED":
            break
        results.append(({"tool": action["tool"], "args": action.get("args")}, res))
    return ep


AGENTS = {"SingleShot": run_single_shot,
          "Loop": lambda llm, q, B: _tool_loop(llm, q, B, False),
          "BudgetAware": lambda llm, q, B: _tool_loop(llm, q, B, True),
          "BudgetAwareRead": lambda llm, q, B: _tool_loop(llm, q, B, True, True)}
