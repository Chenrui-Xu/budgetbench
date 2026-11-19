"""Runner and agents for the BioBudget pilot.

Runner semantics follow BudgetBench: every tool call is priced, the price is
checked against the ceiling B *before* the call is executed, an over-budget
call ends the episode as BUDGET_EXCEEDED, and the terminal action (answer) is
free.

Agents
------
SingleShot   one LLM call, no tools (LAB-Bench zero-shot prompt).
Loop         ReAct-style tool loop; budget-blind (no budget or prices in the
             prompt, no fallback).
BudgetAware  same loop + remaining budget and prices in the prompt + three
             regimes on f = R/B: HIGH (f > 0.6) all affordable tools;
             MEDIUM (0.3 < f <= 0.6) the 3-unit find_orfs is removed;
             LOW (f <= 0.3) no LLM call: answer with the latest best guess,
             or the "insufficient information" option if there is none.
             Tools costing more than the remaining budget are never offered.
"""
from __future__ import annotations

import json
import re
import time

from . import tools as T
from .tasks import render_question

HIGH, MEDIUM = 0.6, 0.3
MAX_RESULT_CHARS = 1500

MCQ_TEMPLATE = """The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.

{question}

You MUST include the letter of the correct answer within the following tags: [ANSWER] and [/ANSWER].
For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is the correct letter.
Always answer in exactly this format of a single letter between the two tags, even if you are unsure.
We require this because we use automatic parsing."""

AGENT_SYSTEM = """You are an expert molecular biologist answering a multiple-choice question about DNA or protein sequences.
You can call deterministic sequence-analysis tools. Long sequences from the question and options can be passed to tools by reference (e.g. "$Q1", "$B2") instead of copying them.

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
    "HIGH": "Budget regime HIGH: you have ample budget. Use tools to compute the answer.",
    "MEDIUM": "Budget regime MEDIUM: budget is limited. Use only targeted, cheap tool calls and move toward answering. find_orfs is not available.",
}


def _usage(resp) -> dict:
    u = getattr(resp, "usage_metadata", None)
    g = lambda k: int(getattr(u, k, 0) or 0) if u is not None else 0  # noqa: E731
    return {"in": g("prompt_token_count"), "out": g("candidates_token_count"),
            "think": g("thoughts_token_count")}


def _parse_json(text: str):
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise ValueError("no JSON object found")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict) or "tool" not in obj:
        raise ValueError("JSON must contain the key 'tool'")
    return obj


class LLM:
    """Thin wrapper so that the agents can be tested with a fake model."""

    def __init__(self, client, model: str, thinking_budget: int, temperature: float,
                 max_output_tokens: int = 4096):
        self.client, self.model = client, model
        self.thinking_budget, self.temperature = thinking_budget, temperature
        self.max_output_tokens = max_output_tokens

    def __call__(self, prompt: str, json_mode: bool):
        from google.genai import types  # imported lazily for offline tests
        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
            response_mime_type="application/json" if json_mode else None,
        )
        delay = 5.0
        for attempt in range(8):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg)
                return (resp.text or ""), _usage(resp)
            except Exception as e:
                msg = str(e)
                # 429 (rate limit) and 5xx are retried with backoff; everything else is fatal
                retryable = any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "500", "502",
                                                   "503", "504", "UNAVAILABLE", "INTERNAL",
                                                   "DEADLINE_EXCEEDED", "timed out"))
                if not retryable or attempt == 7:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)


class Episode:
    def __init__(self, q: dict, B: int):
        self.q, self.B = q, B
        self.spent = 0
        self.steps: list[dict] = []
        self.calls: list[dict] = []
        self.status: str | None = None
        self.choice: str | None = None
        self.best_guess: str | None = None
        self.answered_at: int | None = None
        self.raw_tail: str | None = None

    @property
    def remaining(self) -> int:
        return self.B - self.spent

    def resolve(self, v):
        if isinstance(v, str) and v.strip() in self.q["refs"]:
            return self.q["refs"][v.strip()]
        if isinstance(v, list):
            return [self.resolve(x) for x in v]
        return v

    def tool(self, name: str, args: dict) -> dict:
        """Runner step for a non-terminal tool call (budget checked first)."""
        price = T.PRICES[name]
        if self.spent + price > self.B:
            self.status = "BUDGET_EXCEEDED"
            self.steps.append({"tool": name, "price": price, "executed": False})
            return {}
        self.spent += price
        try:
            res = T.TOOLS[name](**{k: self.resolve(v) for k, v in (args or {}).items()})
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        short_args = {k: (v if not isinstance(v, str) or len(v) <= 40 else v[:37] + "...")
                      for k, v in (args or {}).items()}
        self.steps.append({"tool": name, "args": short_args, "price": price,
                           "result": json.dumps(res)[:300]})
        return res

    def answer(self, letter: str | None):
        letter = (letter or "").strip().upper()[:1]
        self.choice = letter
        self.answered_at = self.spent
        if letter == self.q["answer"]:
            self.status = "OK_SOLVED"
        elif letter == self.q["refuse"]:
            self.status = "ABSTAINED"
        elif letter in self.q["choices"]:
            self.status = "WRONG_ANSWER"
        else:
            self.status = "INVALID_FORMAT"


def _history_text(ep: Episode, results: list) -> str:
    if not results:
        return "(no tool calls yet)"
    lines = []
    for k, (call, res) in enumerate(results, start=1):
        lines.append(f"Step {k}: {json.dumps(call)[:400]}\n  -> {json.dumps(res)[:MAX_RESULT_CHARS]}")
    return "\n".join(lines)


def run_single_shot(llm, q: dict, B: int) -> Episode:
    ep = Episode(q, B)
    text, use = llm(MCQ_TEMPLATE.format(question=render_question(q, with_refs=False)), json_mode=False)
    ep.calls.append(use)
    ep.raw_tail = (text or "")[-300:]
    m = re.search(r"\[ANSWER\]\s*\**\(?\s*([A-Za-z])\b", text or "", re.I)
    if not m:  # the model sometimes ignores the tags and ends with \boxed{X} or \boxed{\text{X}}
        b = re.findall(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-Za-z])\)?\s*\}", text or "")
        m = re.match(r"(.)", b[-1]) if b else None
    ep.answer(m.group(1) if m else None)
    return ep


def _final_read(llm, ep, q, B, question, results, max_llm_calls):
    """Revised LOW fallback: one last model call that can only answer (free)."""
    prompt = (AGENT_SYSTEM.format(tool_docs="  (no tools left)") + "\n" + question + "\n\n"
              + f"Total budget: {B} units. Remaining budget: {ep.remaining} units. "
              + "Budget regime LOW: no tools are available. Read the results below and "
              + "answer now with {\"tool\": \"answer\", \"choice\": \"<letter>\"}.\n\n"
              + "Tool call history:\n" + _history_text(ep, results) + "\n\nAnswer (JSON only):")
    choice = None
    if len(ep.calls) < max_llm_calls + 1:
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


def _tool_loop(llm, q: dict, B: int, budget_aware: bool, final_read: bool = False) -> Episode:
    ep = Episode(q, B)
    results: list = []
    question = render_question(q, with_refs=True)
    max_llm_calls = B + 4
    while ep.status is None:
        regime = None
        offered = list(T.TOOL_DOCS)
        if budget_aware:
            f = ep.remaining / B
            regime = "HIGH" if f > HIGH else ("MEDIUM" if f > MEDIUM else "LOW")
            if regime == "LOW" and final_read:
                _final_read(llm, ep, q, B, question, results, max_llm_calls)
                break
            if regime == "LOW":
                ep.answer(ep.best_guess or q["refuse"])
                ep.steps.append({"tool": "answer", "regime": "LOW", "fallback": True,
                                 "choice": ep.choice})
                break
            offered = [t for t in offered if T.PRICES[t] <= ep.remaining
                       and not (regime == "MEDIUM" and t == T.EXPENSIVE_TOOL)]
        docs = "\n".join("  " + T.TOOL_DOCS[t] + (f"  [price {T.PRICES[t]}]" if budget_aware else "")
                         for t in offered)
        prompt = AGENT_SYSTEM.format(tool_docs=docs) + "\n" + question + "\n\n"
        if budget_aware:
            prompt += (f"Total budget: {B} units. Remaining budget: {ep.remaining} units. "
                       f"The final answer is free.\n{REGIME_TEXT[regime]}\n\n")
        prompt += "Tool call history:\n" + _history_text(ep, results) + "\n\nNext step (JSON only):"

        action, err = None, None
        for attempt in range(2):  # one re-prompt on malformed or unavailable actions
            if len(ep.calls) >= max_llm_calls:
                break
            text, use = llm(prompt if attempt == 0 else prompt + f"\n\nYour previous output was rejected: {err}. Output one valid JSON object.", json_mode=True)
            ep.calls.append(use)
            try:
                a = _parse_json(text)
                name = a.get("tool")
                if name != "answer" and name not in offered:
                    raise ValueError(f"tool '{name}' is not available")
                action = a
                break
            except Exception as e:
                err = str(e)[:200]
        if action is None:
            if budget_aware and final_read:
                _final_read(llm, ep, q, B, question, results, max_llm_calls)
            elif budget_aware:  # fall back exactly as in the LOW regime
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


def run_loop(llm, q, B):
    return _tool_loop(llm, q, B, budget_aware=False)


def run_budget_aware(llm, q, B):
    return _tool_loop(llm, q, B, budget_aware=True)


def run_budget_aware_read(llm, q, B):
    return _tool_loop(llm, q, B, budget_aware=True, final_read=True)


AGENTS = {"SingleShot": run_single_shot, "Loop": run_loop, "BudgetAware": run_budget_aware,
          "BudgetAwareRead": run_budget_aware_read}
