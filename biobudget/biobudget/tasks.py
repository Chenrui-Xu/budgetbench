"""LAB-Bench SeqQA loader for the BioBudget pilot.

Choices are built exactly as in LAB-Bench (ideal + refuse option + distractors,
shuffled, lettered A, B, C, ...), with a per-question seeded shuffle so that every
agent sees the same letters for the same question.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
import string

SUBTASKS = [
    "ORF-seq-AAid", "ORF-seq-AAseq", "ORF-seq-numlen",
    "Prop-seq-gcpercent",
    "RE-seq-lenfrags", "RE-seq-numfrags",
    "PCR-primers-len", "PCR-len-primers", "PCR-seq-primers",
]
REFUSE_CHOICE = "Insufficient information to answer the question"
ALPHABET = string.ascii_uppercase
DNA_RE = re.compile(r"[ACGTU]{15,}")


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def load_questions(data_dir: str, per_subtask: int = 20, seed: int = 0,
                   subtasks=None, file_pattern: str = "{sub}-v1-public.jsonl") -> list[dict]:
    out = []
    for sub in (subtasks or SUBTASKS):
        path = os.path.join(data_dir, file_pattern.format(sub=sub))
        rows = [json.loads(l) for l in open(path) if l.strip()]
        rows.sort(key=lambda r: r["id"])
        rng = random.Random(_seed(seed, sub))
        picked = rng.sample(rows, min(per_subtask, len(rows)))
        for r in picked:
            d = r["distractors"]
            if isinstance(d, str):
                d = ast.literal_eval(d)
            choices = [str(r["ideal"]), REFUSE_CHOICE, *map(str, d)]
            perm = list(range(len(choices)))
            random.Random(_seed(seed, r["id"])).shuffle(perm)
            letters = ALPHABET[: len(choices)]
            shown = {letters[k]: choices[perm[k]] for k in range(len(choices))}
            refs = {}
            for k, m in enumerate(DNA_RE.finditer(r["question"]), start=1):
                refs[f"$Q{k}"] = m.group(0)
            for L, text in shown.items():
                for k, m in enumerate(DNA_RE.finditer(text), start=1):
                    refs[f"${L}{k}"] = m.group(0)
            out.append({
                "id": r["id"], "subtask": sub, "question": r["question"],
                "choices": shown, "answer": letters[perm.index(0)],
                "refuse": letters[perm.index(1)], "ideal": str(r["ideal"]),
                "refs": refs,
            })
    return out


def render_question(q: dict, with_refs: bool) -> str:
    lines = [f"Question: {q['question']}", "", "Options:"]
    lines += [f"({L}) {t}" for L, t in q["choices"].items()]
    if with_refs and q["refs"]:
        lines += ["", "Sequence references you can pass to tools instead of copying sequences:"]
        for name, s in q["refs"].items():
            where = "question" if name.startswith("$Q") else f"option {name[1]}"
            lines.append(f"  {name}: {len(s)} nt from the {where} (starts {s[:12]}...)")
    return "\n".join(lines)
