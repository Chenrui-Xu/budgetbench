"""Re-score SingleShot episodes whose answer was given as a final \\boxed{letter}
instead of the requested [ANSWER] tags. Tool-using agents answer in JSON and are unaffected.
Usage: python rescore.py in.jsonl out.jsonl
"""
import json, re, sys, collections
PAT = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-Za-z])\)?\s*\}?\s*\}")
n = collections.Counter()
with open(sys.argv[2], "w") as out:
    for line in open(sys.argv[1]):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["agent"] == "SingleShot" and r["status"] == "INVALID_FORMAT":
            m = PAT.findall(r.get("raw_tail") or "")
            if m:
                L = m[-1].upper()
                r["status_strict"] = "INVALID_FORMAT"
                r["choice"] = L
                r["status"] = ("OK_SOLVED" if L == r["answer"] else
                               "ABSTAINED" if L == r["refuse"] else "WRONG_ANSWER")
                n[r["status"]] += 1
        out.write(json.dumps(r) + "\n")
print("rescored", dict(n))
