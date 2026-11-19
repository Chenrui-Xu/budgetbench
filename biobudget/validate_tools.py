"""Tool-only oracle: answer every SeqQA question in the 9 subtasks with the
deterministic tools alone, to check that the tools are correct and sufficient.
No LLM involved. Usage: python3 -I validate_tools.py <LAB-Bench/SeqQA dir>"""
import collections
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from biobudget import tools as T  # noqa: E402
from biobudget.tasks import load_questions  # noqa: E402

AA3 = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu",
       "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe",
       "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val", "*": "Stop"}
FULL = {"Ala": "Alanine", "Arg": "Arginine", "Asn": "Asparagine", "Asp": "Aspartic acid",
        "Cys": "Cysteine", "Gln": "Glutamine", "Glu": "Glutamic acid", "Gly": "Glycine",
        "His": "Histidine", "Ile": "Isoleucine", "Leu": "Leucine", "Lys": "Lysine",
        "Met": "Methionine", "Phe": "Phenylalanine", "Pro": "Proline", "Ser": "Serine",
        "Thr": "Threonine", "Trp": "Tryptophan", "Tyr": "Tyrosine", "Val": "Valine"}


def norm_aa(x):
    x = x.strip()
    for one, three in AA3.items():
        if x.upper() in (one, three.upper(), FULL.get(three, "").upper()):
            return one
    return x


def oracle(q):
    sub, text, ch = q["subtask"], q["question"], q["choices"]
    seq = q["refs"].get("$Q1")
    opts = {L: t for L, t in ch.items() if L != q["refuse"]}
    if sub == "ORF-seq-AAid":
        pos = int(re.search(r"position (\d+)", text).group(1))
        prot = T.find_orfs(seq)["orfs_longest_first"][0]["protein"]
        target = prot[pos - 1]
        return [L for L, t in opts.items() if norm_aa(t) == target]
    if sub == "ORF-seq-AAseq":
        prot = T.find_orfs(seq)["orfs_longest_first"][0]["protein"]
        return [L for L, t in opts.items() if t.strip() == prot]
    if sub == "ORF-seq-numlen":
        n = int(re.search(r"greater than (\d+)", text).group(1))
        cnt = sum(1 for o in T._orfs(T._clean(seq), "+") if o["length_aa"] > n)
        return [L for L, t in opts.items() if t.strip() == str(cnt)]
    if sub == "Prop-seq-gcpercent":
        gc = T.gc_content(seq)["gc_percent"]
        return [L for L, t in opts.items() if t.strip() == str(round(gc))]
    if sub in ("RE-seq-lenfrags", "RE-seq-numfrags"):
        enz = re.search(r"enzymes?:?\s*([A-Za-z0-9 ,]+)\?", text).group(1)
        enz = [e for e in re.split(r"[,\s]+", enz) if e and e.lower() != "and"]
        res = T.digest(seq, enz)
        if sub == "RE-seq-numfrags":
            return [L for L, t in opts.items() if t.strip() == str(res["n_fragments"])]
        want = res["fragment_lengths_in_order"]
        return [L for L, t in opts.items() if [int(x) for x in re.findall(r"\d+", t)] == want]
    if sub == "PCR-primers-len":
        f, r = re.search(r"primer pair ([ACGT]+),\s*([ACGT]+)", text).groups()
        tmpl = max(q["refs"].values(), key=len)
        res = T.pcr(tmpl, f, r)
        return [L for L, t in opts.items() if res.get("product") and int(t) in res["amplicon_lengths_bp"]]
    if sub in ("PCR-len-primers", "PCR-seq-primers"):
        if sub == "PCR-len-primers":
            tmpl = q["refs"]["$Q1"]
            want = int(re.search(r"(\d+) bp", text).group(1))
        else:
            tmpl = q["refs"]["$Q1"]  # the amplicon itself; primers must reproduce it
            want = len(tmpl)
        good = []
        for L, t in opts.items():
            ps = re.findall(r"[ACGT]{10,}", t)
            if len(ps) == 2:
                res = T.pcr(tmpl, ps[0], ps[1])
                if res.get("product") and want in res["amplicon_lengths_bp"]:
                    good.append(L)
        return good
    raise ValueError(sub)


def main(data_dir):
    qs = load_questions(data_dir, per_subtask=40)
    tally = collections.defaultdict(lambda: [0, 0, 0])
    bad = []
    for q in qs:
        got = oracle(q)
        t = tally[q["subtask"]]
        t[0] += 1
        if got == [q["answer"]]:
            t[1] += 1
        elif q["answer"] in got:
            t[2] += 1
            bad.append((q["subtask"], q["id"], "ambiguous", got, q["answer"]))
        else:
            bad.append((q["subtask"], q["id"], "wrong", got, q["answer"], q["ideal"][:40]))
    for s, (n, ok, amb) in tally.items():
        print(f"{s:22s} n={n:3d} unique-correct={ok:3d} ambiguous={amb}")
    print("total", sum(v[0] for v in tally.values()), "unique-correct", sum(v[1] for v in tally.values()))
    for b in bad[:25]:
        print(b)


if __name__ == "__main__":
    main(sys.argv[1])
