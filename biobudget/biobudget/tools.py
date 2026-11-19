"""Deterministic sequence tools for the BioBudget pilot (Biopython).

Every tool takes plain strings, returns a small JSON-serialisable dict and never
touches the network or the file system. Long sequences can be passed by
reference ($Q1, $B2, ...) and are resolved by the environment before the tool
is called.
"""
from __future__ import annotations

import re

from Bio import Restriction
from Bio.Seq import Seq

# Price of each tool in budget units. Prices follow the size of the tool's
# typical output (token cost): scalar results 1, full-sequence results 2,
# the six-frame ORF scan 3. The terminal action `answer` is free.
PRICES = {
    "gc_content": 1,
    "digest": 1,
    "pcr": 1,
    "find_sequence": 1,
    "reverse_complement": 2,
    "translate": 2,
    "find_orfs": 3,
    "answer": 0,
}
EXPENSIVE_TOOL = "find_orfs"

STOPS = {"TAA", "TAG", "TGA"}


def _clean(seq: str) -> str:
    s = re.sub(r"[^A-Za-z]", "", str(seq)).upper().replace("U", "T")
    if not s:
        raise ValueError("empty sequence")
    return s


def gc_content(sequence: str) -> dict:
    s = _clean(sequence)
    gc = sum(c in "GC" for c in s)
    return {"length": len(s), "gc_percent": round(100.0 * gc / len(s), 2)}


def reverse_complement(sequence: str) -> dict:
    s = _clean(sequence)
    return {"reverse_complement": str(Seq(s).reverse_complement())}


def translate(sequence: str, strand: str = "+", frame: int = 0) -> dict:
    s = _clean(sequence)
    if strand not in ("+", "-"):
        raise ValueError("strand must be '+' or '-'")
    frame = int(frame)
    if frame not in (0, 1, 2):
        raise ValueError("frame must be 0, 1 or 2")
    if strand == "-":
        s = str(Seq(s).reverse_complement())
    s = s[frame:]
    s = s[: len(s) - len(s) % 3]
    return {"strand": strand, "frame": frame, "protein": str(Seq(s).translate())}


def _orfs(s: str, strand: str = "+") -> list[dict]:
    """ORFs: every ATG followed by an in-frame stop codon (nested ORFs counted
    separately). strand="+" scans the sequence as given (the LAB-Bench
    convention); strand="both" also scans the reverse complement.
    Positions are 1-based on the given strand; length_aa excludes the stop.
    """
    out = []
    n = len(s)
    strands = [("+", s)]
    if strand == "both":
        strands.append(("-", str(Seq(s).reverse_complement())))
    for st, seq in strands:
        for i in range(n - 2):
            if seq[i : i + 3] != "ATG":
                continue
            for j in range(i, n - 2, 3):
                if seq[j : j + 3] in STOPS:
                    prot = str(Seq(seq[i : j + 3]).translate())
                    if st == "+":
                        start, end = i + 1, j + 3
                    else:
                        start, end = n - (j + 3) + 1, n - i
                    out.append({"strand": st, "frame": i % 3, "start": start, "end": end,
                                "length_aa": len(prot) - 1, "protein": prot})
                    break
    # longest first; ties broken by reading frame, then position
    out.sort(key=lambda o: (-o["length_aa"], o["strand"], o["frame"], o["start"]))
    return out


def find_orfs(sequence: str, min_aa: int = 0, strand: str = "+") -> dict:
    s = _clean(sequence)
    if strand not in ("+", "both"):
        raise ValueError("strand must be '+' or 'both'")
    orfs = [o for o in _orfs(s, strand) if o["length_aa"] >= int(min_aa)]
    shown = orfs[:12]
    res = {"n_orfs": len(orfs),
           "definition": "ATG to first in-frame stop; nested ORFs counted separately; "
                         "length_aa excludes the stop codon; strand '+' = sequence as given",
           "orfs_longest_first": shown}
    if len(orfs) > len(shown):
        res["truncated"] = f"{len(orfs) - len(shown)} shorter ORFs not shown"
    return res


def digest(sequence: str, enzymes) -> dict:
    s = _clean(sequence)
    if isinstance(enzymes, str):
        enzymes = [e for e in re.split(r"[,\s]+", enzymes) if e]
    batch = []
    unknown = []
    for name in enzymes:
        enz = getattr(Restriction, name.strip(), None)
        if enz is None or not hasattr(enz, "search"):
            unknown.append(name)
        else:
            batch.append(enz)
    if unknown:
        return {"error": f"unknown enzyme(s): {', '.join(unknown)}"}
    seq = Seq(s)
    cuts = sorted({c for e in batch for c in e.search(seq, linear=True)})
    # Biopython reports the first base after the cut (1-based)
    bounds = [1] + cuts + [len(s) + 1]
    frags = [bounds[k + 1] - bounds[k] for k in range(len(bounds) - 1)]
    frags = [f for f in frags if f > 0]
    per_enzyme = {str(e): e.search(seq, linear=True) for e in batch}
    return {"linear": True, "cut_positions_by_enzyme": per_enzyme, "n_cuts": len(cuts),
            "n_fragments": len(frags), "fragment_lengths_in_order": frags}


def pcr(template: str, forward_primer: str, reverse_primer: str) -> dict:
    t = _clean(template)
    f = _clean(forward_primer)
    r = _clean(reverse_primer)
    rc_r = str(Seq(r).reverse_complement())
    rc_f = str(Seq(f).reverse_complement())
    products = []
    # forward on top strand, reverse primer binds bottom strand downstream
    for i in [m.start() for m in re.finditer(f"(?={f})", t)]:
        for j in [m.start() for m in re.finditer(f"(?={rc_r})", t)]:
            if j + len(rc_r) > i:
                products.append(j + len(rc_r) - i)
    # swapped orientation
    for i in [m.start() for m in re.finditer(f"(?={r})", t)]:
        for j in [m.start() for m in re.finditer(f"(?={rc_f})", t)]:
            if j + len(rc_f) > i:
                products.append(j + len(rc_f) - i)
    if not products:
        return {"product": False,
                "forward_binds": bool(re.search(f, t) or re.search(rc_f, t)),
                "reverse_binds": bool(re.search(r, t) or re.search(rc_r, t))}
    return {"product": True, "amplicon_lengths_bp": sorted(set(products))}


def find_sequence(sequence: str, query: str) -> dict:
    s = _clean(sequence)
    q = _clean(query)
    rc = str(Seq(q).reverse_complement())
    plus = [m.start() + 1 for m in re.finditer(f"(?={q})", s)]
    minus = [m.start() + 1 for m in re.finditer(f"(?={rc})", s)]
    return {"query_length": len(q), "matches_plus_strand": plus[:20],
            "matches_minus_strand": minus[:20]}


TOOLS = {
    "gc_content": gc_content,
    "reverse_complement": reverse_complement,
    "translate": translate,
    "find_orfs": find_orfs,
    "digest": digest,
    "pcr": pcr,
    "find_sequence": find_sequence,
}

TOOL_DOCS = {
    "gc_content": '{"tool": "gc_content", "args": {"sequence": "<seq or $ref>"}} -> GC percent and length',
    "reverse_complement": '{"tool": "reverse_complement", "args": {"sequence": "<seq or $ref>"}} -> reverse complement',
    "translate": '{"tool": "translate", "args": {"sequence": "<seq or $ref>", "strand": "+ or -", "frame": 0, 1 or 2}} -> amino-acid sequence (* = stop)',
    "find_orfs": '{"tool": "find_orfs", "args": {"sequence": "<seq or $ref>", "min_aa": <int>, "strand": "+ or both"}} -> all ORFs (ATG to in-frame stop, nested counted; default strand "+" = sequence as given), longest first, with protein sequences',
    "digest": '{"tool": "digest", "args": {"sequence": "<seq or $ref>", "enzymes": ["<name>", ...]}} -> fragment lengths of a linear digest',
    "pcr": '{"tool": "pcr", "args": {"template": "<seq or $ref>", "forward_primer": "<seq or $ref>", "reverse_primer": "<seq or $ref>"}} -> amplicon length(s) if both primers bind',
    "find_sequence": '{"tool": "find_sequence", "args": {"sequence": "<seq or $ref>", "query": "<seq or $ref>"}} -> exact match positions on both strands',
}
