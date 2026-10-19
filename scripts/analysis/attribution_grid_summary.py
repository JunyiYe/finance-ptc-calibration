#!/usr/bin/env python3
"""
Summarize the full 2x2 error-attribution grid (GPTQ/SparseGPT x FinQA/ConvFinQA):
category distributions for fixed vs control, one-sided Fisher test for
reasoning depletion among fixes, mechanical-category share, and the
BF16-correct fraction of fixed samples.

Reads the error_attribution_*.json files (current, post budget-fix data)
and the adjudication-consistent BF16 detailed results.

Output: experiments/results/analysis/error_attribution_grid.md + stdout.
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math import comb
from calibration_significance import load_detailed

MODEL = "Llama-3.1-8B-Instruct"
A = "experiments/results/analysis"
CATS = ["reasoning", "unit_scale", "arithmetic", "extraction", "format", "other"]
MECH = {"unit_scale", "arithmetic", "extraction", "format"}
SETTINGS = [("GPTQ", "gptq_w4a16", 5, "FinQA"),
            ("GPTQ", "gptq_w4a16", 9, "ConvFinQA"),
            ("SparseGPT", "sparsegpt_unstructured", 5, "FinQA"),
            ("SparseGPT", "sparsegpt_unstructured", 9, "ConvFinQA")]


def fisher_one_sided(a, b, c, d):
    """P(X >= a) for hypergeometric with margins; tests enrichment of
    top-left cell. For depletion pass the complement."""
    n = a + b + c + d
    row1, col1 = a + b, a + c
    lo = max(0, row1 + col1 - n)
    hi = min(row1, col1)
    denom = comb(n, col1)
    p = sum(comb(row1, x) * comb(n - row1, col1 - x) for x in range(a, hi + 1))
    return p / denom


def load(name, tnum, subset):
    suffix = "" if subset == "fixed" else "_unfixed"
    path = f"{A}/error_attribution_{name}_c1toc3{suffix}_{MODEL}_t{tnum}.json"
    d = json.load(open(path))
    ps = d["per_sample"]
    if isinstance(ps, list):
        return {s["id"]: s["category"] for s in ps if s.get("category")}
    return {k: v["category"] if isinstance(v, dict) else v
            for k, v in ps.items() if (v.get("category") if isinstance(v, dict) else v)}


def main():
    md = ["# Error-attribution 2x2 grid (post budget-fix data, "
          "adjudication-consistent)", ""]
    for fam, name, tnum, tname in SETTINGS:
        fixed = load(name, tnum, "fixed")
        ctrl = load(name, tnum, "unfixed")
        bf16 = load_detailed(MODEL, "bf16", tnum)
        nf, nc = len(fixed), len(ctrl)
        row = {c: sum(1 for v in fixed.values() if v == c) for c in CATS}
        rowc = {c: sum(1 for v in ctrl.values() if v == c) for c in CATS}
        # one-sided Fisher: reasoning depleted among fixed
        a = row["reasoning"]; b = nf - a
        c_ = rowc["reasoning"]; d = nc - c_
        # depletion of reasoning in fixed = enrichment of non-reasoning
        p_dep = fisher_one_sided(b, a, d, c_)
        mech_f = sum(row[c] for c in CATS if c in MECH) / nf * 100
        mech_c = sum(rowc[c] for c in CATS if c in MECH) / nc * 100
        hit = [i for i in fixed if i in bf16]
        bf_ok = sum(1 for i in hit if bool(bf16[i]["is_correct"]))
        md += [f"## {fam} x {tname}",
               f"- n fixed = {nf}, n control = {nc}",
               "- distribution (fixed | control, %): " + ", ".join(
                   f"{c} {row[c]/nf*100:.1f}|{rowc[c]/nc*100:.1f}"
                   for c in CATS),
               f"- reasoning fixed {row['reasoning']/nf*100:.1f}% vs control "
               f"{rowc['reasoning']/nc*100:.1f}% (one-sided Fisher, depletion: "
               f"p={p_dep:.3g})",
               f"- mechanical share: fixed {mech_f:.1f}% vs control {mech_c:.1f}%",
               f"- BF16-correct among fixed: {bf_ok}/{len(hit)} = "
               f"{bf_ok/len(hit)*100:.1f}%", ""]
    out = f"{A}/error_attribution_grid.md"
    with open(out, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", out)


if __name__ == "__main__":
    main()
