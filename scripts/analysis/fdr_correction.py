#!/usr/bin/env python3
"""
Benjamini-Hochberg FDR correction over the damage-significance CSVs
(each 6 configs x 10 tasks = 60 tests, corrected within each file).
Adds q_value and sig_fdr columns in place.
"""

import csv
import sys


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def bh(pvals):
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank_from_end, i in enumerate(reversed(order)):
        rank = n - rank_from_end
        prev = min(prev, pvals[i] * n / rank)
        q[i] = prev
    return q


def process(path):
    rows = list(csv.DictReader(open(path)))
    pcol = "p_value" if "p_value" in rows[0] else "p"
    dcol = "diff" if "diff" in rows[0] else "delta_pp"
    q = bh([float(r[pcol]) for r in rows])
    changed = []
    for r, qv in zip(rows, q):
        r["q_value"] = f"{qv:.3g}"
        r["sig_fdr"] = stars(qv)
        if r["sig"] and not r["sig_fdr"]:
            changed.append((r["method"], r["task"], r[dcol],
                            r[pcol], r["q_value"]))
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{path}: {len(rows)} tests")
    if changed:
        for m, t, d, p, qv in changed:
            print(f"  LOST star after FDR: {m} {t} diff={d} p={p} q={qv}")
    else:
        print("  no stars lost after FDR")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "experiments/results/analysis/damage_significance_Llama-3.1-8B-Instruct.csv",
        "experiments/results/analysis/damage_significance_Llama-3.1-8B-Instruct_c2.csv",
    ]
    for p in paths:
        process(p)
