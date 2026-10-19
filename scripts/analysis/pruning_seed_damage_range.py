#!/usr/bin/env python3
"""
Pruning damage seed stability (gap 5): BF16 minus pruned score under three
WikiText calibration draws (seed 42 = the main c1, plus c1s43 / c1s44),
for SparseGPT and Wanda on the six tasks the damage claims cite.

Damage values follow the display-consistency policy: subtraction of scores
rounded to one decimal. Missing seed runs show as blanks so the script can
run mid-pipeline.

Output: experiments/results/analysis/pruning_seed_damage_ranges.csv + stdout.
"""

import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_metrics import load_scores

MODEL = "Llama-3.1-8B-Instruct"
TASKS = [(3, "Headlines"), (8, "FinCausal"), (9, "ConvFinQA"),
         (5, "FinQA"), (4, "CFA"), (1, "FPB")]
METHODS = [("SparseGPT", "sparsegpt_unstructured"), ("Wanda", "wanda_4of8")]
SEEDS = [("s42", "c1"), ("s43", "c1s43"), ("s44", "c1s44")]
OUT = "experiments/results/analysis/pruning_seed_damage_ranges.csv"


def main():
    r = load_scores(MODEL)
    rows = []
    print(f"{'method':10s} {'task':10s} " +
          " ".join(f"{s:>7s}" for s, _ in SEEDS) + "   damage range")
    for mname, base in METHODS:
        for t, tname in TASKS:
            bf = round(r[("bf16", t)], 1)
            damages = {}
            for sname, suffix in SEEDS:
                v = r.get((f"{base}_{suffix}", t))
                if v is not None:
                    damages[sname] = round(bf - round(v, 1), 1)
            cells = " ".join(f"{damages[s]:7.1f}" if s in damages else
                             f"{'--':>7s}" for s, _ in SEEDS)
            rng = (f"{min(damages.values()):.1f}-{max(damages.values()):.1f}"
                   if damages else "--")
            print(f"{mname:10s} {tname:10s} {cells}   {rng}")
            rows.append([mname, tname, f"{bf:.1f}"]
                        + [f"{damages[s]:.1f}" if s in damages else ""
                           for s, _ in SEEDS]
                        + [f"{min(damages.values()):.1f}" if damages else "",
                           f"{max(damages.values()):.1f}" if damages else "",
                           len(damages)])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "task", "bf16",
                    "damage_s42", "damage_s43", "damage_s44",
                    "damage_min", "damage_max", "n_seeds"])
        w.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
