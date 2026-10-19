#!/usr/bin/env python3
"""
What does standard (WikiText) calibration buy over calibration-free RTN?

Per-task GPTQ-c1 minus RTN with the paper's metric policy and paired tests
(accuracy tasks: exact McNemar; F1 tasks: paired bootstrap), on the ids
shared with the canonical BF16 test sets.

Output: experiments/results/analysis/standard_calibration_dividend.{md,csv}
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_metrics import TASK_METRIC  # noqa: E402
from calibration_significance import (  # noqa: E402
    TASKS, TASK_NAMES, load_detailed, mcnemar_exact_p, bootstrap_wf1, stars)

MODEL = "Llama-3.1-8B-Instruct"
OUT = "experiments/results/analysis/standard_calibration_dividend"


def main():
    rng = np.random.default_rng(42)
    rows = []
    for t in TASKS:
        tnum = int(t[1:])
        rtn = load_detailed(MODEL, "rtn_w4a16", tnum)
        gptq = load_detailed(MODEL, "gptq_w4a16_c1", tnum)
        bf = load_detailed(MODEL, "bf16", tnum)
        if not rtn or not gptq or not bf:
            print(f"{t}: missing files, skipped")
            continue
        ids = sorted(set(rtn) & set(gptq) & set(bf))
        metric = TASK_METRIC[tnum]
        if metric == "accuracy":
            b = sum(1 for i in ids if rtn[i]["is_correct"] and not gptq[i]["is_correct"])
            c = sum(1 for i in ids if gptq[i]["is_correct"] and not rtn[i]["is_correct"])
            delta = (c - b) / len(ids) * 100
            p = mcnemar_exact_p(b, c)
            test = "mcnemar"
        else:
            labs = sorted({str(bf[i]["gold_label"]) for i in ids})
            li = {l: k for k, l in enumerate(labs)}
            gold = np.array([li[str(bf[i]["gold_label"])] for i in ids])
            pred_r = np.array([li.get(str(rtn[i].get("predicted_label")), -1) for i in ids])
            pred_g = np.array([li.get(str(gptq[i].get("predicted_label")), -1) for i in ids])
            obs, p = bootstrap_wf1(gold, pred_r, pred_g, len(labs), rng, average=metric)
            delta = obs * 100
            test = f"bootstrap_{metric}"
        rows.append({"task": t, "name": TASK_NAMES[t], "n": len(ids),
                     "metric": metric, "test": test,
                     "delta_pp": round(delta, 3), "p": p, "sig": stars(p)})
        print(f"{TASK_NAMES[t]:<10} GPTQ-c1 − RTN = {delta:+.1f}pp  p={p:.4g} {stars(p)}")

    with open(f"{OUT}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{OUT}.md", "w") as f:
        f.write(f"# Standard-calibration dividend (GPTQ-c1 − RTN), {MODEL}\n\n")
        f.write("| Task | n | Δ (pp) | p | |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['name']} | {r['n']} | {r['delta_pp']:+.1f} | "
                    f"{r['p']:.2g} | {r['sig']} |\n")
    print(f"\nwrote {OUT}.{{md,csv}}")


if __name__ == "__main__":
    main()
