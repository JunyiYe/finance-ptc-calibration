#!/usr/bin/env python3
"""
Damage significance: each WikiText-calibrated configuration (and RTN) vs BF16,
per task, using the same machinery as calibration_significance.py —
exact two-sided McNemar for accuracy tasks, paired bootstrap (B=2000) for
F1 tasks, ids restricted to the canonical intersection with BF16.

Outputs:
  experiments/results/analysis/damage_significance_<model>.csv
  experiments/results/analysis/damage_significance_<model>.md
"""

import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from calibration_significance import (load_detailed, mcnemar_exact_p,
                                      f1_score, bootstrap_wf1, stars,
                                      TASKS, TASK_NAMES)
from paper_metrics import TASK_METRIC

MODEL = "Llama-3.1-8B-Instruct"

ap = argparse.ArgumentParser()
ap.add_argument("--calib", default="c1", choices=["c1", "c2"],
                help="calibration suffix; c2 = the C4-calibrated replication")
CALIB = ap.parse_args().calib

CONFIGS = [
    ("RTN", "rtn_w4a16"),
    ("GPTQ", f"gptq_w4a16_{CALIB}"),
    ("AWQ", f"awq_w4a16_asym_{CALIB}"),
    ("SmoothQuant", f"smoothquant_w8a8_{CALIB}"),
    ("SparseGPT", f"sparsegpt_unstructured_{CALIB}"),
    ("Wanda", f"wanda_4of8_{CALIB}"),
]
OUT = ("experiments/results/analysis/damage_significance_" + MODEL
       + ("" if CALIB == "c1" else "_" + CALIB))

LABELS = {  # label sets for F1 tasks, mirrored from calibration_significance usage
}


def main():
    rows_csv = []
    md = [f"# Damage significance vs BF16 — {MODEL}", "",
          "Exact two-sided McNemar (accuracy tasks); paired bootstrap "
          "B=2000 (F1 tasks); ids = canonical intersection with BF16.", ""]
    header = "| Config | " + " | ".join(TASK_NAMES[t] for t in TASKS) + " |"
    sep = "|---" * (len(TASKS) + 1) + "|"
    md += [header, sep]
    rng = np.random.default_rng(42)
    for cname, cfg in CONFIGS:
        cells = []
        for t in TASKS:
            tnum = int(t[1:])
            ref = load_detailed(MODEL, "bf16", tnum)
            cmp_ = load_detailed(MODEL, cfg, tnum)
            if not ref or not cmp_:
                cells.append("—")
                continue
            ids = sorted(set(ref) & set(cmp_))
            n = len(ids)
            if TASK_METRIC[tnum] == "accuracy":
                a = np.array([bool(ref[i]["is_correct"]) for i in ids])
                b = np.array([bool(cmp_[i]["is_correct"]) for i in ids])
                delta = (b.mean() - a.mean()) * 100
                p = mcnemar_exact_p(int(np.sum(a & ~b)), int(np.sum(~a & b)))
                test = "mcnemar"
                s_ref, s_cmp = a.mean() * 100, b.mean() * 100
            else:
                lab_list = sorted({str(ref[i]["gold_label"]) for i in ids})
                lab_idx = {l: k for k, l in enumerate(lab_list)}
                gold = np.array([lab_idx[str(ref[i]["gold_label"])] for i in ids])
                pa = np.array([lab_idx.get(str(ref[i].get("predicted_label")), -1)
                               for i in ids])
                pb = np.array([lab_idx.get(str(cmp_[i].get("predicted_label")), -1)
                               for i in ids])
                labels = len(lab_list)
                avg = "macro_f1" if TASK_METRIC[tnum] == "macro_f1" else "weighted_f1"
                s_ref = f1_score(gold, pa, labels, avg) * 100
                s_cmp = f1_score(gold, pb, labels, avg) * 100
                delta = s_cmp - s_ref
                _obs, p = bootstrap_wf1(gold, pa, pb, labels, rng, average=avg)
                test = f"bootstrap_{avg}"
            sig = stars(p)
            cells.append(f"{delta:+.1f}{sig}")
            rows_csv.append([MODEL, cname, t, TASK_METRIC[tnum],
                             f"{s_ref:.2f}", f"{s_cmp:.2f}", f"{delta:.2f}",
                             test, f"{p:.3g}", sig])
        md.append(f"| {cname} | " + " | ".join(cells) + " |")
    md += ["", "Significance: \\*\\*\\* p<0.001, \\*\\* p<0.01, \\* p<0.05."]
    with open(OUT + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "task", "metric", "bf16_score",
                    "method_score", "diff", "test_type", "p_value", "sig"])
        w.writerows(rows_csv)
    with open(OUT + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {OUT}.{{csv,md}}")


if __name__ == "__main__":
    main()
