#!/usr/bin/env python3
"""
Significance tests for calibration contrasts under the paper's metric policy.

Per paper_metrics.TASK_METRIC:
  - accuracy tasks (CFA, FinQA, ConvFinQA, TAT-QA): exact two-sided McNemar
    on per-sample paired correctness (adjudicated where available);
  - weighted-F1 tasks (FPB, FOMC, Headlines, FinRED, FinCausal, BigData22):
    paired bootstrap over samples (B=2000, seed 42) on the weighted-F1
    difference.

Usage:
    python scripts/analysis/calibration_significance.py \
        --model Llama-3.1-8B-Instruct --pairs c3:c1 c3:c2 c2:c1

Outputs (experiments/results/analysis/):
    calibration_significance_<model>.md / .csv
"""

import argparse
import csv
import json
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_metrics import TASK_METRIC, FINMIX_REDIRECT  # noqa: E402

RESULTS_ROOT = "experiments/results"
OUT_DIR = "experiments/results/analysis"

TASKS = ["t1", "t2", "t3", "t6", "t8", "t10", "t4", "t5", "t9", "t11"]
TASK_NAMES = {
    "t1": "FPB", "t2": "FOMC", "t3": "Headlines", "t4": "CFA",
    "t5": "FinQA", "t6": "FinRED", "t8": "FinCausal", "t9": "ConvFinQA",
    "t10": "BigData22", "t11": "TAT-QA",
}
METHODS = [
    ("gptq_w4a16", "GPTQ"),
    ("awq_w4a16_asym", "AWQ"),
    ("smoothquant_w8a8", "SmoothQuant"),
    ("sparsegpt_unstructured", "SparseGPT"),
    ("wanda_4of8", "Wanda"),
]
B = 2000
SEED = 42


def load_detailed(model, cfg, tnum):
    cfg = FINMIX_REDIRECT.get((model, cfg), cfg)
    base = os.path.join(RESULTS_ROOT, model)
    for suffix in (f"_t{tnum}_detailed_llm_adjudicated.json",
                   f"_t{tnum}_detailed.json"):
        path = os.path.join(base, f"{cfg}_{model}{suffix}")
        if os.path.exists(path):
            with open(path) as f:
                return {str(r["id"]): r for r in json.load(f)}
    return None


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** (n - 1)
    return min(1.0, p)


def f1_score(gold, pred, labels, average):
    """gold/pred: int arrays; labels: number of gold classes (invalid = -1)."""
    total = len(gold)
    scores, weights = [], []
    for lab in range(labels):
        g = gold == lab
        p = pred == lab
        tp = np.sum(g & p)
        support = np.sum(g)
        if support == 0:
            continue
        prec_den = np.sum(p)
        prec = tp / prec_den if prec_den else 0.0
        rec = tp / support
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        scores.append(f1)
        weights.append(support / total if average == "weighted_f1" else 1.0)
    return float(np.average(scores, weights=weights)) if scores else 0.0


def bootstrap_wf1(gold, pred_b, pred_t, labels, rng, average="weighted_f1"):
    """Paired bootstrap p-value and observed delta for f1(treat)-f1(base)."""
    n = len(gold)
    obs = f1_score(gold, pred_t, labels, average) - f1_score(gold, pred_b, labels, average)
    deltas = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        deltas[i] = (f1_score(gold[idx], pred_t[idx], labels, average)
                     - f1_score(gold[idx], pred_b[idx], labels, average))
    p_lo = (np.sum(deltas <= 0) + 1) / (B + 1)
    p_hi = (np.sum(deltas >= 0) + 1) / (B + 1)
    return obs, min(1.0, 2 * min(p_lo, p_hi))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama-3.1-8B-Instruct")
    ap.add_argument("--pairs", nargs="+", default=["c3:c1", "c3:c2", "c2:c1"])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_rows = []
    md = [f"# Calibration significance — {args.model}", "",
          "Metric policy: weighted-F1 (FinCausal: macro-F1) with paired bootstrap "
          "for classification tasks; accuracy with exact McNemar for QA tasks.", ""]

    for pair in args.pairs:
        treat, base = pair.split(":")
        md += [f"## {treat.upper()} vs {base.upper()}", ""]
        for mkey, mname in METHODS:
            header = f"| {mname} | " + " | ".join(TASK_NAMES[t] for t in TASKS) + " |"
            sep = "|---" * (len(TASKS) + 1) + "|"
            row_d, row_p = [f"| Δ ({treat}−{base}) "], ["| p "]
            any_data = False
            for t in TASKS:
                tnum = int(t[1:])
                a = load_detailed(args.model, f"{mkey}_{base}", tnum)
                bb = load_detailed(args.model, f"{mkey}_{treat}", tnum)
                if not a or not bb:
                    row_d.append("| — ")
                    row_p.append("| — ")
                    continue
                ref = load_detailed(args.model, "bf16", tnum)
                ids = set(a) & set(bb)
                if ref:
                    ids &= set(ref)
                ids = sorted(ids)
                n = len(ids)
                if TASK_METRIC[tnum] == "accuracy":
                    ca = np.array([bool(a[i]["is_correct"]) for i in ids])
                    cb = np.array([bool(bb[i]["is_correct"]) for i in ids])
                    delta = (cb.mean() - ca.mean()) * 100
                    win_t = int(np.sum(~ca & cb))
                    win_b = int(np.sum(ca & ~cb))
                    p = mcnemar_exact_p(win_b, win_t)
                    test = "mcnemar"
                else:
                    lab_list = sorted({str(a[i]["gold_label"]) for i in ids})
                    lab_idx = {l: k for k, l in enumerate(lab_list)}
                    gold = np.array([lab_idx[str(a[i]["gold_label"])] for i in ids])
                    def enc(rows_):
                        return np.array([lab_idx.get(str(rows_[i].get("predicted_label")), -1)
                                         for i in ids])
                    rng = np.random.default_rng(SEED)
                    obs, p = bootstrap_wf1(gold, enc(a), enc(bb), len(lab_list), rng,
                                           average=TASK_METRIC[tnum])
                    delta = obs * 100
                    test = f"bootstrap_{TASK_METRIC[tnum]}"
                any_data = True
                row_d.append(f"| {delta:+.1f}{stars(p)} ")
                row_p.append(f"| {p:.3g} ")
                csv_rows.append({
                    "model": args.model, "method": mkey, "task": t,
                    "treat": treat, "base": base, "n": n, "test": test,
                    "metric": TASK_METRIC[tnum],
                    "delta_pp": round(delta, 3), "p": p, "sig": stars(p),
                })
            if any_data:
                md += [header, sep, "".join(row_d) + "|", "".join(row_p) + "|", ""]
        md.append("")

    md.append("Significance: \\*\\*\\* p<0.001, \\*\\* p<0.01, \\* p<0.05.")
    with open(os.path.join(OUT_DIR, f"calibration_significance_{args.model}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT_DIR, f"calibration_significance_{args.model}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {OUT_DIR}/calibration_significance_{args.model}.{{md,csv}}")


if __name__ == "__main__":
    main()
