#!/usr/bin/env python3
"""
Test-set consistency overrides.

Some June/July runs evaluated superset versions of the test files
(Headlines 20547, FinRED 8928, FinCausal 8628, TAT-QA 1668) while the
canonical protocol uses the deduplicated sets defined by the bf16 runs.
A full audit (2026-07-26) confirmed every such case is a strict SUPERSET
with byte-identical prompts, so restricting to the canonical id set is
mathematically equivalent to a re-run.

This script auto-detects all superset runs for every model, recomputes
their metrics on the canonical (bf16) id set, and writes overrides
consumed by paper_metrics.load_scores. Judged tasks (FinQA, ConvFinQA,
TAT-QA) use adjudicated per-sample correctness.

Output: experiments/results/analysis/testset_overrides.json
"""

import glob
import json
import os
import re

import numpy as np

OUT = "experiments/results/analysis/testset_overrides.json"
MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-8B", "Llama-3.1-70B-Instruct"]
JUDGED = {5, 9, 11}


def cls_metrics(rows):
    labs = sorted({str(r["gold_label"]) for r in rows})
    li = {l: k for k, l in enumerate(labs)}
    g = np.array([li[str(r["gold_label"])] for r in rows])
    p = np.array([li.get(str(r.get("predicted_label")), -1) for r in rows])
    acc = float(np.mean(g == p))
    f1s, supports = [], []
    for k in range(len(labs)):
        tp = np.sum((g == k) & (p == k))
        prec = tp / max(np.sum(p == k), 1)
        rec = tp / np.sum(g == k)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        supports.append(np.sum(g == k))
    supports = np.array(supports, dtype=float)
    return {"accuracy": acc,
            "macro_f1": float(np.mean(f1s)),
            "weighted_f1": float(np.average(f1s, weights=supports))}


def main():
    overrides = {}
    for model in MODELS:
        base = f"experiments/results/{model}"
        # canonical ids come from Llama-8B's bf16 runs (the dedup-1500 sets);
        # Qwen's own bf16 ran the full sets, so cross-model canon is required
        canon_model = "Llama-3.1-8B-Instruct"
        cbase = f"experiments/results/{canon_model}"
        canon = {}
        for t in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]:
            f = f"{cbase}/bf16_{canon_model}_t{t}_detailed.json"
            if os.path.exists(f):
                canon[t] = {str(r["id"]) for r in json.load(open(f))}
        restrict_bf16 = model != canon_model
        for f in sorted(glob.glob(f"{base}/*_detailed*.json")):
            b = os.path.basename(f)
            if b.startswith("bf16") and not restrict_bf16:
                continue
            m = re.search(r"_t(\d+)_detailed", b)
            if not m:
                continue
            t = int(m.group(1))
            if t not in canon:
                continue
            is_adj = "llm_adjudicated" in b
            if (t in JUDGED) != is_adj:
                continue  # judged tasks: use adjudicated file only
            cfg = b.split(f"_{model}")[0]
            rows = json.load(open(f))
            ids = {str(r["id"]) for r in rows}
            if not (canon[t] < ids):
                continue  # not a superset run
            sub = [r for r in rows if str(r["id"]) in canon[t]]
            if len(sub) != len(canon[t]):
                print(f"WARN {model} {cfg} t{t}: {len(sub)} != {len(canon[t])}")
            if t in JUDGED:
                mets = {"accuracy": float(np.mean([bool(r["is_correct"]) for r in sub]))}
            else:
                mets = cls_metrics(sub)
            overrides[f"{model}|{cfg}|{t}"] = mets
            key = "accuracy" if t in JUDGED else "all"
            print(f"{model} {cfg} t{t}: {len(rows)} -> {len(sub)} ({key}: "
                  + ", ".join(f"{k}={v*100:.1f}" for k, v in mets.items()) + ")")
    with open(OUT, "w") as f:
        json.dump(overrides, f, indent=2)
    print(f"\nwrote {OUT} ({len(overrides)} overrides)")


if __name__ == "__main__":
    main()
