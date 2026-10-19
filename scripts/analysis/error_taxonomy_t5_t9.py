#!/usr/bin/env python3
"""
Rule-based error taxonomy for numerical QA tasks (T5 FinQA, T9 ConvFinQA).

Classifies every wrong prediction into:
    no_answer    — no parseable numeric answer extracted
    truncated    — generation hit max tokens before answering
    sign_error   — |pred| == |gold| but opposite sign
    scale_error  — pred == gold x 10^k (k != 0), catches %-vs-decimal and
                   million/billion unit confusions
    near_miss    — within 5% relative error of gold (wrong rounding/precision)
    other_wrong  — parseable number, none of the above (wrong operand or
                   wrong arithmetic)

Then, for each (method, task), reports the c1→c3 transition matrix:
which error types the domain calibration set fixes, and whether the fixed
samples are ones the BF16 baseline also gets right (capability recovery vs
new capability).

Usage:
    python scripts/analysis/error_taxonomy_t5_t9.py --model Llama-3.1-8B-Instruct

Output:
    experiments/results/analysis/error_taxonomy_<model>.md
    experiments/results/analysis/error_taxonomy_<model>.json
"""

import argparse
import json
import os
import re

RESULTS_ROOT = "experiments/results"
OUT_DIR = "experiments/results/analysis"

TASKS = {"t5": "FinQA", "t9": "ConvFinQA"}
METHODS = [
    ("gptq_w4a16", "GPTQ W4A16"),
    ("awq_w4a16_asym", "AWQ W4A16"),
    ("sparsegpt_unstructured", "SparseGPT 50%"),
    ("wanda_4of8", "Wanda 4:8"),
]
CATEGORIES = ["no_answer", "truncated", "sign_error", "scale_error",
              "near_miss", "other_wrong"]

_NUM_RE = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?%?")


def parse_num(s):
    """Parse a numeric answer string; returns float or None."""
    if s is None:
        return None
    s = str(s).strip()
    m = _NUM_RE.search(s)
    if not m:
        return None
    tok = m.group(0)
    neg = tok.startswith("(") or tok.startswith("-") or tok.startswith("-$")
    tok = tok.strip("()%$").replace(",", "").replace("$", "").lstrip("-")
    try:
        v = float(tok)
    except ValueError:
        return None
    return -v if neg else v


def classify(row):
    """Classify one wrong prediction row into an error category."""
    gold = parse_num(row.get("gold_label"))
    pred = parse_num(row.get("predicted_label"))
    if pred is None:
        if row.get("hit_max_tokens"):
            return "truncated"
        return "no_answer"
    if gold is None:
        return "other_wrong"
    if row.get("hit_max_tokens") and pred is None:
        return "truncated"
    ag, ap = abs(gold), abs(pred)
    if ag > 0 and ap > 0:
        # sign error: same magnitude, opposite sign
        if abs(ap - ag) / ag < 0.01 and (gold > 0) != (pred > 0):
            return "sign_error"
        # scale error: off by a power of ten (covers % <-> decimal)
        for k in (1, 2, 3, -1, -2, -3):
            if abs(ap - ag * 10 ** k) / (ag * 10 ** k) < 0.01:
                return "scale_error"
        # near miss: right ballpark, wrong precision/rounding
        if abs(ap - ag) / ag < 0.05:
            return "near_miss"
    return "other_wrong"


def load(model, cfg, tnum):
    base = os.path.join(RESULTS_ROOT, model)
    for suffix in (f"_t{tnum}_detailed_llm_adjudicated.json",
                   f"_t{tnum}_detailed.json"):
        p = os.path.join(base, f"{cfg}_{model}{suffix}")
        if os.path.exists(p):
            with open(p) as f:
                return {str(r["id"]): r for r in json.load(f)}
    return None


def dist_line(counts, total):
    return " | ".join(f"{counts.get(c, 0) / total * 100:.1f}" if total else "—"
                      for c in CATEGORIES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama-3.1-8B-Instruct")
    ap.add_argument("--base", default="c1")
    ap.add_argument("--treat", default="c3")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    md = [f"# Error taxonomy (rule-based) — {args.model}", ""]
    md.append(f"Error distribution over WRONG predictions (% of errors), plus the "
              f"{args.base}→{args.treat} discordant transition analysis.")
    blob = {}

    for tnum, tname in ((k[1:], v) for k, v in TASKS.items()):
        md += ["", f"## T{tnum} {tname}", ""]
        bf16 = load(args.model, "bf16", tnum) or {}
        md.append("| Config | n_err | " + " | ".join(CATEGORIES) + " |")
        md.append("|---" * (len(CATEGORIES) + 2) + "|")

        per_cfg = {}
        for mkey, mname in METHODS:
            for calib in (args.base, args.treat):
                rows = load(args.model, f"{mkey}_{calib}", tnum)
                if not rows:
                    continue
                errs = {i: classify(r) for i, r in rows.items()
                        if not r["is_correct"]}
                per_cfg[(mkey, calib)] = (rows, errs)
                counts = {}
                for c in errs.values():
                    counts[c] = counts.get(c, 0) + 1
                md.append(f"| {mname} {calib} | {len(errs)} | "
                          f"{dist_line(counts, len(errs))} |")
                blob[f"t{tnum}/{mkey}_{calib}"] = {
                    "n_err": len(errs),
                    "counts": counts,
                }

        # discordant transition: base wrong -> treat right
        md += ["", f"### Which {args.base}-errors does {args.treat} fix?", ""]
        md.append(f"| Method | fixed n | of which bf16-correct | "
                  + " | ".join(CATEGORIES) + " |")
        md.append("|---" * (len(CATEGORIES) + 3) + "|")
        for mkey, mname in METHODS:
            if (mkey, args.base) not in per_cfg or (mkey, args.treat) not in per_cfg:
                continue
            rows_b, errs_b = per_cfg[(mkey, args.base)]
            rows_t, _ = per_cfg[(mkey, args.treat)]
            fixed = [i for i in errs_b
                     if i in rows_t and rows_t[i]["is_correct"]]
            counts = {}
            for i in fixed:
                counts[errs_b[i]] = counts.get(errs_b[i], 0) + 1
            n_bf16_ok = sum(1 for i in fixed
                            if i in bf16 and bf16[i]["is_correct"])
            pct_bf16 = f"{n_bf16_ok / len(fixed) * 100:.0f}%" if fixed else "—"
            md.append(f"| {mname} | {len(fixed)} | {pct_bf16} | "
                      f"{dist_line(counts, len(fixed))} |")
            blob[f"t{tnum}/{mkey}_fixed"] = {
                "n_fixed": len(fixed), "bf16_correct": n_bf16_ok,
                "counts": counts,
            }

    md_path = os.path.join(OUT_DIR, f"error_taxonomy_{args.model}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT_DIR, f"error_taxonomy_{args.model}.json"), "w") as f:
        json.dump(blob, f, indent=2)
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
