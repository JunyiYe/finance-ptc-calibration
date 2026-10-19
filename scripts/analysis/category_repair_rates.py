#!/usr/bin/env python3
"""
Per-category conditional repair rates among COMPRESSION-INTRODUCED errors:
restrict to samples where the WikiText-calibrated model is wrong AND BF16
is correct (the model had the capability; compression broke it), then for
each judge-attributed error category k compute, per (method, task) stratum
and pooled,

    P(FinMix repairs | Wiki wrong, BF16 correct, category k)
        = n_repaired(k) / (n_repaired(k) + n_unrepaired(k))

For each category: stratified CMH test of "repair rate in category k vs
all other categories" across the four strata, Mantel-Haenszel common OR
with Robins-Breslow-Greenland 95% CI, two-sided p, Bonferroni x6.

Requires the FULL unfixed attribution runs (every unrepaired error
attributed, not the 200-sample controls).

Secondary output: the same table without the BF16-correct restriction
(full wiki-wrong population), for the appendix/code release.

Output: experiments/results/analysis/category_repair_rates.csv + stdout.
"""

import csv
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math import erf, exp, log, sqrt
from calibration_significance import load_detailed

A = "experiments/results/analysis"
M = "Llama-3.1-8B-Instruct"
CATS = ["reasoning", "unit_scale", "arithmetic", "extraction", "format",
        "other"]
SET = [("gptq_w4a16", "t5", 5, "GPTQ", "FinQA"),
       ("gptq_w4a16", "t9", 9, "GPTQ", "ConvFinQA"),
       ("sparsegpt_unstructured", "t5", 5, "SparseGPT", "FinQA"),
       ("sparsegpt_unstructured", "t9", 9, "SparseGPT", "ConvFinQA")]


def load_cats(name, t, subset):
    suf = "" if subset == "fixed" else "_unfixed"
    ps = json.load(open(
        f"{A}/error_attribution_{name}_c1toc3{suf}_{M}_{t}.json"))["per_sample"]
    it = ps if isinstance(ps, list) else [
        {"id": k, **(v if isinstance(v, dict) else {"category": v})}
        for k, v in ps.items()]
    return {s["id"]: s["category"] for s in it if s.get("category")}


def cmh(tables):
    """tables: list of (a,b,c,d) = (rep_k, rep_other, unrep_k, unrep_other).
    Returns (OR_MH, lo, hi, p_two_sided) testing repair-rate difference of
    category k vs the rest. a/(a+c) vs b/(b+d) framing via standard CMH on
    the 2x2 (k vs other) x (repaired vs unrepaired)."""
    R = S = num = den = 0.0
    sPR = sM = sQS = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n == 0:
            continue
        R += a * d / n
        S += b * c / n
        P = (a + d) / n
        Q = (b + c) / n
        sPR += P * a * d / n
        sM += (P * b * c + Q * a * d) / n
        sQS += Q * b * c / n
        num += a - (a + b) * (a + c) / n
        den += (a + b) * (c + d) * (a + c) * (b + d) / (n**2 * (n - 1))
    if S == 0 or R == 0 or den == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    orr = R / S
    var = sPR / (2 * R * R) + sM / (2 * R * S) + sQS / (2 * S * S)
    lo, hi = exp(log(orr) - 1.96 * sqrt(var)), exp(log(orr) + 1.96 * sqrt(var))
    z = num / sqrt(den)
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return orr, lo, hi, p


def analyse(restrict_bf16):
    label = "BF16-correct subset" if restrict_bf16 else "all wiki-wrong"
    print(f"\n===== {label} =====")
    strata = []
    for name, t, tn, meth, tname in SET:
        bf = load_detailed(M, "bf16", tn)
        fc = load_cats(name, t, "fixed")
        uc = load_cats(name, t, "unfixed")
        if restrict_bf16:
            fc = {i: c for i, c in fc.items() if bool(bf[i]["is_correct"])}
            uc = {i: c for i, c in uc.items() if bool(bf[i]["is_correct"])}
        strata.append((meth, tname, fc, uc))
        print(f"{meth} x {tname}: repaired={len(fc)} unrepaired={len(uc)}")

    rows = []
    print(f"{'category':12s}" + "".join(
        f" {m[:3]}.{tn[:4]:>4s}" for m, tn, _, _ in strata)
        + "   pooled       OR [95% CI]      p       x6")
    for cat in CATS:
        cells = []
        tabs = []
        kf_t = ku_t = 0
        for meth, tname, fc, uc in strata:
            kf = sum(1 for c in fc.values() if c == cat)
            ku = sum(1 for c in uc.values() if c == cat)
            kf_t += kf
            ku_t += ku
            rate = kf / (kf + ku) if kf + ku else float("nan")
            cells.append(f"{rate*100:8.1f}" if kf + ku else "      --")
            rows.append([label, meth, tname, cat, kf, ku, kf + ku,
                         f"{rate:.4f}" if kf + ku else ""])
            tabs.append((kf, len(fc) - kf, ku, len(uc) - ku))
        orr, lo, hi, p = cmh(tabs)
        pooled = kf_t / (kf_t + ku_t) if kf_t + ku_t else float("nan")
        rows.append([label, "pooled", "all", cat, kf_t, ku_t, kf_t + ku_t,
                     f"{pooled:.4f}"])
        rows.append([label, "test", "all", cat, "", "", "",
                     f"OR={orr:.3f};CI=[{lo:.3f},{hi:.3f}];p={p:.3g};"
                     f"bonf6={min(1, p*6):.3g}"])
        print(f"{cat:12s}" + "".join(cells)
              + f" {pooled*100:8.1f}  {orr:5.2f} [{lo:4.2f},{hi:4.2f}]"
              f" {p:8.2g} {min(1, p*6):8.2g}")
    return rows


def main():
    rows = analyse(restrict_bf16=True) + analyse(restrict_bf16=False)
    out = f"{A}/category_repair_rates.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["population", "method", "task", "error_category",
                    "n_repaired", "n_unrepaired", "n_total",
                    "repair_rate_or_test"])
        w.writerows(rows)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
