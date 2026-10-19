#!/usr/bin/env python3
"""
Per-method versions of the section 5.2 analyses.

A. Ability recovery: 2x2 (BF16 status x repaired) over all WikiText-wrong
   samples, separately for GPTQ and SparseGPT; task-stratified CMH OR with
   Robins-Breslow-Greenland 95% CI and two-sided p; BF16-status x method
   interaction tested as z = (lnOR_g - lnOR_s)/sqrt(var_g + var_s).

B. Error-type repair rates restricted to WikiText-wrong AND BF16-correct,
   per method and pooled: per-category task-stratified CMH OR (category vs
   all others), raw and Bonferroni-x6 p; omnibus category-vs-repair
   chi-square (6x2, tasks pooled within method; the stratified per-category
   CMH is the primary inference); category x method interaction z-tests.

Outputs:
  experiments/results/analysis/ability_recovery_by_method.csv
  experiments/results/analysis/error_type_by_method.csv
"""

import csv
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math import erf, exp, log, sqrt
from scipy.stats import chi2_contingency
from calibration_significance import load_detailed

A = "experiments/results/analysis"
M = "Llama-3.1-8B-Instruct"
CATS = ["reasoning", "unit_scale", "arithmetic", "extraction", "format",
        "other"]
METHODS = {"GPTQ": ("gptq_w4a16", "_c3"),
           "SparseGPT": ("sparsegpt_unstructured", "_c3full")}
TASKS = [("t5", 5, "FinQA"), ("t9", 9, "ConvFinQA")]


def load_cats(name, t, subset):
    suf = "" if subset == "fixed" else "_unfixed"
    ps = json.load(open(
        f"{A}/error_attribution_{name}_c1toc3{suf}_{M}_{t}.json"))["per_sample"]
    it = ps if isinstance(ps, list) else [
        {"id": k, **(v if isinstance(v, dict) else {"category": v})}
        for k, v in ps.items()]
    return {s["id"]: s["category"] for s in it if s.get("category")}


def mh_or(tables):
    """tables: (a,b,c,d) rows with a,b = exposed/unexposed successes ...
    here a=repaired&E, b=repaired&notE, c=unrepaired&E, d=unrepaired&notE.
    Returns OR_MH, lo, hi, p, var(lnOR)."""
    R = S = num = den = 0.0
    sPR = sM = sQS = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
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
    if R == 0 or S == 0 or den == 0:
        return (float("nan"),) * 5
    orr = R / S
    var = sPR / (2 * R * R) + sM / (2 * R * S) + sQS / (2 * S * S)
    z = num / sqrt(den)
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return orr, exp(log(orr) - 1.96 * sqrt(var)), \
        exp(log(orr) + 1.96 * sqrt(var)), p, var


def zdiff(or1, v1, or2, v2):
    z = (log(or1) - log(or2)) / sqrt(v1 + v2)
    return z, 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def main():
    # ---------------- A: ability recovery per method -----------------------
    rows_a = []
    per_method = {}
    for mname, (base, c3s) in METHODS.items():
        tabs = []
        for tk, tn, tname in TASKS:
            bf = load_detailed(M, "bf16", tn)
            c1 = load_detailed(M, base + "_c1", tn)
            c3 = load_detailed(M, base + c3s, tn)
            ids = [i for i in set(bf) & set(c1) & set(c3)
                   if not c1[i]["is_correct"]]
            bcr = sum(1 for i in ids if bf[i]["is_correct"]
                      and c3[i]["is_correct"])
            bcu = sum(1 for i in ids if bf[i]["is_correct"]
                      and not c3[i]["is_correct"])
            bwr = sum(1 for i in ids if not bf[i]["is_correct"]
                      and c3[i]["is_correct"])
            bwu = sum(1 for i in ids if not bf[i]["is_correct"]
                      and not c3[i]["is_correct"])
            rows_a.append([mname, tname, bcr, bcu, bwr, bwu])
            # 2x2 for OR(BF16-correct -> repaired): a=repaired&Bok,
            # b=repaired&Bwrong, c=unrepaired&Bok, d=unrepaired&Bwrong
            tabs.append((bcr, bwr, bcu, bwu))
        orr, lo, hi, p, var = mh_or(tabs)
        per_method[mname] = (orr, var)
        rows_a.append([mname, "CMH", f"OR={orr:.3f}", f"CI=[{lo:.3f},{hi:.3f}]",
                       f"p={p:.3g}", ""])
        print(f"A {mname}: CMH OR={orr:.2f} [{lo:.2f},{hi:.2f}] p={p:.3g}")
    zg, pg = zdiff(*per_method["GPTQ"], *per_method["SparseGPT"])
    rows_a.append(["interaction", "GPTQ_vs_SparseGPT",
                   f"z={zg:.2f}", f"p={pg:.3g}",
                   "log-OR difference, RBG variances", ""])
    print(f"A interaction GPTQ vs SparseGPT: z={zg:.2f} p={pg:.3g}")
    with open(f"{A}/ability_recovery_by_method.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "task", "bf16_correct_repaired",
                    "bf16_correct_unrepaired", "bf16_wrong_repaired",
                    "bf16_wrong_unrepaired"])
        w.writerows(rows_a)

    # ---------------- B: error types within BF16-correct -------------------
    rows_b = []
    cat_or = {}
    for mname, (base, c3s) in METHODS.items():
        strata = []
        for tk, tn, tname in TASKS:
            bf = load_detailed(M, "bf16", tn)
            c1 = load_detailed(M, base + "_c1", tn)
            c3 = load_detailed(M, base + c3s, tn)
            sub_f = {i for i in set(bf) & set(c1) & set(c3)
                     if not c1[i]["is_correct"] and bf[i]["is_correct"]
                     and c3[i]["is_correct"]}
            sub_u = {i for i in set(bf) & set(c1) & set(c3)
                     if not c1[i]["is_correct"] and bf[i]["is_correct"]
                     and not c3[i]["is_correct"]}
            fc = load_cats(base, tk, "fixed")
            uc = load_cats(base, tk, "unfixed")
            f_res = {i: fc[i] for i in sub_f if i in fc}
            u_res = {i: uc[i] for i in sub_u if i in uc}
            unres = (len(sub_f) - len(f_res)) + (len(sub_u) - len(u_res))
            strata.append((tname, f_res, u_res, unres))
        # per-category rows and per-method stratified tests
        pooled_f = {}
        pooled_u = {}
        for tname, f_res, u_res, unres in strata:
            for cat in CATS:
                kf = sum(1 for v in f_res.values() if v == cat)
                ku = sum(1 for v in u_res.values() if v == cat)
                rate = kf / (kf + ku) if kf + ku else float("nan")
                rows_b.append([mname, tname, cat, kf, ku, kf + ku,
                               f"{rate:.4f}" if kf + ku else "", unres])
            pooled_f[tname] = f_res
            pooled_u[tname] = u_res
        # omnibus 6x2 chi-square, tasks pooled within method
        table = [[sum(sum(1 for v in pooled_f[t].values() if v == cat)
                      for t in pooled_f),
                  sum(sum(1 for v in pooled_u[t].values() if v == cat)
                      for t in pooled_u)] for cat in CATS]
        table = [r for r in table if sum(r) > 0]
        chi2, pom, dof, _ = chi2_contingency(table)
        rows_b.append([mname, "omnibus", "all", "", "", "",
                       f"chi2={chi2:.2f};dof={dof};p={pom:.3g}", ""])
        print(f"B {mname} omnibus: chi2={chi2:.2f} dof={dof} p={pom:.3g}")
        for cat in CATS:
            tabs = []
            for tname, f_res, u_res, _un in strata:
                a = sum(1 for v in f_res.values() if v == cat)
                b = len(f_res) - a
                c = sum(1 for v in u_res.values() if v == cat)
                d = len(u_res) - c
                tabs.append((a, b, c, d))
            orr, lo, hi, p, var = mh_or(tabs)
            cat_or.setdefault(cat, {})[mname] = (orr, var)
            rows_b.append([mname, "test", cat, "", "", "",
                           f"OR={orr:.3f};CI=[{lo:.3f},{hi:.3f}];p={p:.3g};"
                           f"bonf6={min(1, p*6) if p == p else float('nan'):.3g}",
                           ""])
    # category x method interaction
    for cat in CATS:
        d = cat_or.get(cat, {})
        if len(d) == 2 and all(v[0] == v[0] for v in d.values()):
            z, p = zdiff(*d["GPTQ"], *d["SparseGPT"])
            rows_b.append(["interaction", "GPTQ_vs_SparseGPT", cat, "", "",
                           "", f"z={z:.2f};p={p:.3g};bonf6={min(1,p*6):.3g}",
                           ""])
            print(f"B interaction {cat}: z={z:.2f} p={p:.3g}")
    with open(f"{A}/error_type_by_method.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "task", "category", "n_repaired",
                    "n_unrepaired", "n_total", "repair_rate_or_test",
                    "unresolved_count"])
        w.writerows(rows_b)
    print("wrote ability_recovery_by_method.csv, error_type_by_method.csv")


if __name__ == "__main__":
    main()
