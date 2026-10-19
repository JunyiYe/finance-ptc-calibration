#!/usr/bin/env python3
"""Paper consistency sweep: verify every number the paper cites against the
generated data (34 assertions: structure, table cells, attribution sums,
repair deltas, RQ3 statistics, C4 replication, PPL ordering).

Run from the repo root after ANY result change or table regeneration:
    python3 scripts/analysis/paper_consistency_check.py
Exits with ALL PASS or lists the failing assertions."""
import csv, json, os, re, subprocess, sys
sys.path.insert(0, "scripts/analysis")
from paper_metrics import load_scores, FINMIX_REDIRECT
from math import sqrt

FAIL = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)

tex = {}
for f in os.listdir("paper/sections"):
    tex[f] = open(f"paper/sections/{f}").read()
alltex = "\n".join(tex.values())
tables = {f: open(f"paper/tables/{f}").read() for f in os.listdir("paper/tables")}

# ---- 1. structural checks -------------------------------------------------
check("no em-dashes in sections", "---" not in alltex)
refs = set(re.findall(r"\\ref\{([a-z:0-9]+)\}", alltex + "".join(tables.values())))
labels = set(re.findall(r"\\label\{([a-z:0-9]+)\}",
             alltex + "".join(tables.values()) + open("paper/main.tex").read()))
check("all refs resolve", refs <= labels, f"dangling: {refs - labels}")

# ---- 2. attribution table column sums ------------------------------------
att = tables["attribution_table.tex"]
rows = re.findall(r"^(reasoning|unit/scale|arithmetic|extraction|format|other)\s*&([^\\]+)\\\\",
                  att, re.M)
cols = [[], [], [], []]
for _n, vals in rows:
    for i, v in enumerate(vals.split("&")):
        cols[i].append(float(v))
for i, c in enumerate(cols):
    check(f"attribution col{i} sums to ~100", abs(sum(c) - 100) < 0.6, f"{sum(c):.1f}")

# ---- 3. repair-table deltas equal rounded-subtraction --------------------
r8 = load_scores("Llama-3.1-8B-Instruct")
rep = tables["repair_table.tex"]
pairs = {"GPTQ": ("gptq_w4a16_c3", "gptq_w4a16_c1"),
         "AWQ": ("awq_w4a16_asym_c3", "awq_w4a16_asym_c1"),
         "SmoothQuant": ("smoothquant_w8a8_c3", "smoothquant_w8a8_c1"),
         "SparseGPT": ("sparsegpt_unstructured_c3full", "sparsegpt_unstructured_c1"),
         "Wanda": ("wanda_4of8_c3full", "wanda_4of8_c1")}
TASKS = [1, 2, 3, 6, 8, 10, 4, 5, 9, 11]
for meth, (c3, c1) in pairs.items():
    m = re.search(rf"^{meth} &(.+)\\\\", rep, re.M)
    cells = re.findall(r"([\d.]+) \(\$([+-])([\d.]+)", m.group(1))
    ok = True
    for (v, sign, d), t in zip(cells, TASKS):
        want = round(round(r8[(c3, t)], 1) - round(r8[(c1, t)], 1), 1)
        got = float(d) * (1 if sign == "+" else -1)
        if abs(want - got) > 0.051 or abs(float(v) - round(r8[(c3, t)], 1)) > 0.051:
            ok = False
    check(f"repair-table {meth} cells consistent", ok)

# ---- 4. headline statistics ----------------------------------------------
def collect(model, meths):
    r = {k: v for k, v in load_scores(model).items()}
    pts = []
    for meth in meths:
        for t in TASKS:
            c3 = FINMIX_REDIRECT.get((model, f"{meth}_c3"), f"{meth}_c3")
            bf, c1, c3v = r.get(("bf16", t)), r.get((f"{meth}_c1", t)), r.get((c3, t))
            if None in (bf, c1, c3v):
                continue
            pts.append((bf - c1, c3v - c1, meth, t))
    return pts
L = collect("Llama-3.1-8B-Instruct", ["gptq_w4a16", "awq_w4a16_asym",
    "sparsegpt_unstructured", "wanda_4of8", "smoothquant_w8a8"])
Q = collect("Qwen3-8B", ["gptq_w4a16", "awq_w4a16_asym",
    "sparsegpt_unstructured", "wanda_4of8", "smoothquant_w8a8"])
def spear(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        for i, idx in enumerate(o): r[idx] = i
        return r
    rx, ry = rank(xs), rank(ys); n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
# BigData22 (t10) excluded from the RQ3 analysis set (chance-level task)
allp = [p for p in L + Q if p[3] != 10]
Lx = [p for p in L if p[3] != 10]
Qx = [p for p in Q if p[3] != 10]
check("n = 90 triples", len(allp) == 90, str(len(allp)))
check("rho 0.74 cited as primary", abs(spear(allp) - 0.74) < 0.005
      and "0.74" in alltex, f"{spear(allp):.3f}")
check("rho 0.68 disclosed (all-task value)",
      abs(spear(L + Q) - 0.68) < 0.005 and "0.68" in alltex,
      f"{spear(L + Q):.3f}")
check("no 'predicted before' claim remains",
      "before running them" not in alltex
      and "before those experiments were run" not in alltex
      and "before the run" not in alltex
      and "announced before" not in alltex)
fx = [p[0] for p in Lx]; fy = [p[1] for p in Lx]
n = len(fx); mx = sum(fx) / n; my = sum(fy) / n
sxx = sum((a - mx) ** 2 for a in fx)
slope = sum((a - mx) * (b - my) for a, b in zip(fx, fy)) / sxx
check("fit slope 0.38 in figure", abs(slope - 0.38) < 0.005, f"{slope:.3f}")
b0 = my - slope * mx
sd = sqrt(sum((y - (slope * x + b0)) ** 2 for x, y in zip(fx, fy)) / (n - 2))
TCRIT = 2.017  # df = 43
out = sum(1 for x, y, _m, _t in Qx
          if abs(y - (slope * x + b0)) > TCRIT * sd * sqrt(1 + 1 / n + (x - mx) ** 2 / sxx))
check("45/45 Qwen in band", out == 0 and "forty-five" in alltex, f"outside={out}")
outL = sum(1 for x, y, _m, _t in Lx
           if abs(y - (slope * x + b0)) > TCRIT * sd * sqrt(1 + 1 / n + (x - mx) ** 2 / sxx))
check("3 Llama points outside band (caption)", outL == 3 and "three" in alltex,
      f"outsideL={outL}")

# ---- 5. specific cited values against significance CSVs -------------------
def sig(model):
    out = {}
    with open(f"experiments/results/analysis/calibration_significance_{model}.csv") as f:
        for row in csv.DictReader(f):
            out[(row["method"], row["task"], row["treat"], row["base"])] = (
                float(row["delta_pp"]), row["sig"])
    return out
s8 = sig("Llama-3.1-8B-Instruct")
checks = [
    ("GPTQ FinQA +5.9***", ("gptq_w4a16", "t5", "c3", "c1"), 5.9, "***"),
    ("Wanda FOMC -10.2***", ("wanda_4of8", "t2", "c3", "c1"), -10.2, "***"),
    ("SparseGPT Headlines +38.4***", ("sparsegpt_unstructured", "t3", "c3", "c1"), 38.4, "***"),
]
for name, key, dexp, sexp in checks:
    d, sg = s8[key]
    check(name, abs(d - dexp) < 0.06 and sg == sexp, f"{d:.1f}{sg}")
sp = [s8[("sparsegpt_unstructured", f"t{t}", "c3", "c1")] for t in TASKS]
check("SparseGPT 9/10 significant", sum(1 for d, g in sp if g) == 9)
wa = [s8[("wanda_4of8", f"t{t}", "c3", "c1")] for t in TASKS]
check("Wanda 7 gains 1 loss", sum(1 for d, g in wa if g and d > 0) == 7
      and sum(1 for d, g in wa if g and d < 0) == 1)

# ---- 5. C4 replication claims (section 4.2/4.3) --------------------------
cmp_rows = list(csv.DictReader(
    open("experiments/results/analysis/c1_vs_c2_damage_comparison.csv")))
check("c1-vs-c2 comparison has 50 cells", len(cmp_rows) == 50)
absd = [abs(float(r["c1_minus_c2"])) for r in cmp_rows]
check("mean abs c1-c2 diff ~2 points", 1.5 <= sum(absd) / len(absd) <= 2.5,
      f"{sum(absd)/len(absd):.2f}")
d2 = {(r["method"], r["task"]): float(r["diff"])
      for r in csv.DictReader(
          open("experiments/results/analysis/"
               "damage_significance_Llama-3.1-8B-Instruct_c2.csv"))}
qworst = min(v for (m, t), v in d2.items()
             if m in ("RTN", "GPTQ", "AWQ", "SmoothQuant"))
pworst = min(v for (m, t), v in d2.items() if m in ("SparseGPT", "Wanda"))
check("c2 families separated ~30 points", 25 <= qworst - pworst <= 40,
      f"quant {qworst:.1f} prune {pworst:.1f}")
check("c2 GPTQ spares FinRED hits Headlines",
      d2[("GPTQ", "t6")] > 0 and d2[("GPTQ", "t3")] < -3)
prune_flips = [r for r in cmp_rows if r["same_direction"] == "NO"
               and r["method"] in ("SparseGPT", "Wanda")]
check("pruning sign flips only BigData22",
      {r["task"] for r in prune_flips} == {"BigData22"})
ppl8 = json.load(open("experiments/results/analysis/"
                      "perplexity_Llama-3.1-8B-Instruct.json"))
gptq_ppl = [ppl8["gptq_w4a16_c1"], ppl8["gptq_w4a16_c2"], ppl8["gptq_w4a16_c3"]]
gptq_fq = [r8[("gptq_w4a16_c1", 5)], r8[("gptq_w4a16_c2", 5)],
           r8[("gptq_w4a16_c3", 5)]]
check("GPTQ 3-corpus PPL order opposite to FinQA order",
      gptq_ppl == sorted(gptq_ppl) and gptq_fq == sorted(gptq_fq))
check("web-vs-finmix inversion all damaged methods", all(
    ppl8[a] < ppl8[b] and r8[(a, 5)] < r8[(b, 5)]
    for a, b in [("gptq_w4a16_c2", "gptq_w4a16_c3"),
                 ("sparsegpt_unstructured_c2", "sparsegpt_unstructured_c3full"),
                 ("wanda_4of8_c2", "wanda_4of8_c3full")]))
check("tab:main_c4 exists with stars",
      "main_table_c4.tex" in tables and "$^{***}$" in tables["main_table_c4.tex"])

# ---- 6. tab:gen: every numeric cell equals the rounded source score ----
GEN_TASKS = [1, 2, 3, 6, 8, 10, 4, 5, 9, 11]
GEN_ROWS = {"BF16": "bf16", "RTN": "rtn_w4a16", "GPTQ": "gptq_w4a16",
            "AWQ": "awq_w4a16_asym", "SmoothQuant": "smoothquant_w8a8",
            "SparseGPT": "sparsegpt_unstructured", "Wanda": "wanda_4of8"}
from paper_metrics import FINMIX_REDIRECT
gen = tables["generalization_table.tex"]
panel = None
cur_method = None
bad = []
n_cells = 0
for line in gen.splitlines():
    if "Qwen3-8B}" in line:
        panel = "Qwen3-8B"; continue
    if "Llama-3.1-70B}" in line:
        panel = "Llama-3.1-70B-Instruct"; continue
    m = re.match(r"^\s*([A-Za-z0-9]*)\s*&\s*(--|\\wiki\{\}|\\finmix\{\})\s*&(.+)\\\\", line)
    if not m or panel is None:
        continue
    label, calib, cells = m.groups()
    if label:
        cur_method = label
    base = GEN_ROWS.get(cur_method)
    if base is None:
        continue
    scores = load_scores(panel)
    if calib == "--":
        cfg = base
    elif "wiki" in calib:
        cfg = base + "_c1"
    else:
        cands = [FINMIX_REDIRECT.get((panel, base + "_c3"), base + "_c3"),
                 base + "_c3", base + "_c3full"]
        cfg = next((c for c in cands
                    if any(k[0] == c for k in scores)), cands[0])
    for cell, t in zip(cells.split("&"), GEN_TASKS):
        cell = cell.strip()
        if cell == "--":
            continue
        n_cells += 1
        want = scores.get((cfg, t))
        if want is None or abs(float(cell) - round(want, 1)) > 0.051:
            bad.append((panel[:8], cur_method, t, cell,
                        None if want is None else round(want, 1)))
check(f"tab:gen all {n_cells} cells match source scores", not bad,
      f"mismatches: {bad[:4]}")
from paper_metrics import PLACEHOLDER_SCORES
ph_active = [(m, cfg, t) for (m, cfg, t) in PLACEHOLDER_SCORES
             if not os.path.exists(
                 f"experiments/results/{m}/{cfg}_{m}_t{t}.json")]
if ph_active:
    print(f"WARN placeholder scores ACTIVE for {len(ph_active)} cells: "
          f"{[(c, t) for _, c, t in ph_active]} -- replace before camera-ready")
expected_dashes = sum(
    1 for cfg, t in [("smoothquant_w8a8_c1", 4),
                     ("smoothquant_w8a8_c3full", 4),
                     ("smoothquant_w8a8_c3full", 11)]
    if not os.path.exists(
        f"experiments/results/Llama-3.1-70B-Instruct/{cfg}"
        f"_Llama-3.1-70B-Instruct_t{t}.json")
    and ("Llama-3.1-70B-Instruct", cfg, t) not in PLACEHOLDER_SCORES)
n_dashes = (gen.count("& -- &") + gen.count("& -- \\\\")
            - gen.count("RTN & --") - gen.count("BF16 & --"))
check(f"tab:gen dashes match missing cells ({expected_dashes})",
      n_dashes == expected_dashes, f"dashes={n_dashes}")

# ---- 7. conditional repair rates (section 5.2 evidence) ------------------
cr = [r for r in csv.reader(open(
    "experiments/results/analysis/repair_conditional_2x2.csv")) if r]
strata_rows = [r for r in cr if r[0] in ("gptq", "sparsegpt")]
check("conditional-2x2 has 4 strata", len(strata_rows) == 4)
ok = all(int(r[2]) / (int(r[2]) + int(r[3]))
         > 2 * int(r[4]) / (int(r[4]) + int(r[5])) for r in strata_rows)
check("P(fix|BF16 ok) > 2x P(fix|BF16 wrong) in every stratum", ok)
mh = next(r for r in cr if r and r[0] == "CMH_common_OR")
check("CMH common OR ~6.1 recorded", abs(float(mh[1]) - 6.07) < 0.02, mh[1])

print()
print("RESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}")
