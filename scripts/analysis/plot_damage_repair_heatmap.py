#!/usr/bin/env python3
"""
Heatmaps of per-task score changes for the five calibrated methods,
Llama-3.1-8B, in a shared visual language: red = capability lost,
blue = gained, diverging RdBu with a perceptual stretch near zero.

Default: two panels, damage (WikiText minus BF16) and repair
(FinMix minus WikiText).            -> paper/figures/damage_repair_heatmap.*
--c4:   one panel, C4 minus WikiText (the generic-generic contrast
        behind the replication paragraph).
                                    -> paper/figures/generic_corpus_heatmap.*

All values follow the display-consistency policy (differences of scores
rounded to one decimal). Stars: damage panel from damage_significance_*.csv
(vs BF16); repair and C4 panels from calibration_significance_*.csv
(paired c3:c1 / c2:c1 tests).
"""

import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from paper_metrics import load_scores

MODEL = "Llama-3.1-8B-Instruct"
METHODS = [("GPTQ", "gptq_w4a16", "_c1", "_c3"),
           ("AWQ", "awq_w4a16_asym", "_c1", "_c3"),
           ("SmoothQuant", "smoothquant_w8a8", "_c1", "_c3"),
           ("SparseGPT", "sparsegpt_unstructured", "_c1", "_c3full"),
           ("Wanda", "wanda_4of8", "_c1", "_c3full")]
# BigData22 omitted: near random for every configuration (see tab:main),
# its corpus-dependent swings only distract from the pattern
TASKS = [(1, "FPB"), (2, "FOMC"), (3, "Headlines"), (6, "FinRED"),
         (8, "FinCausal"),
         (4, "CFA"), (5, "FinQA"), (9, "ConvFinQA"), (11, "TAT-QA")]
N_CLS = 5
LIM_POS = 25      # positive side of the shared color range (cells beyond
                  # clip to full saturation); linear (gamma 1) per request
LIM_NEG = 25      # negative side; asymmetric because the deepest negative
                  # cell is only -13.3
GAMMA_POS = 1.0   # blue side: evenly divided, no compression
GAMMA_NEG = 1.2   # red side: mild compression near zero

for fname in ("Arial", "Liberation Sans"):
    if any(fname in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fname
        break
plt.rcParams.update({"axes.linewidth": 0.6, "pdf.fonttype": 42})


def load_sig_damage():
    out = {}
    p = f"experiments/results/analysis/damage_significance_{MODEL}.csv"
    for row in csv.DictReader(open(p)):
        out[(row["method"], int(row["task"][1:]))] = row["sig"]
    return out


def load_sig_pair(treat_prefix, base):
    out = {}
    p = f"experiments/results/analysis/calibration_significance_{MODEL}.csv"
    for row in csv.DictReader(open(p)):
        if row["treat"].startswith(treat_prefix) and row["base"] == base:
            out[(row["method"], int(row["task"][1:]))] = row["sig"]
    return out


def make_norm():
    # white stays anchored at 0; each side is gamma-compressed against
    # its own limit
    def fwd(x):
        x = np.asarray(x, dtype=float)
        lim = np.where(x >= 0, LIM_POS, LIM_NEG)
        g = np.where(x >= 0, GAMMA_POS, GAMMA_NEG)
        return np.sign(x) * (np.abs(x) / lim) ** g

    def inv(y):
        y = np.asarray(y, dtype=float)
        lim = np.where(y >= 0, LIM_POS, LIM_NEG)
        g = np.where(y >= 0, GAMMA_POS, GAMMA_NEG)
        return np.sign(y) * np.abs(y) ** (1 / g) * lim

    return matplotlib.colors.FuncNorm((fwd, inv), vmin=-LIM_NEG, vmax=LIM_POS)


def draw_panel(ax, vals, stars, title, norm, cmap, show_ylabels):
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([m for m, *_ in METHODS], rotation=35,
                       ha="right", fontsize=8)
    ax.set_title(title, fontsize=8.5, pad=5)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            dark = abs(norm(v) - 0.5) > 0.31
            ax.text(j, i - 0.13 if stars[i][j] else i, f"{v:+.1f}",
                    ha="center", va="center", fontsize=6.4,
                    color="white" if dark else "#1a1a1a")
            if stars[i][j]:
                ax.text(j, i + 0.27, stars[i][j], ha="center",
                        va="center", fontsize=5.6,
                        color="white" if dark else "#1a1a1a")
    ax.axhline(N_CLS - 0.5, color="#1a1a1a", linewidth=1.1)
    ax.set_xticks(np.arange(-0.5, len(METHODS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(TASKS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="both", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(1.2)
    if show_ylabels:
        ax.set_yticks(range(len(TASKS)))
        ax.set_yticklabels([n for _, n in TASKS], fontsize=8)
        for y, lab in ((N_CLS / 2 - 0.5, "Classification"),
                       (N_CLS + (len(TASKS) - N_CLS) / 2 - 0.5,
                        "Numerical QA")):
            ax.text(-3.15, y, lab, rotation=90, ha="center", va="center",
                    fontsize=8.5, color="#444444")
    return im


def add_colorbar(fig, im, axes, fraction=0.035,
                 label="points, red = capability lost, blue = gained"):
    cbar = fig.colorbar(im, ax=axes, fraction=fraction, pad=0.02,
                        extend="both",
                        ticks=[-25, -20, -10, 0, 10, 20, 25])
    cbar.set_label(label, fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.outline.set_visible(False)


def save(fig, stem):
    os.makedirs("paper/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/{stem}.{ext}",
                    bbox_inches="tight", dpi=300)
    print(f"wrote paper/figures/{stem}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c4", action="store_true",
                    help="single-panel C4-minus-WikiText figure")
    ap.add_argument("--third", default="wiki", choices=["wiki", "c4"],
                    help="right panel: FinMix-WikiText (total repair, "
                         "matches section 5 numbers) or FinMix-C4 "
                         "(decomposition; middle+right sum to the total)")
    ap.add_argument("--cmap", default="RdBu", choices=["RdBu", "RdYlBu"],
                    help="RdBu = classic two-hue diverging; RdYlBu = "
                         "three-hue with yellow midpoint")
    args = ap.parse_args()

    r = load_scores(MODEL)
    norm = make_norm()
    cmap = plt.get_cmap(args.cmap)

    def rd(cfg, t):
        return round(r[(cfg, t)], 1)

    if args.c4:
        vals = np.array([[rd(base + "_c2", t) - rd(base + "_c1", t)
                          for _, base, _, _ in METHODS] for t, _ in TASKS])
        sig = load_sig_pair("c2", "c1")
        stars = [[sig.get((base, t), "") for _, base, _, _ in METHODS]
                 for t, _ in TASKS]
        fig, ax = plt.subplots(figsize=(3.7, 3.6))
        im = draw_panel(ax, vals, stars,
                        "C4 calibration\n(score minus WikiText)",
                        norm, cmap, show_ylabels=True)
        add_colorbar(fig, im, [ax], fraction=0.05)
        save(fig, "generic_corpus_heatmap")
        return

    damage = np.array([[rd("bf16", t) - rd(base + c1, t)
                        for _, base, c1, _ in METHODS] for t, _ in TASKS])
    generic = np.array([[rd(base + "_c2", t) - rd(base + "_c1", t)
                         for _, base, _, _ in METHODS] for t, _ in TASKS])
    dref = "_c1" if args.third == "wiki" else "_c2"
    domain = np.array([[rd(base + c3, t) - rd(base + dref, t)
                        for _, base, _, c3 in METHODS] for t, _ in TASKS])
    dsig = load_sig_damage()
    gsig = load_sig_pair("c2", "c1")
    msig = load_sig_pair("c3", "c1" if args.third == "wiki" else "c2")
    dstars = [[dsig.get((m, t), "") for m, _, _, _ in METHODS]
              for t, _ in TASKS]
    gstars = [[gsig.get((base, t), "") for _, base, _, _ in METHODS]
              for t, _ in TASKS]
    mstars = [[msig.get((base, t), "") for _, base, _, _ in METHODS]
              for t, _ in TASKS]

    # one shared colormap and colorbar across all three panels: equal
    # numbers get equal colors, so a heavily damaged cell (left) and its
    # large repair (right) match visually. Damage is signed positive.
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.4), sharey=True,
                             gridspec_kw={"wspace": 0.18})
    draw_panel(axes[0], damage, dstars,
               "Damage\n(BF16 $-$ WikiText)",
               norm, cmap, show_ylabels=True)
    draw_panel(axes[1], generic, gstars,
               "Generic-corpus recovery\n(C4 $-$ WikiText)",
               norm, cmap, show_ylabels=False)
    im = draw_panel(axes[2], domain, mstars,
                    "Domain-matched recovery\n(FinMix $-$ WikiText)" if args.third == "wiki"
               else "Domain-matched recovery\n(FinMix $-$ C4)",
                    norm, cmap, show_ylabels=False)
    add_colorbar(fig, im, axes, fraction=0.025, label="points")
    stem = "damage_repair_heatmap"
    if args.third == "c4":
        stem += "_decomp"
    if args.cmap != "RdBu":
        stem += "_" + args.cmap.lower()
    save(fig, stem)


if __name__ == "__main__":
    main()
