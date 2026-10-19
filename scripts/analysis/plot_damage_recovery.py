#!/usr/bin/env python3
"""
Figure 1, two panels (figure*):
  (a) Llama-3.1-8B only: damage vs recovery, least-squares fit and its
      95% prediction band (the relationship is ESTABLISHED here).
  (b) Qwen3-8B only, held out from the fit: same axes, the Llama fit and
      band repeated in grey, Qwen points as triangles (the relationship
      is TESTED here). SparseGPT labeled directly.

x = damage under generic calibration  (BF16 - WikiText), pp
y = recovery from domain calibration  (FinMix - WikiText), pp
One point per (method, task); BigData22 excluded throughout (chance-level,
consistent with the paper's analysis set of nine tasks; n = 45 + 45 = 90).
rho / slope / band-coverage numbers are printed for the caption, not drawn.

Output: paper/figures/fig1_damage_recovery.{pdf,png}
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_metrics import load_scores as _shared_load_scores
from paper_metrics import FINMIX_REDIRECT
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# BigData22 (t10) excluded: chance-level on every model (see tab:main)
TASKS = [1, 2, 3, 4, 5, 6, 8, 9, 11]
METHODS = {  # key -> (label, okabe-ito hex)
    "gptq_w4a16": ("GPTQ", "#0072B2"),
    "awq_w4a16_asym": ("AWQ", "#D55E00"),
    "sparsegpt_unstructured": ("SparseGPT", "#009E73"),
    "wanda_4of8": ("Wanda", "#CC79A7"),
    "smoothquant_w8a8": ("SmoothQuant", "#56B4E9"),
}


def load_points(model):
    r = {k: v / 100 for k, v in _shared_load_scores(model).items()}
    pts = []
    for meth in METHODS:
        for t in TASKS:
            c3_cfg = FINMIX_REDIRECT.get((model, f"{meth}_c3"), f"{meth}_c3")
            bf, c1 = r.get(("bf16", t)), r.get((f"{meth}_c1", t))
            c3 = r.get((c3_cfg, t))
            if c3 is None:
                c3 = r.get((f"{meth}_c3full", t))
            if None in (bf, c1, c3):
                continue
            pts.append(((bf - c1) * 100, (c3 - c1) * 100, meth))
    return pts


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for i, idx in enumerate(order):
            r[idx] = i
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = sqrt(sum((a - mx) ** 2 for a in rx)
               * sum((b - my) ** 2 for b in ry))
    return num / den


def main():
    L = load_points("Llama-3.1-8B-Instruct")
    Q = load_points("Qwen3-8B")
    B = load_points("Llama-3.1-70B-Instruct")
    allp = L + Q
    rho = spearman([p[0] for p in allp], [p[1] for p in allp])
    print(f"n={len(allp)}, Spearman rho={rho:.3f}")

    # Llama-only least-squares fit + 95% prediction band
    fx, fy = [p[0] for p in L], [p[1] for p in L]
    n = len(fx)
    mx, my = sum(fx) / n, sum(fy) / n
    sxx = sum((a - mx) ** 2 for a in fx)
    slope = sum((a - mx) * (b - my) for a, b in zip(fx, fy)) / sxx
    intercept = my - slope * mx
    resid = [b - (slope * a + intercept) for a, b in zip(fx, fy)]
    sd = sqrt(sum(e * e for e in resid) / (n - 2))
    tcrit = 2.017  # t 97.5th percentile, df = 43
    print(f"Llama fit: n={n}, slope={slope:.3f}, intercept={intercept:.2f}")

    def half(x):
        return tcrit * sd * sqrt(1 + 1 / n + (x - mx) ** 2 / sxx)

    for pts, name in ((L, "Llama"), (Q, "Qwen"), (B, "70B")):
        out = [(x, y) for x, y, _m in pts
               if abs(y - (slope * x + intercept)) > half(x)]
        print(f"{name}: {len(pts) - len(out)}/{len(pts)} inside band; "
              f"outside {[(round(a,1), round(b,1)) for a, b in out]}")

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.5, "axes.linewidth": 0.6,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), dpi=300,
                             sharey=True, gridspec_kw={"wspace": 0.07})

    xs_all = [p[0] for p in allp]
    ys_all = [p[1] for p in allp]
    x_lo, x_hi = min(xs_all) - 2, max(xs_all) + 2
    grid = [x_lo + i * (x_hi - x_lo) / 100 for i in range(101)]

    for ax, pts, marker, title in (
            (axes[0], L, "o", "(a) Llama-3.1-8B fit"),
            (axes[1], Q, "o", "(b) Held-out Qwen3-8B"),
            (axes[2], B, "o", "(c) Llama-3.1-70B scale check")):
        ax.axhline(0, color="0.85", lw=0.6, zorder=1)
        ax.axvline(0, color="0.85", lw=0.6, zorder=1)
        diag_end = min(x_hi, max(ys_all) * 1.1)
        ax.plot([0, diag_end], [0, diag_end], color="0.7", lw=0.7, ls=":",
                zorder=1.4)
        ax.fill_between(grid, [slope * g + intercept - half(g) for g in grid],
                        [slope * g + intercept + half(g) for g in grid],
                        color="0.55", alpha=0.13, lw=0, zorder=1.5)
        ax.plot([x_lo, x_hi],
                [intercept + slope * x_lo, intercept + slope * x_hi],
                color="0.3", lw=0.8, ls="--", zorder=2)
        for x, y, meth in pts:
            ax.scatter(x, y, s=24, marker=marker,
                       facecolor=METHODS[meth][1], edgecolor="white",
                       linewidth=0.4, alpha=0.6, zorder=3)
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-12, max(ys_all) + 3)
        ax.set_aspect("equal", adjustable="box")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(width=0.6, length=2.5)

    fig.supxlabel("Damage under WikiText (pp)", fontsize=8.5, y=0.03)
    fig.supylabel("Recovery with FinMix (pp)", fontsize=8.5, x=0.008)

    # label the y=x line in panel (a), rotated along its on-screen direction
    from math import atan2, degrees
    fig.canvas.draw()
    p0 = axes[0].transData.transform((20.0, 20.0))
    p1 = axes[0].transData.transform((30.0, 30.0))
    ang = degrees(atan2(p1[1] - p0[1], p1[0] - p0[0]))
    axes[0].text(27.0, 30.5, "Full repair", fontsize=6.2, color="0.5",
                 rotation=ang, rotation_mode="anchor", ha="center")

    # annotate the flagship held-out point: Qwen SparseGPT FinQA
    sg = [(x, y) for x, y, m in Q if m == "sparsegpt_unstructured"]
    px, py = max(sg, key=lambda p: p[1])
    axes[1].annotate("SparseGPT on FinQA", (px, py),
                     xytext=(px - 1.0, py + 7.5), fontsize=6.8, ha="right",
                     color="0.15",
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45",
                                     shrinkB=4))

    handles = [Line2D([], [], marker="s", linestyle="", markersize=5,
                      markerfacecolor=to_rgba(hexv, 0.6),
                      markeredgecolor="none", label=label)
               for label, hexv in METHODS.values()]
    handles += [Line2D([], [], linestyle="--", color="0.3", lw=0.8,
                       label="Llama fit"),
                Patch(facecolor="0.55", alpha=0.2, lw=0,
                      label="95% prediction band")]
    leg = axes[0].legend(handles=handles, fontsize=6.2, loc="upper left",
                         frameon=False, handletextpad=0.3,
                         labelspacing=0.25, borderaxespad=0.2,
                         alignment="left")

    fig.tight_layout(pad=0.4)
    os.makedirs("paper/figures", exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(f"paper/figures/fig1_damage_recovery.{ext}",
                    bbox_inches="tight", dpi=300)
    print("wrote paper/figures/fig1_damage_recovery.{pdf,png}")


if __name__ == "__main__":
    main()
