#!/usr/bin/env python3
"""
plot_cpca_patterns.py
-------------------------------------------------------------------
Main complex-PCA figure, drawn from the outputs of cpca_group_lags.py, in the
style of plot_dfc_results.py (same group colours, fonts and panel layout,
copied here so the script runs on its own).

Fig_cpca_patterns   one row per component
    Left   The component in all 95 mice (Bolt et al., 2022, published code; the
           'ALL' rows of cpca_group_loadings.csv). Every strong ROI's phase lies
           within a few degrees of 0 or pi (the largest deviation is printed on
           the panel), so each complex loading is shown as a signed weight:
           +|L| for ROIs that peak with the reference ROI (phase near 0) and
           -|L| for ROIs half a cycle away (phase near pi). A display choice;
           nothing is refitted. Bars are sorted; dark = strong ROIs (|loading|
           >= the component's median), light = weak; the four largest of each
           sign are named.
    Right  Traveling index (Bolt et al., 2022, after Feeny, 2008) of each
           group's own fit: 0 = standing wave, 1 = traveling wave. Dashed line:
           the fit on all mice. Open circles: the group's component matched the
           all-mouse component with |cosine similarity| below MATCH_MIN. The
           title gives the smallest BH-corrected permutation q over the 8
           planned contrasts (all contrasts are in cpca_lag_contrasts.csv).
           Bars, drawn when cpca_ti_bootstrap_summary.csv (cpca_ti_bootstrap.py)
           is in --in_dir: 95% bootstrap percentile interval of each group's
           index (its mice resampled with replacement and refitted).

Usage
    python plot_cpca_patterns.py --in_dir dfc_dcc_cpca/cpca_group_lags \
        --formats pdf png tiff
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ALPHA = 0.05
# Weak-match cutoff on |cosine| between a group component and the all-mouse
# component: 0.85, the lower bound of 'fair similarity' for Tucker's congruence
# coefficient (Lorenzo-Seva and ten Berge, 2006, Methodology 2:57-64).
MATCH_MIN = 0.85
N_NAMED = 4

# ---- style, copied from plot_dfc_results.py ----
GROUP_ORDER_PREF = ["NTG_Y", "NTG_M", "NTG_A",
                    "CRND8_Y", "CRND8_M", "5XFAD_Y", "5XFAD_M"]
GROUP_COLORS = {
    "NTG_Y": "#e06666",
    "NTG_M": "#a73dd9",
    "NTG_A": "#a64d79",
    "CRND8_Y": "#3d85c6",
    "CRND8_M": "#38761d",
    "5XFAD_Y": "#8fce00",
    "5XFAD_M": "#d4b75d",
}
FALLBACK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
SECONDARY = "#52514e"
BASELINE = "#c3c2b7"
PANEL_SIZE = 8 / 3
TITLE_FONTSIZE = 8
YLABEL_FONTSIZE = 11
XTICK_FONTSIZE = 9
YTICK_FONTSIZE = 10
PNG_DPI = 600
TIFF_DPI = 600

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.2,
    "savefig.dpi": 350,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                     "ps.fonttype": 42})
warnings.filterwarnings("ignore", message=".*findfont.*")


def fmt_p(p, n_perm=None):
    if not np.isfinite(p):
        return "NA"
    if n_perm is not None and p <= 1.0 / (n_perm + 1) + 1e-15:
        return f"<{1.0 / n_perm:.1g}"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3g}"


def group_order(groups):
    present = [str(g) for g in pd.unique(groups)]
    ordered = [g for g in GROUP_ORDER_PREF if g in present]
    ordered += sorted(g for g in present if g not in ordered)
    return ordered


def build_palette(groups):
    pal, i = {}, 0
    for g in groups:
        if g in GROUP_COLORS:
            pal[g] = GROUP_COLORS[g]
        else:
            pal[g] = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
            i += 1
    return pal


def format_group_axis(ax, groups, title, ylabel):
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=4, labelsize=XTICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)
    sns.despine(ax=ax)


def panel_label(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, out_dir, stem, formats):
    for ext in formats:
        path = os.path.join(out_dir, f"{stem}.{ext}")
        if ext == "png":
            fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
        elif ext in ("tif", "tiff"):
            fig.savefig(path, dpi=TIFF_DPI, bbox_inches="tight",
                        pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}")



def pattern_panel(ax, a, k):
    """Signed loadings of component k in all mice (a: 'ALL' rows, by region)."""
    pole = np.where(np.cos(a.phase_rad) >= 0, 1.0, -1.0)
    signed = pd.Series(pole * a.magnitude, index=a.index).sort_values(ascending=False)
    strong = a.strong.reindex(signed.index).to_numpy(bool)
    dev = np.degrees(np.abs(np.angle(np.exp(2j * a.phase_rad[a.strong.astype(bool)]))) / 2)
    x = np.arange(len(signed))
    colours = np.where(signed.to_numpy() >= 0, "#e34948", "#2a78d6")
    ax.bar(x[strong], signed.to_numpy()[strong], width=0.8, color=colours[strong], lw=0)
    ax.bar(x[~strong], signed.to_numpy()[~strong], width=0.8, color=colours[~strong],
           alpha=0.35, lw=0)
    ax.axhline(0, color=BASELINE, lw=0.8)
    lab = a.label.reindex(signed.index)
    ax.text(0.98, 0.97, "Largest +: " + ", ".join(lab.iloc[:N_NAMED]),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=SECONDARY)
    if (signed < 0).any():
        ax.text(0.02, 0.03, "Largest −: " + ", ".join(lab.iloc[::-1].iloc[:N_NAMED]),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=SECONDARY)
    ax.text(0.98, 0.86, f"strong ROIs within {dev.max():.0f}° of 0 or π",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=SECONDARY)
    ax.set_xlim(-1, len(x))
    lo, hi = min(0.0, signed.min()), max(0.0, signed.max())
    ax.set_ylim(lo - 0.3 * (hi - lo) * (lo < 0), hi + 0.3 * (hi - lo))
    ax.set_xticks([])
    ax.set_xlabel("ROIs, sorted by loading", fontsize=8)
    ax.set_ylabel("Loading (signed by phase)", fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.set_title(f"Component {k}: pattern in all mice (n = {int(a.n_mice.iloc[0])})",
                 fontsize=8, loc="left")
    sns.despine(ax=ax)


def ti_panel(ax, t, con, groups, palette, top, boot=None):
    """Traveling index of each group fit (t: one component, indexed by group);
    boot: bootstrap summary for the same component, indexed by group, or None."""
    if boot is not None:
        for i, g in enumerate(groups):
            if g in boot.index:
                lo, hi = boot.loc[g, "ci_lo"], boot.loc[g, "ci_hi"]
                ax.vlines(i, lo, hi, color=palette[g], lw=1.2, zorder=2)
                ax.hlines([lo, hi], i - 0.12, i + 0.12, color=palette[g], lw=1.2, zorder=2)
    ax.axhline(t.loc["ALL", "traveling_index"], color=SECONDARY, lw=1, ls="--", zorder=1)
    ax.text(len(groups) - 0.5, t.loc["ALL", "traveling_index"], "all mice", ha="right",
            va="top", fontsize=7, color=SECONDARY)
    for i, g in enumerate(groups):
        weak = t.loc[g, "match_similarity"] < MATCH_MIN
        ax.scatter(i, t.loc[g, "traveling_index"], s=45, zorder=3,
                   facecolor="white" if weak else palette[g], edgecolor=palette[g],
                   linewidth=1.3)
    sig = con[con.q_TI_BH < ALPHA]
    if len(sig):
        title = "Traveling index by group; " + "; ".join(
            f"{r.contrast}: q={fmt_p(r.q_TI_BH)}" for r in sig.itertuples())
    else:
        title = f"Traveling index by group (all q ≥ {con.q_TI_BH.min():.2f})"
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_ylabel("Traveling index", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, top)
    sns.despine(ax=ax)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--in_dir", default=os.path.join("dfc_dcc_cpca", "cpca_group_lags"))
    ap.add_argument("--out_dir", default=None, help="default: --in_dir")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = ap.parse_args()
    out = args.out_dir or args.in_dir
    os.makedirs(out, exist_ok=True)
    load = pd.read_csv(os.path.join(args.in_dir, "cpca_group_loadings.csv"))
    ti = pd.read_csv(os.path.join(args.in_dir, "cpca_group_traveling_index.csv"))
    con = pd.read_csv(os.path.join(args.in_dir, "cpca_lag_contrasts.csv"))
    bpath = os.path.join(args.in_dir, "cpca_ti_bootstrap_summary.csv")
    boot = pd.read_csv(bpath) if os.path.exists(bpath) else None

    groups = group_order(ti.group[ti.group != "ALL"])
    palette = build_palette(groups)
    comps = sorted(load.component.unique())
    ymax = ti.traveling_index.max() if boot is None else max(ti.traveling_index.max(),
                                                             boot.ci_hi.max())
    top = min(1.05, max(0.6, 1.15 * ymax))
    fig = plt.figure(figsize=(10.0, 3.4 * len(comps)))
    gs = fig.add_gridspec(len(comps), 2, width_ratios=[1.4, 1.0], wspace=0.28, hspace=0.6)
    letters = iter("ABCDEFGHIJ")
    for row, k in enumerate(comps):
        a = load[(load.group == "ALL") & (load.component == k)].set_index("region")
        ax = fig.add_subplot(gs[row, 0])
        pattern_panel(ax, a, k)
        panel_label(ax, next(letters), x=-0.1, y=1.04)
        ax = fig.add_subplot(gs[row, 1])
        ti_panel(ax, ti[ti.component == k].set_index("group"), con[con.component == k],
                 groups, palette, top,
                 None if boot is None else boot[boot.component == k].set_index("group"))
        panel_label(ax, next(letters), x=-0.16, y=1.04)
    note = (f"Open circles: group component matched the all-mouse component "
            f"with |cosine| < {MATCH_MIN:g}. Dashed line: fit on all mice.")
    if boot is not None:
        note += (f"\nBars: 95% bootstrap interval ({int(boot.n_boot.min()):,} resamples of each "
                 f"group's mice, refitted); group differences are tested by permutation (q in titles).")
    fig.text(0.5, 0.035, note, ha="center", va="top", fontsize=7, color=SECONDARY,
             **({} if boot is None else {"linespacing": 1.4}))
    save(fig, out, "Fig_cpca_patterns", args.formats)


if __name__ == "__main__":
    main()
