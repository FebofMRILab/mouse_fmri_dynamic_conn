"""
plot_dcc_example.py
-------------------------------------------------------------------
Supplement figure in the units of Lindquist et al. (2014): the conditional
correlation rho_ij(t) of one mouse (published DCC estimator, dcc_mvgarch.m),
against the same quantity in its 19 phase-randomised surrogates.

  A-C  rho(t) for components 1-2, 1-3, 2-3 (fixed in advance): the mouse (line),
       the surrogates' pointwise 2.5-97.5 % range (band) and median, and the
       static correlation (dotted)

The former panel D (SD over time of rho(t) for every pair, mouse against its
surrogates) was removed on 11 Nov 2026: a single mouse's pair-by-pair scatter
invited a reading the paper does not make. dcc_example_traces.m still writes
dcc_example_pair_sd.csv; it is no longer read here.

Reads the folder written by dcc_example_traces.m:
    python plot_dcc_example.py --in_dir dfc_dcc_cpca/dcc_example
Writes FigS_dcc_example.{pdf,png} and FigS_dcc_example_caption.txt there.
Display only; the inference is the test across mice (plot_dfc_results.py).
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.lines import Line2D            # noqa: E402
from matplotlib.patches import Patch           # noqa: E402

import plot_dfc_results as pdr                 # noqa: E402  (same folder)

OBS = pdr.INK
SURR = pdr.NULL_COLORS["phase"]                # the phase null's colour in every DFC figure


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--in_dir", default=os.path.join("dfc_dcc_cpca", "dcc_example"))
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = ap.parse_args()
    d = args.in_dir
    tr = pd.read_csv(os.path.join(d, "dcc_example_traces.csv"))
    par = pd.read_csv(os.path.join(d, "dcc_example_params.csv"))
    info = open(os.path.join(d, "dcc_example_info.txt")).read()
    mouse = re.search(r"mouse_id: (\d+)", info).group(1)
    group = re.search(r"group: (\S+)", info).group(1)

    inp = re.search(r"Input: (\d+) group-PCA components .*?\(TR ([0-9.]+) s\)", info)
    n_comp, tr_eff = (inp.group(1), inp.group(2)) if inp else ("?", "?")
    n_surr = int((par.source != "observed").sum())

    pairs = list(dict.fromkeys(tr.pair))
    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(len(pairs), 1, hspace=0.45)
    for i, pr in enumerate(pairs):
        ax = fig.add_subplot(gs[i, 0])
        x = tr[tr.pair == pr]
        t = x.time_s.to_numpy() / 60
        ax.fill_between(t, x.surr_p025, x.surr_p975, color=SURR, alpha=0.18, lw=0, zorder=1)
        ax.plot(t, x.surr_p50, color=SURR, lw=0.9, alpha=0.9, zorder=2)
        ax.axhline(x.static_r.iloc[0], color=pdr.SECONDARY, lw=0.9, ls=":", zorder=2)
        ax.plot(t, x.observed_r, color=OBS, lw=1.1, zorder=3)
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlim(t.min(), t.max())
        a, b = pr.split("-")
        ax.set_title(f"Components {a} and {b}  (static r = {x.static_r.iloc[0]:.2f})",
                     fontsize=pdr.TITLE_FONTSIZE + 1, loc="left")
        ax.set_ylabel("ρ(t)", fontsize=9)
        ax.tick_params(labelsize=8)
        if i < len(pairs) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (min)", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(-0.09, 1.08, "ABC"[i], transform=ax.transAxes, fontsize=11, fontweight="bold")
    handles = [Line2D([], [], color=OBS, lw=1.1, label=f"mouse {mouse} ({group})"),
               Patch(facecolor=SURR, alpha=0.18, label="phase surrogates, 2.5-97.5 %"),
               Line2D([], [], color=SURR, lw=0.9, label="surrogate median"),
               Line2D([], [], color=pdr.SECONDARY, lw=0.9, ls=":", label="static correlation")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.035))
    pdr.save(fig, d, "FigS_dcc_example", args.formats)

    cap = (
        f"Figure S-x. Conditional correlations in one mouse (mouse {mouse}, {group}; drawn at random "
        f"from the NTG_Y group, see dcc_example_info.txt). Conditional correlations ρ(t) were estimated "
        f"with the published DCC(1,1) estimator (dcc_mvgarch.m; Lindquist et al., 2014) on the "
        f"pipeline's input for this mouse: {n_comp} group-PCA components, odd volumes (TR {tr_eff} s), "
        f"AR(2) residuals. (A-C) ρ(t) for the three largest components (pairs fixed in advance), with the "
        f"pointwise 2.5-97.5 % range and median of the same quantity in the mouse's {n_surr} "
        f"phase-randomised surrogates, which keep its spectra and static correlation but remove "
        f"non-stationarity. Dotted line: static correlation. "
        "Display only: the inference is the test across mice.")
    open(os.path.join(d, "FigS_dcc_example_caption.txt"), "w").write(cap + "\n")
    print(cap)


if __name__ == "__main__":
    main()
