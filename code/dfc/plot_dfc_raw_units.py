"""
plot_dfc_raw_units.py
-------------------------------------------------------------------
Raw-unit view of the pre-specified non-stationarity measures (Supplement).

For each measure, every mouse's observed value is plotted against the mean of
its own 19 phase-randomised surrogates, in the measure's natural units (DCC a,
SD of the conditional correlation, GARCH alpha, ...). It shows what the z-scores
of Fig_dfc_nonstationarity are made of. Descriptive only: the inference is the
calibrated test on z (plot_dfc_results.py). The stationary replicas' raw values
are not in dfc_subject_metrics.csv (only their z), so the calibration itself is
not shown in raw units.

    python plot_dfc_raw_units.py --in_dir dfc_dcc_cpca

Writes <in_dir>/figures/FigS_dfc_raw_units.{pdf,png} and
<in_dir>/figures/table_raw_units.csv. Styling, labels and group colours come
from plot_dfc_results.py.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.lines import Line2D            # noqa: E402

import plot_dfc_results as pdr                 # noqa: E402  (same folder)

UNITS = {
    "dcc_a": "unitless (0 = constant correlation)",
    "dcc_LR": "2 x log-likelihood gain over constant correlation",
    "corr_fluct": "SD over time of the conditional correlation, mean over pairs",
    "garch_alpha": "unitless (0 = constant variance), mean over components",
    "garch_LR": "2 x log-likelihood gain over constant variance, mean over components",
    "vol_cv": "CV of the conditional SD, mean over components",
    "cpca_roi_c1_env_cv": "CV of the component's amplitude envelope",
}


def iqr_text(x, fmt):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return "NA"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:{fmt}} [{q1:{fmt}}, {q3:{fmt}}]"


def fmt_for(values):
    m = np.nanmedian(np.abs(values))
    return ".2f" if m >= 1 else (".3f" if m >= 0.01 else ".4f")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--in_dir", default="dfc_dcc_cpca")
    ap.add_argument("--out_dir", default=None, help="default: <in_dir>/figures")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = ap.parse_args()
    out = args.out_dir or os.path.join(args.in_dir, "figures")
    os.makedirs(out, exist_ok=True)

    d = pdr.load_all(args.in_dir)["sub"]
    groups = pdr.group_order(d.group)
    palette = pdr.build_palette(groups)
    metrics = [m for m in pdr.PRIMARY_TESTS if m in set(d.metric)]

    rows = []
    ncol = 4
    nrow = int(np.ceil((len(metrics) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(pdr.PANEL_SIZE * ncol * 1.05, pdr.PANEL_SIZE * nrow * 1.05),
                             squeeze=False)
    for i, m in enumerate(metrics):
        ax = axes.flat[i]
        x = d[d.metric == m].copy()
        x["diff"] = x.observed - x.phase_mean
        ok = x[np.isfinite(x.observed) & np.isfinite(x.phase_mean)]
        lo = min(ok.observed.min(), ok.phase_mean.min())
        hi = max(ok.observed.max(), ok.phase_mean.max())
        pad = 0.05 * (hi - lo if hi > lo else 1)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color=pdr.BASELINE, lw=1.0, zorder=1)
        for g in groups:
            s = ok[ok.group == g]
            ax.scatter(s.phase_mean, s.observed, s=16, color=palette[g], edgecolor="white",
                       linewidth=0.35, alpha=pdr.POINT_ALPHA, zorder=3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        n_above = int((ok["diff"] > 0).sum())
        f = fmt_for(ok.observed)
        ax.set_title(pdr.long_label(m), fontsize=pdr.TITLE_FONTSIZE + 1)
        ax.text(0.97, 0.03, f"above line: {n_above}/{len(ok)}\nmedian diff: {np.median(ok['diff']):+{f}}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=7, color=pdr.SECONDARY,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5), zorder=4)
        ax.set_xlabel("Mean of own phase surrogates", fontsize=8)
        ax.set_ylabel("Observed", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        rows.append({
            "metric": m, "label": pdr.long_label(m), "unit": UNITS.get(m, ""), "n": len(ok),
            "observed_median_IQR": iqr_text(ok.observed, f),
            "surrogate_mean_median_IQR": iqr_text(ok.phase_mean, f),
            "difference_median_IQR": iqr_text(ok["diff"], f),
            "surrogate_SD_median": f"{np.median(ok.phase_sd):{f}}",
            "mice_observed_above_surrogate_mean": n_above,
            "median_z_observed": round(float(np.nanmedian(ok.phase_z)), 3),
            "median_z_replica": round(float(np.nanmedian(ok.phase_replica_z)), 3),
        })
    for j in range(len(metrics), nrow * ncol):
        axes.flat[j].axis("off")
    key = axes.flat[len(metrics)]
    handles = [Line2D([], [], marker="o", ls="", color=palette[g], markeredgecolor="white",
                      markersize=6, label=g) for g in groups]
    handles.append(Line2D([], [], color=pdr.BASELINE, lw=1.0, label="observed = surrogate mean"))
    key.legend(handles=handles, loc="center left", frameon=False, fontsize=7.5, title="Group",
               title_fontsize=8)
    fig.suptitle("Non-stationarity measures in their own units: each mouse vs its phase surrogates",
                 fontsize=pdr.TITLE_FONTSIZE + 2, y=1.0)
    fig.tight_layout()
    pdr.save(fig, out, "FigS_dfc_raw_units", args.formats)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "table_raw_units.csv"), index=False)
    print(tab[["metric", "observed_median_IQR", "surrogate_mean_median_IQR",
               "mice_observed_above_surrogate_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
