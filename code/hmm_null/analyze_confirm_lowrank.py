#!/usr/bin/env python3
"""
analyze_confirm_lowrank.py
-------------------------------------------------------------------
Pre-registered analysis of the confirmatory PCA-HMM run
(run4c_confirm_lowrank.m). Reads <root>/observed and <root>/replica.

Per model (lowrank 15, 25) x K (2, 3, 4), with demeaned state-FC split-half
similarity per split r (mean over matched states, as in MATLAB):
  uncalibrated   observed vs phase null: median difference, one-sided
                 Mann-Whitney
  procedure bias replica vs replica null (should be ~0 if the procedure is
                 unbiased; it is what the calibration removes)
  CALIBRATED     c_r = (obs_r - null_r) - (rep_r - repnull_r), paired over r;
                 one-sided Wilcoxon signed-rank c > 0; Holm across all cells
  weakest state  median over splits of the least-replicating matched state,
                 observed vs null
  DECISION       state structure only if calibrated Holm p < 0.05 AND the
                 weakest state replicates better in observed than null
Drift (full fits, observed data): per state, occupancy in the 2nd minus 1st
scan half per mouse; two-sided Wilcoxon across mice, Holm across states
within each model x K.

Usage
    python analyze_confirm_lowrank.py --root hmm_confirm_lowrank
Outputs in <root>/analysis: confirm_cells.csv, confirm_occupancy_drift.csv,
Fig_confirm_lowrank.{pdf,png}
"""

import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 1.2,
                     "savefig.dpi": 350, "savefig.facecolor": "white",
                     "figure.facecolor": "white"})
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
warnings.filterwarnings("ignore", message=".*findfont.*")

INK, SECONDARY, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [("obs", "Observed", INK), ("null", "Phase null", "#2a78d6"),
          ("rep", "Stationary replica", MUTED), ("repnull", "Replica's phase null", "#eb6834")]
ALPHA = 0.05
SEED = 20261001


def holm(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    m = q.size
    if m:
        o = np.argsort(q)
        adj = np.empty(m)
        run = 0.0
        for rank, i in enumerate(o):
            run = max(run, (m - rank) * q[i])
            adj[i] = min(1.0, run)
        out[ok] = adj
    return out


def per_split(path, sfx=""):
    """Per (condition, repetition, K): mean and min over matched states of the
    demeaned similarity. 'null' must not be read as missing."""
    f = os.path.join(path, f"split_half_per_state{sfx}.csv")
    if not os.path.isfile(f):
        return None
    d = pd.read_csv(f, keep_default_na=False, na_values=[""])
    d = d[d.metric == "demeaned"]
    g = d.groupby(["condition", "repetition", "K"]).similarity.agg(["mean", "min"]).reset_index()
    return g


def series(g, cond, K, col="mean"):
    s = g[(g.condition == cond) & (g.K == K)].set_index("repetition")[col]
    return s.sort_index()


def wilcoxon_greater(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    if x.size < 6:
        return np.nan
    return float(stats.wilcoxon(x, alternative="greater").pvalue)


def mw_greater(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan
    return float(stats.mannwhitneyu(x, y, alternative="greater").pvalue)


def median_ci(x, rng, n_boot=10000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan, np.nan, np.nan
    m = np.median(x[rng.integers(0, x.size, (n_boot, x.size))], axis=1)
    return float(np.median(x)), *np.percentile(m, [2.5, 97.5]).tolist()


def rank_of(label):
    digits = re.sub(r"[^0-9]", "", label)
    return digits or label


def fmt_p(p):
    if not np.isfinite(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="hmm_confirm_lowrank")
    ap.add_argument("--null", default="phase", choices=["phase", "iaaft"],
                    help="phase = the pre-registered decision null")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    sfx = "" if args.null == "phase" else "_iaaft"
    out = args.out_dir or os.path.join(args.root, "analysis")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    labels = sorted({os.path.basename(p).replace("K_selection_", "")
                     for p in glob.glob(os.path.join(args.root, "observed", "K_selection_*"))
                     if os.path.isdir(p)},
                    key=lambda s: int(rank_of(s)) if rank_of(s).isdigit() else 0)
    if not labels:
        raise SystemExit(f"No observed K_selection_* folders under {args.root}")

    cells, data = [], {}
    for lab in labels:
        go = per_split(os.path.join(args.root, "observed", f"K_selection_{lab}"), sfx)
        gr = per_split(os.path.join(args.root, "replica", f"K_selection_{lab}"), sfx)
        if go is None:
            print(f"{lab}: no observed per-state file; skipped")
            continue
        if gr is None:
            print(f"{lab}: NO REPLICA ARM -> calibrated test not available")
        data[lab] = (go, gr)
        for K in sorted(go.K.unique()):
            o, n = series(go, "observed", K), series(go, "null", K)
            ow, nw = series(go, "observed", K, "min"), series(go, "null", K, "min")
            row = dict(model=lab, K=int(K), n_splits=len(o), n_null=len(n),
                       median_obs=o.median(), median_null=n.median(),
                       delta_uncal=o.median() - n.median(), p_uncal=mw_greater(o, n),
                       weakest_obs=ow.median(), weakest_null=nw.median())
            if gr is not None:
                r, rn = series(gr, "observed", K), series(gr, "null", K)
                row.update(median_rep=r.median(), median_repnull=rn.median(),
                           delta_replica=r.median() - rn.median(), p_replica=mw_greater(r, rn))
                c = ((o - n) - (r - rn)).dropna()
                med, lo, hi = median_ci(c, rng)
                row.update(n_paired=len(c), calibrated_median=med, ci_low=lo, ci_high=hi,
                           p_calibrated=wilcoxon_greater(c))
            cells.append(row)
    T = pd.DataFrame(cells)
    if "p_calibrated" not in T:
        T["p_calibrated"] = np.nan
    T["p_calibrated_holm"] = holm(T.p_calibrated)
    T["p_uncal_holm"] = holm(T.p_uncal)
    T["weakest_state_ok"] = T.weakest_obs > T.weakest_null
    T["state_structure"] = (T.p_calibrated_holm < ALPHA) & T.weakest_state_ok
    T["null_used"] = args.null
    T.to_csv(os.path.join(out, "confirm_cells.csv"), index=False)

    # ---------------- drift from the full fits ----------------
    drift = []
    occ_all = {}
    for lab in labels:
        f = os.path.join(args.root, "observed", f"occupancy_{lab}.csv")
        if not os.path.isfile(f):
            continue
        occ = pd.read_csv(f)
        occ_all[lab] = occ
        for K, ok in occ.groupby("K"):
            rows = []
            for k, sk in ok.groupby("state"):
                w = sk.pivot_table(index="subject_id", columns="quarter", values="fo_quarter")
                d = (w[3] + w[4]) / 2 - (w[1] + w[2]) / 2
                p = float(stats.wilcoxon(d).pvalue) if (d != 0).sum() > 5 else np.nan
                rows.append(dict(model=lab, K=int(K), state=int(k), n=len(d),
                                 fo_total=sk.drop_duplicates("subject_id").fo_total.mean(),
                                 median_change=d.median(), frac_increase=float((d > 0).mean()),
                                 p_two_sided=p))
            rows = pd.DataFrame(rows)
            rows["p_holm_within_K"] = holm(rows.p_two_sided)
            drift.append(rows)
    D = pd.concat(drift, ignore_index=True) if drift else pd.DataFrame()
    if not D.empty:
        D.to_csv(os.path.join(out, "confirm_occupancy_drift.csv"), index=False)

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(8.0, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.6, wspace=0.75)
    letters = iter("ABCDEF")
    for i, lab in enumerate(labels[:2]):
        ax = fig.add_subplot(gs[0, i])
        go, gr = data.get(lab, (None, None))
        offs = {"obs": -0.27, "null": -0.09, "rep": 0.09, "repnull": 0.27}
        for key, name, col in SERIES:
            src, cond = (go, "observed") if key in ("obs", "null") else (gr, "observed")
            if key in ("null", "repnull"):
                cond = "null"
            if src is None:
                continue
            Ks = sorted(src.K.unique())
            meds = []
            for K in Ks:
                v = series(src, cond, K).to_numpy(float)
                ax.scatter(K + offs[key] + rng.uniform(-0.04, 0.04, v.size), v, s=5,
                           color=col, alpha=0.4, lw=0)
                meds.append(np.median(v))
            ax.plot(np.array(Ks) + offs[key], meds, "o-", color=col, ms=3.5, lw=1.1,
                    mec="white", mew=0.5)
        ax.set_xticks(sorted(go.K.unique()))
        ax.set_xlabel("K", fontsize=8.5)
        ax.set_ylabel("Split-half similarity (demeaned)", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_title(f"PCA-HMM, rank {rank_of(lab)}", fontsize=8.5, loc="left")
        sns.despine(ax=ax)
        ax.text(-0.3, 1.14, next(letters), transform=ax.transAxes, fontsize=11,
                fontweight="bold")
    fig.legend(handles=[Line2D([], [], marker="o", color=c, ms=3.5, lw=1.1, label=n)
                        for _, n, c in SERIES],
               loc="lower center", bbox_to_anchor=(0.36, 0.975), ncol=4, frameon=False,
               fontsize=7)

    # calibrated forest
    ax = fig.add_subplot(gs[0, 2])
    y = np.arange(len(T))[::-1]
    for yi, (_, r) in zip(y, T.iterrows()):
        if np.isfinite(r.get("calibrated_median", np.nan)):
            ax.plot([r.ci_low, r.ci_high], [yi, yi], color=SECONDARY, lw=1.1)
            ax.plot(r.calibrated_median, yi, "o", ms=4.5,
                    mfc=INK if r.state_structure else "white", mec=INK, mew=1.0)
        ps = fmt_p(r.p_calibrated_holm)
        ax.text(1.04, yi, ("p" if ps.startswith("<") else "p=") + ps,
                transform=ax.get_yaxis_transform(),
                fontsize=6.3, va="center", color=INK if r.state_structure else SECONDARY)
    ax.axvline(0, color=BASELINE, lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"rank {rank_of(m)}, K={k}" for m, k in zip(T.model, T.K)],
                       fontsize=7)
    ax.set_xlabel("Calibrated difference\n(median, 95% CI)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("Pre-registered test (Holm p)", fontsize=8, loc="left")
    sns.despine(ax=ax)
    ax.text(-0.9, 1.14, next(letters), transform=ax.transAxes, fontsize=11, fontweight="bold")

    # drift forests (shared x-axis so the two models are comparable)
    bottom = gs[1, :].subgridspec(1, 2, wspace=0.8)
    drift_axes = []
    for i, lab in enumerate(labels[:2]):
        ax = fig.add_subplot(bottom[0, i], sharex=drift_axes[0] if drift_axes else None)
        drift_axes.append(ax)
        d = D[D.model == lab] if not D.empty else pd.DataFrame()
        if d.empty:
            ax.text(0.5, 0.5, "No occupancy file", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.axis("off")
            continue
        yy = np.arange(len(d))[::-1]
        occ = occ_all[lab]
        for yi, (_, r) in zip(yy, d.iterrows()):
            sk = occ[(occ.K == r.K) & (occ.state == r.state)]
            w = sk.pivot_table(index="subject_id", columns="quarter", values="fo_quarter")
            ch = ((w[3] + w[4]) / 2 - (w[1] + w[2]) / 2).to_numpy()
            med, lo, hi = median_ci(ch, rng, 4000)
            sig = r.p_holm_within_K < ALPHA
            ax.plot([lo, hi], [yi, yi], color=SECONDARY, lw=1.1)
            ax.plot(med, yi, "o", ms=4.5, mfc=INK if sig else "white", mec=INK, mew=1.0)
        ax.axvline(0, color=BASELINE, lw=0.8, zorder=0)
        ax.set_yticks(yy)
        ax.set_yticklabels([f"K={k}, state {s} ({100 * f:.0f}%)"
                            for k, s, f in zip(d.K, d.state, d.fo_total)], fontsize=6.5)
        ax.set_xlabel("Occupancy, 2nd − 1st scan half", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"Drift, rank {rank_of(lab)} (filled: Holm p < 0.05)",
                     fontsize=8, loc="left")
        sns.despine(ax=ax)
        ax.text(-0.55, 1.06, next(letters), transform=ax.transAxes, fontsize=11,
                fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"Fig_confirm_lowrank.{ext}"), bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
    plt.close(fig)

    # ---------------- console ----------------
    pd.set_option("display.width", 220)
    cols = ["model", "K", "n_paired", "delta_uncal", "p_uncal", "delta_replica",
            "calibrated_median", "ci_low", "ci_high", "p_calibrated", "p_calibrated_holm",
            "weakest_obs", "weakest_null", "state_structure"]
    print(f"\nPre-registered cells ({args.null} null):")
    print(T[[c for c in cols if c in T]].round(4).to_string(index=False))
    hits = T[T.state_structure]
    print("\nDECISION: " + ("state structure supported for " +
                            ", ".join(f"{m} K={k}" for m, k in zip(hits.model, hits.K))
                            if len(hits) else "no cell meets the pre-registered criterion"))
    if not D.empty:
        print("\nOccupancy drift (Holm within K):")
        print(D.round(4).to_string(index=False))
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
