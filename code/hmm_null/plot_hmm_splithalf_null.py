#!/usr/bin/env python3
"""
plot_hmm_splithalf_null.py
-------------------------------------------------------------------
Manuscript summary of the HMM split-half K selection against stationary
surrogates (output folder of run4b_hmm_splithalf_null.m, e.g.
hmm_raw_timeseries_null/), optionally with the zeromean = 0 activation-state
diagnostic (hmm_raw_timeseries_null_zeromean0/).

What the figure shows
  A  Raw split-half similarity of state FC for the primary model, per K:
     observed splits vs phase-randomised and IAAFT surrogate fits. High raw
     reproducibility is reached by stationary surrogates too, because it is
     dominated by the FC every state shares.
  B  The same after removing the shared FC (demeaned state FC): the part
     that would distinguish states.
  C  (with --zeromean0_dir) split-half similarity of the state MEAN maps
     (activation states) for the primary model, observed vs nulls.
  D  Every model x K: median observed - median null (demeaned), for each
     null. Marks: dot = one-sided Mann-Whitney p < 0.05 (uncorrected);
     star = p < 0.05 / (number of K), Bonferroni across K within the model.
     Rows marked with a dagger use means from the K-selection table because
     the per-split file was missing.

Per-split values are recomputed from split_half_per_state*.csv (mean over
matched states, as the MATLAB code scores a split), so median deltas and
Mann-Whitney p do not depend on which columns an older run wrote. If a
per-state file is missing, the K-selection CSV is used instead.

Usage
    python plot_hmm_splithalf_null.py --in_dir hmm_raw_timeseries_null \
        --zeromean0_dir hmm_raw_timeseries_null_zeromean0 --primary gaussian_pca_25
Outputs (in --out_dir, default <in_dir>/figures):
    Fig_hmm_splithalf_null.{pdf,png}
    table_hmm_splithalf_null.csv    every model x K x null: medians, delta, p
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ---------------- style (same calls as the static FC figures) ----------------
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 1.2,
                     "savefig.dpi": 350, "savefig.facecolor": "white",
                     "figure.facecolor": "white"})
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
warnings.filterwarnings("ignore", message=".*findfont.*")

INK, SECONDARY, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"observed": ("Observed", INK),
          "phase": ("Phase-randomised null", "#2a78d6"),
          "iaaft": ("IAAFT null", "#eb6834")}
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#1f5fae", "#2a78d6", "#f0efec", "#e34948", "#b3262f"])
SEED = 20260929


# ---------------- data ----------------
def model_label(m):
    g = re.match(r"gaussian_pca_(\d+)", m)
    if g:
        return f"Gaussian, PCA-{g.group(1)}"
    g = re.match(r"lowrank_(\d+)", m)
    if g:
        return f"PCA-HMM, rank {g.group(1)}"
    if m == "gaussian_full":
        return "Gaussian, all ROIs"
    return m


def model_order(models, primary):
    def key(m):
        g = re.match(r"(gaussian_pca|lowrank)_(\d+)", m)
        if m == primary:
            return (0, 0, 0)
        if g:
            return (1 if g.group(1) == "gaussian_pca" else 2, int(g.group(2)), 0)
        return (3, 0, 0)
    return sorted(models, key=key)


def per_split_scores(path):
    """Per (condition, repetition, K, metric): mean of the matched-state
    similarities, exactly as hmm_select_k_splithalf_null.m scores a split.
    'null' must not be read as missing."""
    d = pd.read_csv(path, keep_default_na=False, na_values=[""])
    d["condition"] = d["condition"].astype(str)
    g = d.groupby(["condition", "repetition", "K", "metric"], as_index=False).similarity.mean()
    return g.rename(columns={"similarity": "score"})


def load_folder(folder):
    """Returns {model: {null: scores DataFrame or None}} and {model: {null: K table}}."""
    scores, tables = {}, {}
    for sub in sorted(glob.glob(os.path.join(folder, "K_selection_*"))):
        # Model results are sub-FOLDERS; run4b also writes a top-level
        # K_selection_summary.csv, which is not a model.
        if not os.path.isdir(sub) or not glob.glob(os.path.join(sub, "split_half_*.csv")):
            continue
        m = os.path.basename(sub).replace("K_selection_", "")
        scores[m], tables[m] = {}, {}
        for null, sfx in (("phase", ""), ("iaaft", "_iaaft")):
            ps = os.path.join(sub, f"split_half_per_state{sfx}.csv")
            ks = os.path.join(sub, f"split_half_K_selection{sfx}.csv")
            scores[m][null] = per_split_scores(ps) if os.path.isfile(ps) else None
            tables[m][null] = pd.read_csv(ks) if os.path.isfile(ks) else None
    return scores, tables


def mw_greater(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan
    return float(stats.mannwhitneyu(x, y, alternative="greater").pvalue)


def summarise(scores, tables, metric):
    rows = []
    for m in scores:
        for null in ("phase", "iaaft"):
            S, T = scores[m][null], tables[m][null]
            if S is not None:
                s = S[S.metric == metric]
                for K, sk in s.groupby("K"):
                    o = sk[sk.condition == "observed"].score
                    n = sk[sk.condition == "null"].score
                    if len(o) == 0 or len(n) == 0:
                        continue
                    rows.append(dict(model=m, null=null, K=int(K), metric=metric,
                                     n_obs=len(o), n_null=len(n),
                                     median_observed=o.median(), median_null=n.median(),
                                     delta_median=o.median() - n.median(),
                                     p_mw=mw_greater(o, n), source="per-split"))
            elif T is not None:
                col = {"raw": ("mean_reproducibility", "null_mean_raw"),
                       "demeaned": ("mean_demeaned", "null_mean_demeaned"),
                       "meanmap": ("mean_meanmap", "null_mean_meanmap")}[metric]
                if col[0] not in T:
                    continue
                pcol = {"demeaned": "p_mw_demeaned", "meanmap": "p_mw_meanmap"}.get(metric)
                for _, r in T.iterrows():
                    rows.append(dict(model=m, null=null, K=int(r.K), metric=metric,
                                     n_obs=r.get("n_successful_splits", np.nan),
                                     n_null=r.get("n_successful_null", np.nan),
                                     median_observed=r[col[0]], median_null=r[col[1]],
                                     delta_median=r[col[0]] - r[col[1]],
                                     p_mw=r[pcol] if pcol and pcol in T else np.nan,
                                     source="K-selection table (means)"))
    out = pd.DataFrame(rows)
    if not out.empty:
        nK = out.groupby(["model", "null"]).K.transform("nunique")
        out["bonferroni_across_K"] = out.p_mw < 0.05 / nK
    return out


# ---------------- plotting ----------------
def strip_panel(ax, scores_m, metric, title, ylabel, rng):
    """Per-split points and medians for observed and each null, by K."""
    offsets = {"observed": -0.24, "phase": 0.0, "iaaft": 0.24}
    Ks = set()
    for null in ("phase", "iaaft"):
        S = scores_m.get(null)
        if S is not None:
            Ks |= set(S.loc[S.metric == metric, "K"].unique())
    Ks = sorted(Ks)
    if not Ks:
        ax.text(0.5, 0.5, "No per-split data", ha="center", va="center", transform=ax.transAxes)
        return
    for series in ("observed", "phase", "iaaft"):
        src = scores_m.get("phase" if series == "observed" else series)
        if series == "observed" and src is None:
            src = scores_m.get("iaaft")
        if src is None:
            continue
        cond = "observed" if series == "observed" else "null"
        s = src[(src.metric == metric) & (src.condition == cond)]
        lab, col = SERIES[series]
        meds = []
        for K in Ks:
            v = s.loc[s.K == K, "score"].to_numpy(float)
            x = K + offsets[series] + rng.uniform(-0.07, 0.07, v.size)
            ax.scatter(x, v, s=7, color=col, alpha=0.45, lw=0, zorder=2)
            meds.append(np.median(v) if v.size else np.nan)
        ax.plot(np.array(Ks) + offsets[series], meds, "-", color=col, lw=1.2, zorder=3)
        ax.plot(np.array(Ks) + offsets[series], meds, "o", ms=3.5, color=col,
                mec="white", mew=0.5, zorder=4, label=lab)
    ax.set_xticks(Ks)
    ax.set_xlabel("Number of states, K", fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8, loc="left")
    sns.despine(ax=ax)


def heat_panel(ax, summ, null, models, Ks, vmax, show_y):
    s = summ[summ.null == null]
    M = np.full((len(models), len(Ks)), np.nan)
    P = np.full_like(M, np.nan)
    B = np.zeros_like(M, dtype=bool)
    for i, m in enumerate(models):
        for j, K in enumerate(Ks):
            r = s[(s.model == m) & (s.K == K)]
            if len(r):
                M[i, j] = r.delta_median.iloc[0]
                P[i, j] = r.p_mw.iloc[0]
                B[i, j] = bool(r.bonferroni_across_K.iloc[0])
    im = ax.imshow(M, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6, color=MUTED)
                continue
            mark = "*" if B[i, j] else ("•" if P[i, j] < 0.05 else "")
            ax.text(j, i, f"{M[i, j]:.2f}{mark}", ha="center", va="center", fontsize=5.8,
                    color="white" if abs(M[i, j]) > 0.6 * vmax else INK)
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels(Ks, fontsize=7)
    ax.set_xlabel("K", fontsize=8.5)
    ax.set_yticks(range(len(models)))
    fb = set(summ.loc[summ.source != "per-split", "model"])
    ax.set_yticklabels([model_label(m) + (" †" if m in fb else "") for m in models]
                       if show_y else [], fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(SERIES[null][0], fontsize=8, loc="left")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="hmm_raw_timeseries_null")
    ap.add_argument("--zeromean0_dir", default=None)
    ap.add_argument("--primary", default="gaussian_pca_25")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = ap.parse_args()
    out = args.out_dir or os.path.join(args.in_dir, "figures")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    scores, tables = load_folder(args.in_dir)
    if not scores:
        raise SystemExit(f"No K_selection_* folders in {args.in_dir}")
    models = model_order(list(scores), args.primary)
    primary = args.primary if args.primary in scores else models[0]
    print(f"Models: {', '.join(models)} (primary: {primary})")
    summ = pd.concat([summarise(scores, tables, "raw"),
                      summarise(scores, tables, "demeaned")], ignore_index=True)
    zsumm = None
    zscores = None
    if args.zeromean0_dir:
        zscores, ztables = load_folder(args.zeromean0_dir)
        zsumm = summarise(zscores, ztables, "meanmap")
        zsumm["run"] = "zeromean0"
        dz = summarise(zscores, ztables, "demeaned")
        dz["run"] = "zeromean0"
        zsumm = pd.concat([zsumm, dz], ignore_index=True)
    table = summ.assign(run="zeromean1")
    if zsumm is not None:
        table = pd.concat([table, zsumm], ignore_index=True)
    table.to_csv(os.path.join(out, "table_hmm_splithalf_null.csv"), index=False)

    # ---------------- figure ----------------
    have_d = zscores is not None and primary in zscores
    fig = plt.figure(figsize=(8.0, 7.2))
    gs = fig.add_gridspec(2, 3 if have_d else 2, height_ratios=[1.0, 1.15],
                          hspace=0.55, wspace=0.42)
    top = [fig.add_subplot(gs[0, i]) for i in range(3 if have_d else 2)]
    strip_panel(top[0], scores[primary], "raw", f"{model_label(primary)}: state FC",
                "Split-half similarity (raw)", rng)
    strip_panel(top[1], scores[primary], "demeaned", "Shared FC removed",
                "Split-half similarity (demeaned)", rng)
    if have_d:
        strip_panel(top[2], zscores[primary], "meanmap", "Activation states (zeromean = 0)",
                    "Split-half similarity (state means)", rng)
    for ax, letter in zip(top, "ABC"):
        ax.text(-0.3, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="-", color=SERIES[k][1], ms=3.5, lw=1.2,
                      label=SERIES[k][0]) for k in ("observed", "phase", "iaaft")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.965), ncol=3,
               frameon=False, fontsize=7.5)

    # C: heatmaps
    Ks = sorted(summ.K.unique())
    dm = summ[summ.metric == "demeaned"]
    vmax = max(0.05, float(np.nanpercentile(np.abs(dm.delta_median), 98)))
    sub = gs[1, :].subgridspec(1, 2, wspace=0.06)
    axl = fig.add_subplot(sub[0, 0])
    axr = fig.add_subplot(sub[0, 1])
    heat_panel(axl, dm, "phase", models, Ks, vmax, True)
    im = heat_panel(axr, dm, "iaaft", models, Ks, vmax, False)
    cb = fig.colorbar(im, ax=[axl, axr], fraction=0.025, pad=0.015)
    cb.set_label("Median observed − median null\n(demeaned similarity)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    axl.text(-0.42, 1.06, "D" if have_d else "C", transform=axl.transAxes, fontsize=11,
             fontweight="bold")
    nK = len(Ks)
    note = (f"• one-sided Mann–Whitney p < 0.05 (uncorrected);  "
            f"* p < 0.05/{nK} (Bonferroni across K)")
    if (dm.source != "per-split").any():
        note += ";  † means from the K-selection table (no per-split file)"
    axl.text(0.0, -0.2, note, transform=axl.transAxes, fontsize=6.5, color=SECONDARY, va="top")

    for ext in args.formats:
        fig.savefig(os.path.join(out, f"Fig_hmm_splithalf_null.{ext}"), bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
    plt.close(fig)

    # ---------------- console summary ----------------
    pd.set_option("display.width", 200)
    d = dm.pivot_table(index=["model", "K"], columns="null",
                       values=["delta_median", "p_mw"]).round(3)
    print("\nDemeaned FC: median observed - median null, one-sided Mann-Whitney p")
    print(d.to_string())
    both = dm.pivot_table(index=["model", "K"], columns="null", values="bonferroni_across_K",
                          aggfunc="first")
    if {"phase", "iaaft"} <= set(both.columns):
        ok = both[(both.phase == True) & (both.iaaft == True)]  # noqa: E712
        print(f"\nModel x K cells passing Bonferroni across K under BOTH nulls: "
              f"{len(ok)}" + (f" -> {list(ok.index)}" if len(ok) else ""))
    if zsumm is not None:
        z = zsumm[(zsumm.metric == "meanmap") & (zsumm.model == primary)]
        print(f"\nActivation states ({primary}, zeromean = 0): min p = "
              f"{z.p_mw.min():.3g} over K and nulls")
    src = table.source.value_counts().to_dict()
    print(f"\nValues from: {src}")
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
