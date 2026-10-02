#!/usr/bin/env python3
"""
plot_dfc_results.py
-------------------------------------------------------------------
Manuscript figures and tables for the window-free dynamic FC analysis
(output folder of run_dcc_cpca_null.m: DCC/GARCH on group-PCA components
and complex PCA, each tested against the subject's own stationary
surrogates, with the stationary-replica calibration).

Styling matches the static FC figures (full_analysis_abeta_mice.py,
Figure 6): same GROUP_COLORS, the same rcParams + seaborn theme calls in
the same order, the same box/point styling and per-axis text sizes. Group
statistics follow the rest of the study and use the same module
(welch_tests_scipy.py, SciPy >= 1.16): Welch ANOVA across the 7 groups with
permutation p (scipy permutation_test of f_oneway(equal_var=False); Holm
across the panels of a figure, shown in each title), and the same 8 planned
contrasts as ordinary two-sided Welch t tests (scipy ttest_ind,
equal_var=False) with pooled BH-FDR. Corrections: statsmodels multipletests.

Inputs (all written by run_dcc_cpca_null.m into one folder)
    dfc_subject_metrics.csv          per subject x metric: observed, z, p
    dfc_group_tests_calibrated.csv   observed z vs stationary-replica z
    dfc_group_tests.csv              observed vs surrogate, per null
    cpca_components_{roi,pca}.csv    explained variance, refit checks
    cpca_loadings_{roi,pca_in_roi}.csv
    cpca_cycle_{roi,pca}_c{k}.csv    one propagation cycle per component
    pca_explained.csv, dfc_config.txt

Figures (in --out_dir; PDF with editable text, PNG, optional TIFF)
    Fig_dfc_nonstationarity    calibrated test: observed vs replica z (A),
                               median difference with 95% CI (B)
    Fig_dfc_group_z            z by group, 3 x 3 grid in Figure 6 layout
    Fig_cpca_propagation_roi   complex PCA: loadings (polar), lags, cycle
    Fig_cpca_static_by_group   static cPCA measures by group (descriptive)
    FigS_cpca_propagation_pca  same as above for the PCA-25 space
    FigS_cpca_roi_lags         every ROI's lag, per component (labelled)
    FigS_dfc_null_agreement    phase vs IAAFT null; share of mice p <= 0.05
    FigS_pca_cpca_checks       PCA scree, cPCA variance, ROI vs PCA space
Tables
    Tables_dfc.docx            formatted tables + figure notes
    table_*.csv                complete numeric results

Usage
    python plot_dfc_results.py --in_dir dfc_dcc_cpca
    python plot_dfc_results.py --in_dir dfc_dcc_cpca --roi_labels roi_labels.csv \
        --lag_covariate lag_covariate.csv --formats pdf png tiff

    roi_labels.csv   columns: region (1..N, atlas order), label
    lag_covariate.csv  output of compute_lag_covariate.py (columns
                     subject_id, coupling_magnitude, ...). If given, the lag
                     index is adjusted for coupling magnitude (pre-specified
                     covariate); otherwise it is shown unadjusted and
                     labelled as such. Do not use zero-lag or thresholded FC
                     strength here (see compute_lag_covariate.py).
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests

from welch_tests_scipy import (welch_anova_df, welch_anova_permutation,  # study's Welch tests
                               welch_t_test_df)

# ============================================================
# CONFIGURATION
# ============================================================
GROUP_ORDER_PREF = ["NTG_Y", "NTG_M", "NTG_A",
                    "CRND8_Y", "CRND8_M", "5XFAD_Y", "5XFAD_M"]

# Must be identical to GROUP_COLORS in full_analysis_abeta_mice.py.
GROUP_COLORS = {
    "NTG_Y": "#e06666",
    "NTG_M": "#a73dd9",
    "NTG_A": "#a64d79",
    "CRND8_Y": "#3d85c6",
    "CRND8_M": "#38761d",
    "5XFAD_Y": "#8fce00",
    "5XFAD_M": "#d4b75d",
}
# Groups not listed above (e.g. test data) take these in fixed order.
FALLBACK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# The same 8 planned contrasts as every other analysis in the study.
PAIRWISE_COMPARISONS = [
    ("NTG_Y", "NTG_M"), ("NTG_Y", "NTG_A"), ("NTG_Y", "CRND8_Y"),
    ("NTG_M", "CRND8_M"), ("NTG_Y", "5XFAD_Y"), ("NTG_M", "5XFAD_M"),
    ("CRND8_Y", "CRND8_M"), ("5XFAD_Y", "5XFAD_M"),
]
FDR_POOL = "family"      # "family": BH over all contrasts x metrics of a figure
                         # "metric": BH over the 8 contrasts of each metric
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000
RANDOM_SEED = 20260920
ALPHA = 0.05

# Pre-specified non-stationarity measures (calibrated test; Holm family).
PRIMARY_TESTS = ["dcc_a", "dcc_LR", "corr_fluct",
                 "garch_alpha", "garch_LR", "vol_cv",
                 "cpca_roi_c1_env_cv"]

# Figure 6-style grid of z by group: rows x columns.
GROUP_Z_GRID = [["dcc_a", "dcc_LR", "corr_fluct"],
                ["garch_alpha", "garch_LR", "vol_cv"],
                ["cpca_roi_c1_env_cv", "cpca_roi_c2_env_cv", "cpca_roi_c3_env_cv"]]

# Static complex-PCA measures by group (observed values, not z).
STATIC_GRID = [["cpca_roi_c1_var_expl", "cpca_roi_c2_var_expl", "cpca_roi_c3_var_expl"],
               ["cpca_roi_c1_freq_hz", "cpca_roi_lag_index", None]]

FAMILY = {"dcc": "Coupling (DCC)", "garch": "Amplitude (GARCH)",
          "cpca": "Propagation (complex PCA)"}

# Neutral inks and the diverging map (blue <-> red, grey midpoint).
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
REPLICA_FILL = "#e1e0d9"
NULL_COLORS = {"phase": "#2a78d6", "iaaft": "#eb6834"}
SPACE_COLORS = {"roi": "#2a78d6", "pca": "#eb6834"}
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#1f5fae", "#2a78d6", "#f0efec", "#e34948", "#b3262f"])

# Figure 6 geometry and styling.
PANEL_SIZE = 8 / 3
BOX_WIDTH = 0.56
BOX_LINEWIDTH = 1.1
JITTER_WIDTH = 0.14
POINT_SIZE = 4.5 ** 2
POINT_ALPHA = 0.8
POINT_EDGEWIDTH = 0.35
TITLE_FONTSIZE = 8
YLABEL_FONTSIZE = 11
XTICK_FONTSIZE = 9
YTICK_FONTSIZE = 10
PNG_DPI = 600
TIFF_DPI = 600

# Same calls, in the same order, as full_analysis_abeta_mice.py. The
# seaborn theme runs last there and overrides most of the rcParams above it
# (paper-context sizes; Arial if installed, otherwise DejaVu Sans).
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


# ============================================================
# LABELS
# ============================================================
def parse_cpca(metric):
    """'cpca_roi_c2_env_cv' -> ('roi', 2, 'env_cv'); lag index -> k = None."""
    parts = metric.split("_")
    if parts[0] != "cpca":
        return None
    space = parts[1]
    if parts[2].startswith("c") and parts[2][1:].isdigit():
        return space, int(parts[2][1:]), "_".join(parts[3:])
    return space, None, "_".join(parts[2:])


def family_of(metric):
    if metric.startswith("dcc") or metric == "corr_fluct":
        return "dcc"
    if metric.startswith("garch") or metric == "vol_cv":
        return "garch"
    return "cpca"


def short_label(metric):
    fixed = {"dcc_a": "DCC a", "dcc_LR": "DCC LR", "corr_fluct": "Corr. fluctuation",
             "garch_alpha": "GARCH α", "garch_LR": "GARCH LR",
             "vol_cv": "Volatility CV", "dcc_b": "DCC b",
             "dcc_persist": "DCC a + b", "garch_persist": "GARCH α + β"}
    if metric in fixed:
        return fixed[metric]
    c = parse_cpca(metric)
    if c is None:
        return metric
    space, k, what = c
    tag = "" if space == "roi" else " (PCA space)"
    name = {"env_cv": "Envelope CV", "var_expl": "Variance expl.",
            "freq_hz": "Frequency", "lag_index": "Lag index"}.get(what, what)
    return f"{name}, C{k}{tag}" if k else f"{name}{tag}"


def long_label(metric):
    if metric.endswith("_adj"):
        return long_label(metric[:-4]) + " (adjusted)"
    fixed = {"dcc_a": "DCC news parameter a",
             "dcc_LR": "DCC likelihood ratio",
             "corr_fluct": "Correlation fluctuation",
             "garch_alpha": "GARCH α (mean)",
             "garch_LR": "GARCH likelihood ratio",
             "vol_cv": "Volatility CV"}
    if metric in fixed:
        return fixed[metric]
    c = parse_cpca(metric)
    if c is None:
        return metric
    space, k, what = c
    tag = "" if space == "roi" else " (PCA space)"
    if what == "env_cv":
        return f"Envelope CV, comp. {k}{tag}"
    if what == "var_expl":
        return f"Var. explained, C{k} (%){tag}"
    if what == "freq_hz":
        return f"Frequency, C{k} (Hz){tag}"
    if what == "lag_index":
        return f"Lag index{tag}"
    return metric


# ============================================================
# STATISTICS
# ============================================================
def _adjust(p, method):
    """statsmodels multipletests on the finite p-values; NaN stays NaN."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method=method)[1]
    return out


def holm(p):
    return _adjust(p, "holm")


def bh(p):
    return _adjust(p, "fdr_bh")


def welch_anova_perm(values, groups, order, n_perm, seed):
    """Welch ANOVA across `order`; p by permuting group labels
    (welch_tests_scipy.welch_anova_permutation). Returns F, df1, df2, p
    (NaN if any group has < 2 values or no spread, as before)."""
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    keep = np.isfinite(values) & np.isin(groups, order)
    x = values[keep]
    lab = np.array([order.index(g) for g in groups[keep]])
    k = len(order)
    counts = np.bincount(lab, minlength=k)
    if k < 2 or counts.min() < 2:
        return np.nan, np.nan, np.nan, np.nan
    if any(np.var(x[lab == g], ddof=1) == 0 for g in range(k)):
        return np.nan, np.nan, np.nan, np.nan
    samples = [x[lab == g] for g in range(k)]
    F, p = welch_anova_permutation(samples, n_permutations=n_perm, random_seed=seed)
    df1, df2 = welch_anova_df(samples)
    return float(F), df1, df2, float(p)


def welch_t(a, b):
    """Ordinary two-sided Welch t test of a vs b (welch_tests_scipy.welch_t_test_df).
    Returns t, df, mean difference, p."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    t, df, p = welch_t_test_df(a, b)
    if not np.isfinite(t):
        return np.nan, np.nan, np.nan, np.nan
    return float(t), float(df), float(a.mean() - b.mean()), float(p)


def median_ci(x, n_boot, seed, level=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    med = np.median(x[rng.integers(0, x.size, size=(n_boot, x.size))], axis=1)
    lo, hi = np.percentile(med, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(np.median(x)), float(lo), float(hi)


def fmt_p(p, n_perm=None):
    if not np.isfinite(p):
        return "NA"
    if n_perm is not None and p <= 1.0 / (n_perm + 1) + 1e-15:
        return f"<{1.0 / n_perm:.1g}"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3g}"


def p_text(p, n_perm=None):
    s = fmt_p(p, n_perm)
    return "p" + s if s.startswith("<") else "p=" + s


def planned_pairs(groups_present):
    pairs = [c for c in PAIRWISE_COMPARISONS
             if c[0] in groups_present and c[1] in groups_present]
    if not pairs and len(groups_present) == 2:
        pairs = [tuple(groups_present)]
    return pairs


def group_tests(frame, metrics, value_col, groups, tag, n_perm):
    """Omnibus Welch ANOVA with permutation p (Holm across `metrics`) and
    planned contrasts as ordinary Welch t tests (BH pooled over contrasts x
    metrics, or per metric)."""
    omni, cons = [], []
    pairs = planned_pairs(groups)
    for j, m in enumerate(metrics):
        sub = frame[frame.metric == m]
        F, df1, df2, p = welch_anova_perm(sub[value_col], sub.group, groups,
                                          n_perm, RANDOM_SEED + j)
        omni.append(dict(figure=tag, metric=m, label=long_label(m),
                         value=value_col, n=int(sub[value_col].notna().sum()),
                         welch_F=F, df1=df1, df2=df2, p_perm=p))
        for ga, gb in pairs:
            t, df, diff, pc = welch_t(sub.loc[sub.group == ga, value_col],
                                      sub.loc[sub.group == gb, value_col])
            cons.append(dict(figure=tag, metric=m, label=long_label(m),
                             value=value_col, contrast=f"{ga} vs {gb}",
                             n_a=int(sub.loc[sub.group == ga, value_col].notna().sum()),
                             n_b=int(sub.loc[sub.group == gb, value_col].notna().sum()),
                             mean_diff=diff, welch_t=t, df=df, p_welch=pc))
    omni = pd.DataFrame(omni)
    if not omni.empty:
        omni["p_Holm"] = holm(omni.p_perm)
    cons = pd.DataFrame(cons, columns=["figure", "metric", "label", "value", "contrast",
                                       "n_a", "n_b", "mean_diff", "welch_t", "df", "p_welch"])
    cons["q_BH"] = np.nan
    if not cons.empty:
        if FDR_POOL == "metric":
            cons["q_BH"] = cons.groupby("metric").p_welch.transform(lambda s: bh(s.values))
        else:
            cons["q_BH"] = bh(cons.p_welch)
    return omni, cons


# ============================================================
# DATA
# ============================================================
def read_csv(path, required=True):
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(path)
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def read_config(path):
    cfg = {}
    if os.path.isfile(path):
        for line in open(path):
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


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


def load_all(in_dir):
    d = read_csv(os.path.join(in_dir, "dfc_subject_metrics.csv"))
    d["group"] = d["group"].astype(str).str.strip()
    d["metric"] = d["metric"].astype(str).str.strip()
    num = [c for c in d.columns if c not in ("group", "metric", "role")]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "phase_replica_z" not in d.columns:
        d["phase_replica_z"] = np.nan
    # Lag index: sqrt(sum Im^2 / sum |C|^2) cannot exceed 1 in expectation. The
    # split-half denominator can approach 0 when zero-lag coupling is absent,
    # which is always the case in group-PCA space (components are uncorrelated
    # at zero lag by construction), so only the ROI-space index is used.
    li = d.metric.str.endswith("lag_index") & (d.observed > 1 + 1e-6)
    if li.any():
        print(f"  {int(li.sum())} lag-index value(s) > 1 set to NaN "
              f"({', '.join(sorted(d.loc[li, 'metric'].unique()))})")
        d.loc[li, "observed"] = np.nan
    D = dict(sub=d)
    D["cal"] = read_csv(os.path.join(in_dir, "dfc_group_tests_calibrated.csv"), False)
    D["gt"] = read_csv(os.path.join(in_dir, "dfc_group_tests.csv"), False)
    D["pca"] = read_csv(os.path.join(in_dir, "pca_explained.csv"), False)
    D["cfg"] = read_config(os.path.join(in_dir, "dfc_config.txt"))
    for sp in ("roi", "pca"):
        D[f"comp_{sp}"] = read_csv(os.path.join(in_dir, f"cpca_components_{sp}.csv"), False)
        cyc = {}
        k = 1
        while os.path.isfile(os.path.join(in_dir, f"cpca_cycle_{sp}_c{k}.csv")):
            cyc[k] = read_csv(os.path.join(in_dir, f"cpca_cycle_{sp}_c{k}.csv"))
            k += 1
        D[f"cycle_{sp}"] = cyc
    D["load_roi"] = read_csv(os.path.join(in_dir, "cpca_loadings_roi.csv"), False)
    D["load_pca_in_roi"] = read_csv(os.path.join(in_dir, "cpca_loadings_pca_in_roi.csv"), False)
    return D


def roi_names(path, n):
    names = {i: f"ROI {i}" for i in range(1, n + 1)}
    if path:
        lab = read_csv(path)
        col = "label" if "label" in lab.columns else lab.columns[1]
        for r, l in zip(lab["region"], lab[col]):
            names[int(r)] = str(l)
    return names


def add_adjusted_lag(d, path, column="coupling_magnitude"):
    """Adds cpca_{space}_lag_index_adj: the lag index residualised on the
    per-mouse coupling magnitude (OLS across all mice), mean added back."""
    if not path:
        return d, False
    cov = read_csv(path)
    if column not in cov.columns:
        print(f"  {path} has no '{column}' column; using '{cov.columns[1]}'")
        column = cov.columns[1]
    cov = cov[["subject_id", column]].rename(columns={column: "_cov"})
    rows = []
    for sp in ("roi",):          # PCA-space lag index is not interpretable (see load_all)
        m = f"cpca_{sp}_lag_index"
        sub = d[d.metric == m].merge(cov, on="subject_id", how="left")
        ok = sub.observed.notna() & sub._cov.notna()
        if ok.sum() < 5:
            continue
        X = np.c_[np.ones(ok.sum()), sub.loc[ok, "_cov"]]
        beta, *_ = np.linalg.lstsq(X, sub.loc[ok, "observed"], rcond=None)
        adj = sub.observed - beta[1] * (sub._cov - sub.loc[ok, "_cov"].mean())
        new = sub.drop(columns="_cov").copy()
        new["metric"] = m + "_adj"
        new["observed"] = np.where(ok, adj, np.nan)
        rows.append(new)
        r = np.corrcoef(sub.loc[ok, "observed"], sub.loc[ok, "_cov"])[0, 1]
        print(f"  {m}: slope on {column} = {beta[1]:.3g}, r = {r:.2f} (n = {int(ok.sum())})")
    if rows:
        d = pd.concat([d] + rows, ignore_index=True)
    return d, bool(rows)


# ============================================================
# PLOTTING HELPERS
# ============================================================
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


def panel_label(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left", color=INK)


def symlog_axis(ax, which="y", linthresh=2):
    """Symmetric-log axis with plain-number ticks (0, ±2, ±5, ±10, ...)."""
    getattr(ax, f"set_{which}scale")("symlog", linthresh=linthresh, linscale=1)
    lo, hi = getattr(ax, f"get_{which}lim")()
    top = max(abs(lo), abs(hi), linthresh)
    steps = ((1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
             if np.log10(top / linthresh) < 1.3 else (1, 10, 100, 1000, 10000))
    cand = [0] + [s * v for v in steps for s in (1, -1)]
    ticks = sorted(v for v in cand if lo <= v <= hi and (abs(v) >= linthresh or v == 0))
    getattr(ax, f"set_{which}ticks")(ticks)
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−"))
    getattr(ax, f"{which}axis").set_major_formatter(fmt)
    getattr(ax, f"{which}axis").set_minor_locator(matplotlib.ticker.NullLocator())


def use_symlog(values, limit=12):
    v = np.abs(np.asarray(values, float))
    v = v[np.isfinite(v)]
    return v.size > 0 and np.percentile(v, 99) > limit


def box_strip(ax, frame, ycol, groups, palette, rng):
    """Figure 6 styling: seaborn box (no fliers) + reproducibly jittered
    subject points in the group colour with white edges."""
    frame = frame[frame["group"].isin(groups)].copy()
    frame["group"] = frame["group"].astype(str)
    sns.boxplot(data=frame, x="group", y=ycol, order=groups, hue="group",
                hue_order=groups, palette=palette, width=BOX_WIDTH,
                showfliers=False, linewidth=BOX_LINEWIDTH, legend=False,
                dodge=False, ax=ax)
    for i, g in enumerate(groups):
        y = frame.loc[frame.group == g, ycol].to_numpy(float)
        y = y[np.isfinite(y)]
        ax.scatter(i + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, y.size), y,
                   s=POINT_SIZE, facecolor=palette[g], edgecolor="white",
                   linewidth=POINT_EDGEWIDTH, alpha=POINT_ALPHA, zorder=3)


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


def pad_y(ax, values, lower=0.07, upper=0.12):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0 or ax.get_yscale() != "linear":
        return
    lo, hi = v.min(), v.max()
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    ax.set_ylim(lo - lower * span, hi + upper * span)


# ============================================================
# FIGURE 1: calibrated non-stationarity test
# ============================================================
def fig_nonstationarity(D, out_dir, formats, n_boot):
    d = D["sub"]
    tests = [m for m in pd.unique(d.loc[d.role == "test", "metric"])]
    prim = [m for m in PRIMARY_TESTS if m in tests]
    expl = [m for m in tests if m not in prim]
    calibrated = d.loc[d.role == "test", "phase_replica_z"].notna().any()
    cal = D["cal"].set_index("metric") if (calibrated and D["cal"] is not None) else None
    gt = D["gt"]
    rows = []
    for i, m in enumerate(prim + expl):
        sub = d[d.metric == m]
        zo = sub.phase_z.to_numpy(float)
        zr = sub.phase_replica_z.to_numpy(float)
        diff = zo - zr if calibrated else zo
        med, lo, hi = median_ci(diff, n_boot, RANDOM_SEED + i)
        if cal is not None and m in cal.index:
            p = float(cal.loc[m, "wilcoxon_p_greater"])
        elif gt is not None:
            r = gt[(gt.metric == m) & (gt.null == "phase")]
            p = float(r.wilcoxon_p_greater.iloc[0]) if len(r) else np.nan
        else:
            p = np.nan
        rows.append(dict(metric=m, label=long_label(m), family=FAMILY[family_of(m)],
                         prespecified=m in prim, n=int(np.isfinite(diff).sum()),
                         median_z_observed=np.nanmedian(zo),
                         median_z_replica=np.nanmedian(zr) if calibrated else np.nan,
                         median_difference=med, ci_low=lo, ci_high=hi,
                         p_wilcoxon_greater=p))
    T = pd.DataFrame(rows)
    T["p_Holm_prespecified"] = np.nan
    T.loc[T.prespecified, "p_Holm_prespecified"] = holm(T.loc[T.prespecified, "p_wilcoxon_greater"])
    T["test"] = ("one-sided Wilcoxon on z_observed - z_replica" if calibrated
                 else "one-sided Wilcoxon on observed - surrogate mean (UNCALIBRATED)")

    fig = plt.figure(figsize=(8.0, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.95)
    rng = np.random.default_rng(RANDOM_SEED)

    # ---- A: observed z vs stationary-replica z, pre-specified measures
    ax = fig.add_subplot(gs[0, 0])
    allz = []
    for i, m in enumerate(prim):
        sub = d[d.metric == m]
        series = [(-0.19 if calibrated else 0.0, sub.phase_z.to_numpy(float), "white", INK)]
        if calibrated:
            series.append((0.19, sub.phase_replica_z.to_numpy(float), REPLICA_FILL, MUTED))
        for off, v, face, pc in series:
            v = v[np.isfinite(v)]
            allz.append(v)
            ax.boxplot(v, positions=[i + off], widths=0.3, showfliers=False,
                       patch_artist=True, manage_ticks=False,
                       medianprops=dict(color=INK, lw=1.1),
                       boxprops=dict(facecolor=face, edgecolor=SECONDARY, lw=0.8),
                       whiskerprops=dict(color=SECONDARY, lw=0.8),
                       capprops=dict(color=SECONDARY, lw=0.8))
            ax.scatter(i + off + rng.uniform(-0.085, 0.085, v.size), v, s=6,
                       color=pc, alpha=0.55, lw=0, zorder=3)
    allz = np.concatenate(allz) if allz else np.array([0.0])
    if use_symlog(allz):
        symlog_axis(ax, "y", 2)
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=0)
    ax.set_xticks(range(len(prim)))
    ax.set_xticklabels([short_label(m) for m in prim], rotation=40, ha="right",
                       rotation_mode="anchor", fontsize=8)
    ax.set_xlim(-0.6, len(prim) - 0.4)
    ax.set_ylabel("z vs own phase surrogates", fontsize=9.6)
    ax.tick_params(axis="y", labelsize=8)
    fams = [family_of(m) for m in prim]
    for i in range(1, len(prim)):
        if fams[i] != fams[i - 1]:
            ax.axvline(i - 0.5, color=GRID, lw=0.8, zorder=0)
    tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "linear":
        ax.set_ylim(lo, hi + 0.14 * (hi - lo))
    else:
        ax.set_ylim(lo, hi * 3 if hi > 0 else hi)
        symlog_axis(ax, "y", 2)
    starts = [0] + [i for i in range(1, len(prim)) if fams[i] != fams[i - 1]] + [len(prim)]
    for a, b in zip(starts[:-1], starts[1:]):
        name = {"dcc": "DCC", "garch": "GARCH", "cpca": "cPCA"}[fams[a]]
        ax.text((a + b - 1) / 2, 0.985, name, transform=tr, ha="center",
                va="top", fontsize=8, color=SECONDARY)
    handles = [Patch(facecolor="white", edgecolor=SECONDARY, label="Observed")]
    if calibrated:
        handles.append(Patch(facecolor=REPLICA_FILL, edgecolor=SECONDARY,
                             label="Stationary replica"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.5,
              bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.2,
              borderaxespad=0.2)
    sns.despine(ax=ax)
    panel_label(ax, "A", x=-0.2, y=1.02)

    # ---- B: forest of median (z_obs - z_replica)
    ax = fig.add_subplot(gs[0, 1])
    order = prim + expl
    y = np.arange(len(order))[::-1].astype(float)
    if expl:
        y[len(prim):] -= 0.6
    for yi, m in zip(y, order):
        r = T[T.metric == m].iloc[0]
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=SECONDARY, lw=1.2,
                solid_capstyle="round")
        ax.plot(r.median_difference, yi, "o", ms=5,
                mfc=INK if r.prespecified else "white", mec=INK, mew=1.0, zorder=3)
        txt = p_text(r.p_wilcoxon_greater)
        if r.prespecified:
            txt += f" ({fmt_p(r.p_Holm_prespecified)})"
        ax.text(1.03, yi, txt, transform=matplotlib.transforms.blended_transform_factory(
            ax.transAxes, ax.transData), va="center", ha="left", fontsize=6.5,
            color=INK if r.prespecified else SECONDARY)
    ax.axvline(0, color=BASELINE, lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(m) for m in order], fontsize=7.5)
    for tl, m in zip(ax.get_yticklabels(), order):
        tl.set_color(INK if m in prim else SECONDARY)
    if expl:
        ax.axhline((y[len(prim) - 1] + y[len(prim)]) / 2, color=GRID, lw=0.8)
        ax.text(1.03, y[len(prim)] + 0.75, "Exploratory",
                transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                fontsize=6.5, color=SECONDARY, style="italic", va="center")
    ax.text(1.03, y[0] + 0.9, "p (Holm)",
            transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
            fontsize=6.5, color=SECONDARY, va="center")
    ax.set_ylim(y.min() - 0.7, y.max() + 1.3)
    xl = ("Median z$_{observed}$ − z$_{replica}$ (95% CI)" if calibrated
          else "Median z vs surrogates (95% CI)")
    ax.set_xlabel(xl, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    if use_symlog(T[["ci_low", "ci_high"]].to_numpy()):
        symlog_axis(ax, "x", 1)
    sns.despine(ax=ax)
    panel_label(ax, "B", x=-0.62, y=1.0)
    if not calibrated:
        fig.suptitle("UNCALIBRATED: no stationary-replica z in this run", color="#e34948",
                     fontsize=9, y=1.02)
    save(fig, out_dir, "Fig_dfc_nonstationarity", formats)
    return T, calibrated


# ============================================================
# FIGURE 2: z by group (Figure 6 layout)
# ============================================================
def fig_group_z(D, out_dir, formats, groups, palette, n_perm):
    d = D["sub"]
    grid = [[m for m in row if m in set(d.metric)] for row in GROUP_Z_GRID]
    grid = [r for r in grid if r]
    metrics = [m for r in grid for m in r]
    omni, cons = group_tests(d, metrics, "phase_z", groups, "Fig_dfc_group_z", n_perm)
    ncol = max(len(r) for r in grid)
    fig, axes = plt.subplots(len(grid), ncol, figsize=(PANEL_SIZE * ncol, PANEL_SIZE * len(grid)),
                             constrained_layout=True, squeeze=False)
    rng = np.random.default_rng(RANDOM_SEED)
    for r, row in enumerate(grid):
        for c in range(ncol):
            ax = axes[r, c]
            if c >= len(row):
                ax.axis("off")
                continue
            m = row[c]
            sub = d[d.metric == m]
            rep = sub.phase_replica_z.to_numpy(float)
            rep = rep[np.isfinite(rep)]
            if rep.size >= 10:
                lo, hi = np.percentile(rep, [2.5, 97.5])
                ax.axhspan(lo, hi, color=GRID, alpha=0.7, lw=0, zorder=0)
            ax.axhline(0, color=BASELINE, lw=0.8, zorder=0)
            box_strip(ax, sub, "phase_z", groups, palette, rng)
            if use_symlog(sub.phase_z):
                symlog_axis(ax, "y", 2)
            p = omni.loc[omni.metric == m, "p_Holm"].iloc[0]
            format_group_axis(ax, groups, f"{FAMILY[family_of(m)]}; {p_text(p, n_perm)}",
                              f"{long_label(m)} (z)")
            pad_y(ax, np.r_[sub.phase_z.to_numpy(float), rep])
    save(fig, out_dir, "Fig_dfc_group_z", formats)
    return omni, cons


# ============================================================
# FIGURE 3: complex PCA propagation
# ============================================================
def loadings_frame(D, space):
    L = D["load_roi"] if space == "roi" else D["load_pca_in_roi"]
    return None if L is None else L.copy()


def fig_cpca(D, out_dir, formats, space, stem, names):
    L = loadings_frame(D, space)
    cyc = D[f"cycle_{space}"]
    comp = D[f"comp_{space}"]
    if L is None or not cyc or comp is None:
        print(f"  skip {stem}: missing complex-PCA files for the {space} space")
        return
    K = min(int(L.component.max()), len(cyc))
    vmax = np.nanpercentile(np.abs(np.concatenate(
        [cyc[k].iloc[:, 1:].to_numpy(float).ravel() for k in range(1, K + 1)])), 99)
    fig = plt.figure(figsize=(8.0, 2.55 * K))
    gs = fig.add_gridspec(K, 5, width_ratios=[1.0, 0.3, 1.2, 0.1, 1.55],
                          wspace=0.12, hspace=0.75, left=0.06, right=0.99)
    letters = iter("ABCDEFGHIJKLMNOP")
    for k in range(1, K + 1):
        lk = L[L.component == k].sort_values("region").reset_index(drop=True)
        cr = comp[comp.component == k].iloc[0]
        f = float(cr.median_freq_hz)
        row = k - 1

        # ---- polar loadings (angle = phase, radius = magnitude)
        ax = fig.add_subplot(gs[row, 0], projection="polar")
        ax.scatter(lk.phase_rad, lk.magnitude, s=9, color=INK, alpha=0.7, lw=0)
        ax.set_rlabel_position(135)
        ax.tick_params(labelsize=6.5, pad=-1)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["0", "π/2", "π", "−π/2"])
        ax.grid(color=GRID, lw=0.6)
        ax.spines["polar"].set_color(BASELINE)
        ax.spines["polar"].set_linewidth(0.6)
        rmax = lk.magnitude.max() * 1.1
        ax.set_ylim(0, rmax)
        ax.set_yticks(np.round(np.linspace(0, rmax, 3)[1:], 2))
        for _, r in lk.nlargest(3, "magnitude").iterrows():
            nm = names[int(r.region)]
            if not nm.startswith("ROI "):
                ax.annotate(nm, (r.phase_rad, r.magnitude), fontsize=5.5, color=SECONDARY,
                            xytext=(3, 2), textcoords="offset points")
        panel_label(ax, next(letters), x=-0.12, y=1.02)
        # row header: component, variance, frequency
        pos = ax.get_position()
        fig.text(pos.x0 - 0.02, pos.y1 + 0.028,
                 f"Component {k}   ·   {100 * cr.explained:.1f}% of analytic variance   ·   "
                 f"{f:.3f} Hz (period {1 / f:.0f} s)",
                 fontsize=8.5, color=INK, ha="left", va="bottom")

        # ---- lags, sorted; marker size by |loading|
        ax = fig.add_subplot(gs[row, 2])
        srt = lk.sort_values("lag_sec").reset_index(drop=True)
        strong = (srt.magnitude >= srt.magnitude.median()).to_numpy()
        size = (4 + 40 * (srt.magnitude / srt.magnitude.max()) ** 2).to_numpy()
        xs = np.arange(len(srt))
        ax.scatter(xs[~strong], srt.lag_sec[~strong], s=size[~strong], color=MUTED,
                   alpha=0.6, lw=0, label="|loading| < median")
        ax.scatter(xs[strong], srt.lag_sec[strong], s=size[strong], color=INK,
                   alpha=0.85, lw=0, label="|loading| ≥ median")
        ax.axhline(0, color=BASELINE, lw=0.8, zorder=0)
        ax.set_xlabel("ROIs, sorted by lag", fontsize=8)
        ax.set_ylabel("Lag (s)", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-2, len(srt) + 1)
        st = srt[strong]
        # name the earliest and latest strong ROIs in the empty lower-right corner
        e_names = [names[int(r)] for r in st.region.head(3)]
        l_names = [names[int(r)] for r in st.region.tail(3)[::-1]]
        if not all(n.startswith("ROI ") for n in e_names + l_names):
            ax.text(0.98, 0.03, "Earliest: " + ", ".join(e_names) + "\nLatest: "
                    + ", ".join(l_names), transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=5.5, color=SECONDARY, linespacing=1.4)
        lo, hi = srt.lag_sec.min(), srt.lag_sec.max()
        ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))
        if row == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=6, handletextpad=0.1,
                      markerscale=0.8, borderaxespad=0.1)
        sns.despine(ax=ax)
        panel_label(ax, next(letters), x=-0.3, y=1.02)

        # ---- one propagation cycle, ROIs sorted by phase (earliest at top)
        c = cyc[k]
        th = c.iloc[:, 0].to_numpy(float)
        M = c.iloc[:, 1:].to_numpy(float)
        order = lk.sort_values("phase_rad").region.to_numpy() - 1
        axm = fig.add_subplot(gs[row, 3])
        axm.imshow(lk.magnitude.to_numpy()[order][:, None], aspect="auto", cmap="Greys",
                   vmin=0, vmax=lk.magnitude.max(), interpolation="nearest")
        axm.set_xticks([0])
        axm.set_xticklabels(["|v|"], fontsize=6.5)
        axm.tick_params(axis="x", length=0)
        axm.set_yticks([])
        axm.set_ylabel("ROIs, sorted by phase", fontsize=8)
        for s in axm.spines.values():
            s.set_visible(False)
        pm = axm.get_position()
        fig.text(pm.x0 - 0.045, pm.y1 + 0.004, next(letters), fontsize=11,
                 fontweight="bold", va="bottom", ha="left", color=INK)
        ax = fig.add_subplot(gs[row, 4])
        step = th[1] - th[0]
        im = ax.imshow(M[:, order].T, aspect="auto", cmap=DIVERGING, vmin=-vmax, vmax=vmax,
                       extent=[th[0] - step / 2, th[-1] + step / 2, len(order) - 0.5, -0.5],
                       interpolation="nearest")
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels(["−π", "−π/2", "0", "π/2", "π"])
        ax.set_xlabel("Component phase (rad)", fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label("Mean BOLD (SD)", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_visible(False)
    save(fig, out_dir, stem, formats)


def fig_roi_lags(D, out_dir, formats, names):
    L = D["load_roi"]
    if L is None:
        return
    K = int(L.component.max())
    n = int(L.region.max())
    fig, axes = plt.subplots(1, K, figsize=(2.65 * K, max(3.0, 0.085 * n + 0.8)),
                             gridspec_kw=dict(wspace=0.85))
    axes = np.atleast_1d(axes)
    for k, ax in zip(range(1, K + 1), axes):
        lk = L[L.component == k].sort_values("lag_sec").reset_index(drop=True)
        yy = np.arange(len(lk))[::-1]
        strong = lk.magnitude >= lk.magnitude.median()
        ax.barh(yy, lk.lag_sec, height=0.7, color=np.where(strong, INK, BASELINE), lw=0)
        ax.axvline(0, color=SECONDARY, lw=0.6)
        ax.set_yticks(yy)
        ax.set_yticklabels([names[int(r)] for r in lk.region], fontsize=4.8)
        ax.set_ylim(-0.8, len(lk) - 0.2)
        ax.tick_params(axis="y", length=0, pad=1)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("Lag (s)", fontsize=8)
        ax.set_title(f"Component {k}", fontsize=9, loc="left")
        sns.despine(ax=ax, left=True)
    fig.legend(handles=[Patch(color=INK, label="|loading| ≥ median"),
                        Patch(color=BASELINE, label="|loading| < median")],
               loc="lower center", bbox_to_anchor=(0.5, 0.995), frameon=False,
               fontsize=7, ncol=2)
    save(fig, out_dir, "FigS_cpca_roi_lags", formats)


# ============================================================
# FIGURE 4: static cPCA measures by group
# ============================================================
def fig_static(D, out_dir, formats, groups, palette, n_perm, adjusted):
    d = D["sub"].copy()
    grid = [list(r) for r in STATIC_GRID]
    if adjusted:
        grid = [[("cpca_roi_lag_index_adj" if m == "cpca_roi_lag_index" else m) for m in r]
                for r in grid]
    metrics = [m for r in grid for m in r if m is not None and m in set(d.metric)]
    for m in metrics:
        if m.endswith("var_expl"):
            d.loc[d.metric == m, "observed"] *= 100
    omni, cons = group_tests(d, metrics, "observed", groups, "Fig_cpca_static_by_group", n_perm)
    ncol = max(len(r) for r in grid)
    fig, axes = plt.subplots(len(grid), ncol, figsize=(PANEL_SIZE * ncol, PANEL_SIZE * len(grid)),
                             constrained_layout=True, squeeze=False)
    rng = np.random.default_rng(RANDOM_SEED)
    for r, row in enumerate(grid):
        for c in range(ncol):
            ax = axes[r, c]
            m = row[c] if c < len(row) else None
            if m is None or m not in metrics:
                ax.axis("off")
                continue
            sub = d[d.metric == m]
            box_strip(ax, sub, "observed", groups, palette, rng)
            p = omni.loc[omni.metric == m, "p_Holm"].iloc[0]
            lab = long_label(m.replace("_adj", ""))
            if "lag_index" in m:
                lab += " (adjusted)" if m.endswith("_adj") else " (unadjusted)"
            format_group_axis(ax, groups, f"Static; {p_text(p, n_perm)}", lab)
            pad_y(ax, sub.observed)
    save(fig, out_dir, "Fig_cpca_static_by_group", formats)
    return omni, cons


# ============================================================
# SUPPLEMENTARY: null agreement and PCA / cPCA checks
# ============================================================
def fig_null_agreement(D, out_dir, formats):
    d = D["sub"]
    tests = list(pd.unique(d.loc[d.role == "test", "metric"]))
    both = [m for m in tests if d.loc[d.metric == m, "iaaft_z"].notna().any()]
    show = [m for m in ["garch_alpha", "garch_LR", "vol_cv", "cpca_roi_c1_env_cv"] if m in both]
    if not show:
        print("  skip FigS_dfc_null_agreement: no IAAFT z values")
        return None
    fig = plt.figure(figsize=(8.0, 5.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25], hspace=0.6, wspace=0.55)
    for j, m in enumerate(show):
        ax = fig.add_subplot(gs[0, j])
        sub = d[d.metric == m]
        x, y = sub.phase_z.to_numpy(float), sub.iaaft_z.to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=8, color=INK, alpha=0.55, lw=0)
        lo = np.nanmin(np.r_[x[ok], y[ok]]) if ok.any() else -1
        hi = np.nanmax(np.r_[x[ok], y[ok]]) if ok.any() else 1
        pad = 0.08 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=BASELINE, lw=0.8, zorder=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        rho = pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman") if ok.sum() > 3 else np.nan
        ax.text(0.04, 0.96, f"ρ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=7)
        ax.set_title(short_label(m), fontsize=8)
        ax.set_xlabel("z, phase null", fontsize=8)
        if j == 0:
            ax.set_ylabel("z, IAAFT null", fontsize=8)
        ax.tick_params(labelsize=7)
        sns.despine(ax=ax)
        if j == 0:
            panel_label(ax, "A", x=-0.45, y=1.08)
    # share of mice with p <= 0.05 per metric and null
    ax = fig.add_subplot(gs[1, :])
    n_surr = int(D["cfg"].get("n_surr", "19") or 19)
    pmin = 1.0 / (n_surr + 1)
    rows = []
    for m in tests:
        sub = d[d.metric == m]
        for null in ("phase", "iaaft"):
            p = sub[f"{null}_p"].to_numpy(float)
            p = p[np.isfinite(p)]
            if null == "iaaft" and not sub.iaaft_z.notna().any():
                continue
            if p.size:
                rows.append(dict(metric=m, null=null, n=p.size,
                                 frac=np.mean(p <= ALPHA + 1e-12)))
    F = pd.DataFrame(rows)
    x0 = np.arange(len(tests))
    n_mice = int(F.n.max()) if len(F) else 0
    if n_mice and pmin <= ALPHA + 1e-12:
        from scipy.stats import binom
        lo, hi = binom.ppf([0.025, 0.975], n_mice, pmin) / n_mice
        ax.axhspan(lo, hi, color=GRID, alpha=0.7, lw=0, zorder=0)
        ax.axhline(pmin, color=BASELINE, lw=0.8, zorder=0)
    for off, null in [(-0.14, "phase"), (0.14, "iaaft")]:
        f = F[F.null == null].set_index("metric")
        xs = [i + off for i, m in enumerate(tests) if m in f.index]
        ys = [f.loc[m, "frac"] for m in tests if m in f.index]
        ax.scatter(xs, ys, s=22, color=NULL_COLORS[null], edgecolor="white", lw=0.5, zorder=3,
                   label={"phase": "Phase randomisation", "iaaft": "IAAFT"}[null])
    ax.set_xticks(x0)
    ax.set_xticklabels([short_label(m) for m in tests], rotation=40, ha="right",
                       rotation_mode="anchor", fontsize=7)
    ax.set_ylabel(f"Mice with p ≤ {ALPHA:g}\n(fraction)", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(0, max(0.3, F.frac.max() * 1.15 if len(F) else 0.3))
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    sns.despine(ax=ax)
    panel_label(ax, "B", x=-0.07, y=1.04)
    save(fig, out_dir, "FigS_dfc_null_agreement", formats)
    return F


def complex_loadings(L, n_regions):
    K = int(L.component.max())
    V = np.zeros((n_regions, K), complex)
    for _, r in L.iterrows():
        V[int(r.region) - 1, int(r.component) - 1] = r.magnitude * np.exp(1j * r.phase_rad)
    return V / np.linalg.norm(V, axis=0, keepdims=True)


def fig_checks(D, out_dir, formats):
    pdim = D["cfg"].get("pca_dim", "25")
    fig = plt.figure(figsize=(8.0, 2.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 0.9], wspace=0.55)
    pca = D["pca"]
    if pca is not None:
        ax = fig.add_subplot(gs[0, 0])
        e = 100 * pca.explained.to_numpy(float)
        x = pca.component.to_numpy()
        ax.bar(x, e, width=0.7, color=BASELINE, lw=0, label="Per component")
        ax.plot(x, np.cumsum(e), "-o", color=INK, ms=2.5, lw=1.0, label="Cumulative")
        ax.text(x[-1], np.cumsum(e)[-1] + 2, f"{np.cumsum(e)[-1]:.1f}%", ha="right",
                va="bottom", fontsize=7)
        ax.set_xlabel("Group PCA component", fontsize=8)
        ax.set_ylabel("Variance explained (%)", fontsize=8.5)
        ax.set_ylim(0, max(np.cumsum(e)[-1] * 1.15, 10))
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=7, loc="center right")
        sns.despine(ax=ax)
        panel_label(ax, "A", x=-0.25)
    ax = fig.add_subplot(gs[0, 1])
    any_space = False
    for off, sp in [(-0.12, "roi"), (0.12, "pca")]:
        c = D[f"comp_{sp}"]
        if c is None:
            continue
        any_space = True
        lab = {"roi": "ROI space", "pca": f"PCA-{pdim} space"}[sp]
        ax.plot(c.component + off, 100 * c.explained, "o", ms=5, color=SPACE_COLORS[sp],
                label=f"{lab}")
        if "refit_explained_iaaft" in c.columns:
            ax.plot(c.component + off + 0.07, 100 * c.refit_explained_iaaft, "o", ms=5,
                    mfc="none", mec=SPACE_COLORS[sp], mew=0.9)
    if any_space:
        ax.plot([], [], "o", mfc="none", mec=SECONDARY, label="Refit on IAAFT surrogates")
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Complex PCA component", fontsize=8)
        ax.set_ylabel("Analytic variance (%)", fontsize=8.5)
        ax.set_ylim(0, None)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=6.5, loc="lower left")
        sns.despine(ax=ax)
        panel_label(ax, "B", x=-0.3)
    La, Lb = D["load_roi"], D["load_pca_in_roi"]
    if La is not None and Lb is not None:
        n = int(max(La.region.max(), Lb.region.max()))
        A, B = complex_loadings(La, n), complex_loadings(Lb, n)
        S = np.abs(A.conj().T @ B)
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(S, cmap="Blues", vmin=0, vmax=1)
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if S[i, j] > 0.6 else INK)
        ax.set_xticks(range(S.shape[1]))
        ax.set_xticklabels([f"C{j + 1}" for j in range(S.shape[1])])
        ax.set_yticks(range(S.shape[0]))
        ax.set_yticklabels([f"C{i + 1}" for i in range(S.shape[0])])
        ax.set_xlabel(f"PCA-{pdim} space\n(mapped to ROIs)", fontsize=8)
        ax.set_ylabel("ROI space", fontsize=8.5)
        ax.set_title("|⟨v$_{ROI}$, v$_{PCA}$⟩|", fontsize=8)
        ax.tick_params(labelsize=7, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        panel_label(ax, "C", x=-0.45, y=1.18)
    save(fig, out_dir, "FigS_pca_cpca_checks", formats)


# ============================================================
# TABLES
# ============================================================
def static_descriptives(D, groups, adjusted):
    d = D["sub"]
    ms = [m for r in STATIC_GRID for m in r if m is not None]
    if adjusted:
        ms.append("cpca_roi_lag_index_adj")
    rows = []
    for m in ms:
        sub = d[d.metric == m]
        if sub.empty:
            continue
        r = {"Measure": long_label(m.replace("_adj", "")) + (" (adjusted for coupling magnitude)" if m.endswith("_adj") else "")}
        scale = 100 if m.endswith("var_expl") else 1
        for g in groups:
            x = scale * sub.loc[sub.group == g, "observed"].dropna()
            r[g] = (f"{x.median():.3g} [{x.quantile(.25):.3g}–{x.quantile(.75):.3g}]"
                    if len(x) else "—")
        rows.append(r)
    return pd.DataFrame(rows)


def write_docx(path, T1, omni, cons, static_tab, notes):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, Inches

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Inches(11), Inches(8.5)
    for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, m, Inches(0.7))
    st = doc.styles["Normal"]
    st.font.name = "Arial"
    st.font.size = Pt(9)
    st.paragraph_format.space_after = Pt(0)
    st.paragraph_format.space_before = Pt(0)

    def borders(tbl):
        b = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom"):
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), "8")
            e.set(qn("w:color"), "000000")
            b.append(e)
        for edge in ("left", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "nil")
            b.append(e)
        tbl._tbl.tblPr.append(b)
        for cell in tbl.rows[0].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            bd = OxmlElement("w:tcBorders")
            e = OxmlElement("w:bottom")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), "6")
            e.set(qn("w:color"), "000000")
            bd.append(e)
            tcPr.append(bd)

    def add_table(title, df, note):
        h = doc.add_paragraph()
        r = h.add_run(title)
        r.bold = True
        r.font.size = Pt(10)
        h.paragraph_format.space_after = Pt(4)
        if df is None or df.empty:
            doc.add_paragraph("No rows.")
        else:
            t = doc.add_table(rows=1, cols=df.shape[1])
            for j, c in enumerate(df.columns):
                cell = t.rows[0].cells[j]
                cell.text = ""
                run = cell.paragraphs[0].add_run(str(c))
                run.bold = True
                run.font.size = Pt(8.5)
            for _, row in df.iterrows():
                cells = t.add_row().cells
                for j, v in enumerate(row):
                    cells[j].text = ""
                    run = cells[j].paragraphs[0].add_run(str(v))
                    run.font.size = Pt(8.5)
            for row in t.rows:
                for j, cell in enumerate(row.cells):
                    cell.paragraphs[0].alignment = (WD_ALIGN_PARAGRAPH.LEFT if j == 0
                                                    else WD_ALIGN_PARAGRAPH.CENTER)
            borders(t)
        n = doc.add_paragraph()
        rn = n.add_run(note)
        rn.italic = True
        rn.font.size = Pt(8)
        n.paragraph_format.space_before = Pt(3)
        doc.add_paragraph()

    tA = pd.DataFrame({
        "Measure": T1.label,
        "Family": T1.family,
        "Pre-specified": np.where(T1.prespecified, "yes", "exploratory"),
        "n": T1.n,
        "Median z (observed)": T1.median_z_observed.map("{:.2f}".format),
        "Median z (replica)": T1.median_z_replica.map(lambda v: "—" if not np.isfinite(v) else f"{v:.2f}"),
        "Median difference [95% CI]": [f"{a:.2f} [{b:.2f}, {c:.2f}]" for a, b, c in
                                       zip(T1.median_difference, T1.ci_low, T1.ci_high)],
        "p": T1.p_wilcoxon_greater.map(fmt_p),
        "p (Holm)": T1.p_Holm_prespecified.map(lambda v: "—" if not np.isfinite(v) else fmt_p(v)),
    })
    add_table("Table S-A. Calibrated test of non-stationarity (phase null).", tA, notes["A"])

    tB = pd.DataFrame({
        "Figure": omni.figure.str.replace("Fig_", ""),
        "Measure": omni.label,
        "Outcome": np.where(omni.value == "phase_z", "z (phase null)", "observed"),
        "n": omni.n,
        "Welch F (df)": [("—" if not np.isfinite(F) else f"{F:.2f} ({a:.0f}, {b:.1f})")
                         for F, a, b in zip(omni.welch_F, omni.df1, omni.df2)],
        "p (permutation)": omni.p_perm.map(lambda v: fmt_p(v, N_PERMUTATIONS)),
        "p (Holm)": omni.p_Holm.map(lambda v: fmt_p(v, N_PERMUTATIONS)),
    })
    add_table("Table S-B. Group comparisons: Welch ANOVA across groups.", tB, notes["B"])

    sig = cons[cons.q_BH < ALPHA] if len(cons) else cons
    tC = pd.DataFrame({
        "Figure": sig.figure.str.replace("Fig_", ""),
        "Measure": sig.label,
        "Contrast": sig.contrast,
        "n": [f"{a}/{b}" for a, b in zip(sig.n_a, sig.n_b)],
        "Mean difference": sig.mean_diff.map("{:.3g}".format),
        "Welch t (df)": [f"{t:.2f} ({df:.1f})" for t, df in zip(sig.welch_t, sig.df)],
        "p (Welch t)": sig.p_welch.map(fmt_p),
        "q (BH)": sig.q_BH.map(fmt_p),
    }) if len(sig) else pd.DataFrame()
    add_table("Table S-C. Planned contrasts with q < 0.05.", tC, notes["C"])
    add_table("Table S-D. Static complex-PCA measures by group (median [IQR]).",
              static_tab, notes["D"])

    h = doc.add_paragraph()
    r = h.add_run("Figure notes (draft legend material)")
    r.bold = True
    r.font.size = Pt(10)
    for line in notes["figs"]:
        p = doc.add_paragraph(line)
        p.paragraph_format.space_after = Pt(4)
    doc.save(path)


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--in_dir", default="dfc_dcc_cpca")
    ap.add_argument("--out_dir", default=None, help="default: <in_dir>/figures")
    ap.add_argument("--roi_labels", default=None)
    ap.add_argument("--lag_covariate", default=None,
                    help="CSV from compute_lag_covariate.py")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--n_perm", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--n_boot", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()
    out = args.out_dir or os.path.join(args.in_dir, "figures")
    os.makedirs(out, exist_ok=True)

    D = load_all(args.in_dir)
    d = D["sub"]
    groups = group_order(d.group)
    palette = build_palette(groups)
    n_sub = d.subject_id.nunique()
    n_roi = int(D["load_roi"].region.max()) if D["load_roi"] is not None else 0
    names = roi_names(args.roi_labels, n_roi)
    cfg = D["cfg"]
    print(f"Subjects: {n_sub}  ROIs: {n_roi}  Groups: "
          + ", ".join(f"{g} ({d.loc[d.group == g, 'subject_id'].nunique()})" for g in groups))
    print(f"Config: TR={cfg.get('TR', '?')} s, n_surr={cfg.get('n_surr', '?')}, "
          f"n_replica={cfg.get('n_replica', '0')}, pca_dim={cfg.get('pca_dim', '?')}, "
          f"dcc_decimate={cfg.get('dcc_decimate', '?')}")
    D["sub"], adjusted = add_adjusted_lag(d, args.lag_covariate)

    print("Figures:")
    T1, calibrated = fig_nonstationarity(D, out, args.formats, args.n_boot)
    omni_z, cons_z = fig_group_z(D, out, args.formats, groups, palette, args.n_perm)
    fig_cpca(D, out, args.formats, "roi", "Fig_cpca_propagation_roi", names)
    omni_s, cons_s = fig_static(D, out, args.formats, groups, palette, args.n_perm, adjusted)
    fig_cpca(D, out, args.formats, "pca", "FigS_cpca_propagation_pca", names)
    fig_roi_lags(D, out, args.formats, names)
    frac = fig_null_agreement(D, out, args.formats)
    fig_checks(D, out, args.formats)

    omni = pd.concat([omni_z, omni_s], ignore_index=True)
    cons = pd.concat([cons_z, cons_s], ignore_index=True)
    static_tab = static_descriptives(D, groups, adjusted)
    T1.to_csv(os.path.join(out, "table_calibrated_test.csv"), index=False)
    omni.to_csv(os.path.join(out, "table_group_omnibus.csv"), index=False)
    cons.to_csv(os.path.join(out, "table_group_contrasts_all.csv"), index=False)
    static_tab.to_csv(os.path.join(out, "table_static_by_group.csv"), index=False)
    if frac is not None:
        frac.to_csv(os.path.join(out, "table_fraction_p05.csv"), index=False)
    if D["load_roi"] is not None:
        lab = D["load_roi"].copy()
        lab.insert(1, "label", lab.region.map(names))
        lab.to_csv(os.path.join(out, "cpca_loadings_roi_labeled.csv"), index=False)

    # notes and draft legend material, with numbers from this run
    n_surr = cfg.get("n_surr", "19")
    rep_note = ("Each mouse's stationary replica (a Gaussian series with the same auto- and "
                "cross-spectra, cut from a longer series so its ends do not join) went through "
                f"the identical procedure with its own {n_surr} phase surrogates.")
    fam = "pooled across all contrasts and measures of a figure" if FDR_POOL == "family" \
        else "across the planned contrasts of each measure"
    prim_rows = T1[T1.prespecified]
    notes = {
        "A": ("z: (observed − mean of the mouse's own phase-randomised surrogates) / SD of the "
              "surrogates. " + (rep_note + " p: one-sided Wilcoxon signed-rank test that "
                                "z_observed − z_replica > 0 across mice. " if calibrated else
                                "UNCALIBRATED run (no replica): p is the one-sided Wilcoxon on "
                                "observed − surrogate mean. ") +
              f"Holm correction across the {int(T1.prespecified.sum())} pre-specified measures. "
              f"95% CI: percentile bootstrap of the median ({args.n_boot} resamples). DCC measures "
              "are tested against the phase null only (IAAFT is not a valid null for DCC on "
              "band-limited data)."),
        "B": (f"Welch ANOVA across groups; p from {args.n_perm} permutations of group labels; "
              "Holm correction across the panels of each figure (as shown in panel titles). "
              "Group comparisons of non-stationarity use z, so any bias common to all mice cancels."),
        "C": ("Ordinary two-sided Welch t tests (Welch–Satterthwaite df); Benjamini–Hochberg "
              f"FDR {fam}. Complete results: table_group_contrasts_all.csv."
              + ("" if len(cons[cons.q_BH < ALPHA]) else " No planned contrast reached q < 0.05.")),
        "D": ("Static measures describe stationary lagged coupling (phase randomisation preserves "
              "variance explained and frequency exactly); they are for group comparison only, not "
              "evidence of dynamics. The lag index is a split-half estimate; "
              + ("it is adjusted for each mouse's coupling magnitude (RMS |complex coherence| "
                 "over all ROI pairs, split-half), which is lag-invariant." if adjusted else
                 "it is shown unadjusted here; group comparisons should adjust for coupling "
                 "magnitude (compute_lag_covariate.py), not zero-lag FC strength.")),
        "figs": [
            "Fig_dfc_nonstationarity. (A) Per-mouse z of each pre-specified non-stationarity measure "
            "against the mouse's own phase-randomised surrogates (white boxes) and the same "
            "procedure applied to a stationary replica of each mouse (grey boxes); p above each "
            "pair: one-sided Wilcoxon on z_observed − z_replica. (B) Median difference with 95% "
            "bootstrap CI for all measures; filled markers pre-specified, open markers exploratory; "
            "p (Holm across pre-specified measures). "
            + (f"Pre-specified measures: p = {', '.join(fmt_p(v) for v in prim_rows.p_wilcoxon_greater)} "
               f"(n = {int(prim_rows.n.max()) if len(prim_rows) else 0} mice)." if len(prim_rows) else ""),
            "Fig_dfc_group_z. z by group for each measure (boxes: median and IQR; whiskers 1.5 IQR; "
            "points: mice). Grey band: central 95% of stationary-replica z across all mice (the "
            "stationary reference); line: z = 0. Title p: Welch ANOVA, permutation, Holm across panels.",
            "Fig_cpca_propagation_roi. Group complex PCA in ROI space. Left: complex loadings "
            "(angle = phase relative to the largest-loading ROI, radius = magnitude; positive phase "
            "= later). Middle: lag (s) = phase / (2π × component frequency), ROIs sorted; marker size "
            "∝ |loading|. Right: one propagation cycle, mean standardised BOLD in 20 bins of the "
            "component's phase, ROIs sorted by phase (earliest at top); the strip gives |loading|.",
            "Fig_cpca_static_by_group. Static complex-PCA measures by group; title p: Welch ANOVA, "
            "permutation, Holm across panels. Descriptive of stationary lagged coupling.",
            "FigS_dfc_null_agreement. (A) Per-mouse z under the phase and IAAFT nulls (measures "
            "for which both nulls are valid). (B) Fraction of mice with p ≤ 0.05 against their own "
            f"surrogates ({n_surr} per null: p ≤ 0.05 means the observed value exceeded every "
            "surrogate); line: 0.05 expected under the null, band: central 95% binomial range.",
            "FigS_pca_cpca_checks. (A) Group PCA variance. (B) Complex PCA variance in ROI and PCA-25 "
            "space; open markers: refit on IAAFT surrogates (phase surrogates reproduce the fit "
            "exactly). (C) Similarity |⟨v_ROI, v_PCA⟩| between ROI-space components and PCA-space "
            "components mapped back to ROIs.",
        ],
    }
    try:
        write_docx(os.path.join(out, "Tables_dfc.docx"), T1, omni, cons, static_tab, notes)
    except ImportError:
        print("python-docx not installed; wrote CSV tables only (pip install python-docx).")

    print("\nCalibrated test (pre-specified):" if calibrated else "\nUNCALIBRATED test:")
    print(T1[["metric", "n", "median_difference", "ci_low", "ci_high",
              "p_wilcoxon_greater", "p_Holm_prespecified"]].to_string(index=False))
    print("\nGroup omnibus:")
    print(omni[["figure", "metric", "welch_F", "p_perm", "p_Holm"]].to_string(index=False))
    print(f"\nPlanned contrasts with q < {ALPHA}: {int((cons.q_BH < ALPHA).sum()) if len(cons) else 0}")
    print(f"Done. Outputs in {out}")


if __name__ == "__main__":
    main()
