#!/usr/bin/env python3
"""
Manuscript-quality HMM graph-metric figures.

Styling matches Figure 6 (static graph metrics) in full_analysis_abeta_mice.py:
same palette, group labels, effective font sizes, box/point styling, panel
size and layout (rows = state, columns = metric), and Welch ANOVA p-values
Holm-corrected across all state x metric panels.

Input
-----
graph_metrics_per_subject_per_state_density15.csv

Required columns
----------------
group
state
subject_id
strength_mean
transitivity_bin
Eglob_bin

Outputs
-------
hmm_graph_figures/
    strength_mean_summary.csv
    transitivity_bin_summary.csv
    Eglob_bin_summary.csv

    strength_mean_by_group_state.png
    strength_mean_by_group_state.tiff
    strength_mean_by_group_state.svg

    transitivity_bin_by_group_state.png
    transitivity_bin_by_group_state.tiff
    transitivity_bin_by_group_state.svg

    Eglob_bin_by_group_state.png
    Eglob_bin_by_group_state.tiff
    Eglob_bin_by_group_state.svg

    combined_graph_metrics_by_group_state.png
    combined_graph_metrics_by_group_state.tiff
    combined_graph_metrics_by_group_state.svg
"""

# ============================================================
# HEADLESS MATPLOTLIB BACKEND
# ============================================================

import matplotlib
matplotlib.use("Agg")

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from welch_permutation import welch_anova_permutation, welch_t_permutation
from itertools import combinations

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# USER CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "graph_metrics_per_subject_per_state_density15.csv"
)

OUTPUT_DIR = Path("hmm_graph_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_K = 3
SELECTED_PCA = 50

GROUP_ORDER = [
    "NTG_Y",
    "NTG_M",
    "NTG_A",
    "CRND8_Y",
    "CRND8_M",
    "5XFAD_Y",
    "5XFAD_M",
]

# Tick labels: identical to the static FC figures (raw group names).
GROUP_LABELS = {group: group for group in GROUP_ORDER}

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

# Same column order and sentence-case labels as Figure 6
# (global efficiency, transitivity, mean network strength).
METRICS = {
        "strength_mean": {
        "ylabel": "Mean network strength",
        "filename": "strength_mean_by_group_state",
        "summary_filename": "strength_mean_summary.csv",
    },
        "transitivity_bin": {
        "ylabel": "Binarized transitivity",
        "filename": "transitivity_bin_by_group_state",
        "summary_filename": "transitivity_bin_summary.csv",
    },
    "Eglob_bin": {
        "ylabel": "Binarized global efficiency",
        "filename": "Eglob_bin_by_group_state",
        "summary_filename": "Eglob_bin_summary.csv",
    },
}

# Panel size matches Figure 6 (8 x 8 in for a 3 x 3 grid).
PANEL_SIZE = 8 / 3

PNG_DPI = 350          # same as save_figure() in the static script
TIFF_DPI = 600

# Figure 6 styling (sns.boxplot / sns.stripplot arguments).
BOX_WIDTH = 0.56
BOX_LINEWIDTH = 1.1
JITTER_WIDTH = 0.14
POINT_SIZE = 4.5 ** 2  # stripplot size=4.5 (points) -> scatter area
POINT_ALPHA = 0.8
POINT_EDGEWIDTH = 0.35

# Figure 6 per-axis text sizes.
TITLE_FONTSIZE = 8
YLABEL_FONTSIZE = 11
XTICK_FONTSIZE = 9
YTICK_FONTSIZE = 10

N_PERMUTATIONS = 10000

# Planned between-group comparisons: the same 8 used in every other analysis.
PAIRWISE_COMPARISONS = [
    ("NTG_Y", "NTG_M"), ("NTG_Y", "NTG_A"), ("NTG_Y", "CRND8_Y"),
    ("NTG_M", "CRND8_M"), ("NTG_Y", "5XFAD_Y"), ("NTG_M", "5XFAD_M"),
    ("CRND8_Y", "CRND8_M"), ("5XFAD_Y", "5XFAD_M"),
]
RANDOM_SEED = 20260913


# ============================================================
# MANUSCRIPT STYLE
# ============================================================

# Same calls, in the same order, as full_analysis_abeta_mice.py.
# Note: sns.set_theme() runs last there, so it overrides most of the
# rcParams above (context="paper" font sizes, axis line width). Copying the
# sequence exactly reproduces the static figures' effective style,
# including the font fallback (Arial if installed, otherwise DejaVu Sans).
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

# Export-only settings (do not change appearance): editable text in vectors.
plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Font actually resolved for text (for the manifest).
FONT_FAMILY = font_manager.FontProperties(
    fname=font_manager.findfont(font_manager.FontProperties(family=["sans-serif"]))
).get_name()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def natural_sort_key(value):
    """
    Sort labels containing numbers naturally.

    For example:
        State 2 comes before State 10.
    """
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
    ]

def interquartile_range(values):
    """Calculate Q3 minus Q1 after excluding missing values."""
    values = pd.Series(values).dropna()

    if values.empty:
        return np.nan

    return values.quantile(0.75) - values.quantile(0.25)

def welch_anova_p(dataframe, outcome, transform=None):
    """
    Permutation p-value for a Welch ANOVA across GROUP_ORDER.
    NaN if any group has fewer than two observations or zero variance.
    """
    samples = []
    for group in GROUP_ORDER:
        values = dataframe.loc[dataframe["group"] == group, outcome].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if transform is not None:
            values = transform(values)
        samples.append(values)

    if any(len(x) < 2 or np.var(x, ddof=1) == 0 for x in samples):
        return np.nan

    _, p_value = welch_anova_permutation(
        samples, n_permutations=N_PERMUTATIONS, random_seed=RANDOM_SEED
    )
    return p_value

def holm_adjusted_p(data, state_order):
    """
    Welch ANOVA for every state x metric panel, Holm-corrected as one family
    (the same family structure as Figure 6: density x metric).
    """
    raw = {
        (state, metric): welch_anova_p(data.loc[data["state"] == state], metric)
        for state in state_order
        for metric in METRICS
    }
    keys = [k for k, p in raw.items() if np.isfinite(p)]
    adjusted = {k: np.nan for k in raw}
    if keys:
        _, p_adj, _, _ = multipletests(
            [raw[k] for k in keys], alpha=0.05, method="holm"
        )
        adjusted.update(dict(zip(keys, p_adj)))
    return raw, adjusted


def welch_anova_full(dataframe, outcome):
    """
    Welch ANOVA across GROUP_ORDER: F, numerator df, Welch denominator df and
    permutation p. NaNs if any group has < 2 observations or zero variance.
    """
    samples = []
    for group in GROUP_ORDER:
        values = dataframe.loc[dataframe["group"] == group, outcome].to_numpy(dtype=float)
        samples.append(values[np.isfinite(values)])

    if any(len(x) < 2 or np.var(x, ddof=1) == 0 for x in samples):
        return np.nan, np.nan, np.nan, np.nan

    F, p = welch_anova_permutation(samples, n_permutations=N_PERMUTATIONS,
                                   random_seed=RANDOM_SEED)
    n = np.array([len(x) for x in samples], dtype=float)
    v = np.array([np.var(x, ddof=1) for x in samples])
    k = len(samples)
    w = n / v
    lam = (((1 - w / w.sum()) ** 2) / (n - 1)).sum()
    return F, k - 1, (k ** 2 - 1) / (3 * lam), p


def paired_signflip_test(differences):
    """
    Two-sided permutation test that the mean within-subject difference is 0,
    using the paired t statistic and random sign flips. Enumerates all sign
    patterns exactly when there are few subjects.
    Returns (t, p, n).
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3 or np.std(d, ddof=1) == 0:
        return np.nan, np.nan, n

    def t_stat(x):
        sd = np.std(x, ddof=1)
        return np.nan if sd == 0 else x.mean() / (sd / np.sqrt(x.size))

    observed = t_stat(d)
    if 2 ** n <= N_PERMUTATIONS:
        signs = np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T
        null = np.array([t_stat(s * d) for s in signs])
        p = np.mean(np.abs(null) >= abs(observed) - 1e-12)
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        exceed = 0
        for _ in range(N_PERMUTATIONS):
            t_null = t_stat(rng.choice([-1.0, 1.0], size=n) * d)
            exceed += np.isfinite(t_null) and abs(t_null) >= abs(observed)
        p = (1 + exceed) / (N_PERMUTATIONS + 1)
    return observed, p, n


def bh(values):
    values = np.asarray(values, dtype=float)
    q = np.full_like(values, np.nan)
    ok = np.isfinite(values)
    if ok.any():
        q[ok] = multipletests(values[ok], method="fdr_bh")[1]
    return q


def add_vertical_padding(ax, values, lower=0.07, upper=0.12):
    """Add consistent vertical whitespace around observed values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    y_min = values.min()
    y_max = values.max()
    y_range = y_max - y_min

    if y_range == 0:
        y_range = max(abs(y_max), 1.0)

    ax.set_ylim(
        y_min - lower * y_range,
        y_max + upper * y_range,
    )


def save_manuscript_figure(fig, output_stem):
    """
    Save raster and vector versions of a figure.

    PNG:
        convenient for presentations and general viewing

    TIFF:
        high-resolution raster file commonly accepted by journals

    SVG:
        editable vector file for Illustrator or Inkscape
    """
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    tiff_path = OUTPUT_DIR / f"{output_stem}.tiff"
    svg_path = OUTPUT_DIR / f"{output_stem}.svg"

    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        tiff_path,
        dpi=TIFF_DPI,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {tiff_path}")
    print(f"Saved: {svg_path}")


def draw_box_and_points(
    ax,
    subset,
    outcome,
    group_order,
    rng,
):
    """
    Draw one state panel using a boxplot plus deterministic jitter.

    The jitter is generated explicitly rather than delegated to
    seaborn, making the figure exactly reproducible.
    """
    sns.boxplot(
        data=subset,
        x="group",
        y=outcome,
        order=group_order,
        hue="group",
        hue_order=group_order,
        palette=GROUP_COLORS,
        width=BOX_WIDTH,
        showfliers=False,
        linewidth=BOX_LINEWIDTH,
        legend=False,
        # "group" is Categorical, which makes seaborn dodge by hue and draw
        # each box at 1/7 width. dodge=False keeps full-width boxes.
        dodge=False,
        ax=ax,
    )

    # Draw subject points manually for reproducible jitter.
    for group_index, group in enumerate(group_order):
        group_values = (
            subset.loc[subset["group"] == group, outcome]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(group_values) == 0:
            continue

        x_positions = (
            group_index
            + rng.uniform(
                low=-JITTER_WIDTH,
                high=JITTER_WIDTH,
                size=len(group_values),
            )
        )

        ax.scatter(
            x_positions,
            group_values,
            s=POINT_SIZE,
            facecolor=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=POINT_EDGEWIDTH,
            alpha=POINT_ALPHA,
            zorder=3,
        )

def format_state_axis(
    ax,
    state_label,
    ylabel,
    p_adjusted,
):
    """Figure 6 formatting for one state panel."""
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[group] for group in GROUP_ORDER])

    p_text = "NA" if not np.isfinite(p_adjusted) else f"{p_adjusted:.3g}"
    ax.set_title(f"State {state_label}; p={p_text}", fontsize=TITLE_FONTSIZE)

    ax.set_xlabel("", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", pad=4, labelsize=XTICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)

    sns.despine(ax=ax)


def create_metric_figure(
    data,
    outcome,
    ylabel,
    filename,
    state_order,
    p_adjusted_map,
):
    """One row of state panels for a single metric (Figure 6 panel size)."""
    n_states = len(state_order)

    fig, axes = plt.subplots(
        1,
        n_states,
        figsize=(PANEL_SIZE * n_states, PANEL_SIZE + 0.3),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    rng = np.random.default_rng(RANDOM_SEED)

    finite_values = data[outcome].replace([np.inf, -np.inf], np.nan).dropna()

    for state_index, state in enumerate(state_order):
        ax = axes[state_index]
        subset = data.loc[data["state"] == state].copy()

        draw_box_and_points(ax, subset, outcome, GROUP_ORDER, rng)
        format_state_axis(
            ax,
            state_label=state,
            ylabel=ylabel,
            p_adjusted=p_adjusted_map[(state, outcome)],
        )
        add_vertical_padding(ax, finite_values)

        # Shared y-axis: label only the first panel.
        if state_index > 0:
            ax.set_ylabel("")

    save_manuscript_figure(fig, filename)
    plt.close(fig)


def create_combined_figure(data, state_order, p_adjusted_map):
    """
    Grid matching Figure 6: rows = HMM states (Figure 6 rows = densities),
    columns = metrics in the same order, every panel with its own y-axis
    label and Holm-adjusted p-value in the title.
    Each metric keeps one y-range across states so states are comparable.
    """
    n_states = len(state_order)
    n_metrics = len(METRICS)

    fig, axes = plt.subplots(
        n_states,
        n_metrics,
        figsize=(PANEL_SIZE * n_metrics, PANEL_SIZE * n_states),
        constrained_layout=True,
        squeeze=False,
    )

    rng = np.random.default_rng(RANDOM_SEED)

    for col, (outcome, settings) in enumerate(METRICS.items()):
        finite_values = data[outcome].replace([np.inf, -np.inf], np.nan).dropna()

        for row, state in enumerate(state_order):
            ax = axes[col, row]
            subset = data.loc[data["state"] == state].copy()

            draw_box_and_points(ax, subset, outcome, GROUP_ORDER, rng)
            format_state_axis(
                ax,
                state_label=state,
                ylabel=settings["ylabel"],
                p_adjusted=p_adjusted_map[(state, outcome)],
            )
            add_vertical_padding(ax, finite_values)

    save_manuscript_figure(fig, "combined_graph_metrics_by_group_state")
    plt.close(fig)


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE.resolve()}\n"
        "Place this script in the same directory as "
        "graph_metrics_per_subject_per_state_density15.csv, "
        "or change INPUT_FILE near the top of the script."
    )

data = pd.read_csv(INPUT_FILE)

required_columns = {
    "group",
    "state",
    "subject_id",
    "strength_mean",
    "transitivity_bin",
    "Eglob_bin",
}

missing_columns = required_columns.difference(data.columns)

if missing_columns:
    raise ValueError(
        "The input file is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )

# Standardize text fields.
data["group"] = data["group"].astype(str).str.strip()
data["state"] = data["state"].astype(str).str.strip()
data["subject_id"] = data["subject_id"].astype(str).str.strip()

# Convert graph metrics to numeric values.
for metric in METRICS:
    data[metric] = pd.to_numeric(
        data[metric],
        errors="coerce",
    )

# Match the R script's initial filtering step.
data = data.loc[
    np.isfinite(data["strength_mean"])
].copy()

unexpected_groups = sorted(
    set(data["group"].dropna()).difference(GROUP_ORDER)
)

if unexpected_groups:
    print(
        "Warning: excluding unexpected groups: "
        + ", ".join(unexpected_groups)
    )

data = data.loc[
    data["group"].isin(GROUP_ORDER)
].copy()

data["group"] = pd.Categorical(
    data["group"],
    categories=GROUP_ORDER,
    ordered=True,
)

if data.empty:
    raise ValueError(
        "No usable observations remain after filtering."
    )

state_order = sorted(
    data["state"].dropna().unique(),
    key=natural_sort_key,
)

data["state"] = pd.Categorical(
    data["state"],
    categories=state_order,
    ordered=True,
)

print(f"Loaded observations: {len(data)}")
print(f"Unique subjects: {data['subject_id'].nunique()}")
print(f"States: {state_order}")

print("\nObservations by group and state:")
print(
    data.groupby(
        ["group", "state"],
        observed=False,
    )
    .size()
    .unstack(fill_value=0)
)


# ============================================================
# DESCRIPTIVE SUMMARY TABLES
# ============================================================

for metric, settings in METRICS.items():
    summary = (
        data.groupby(
            ["group", "state"],
            observed=False,
            sort=False,
        )[metric]
        .agg(
            n="count",
            mean="mean",
            sd="std",
            median="median",
            IQR=interquartile_range,
        )
        .reset_index()
    )

    summary_path = (
        OUTPUT_DIR / settings["summary_filename"]
    )

    summary.to_csv(
        summary_path,
        index=False,
        float_format="%.8f",
    )

    print(f"\nSaved summary: {summary_path}")
    print(summary.to_string(index=False))


# ============================================================
# GROUP TESTS (Welch ANOVA, Holm across states x metrics)
# ============================================================

raw_p_map, p_adjusted_map = holm_adjusted_p(data, state_order)

omnibus_rows = []
for state in state_order:
    for metric in METRICS:
        F, df1, df2, p = welch_anova_full(data.loc[data["state"] == state], metric)
        omnibus_rows.append({
            "state": state, "metric": metric, "test": "Welch ANOVA (permutation)",
            "F": F, "df1": df1, "df2_Welch": df2,
            "p_raw": p, "p_Holm": p_adjusted_map[(state, metric)],
        })
stats_table = pd.DataFrame(omnibus_rows)
stats_path = OUTPUT_DIR / "group_tests_by_state.csv"
stats_table.to_csv(stats_path, index=False)
print(f"\nSaved group tests: {stats_path}")
print(stats_table.round(4).to_string(index=False))


# ============================================================
# POST HOC 1: planned group comparisons within each state
# Welch t, permutation p; BH pooled across comparisons x states per metric
# ============================================================

group_rows = []
for metric in METRICS:
    for state in state_order:
        sub = data.loc[data["state"] == state]
        for group_a, group_b in PAIRWISE_COMPARISONS:
            a = sub.loc[sub["group"] == group_a, metric].to_numpy(dtype=float)
            b = sub.loc[sub["group"] == group_b, metric].to_numpy(dtype=float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 2 or len(b) < 2 or (np.var(a, ddof=1) == 0 and np.var(b, ddof=1) == 0):
                t_value, p_value = np.nan, np.nan
            else:
                t_value, p_value = welch_t_permutation(
                    a, b, n_permutations=N_PERMUTATIONS, random_seed=RANDOM_SEED)
            group_rows.append({
                "metric": metric, "state": state,
                "comparison": f"{group_a} vs {group_b}",
                "n_a": len(a), "n_b": len(b),
                "mean_a": a.mean() if len(a) else np.nan,
                "mean_b": b.mean() if len(b) else np.nan,
                "difference": (a.mean() - b.mean()) if len(a) and len(b) else np.nan,
                "t_Welch": t_value, "p_permutation": p_value,
            })

group_posthoc = pd.DataFrame(group_rows)
group_posthoc["q_BH_pooled"] = np.nan
for metric, index in group_posthoc.groupby("metric").groups.items():
    group_posthoc.loc[index, "q_BH_pooled"] = bh(group_posthoc.loc[index, "p_permutation"])

group_posthoc_path = OUTPUT_DIR / "posthoc_group_comparisons_by_state.csv"
group_posthoc.to_csv(group_posthoc_path, index=False)
print(f"\nSaved group-wise post hoc tests: {group_posthoc_path}")
significant = group_posthoc[group_posthoc["q_BH_pooled"] < 0.05]
print(f"  {len(significant)} of {group_posthoc['p_permutation'].notna().sum()} "
      "testable comparisons at q < 0.05")
if len(significant):
    print(significant.round(4).to_string(index=False))


# ============================================================
# POST HOC 2: state comparisons within each group
# The same animal contributes to every state it visited, so states are
# compared within animals: paired differences, sign-flip permutation.
# BH pooled across state pairs x groups per metric.
# ============================================================

state_rows = []
for metric in METRICS:
    wide = (data.pivot_table(index=["subject_id", "group"], columns="state",
                             values=metric, observed=True)
            .reset_index())
    for group in GROUP_ORDER:
        sub = wide.loc[wide["group"] == group]
        for state_a, state_b in combinations(state_order, 2):
            if state_a not in sub.columns or state_b not in sub.columns:
                continue
            pair = sub[[state_a, state_b]].dropna()
            differences = (pair[state_a] - pair[state_b]).to_numpy(dtype=float)
            t_value, p_value, n_pairs = paired_signflip_test(differences)
            state_rows.append({
                "metric": metric, "group": group,
                "comparison": f"{state_a} vs {state_b}",
                "n_paired_animals": n_pairs,
                "mean_a": pair[state_a].mean() if n_pairs else np.nan,
                "mean_b": pair[state_b].mean() if n_pairs else np.nan,
                "mean_difference": differences.mean() if n_pairs else np.nan,
                "t_paired": t_value, "p_permutation": p_value,
            })

state_posthoc = pd.DataFrame(state_rows)
state_posthoc["q_BH_pooled"] = np.nan
for metric, index in state_posthoc.groupby("metric").groups.items():
    state_posthoc.loc[index, "q_BH_pooled"] = bh(state_posthoc.loc[index, "p_permutation"])

state_posthoc_path = OUTPUT_DIR / "posthoc_state_comparisons_by_group.csv"
state_posthoc.to_csv(state_posthoc_path, index=False)
print(f"\nSaved state-wise post hoc tests: {state_posthoc_path}")
significant = state_posthoc[state_posthoc["q_BH_pooled"] < 0.05]
print(f"  {len(significant)} of {state_posthoc['p_permutation'].notna().sum()} "
      "testable comparisons at q < 0.05")
if len(significant):
    print(significant.round(4).to_string(index=False))


# ============================================================
# INDIVIDUAL MANUSCRIPT FIGURES
# ============================================================

for metric, settings in METRICS.items():
    print(f"\nCreating figure for: {metric}")

    create_metric_figure(
        data=data,
        outcome=metric,
        ylabel=settings["ylabel"],
        filename=settings["filename"],
        state_order=state_order,
        p_adjusted_map=p_adjusted_map,
    )


# ============================================================
# COMBINED THREE-METRIC FIGURE
# ============================================================

print("\nCreating combined graph-metric figure")

create_combined_figure(
    data=data,
    state_order=state_order,
    p_adjusted_map=p_adjusted_map,
)


# ============================================================
# COLOR-KEY FIGURE
# ============================================================

legend_handles = [
    Patch(
        facecolor=GROUP_COLORS[group],
        edgecolor="#222222",
        linewidth=0.8,
        label=GROUP_LABELS[group],
    )
    for group in GROUP_ORDER
]

legend_fig, legend_ax = plt.subplots(
    figsize=(11, 1.4)
)

legend_ax.legend(
    handles=legend_handles,
    loc="center",
    ncol=len(GROUP_ORDER),
    frameon=False,
    title="Experimental group",
    columnspacing=1.4,
    handlelength=1.5,
)

legend_ax.axis("off")

save_manuscript_figure(
    legend_fig,
    "experimental_group_color_key",
)

plt.close(legend_fig)


# ============================================================
# ANALYSIS MANIFEST
# ============================================================

manifest = [
    "HMM graph-metric manuscript figure generation",
    f"Input file: {INPUT_FILE.resolve()}",
    f"Output directory: {OUTPUT_DIR.resolve()}",
    f"Selected K: {SELECTED_K}",
    f"Selected HMM-PCA components: {SELECTED_PCA}",
    f"Number of observations: {len(data)}",
    f"Number of unique subjects: {data['subject_id'].nunique()}",
    f"States: {', '.join(map(str, state_order))}",
    f"Group order: {', '.join(GROUP_ORDER)}",
    f"Font family used: {FONT_FAMILY}",
    f"PNG resolution: {PNG_DPI} dpi",
    f"TIFF resolution: {TIFF_DPI} dpi",
    "Figures: boxplots with reproducibly jittered subject points",
    "Styling: matched to Figure 6 of full_analysis_abeta_mice.py",
    "Group tests: Welch ANOVA (permutation) per state x metric, Holm-corrected across all panels",
    "Y-axis scaling: fixed across state panels within each metric",
]

manifest_path = OUTPUT_DIR / "figure_generation_manifest.txt"

manifest_path.write_text(
    "\n".join(manifest) + "\n",
    encoding="utf-8",
)

print(f"\nSaved manifest: {manifest_path}")
print("\nFigure generation completed successfully.")
print(f"Results directory: {OUTPUT_DIR.resolve()}")