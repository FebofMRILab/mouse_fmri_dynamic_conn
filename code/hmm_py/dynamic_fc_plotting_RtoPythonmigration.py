#!/usr/bin/env python3
"""
Manuscript-quality HMM-MAR plots.

Inputs (run4d tables of the reported fit; hmm_model_selection.py, same
settings as hmm_config.m: HMM-PCA rank 15, K = 4)
------
<signals>/hmm_switching_rates_all_K_all_lowrank.csv
<signals>/hmm_maxFO_all_K_all_lowrank.csv
<signals>/hmm_dwell_times_all_K_all_lowrank.csv
(<signals> = HMM_OUT_BASE, default /blue/febo/share/Abeta_Age/rabies/signals)
Every table must carry the fit ID in <model folder>/fit_id.txt.

Outputs
-------
<signals>/hmm_dynamics_figures_hmmpca15_k4/

Styling matches Figure 6 (static graph metrics) in full_analysis_abeta_mice.py
and the HMM graph-metric figures: same palette, group labels, effective font
sizes, box/point styling and panel size. Panel titles carry Welch ANOVA
permutation p-values (welch_tests_scipy.py; Holm-corrected across states
within each figure), as in Figure 6. hmm_dynamic_stats.py computes the same
tests with the same seed, so titles and tables agree.
"""

# ============================================================
# HEADLESS BACKEND — SAFE FOR HIPERGATOR
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
import matplotlib.ticker as mticker
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests

import hmm_model_selection as sel
from welch_tests_scipy import welch_anova_permutation

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURATION
# ============================================================

# The reported fit (hmm_model_selection.py; same settings as hmm_config.m).
SELECTED_K = sel.K
SELECTED_COMPONENTS = sel.N_COMPONENTS

OUTPUT_DIR = sel.OUT_BASE / f"hmm_dynamics_figures_{sel.TAG}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWITCHING_FILE = sel.table_path("switching_rates")
MAXFO_FILE = sel.table_path("maxFO")
DWELL_FILE = sel.table_path("dwell_times")
FIT_IDS = set()

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

PNG_DPI = 350          # same as save_figure() in the static script
TIFF_DPI = 600

N_PERMUTATIONS = 10000
RANDOM_SEED = 20260911

# Panel size matches Figure 6 (8 x 8 in for a 3 x 3 grid).
PANEL_SIZE = 8 / 3

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

# Group tests shown in panel titles.
SHOW_P_VALUES = True
# Dwell times are right-skewed; test log10(dwell) for Figures 5 and 6.
DWELL_TEST_ON_LOG = True

# ============================================================
# MANUSCRIPT STYLE
# ============================================================

# Same calls, in the same order, as full_analysis_abeta_mice.py.
# sns.set_theme() runs last there and overrides most of these rcParams, so
# copying the sequence reproduces the static figures' effective style,
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
# GENERAL HELPERS
# ============================================================

def natural_sort_key(value):
    """Sort numeric/string state labels naturally."""
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
    ]


def interquartile_range(values):
    """Calculate Q3 minus Q1."""
    values = pd.Series(values).dropna()

    if values.empty:
        return np.nan

    return values.quantile(0.75) - values.quantile(0.25)


def save_manuscript_figure(fig, output_stem):
    """Save PNG, TIFF, and editable SVG versions."""
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


def add_vertical_padding(ax, values, lower=0.08, upper=0.13):
    """Add whitespace around the observed y-axis range."""
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


def holm(p_values):
    """Holm adjustment over the finite p-values; NaN stays NaN."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    finite = np.isfinite(p_values)
    if finite.any():
        adjusted[finite] = multipletests(p_values[finite], method="holm")[1]
    return adjusted

STATS_ROWS = []

def record_tests(figure, outcome, labels, raw, adjusted, test):
    for label, p_raw, p_adj in zip(labels, raw, adjusted):
        STATS_ROWS.append({
            "figure": figure,
            "outcome": outcome,
            "panel": label,
            "test": test,
            "p_raw": p_raw,
            "p_Holm_within_figure": p_adj,
        })


def panel_title(label, p_value):
    """Figure 6 title format: '<label>; p=<value>'."""
    if not SHOW_P_VALUES:
        return label
    p_text = "NA" if not np.isfinite(p_value) else f"{p_value:.3g}"
    return f"{label}; p={p_text}" if label else f"p={p_text}"


def format_group_axis(ax, ylabel, title):
    """Figure 6 formatting for one panel."""
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[group] for group in GROUP_ORDER])

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
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


def draw_distribution(ax, dataframe, outcome, rng):
    """Figure 6 boxplot plus reproducibly jittered points."""
    plot_data = dataframe.loc[np.isfinite(dataframe[outcome])].copy()

    sns.boxplot(
        data=plot_data,
        x="group",
        y=outcome,
        order=GROUP_ORDER,
        hue="group",
        hue_order=GROUP_ORDER,
        palette=GROUP_COLORS,
        width=BOX_WIDTH,
        showfliers=False,
        linewidth=BOX_LINEWIDTH,
        legend=False,
        # "group" is Categorical, which makes seaborn dodge by hue and draw
        # each box at 1/7 width, offset from its points. dodge=False fixes it.
        dodge=False,
        ax=ax,
    )

    for group_index, group in enumerate(GROUP_ORDER):
        values = (
            plot_data.loc[plot_data["group"] == group, outcome]
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(values) == 0:
            continue

        x_positions = group_index + rng.uniform(
            -JITTER_WIDTH, JITTER_WIDTH, size=len(values)
        )
        ax.scatter(
            x_positions,
            values,
            s=POINT_SIZE,
            facecolor=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=POINT_EDGEWIDTH,
            alpha=POINT_ALPHA,
            zorder=3,
        )


def create_subject_metric_figure(dataframe, outcome, ylabel, filename):
    """One Figure 6-sized panel for a metric with one row per subject."""
    fig, ax = plt.subplots(
        figsize=(PANEL_SIZE, PANEL_SIZE + 0.3),
        constrained_layout=True,
    )
    rng = np.random.default_rng(RANDOM_SEED)

    p_value = welch_anova_p(dataframe, outcome)
    record_tests(filename, outcome, ["all subjects"], [p_value], [p_value], "Welch ANOVA (permutation)")

    draw_distribution(ax, dataframe, outcome, rng)
    format_group_axis(ax, ylabel, panel_title("", p_value))
    add_vertical_padding(ax, dataframe[outcome])

    save_manuscript_figure(fig, filename)
    plt.close(fig)


def create_state_metric_figure(
    dataframe,
    outcome,
    ylabel,
    filename,
    state_order,
    log_scale=False,
    test_transform=None,
    test_name="Welch ANOVA",
):
    """One row of state panels, shared y-axis, Holm across states."""
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

    raw_p = [
        welch_anova_p(dataframe.loc[dataframe["state"] == s], outcome, test_transform)
        for s in state_order
    ]
    adjusted_p = holm(raw_p)
    record_tests(filename, outcome, state_order, raw_p, adjusted_p, test_name)

    all_values = (
        dataframe[outcome]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    for state_index, state in enumerate(state_order):
        ax = axes[state_index]
        subset = dataframe.loc[dataframe["state"] == state].copy()

        draw_distribution(ax, subset, outcome, rng)
        format_group_axis(
            ax,
            ylabel if state_index == 0 else "",
            panel_title(str(state), adjusted_p[state_index]),
        )

        if log_scale:
            ax.set_yscale("log")
            # Ticks at 1-2-5 steps, printed as plain numbers (not 10^x).
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda value, _: f"{value:g}")
            )
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        else:
            add_vertical_padding(ax, all_values)

    save_manuscript_figure(fig, filename)
    plt.close(fig)


def wilson_interval(successes, total, z=1.96):
    """Wilson 95% confidence interval for a binomial proportion."""
    if total == 0:
        return np.nan, np.nan

    proportion = successes / total
    denominator = 1 + (z ** 2 / total)

    center = (
        proportion
        + z ** 2 / (2 * total)
    ) / denominator

    half_width = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z ** 2 / (4 * total ** 2)
        )
        / denominator
    )

    return center - half_width, center + half_width


# ============================================================
# LOAD INPUT FILES
# ============================================================

required_files = [
    SWITCHING_FILE,
    MAXFO_FILE,
    DWELL_FILE,
]

missing_files = [
    path for path in required_files
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "Missing input files:\n"
        + "\n".join(str(path.resolve()) for path in missing_files)
    )

switching_all = pd.read_csv(SWITCHING_FILE)
maxfo_all = pd.read_csv(MAXFO_FILE)
state_all = pd.read_csv(DWELL_FILE)


# ============================================================
# SELECT K AND PCA SOLUTION
# ============================================================

def select_solution(dataframe, table_name):
    required = {
        "group",
        "subject_id",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"{table_name} is missing columns: "
            + ", ".join(sorted(missing))
        )

    selected = sel.select_model(dataframe, table_name)
    sel.check_fit_id(selected, table_name, FIT_IDS)

    selected["group"] = (
        selected["group"].astype(str).str.strip()
    )
    selected["subject_id"] = (
        selected["subject_id"].astype(str).str.strip()
    )

    unexpected = sorted(
        set(selected["group"]).difference(GROUP_ORDER)
    )

    if unexpected:
        print(
            f"Warning: excluding unexpected groups from "
            f"{table_name}: {unexpected}"
        )

    selected = selected.loc[
        selected["group"].isin(GROUP_ORDER)
    ].copy()

    selected["group"] = pd.Categorical(
        selected["group"],
        categories=GROUP_ORDER,
        ordered=True,
    )

    return selected


switching = select_solution(
    switching_all,
    "switching-rate data",
)

maxfo = select_solution(
    maxfo_all,
    "maximum-FO data",
)

state_metrics = select_solution(
    state_all,
    "state-level data",
)


# ============================================================
# VALIDATE AND FORMAT VARIABLES
# ============================================================

required_switching = {"switching_rate"}
required_maxfo = {"max_FO"}
required_state = {
    "state",
    "fractional_occupancy",
    "n_dwell_episodes",
    "mean_dwell_seconds",
}

if missing := required_switching.difference(switching.columns):
    raise ValueError(
        "Switching data missing: " + ", ".join(sorted(missing))
    )

if missing := required_maxfo.difference(maxfo.columns):
    raise ValueError(
        "Max-FO data missing: " + ", ".join(sorted(missing))
    )

if missing := required_state.difference(state_metrics.columns):
    raise ValueError(
        "State data missing: " + ", ".join(sorted(missing))
    )

numeric_columns = {
    "switching_rate": switching,
    "max_FO": maxfo,
    "fractional_occupancy": state_metrics,
    "n_dwell_episodes": state_metrics,
    "mean_dwell_seconds": state_metrics,
}

for column, dataframe in numeric_columns.items():
    dataframe[column] = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

raw_state_values = (
    state_metrics["state"]
    .dropna()
    .astype(str)
    .unique()
)

raw_state_order = sorted(
    raw_state_values,
    key=natural_sort_key,
)

state_label_map = {
    state: (
        state
        if str(state).lower().startswith("state")
        else f"State {state}"
    )
    for state in raw_state_order
}

state_metrics["state"] = (
    state_metrics["state"]
    .astype(str)
    .map(state_label_map)
)

STATE_ORDER = [
    state_label_map[state]
    for state in raw_state_order
]

state_metrics["state"] = pd.Categorical(
    state_metrics["state"],
    categories=STATE_ORDER,
    ordered=True,
)

# Duplicate checks matching the R script.
if switching["subject_id"].duplicated().any():
    raise ValueError(
        "Duplicate subject rows found in switching data."
    )

if maxfo["subject_id"].duplicated().any():
    raise ValueError(
        "Duplicate subject rows found in Max-FO data."
    )

if state_metrics.duplicated(
    subset=["subject_id", "state"]
).any():
    raise ValueError(
        "Duplicate subject-state rows found in state data."
    )

print(f"Switching subjects: {switching['subject_id'].nunique()}")
print(f"Max-FO subjects: {maxfo['subject_id'].nunique()}")
print(f"State-level subjects: {state_metrics['subject_id'].nunique()}")
print(f"States: {STATE_ORDER}")


# ============================================================
# SAVE SELECTED DATASETS
# ============================================================

switching.to_csv(
    OUTPUT_DIR
    / f"switching_selected_{sel.TAG}.csv",
    index=False,
)

maxfo.to_csv(
    OUTPUT_DIR
    / f"maxFO_selected_{sel.TAG}.csv",
    index=False,
)

state_metrics.to_csv(
    OUTPUT_DIR
    / f"state_metrics_selected_{sel.TAG}.csv",
    index=False,
)


# ============================================================
# DESCRIPTIVE SUMMARIES
# ============================================================

def group_summary(dataframe, outcome):
    return (
        dataframe.groupby(
            "group",
            observed=False,
            sort=False,
        )[outcome]
        .agg(
            n="count",
            mean="mean",
            sd="std",
            median="median",
            IQR=interquartile_range,
        )
        .reset_index()
    )


switching_summary = group_summary(
    switching,
    "switching_rate",
)

maxfo_summary = group_summary(
    maxfo,
    "max_FO",
)

state_summary = (
    state_metrics.groupby(
        ["group", "state"],
        observed=False,
        sort=False,
    )
    .agg(
        n_subjects=("subject_id", "nunique"),
        n_with_state=(
            "n_dwell_episodes",
            lambda x: int(np.sum(x > 0)),
        ),
        state_presence_rate=(
            "n_dwell_episodes",
            lambda x: np.mean(x > 0),
        ),
        mean_FO=("fractional_occupancy", "mean"),
        sd_FO=("fractional_occupancy", "std"),
        median_FO=("fractional_occupancy", "median"),
        mean_dwell_seconds=("mean_dwell_seconds", "mean"),
        median_dwell_seconds=("mean_dwell_seconds", "median"),
    )
    .reset_index()
)

switching_summary.to_csv(
    OUTPUT_DIR / "switching_summary.csv",
    index=False,
)

maxfo_summary.to_csv(
    OUTPUT_DIR / "maxFO_summary.csv",
    index=False,
)

state_summary.to_csv(
    OUTPUT_DIR / "state_summary.csv",
    index=False,
)


# ============================================================
# FIGURE 1: SWITCHING RATE
# ============================================================

create_subject_metric_figure(
    dataframe=switching,
    outcome="switching_rate",
    ylabel="Switching rate",
    filename="figure1_switching_rate_by_group",
)


# ============================================================
# FIGURE 2: MAXIMUM FRACTIONAL OCCUPANCY
# ============================================================

create_subject_metric_figure(
    dataframe=maxfo,
    outcome="max_FO",
    ylabel="Maximum fractional occupancy",
    filename="figure2_maxFO_by_group",
)


# ============================================================
# FIGURE 3: FRACTIONAL OCCUPANCY BY STATE
# ============================================================

create_state_metric_figure(
    dataframe=state_metrics,
    outcome="fractional_occupancy",
    ylabel="Fractional occupancy",
    filename="figure3_fractional_occupancy_by_group_state",
    state_order=STATE_ORDER,
)


# ============================================================
# FIGURE 4: STATE PRESENCE
# ============================================================

presence_rows = []

for (group, state), subset in state_metrics.groupby(
    ["group", "state"],
    observed=False,
):
    total = len(subset)
    successes = int(
        np.sum(subset["n_dwell_episodes"] > 0)
    )

    proportion = successes / total if total else np.nan
    ci_low, ci_high = wilson_interval(successes, total)

    presence_rows.append({
        "group": group,
        "state": state,
        "n": total,
        "n_present": successes,
        "proportion_present": proportion,
        "ci_low": ci_low,
        "ci_high": ci_high,
    })

presence_summary = pd.DataFrame(presence_rows)

presence_summary.to_csv(
    OUTPUT_DIR / "state_presence_summary.csv",
    index=False,
)

n_states = len(STATE_ORDER)

fig, axes = plt.subplots(
    1,
    n_states,
    figsize=(PANEL_SIZE * n_states, PANEL_SIZE + 0.3),
    sharey=True,
    constrained_layout=True,
    squeeze=False,
)
axes = axes.ravel()

for state_index, state in enumerate(STATE_ORDER):
    ax = axes[state_index]

    subset = (
        presence_summary.loc[presence_summary["state"] == state]
        .set_index("group")
        .reindex(GROUP_ORDER)
    )

    x = np.arange(len(GROUP_ORDER))
    heights = subset["proportion_present"].to_numpy(dtype=float)
    lower_error = heights - subset["ci_low"].to_numpy(dtype=float)
    upper_error = subset["ci_high"].to_numpy(dtype=float) - heights

    ax.bar(
        x,
        heights,
        width=BOX_WIDTH,
        color=[GROUP_COLORS[group] for group in GROUP_ORDER],
        edgecolor=".2",
        linewidth=BOX_LINEWIDTH,
        zorder=2,
    )
    ax.errorbar(
        x,
        heights,
        yerr=np.vstack([lower_error, upper_error]),
        fmt="none",
        ecolor=".2",
        elinewidth=BOX_LINEWIDTH,
        capsize=3,
        capthick=BOX_LINEWIDTH,
        zorder=3,
    )

    ax.set_ylim(0, 1.06)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])

    # No group test here: presence is a proportion, not a Welch outcome.
    format_group_axis(
        ax,
        "Subjects entering state" if state_index == 0 else "",
        str(state),
    )

save_manuscript_figure(fig, "figure4_state_presence_by_group")
plt.close(fig)


# ============================================================
# FIGURES 5 AND 6: MEAN DWELL TIME (linear and log scale)
# ============================================================

dwell_observed = state_metrics.loc[
    (state_metrics["n_dwell_episodes"] > 0)
    & np.isfinite(state_metrics["mean_dwell_seconds"])
    & (state_metrics["mean_dwell_seconds"] > 0)
].copy()

dwell_test = dict(
    test_transform=np.log10 if DWELL_TEST_ON_LOG else None,
    test_name="Welch ANOVA on log10(dwell)" if DWELL_TEST_ON_LOG else "Welch ANOVA",
)

create_state_metric_figure(
    dataframe=dwell_observed,
    outcome="mean_dwell_seconds",
    ylabel="Mean dwell time (s)",
    filename="figure5_mean_dwell_by_group_state",
    state_order=STATE_ORDER,
    **dwell_test,
)

# Same test as Figure 5, so both figures show the same p-values.
create_state_metric_figure(
    dataframe=dwell_observed,
    outcome="mean_dwell_seconds",
    ylabel="Mean dwell time (s)",
    filename="figure6_mean_dwell_by_group_state_log_scale",
    state_order=STATE_ORDER,
    log_scale=True,
    **dwell_test,
)


# ============================================================
# GROUP TESTS SHOWN IN PANEL TITLES
# ============================================================

tests_table = pd.DataFrame(STATS_ROWS).drop_duplicates(
    subset=["outcome", "panel", "test"]
)
tests_table["fit_id"] = next(iter(FIT_IDS), "not recorded")
tests_path = OUTPUT_DIR / "group_tests_shown_in_figures.csv"
tests_table.to_csv(tests_path, index=False)
print(f"\nSaved group tests: {tests_path}")
print(tests_table.to_string(index=False))


# ============================================================
# ANALYSIS MANIFEST
# ============================================================

manifest = [
    "HMM-MAR manuscript figure generation",
    f"Model: {sel.LABEL}",
    f"Model folder: {sel.MODEL_DIR}",
    f"Fit ID: {next(iter(FIT_IDS), 'not recorded')}",
    f"Switching-rate subjects: {switching['subject_id'].nunique()}",
    f"Max-FO subjects: {maxfo['subject_id'].nunique()}",
    f"State-level subjects: {state_metrics['subject_id'].nunique()}",
    f"States: {', '.join(map(str, STATE_ORDER))}",
    f"Group order: {', '.join(GROUP_ORDER)}",
    f"Font family: {FONT_FAMILY}",
    f"PNG resolution: {PNG_DPI} dpi",
    f"TIFF resolution: {TIFF_DPI} dpi",
    "X-axis labels: 45-degree rotation applied after seaborn plotting",
    "Styling: matched to Figure 6 of full_analysis_abeta_mice.py",
    "Distributions: boxplot plus individual points (no violins, as in Figure 6)",
    (
        "Panel p-values: Welch ANOVA, permutation p (welch_tests_scipy.py: scipy "
        "permutation_test of f_oneway(equal_var=False)), Holm-corrected across "
        "states within each figure"
        if SHOW_P_VALUES else "Panel p-values: not shown"
    ),
    (
        "Dwell-time test on log10(mean dwell)"
        if DWELL_TEST_ON_LOG else "Dwell-time test on raw mean dwell"
    ),
    "State presence: binomial proportions with Wilson 95% confidence intervals",
    "Dwell time: subjects not entering a state excluded",
    "Full group statistics (contrasts, state comparisons, presence): "
    "hmm_dynamic_stats.py, same tests and seed",
]

manifest_path = OUTPUT_DIR / "figure_generation_manifest.txt"

manifest_path.write_text(
    "\n".join(manifest) + "\n",
    encoding="utf-8",
)

print("\nFigure generation completed successfully.")