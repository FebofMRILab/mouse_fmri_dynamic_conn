#!/usr/bin/env python3
# ================================================================
# Abeta HUMAN rsfMRI CONNECTIVITY AND GRAPH-THEORY ANALYSIS
# Groups: Low, High
# Matrices: 54 subjects × 121 ROIs × 121 ROIs
# ================================================================

import matplotlib
matplotlib.use("Agg")  # Safe on HiPerGator/headless systems

import os
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from welch_permutation import welch_anova_edgewise_permutation
from welch_permutation import welch_anova_permutation
from welch_permutation import welch_t_edgewise_permutation
from welch_permutation import welch_t_permutation

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

OUT = "fc_analysis_humans"
os.makedirs(OUT, exist_ok=True)

GROUP_ORDER = ["Low", "High"]

GROUP_COLORS = {
    "Low": "#e06666",
    "High": "#a73dd9",
}

PAIRWISE_COMPARISONS = [
    ("Low", "High"),
]

GRAPH_DENSITIES = [0.10, 0.20, 0.30]
CIRCULAR_DENSITY = 0.10

NBS_T_THRESHOLD = 2.5
NBS_PERMUTATIONS = 10000
RANDOM_SEED = 20260911

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.2,
    "savefig.dpi": 350,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

sns.set_theme(style="ticks", context="paper")

# ============================================================
# LOAD ROI NAMES
# ============================================================

roi_df = pd.read_csv("roi_names_order.csv")

if "roi_name" not in roi_df.columns:
    raise ValueError("roi_names_order.csv must contain a 'roi_name' column.")

roi_names = roi_df["roi_name"].astype(str).tolist()
N_ROI = len(roi_names)

if N_ROI != 121:
    raise ValueError(f"Expected 121 ROIs, but found {N_ROI}.")

print(f"ROI count: {N_ROI}")

# ============================================================
# LOAD SUBJECT METADATA
# ============================================================

meta = pd.read_csv("subject_group_order_mappings.csv")
meta = meta.rename(columns={"index": "subject_index"})

required_meta = {"subject_index", "genotype"}
missing_meta = required_meta.difference(meta.columns)

if missing_meta:
    raise ValueError(
        "Metadata is missing columns: " + ", ".join(sorted(missing_meta))
    )

meta["subject_index"] = pd.to_numeric(
    meta["subject_index"], errors="raise"
).astype(int)

meta["genotype"] = meta["genotype"].astype(str).str.strip()

unexpected = sorted(set(meta["genotype"]).difference(GROUP_ORDER))

if unexpected:
    raise ValueError(
        f"Unexpected genotype labels found: {unexpected}. "
        f"Expected only {GROUP_ORDER}."
    )

meta = meta.loc[meta["genotype"].isin(GROUP_ORDER)].copy()
meta["genotype"] = pd.Categorical(
    meta["genotype"],
    categories=GROUP_ORDER,
    ordered=True,
)
meta = meta.reset_index(drop=True)

print("\nGroup sizes:")
print(meta["genotype"].value_counts(sort=False))

expected_counts = {"Low": 28, "High": 26}
observed_counts = meta["genotype"].value_counts().to_dict()

for group, expected_n in expected_counts.items():
    observed_n = int(observed_counts.get(group, 0))
    if observed_n != expected_n:
        print(
            f"Warning: expected {expected_n} {group} mice, "
            f"but found {observed_n}."
        )

# ============================================================
# LOAD 54 STACKED 121 × 121 MATRICES
# ============================================================

merged = pd.read_csv("merged.csv")

# Remove pandas-generated index columns.
merged = merged.loc[
    :, ~merged.columns.astype(str).str.startswith("Unnamed")
]

merged = merged.apply(pd.to_numeric, errors="coerce")
merged = merged.dropna(axis=1, how="all")

if merged.shape[1] != N_ROI:
    raise ValueError(
        f"Expected {N_ROI} matrix columns, found {merged.shape[1]}."
    )

# ============================================================
# REMOVE REPEATED HEADER ROWS BETWEEN STACKED MATRICES
# ============================================================

raw_values = merged.to_numpy(dtype=float)
expected_header = np.arange(1, N_ROI + 1, dtype=float)

# Strategy 1: exact integer sequence match with tight tolerance.
header_match_tight = np.isclose(
    raw_values,
    expected_header[None, :],
    atol=1e-8,
    rtol=0,
).all(axis=1)

# Strategy 2: broader tolerance in case of floating-point variation.
header_match_broad = np.isclose(
    raw_values,
    expected_header[None, :],
    atol=1e-3,
    rtol=0,
).all(axis=1)

# Report how many each strategy finds.
print(
    f"Header rows detected (tight atol=1e-8):  {header_match_tight.sum()}"
)
print(
    f"Header rows detected (broad atol=1e-3):  {header_match_broad.sum()}"
)

# Strategy 3: positional detection.
# Every (N_ROI + 1)-th row starting from row N_ROI should be a header,
# i.e. rows at positions N_ROI, 2*(N_ROI+1)-1, etc.
# More robustly: any row whose first value rounds to 1 and last rounds to N_ROI
# and is monotonically increasing by 1.
diffs = np.diff(raw_values, axis=1)
header_match_monotone = (
    np.abs(raw_values[:, 0] - 1.0) < 0.5
) & (
    np.abs(raw_values[:, -1] - N_ROI) < 0.5
) & (
    np.abs(diffs - 1.0).max(axis=1) < 0.5
)

print(
    f"Header rows detected (monotone sequence): {header_match_monotone.sum()}"
)

# Use the strategy that finds the most headers, up to n_subjects_total - 1.
n_expected_headers = len(meta) - 1

candidates = {
    "tight":    (header_match_tight,    header_match_tight.sum()),
    "broad":    (header_match_broad,    header_match_broad.sum()),
    "monotone": (header_match_monotone, header_match_monotone.sum()),
}

# Pick the strategy whose count is closest to n_expected_headers.
best_strategy = min(
    candidates,
    key=lambda k: abs(candidates[k][1] - n_expected_headers),
)

repeated_header_mask = candidates[best_strategy][0]

print(
    f"\nUsing '{best_strategy}' strategy: "
    f"{repeated_header_mask.sum()} header rows removed."
)

if repeated_header_mask.sum() != n_expected_headers:
    print(
        f"Warning: expected {n_expected_headers} header rows "
        f"but found {repeated_header_mask.sum()}. "
        f"Inspect merged.csv around every {N_ROI}th row for anomalies."
    )

merged = (
    merged.loc[~repeated_header_mask]
    .dropna(axis=0, how="all")
    .reset_index(drop=True)
)

n_subjects_total = len(meta)
expected_shape = (n_subjects_total * N_ROI, N_ROI)

print("Parsed merged.csv shape:", merged.shape)

if merged.shape != expected_shape:
    # Diagnose which block is the wrong size.
    n_rows = merged.shape[0]
    n_complete = n_rows // N_ROI
    n_remainder = n_rows % N_ROI

    raise ValueError(
        f"Expected stacked data shape {expected_shape}, "
        f"but obtained {merged.shape}. "
        f"This corresponds to {n_complete} complete matrices "
        f"with {n_remainder} leftover rows. "
        f"Check merged.csv for a missing or extra matrix block."
    )

n_subjects_total = len(meta)
expected_shape = (n_subjects_total * N_ROI, N_ROI)

print("Parsed merged.csv shape:", merged.shape)

if merged.shape != expected_shape:
    raise ValueError(
        f"Expected stacked data shape {expected_shape}, "
        f"but obtained {merged.shape}."
    )

matrices_all = merged.to_numpy(dtype=float).reshape(
    n_subjects_total,
    N_ROI,
    N_ROI,
)

if not np.isfinite(matrices_all).all():
    bad = np.argwhere(~np.isfinite(matrices_all))
    raise ValueError(
        f"Non-finite values remain. First locations: {bad[:10].tolist()}"
    )

symmetry_error = np.max(
    np.abs(matrices_all - matrices_all.transpose(0, 2, 1))
)

diagonal_error = np.max(
    np.abs(
        np.diagonal(matrices_all, axis1=1, axis2=2) - 1
    )
)

print("matrices_all shape:", matrices_all.shape)
print("Maximum symmetry error:", symmetry_error)
print("Maximum diagonal error:", diagonal_error)

if symmetry_error > 1e-6:
    raise ValueError("At least one reconstructed matrix is not symmetric.")

if diagonal_error > 1e-6:
    raise ValueError("At least one matrix has an invalid diagonal.")

# Map metadata rows to matrix blocks in merged.csv.
#
# subject_index is the BIDS subject number (1..58 with gaps), NOT the block
# position: merged.csv holds one block per signal file, stacked in ascending
# signal number. If merged_block_order.txt (written by run6) is present it is
# used directly; otherwise blocks are assumed to follow ascending subject
# number, which is how the signal files are sorted.
subject_numbers = meta["subject_index"].to_numpy(dtype=int)

if len(matrices_all) != len(meta):
    raise ValueError(
        f"merged.csv has {len(matrices_all)} matrices but the mapping file has "
        f"{len(meta)} subjects."
    )

if os.path.exists("merged_block_order.txt"):
    block_order = pd.read_csv("merged_block_order.txt", sep="\t")
    block_of_subject = dict(zip(block_order["signal_number"].astype(int),
                                block_order["block"].astype(int) - 1))
    missing = [s for s in subject_numbers if s not in block_of_subject]
    if missing:
        raise ValueError(f"Subjects not in merged_block_order.txt: {missing}")
    matrix_positions = np.array([block_of_subject[s] for s in subject_numbers])
    print("Block mapping: from merged_block_order.txt")
else:
    rank = np.argsort(np.argsort(subject_numbers))
    matrix_positions = rank
    print("Block mapping: ascending subject number (no merged_block_order.txt)")

if len(set(matrix_positions)) != len(matrix_positions):
    raise ValueError("Two subjects map to the same matrix block.")
if matrix_positions.min() < 0 or matrix_positions.max() >= len(matrices_all):
    raise ValueError("Metadata matrix indices are out of range.")

print("First subjects -> blocks:",
      list(zip(subject_numbers[:4].tolist(), (matrix_positions[:4] + 1).tolist())),
      "... last:", (int(subject_numbers[-1]), int(matrix_positions[-1]) + 1))

matrices_r = matrices_all[matrix_positions].copy()

# Fisher-z values are used for inferential statistics.
matrices_z = np.arctanh(
    np.clip(matrices_r, -0.999999, 0.999999)
)

for subject_matrix in matrices_z:
    np.fill_diagonal(subject_matrix, np.nan)

groups = {
    group: meta["genotype"].astype(str).to_numpy() == group
    for group in GROUP_ORDER
}

print("\nFinal matrix shape:", matrices_r.shape)

for group in GROUP_ORDER:
    print(f"{group}: n={groups[group].sum()}")


# ============================================================
# ROI PAIRS
# ============================================================

triu_i, triu_j = np.triu_indices(N_ROI, k=1)
N_EDGES = len(triu_i)

print(f"Unique ROI pairs: {N_EDGES}")

edge_values_z = matrices_z[:, triu_i, triu_j]

prefrontal_rois = [
    "L Cont_PFCl_1", "L Default_PFC_1",
    "L Default_PFC_2", "L Default_PFC_3",
    "L Default_PFC_4", "L Default_PFC_5",
    "L Default_PFC_6","L Default_PFC_7",
    "R Cont_PFCl_1", "R Cont_PFCl_2",
    "R Cont_PFCl_3", "R Cont_PFCl_4",
    "R Cont_PFCl_1", "R Cont_PFCmp_1",
    "R Default_PFCv_1","R Default_PFCv_2",
    "R Default_PFCdPFCm_1","R Default_PFCdPFCm_2",
    "R Default_PFCdPFCm_3",
]

posterior_rois = [
    "L Default_pCunPCC_1", "L Default_pCunPCC_2",
    "L Cont_Par_1", "L Default_Par_1",
    "L Default_Par_2", "R Cont_Par_1",
    "R Cont_Par_2", "R Default_Par_1",
    "R Cont_pCun_1","R Default_pCunPCC_1", 
    "R Default_pCunPCC_2",
]

missing_rois = [x for x in prefrontal_rois + posterior_rois if x not in roi_names]
if missing_rois:
    raise ValueError(f"Hypothesis ROIs not in roi_names: {missing_rois}")

pf_idx = [roi_names.index(x) for x in prefrontal_rois]
posterior_idx = [roi_names.index(x) for x in posterior_rois]

hypothesis_pairs = [
    (min(i, j), max(i, j))
    for i in pf_idx
    for j in posterior_idx
    if i != j
]
hypothesis_set = set(hypothesis_pairs)

# ============================================================
# GENERAL PLOT HELPERS
# ============================================================

def save_figure(fig, filename):
    path = os.path.join(OUT, filename)
    fig.savefig(
        path,
        dpi=350,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved: {path}")

def heatmap(
    matrix,
    title,
    filename,
    colorbar_label,
    cmap="RdBu_r",
    symmetric=True,
):
    values = matrix[np.isfinite(matrix)]
    values = values[values != 0]

    if symmetric:
        limit = (
            np.percentile(np.abs(values), 98)
            if len(values)
            else 1
        )
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = None, None

    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(18, 12))

    sns.heatmap(
        matrix,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0 if symmetric else None,
        square=True,
        cbar_kws={
            "label": colorbar_label,
            "shrink": 0.72,
        },
        ax=ax,
    )

    ax.set_title(title, pad=14)
    ax.tick_params(axis="x", rotation=90, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    save_figure(fig, filename)


# ============================================================
# FIGURE 1: GROUP-MEAN FC MATRICES
# ============================================================

print("\n[Figure 1] Group-mean FC matrices by A\u03B2 Condition")

group_mean_z = {
    group: np.nanmean(matrices_z[groups[group]], axis=0)
    for group in GROUP_ORDER
}

group_mean_r = {
    group: np.tanh(group_mean_z[group])
    for group in GROUP_ORDER
}

off_diagonal_values = np.concatenate([
    group_mean_r[group][~np.eye(N_ROI, dtype=bool)]
    for group in GROUP_ORDER
])

mean_limit = np.percentile(np.abs(off_diagonal_values), 99)

n_groups = len(GROUP_ORDER)

# For 7 groups use a 2-row layout to keep each panel readable.
n_cols = 2
n_rows = int(np.ceil(n_groups / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(n_cols * 8, n_rows * 8),
    constrained_layout=True,
)

# Flatten to a 1-D list regardless of layout shape.
axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

for ax_idx, group in enumerate(GROUP_ORDER):
    ax = axes_flat[ax_idx]

    sns.heatmap(
        group_mean_r[group],
        mask=np.triu(
            np.ones((N_ROI, N_ROI), dtype=bool),
            k=1,
        ),
        cmap="RdBu_r",
        center=0,
        vmin=-mean_limit,
        vmax=mean_limit,
        square=True,
        cbar=True,
        cbar_kws={"label": "Mean Pearson r", "shrink": 0.7},
        ax=ax,
    )

    ax.set_title(
        f"{group} A\u03B2\n (n={groups[group].sum()})",
        fontsize=11,
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

# Hide any unused panels in the grid.
for unused_ax in axes_flat[n_groups:]:
    unused_ax.set_visible(False)

fig.suptitle(
    "Resting-State Functional Connectivity by Group",
    fontsize=12,
    fontweight="bold",
)

save_figure(fig, "figure1_group_mean_connectivity.png")

# ============================================================
# FIGURE 2: PAIRWISE GROUP-DIFFERENCE MATRICES
# ============================================================

print("\n[Figure 2] Pairwise A\u03B2 group differences")

difference_matrices = {}

n_comparisons = len(PAIRWISE_COMPARISONS)
n_cols = 2
n_rows = int(np.ceil(n_comparisons / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(n_cols * 8, n_rows * 8),
    constrained_layout=True,
)

axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

for ax_idx, (group_a, group_b) in enumerate(PAIRWISE_COMPARISONS):
    ax = axes_flat[ax_idx]
    
    difference = group_mean_z[group_a] - group_mean_z[group_b]
    difference_matrices[(group_a, group_b)] = difference

    values = difference[np.isfinite(difference)]
    limit = np.percentile(np.abs(values), 98)

    sns.heatmap(
        difference,
        mask=np.triu(
            np.ones((N_ROI, N_ROI), dtype=bool),
            k=1,
        ),
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        square=True,
        cbar=True,
        cbar_kws={"label": "Δ Fisher z", "shrink": 0.7},
        ax=ax,
    )

    ax.set_title(f"{group_a} − {group_b}")

# Hide any unused panels.
for unused_ax in axes_flat[n_comparisons:]:
    unused_ax.set_visible(False)
    
fig.suptitle(
    "Pairwise A\u03B2 Group Differences",
    fontsize=12,
    fontweight="bold",
)

save_figure(fig, "figure2_pairwise_connectivity_differences.png")


# ============================================================
# EDGEWISE OMNIBUS ANOVA
# ============================================================

print("\n[Statistics] Edgewise Welch test (Low vs High)")

anova_f = np.zeros(N_EDGES)
anova_p = np.ones(N_EDGES)

masks = [groups[g] for g in GROUP_ORDER]
anova_f, anova_p = welch_anova_edgewise_permutation(
    edge_values_z, masks, n_permutations=10000, random_seed=RANDOM_SEED
)

anova_reject, anova_q, _, _ = multipletests(
    anova_p,
    alpha=0.05,
    method="fdr_bh",
)

print(
    f"Omnibus FDR-significant edges: "
    f"{anova_reject.sum()} / {N_EDGES}"
)

anova_table = pd.DataFrame({
    "ROI_1": [roi_names[i] for i in triu_i],
    "ROI_2": [roi_names[j] for j in triu_j],
    "ROI_1_index": triu_i + 1,
    "ROI_2_index": triu_j + 1,
    "F_statistic": anova_f,
    "p_raw": anova_p,
    "q_FDR": anova_q,
    "FDR_significant": anova_reject,
    "hypothesis_pair": [
        (i, j) in hypothesis_set
        for i, j in zip(triu_i, triu_j)
    ],
}).sort_values(["q_FDR", "p_raw"])

anova_table.to_csv(
    os.path.join(OUT, "edgewise_omnibus_ANOVA.csv"),
    index=False,
)


# ============================================================
# FIGURE 3: OMNIBUS ANOVA MAP
# ============================================================

f_matrix = np.zeros((N_ROI, N_ROI))
q_matrix = np.ones((N_ROI, N_ROI))

f_matrix[triu_i, triu_j] = anova_f
f_matrix[triu_j, triu_i] = anova_f

q_matrix[triu_i, triu_j] = anova_q
q_matrix[triu_j, triu_i] = anova_q

fig, axes = plt.subplots(
    1,
    2,
    figsize=(20, 10),
    constrained_layout=True,
)

f_limit = np.percentile(anova_f, 99)

sns.heatmap(
    f_matrix,
    mask=np.triu(
        np.ones((N_ROI, N_ROI), dtype=bool),
        k=1,
    ),
    cmap="magma",
    vmin=0,
    vmax=f_limit,
    square=True,
    cbar_kws={"label": "Welch F (= t²)", "shrink": 0.7},
    ax=axes[0],
)

axes[0].set_title("Low vs High A\u03B2 Effect")

significance_matrix = (q_matrix < 0.05).astype(float)
np.fill_diagonal(significance_matrix, 0)

sns.heatmap(
    significance_matrix,
    mask=np.triu(
        np.ones((N_ROI, N_ROI), dtype=bool),
        k=1,
    ),
    cmap=["white", "#7A0177"],
    vmin=0,
    vmax=1,
    square=True,
    cbar=False,
    ax=axes[1],
)

axes[1].set_title("FDR-Significant Edges (q < 0.05)")

save_figure(fig, "figure3_edgewise_omnibus_ANOVA.png")


# ============================================================
# PAIRWISE WELCH TESTS WITH POOLED FDR
# (one BH family: all planned comparisons x all edges)
# ============================================================

print("\n[Statistics] Pairwise Welch tests (pooled BH-FDR)")

pairwise_results = {}

# Pass 1: run every planned comparison.
for group_a, group_b in PAIRWISE_COMPARISONS:
    t_values, p_values = welch_t_edgewise_permutation(
        edge_values_z[groups[group_a]],
        edge_values_z[groups[group_b]],
        n_permutations=10000,
        random_seed=RANDOM_SEED,
    )
    pairwise_results[(group_a, group_b)] = {
        "t": np.nan_to_num(t_values),
        "p": np.nan_to_num(p_values, nan=1.0),
    }

# Pooled correction across comparisons x edges.
comparisons = list(pairwise_results)
all_p = np.concatenate([pairwise_results[c]["p"] for c in comparisons])

reject_all, q_all, _, _ = multipletests(
    all_p,
    alpha=0.05,
    method="fdr_bh",
)

N_PAIRWISE_TESTS = all_p.size
print(
    f"Pooled BH family: {len(comparisons)} comparisons x "
    f"{N_EDGES} edges = {N_PAIRWISE_TESTS} tests"
)

for c, rej, q in zip(
    comparisons,
    np.split(reject_all, len(comparisons)),
    np.split(q_all, len(comparisons)),
):
    pairwise_results[c]["q"] = q
    pairwise_results[c]["reject"] = rej

# Pass 2: write one table per comparison.
for (group_a, group_b), res in pairwise_results.items():
    values_a = edge_values_z[groups[group_a]]
    values_b = edge_values_z[groups[group_b]]

    result_table = pd.DataFrame({
        "ROI_1": [roi_names[i] for i in triu_i],
        "ROI_2": [roi_names[j] for j in triu_j],
        "t_statistic": res["t"],
        "p_raw": res["p"],
        "q_FDR_pooled": res["q"],
        "FDR_significant_pooled": res["reject"],
        "mean_difference_Fisher_z": (
            values_a.mean(axis=0) - values_b.mean(axis=0)
        ),
        "hypothesis_pair": [
            (i, j) in hypothesis_set
            for i, j in zip(triu_i, triu_j)
        ],
    }).sort_values(["q_FDR_pooled", "p_raw"])

    result_table.to_csv(
        os.path.join(
            OUT,
            f"edgewise_{group_a}_vs_{group_b}.csv",
        ),
        index=False,
    )

    print(
        f"{group_a} vs {group_b}: "
        f"{res['reject'].sum()} FDR-significant edges (pooled)"
    )


# ============================================================
# FIGURE 4: TOP OMNIBUS EDGES
# ============================================================

print("\n[Figure 4] Top omnibus edges")

TOP_EDGES = min(10, N_EDGES)
top_edge_indices = np.argsort(anova_p)[:TOP_EDGES]

plot_rows = []

for edge_rank, edge_index in enumerate(top_edge_indices, start=1):
    i = triu_i[edge_index]
    j = triu_j[edge_index]
    edge_name = f"{roi_names[i]} ↔ {roi_names[j]}"

    for subject in range(len(meta)):
        plot_rows.append({
            "Edge": edge_name,
            "Group": str(meta.loc[subject, "genotype"]),
            "Fisher_z": edge_values_z[subject, edge_index],
            "p_raw": anova_p[edge_index],
            "q_FDR": anova_q[edge_index],
            "rank": edge_rank,
        })

top_edge_data = pd.DataFrame(plot_rows)

fig, axes = plt.subplots(
    5,
    2,
    figsize=(7, 9),
    constrained_layout=True,
)

for ax, edge_name in zip(
    axes.flat,
    top_edge_data["Edge"].drop_duplicates(),
):
    subset = top_edge_data.loc[top_edge_data["Edge"] == edge_name]
    edge_index = top_edge_indices[
        list(top_edge_data["Edge"].drop_duplicates()).index(edge_name)
    ]

    sns.boxplot(
        data=subset,
        x="Group",
        y="Fisher_z",
        hue="Group",
        order=GROUP_ORDER,
        palette=GROUP_COLORS,
        showfliers=False,
        width=0.36,
        linewidth=1.1,
        legend=False,
        ax=ax,
    )

    sns.stripplot(
        data=subset,
        x="Group",
        y="Fisher_z",
        hue="Group",
        order=GROUP_ORDER,
        palette=GROUP_COLORS,
        jitter=0.16,
        size=4.5,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.8,
        legend=False,
        ax=ax,
    )

    ax.set_title(
        f"{edge_name}\n"
        f"p={anova_p[edge_index]:.3g}, "
        f"q={anova_q[edge_index]:.3g}",
        fontsize=9,
    )
    ax.set_xlabel("",fontsize=12)
    ax.set_ylabel("Fisher z",fontsize=12)
    sns.despine(ax=ax)
    
    # Axis tick-label sizes
    ax.tick_params(axis="x", rotation=0, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

#fig.suptitle(
#    "Top Edges from the Omnibus Low-High A\u03B2 Groups Analysis",
#    fontsize=10,
#    fontweight="bold",
#)

save_figure(fig, "figure4_top_omnibus_edges.png")

# ============================================================
# FIGURE 5: HYPOTHESIS NETWORK SUMMARY
# ============================================================

print("\n[Figure 5] Prefrontal-to-posterior connectivity")

hypothesis_edge_indices = [
    edge
    for edge, (i, j) in enumerate(zip(triu_i, triu_j))
    if (i, j) in hypothesis_set
]

hypothesis_fc = np.nanmean(
    edge_values_z[:, hypothesis_edge_indices],
    axis=1,
)

hypothesis_data = meta[["subject_index", "genotype"]].copy()
hypothesis_data["mean_hypothesis_FC"] = hypothesis_fc

hypothesis_data.to_csv(
    os.path.join(OUT, "hypothesis_network_subject_values.csv"),
    index=False,
)

hyp_samples = [
    hypothesis_fc[groups[group]]
    for group in GROUP_ORDER
]

hyp_f, hyp_p = welch_anova_permutation(hyp_samples, n_permutations=10000, random_seed=RANDOM_SEED)

fig, ax = plt.subplots(figsize=(6.5, 6))

sns.violinplot(
    data=hypothesis_data,
    x="genotype",
    y="mean_hypothesis_FC",
    hue="genotype",
    order=GROUP_ORDER,
    palette=GROUP_COLORS,
    inner=None,
    cut=0,
    alpha=0.25,
    legend=False,
    ax=ax,
)

sns.boxplot(
    data=hypothesis_data,
    x="genotype",
    y="mean_hypothesis_FC",
    hue="genotype",
    order=GROUP_ORDER,
    palette=GROUP_COLORS,
    width=0.25,
    showfliers=False,
    linewidth=1.2,
    legend=False,
    ax=ax,
)

sns.stripplot(
    data=hypothesis_data,
    x="genotype",
    y="mean_hypothesis_FC",
    hue="genotype",
    order=GROUP_ORDER,
    palette=GROUP_COLORS,
    jitter=0.15,
    size=6,
    edgecolor="white",
    linewidth=0.4,
    alpha=0.8,
    legend=False,
    ax=ax,
)

ax.set_title(
    "Prefrontal–Posterior Connectivity\n"
    f"Welch ANOVA: F={hyp_f:.2f}, p={hyp_p:.3g}"
)

ax.set_xlabel("",fontsize=14)
ax.set_ylabel("Mean connectivity (Fisher z)",fontsize=14)
sns.despine(ax=ax)

# Axis tick-label sizes
ax.tick_params(axis="x", rotation=0, labelsize=12)
ax.tick_params(axis="y", labelsize=12)

save_figure(fig, "figure5_hypothesis_network_connectivity.png")

# ============================================================
# PROPORTIONAL THRESHOLDING
# ============================================================

def proportional_threshold_positive(matrix, density):
    """
    Retain the strongest positive edges so that the requested
    density refers to all possible undirected edges.

    Returns
    -------
    weighted : symmetric weighted adjacency matrix
    binary : symmetric binary adjacency matrix
    """
    matrix = np.asarray(matrix, dtype=float).copy()
    matrix = np.nan_to_num(matrix, nan=0.0)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)

    edge_weights = matrix[triu_i, triu_j]
    positive_indices = np.where(edge_weights > 0)[0]

    n_keep = int(np.ceil(density * N_EDGES))
    n_keep = min(n_keep, len(positive_indices))

    weighted = np.zeros((N_ROI, N_ROI), dtype=float)

    if n_keep == 0:
        return weighted, weighted.copy()

    ranked_positive = positive_indices[
        np.argsort(edge_weights[positive_indices])[::-1]
    ]
    selected = ranked_positive[:n_keep]

    selected_i = triu_i[selected]
    selected_j = triu_j[selected]
    selected_weights = edge_weights[selected]

    weighted[selected_i, selected_j] = selected_weights
    weighted[selected_j, selected_i] = selected_weights

    binary = (weighted > 0).astype(float)

    return weighted, binary


# ============================================================
# GRAPH-THEORY METRICS AT MULTIPLE DENSITIES
# ============================================================

print("\n[Graph theory] Subject-level network metrics")

network_rows = []
node_strength_rows = []

for subject in range(len(meta)):
    subject_r = matrices_r[subject].copy()
    np.fill_diagonal(subject_r, 0)

    for density in GRAPH_DENSITIES:
        weighted, binary = proportional_threshold_positive(
            subject_r,
            density,
        )

        graph_binary = nx.from_numpy_array(binary)
        graph_weighted = nx.from_numpy_array(weighted)

        node_strength = weighted.sum(axis=1)

        network_rows.append({
            "subject_index": meta.loc[subject, "subject_index"],
            "genotype": str(meta.loc[subject, "genotype"]),
            "density": density,
            "global_efficiency": nx.global_efficiency(graph_binary),
            # Global clustering is reported as transitivity: the fraction of
            # connected triples that close into triangles. Unlike the unweighted
            # mean of nodal clustering, it is not pulled toward zero by nodes of
            # degree < 2 in sparse graphs. Mean clustering is kept in the CSV
            # for reference but is not plotted or tested.
            "transitivity": nx.transitivity(graph_binary),
            "mean_clustering": nx.average_clustering(graph_binary),
            "mean_network_strength": node_strength.mean(),
        })

        for roi_index, strength in enumerate(node_strength):
            node_strength_rows.append({
                "subject_index": meta.loc[subject, "subject_index"],
                "genotype": str(meta.loc[subject, "genotype"]),
                "density": density,
                "roi_index": roi_index + 1,
                "roi_name": roi_names[roi_index],
                "node_strength": strength,
            })

network_metrics = pd.DataFrame(network_rows)
node_strength_data = pd.DataFrame(node_strength_rows)

network_metrics.to_csv(
    os.path.join(OUT, "network_metrics_by_density.csv"),
    index=False,
)

node_strength_data.to_csv(
    os.path.join(OUT, "node_strength_by_density.csv"),
    index=False,
)


# ============================================================
# FIGURE 6: GRAPH METRICS
# ============================================================

print("\n[Figure 6] Graph metrics")

metric_definitions = [
    ("mean_network_strength", "Mean network strength"),
    ("transitivity", "Transitivity"),
    ("global_efficiency", "Global efficiency"),
]

# Welch ANOVA p-values for every density x metric, Holm-corrected as one family
raw_p = {}
for density in GRAPH_DENSITIES:
    subset = network_metrics.loc[np.isclose(network_metrics["density"], density)]
    for metric, _ in metric_definitions:
        samples = [
            subset.loc[subset["genotype"] == g, metric].to_numpy()
            for g in GROUP_ORDER
        ]
        _, p_value = welch_anova_permutation(
            samples, n_permutations=10000, random_seed=RANDOM_SEED
        )
        raw_p[(density, metric)] = p_value

keys = list(raw_p)
_, p_adj_values, _, _ = multipletests(
    [raw_p[k] for k in keys], alpha=0.05, method="holm"
)
p_adjusted_map = dict(zip(keys, p_adj_values))

fig, axes = plt.subplots(
    len(GRAPH_DENSITIES),
    len(metric_definitions),
    figsize=(8, 8),
    constrained_layout=True,
)

for row, density in enumerate(GRAPH_DENSITIES):
    subset = network_metrics.loc[
        np.isclose(network_metrics["density"], density)
    ]

    for col, (metric, label) in enumerate(metric_definitions):
        ax = axes[col, row]

        sns.boxplot(
            data=subset,
            x="genotype",
            y=metric,
            hue="genotype",
            order=GROUP_ORDER,
            palette=GROUP_COLORS,
            showfliers=False,
            width=0.36,
            linewidth=1.1,
            legend=False,
            ax=ax,
        )

        sns.stripplot(
            data=subset,
            x="genotype",
            y=metric,
            hue="genotype",
            order=GROUP_ORDER,
            palette=GROUP_COLORS,
            jitter=0.14,
            size=4.5,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.8,
            legend=False,
            ax=ax,
        )

        p_adjusted = p_adjusted_map[(density, metric)]

        ax.set_title(
            #f"{label}\n"
            f"Density={density:.2f}; p$_{{Holm}}$={p_adjusted:.3g}",
            fontsize=9,
        )
        ax.set_xlabel("",fontsize=10)
        ax.set_ylabel(label if col == 0 else label,fontsize=12)
        sns.despine(ax=ax)
        # Axis tick-label sizes
        ax.tick_params(axis="x", rotation=0, labelsize=12)
        ax.tick_params(axis="y", labelsize=11)

#fig.suptitle(
#    "Graph-Theory Metrics Across Network Densities",
#    fontsize=10,
#    fontweight="bold",
#)

save_figure(fig, "figure6_graph_metrics_by_density.png")


# ============================================================
# NODE-STRENGTH PROFILE
# ============================================================

print("\n[Figure 7] Whole-brain node strength")

strength_density = 0.20

strength_plot_data = node_strength_data.loc[
    np.isclose(node_strength_data["density"], strength_density)
].copy()

strength_summary = (
    strength_plot_data
    .groupby(["genotype", "roi_name"], observed=False)
    ["node_strength"]
    .agg(["mean", "sem"])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(22, 8))

x = np.arange(N_ROI)
offsets = np.linspace(-0.2, 0.2, len(GROUP_ORDER))

for offset, group in zip(offsets, GROUP_ORDER):
    group_summary = (
        strength_summary.loc[
            strength_summary["genotype"] == group
        ]
        .set_index("roi_name")
        .reindex(roi_names)
    )

    ax.errorbar(
        x + offset,
        group_summary["mean"],
        yerr=group_summary["sem"],
        color=GROUP_COLORS[group],
        marker="o",
        markersize=3.5,
        linewidth=1.1,
        capsize=2,
        label=group,
    )

for roi_index in pf_idx + posterior_idx:
    ax.axvspan(
        roi_index - 0.5,
        roi_index + 0.5,
        color="#F2C14E",
        alpha=0.10,
        zorder=0,
    )

ax.set_xticks(x)
ax.set_xticklabels(roi_names, rotation=90, fontsize=8)
ax.set_ylabel("Node strength")
ax.set_xlabel("ROI")
ax.set_title(
    f"Whole-Brain Node Strength at Density {strength_density:.2f}"
)
ax.legend(frameon=False, title="Group", ncol=len(GROUP_ORDER))
sns.despine(ax=ax)

save_figure(fig, "figure7_node_strength_profile.png")


# ============================================================
# BRAINNET VIEWER EXPORT
# ============================================================

def export_brainnet_edge(matrix, filename):
    """
    Export a pure numeric square matrix for BrainNet Viewer.

    No row names, column names, commas, or index values are written.
    """
    matrix = np.asarray(matrix, dtype=float).copy()
    matrix = np.nan_to_num(matrix, nan=0.0)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)

    path = os.path.join(OUT, filename)

    np.savetxt(
        path,
        matrix,
        fmt="%.8f",
        delimiter="\t",
    )

    print(f"Saved BrainNet edge matrix: {path}")


# ============================================================
# CIRCULAR NETWORK PLOTS
# ============================================================

def circular_network_plot(
    weighted,
    #title,
    filename,
    density,
):
    graph = nx.from_numpy_array(weighted)
    position = nx.circular_layout(graph)

    strengths = weighted.sum(axis=1)

    if np.ptp(strengths) > 0:
        node_sizes = 100 + 900 * (
            strengths - strengths.min()
        ) / np.ptp(strengths)
    else:
        node_sizes = np.full(N_ROI, 250)

    node_colors = [
        "#D6A84B" if i in pf_idx
        else "#7E57C2" if i in posterior_idx
        else "#77AADD"
        for i in range(N_ROI)
    ]

    edges = list(graph.edges(data=True))
    edge_weights = np.array(
        [edge_data["weight"] for _, _, edge_data in edges]
    )

    if len(edge_weights):
        edge_min = edge_weights.min()
        edge_range = np.ptp(edge_weights)

        if edge_range == 0:
            normalized_weights = np.ones(len(edge_weights))
        else:
            normalized_weights = (
                edge_weights - edge_min
            ) / edge_range
    else:
        normalized_weights = np.array([])

    fig, ax = plt.subplots(figsize=(12, 12))

    print(
        #f"{title}: drawing {len(edges)} edges; "
        f"nonzero matrix entries={np.count_nonzero(weighted)}"
    )

    # Draw every edge individually inside the loop to support curvature.
    for edge_index, (node_a, node_b, edge_data) in enumerate(edges):
        normalized = normalized_weights[edge_index]

        drawn_edges = nx.draw_networkx_edges(
            graph,
            position,
            edgelist=[(node_a, node_b)],
            width=0.8 + 3.2 * normalized,
            edge_color=[plt.cm.inferno(0.20 + 0.70 * normalized)],
            alpha=0.30 + 0.65 * normalized,
            connectionstyle="arc3,rad=0.25",
            arrows=True,
            arrowstyle="-",
            min_source_margin=3,
            min_target_margin=3,
            ax=ax,
        )

        if drawn_edges is not None:
            if isinstance(drawn_edges, list):
                for artist in drawn_edges:
                    artist.set_zorder(1)
            else:
                drawn_edges.set_zorder(1)

    # Draw nodes after edges so nodes remain visually clean.
    node_collection = nx.draw_networkx_nodes(
        graph,
        position,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white",
        linewidths=0.8,
        ax=ax,
    )

    node_collection.set_zorder(2)

    # Draw ROI labels around the circle.
    for node, (x_pos, y_pos) in position.items():
        angle = np.degrees(np.arctan2(y_pos, x_pos))
        rotation = angle

        if angle < -90 or angle > 90:
            rotation += 180
            horizontal_alignment = "right"
        else:
            horizontal_alignment = "left"

        ax.text(
            1.10 * x_pos,
            1.10 * y_pos,
            roi_names[node],
            rotation=rotation,
            rotation_mode="anchor",
            ha=horizontal_alignment,
            va="center",
            fontsize=17,
        )

   # legend_handles = [
    #    plt.Line2D(
     #       [0], [0],
      #      marker="o",
       #     color="none",
    #        markerfacecolor="#D6A84B",
     #       markeredgecolor="white",
      #      markersize=16,
       #     label="Prefrontal",
      #  ),
      #  plt.Line2D(
      #      [0], [0],
      #      marker="o",
      #      color="none",
      #      markerfacecolor="#7E57C2",
      #      markeredgecolor="white",
      #      markersize=16,
      #      label="Posterior / somatosensory",
      #  ),
      #  plt.Line2D(
       #   #  [0], [0],
         #   marker="o",
         #   color="none",
        #    markerfacecolor="#77AADD",
       #     markeredgecolor="white",
      #      markersize=16,
     #       label="Other ROI",
    #    ),
  #  ]

   # ax.legend(
     #   handles=legend_handles,
    #    loc="upper right",
      #  bbox_to_anchor=(1.12, 1.05),
     #   frameon=False,
    #    fontsize=18,
   # )

  #  ax.set_title(
   #     f"{title}\n"
   #     f"Proportional density = {density:.2f}; "
   #     "node size = node strength",
   #     fontsize=12,
  #      fontweight="bold",
   #     pad=24,
   # )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.32, 1.32)
    ax.set_ylim(-1.32, 1.32)

    save_figure(fig, filename)


# ============================================================
# FIGURE 8: GROUP CIRCULAR NETWORKS
# ============================================================

print("\n[Figure 8] Circular group networks")

for group in GROUP_ORDER:
    # Threshold the group-mean raw correlation matrix at the same
    # density used for node strength and edge visualization.
    mean_matrix = np.nanmean(matrices_r[groups[group]], axis=0)
    np.fill_diagonal(mean_matrix, 0)

    weighted_group, binary_group = proportional_threshold_positive(
        mean_matrix,
        CIRCULAR_DENSITY,
    )

    # Export the exact weighted network displayed in the plot.
    export_brainnet_edge(
        weighted_group,
        f"BrainNet_{group}_density_{CIRCULAR_DENSITY:.2f}.edge",
    )

    circular_network_plot(
        weighted=weighted_group,
        #title=f"{group} A\u03B2 Mean Functional Network",
        filename=(
            f"figure8_circular_network_{group}_"
            f"density_{CIRCULAR_DENSITY:.2f}.png"
        ),
        density=CIRCULAR_DENSITY,
    )


# ============================================================
# GRAPH-METRIC STATISTICS
# ============================================================

print("\n[Statistics] Graph metrics across groups and densities")

graph_stat_rows = []

for density in GRAPH_DENSITIES:
    density_data = network_metrics.loc[
        np.isclose(network_metrics["density"], density)
    ]

    for metric, metric_label in metric_definitions:
        samples = [
            density_data.loc[
                density_data["genotype"] == group,
                metric,
            ].dropna().to_numpy()
            for group in GROUP_ORDER
        ]

        permute_f, permute_p = welch_anova_permutation(samples, n_permutations=10000, random_seed=RANDOM_SEED)

        graph_stat_rows.append({
            "density": density,
            "metric": metric,
            "test": "Welch ANOVA",
            "comparison": "All groups (omnibus)",
            "statistic": permute_f,
            "p_raw": permute_p,
        })

        for group_a, group_b in PAIRWISE_COMPARISONS:
            values_a = density_data.loc[
                density_data["genotype"] == group_a, metric
            ].dropna().to_numpy()
            values_b = density_data.loc[
                density_data["genotype"] == group_b, metric
            ].dropna().to_numpy()

            t_value, p_value = welch_t_permutation(
                values_a,
                values_b,
                n_permutations=10000,
                random_seed=RANDOM_SEED,
            )

            graph_stat_rows.append({
                "density": density,
                "metric": metric,
                "test": "Welch t-test",
                "comparison": f"{group_a} vs {group_b}",
                "statistic": t_value,
                "p_raw": p_value,
            })

graph_statistics = pd.DataFrame(graph_stat_rows)

# FDR correction separately for omnibus and pairwise metric tests.
graph_statistics["q_FDR"] = np.nan
graph_statistics["FDR_significant"] = False

for test_name in graph_statistics["test"].unique():
    test_mask = graph_statistics["test"] == test_name

    reject, corrected_p, _, _ = multipletests(
        graph_statistics.loc[test_mask, "p_raw"].to_numpy(),
        alpha=0.05,
        method="fdr_bh",
    )

    graph_statistics.loc[test_mask, "q_FDR"] = corrected_p
    graph_statistics.loc[test_mask, "FDR_significant"] = reject

graph_statistics.to_csv(
    os.path.join(OUT, "graph_metric_statistics.csv"),
    index=False,
)

print(
    "Saved:",
    os.path.join(OUT, "graph_metric_statistics.csv"),
)


# ============================================================
# NETWORK-BASED STATISTIC (one-sided contrasts, standard NBS)
# ============================================================

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def welch_t_vectorized(group_a_edges, group_b_edges):
    """
    Edgewise Welch t statistics (subjects x edges inputs).
    Positive values indicate greater connectivity in group A.
    """
    group_a_edges = np.asarray(group_a_edges, dtype=float)
    group_b_edges = np.asarray(group_b_edges, dtype=float)

    n_a = group_a_edges.shape[0]
    n_b = group_b_edges.shape[0]

    mean_a = np.nanmean(group_a_edges, axis=0)
    mean_b = np.nanmean(group_b_edges, axis=0)

    variance_a = np.nanvar(group_a_edges, axis=0, ddof=1)
    variance_b = np.nanvar(group_b_edges, axis=0, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = (mean_a - mean_b) / np.sqrt(
            variance_a / n_a + variance_b / n_b
        )

    return np.nan_to_num(t_values, nan=0.0, posinf=0.0, neginf=0.0)


def _label_suprathreshold_edges(t_values, threshold, tail):
    """
    One-sided suprathreshold set.
    tail = +1 keeps t > threshold  (A > B contrast)
    tail = -1 keeps t < -threshold (A < B contrast)

    Returns the suprathreshold edge indices and the connected-component
    label of each of those edges.
    """
    suprathreshold = (tail * np.asarray(t_values)) > threshold
    supra_idx = np.flatnonzero(suprathreshold)

    if supra_idx.size == 0:
        return supra_idx, None

    adjacency = coo_matrix(
        (
            np.ones(supra_idx.size),
            (triu_i[supra_idx], triu_j[supra_idx]),
        ),
        shape=(N_ROI, N_ROI),
    )

    _, node_labels = connected_components(adjacency, directed=False)

    # Both endpoints of an edge share a label.
    return supra_idx, node_labels[triu_i[supra_idx]]


def max_component_extent(t_values, threshold, tail):
    """Largest component extent (number of edges); used in permutations."""
    _, edge_labels = _label_suprathreshold_edges(t_values, threshold, tail)

    if edge_labels is None:
        return 0

    return int(np.bincount(edge_labels).max())


def suprathreshold_components(t_values, threshold, tail):
    """
    Connected components of a one-sided suprathreshold network.
    Component extent is the number of suprathreshold edges.
    """
    supra_idx, edge_labels = _label_suprathreshold_edges(
        t_values, threshold, tail
    )

    if edge_labels is None:
        return []

    components = []

    for label in np.unique(edge_labels):
        edge_indices = supra_idx[edge_labels == label]
        nodes = np.unique(
            np.concatenate([triu_i[edge_indices], triu_j[edge_indices]])
        )

        components.append({
            "nodes": nodes.tolist(),
            "edge_indices": edge_indices,
            "extent": int(edge_indices.size),
            "tail": "positive" if tail > 0 else "negative",
        })

    components.sort(key=lambda c: c["extent"], reverse=True)

    return components


NBS_TAILS = {"positive": 1, "negative": -1}


def run_nbs(
    matrices_group_a,
    matrices_group_b,
    primary_threshold=3.1,
    n_permutations=5000,
    random_seed=20260911,
):
    """
    Standard NBS: two one-sided contrasts (A > B and A < B), each with
    its own permutation null of the maximum component extent.

    Returns
    -------
    observed_t : array
        Edgewise Welch t statistics (A - B).
    components : list of dicts
        Observed components from both contrasts, each with
        'tail' and a within-contrast FWER p-value.
    null_max_extents : dict
        {'positive': array, 'negative': array}
    """
    edge_a = matrices_group_a[:, triu_i, triu_j]
    edge_b = matrices_group_b[:, triu_i, triu_j]

    observed_t = welch_t_vectorized(edge_a, edge_b)

    observed_components = {
        name: suprathreshold_components(observed_t, primary_threshold, tail)
        for name, tail in NBS_TAILS.items()
    }

    combined = np.vstack([edge_a, edge_b])
    n_group_a = edge_a.shape[0]
    n_total = combined.shape[0]

    rng = np.random.default_rng(random_seed)
    null_max_extents = {
        name: np.zeros(n_permutations, dtype=int) for name in NBS_TAILS
    }
    # Suprathreshold edge counts under the null, for the chance baseline.
    null_supra_counts = {
        name: np.zeros(n_permutations, dtype=int) for name in NBS_TAILS
    }

    for permutation in range(n_permutations):
        shuffled = rng.permutation(n_total)

        permuted_t = welch_t_vectorized(
            combined[shuffled[:n_group_a]],
            combined[shuffled[n_group_a:]],
        )

        # Same relabelling feeds both contrasts.
        for name, tail in NBS_TAILS.items():
            null_max_extents[name][permutation] = max_component_extent(
                permuted_t, primary_threshold, tail
            )
            null_supra_counts[name][permutation] = int(
                np.sum(tail * permuted_t > primary_threshold)
            )

        if (permutation + 1) % 1000 == 0 or permutation == n_permutations - 1:
            print(f"    Permutation {permutation + 1} of {n_permutations}")

    components = []

    for name in NBS_TAILS:
        null = null_max_extents[name]

        for component in observed_components[name]:
            component["p_FWER"] = (
                1 + np.sum(null >= component["extent"])
            ) / (n_permutations + 1)
            components.append(component)

    components.sort(key=lambda c: c["extent"], reverse=True)

    # Returned alongside the max-extent nulls without changing their layout:
    # null_max_extents[name] is still the per-permutation max extent array.
    null_max_extents["supra_counts"] = null_supra_counts

    return observed_t, components, null_max_extents


# ============================================================
# RUN NBS FOR ALL PLANNED PAIRWISE COMPARISONS
# ============================================================

def critical_extent(null, alpha):
    """
    Smallest component extent that would reach p < alpha against this null,
    using the same (1 + count) / (N + 1) rule as the reported p-values.
    """
    null = np.asarray(null)
    n = null.size
    for extent in range(1, int(null.max()) + 2):
        if (1 + np.sum(null >= extent)) / (n + 1) < alpha:
            return extent
    return int(null.max()) + 1


print("\n[NBS] Pairwise network-based statistics (one-sided contrasts)")

# Each planned comparison is tested in both directions.
N_NBS_CONTRASTS = len(PAIRWISE_COMPARISONS) * len(NBS_TAILS)

nbs_summary_rows = []
nbs_edge_rows = []
nbs_test_rows = []
nbs_results = {}

for comparison_number, (group_a, group_b) in enumerate(
    PAIRWISE_COMPARISONS,
    start=1,
):
    comparison_name = f"{group_a}_vs_{group_b}"
    contrast_labels = {
        "positive": f"{group_a} > {group_b}",
        "negative": f"{group_a} < {group_b}",
    }

    print(f"\n  NBS comparison: {group_a} vs {group_b}")
    print(
        f"  Sample sizes: {group_a}={groups[group_a].sum()}, "
        f"{group_b}={groups[group_b].sum()}"
    )
    print(f"  Primary threshold: t > {NBS_T_THRESHOLD:.2f} (each direction)")

    observed_t, components, null_distribution = run_nbs(
        matrices_group_a=matrices_z[groups[group_a]],
        matrices_group_b=matrices_z[groups[group_b]],
        primary_threshold=NBS_T_THRESHOLD,
        n_permutations=NBS_PERMUTATIONS,
        random_seed=RANDOM_SEED + comparison_number,
    )

    nbs_results[(group_a, group_b)] = {
        "t": observed_t,
        "components": components,
        "null": null_distribution,
    }

    # Save both null distributions for reproducibility.
    pd.DataFrame({
        f"max_extent_{contrast_labels[name]}": null_distribution[name]
        for name in NBS_TAILS
    }).to_csv(
        os.path.join(OUT, f"NBS_null_distribution_{comparison_name}.csv"),
        index=False,
    )

    # One row per test (comparison x direction): observed vs chance.
    for name, tail in NBS_TAILS.items():
        tail_components = [c for c in components if c["tail"] == name]
        null = null_distribution[name]
        supra_null = null_distribution["supra_counts"][name]
        significant_components = [
            c for c in tail_components
            if min(1.0, c["p_FWER"] * N_NBS_CONTRASTS) < 0.05
        ]
        largest = tail_components[0] if tail_components else None
        nbs_test_rows.append({
            "comparison": f"{group_a} vs {group_b}",
            "contrast": contrast_labels[name],
            "n_group_a": int(groups[group_a].sum()),
            "n_group_b": int(groups[group_b].sum()),
            "primary_threshold_t": NBS_T_THRESHOLD,
            "n_permutations": NBS_PERMUTATIONS,
            "suprathreshold_edges_observed": int(np.sum(tail * observed_t > NBS_T_THRESHOLD)),
            "suprathreshold_edges_chance_mean": float(supra_null.mean()),
            "suprathreshold_edges_chance_95th": float(np.percentile(supra_null, 95)),
            "largest_component_edges": largest["extent"] if largest else 0,
            "largest_component_p_FWER": largest["p_FWER"] if largest else 1.0,
            "largest_component_p_Bonferroni": (
                min(1.0, largest["p_FWER"] * N_NBS_CONTRASTS) if largest else 1.0),
            "null_max_extent_median": float(np.median(null)),
            "critical_extent_FWER_0.05": critical_extent(null, 0.05),
            "critical_extent_Bonferroni_0.05": critical_extent(null, 0.05 / N_NBS_CONTRASTS),
            "n_significant_components": len(significant_components),
            "n_significant_edges": int(sum(c["extent"] for c in significant_components)),
        })

    # Report contrasts with no suprathreshold component.
    tails_found = {component["tail"] for component in components}

    for name in NBS_TAILS:
        if name not in tails_found:
            nbs_summary_rows.append({
                "comparison": f"{group_a} vs {group_b}",
                "contrast": contrast_labels[name],
                "component": 0,
                "n_nodes": 0,
                "n_edges": 0,
                "p_FWER_within_contrast": 1.0,
                "p_Bonferroni_all_contrasts": 1.0,
                "significant": False,
            })
            print(f"  {contrast_labels[name]}: no suprathreshold component.")

    for component_number, component in enumerate(components, start=1):
        p_adjusted = min(1.0, component["p_FWER"] * N_NBS_CONTRASTS)
        component["p_adjusted"] = p_adjusted
        significant_component = p_adjusted < 0.05
        contrast = contrast_labels[component["tail"]]

        nbs_summary_rows.append({
            "comparison": f"{group_a} vs {group_b}",
            "contrast": contrast,
            "component": component_number,
            "n_nodes": len(component["nodes"]),
            "n_edges": component["extent"],
            "p_FWER_within_contrast": component["p_FWER"],
            "p_Bonferroni_all_contrasts": p_adjusted,
            "significant": significant_component,
        })

        for edge_index in component["edge_indices"]:
            i = triu_i[edge_index]
            j = triu_j[edge_index]

            nbs_edge_rows.append({
                "comparison": f"{group_a} vs {group_b}",
                "contrast": contrast,
                "component": component_number,
                "ROI_1": roi_names[i],
                "ROI_2": roi_names[j],
                "ROI_1_index": i + 1,
                "ROI_2_index": j + 1,
                "t_statistic": observed_t[edge_index],
                "p_FWER_within_contrast": component["p_FWER"],
                "p_Bonferroni_all_contrasts": p_adjusted,
                "significant": significant_component,
            })

        print(
            f"  Component {component_number} ({contrast}): "
            f"{len(component['nodes'])} nodes, "
            f"{component['extent']} edges, "
            f"p-FWER={component['p_FWER']:.4f}, "
            f"p-adjusted={p_adjusted:.4f}"
        )


nbs_tests = pd.DataFrame(nbs_test_rows)
nbs_tests.to_csv(os.path.join(OUT, "NBS_test_summary.csv"), index=False)
print("\nNBS per-test summary:")
print(nbs_tests[["contrast", "suprathreshold_edges_observed",
                 "suprathreshold_edges_chance_mean", "largest_component_edges",
                 "critical_extent_Bonferroni_0.05", "n_significant_edges"]]
      .to_string(index=False))

nbs_summary = pd.DataFrame(nbs_summary_rows)

nbs_summary.to_csv(
    os.path.join(OUT, "NBS_component_summary.csv"),
    index=False,
)

nbs_edge_columns = [
    "comparison",
    "contrast",
    "component",
    "ROI_1",
    "ROI_2",
    "ROI_1_index",
    "ROI_2_index",
    "t_statistic",
    "p_FWER_within_contrast",
    "p_Bonferroni_all_contrasts",
    "significant",
]

if nbs_edge_rows:
    nbs_edges = pd.DataFrame(nbs_edge_rows)
else:
    nbs_edges = pd.DataFrame(columns=nbs_edge_columns)

nbs_edges.to_csv(
    os.path.join(OUT, "NBS_component_edges.csv"),
    index=False,
)


# ============================================================
# EXPORT SIGNIFICANT NBS COMPONENTS FOR BRAINNET VIEWER
# ============================================================

print("\n[NBS] Exporting significant BrainNet edge matrices")

for group_a, group_b in PAIRWISE_COMPARISONS:
    comparison_name = f"{group_a}_vs_{group_b}"
    result = nbs_results[(group_a, group_b)]

    t_values = result["t"]
    components = result["components"]

    significant_t_matrix = np.zeros(
        (N_ROI, N_ROI),
        dtype=float,
    )

    significant_binary_matrix = np.zeros(
        (N_ROI, N_ROI),
        dtype=float,
    )

    significant_component_count = 0

    for component_number, component in enumerate(
        components,
        start=1,
    ):
        adjusted_p = component[
            "p_adjusted"
        ]

        if adjusted_p >= 0.05:
            continue

        significant_component_count += 1

        component_t_matrix = np.zeros(
            (N_ROI, N_ROI),
            dtype=float,
        )

        component_binary_matrix = np.zeros(
            (N_ROI, N_ROI),
            dtype=float,
        )

        for edge_index in component["edge_indices"]:
            i = triu_i[edge_index]
            j = triu_j[edge_index]
            t_value = t_values[edge_index]

            component_t_matrix[i, j] = t_value
            component_t_matrix[j, i] = t_value

            component_binary_matrix[i, j] = 1
            component_binary_matrix[j, i] = 1

            significant_t_matrix[i, j] = t_value
            significant_t_matrix[j, i] = t_value

            significant_binary_matrix[i, j] = 1
            significant_binary_matrix[j, i] = 1

        export_brainnet_edge(
            component_t_matrix,
            (
                f"NBS_{comparison_name}_component_"
                f"{component_number}_tstats.edge"
            ),
        )

        export_brainnet_edge(
            component_binary_matrix,
            (
                f"NBS_{comparison_name}_component_"
                f"{component_number}_binary.edge"
            ),
        )

    if significant_component_count > 0:
        export_brainnet_edge(
            significant_t_matrix,
            f"NBS_{comparison_name}_all_significant_tstats.edge",
        )

        export_brainnet_edge(
            significant_binary_matrix,
            f"NBS_{comparison_name}_all_significant_binary.edge",
        )
    else:
        print(
            f"  {comparison_name}: no component survived "
            "correction across all one-sided NBS contrasts."
        )


# ============================================================
# FIGURE 9: NBS COMPONENT MATRICES
# ============================================================

print("\n[Figure 9] NBS component maps")

n_comparisons = len(PAIRWISE_COMPARISONS)
n_cols = 2
n_rows = int(np.ceil(n_comparisons / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(n_cols * 6, n_rows * 6),
    constrained_layout=True,
)

axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

for ax_idx, (group_a, group_b) in enumerate(PAIRWISE_COMPARISONS):
    ax = axes_flat[ax_idx]
    
    comparison_name = f"{group_a}_vs_{group_b}"
    result = nbs_results[(group_a, group_b)]

    plot_matrix = np.zeros((N_ROI, N_ROI), dtype=float)

    for component in result["components"]:
        # Display only components significant after accounting for
        # all eight planned comparisons.
        if (
            component["p_adjusted"]
            >= 0.05
        ):
            continue

        for edge_index in component["edge_indices"]:
            i = triu_i[edge_index]
            j = triu_j[edge_index]
            t_value = result["t"][edge_index]

            plot_matrix[i, j] = t_value
            plot_matrix[j, i] = t_value

    finite_nonzero = plot_matrix[
        np.isfinite(plot_matrix) & (plot_matrix != 0)
    ]

    if len(finite_nonzero):
        limit = np.max(np.abs(finite_nonzero))
    else:
        limit = NBS_T_THRESHOLD + 0.5

    sns.heatmap(
        plot_matrix,
        mask=np.triu(
            np.ones((N_ROI, N_ROI), dtype=bool),
            k=1,
        ),
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        square=True,
        cbar=True,
        cbar_kws={
            "label": f"t statistic: {group_a} − {group_b}",
            "shrink": 0.68,
        },
        ax=ax,
    )

    significant_components = [
        component
        for component in result["components"]
        if component["p_adjusted"] < 0.05
    ]

    ax.set_title(
        f"{group_a} vs {group_b}\n"
        f"{len(significant_components)} significant component(s)",
        fontsize=10,
    )
    ax.tick_params(axis="x", rotation=90, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

# Hide any unused panels.
for unused_ax in axes_flat[n_comparisons:]:
    unused_ax.set_visible(False)
    
fig.suptitle(
    "Network-Based Statistic: Significant Components",
    fontsize=10,
    fontweight="bold",
)

save_figure(fig, "figure9_NBS_significant_components.png")


# ============================================================
# FIGURE 10: NBS CIRCULAR NETWORKS
# ============================================================

print("\n[Figure 10] NBS circular networks")

for group_a, group_b in PAIRWISE_COMPARISONS:
    result = nbs_results[(group_a, group_b)]
    comparison_name = f"{group_a}_vs_{group_b}"

    nbs_weighted = np.zeros((N_ROI, N_ROI), dtype=float)

    for component in result["components"]:
        if (
            component["p_adjusted"]
            >= 0.05
        ):
            continue

        for edge_index in component["edge_indices"]:
            i = triu_i[edge_index]
            j = triu_j[edge_index]

            # Absolute t statistics control node strength and edge
            # width; edge direction remains available in the CSV
            # and signed BrainNet matrix.
            weight = abs(result["t"][edge_index])

            nbs_weighted[i, j] = weight
            nbs_weighted[j, i] = weight

    if np.count_nonzero(nbs_weighted) == 0:
        print(
            f"  No NBS circular plot generated for "
            f"{comparison_name}: no significant component."
        )
        continue

    circular_network_plot(
        weighted=nbs_weighted,
        #title=f"NBS Network: {group_a} vs {group_b}",
        filename=(
            f"figure10_NBS_circular_{comparison_name}.png"
        ),
        density=NBS_T_THRESHOLD,
    )


# ============================================================
# ANALYSIS MANIFEST
# ============================================================

manifest_lines = [
    "Abeta human functional-connectivity analysis",
    f"Groups: {', '.join(GROUP_ORDER)}",
    f"Subjects: {len(meta)}",
    f"ROIs: {N_ROI}",
    f"Unique edges: {N_EDGES}",
    (
        "Group sizes: "
        + ", ".join(
            f"{group}={groups[group].sum()}"
            for group in GROUP_ORDER
        )
    ),
    (
        "Graph densities: "
        + ", ".join(f"{density:.2f}" for density in GRAPH_DENSITIES)
    ),
    f"Circular-network density: {CIRCULAR_DENSITY:.2f}",
    f"NBS primary threshold: t > {NBS_T_THRESHOLD:.2f} per one-sided contrast",
    f"NBS permutations: {NBS_PERMUTATIONS}",
    (
        "NBS inference: component extent, one-sided contrasts "
        "(A > B and A < B), maximum-component FWER per contrast"
    ),
    (
        "Additional NBS multiplicity adjustment: Bonferroni across "
        f"{len(PAIRWISE_COMPARISONS) * 2} contrasts"
    ),
    (
        "BrainNet .edge files: tab-delimited numeric matrices "
        "without row or column labels"
    ),
]

with open(
    os.path.join(OUT, "analysis_manifest.txt"),
    "w",
    encoding="utf-8",
) as manifest_file:
    manifest_file.write("\n".join(manifest_lines) + "\n")


# ============================================================
# FINISHED
# ============================================================

print("\n====================================================")
print("Abeta human connectivity analysis completed.")
print(f"Results directory: {os.path.abspath(OUT)}")
print("====================================================")