"""
Group statistics for HMM temporal dynamics (mouse Abeta study).

Python replacement for hmmpca_R_plots_mixedmodels.r, using the same
statistical approach as the static-FC, graph-metric and nodal analyses:

  * omnibus test   Welch ANOVA across the 7 groups, p by permutation
  * planned tests  the same 8 Welch t contrasts, p by permutation,
                   BH-FDR pooled across contrasts (and states) per outcome
  * per-state      each HMM state analysed separately, Holm across states
  * state post hoc within each group, every pair of states compared on FO and
                   dwell time. The same animal contributes to every state it
                   visited, so these are paired: within-animal differences,
                   sign-flip permutation, BH pooled across state pairs x groups
                   per outcome

Outcomes
  switching rate, maximum FO         one value per subject
  fractional occupancy (per state)   raw FO; permutation removes the need
                                     for the logit transform
  dwell time (per state)             log10(mean dwell), entrants only
  state presence (per state)         proportion entering the state; tested
                                     with a permutation chi-square, which
                                     stays valid when cells are all-0/all-1

Why not the R mixed models
  * Tukey tests all 21 pairs, including uninterpretable ones (different
    genotype AND age), and assumes equal variances and normality.
  * FO sums to 1 across states within each subject, so the group main effect
    is fixed by construction and the random intercept cannot represent the
    negative dependence between states; per-state tests avoid both problems.
  * The presence GLMM fails under complete separation (every animal in a
    cell enters the state), which is common here.
  * Empty group x state cells make emmeans contrasts non-estimable; the
    per-state tests report such cells as untestable instead of failing.
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from welch_permutation import welch_anova_permutation, welch_t_permutation

# ---------------- configuration ----------------
DATA_DIR = Path(os.environ.get("HMM_DATA_DIR", "../signals"))
SELECTED_K = 3
SELECTED_PCA = 50            # must match the model used for the figures
N_PERMUTATIONS = int(os.environ.get("HMM_N_PERM", 10000))
RANDOM_SEED = 20260911        # same seed as the HMM plotter, so titles and tables agree
OUT_DIR = DATA_DIR / f"hmm_dynamics_stats_hmmpca{SELECTED_PCA}_k{SELECTED_K}"

GROUP_ORDER = ["NTG_Y", "NTG_M", "NTG_A", "CRND8_Y", "CRND8_M", "5XFAD_Y", "5XFAD_M"]
PAIRWISE_COMPARISONS = [
    ("NTG_Y", "NTG_M"), ("NTG_Y", "NTG_A"),
    ("NTG_Y", "CRND8_Y"), ("NTG_M", "CRND8_M"),
    ("NTG_Y", "5XFAD_Y"), ("NTG_M", "5XFAD_M"),
    ("CRND8_Y", "CRND8_M"), ("5XFAD_Y", "5XFAD_M"),
]


# ---------------- loading ----------------
def load(name):
    frame = pd.read_csv(DATA_DIR / name)
    frame = frame[(frame["K_requested"] == SELECTED_K)
                  & (frame["pca_components"] == SELECTED_PCA)].copy()
    if frame.empty:
        raise ValueError(f"No rows in {name} for K={SELECTED_K}, PCA={SELECTED_PCA}.")
    frame["group"] = frame["group"].astype(str).str.strip()
    unexpected = sorted(set(frame["group"]) - set(GROUP_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected groups in {name}: {unexpected}")
    return frame


def samples_by_group(frame, column, transform=None):
    out = []
    for group in GROUP_ORDER:
        x = frame.loc[frame["group"] == group, column].to_numpy(float)
        x = x[np.isfinite(x)]
        if transform is not None:
            x = transform(x)
        out.append(x)
    return out


def testable(samples):
    return all(len(x) >= 2 and np.var(x, ddof=1) > 0 for x in samples)


# ---------------- tests ----------------
def omnibus(frame, column, transform=None):
    samples = samples_by_group(frame, column, transform)
    if not testable(samples):
        return np.nan, np.nan
    return welch_anova_permutation(samples, N_PERMUTATIONS, RANDOM_SEED)


def planned(frame, column, transform=None):
    rows = []
    for group_a, group_b in PAIRWISE_COMPARISONS:
        a = frame.loc[frame["group"] == group_a, column].to_numpy(float)
        b = frame.loc[frame["group"] == group_b, column].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if transform is not None:
            a, b = transform(a), transform(b)
        if len(a) < 2 or len(b) < 2 or (np.var(a, ddof=1) == 0 and np.var(b, ddof=1) == 0):
            t, p = np.nan, np.nan
        else:
            t, p = welch_t_permutation(a, b, N_PERMUTATIONS, RANDOM_SEED)
        rows.append({"comparison": f"{group_a} vs {group_b}",
                     "n_a": len(a), "n_b": len(b),
                     "mean_a": a.mean() if len(a) else np.nan,
                     "mean_b": b.mean() if len(b) else np.nan,
                     "t": t, "p_permutation": p})
    return rows


def welch_df(samples):
    """Numerator and Welch-approximated denominator df for Welch's ANOVA."""
    n = np.array([len(x) for x in samples], dtype=float)
    v = np.array([np.var(x, ddof=1) for x in samples])
    k = len(samples)
    w = n / v
    lam = (((1 - w / w.sum()) ** 2) / (n - 1)).sum()
    return k - 1, (k ** 2 - 1) / (3 * lam)


def paired_signflip_test(differences):
    """
    Two-sided permutation test that the mean within-animal difference is 0,
    using the paired t statistic and random sign flips. All sign patterns are
    enumerated exactly when there are few animals (2**n <= N_PERMUTATIONS).
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


def state_comparisons(frame, column, state_order, transform=None):
    """Paired comparisons of every pair of states within each group."""
    values = frame[["subject_id", "group", "state", column]].copy()
    if transform is not None:
        values[column] = transform(values[column].to_numpy(float))
    wide = values.pivot_table(index=["subject_id", "group"], columns="state",
                              values=column).reset_index()
    rows = []
    for group in GROUP_ORDER:
        sub = wide[wide["group"] == group]
        for state_a, state_b in combinations(state_order, 2):
            if state_a not in sub.columns or state_b not in sub.columns:
                continue
            pair = sub[[state_a, state_b]].dropna()
            differences = (pair[state_a] - pair[state_b]).to_numpy(float)
            t, p, n = paired_signflip_test(differences)
            rows.append({"group": group, "comparison": f"{state_a} vs {state_b}",
                         "n_paired_animals": n,
                         "mean_a": pair[state_a].mean() if n else np.nan,
                         "mean_b": pair[state_b].mean() if n else np.nan,
                         "mean_difference": differences.mean() if n else np.nan,
                         "t_paired": t, "p_permutation": p})
    return rows


def presence_test(frame):
    """
    Permutation chi-square for group x (entered / did not enter).
    Valid with all-0 or all-1 cells, unlike the binomial GLMM.
    """
    present = (frame["n_dwell_episodes"] > 0).to_numpy(int)
    labels = frame["group"].to_numpy()
    if present.min() == present.max():
        return np.nan, np.nan          # everyone (or no one) entered: nothing to test

    def chi2(label_vector):
        table = pd.crosstab(label_vector, present).reindex(GROUP_ORDER).fillna(0).to_numpy()
        table = table[table.sum(1) > 0]
        expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nansum((table - expected) ** 2 / expected)

    observed = chi2(labels)
    rng = np.random.default_rng(RANDOM_SEED)
    exceed = sum(chi2(rng.permutation(labels)) >= observed for _ in range(N_PERMUTATIONS))
    return observed, (1 + exceed) / (N_PERMUTATIONS + 1)


def bh(values):
    values = np.asarray(values, float)
    q = np.full_like(values, np.nan)
    ok = np.isfinite(values)
    if ok.any():
        q[ok] = multipletests(values[ok], method="fdr_bh")[1]
    return q


def holm(values):
    values = np.asarray(values, float)
    out = np.full_like(values, np.nan)
    ok = np.isfinite(values)
    if ok.any():
        out[ok] = multipletests(values[ok], method="holm")[1]
    return out

SUMMARY_COLUMNS = [
    "fractional_occupancy", "map_TR_count", "n_dwell_episodes",
    "mean_dwell_TR", "median_dwell_TR", "mean_dwell_seconds", "median_dwell_seconds",
]
# Dwell measures are undefined for animals that never entered a state, so they
# are summarised over entrants only; occupancy and counts use every animal.
DWELL_COLUMNS = {"mean_dwell_TR", "median_dwell_TR",
                 "mean_dwell_seconds", "median_dwell_seconds"}
# Decimals per column; occupancy is a proportion, so it gets two.
COLUMN_DECIMALS = {"fractional_occupancy": 2}
 
 
def median_range_table(frame, by=("state",), decimals=1):
    """
    One row per level of `by`, each metric formatted as 'median [min-max]'.
    Returns (formatted table, numeric table with median/min/max columns).
    """
    columns = [c for c in SUMMARY_COLUMNS if c in frame.columns]
    formatted, numeric = [], []
 
    for keys, group in frame.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row_f = dict(zip(by, keys))
        row_n = dict(zip(by, keys))
        row_f["n_animals"] = row_n["n_animals"] = len(group)
 
        for column in columns:
            values = group[column]
            if column in DWELL_COLUMNS:
                values = values[group["n_dwell_episodes"] > 0]
            values = values.dropna().to_numpy(float)
 
            if values.size == 0:
                row_f[column] = "NA"
                median = low = high = np.nan
            else:
                median, low, high = np.median(values), values.min(), values.max()
                d = COLUMN_DECIMALS.get(column, decimals)
                row_f[column] = f"{median:.{d}f} [{low:.{d}f}–{high:.{d}f}]"
            row_n.update({f"{column}_median": median, f"{column}_min": low,
                          f"{column}_max": high})
 
        formatted.append(row_f)
        numeric.append(row_n)
 
    return pd.DataFrame(formatted), pd.DataFrame(numeric)
    
# ---------------- main ----------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    switching = load("pcahmm_switching_rates_all_K_all_PCA.csv")
    maxfo = load("pcahmm_maxFO_all_K_all_PCA.csv")
    states = load("pcahmm_dwell_times_all_K_all_PCA.csv")
    state_order = sorted(states["state"].unique())

    summary_state, summary_state_numeric = median_range_table(states, by=("state",))
 
    summary_state.to_csv(OUT_DIR / "dynamic_metrics_median_range_by_state.csv", index=False)
    summary_state_numeric.to_csv(OUT_DIR / "dynamic_metrics_median_range_by_state_numeric.csv",
                             index=False)
 
# Same table split by group, useful for a supplementary table
    summary_group, summary_group_numeric = median_range_table(states, by=("state", "group"))
    summary_group["group"] = pd.Categorical(summary_group["group"], GROUP_ORDER, ordered=True)
    summary_group = summary_group.sort_values(["state", "group"])
    summary_group.to_csv(OUT_DIR / "dynamic_metrics_median_range_by_state_group.csv", index=False)
 
    print("\nDynamic metrics, median [range] by state:")
    print(summary_state.to_string(index=False))

    print(f"K={SELECTED_K}, PCA={SELECTED_PCA}, {N_PERMUTATIONS} permutations")
    print(switching.groupby("group").size().reindex(GROUP_ORDER).to_string())

    omnibus_rows, planned_rows = [], []

    # subject-level outcomes
    for outcome, frame, column in [("switching_rate", switching, "switching_rate"),
                                   ("max_FO", maxfo, "max_FO")]:
        F, p = omnibus(frame, column)
        samples = samples_by_group(frame, column)
        df1, df2 = welch_df(samples) if testable(samples) else (np.nan, np.nan)
        omnibus_rows.append({"outcome": outcome, "state": "all", "test": "Welch ANOVA",
                             "statistic": F, "df1": df1, "df2_Welch": df2,
                             "p_permutation": p, "p_Holm": p})
        for row in planned(frame, column):
            planned_rows.append({"outcome": outcome, "state": "all", **row})

    # per-state outcomes
    dwell_frame = states[(states["n_dwell_episodes"] > 0)
                         & np.isfinite(states["mean_dwell_seconds"])
                         & (states["mean_dwell_seconds"] > 0)]
    per_state = [
        ("fractional_occupancy", states, "fractional_occupancy", None, "Welch ANOVA"),
        ("dwell_time_log10", dwell_frame, "mean_dwell_seconds", np.log10, "Welch ANOVA on log10"),
    ]
    for outcome, frame, column, transform, test in per_state:
        block = []
        for state in state_order:
            sub = frame[frame["state"] == state]
            F, p = omnibus(sub, column, transform)
            samples = samples_by_group(sub, column, transform)
            df1, df2 = welch_df(samples) if testable(samples) else (np.nan, np.nan)
            block.append({"outcome": outcome, "state": state, "test": test,
                          "statistic": F, "df1": df1, "df2_Welch": df2,
                          "p_permutation": p})
            for row in planned(sub, column, transform):
                planned_rows.append({"outcome": outcome, "state": state, **row})
        adjusted = holm([r["p_permutation"] for r in block])
        for r, a in zip(block, adjusted):
            r["p_Holm"] = a
        omnibus_rows.extend(block)

    # presence
    block, presence_rows = [], []
    for state in state_order:
        sub = states[states["state"] == state]
        chi2, p = presence_test(sub)
        block.append({"outcome": "state_presence", "state": state,
                      "test": "permutation chi-square", "statistic": chi2, "p_permutation": p})
        for group in GROUP_ORDER:
            g = sub[sub["group"] == group]
            presence_rows.append({"state": state, "group": group, "n": len(g),
                                  "n_present": int((g["n_dwell_episodes"] > 0).sum())})
    adjusted = holm([r["p_permutation"] for r in block])
    for r, a in zip(block, adjusted):
        r["p_Holm"] = a
    omnibus_rows.extend(block)

    omnibus_table = pd.DataFrame(omnibus_rows)
    planned_table = pd.DataFrame(planned_rows)
    planned_table["q_BH_pooled"] = np.nan
    for outcome, idx in planned_table.groupby("outcome").groups.items():
        planned_table.loc[idx, "q_BH_pooled"] = bh(planned_table.loc[idx, "p_permutation"])

    # state comparisons within each group (paired, BH pooled per outcome)
    state_rows = []
    for outcome, frame, column, transform in [
            ("fractional_occupancy", states, "fractional_occupancy", None),
            ("dwell_time_log10", dwell_frame, "mean_dwell_seconds", np.log10)]:
        for row in state_comparisons(frame, column, state_order, transform):
            state_rows.append({"outcome": outcome, **row})
    state_table = pd.DataFrame(state_rows)
    state_table["q_BH_pooled"] = np.nan
    for outcome, idx in state_table.groupby("outcome").groups.items():
        state_table.loc[idx, "q_BH_pooled"] = bh(state_table.loc[idx, "p_permutation"])
    state_table.to_csv(OUT_DIR / "posthoc_state_comparisons_by_group.csv", index=False)

    omnibus_table.to_csv(OUT_DIR / "omnibus_tests.csv", index=False)
    planned_table.to_csv(OUT_DIR / "planned_contrasts.csv", index=False)
    pd.DataFrame(presence_rows).to_csv(OUT_DIR / "state_presence_counts.csv", index=False)

    print("\nOmnibus tests:")
    print(omnibus_table.round(4).to_string(index=False))
    sig = planned_table[planned_table["q_BH_pooled"] < 0.05]
    print(f"\nPlanned contrasts surviving pooled BH: {len(sig)} of "
          f"{planned_table['p_permutation'].notna().sum()} testable")
    if len(sig):
        print(sig[["outcome", "state", "comparison", "t", "p_permutation", "q_BH_pooled"]]
              .round(4).to_string(index=False))
    state_sig = state_table[state_table["q_BH_pooled"] < 0.05]
    print(f"\nState comparisons surviving pooled BH: {len(state_sig)} of "
          f"{state_table['p_permutation'].notna().sum()} testable")
    if len(state_sig):
        print(state_sig[["outcome", "group", "comparison", "n_paired_animals",
                         "mean_difference", "t_paired", "p_permutation", "q_BH_pooled"]]
              .round(4).to_string(index=False))
    print(f"\nResults written to {OUT_DIR}")


if __name__ == "__main__":
    main()