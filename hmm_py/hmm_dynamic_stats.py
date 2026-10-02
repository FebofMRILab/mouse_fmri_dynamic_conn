"""
Group statistics for HMM temporal dynamics (mouse Abeta study).

Python replacement for hmmpca_R_plots_mixedmodels.r, using the same
statistical approach, and the same module (welch_tests_scipy.py, SciPy >= 1.16),
as the static-FC, graph-metric and nodal analyses:

  * omnibus test   Welch ANOVA across the 7 groups, p by permutation
                   (scipy permutation_test of f_oneway(equal_var=False));
                   df from statsmodels anova_oneway
  * planned tests  the same 8 contrasts as ordinary two-sided Welch t tests
                   (scipy ttest_ind, equal_var=False), BH-FDR pooled across
                   contrasts (and states) per outcome
  * per-state      each HMM state analysed separately, Holm across states
  * state post hoc within each group, every pair of states compared on FO and
                   dwell time. The same animal contributes to every state it
                   visited, so these are paired: within-animal differences,
                   sign-flip permutation of the paired t (scipy
                   permutation_test), BH pooled across state pairs x groups
                   per outcome

Model and inputs (hmm_model_selection.py, same settings as hmm_config.m):
  the reported fit, HMM-PCA rank 15, K = 4, from run4d's tables
  <HMM_OUT_BASE>/hmm_{switching_rates,maxFO,dwell_times}_all_K_all_lowrank.csv;
  every table must carry the same fit ID as <model folder>/fit_id.txt.

Outcomes
  switching rate, maximum FO         one value per subject
  fractional occupancy (per state)   raw FO; permutation removes the need
                                     for the logit transform
  dwell time (per state)             log10(mean dwell), entrants only
  state presence (per state)         proportion entering the state; tested
                                     with a permutation chi-square, which
                                     stays valid when cells are all-0/all-1
                                     (scipy permutation_test of the Pearson
                                     chi-square from scipy chi2_contingency)

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

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hmm_model_selection as sel                         # noqa: E402
from welch_tests_scipy import (paired_signflip_test, welch_anova_df,  # noqa: E402
                               welch_anova_permutation, welch_t_test_df)

# ---------------- configuration ----------------
SELECTED_K = sel.K
N_PERMUTATIONS = int(os.environ.get("HMM_N_PERM", 10000))
RANDOM_SEED = 20260911        # same seed as the HMM plotter, so titles and tables agree
OUT_DIR = sel.OUT_BASE / f"hmm_dynamics_stats_{sel.TAG}"
FIT_IDS = set()

GROUP_ORDER = ["NTG_Y", "NTG_M", "NTG_A", "CRND8_Y", "CRND8_M", "5XFAD_Y", "5XFAD_M"]
PAIRWISE_COMPARISONS = [
    ("NTG_Y", "NTG_M"), ("NTG_Y", "NTG_A"),
    ("NTG_Y", "CRND8_Y"), ("NTG_M", "CRND8_M"),
    ("NTG_Y", "5XFAD_Y"), ("NTG_M", "5XFAD_M"),
    ("CRND8_Y", "CRND8_M"), ("5XFAD_Y", "5XFAD_M"),
]


# ---------------- loading ----------------
def load(kind):
    path = sel.table_path(kind)
    frame = sel.select_model(pd.read_csv(path), path.name)
    sel.check_fit_id(frame, path.name, FIT_IDS)
    name = path.name
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
    return welch_anova_permutation(samples, n_permutations=N_PERMUTATIONS,
                                   random_seed=RANDOM_SEED)


def planned(frame, column, transform=None):
    rows = []
    for group_a, group_b in PAIRWISE_COMPARISONS:
        a = frame.loc[frame["group"] == group_a, column].to_numpy(float)
        b = frame.loc[frame["group"] == group_b, column].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if transform is not None:
            a, b = transform(a), transform(b)
        if len(a) < 2 or len(b) < 2 or (np.var(a, ddof=1) == 0 and np.var(b, ddof=1) == 0):
            t, df, p = np.nan, np.nan, np.nan
        else:
            t, df, p = welch_t_test_df(a, b)
        rows.append({"comparison": f"{group_a} vs {group_b}",
                     "n_a": len(a), "n_b": len(b),
                     "mean_a": a.mean() if len(a) else np.nan,
                     "mean_b": b.mean() if len(b) else np.nan,
                     "t": t, "df_Welch": df, "p_welch": p})
    return rows


def welch_df(samples):
    """Numerator and Welch denominator df of Welch's ANOVA (statsmodels)."""
    return welch_anova_df(samples)


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
            t, p, n = paired_signflip_test(differences, N_PERMUTATIONS, RANDOM_SEED)
            rows.append({"group": group, "comparison": f"{state_a} vs {state_b}",
                         "n_paired_animals": n,
                         "mean_a": pair[state_a].mean() if n else np.nan,
                         "mean_b": pair[state_b].mean() if n else np.nan,
                         "mean_difference": differences.mean() if n else np.nan,
                         "t_paired": t, "p_permutation": p})
    return rows


def presence_test(frame):
    """
    Permutation chi-square for group x (entered / did not enter): group labels
    permuted against presence (scipy permutation_test, permutation_type =
    'pairings'), Pearson chi-square from scipy chi2_contingency.
    Valid with all-0 or all-1 cells, unlike the binomial GLMM.
    """
    present = (frame["n_dwell_episodes"] > 0).to_numpy(int)
    codes = frame["group"].map({g: i for i, g in enumerate(GROUP_ORDER)}).to_numpy(int)
    if present.min() == present.max():
        return np.nan, np.nan          # everyone (or no one) entered: nothing to test

    def chi2(group_codes, entered):
        table = np.zeros((len(GROUP_ORDER), 2))
        np.add.at(table, (group_codes, entered), 1)
        table = table[table.sum(1) > 0]
        return stats.chi2_contingency(table, correction=False).statistic

    res = stats.permutation_test((codes, present), chi2, permutation_type="pairings",
                                 alternative="greater", n_resamples=N_PERMUTATIONS,
                                 rng=RANDOM_SEED)
    return float(res.statistic), float(res.pvalue)


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
    switching = load("switching_rates")
    maxfo = load("maxFO")
    states = load("dwell_times")
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

    fit_id = FIT_IDS.pop() if FIT_IDS else "not recorded"
    print(f"{sel.LABEL}, fit {fit_id[:12]}, {N_PERMUTATIONS} permutations")
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
        planned_table.loc[idx, "q_BH_pooled"] = bh(planned_table.loc[idx, "p_welch"])

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

    for t in (omnibus_table, planned_table, state_table):
        t["fit_id"] = fit_id
    state_table.to_csv(OUT_DIR / "posthoc_state_comparisons_by_group.csv", index=False)
    omnibus_table.to_csv(OUT_DIR / "omnibus_tests.csv", index=False)
    planned_table.to_csv(OUT_DIR / "planned_contrasts.csv", index=False)
    pd.DataFrame(presence_rows).assign(fit_id=fit_id).to_csv(
        OUT_DIR / "state_presence_counts.csv", index=False)

    print("\nOmnibus tests:")
    print(omnibus_table.round(4).to_string(index=False))
    sig = planned_table[planned_table["q_BH_pooled"] < 0.05]
    print(f"\nPlanned contrasts surviving pooled BH: {len(sig)} of "
          f"{planned_table['p_welch'].notna().sum()} testable")
    if len(sig):
        print(sig[["outcome", "state", "comparison", "t", "df_Welch", "p_welch", "q_BH_pooled"]]
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