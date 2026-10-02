# HMM statistics and figures: changes

**Rule.** Use published code and adapt our data to it. A deviation is allowed only to fix a documented flaw, and then it comes with the exact code and the evidence.

**Published code these scripts call:**
- **SciPy ≥ 1.16:** `permutation_test`, `f_oneway(equal_var=False)`, `ttest_ind(equal_var=False)`, `ttest_1samp`, `chi2_contingency`.
- **statsmodels:** `anova_oneway`, `multipletests`.
- **python-docx** and **matplotlib** for tables and figures.

All of these are reached through `welch_tests_scipy.py`, the same module (byte-identical copy) used by the static-FC and DFC scripts.

## All four HMM scripts

| # | Before | Now | Evidence or effect |
|---|---|---|---|
| P1 | `SELECTED_PCA = 50`, i.e. the HMM-PCA 50 fit. `hmm_dynamic_stats.py` read `pcahmm_*_all_K_all_PCA.csv`, while the dynamics plotter read `hmm_*_all_K_all_PCA.csv`: two different file sets for the same tables and figures. | One setting for all: `hmm_model_selection.py` uses the same defaults, environment variables and folder names as `hmm_config.m`. The reported fit (HMM-PCA rank 15, K = 4; `hmm_v2/CHANGES.md`, "Reported model") is read from run4d's `hmm_*_all_K_all_lowrank.csv` and the `hmmmar_simple_k4_hmmpca15` folder. | **Check that the current draft's dynamics tables and figures came from the same files.** The two previous scripts read different ones. |
| P2 | No link between a table and its fit | Every table must carry one fit ID, equal to `fit_id.txt`. The ID is written into every output table and manifest. | S7: tables from another fit are refused by all three scripts that read run4d/run5 tables |
| P3 | `HMM_DATA_DIR` named the tables folder (`hmm_dynamic_stats.py`). In `hmm_config.m` the same name is the signal-file folder. | `HMM_OUT_BASE`, as in `hmm_config.m` | One meaning per variable. Its default is `/blue/febo/share/Abeta_Age/rabies/signals`, as in `hmm_config.m`. |
| P4 | Output folders: `R_hmm_analysis_pca50_k3_python`, `hmm_dynamics_stats_hmmpca50_k3`, `hmm_graph_figures` in the working folder, and `bootstrap_supplement` in the working folder | `<HMM_OUT_BASE>/hmm_dynamics_figures_hmmpca15_k4`, `<HMM_OUT_BASE>/hmm_dynamics_stats_hmmpca15_k4`, `<model folder>/hmm_graph_figures`, `<model folder>/bootstrap_supplement` | Outputs sit with the fit they describe |

## Group tests (dynamics statistics, both figure scripts)

| # | Before | Now (published) | Evidence (`tests/test_hmm_python_stats.py`) |
|---|---|---|---|
| G1 | Welch F and permutation p from our own loop (`welch_permutation.py`) | `welch_anova_permutation`: `scipy.stats.permutation_test` on `f_oneway(equal_var=False)` | S1, S5: F identical (≤ 5e-15). **p identical too** (max \|Δp\| = 0): for the same seed, SciPy draws the same permutations as the previous loop. |
| G2 | Welch denominator df from our own formula | `welch_anova_df`: statsmodels `anova_oneway(use_var='unequal')` | S1, S5: identical (≤ 5e-15) |
| G3 | 8 planned contrasts: permutation p of Welch's t | Ordinary two-sided Welch t: `welch_t_test_df`, i.e. `scipy.stats.ttest_ind(equal_var=False)` (your decision). Column `p_permutation` is now `p_welch`, and `df_Welch` is added. | S2, S5: t identical (≤ 9e-16). p moved by about 0.005 (median), at most 0.06. |
| G4 | State comparisons within animals: our own sign-flip loop on the paired t | `paired_signflip_test`: `scipy.stats.permutation_test`, one sample of differences (`permutation_type='samples'`, i.e. sign flips) with the paired t (`ttest_1samp`). Exact when 2ⁿ ≤ 10,000, as before. | S3, S5: t identical. p identical wherever the test is exact (groups of ≤ 13 mice; 30 tests). For larger groups the difference is Monte Carlo noise (≤ 0.01). |
| G5 | State presence: our own permutation loop on the chi-square | `scipy.stats.permutation_test` (`permutation_type='pairings'`: group labels permuted against presence) on the Pearson chi-square from `scipy.stats.chi2_contingency` | S4: statistic identical; p differs by Monte Carlo noise only |
| G6 | — | Figure titles and statistics tables use the same function and seed | S6: title p equals table p exactly |

**Unchanged:**
- Holm across states, or across states × metrics, for the omnibus tests.
- BH pooled per outcome for contrasts and state comparisons.
- The `log10` dwell test on entrants only.
- The NaN rules.
- Figure styling.

## Script-specific

**`brainnetwork_plotter_Rmigration_bin_transitivity.py`**
- **The bug.** The combined figure's grid was declared as states × metrics (`plt.subplots(n_states, n_metrics)`) but filled as `axes[metric, state]`. With 3 states and 3 metrics this drew one row per metric. With any other K it would stop with an `IndexError`.
- **The fix.** The grid is now declared metrics × states. The picture is identical for K = 3, and the script works for any K.

**`hmm_bootstrap_stability_plotter.py`**
- **Neff threshold.** `MIN_NEFF` 50 → 83, the value in `hmm_config.m` used by run5 and run5B. It can be changed with `--min_neff`.
- **Dwell column.** It now reads the updated run5B CSV, where `mean_dwell_volumes` became `mean_dwell_TR`. The previous plotter stops with an `AttributeError` (no `mean_dwell_volumes`) on the new file (seen in the test).
- **Count label.** `map_count` is labelled "Viterbi volumes" when the CSV comes from the updated run5B.
- **True/false columns.** These are now parsed from text as well: `astype(bool)` turns the strings "true" and "false" both into True.
- **Panel B.** The colour scale now runs 0–100%. With 50–100%, every cell below 50% was drawn white, which is common with Neff ≥ 83.
- **Panel F.** The legend moved below the axis, because it covered the Spearman box.
- **Panel D.** The y-label is now on two lines, because it ran into the panel letter.
- **Not changed (flag).** This script colours groups by genotype and age (grey, blue, orange shades), not with `GROUP_COLORS` from the static figures.

**`roi_group_table.py`** (replaces `statistical_table.r`) is now in `static_v2/` with the static nodal analysis; its changes are in `static_v2/CHANGES.md` (C9).

## Tests

`python tests/test_hmm_python_stats.py`, from this folder. It runs every script on simulated inputs in the updated MATLAB formats (`tests/make_mock_inputs.py`), and runs the previous scripts (`tests/legacy/`, verbatim) on the same data. S1–S8 run at K = 3 (the previous scripts were written for it); S9 runs every updated script at K = 4, the reported model; S10 checks the defaults. **Run here: 14 passed, 0 failed (about 4 min).**

| Check | Result |
|---|---|
| S1 Welch F, updated vs previous | 1e-16 |
| S1 Welch df | 0 |
| S2 contrast t | 4e-16 |
| S3 paired t | 1e-15 |
| S3 exact sign-flip p | 0 |
| S4 presence chi-square | 0 |
| S5 graph script: F, df, t, paired t and exact p | ≤ 5e-15 |
| S6 title p = table p | 0 |
| S7 another fit refused | 3/3 |
| S8 all outputs written | 4/4 |
| S9 K = 4: all four scripts run, four states in the outputs | 4/4 |
| S10 defaults: K = 4, rank 15, data folder on `/blue/febo/share` | pass |
| Omnibus permutation p, updated vs previous (reported) | identical |
| Contrast p, parametric vs permutation (reported) | median 0.005, max 0.06 |
