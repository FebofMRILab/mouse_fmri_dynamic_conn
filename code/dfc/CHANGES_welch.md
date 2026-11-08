# DFC plotting: group statistics from the study's Welch module

`plot_dfc_results.py`, and `dfc_group_calibrated.py` through it, now take every group test from `welch_tests_scipy.py`. That is the same module the static-FC scripts use; this folder holds an identical copy. The tests are SciPy and statsmodels functions. The figures, tables and file names are unchanged.

| # | Step | Before | Now | Evidence (`tests/test_dfc_group_stats.py`) |
|---|---|---|---|---|
| W1 | Omnibus across the 7 groups | Welch's F with our own formula and permutation loop | `welch_anova_permutation`: `scipy.stats.permutation_test` on `f_oneway(equal_var=False)`, 10,000 permutations. The df shown in Table S-B come from statsmodels `anova_oneway(use_var='unequal')`. | F identical (1.4e-16), df identical (7e-15). The permutation p differs only by Monte Carlo error (max 0.014 here). |
| W2 | 8 planned contrasts | Permutation p of Welch's t, our own loop | Ordinary two-sided Welch t test: `welch_t_test_df`, i.e. `scipy.stats.ttest_ind(equal_var=False)`. Column `p_perm` is now `p_welch`. | t, df and mean difference identical (≤ 1e-14). p moves by about 0.002 (median), at most 0.03. |
| W3 | Holm (omnibus across panels) and BH (contrasts) | Our own functions | statsmodels `multipletests` (`holm`, `fdr_bh`) | Identical (≤ 3e-17) |

**Why W2.** It is the study's decision for every pairwise contrast. With the 5-mouse group the more variable one, permuting Welch's t rejected 10–13% of true nulls, while the ordinary Welch t stayed near 5% (`static_v2/tests/simulate_type1_two_groups.py`).

**Unchanged:**
- The 8 contrasts.
- Holm across a figure's panels for the omnibus test, and pooled BH for the contrasts.
- The NaN rule for groups with fewer than 2 values or no spread.

The calibrated non-stationarity tests (one-sided Wilcoxon of observed vs replica z) compare each mouse with its own replica, not groups with each other, so they are unchanged.

**Requires** SciPy ≥ 1.16 and statsmodels. The `fc_static` environment from `hipergator_tests/setup_python_env.sh` has both.

**Run the test** from this folder:

```bash
python tests/test_dfc_group_stats.py
```

It compares the updated script with the previous one, which is kept in `tests/legacy/`. Locally: 9 checks passed, 0 failed. An end-to-end run on the 95-mouse mock data produced all figures, tables and `Tables_dfc.docx`.
