# Static FC scripts: changes to use published code

**Scripts:**
- `full_analysis_abeta_mice_permutation_transitivity.py`
- `full_analysis_abeta_humans_permutations.py`

**Rule.** Use published code and adapt our data to it. A deviation is allowed only to fix a documented flaw, with the exact code and the evidence.

**Why Welch statistics.** The groups differ greatly in size (5 to 25 mice) and in variance. Every test therefore keeps a Welch (unequal-variance) statistic.

**Published code used:**
- SciPy ≥ 1.16 (`f_oneway(equal_var=False)` was added in 1.16);
- statsmodels;
- networkx;
- bctpy 0.6.1 (Brain Connectivity Toolbox for Python), including its NBS (Zalesky et al. 2010).

## What changed

| # | Step | Before | Now | Effect on results |
|---|---|---|---|---|
| C1 | Welch's F and t | Our own formulas (`welch_permutation.py`) | `scipy.stats.f_oneway(..., equal_var=False)` and `scipy.stats.ttest_ind(..., equal_var=False)` | Identical statistics (S1–S3: ≤ 5e-15; also equal to `statsmodels anova_oneway(use_var='unequal')`) |
| C2 | Omnibus permutation p-values (edgewise 7-group ANOVA, graph metrics, hypothesis network) | Our own permutation loops | `scipy.stats.permutation_test` on Welch's F, via `welch_tests_scipy.py` | Same statistic and null; p differs only by Monte Carlo error (S3b) |
| C2b | Pairwise contrasts (edgewise and graph metrics) | Permutation p of Welch's t (our own loops) | Ordinary Welch t-test, `scipy.stats.ttest_ind(equal_var=False)` (your decision, from S4) | Same t; p from the Welch–Satterthwaite t distribution. For 9 vs 5 mice the p-values moved by up to 0.05 (S3d). |
| C3 | Proportional threshold | Our own code, keeping ceil(p·E) edges | `bct.threshold_proportional` on the positive-only matrix (round(p·E), as in MATLAB BCT) | Mouse: one edge fewer at each density (332/498/664 instead of 333/499/665). Every BCT edge is also in the previous set (S5). Human: unchanged. |
| C4 | Nodal efficiency (mouse) | Our own shortest-path code | `networkx.harmonic_centrality / (n − 1)` | Identical (S6: 4e-16) |
| C5 | NBS | Our own implementation | bctpy `nbs_bct` with the three documented fixes below (`bctpy_nbs_welch.py`), called through `nbs_published.run_nbs` with the same inputs and outputs | Observed components identical; p and null differ only by the random stream (S7–S9, end-to-end runs) |
| C6 | Graph-metric omnibus family (Fig. 6) | Holm | BH-FDR, the same as the statistics table (your decision) | Figure 6 titles now show q_BH |
| C7 | Text | Mouse manifest said "\|t\| > 3.1"; a comment said "eight comparisons" | "t > 3.1 per one-sided contrast"; "16 one-sided contrasts (8 × 2)" | None |
| C8 | Shared module | — | `welch_tests_scipy.py` gains `welch_t_test_df` (the same test, also returning the Welch–Satterthwaite df), `welch_anova_df` (statsmodels `anova_oneway` df, for reporting) and `paired_signflip_test` (`scipy.stats.permutation_test`, for within-animal state comparisons). The DFC, HMM and ROI-table scripts use them; the static scripts do not. | None: S1–S9 rerun after the change, 17 checks, 0 failed |

## C9: `roi_group_table.py` (new; replaces `statistical_table.r`)

| | `statistical_table.r` | `roi_group_table.py` |
|---|---|---|
| Omnibus per ROI | Kruskal–Wallis, BH across ROIs | Welch ANOVA, p from 10,000 permutations, df from statsmodels; BH across ROIs |
| Post hoc | Dunn's test (FSA), all 21 pairs, Holm within each ROI | The 8 planned contrasts, ordinary Welch t; BH pooled across contrasts × ROIs (as for the edgewise contrasts) |
| Kept | — | Input file, group order and names, the mean ± SD table, p formatting with stars, CSV + Word (booktabs) + multipage landscape PDF |

**Why.** The study's decision is to use Welch tests throughout, because of the extreme group sizes and variances. The 8 planned contrasts are the ones every other analysis uses. Dunn's 21 pairs include contrasts that differ in both genotype and age, which is the reason `hmm_dynamic_stats.py` gives for not using Tukey. R is not available here, so the R output was not reproduced. The tests differ anyway.

**Evidence** (`tests/test_roi_group_table.py`, 4 of 4 passed here):
- R1: Welch F and df equal statsmodels `anova_oneway` (≤ 4e-15).
- R2: contrast t, df and p equal `scipy.stats.ttest_ind(equal_var=False)` (≤ 7e-15).
- R3: both BH adjustments equal statsmodels `fdr_bh`.
- R4: the CSV, Word and PDF tables are written.

The previous R script is kept verbatim in `tests/legacy/statistical_table_previous.r`.



## NBS fixes to bctpy (`fixes/F01-F03_nbs.patch`; everything else is bctpy 0.6.1 unchanged)

**F01: Welch t instead of pooled-variance t.**
```python
denom = np.sqrt(np.var(x, ddof=1) / n1 + np.var(y, ddof=1) / n2)
```
- **Reason:** extreme group sizes and variances.
- **Evidence:**
  - Permutation tests of a studentized (Welch) statistic stay asymptotically valid when variances differ; pooled statistics do not (Janssen 1997; Chung & Romano 2013).
  - S4 shows this at our group sizes. With equal means, and the n = 5 group's SD three times the other group's, permuting the pooled t rejects 11–21% of the time. Permuting the Welch t rejects 10–13%.
  - The previous custom NBS already used the Welch t.

**F02: component p-value (1 + b)/(k + 1) instead of b/k.**
```python
pvals[i] = (1 + np.size(np.where(null >= sz_links[i]))) / (k + 1)
```
- **Evidence:** Phipson & Smyth (2010) show that b/k is anti-conservative when permutations are drawn at random and can be exactly 0. SciPy uses the same correction.

**F03: no error when nothing passes the threshold.**
- bctpy raises `BCTParamError` when no edge exceeds the threshold. Now it returns no components and still computes the null distribution.
- **Evidence:** no suprathreshold edge is a valid outcome, and the human analysis reports critical extents from the null even when nothing is significant.

**S7 confirms the patched code.** Its null distribution, components and p-values were reproduced exactly from scipy's Welch t plus extent counting on the same relabelings.

**Runtime:** bctpy loops over edges.
- About 0.09 s per permutation per direction at 82 ROIs, so roughly 0.5 h per mouse comparison (about 4 h for all 8).
- About 0.2 s at 121 ROIs, so about 1.1 h for the human comparison.
- That is longer than the previous vectorised code, but it runs on one core.

## Why pairwise contrasts use the ordinary Welch t-test (S4; decision C2b)

| Equal means; SD of the n = 5 group × | pooled t, permutation | Welch t, permutation | Welch t, parametric |
|---|---|---|---|
| 9 vs 5, × 1 | 0.052 | 0.054 | 0.061 |
| 9 vs 5, × 3 | 0.115 | 0.101 | 0.052 |
| 19 vs 5, × 3 | 0.206 | 0.133 | 0.054 |
| 19 vs 5, × 0.33 | 0.004 | 0.020 | 0.049 |

*Each value is the rejection rate at α = 0.05 over 2000 null datasets of normal data.*

- **The problem.** When the 5-mouse group has the larger variance, permuting a Welch statistic is still inflated, at 10–13%. The parametric Welch t stays near 5%.
- **Skewed data.** With lognormal data and the n = 5 SD three times the other group's, all methods inflate, and the parametric Welch t inflates least:
  - 9 vs 5: pooled permutation 0.168, Welch permutation 0.145, parametric Welch 0.124;
  - 19 vs 5: 0.269, 0.188 and 0.133.
  - These are in `tests/simulate_type1_two_groups_output.txt`.
- **The 7-group omnibus is fine.** Permuting Welch's F stays near nominal (0.05–0.07), and so does the parametric Welch ANOVA. The classical F reaches 0.21–0.24 (`tests/simulate_type1_seven_groups_output.txt`).
- **Decision.** All pairwise contrasts use the ordinary Welch t-test (`scipy.stats.ttest_ind(equal_var=False)`). Using one method for every contrast keeps them comparable.
  - The omnibus tests keep Welch's F with permutations, which stayed near nominal.
  - NBS keeps its permutation null, which it needs for the component extent. Its two contrasts with CRND8_M (n = 5) carry the same caveat as above.

## Not changed (flagged)

- **Mouse group count check.** The mouse script expects NTG_A = 24 (94 mice in total). The header and cohort say 95, with NTG_A = 25. The check only prints a warning.
- **Duplicate human tests.** With two groups, the human edgewise "omnibus" Welch ANOVA tests the same hypothesis as the pairwise Welch t-test. The same applies to the graph-metric omnibus and pairwise tests. The statistics are related (F = t²), but the p-values now come from different methods (permutation vs Welch t). Report the pairwise Welch t-test, which is better calibrated.
- **Seeds.** Every permutation call uses `RANDOM_SEED`. This is harmless for marginal p-values.
- **Unused code.** `heatmap()` and some imports are never used. This is cosmetic, so it was left to keep the changes reviewable.

## Tests

- **`tests/test_static_published.py`:** S1–S9, 17 checks, 0 failed. The run is in `tests/test_static_published_output.txt`.
- **End-to-end runs.** I ran both scripts, previous and updated, on synthetic inputs with the real layout: 95 mice × 82 ROIs, and 54 people × 121 ROIs with BIDS labels 1–58 and gaps. Permutations were reduced (199 per test, 40 for NBS) to keep the runs short.
  - Both versions finished and wrote the same set of files.
  - Edgewise F and t were identical (≤ 1.4e-14).
  - NBS observed components were identical (same nodes and edge counts).
  - Graph metrics changed only through the one-edge threshold difference in the mouse script.
  - p-values differed by Monte Carlo error. This run predates C2b; a rerun with the ordinary Welch t-test also finished (see `tests/end_to_end_synthetic_results.txt`).
- **`tests/legacy/`:** verbatim copies of the previous code (`welch_permutation.py`, both scripts, extracted functions), used only by the tests.

## References

- Chung E, Romano JP (2013) Exact and asymptotically robust permutation tests. *Ann Stat* 41:484–507.
- Janssen A (1997) Studentized permutation tests for non-i.i.d. hypotheses and the generalized Behrens–Fisher problem. *Stat Probab Lett* 36:9–21.
- Phipson B, Smyth GK (2010) Permutation p-values should never be zero. *Stat Appl Genet Mol Biol* 9:39.
- Zalesky A, Fornito A, Bullmore ET (2010) Network-based statistic. *NeuroImage* 53:1197–1207.
- Rubinov M, Sporns O (2010) Complex network measures of brain connectivity. *NeuroImage* 52:1059–1069.
- Achard S, Bullmore E (2007) Efficiency and cost of economical brain functional networks. *PLoS Comput Biol* 3:e17.
