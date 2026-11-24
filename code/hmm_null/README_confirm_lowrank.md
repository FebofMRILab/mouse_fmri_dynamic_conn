# Confirmatory PCA-HMM test (spin-echo / low signal rank)

Pre-specified on 2026-09-30, before any of these fits were run.

## Why

The exploratory heatmap (`Fig_hmm_splithalf_null`) suggested that PCA-HMM at
rank 15/25 might beat the null at K = 2–4. Those cells can't serve as evidence,
because:
- each rested on only 5 null fits;
- the strongest cells were against IAAFT, which is biased low on band-passed
  data;
- the hypothesis was formed after seeing 126 cells.

The spin-echo argument does give an a priori reason to test low rank. At
11.1 T, spin echo weights small vessels and has lower BOLD sensitivity, so the
signal should sit in a small subspace buried in thermal noise. This is a new
test with fresh randomness and a decision rule fixed in advance.

## Pre-specified design

| | |
|---|---|
| Models | PCA-HMM (`options.lowrank`) rank 15 and rank 25 |
| K | 2, 3, 4 (6 cells in total) |
| Randomness | seed 20261001 (the exploratory run used 123); 30 split-halves; 30 phase-null fits per K |
| Replica arm | every mouse replaced by a stationary Gaussian replica with the same auto- and cross-spectra, run through the identical procedure: calibrates the HMM the way the DCC analysis was calibrated |
| Statistic | per split r: c_r = (obs_r − null_r) − (rep_r − repnull_r), demeaned state-FC split-half similarity (mean over matched states) |
| Test | one-sided Wilcoxon signed-rank c > 0, Holm across the 6 cells |
| Decision | state structure only where Holm p < 0.05 **and** the weakest matched state replicates better in observed than in null splits |
| IAAFT | descriptive only |
| Drift check | full fits at K = 2–4 on the observed data; each state's occupancy in the 2nd minus the 1st scan half per mouse; two-sided Wilcoxon, Holm across states within K (at K = 2 the two states' changes are each other's negative, so the test is counted once: E50, 27 Oct 2026) |

A cell that passes the test but whose state occupancy drifts between scan
halves is reported as drift (e.g. sedation), not as recurring states.

## Run on HiPerGator

Put these in the folder with `hmm_signals/`, together with your
`hmm_fit_raw_timeseries.m` and the existing HMM package files:
- `run4c_confirm_lowrank.m`
- `dfc_stationary_replica.m`
- `submit_confirm_lowrank.sh`
- `slurm/confirm_*.sbatch`
- `analyze_confirm_lowrank.py`
- `mp_signal_rank.py`

The defaults are the mouse cohort: 95 scans, TR 2 s. For the human data,
set `HMM_N_SUBJECTS`, `HMM_TR`, `HMM_DATA_DIR` and `HMM_OUT_DIR`.

```bash
mkdir -p logs
sbatch slurm/confirm_quicktest.sbatch        # ~minutes: checks paths, replicas, full fit
bash submit_confirm_lowrank.sh               # 64 chunk tasks (<=15 at once) + 4 finish jobs
```

**Cost.** Each model × arm has (30 + 30) × 3 = 180 split-half jobs, each
fitting both halves. That is 720 jobs in total, about 12–18 CPU-hours at your
earlier fit times, or about 1–1.5 h on 15 cores. The two observed-arm finish
jobs then run the full fits: 15 initialisations × 3 K per model, a few hours.
At most 16 cores are in use at any time.

**When everything has finished:**

```bash
python analyze_confirm_lowrank.py --root hmm_confirm_lowrank
```

This writes `hmm_confirm_lowrank/analysis/`:
- `confirm_cells.csv`. It holds median c (the statistic of `MODEL_SELECTION_RULE.md` item 6) and its bootstrap CI. Added 27 Oct 2026 (E51), all descriptive:
  - the Hodges–Lehmann estimate of c with its 95% CI, the interval that inverts the Wilcoxon signed-rank test: exact for n ≤ 50, as `scipy.stats.wilcoxon`, normal approximation above;
  - its achieved coverage;
  - the matched-pairs rank-biserial correlation, a scale-free effect size.
- `confirm_occupancy_drift.csv`. It adds the Hodges–Lehmann change with its 95% CI and a `note` column for the K = 2 one-test rows.
- `Fig_confirm_lowrank.{pdf,png}`. The forests show the Hodges–Lehmann estimate and 95% CI, labelled unadjusted; filled markers mean Holm p < 0.05. Holm p < 0.05 implies that the CI excludes 0. The earlier bootstrap CI of the median could cross 0 for a filled cell.

It also prints the decision.

**Optional, descriptive IAAFT:**

```bash
HMM_NULL=iaaft bash submit_confirm_lowrank.sh
python analyze_confirm_lowrank.py --root hmm_confirm_lowrank --null iaaft
```

## Confirming the step 5 screen candidate (27 Oct 2026; `MODEL_SELECTION_RULE.md` item 4, D32)

`run4c_confirm_candidate.m` is a copy of `run4c_confirm_lowrank.m` that differs only in the model and K (from `HMM_CAND_MODEL`, `HMM_CAND_K`), the seeds (20261026, 20261027), the output folder (`hmm_confirm_candidate`) and the PREREGISTRATION text; `diff` the two files to check. `hmm_candidate_options.m` sets the observation model exactly as step 5 does (`gaussian_pca_d` → `options.pca = d`; `lowrank_r` → `options.lowrank = r`; `gaussian_full` → neither). `submit_confirm_candidate.sh` reads the model and K cells from `screen_candidate.txt` (written by `select_screen_candidate.py`), refuses when the selector reports no lead or a candidate covered by step 6, and refuses to reuse an existing `hmm_confirm_candidate/` (E48). 32 chunk tasks (max 15 at once), then 2 finish jobs. Analyse with `python analyze_confirm_lowrank.py --root hmm_confirm_candidate` (Holm across this test's 3 cells).

Tests: `hmm_candidate_options.m` and `hmm_candidate_K.m` 11/11 in Octave 8.4; the submit script with a mock `sbatch` (candidate at K 2-4 and at K 5-7; covered by step 6, no lead, existing folder and IAAFT-before-phase refused; stray `HMM_*` variables removed).

## Signal rank without any HMM (`mp_signal_rank.py`)

For each subject, the script counts the eigenvalues of the correlation matrix
above noise in two ways:
- **Parallel analysis (primary).** The comparison is against circularly
  shifted surrogates, which keep each ROI's autocorrelation and lose all
  covariance.
- **Marchenko–Pastur edge.** This uses the effective sample size (Bartlett
  estimate: about 205 for 570 volumes band-passed at 0.01–0.1 Hz, TR 2 s).

It also reports the share of variance above noise, which can be compared
between datasets with different numbers of ROIs.

```bash
python mp_signal_rank.py --dataset mouse_SE11T hmm_signals 95 2 \
                         --dataset human_GE3T /path/to/human/hmm_signals 54 3
```

**Validation on simulated band-passed data:**
- Pure noise gave a rank of 0 by both methods.
- A planted rank of 12 at low SNR was recovered as 10–11; the weakest
  components fall below noise, as they should.


### Group level (`mp_signal_rank_group.py`, 15 Oct 2026)

The same two thresholds on the pooled data that the group PCA and HMM-PCA are estimated from (each mouse's standardised series, concatenated; T_eff = sum of the mice's Bartlett T_eff; circular shifts within each mouse). Functions are imported unchanged from `mp_signal_rank.py`. With ~95 × 208 effective samples both thresholds sit near 1, so the script also reports the split-half overlap of the top-k group subspace (two random halves of the mice, stratified by group): how much of the pooled structure is shared across mice rather than specific to some. Outputs: `signal_rank_group/` (CSV per dataset, the spectrum with thresholds and overlap by k, `FigS_signal_rank_group`). Checked on simulated mice with 10 shared directions plus 3 mouse-specific ones each: pooled MP and PA both gave 9, and the split-half overlap stayed above 0.9 up to k = 9 and fell after it.

## What counts for and against the spin-echo account

Three predictions should agree:
1. PCA-HMM rank 15 ≥ rank 25 in the calibrated test (this run).
2. Calibrated DCC is stronger at PCA-15 than at PCA-25, and weaker at PCA-50.
   This uses the pre-specified DFC sensitivity runs:
   `DFC_PCA_DIM=15 DFC_OUT_DIR=dfc_dcc_cpca_pca15 bash submit_dfc.sh`, and the
   same with 50.
3. Mouse signal rank of roughly 10–20 by parallel analysis.

If rank 15 fails the calibrated test, or DCC does not strengthen at PCA-15,
the account is not supported. The species comparison (awake GE 3 T against
sedated SE 11.1 T) remains confounded either way.
