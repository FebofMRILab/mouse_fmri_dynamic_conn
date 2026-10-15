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
| Drift check | full fits at K = 2–4 on the observed data; each state's occupancy in the 2nd minus the 1st scan half per mouse; two-sided Wilcoxon, Holm across states within K |

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

The defaults are the mouse cohort: 95 subjects, TR 2 s. For the human data,
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
- `confirm_cells.csv`
- `confirm_occupancy_drift.csv`
- `Fig_confirm_lowrank.{pdf,png}`

It also prints the decision.

**Optional, descriptive IAAFT:**

```bash
HMM_NULL=iaaft bash submit_confirm_lowrank.sh
python analyze_confirm_lowrank.py --root hmm_confirm_lowrank --null iaaft
```

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
