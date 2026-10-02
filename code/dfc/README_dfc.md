# Window-free dynamic FC: DCC and complex PCA against stationary surrogates

## What it does

For every subject, each measure is computed on the real data and on the
subject's own surrogates (19 per null by default):

- **phase**: multivariate phase randomisation. Keeps every auto- and
  cross-spectrum, so static FC and lag structure are preserved exactly, and
  removes non-stationarity.
- **iaaft**: multivariate IAAFT. Also keeps each region's value
  distribution (heavy tails, bursts).

| Analysis | Space | Measures |
|---|---|---|
| AR(2)–GARCH(1,1)–DCC(1,1) | 25 group-PCA components | `dcc_a`, `dcc_LR`, `corr_fluct` (correlation); `garch_alpha`, `garch_LR`, `vol_cv` (amplitude) |
| Complex PCA (Bolt et al. 2022) | ROI space (primary) and PCA-25 (check) | `env_cv` (episodic propagation), `var_expl`, `freq_hz`, `lag_index`, loadings (magnitude, phase, lag) |

## Band-passed data: why DCC/GARCH are decimated and tested against phase only

The mouse data are band-passed at 0.01–0.1 Hz but sampled at TR = 2 s, so there
is no power between 0.1 and 0.25 Hz. A series with an empty band cannot be
whitened by any AR model (residual lag-1 autocorrelation stayed at ~0.64 up to
AR(20)). DCC and GARCH then fit the smoothness as "dynamics": stationary
simulated data gave DCC a ≈ 0.44 and LR ≈ 10,500, the same size as the quick
test's surrogates.

- **Fix:** DCC/GARCH run on data decimated to just above twice the low-pass edge
  (TR 2 s → 4 s; Nyquist 0.125 Hz > 0.1 Hz, so nothing in the band is lost),
  averaged over the two polyphase components (odd and even volumes), so no data
  are discarded. Stationary data then give a ≈ 0.05. Power improved: FC
  switching was detected at z = 10–13, against 5.9 without decimation. Complex
  PCA keeps the full sampling rate.
- **The log reports** the power above 0.1 Hz and above the new Nyquist frequency.
  It warns if the data are not actually low-passed where `lowpass_hz` says,
  because decimation would then alias.
- **Which null is valid for which measure** (tested on stationary simulated data):

| Measure | Phase null | IAAFT null |
|---|---|---|
| DCC (`dcc_a`, `dcc_LR`, `corr_fluct`) | valid | **invalid**: its rank step leaks power into the empty band (false z = 1.6–2 after decimation, 15–160 before). Not computed. |
| GARCH (`garch_alpha`, `garch_LR`, `vol_cv`) | valid | valid |
| Complex PCA `env_cv` | valid | valid |

The same leakage explains why the HMM IAAFT null came out slightly *lower* than
the phase null at K = 3–4 in the mouse run. For band-passed data, the phase
null is the reference.

**Human data (TR 3 s):** decimation is impossible without aliasing if the
low-pass edge is 0.08–0.1 Hz (the factor comes out as 1). DCC there would need
data that were high-passed only.

## Calibrated test: stationary replicas (phase null)

Comparing each mouse with its own phase surrogates assumes that observed data
and surrogates are exchangeable when nothing is dynamic. That can fail in
small ways, and across 95 mice a small per-mouse bias becomes "significant".
On stationary simulated data, DCC leaned positive in every test run: mean rank
0.54–0.60 against the 0.5 expected, and mean z about +0.5. None of these
reached significance.

**What the calibration does.** For each mouse, `dfc_stationary_replica.m`
makes a stationary Gaussian replica with (nearly) the same auto- and
cross-spectra, lags included. It is cut from a longer series, so its ends don't
join up, as with trimmed real data. The replica goes through the identical
procedure (its own 19 phase surrogates), giving a replica z for every measure.
When nothing is dynamic, observed z and replica z share one distribution,
whatever bias the procedure has.

**The calibrated test** is a one-sided Wilcoxon on (z_observed − z_replica)
across mice, in `dfc_group_tests_calibrated.csv`. Each mouse's replica z is
the `phase_replica_z` column of `dfc_subject_metrics.csv`.

**Decision rule.** A claim that mouse FC is non-stationary requires the
calibrated test (p < 0.05). The uncalibrated one-sample test is descriptive
only. Genotype contrasts on z don't need it, because a bias common to all mice
cancels.

**Cost and reuse.** The replica work roughly doubles the phase-null compute.
It is stored separately (`subjects/sub###_replica.mat`), so it can be added
after a full run: resubmit, and the main checkpoints are reused. Set
`DFC_N_REPLICA=0` to switch it off.

**Replica checks (simulated data).** Static FC correlates 0.98 with the
observed data, the lag index matches (0.437 vs 0.438), and the ends are
non-periodic like the observed data. The taper adds a little power above
0.1 Hz (0.37% vs 0.16%).

## How to read the results: the `role` column

- **test**: non-stationarity measures. Observed > surrogate is evidence of
  dynamics.
- **static**: `var_expl`, `freq_hz` (preserved exactly by the phase null) and
  `lag_index`. They describe lagged, propagating connectivity, which is a
  stationary property. Use them for genotype and age comparisons, not as
  evidence of dynamics.
  - `lag_index` is a split-half estimate. The naive ratio is ~0.7 for pure noise
    and inflated when coupling is weak; the split-half version gives 0.00 for
    noise and 0.15 for weak zero-lag coupling. That residual bias remains, so
    group comparisons adjust for each mouse's **coupling magnitude**: the RMS
    |complex coherence| over all ROI pairs, split-half, written by
    `compute_lag_covariate.py` (its lag index matches MATLAB's to 1e-11).
    - Do **not** adjust for zero-lag FC strength. A pair's zero-lag
      correlation is about |coherence| × cos(phase lag), so it falls as lags
      grow, and adjusting for it would remove part of the lag difference being
      tested. A thresholded measure (e.g. top 15% of edges) also covers
      different pairs from the lag index, which uses all of them.
  - Use the ROI-space `lag_index` only. `cpca_pca_lag_index` is not
    interpretable: group-PCA components are uncorrelated at zero lag by
    construction, so its denominator is mostly noise (values far above 1 on
    simulated data). The plotting script ignores it.
- **diagnostic**: `dcc_b`, `dcc_persist`, `garch_persist`. Not identified when
  `dcc_a` or `garch_alpha` ≈ 0.

**DCC vs GARCH.** DCC measures time-varying *coupling* between components.
GARCH measures time-varying *amplitude* of each component, which also responds
to FC state switches because they change the component variances. For an
FC-specific claim, rely on DCC.

## Outputs (`dfc_dcc_cpca/`)

| File | Contents |
|---|---|
| `dfc_subject_metrics.csv` | Long format: subject, group, metric, role, observed, and for each null the surrogate mean, SD, z and p = (1 + #surr ≥ obs)/(n + 1) |
| `dfc_group_tests_calibrated.csv` | Phase null, test measures: median observed z, median replica z, one-sided Wilcoxon on their difference, median difference per group |
| `dfc_group_tests.csv` | Per metric and null: median observed − surrogate, one-sided Wilcoxon p (all subjects), fraction of subjects with p ≤ 0.05 (with 19 surrogates, 0.05 is the smallest attainable p: observed above every surrogate), median z per group |
| `cpca_components_{roi,pca}.csv` | Explained variance, median frequency, and a refit check on surrogates (phase: similarity = 1 by construction) |
| `cpca_loadings_*.csv` | Region, component, magnitude, phase (rad), lag (s, positive = later) |
| `cpca_cycle_{space}_c{k}.csv` | One propagation cycle: phase-binned mean signal per ROI (20 bins), for movies and figures |
| `pca_explained.csv`, `dfc_config.txt` | Group PCA variance, settings used |

Genotype comparisons are made by `plot_dfc_results.py` (and
`dfc_group_calibrated.py`) from `dfc_subject_metrics.csv` (z for the test
measures; observed values for the static ones), with the study's Welch tests
(`welch_tests_scipy.py`: Welch ANOVA with permutation p; ordinary Welch t for
the 8 planned contrasts). See `CHANGES_welch.md`.

## Run on HiPerGator

Put every file in the folder that contains `hmm_signals/`. Set `n_subjects`,
`TR` and `expected_T` at the top of `run_dcc_cpca_null.m` (mouse: 95, 2, 570;
human: 3 s and that dataset's values).

```bash
mkdir -p logs
sbatch slurm/dfc_quicktest.sbatch      # 6 subjects, 2 surrogates: checks the setup, times one subject
bash submit_dfc.sh                     # 32 chunks (max 15 running) + finish job
```

**Pre-specified sensitivity:** DCC at other dimensions, in its own folder.

```bash
DFC_PCA_DIM=15 DFC_OUT_DIR=dfc_dcc_cpca_pca15 bash submit_dfc.sh
DFC_PCA_DIM=50 DFC_OUT_DIR=dfc_dcc_cpca_pca50 bash submit_dfc.sh
```

**Runtime.** Each subject needs 1 + 2 × 19 = 39 full fits; DCC dominates. The
quick-test log gives minutes per subject; the whole run takes about
95 × that / 15. A resubmitted job reuses every finished subject.

## Validation (in Octave, on simulated data)

- **Analytic signal:** matches sin/cos to 1e-14.
- **DCC:** true (a, b) = (0.05, 0.90) recovered as (0.050, 0.909); constant
  correlation gives a = 0 and LR = 0.
- **GARCH:** (0.12, 0.80) recovered as (0.13, 0.78).
- **Complex PCA:** simulated travelling-wave lags recovered (r = 0.99–1.00,
  slope 0.95–1.07, correct sign). The phase-surrogate complex covariance is
  identical to 1e-15.
- **End-to-end** (two chunks + finish): subjects with FC switching and bursty
  waves gave z = 5–400 on every test measure; stationary subjects gave |z| ≤ 1.5.
- **End-to-end on band-passed data** (570 volumes, TR 2 s, 0.01–0.1 Hz):
  switching subjects gave z = 4–23 on every valid measure and null;
  stationary subjects gave |z| ≤ 0.7.
- **Calibrated test, end to end** (12 non-periodic band-passed mice, two chunks
  + finish):
  - Stationary mice: median (z_obs − z_replica) = −0.03 to 0.08 for DCC and
    GARCH, and −0.02 for env_cv of component 1.
  - FC-switching mice: 6–19 for DCC and GARCH.
  - Stationary mice also gave about +1 for env_cv of components 2–3. That is
    unresolved at n = 6, so treat components 2–3 with caution; component 1 is
    the pre-specified measure.
- **Statistics:** Wilcoxon and Mann–Whitney p-values match scipy exactly.

## Caveats

- `lag_sec` converts phase using the component's mean frequency, which is an
  approximation for broadband BOLD. The phase itself is the primary quantity.
- For a stationary Gaussian process the envelope is Rayleigh-distributed, so
  `env_cv` ≈ 0.52. Band-pass filtering and finite length shift this, which is
  why each subject has its own surrogates.
- `ar_order = 2` is fixed in advance. Autocorrelation left over after
  prewhitening is present in the surrogates too, so it cancels in the comparison.
