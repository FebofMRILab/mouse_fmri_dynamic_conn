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

## Supplement figures in natural units (14 Oct 2026)

The main figures show each measure as z against the mouse's own phase surrogates. Two Supplement figures show the same results in the measures' own units:

- `plot_dfc_raw_units.py` → `figures/FigS_dfc_raw_units` and `figures/table_raw_units.csv`: for each pre-specified measure, every mouse's observed value against the mean of its 19 phase surrogates (DCC a, SD of the conditional correlation, GARCH α, …), from `dfc_subject_metrics.csv`. Descriptive; the replicas' raw values are not stored, so the calibration is shown only as z.
- `dcc_example_traces.m` + `plot_dcc_example.py` → `dcc_example/FigS_dcc_example`: ρ(t), between −1 and 1 as in Lindquist et al. (2014), for one NTG_Y mouse drawn at random (`rng(20261014)`), computed with the published estimator (`dcc_mvgarch.m`) on the pipeline's input for that mouse and on its 19 phase surrogates (the pipeline's own seeds). Components 1–2, 1–3 and 2–3 were fixed in advance; panel D shows the SD of ρ(t) for all 300 pairs. Display only.
- Both run in `hipergator_tests/run_dcc_example.sbatch`. Test: `tests/test_dcc_example_traces.m` (X1–X7, Octave, with a stand-in for `dcc_mvgarch.m`): the mouse is NTG_Y and reproducible; `meta_data.csv` with a UTF-8 byte-order mark and CRLF line endings (as Excel writes it) is read with the pipeline's own reader; the group PCA and the surrogates equal the pipeline's own (`local_group_pca`, `local_surrogate`, extracted unchanged; differences 4.7e-16 and 0).

## Slice order and RABIES slice-timing correction (22 Oct 2026)

**What RABIES did.** RABIES 0.6.1 ran with `--apply_STC --TR 2` and no other slice-timing options. The settings saved in `preprocess_outputs/rabies_preprocess.pkl` are `stc_axis = Y` and `args_3dTshift = -Fourier -tpattern alt-z`.

**What was acquired.** The BOLD NIfTIs are 64 × 48 × 14. The 14 slices, 0.9 mm thick, lie along the third voxel axis (Z), which the header labels S in scanner convention. The ParaVision `method` files (scans 16–19) give:
- `PVM_ObjOrderScheme = Interlaced`
- `PVM_ObjOrderList = 0 2 4 6 8 10 12 1 3 5 7 9 11 13`
- slice orientation "axial", which for a mouse in the horizontal bore means coronal slices

**The mismatch.** The correction ran across the 48 in-plane rows instead of the slices. That probably left the real slice offsets uncorrected (up to 1.86 s, about 1 s between adjacent slices) and added false offsets of up to 2 s between rows. The NIfTI header also gives TR = 1.5 s; the TR is 2 s.

**`stc_axis_check.py`** is new lab code. It confirms this from the data and identifies the slice order for a rerun.
- **Method.** It takes each slice's lag relative to the whole-brain mean (cross-spectral phase, 0.01–0.1 Hz), then the even-minus-odd contrast and the slopes on every interleaved and sequential pattern.
- **Raw input.** Run on `bold_datasink/input_bold`, the contrast is about +1 s if even slices were acquired first (`alt+z` along Z) and about −1 s if odd slices were (`alt-z`).
- **Corrected output.** Run on RABIES's slice-timing-correction output, an uncorrected slice axis keeps the zig-zag, and a correction applied along the rows shows a slope of +1 on the pattern RABIES assumed.
- **Validation.** `tests/stc_axis/run_test.sh` gives +1.00 / −1.00 s for the two directions, 0.96 s with an added 0.5-s real gradient, and a row slope of +1.00 for the RABIES-like correction along Y.

```bash
python stc_axis_check.py --inputs "preprocess_outputs/bold_datasink/input_bold/*.nii.gz" \
  --TR 2 --slice_axis 2 --row_axis 1 --order 0,2,4,6,8,10,12,1,3,5,7,9,11,13 --out_dir stc_check_raw
```

## Slice timing vs lag (diagnostic, 21 Oct 2026)

`slice_timing_lag_check.py` is new lab code, not taken from a published package. It checks whether the complex-PCA lags could come from slice acquisition timing that RABIES's slice-timing correction (STC) left in place, for example because of a wrong `--tpattern`, direction or axis. It changes no analysis step.

**Why lags can come from slice timing.** Each slice is acquired at its own time within the TR, but all slices of a volume get the same time stamp. A slice acquired later therefore records the signal ahead of its label, and its regions look earlier: "the observed time courses of a hemodynamic response reach their maximum amplitude earlier in slices acquired later" (Sladky et al., 2011). Without STC, lag = constant − acquisition time, a slope of −1. After a correct STC the slope is 0.

**Why the test compares mice within each ROI.** Averaged over mice, an ROI's acquisition time is almost a straight line in its position along the slice axis, because the odd/even alternation of interleaved orders averages out. A real front-to-back lag would then look like slice timing: in simulation it gave a group-level slope of 0.91. But each mouse sits differently in the slice package. The same ROI is acquired at different times in different mice, while its anatomy, and any real lag, stays the same.

**Steps**
1. **Acquisition time per mouse and ROI.** From each mouse's RABIES native-space labels (`bold_datasink/native_labels`), count the ROI's voxels in every slice. The ROI's acquisition time is the voxel-weighted mean slice time.
2. **Lag per mouse and ROI.** Regress the mouse's analytic signal on the time courses of all group components jointly (dual regression; Beckmann et al., 2009; Nickerson et al., 2017). The group components are not refitted. The phase is folded onto ±π/2, so an opposite-sign member isn't read as a half-cycle lag, and converted to seconds as in `cpca_loadings_roi.csv`.
3. **Regression.** lag = ROI effect + mouse effect + b × acquisition time, weighted by |loading|². b uses only how acquisition time differs between mice within an ROI.
4. **Inference.**
   - CI: bootstrap over mice.
   - p: the acquisition-time maps are shuffled between mice, within group with `--strata group`, giving (1 + b)/(k + 1).
   - Shuffling slice times across slices instead was anticonservative (p < 0.05 in 23–46% of null runs).

Candidate orders are the six AFNI `3dTshift` patterns (`seq±z`, `alt±z`, `alt±z2`), which RABIES `--tpattern` uses. With `--method`, the scanner's own order (`PVM_ObjOrderList`) is added in both index directions.

**Reading the result**

| Situation | Slope for the true order | Other orders |
|---|---|---|
| STC correct | about 0 | about 0 |
| No STC | about −1 (−0.8 to −0.9 in simulation) | non-zero |
| STC in the wrong direction | below −1 (−1.1 to −1.4 in simulation) | non-zero |

`within_roi_sd_t` is how much acquisition time varies between mice within an ROI. If it is below 0.05 s, the test has little power.

**Slice fractions without native labels (RABIES 0.6.1, 24 Oct 2026).** RABIES 0.6.1 writes no native-space labels, so `make_slice_maps.py` (new lab code) builds the same information in commonspace:
1. For each mouse, a slice-number map on the native EPI grid (`bold_datasink/input_bold`).
2. The map is pushed into commonspace with the transforms RABIES used for `commonspace_bold`, with ANTs `antsApplyTransforms` and label interpolation. The order is unbiased-to-atlas warp and affine, then anat-to-unbiased warp and affine, then bold-to-anat warp and affine.
3. For every ROI of the time-series atlas (run3's `--ROI_labels_file`), it counts how many commonspace voxels came from each slice. The ROI signals were averaged over the same commonspace voxels.

Regions are numbered in RABIES's order (sorted atlas values), and the script checks the count against the signal-file columns. It writes `slice_fractions.csv`, `slice_timing_pairs.csv` and `qc.csv`, and has two built-in checks:
- **Registration chain:** the mean of the first 50 native volumes, pushed through the same chain, is correlated with RABIES's own `commonspace_bold` (expected r > 0.9).
- **Atlas, region order and mouse ↔ signal-file mapping** (`--cleaned_glob`): ROI means of RABIES's confound-corrected time series are correlated with the columns of `signal<N>.txt` (expected r near 1 for every ROI).

```bash
module load ants
python make_slice_maps.py --meta meta_data_with_bids.csv --rabies_preproc ../preprocess_outputs \
  --atlas /blue/febo/share/civm_atlas/BrainNetCoord.nii --signals_dir hmm_signals \
  --out_dir slice_maps --qc_n 0 \
  --cleaned_glob "../confound_correction_outputs/confound_correction_datasink/cleaned_timeseries/**/*sub-{id}_*.nii*"
python slice_timing_lag_check.py --loadings dfc_dcc_cpca/cpca_loadings_roi.csv \
  --pairs slice_maps/slice_timing_pairs.csv --fractions slice_maps/slice_fractions.csv \
  --strata group --TR 2 --rabies_tpattern alt+z \
  --order 0,2,4,6,8,10,12,1,3,5,7,9,11,13 --out_dir slice_timing_check
```

**Parcellation integrity (24 Oct 2026).** `BrainNetCoord.nii` is built in `civm/roi_extraction.bash` by summing the 82 hole-filled masks `ROIs_Lo/roi<i>.nii.gz` with weight i and keeping values ≤ 82. A voxel in two masks therefore gets the sum of their indices: another region's label if the sum is ≤ 82, or no label if it is larger. `check_parcellation.py` (new, reads files only) counts, region by region, the voxels relabelled, dropped and received from other regions this way. It also writes `roi_labels.csv` from the curated list (regions.txt order = region 1–82). On a synthetic atlas with one overlapping mask it found all overlap voxels and every affected region. `check_roi_signals.py --masks seeds_list.txt` compares the signal files with means over the masks instead of the atlas, which shows which of the two definitions the signal files were made from.

```bash
python check_parcellation.py --atlas /blue/febo/share/civm_atlas/BrainNetCoord.nii --seeds seeds_list.txt \
  --regions /blue/febo/share/civm/regions.txt --ontology RCCF_Ontology_2022-08-19.xls --out_dir parcellation_check
```

**ROI time series from RABIES's cleaned data (24 Oct 2026).** On sub-001, the correlation matrix of the ROI means recomputed from RABIES's cleaned time series with `atlas_commonspace.nii.gz` equals RABIES's own FC matrix (r = 1.0000, max difference 0.0000), so atlas, region order and grid are identical to RABIES's. The previous `hmm_signals/signal1.txt` does not match it (r = 0.81, max difference 0.58): it was made by another route. Cause: the lab script `organize_signals.m` extracted each ROI with `fslmeants --eig`, the first eigenvariate (FSL `fslmeants.cc`: the first eigenvector of the voxel-demeaned time × time covariance, scaled by √T, sign set to agree with the mean), not the mean that RABIES uses (`parcellated_FC_matrix` in RABIES 0.6.1 `analysis_functions.py`: "take the mean ROI timeseries"). In 10 mice, the eigenvariate correlated with the ROI mean at median r 0.94, falling with ROI size (Spearman ρ = −0.64): CA1 and DG 0.59–0.71, small nuclei 0.98–0.99, and below 0.5 for 23 of 820 mouse × ROI pairs (as low as 0.02), so in some mice the eigenvariate follows a different signal from the region as a whole. Fix (adopted 24 Oct 2026, D29: the ROI mean for every analysis): `organize_signals.m` with `--eig` removed (`code/organize_signals/`, edits marked `EDIT`), writing to `hmm_signals_mean/` and checking every mouse against RABIES's FC matrix. Test of the check (Octave, synthetic RABIES-format FC file): ROI mean 1.2e-15; FSL-style eigenvariate 0.24 (flagged). `extract_roi_signals.py`, written before the lab script was seen, is withdrawn: the lab script with this one-option change does the same.

If the ROI signal check is not near 1, `check_roi_signals.py` (new, reads files only) shows why, ROI by ROI. It correlates each signal column with every atlas region and reports which region it matches best. It separates three causes, each tested on the synthetic tree:
- **One mouse low everywhere:** a signal file belongs to another mouse.
- **The same ROIs low in every mouse, matching a fixed other region:** the region order differs or the atlas is mirrored. It reports the share of best matches that are the opposite-hemisphere homologue (100% for a mirrored atlas).
- **Small ROIs slightly low but matching themselves:** a difference in how small regions were resampled.

```bash
python check_roi_signals.py --slice_maps slice_maps --roi_labels roi_labels.csv --n_mice 10 \
  --cleaned_glob "../confound_correction_outputs/confound_correction_datasink/cleaned_timeseries/**/*sub-{id}_*.nii*"
```

**Validation** (`tests/slice_maps/run_test.sh`, about 6 minutes with antspyx). It builds a synthetic RABIES output tree:
- 12 mice, with 14 native slices along voxel axis 2 acquired `alt+z` at TR 2 s.
- Per mouse, a rigid bold-to-anat transform (slice placement shifted by up to ±1.5 slices) and an affine plus smooth warp for anat-to-unbiased.
- A group unbiased-to-atlas affine plus warp, and an 82-ROI atlas.
- `commonspace_bold` made with the same chain; ROI series are atlas means.

Results:
- 82 regions found, matching 82 columns.
- Registration r = 1.000; a deliberately wrong bold-to-anat transform gave r = 0.18 and a warning.
- ROI signal r = 1.000.
- Lag check with no slice-timing correction: −0.64 [−0.76, −0.51] for C1 and −1.07 [−1.42, −0.80] for C2, p = 0.0005.
- With correct correction: −0.01 [−0.13, 0.12] and 0.13 [−0.17, 0.36].
- Linear resampling in commonspace mixes adjacent slices, which attenuates the no-correction slope below −1.

**Inputs**
- `cpca_loadings_roi.csv` from the DFC run.
- `slice_timing_pairs.csv`: one row per mouse, with columns `signal_file` (the `hmm_signals/signal<N>.txt` used by `run_dcc_cpca_null.m`), `label_file` (that mouse's RABIES `native_labels` NIfTI) and `group`.
- `roi_labels.csv` (as for `plot_dfc_results.py`). Add an `atlas_value` column if the label integers are not 1–82 in the column order of the signal files.
- One scan's ParaVision `method` file, if available.

The native labels must be on the acquisition grid: the slice count printed must equal the raw EPI's.

Building the pairs file. Add a `bids_id` column to a copy of `meta_data.csv` (the BIDS subject label, with or without `sub-`), then run `make_slice_timing_pairs.py`:

```bash
python make_slice_timing_pairs.py --meta meta_data_with_bids.csv \
  --labels_dir /blue/febo/share/Abeta_Age/rabies/<preprocess_outputs>/bold_datasink/native_labels
```

It lists every problem at once (empty `bids_id`, no label file or more than one for an ID, missing signal file, one label file paired with two mice) and writes nothing until all rows match. `sub-1` never matches `sub-10`. It also reads Excel's CSV quirks: byte-order mark, CRLF line endings, blank trailing rows and IDs read as `7.0`.

**Run** (in `signals/`, after `bash hipergator_tests/link_into_signals.sh`). The environment needs `nibabel`; in an existing environment add it once with `conda run -p /blue/febo/$USER/envs/fc_static pip install nibabel`.

```bash
sbatch -A febo --qos febo -c 1 --mem 8gb -t 02:00:00 -o stc_%j.log --wrap \
  "module load conda; conda run --no-capture-output -p /blue/febo/$USER/envs/fc_static \
   python slice_timing_lag_check.py --loadings dfc_dcc_cpca/cpca_loadings_roi.csv \
   --pairs slice_timing_pairs.csv --strata group --roi_labels roi_labels.csv --TR 2 \
   --rabies_tpattern <as given to RABIES> --method <one scan's method file> \
   --out_dir slice_timing_check"
```

Outputs (`slice_timing_check/`):
- `slice_timing_lag_slopes.csv` (one row per component × order)
- `slice_timing_mouse_roi.csv` (acquisition times and lags per mouse and ROI)
- `slice_timing_label_files.csv` (slice axis, orientation and voxel size per file)
- `FigS_slice_timing_lag`:
  - A: component-1 lag against acquisition time, both relative to the ROI mean, with the −1 line for no STC.
  - B: slope with 95% CI for every order and component.

**Validation** (`tests/slice_timing/run_test.sh`, Octave + Python, about 80 s). The simulation has 20 mice and 20 coronal slices acquired `alt+z`, at TR 2 s with 570 volumes. It contains a global component, a seesaw component and ROI noise, all band-passed 0.01–0.1 Hz, and the group complex PCA is fitted by `dfc_cpca_fit.m`.

| Scenario | C1 slope | C2 slope |
|---|---|---|
| No STC | −0.88 [−1.01, −0.77] | −0.83 |
| STC in the wrong direction | −1.40 | −1.06 |
| Correct STC | 0.05 [−0.07, 0.16] | 0.01 |
| Real 1-s front-to-back lag, correct STC | 0.05 | 0.01 |

- The real-lag scenario gave a group-level slope of 0.91.
- Over 41 null runs, p < 0.05 occurred in 0–7%, and the mean slope was 0.00 (SD 0.03–0.04).
- Three faults were found and fixed during development (provenance E45): the group-level design, the slice-permutation null, and one-component-at-a-time projection, which let component 1 leak into component 2.

## ROI signal homogeneity (Supplement, 24 Oct 2026)

`roi_signal_homogeneity.py` (new; descriptive, plus the study's Welch tests) asks how well one time series represents each ROI, now that every analysis uses the ROI mean (D29). For every mouse and ROI, from RABIES's cleaned commonspace data and `slice_maps/atlas_commonspace.nii.gz`:

| Column | Meaning |
|---|---|
| `r2_mean` | Share of the ROI's voxel variance explained by the ROI mean: the mean over voxels of r(voxel, ROI mean)². Voxels are standardised, so each counts once. |
| `r2_pc1` | The most any single time series can explain: largest eigenvalue of the voxel correlation matrix / number of voxels (first principal component; Eckart & Young 1936). Always ≥ `r2_mean`; the gap is what the mean costs against the best single series. |
| `r2_pc2` | The same for the second component. |
| `mean_r` | Mean correlation over all voxel pairs: resting-state homogeneity as in Schaefer et al. 2018. |
| `r_mean_eig` | r between the ROI mean and the first eigenvariate computed as `fslmeants --eig` does (E47), for all mice. |
| `n_vox`, `n_vox_used` | Atlas voxels; voxels with non-zero variance (constant voxels outside RABIES's brain mask are left out of the voxel measures). |

Per mouse, the measures are averaged over ROIs weighted by voxel count (the share of all ROI voxel variance explained by the ROI means). Group tests: Welch's F with a permutation null for the per-mouse value and for each ROI (BH over the 82 ROIs), and the study's 8 planned contrasts (Welch t, BH over the 8). Outputs in `roi_homogeneity/`: `roi_homogeneity_mouse_roi.csv`, `_by_roi.csv`, `_by_mouse.csv`, `_group_tests.csv`, `roi_homogeneity_report.txt`, `FigS_roi_homogeneity.png/.pdf` (A: per ROI, ROI mean vs first-component ceiling; B: against ROI size; C: per mouse by group; D: mean vs first eigenvariate).

Built-in checks: `--eig_dir hmm_signals_eig` compares the script's eigenvariate with FSL's own output, and `--signals_dir hmm_signals` compares its ROI means with the step 0b files; both should give r = 1.000 for every mouse and ROI, and the report lists any mouse below 0.999.

How to read it:
- High `r2_mean` with `r2_pc1` close to it: one time series represents the ROI well.
- A large `r2_pc2` (well above that of small, one-signal ROIs; noise and smoothing set its floor): the ROI holds a second signal. These are candidates for splitting (ROI-selection subproject).
- The gap between `r2_mean` and `r2_pc1` shows how much the mean loses by mixing the parts. It is near zero when two parts are equal in size, so a small gap alone does not show a single signal (`r2_pc2` does).
- Low `r2_mean` and low `r2_pc1` alike: noise dominates, or the signal is spread over many small parts; these measures cannot tell the two apart.
- Small ROIs score higher partly by geometry: neighbouring voxels share signal through the point-spread function and RABIES's 0.5-mm smoothing. Panel B shows the size trend; the measures are not size-adjusted.
- A group difference in these measures would mean the ROI signals differ in quality between groups, which matters when interpreting group FC differences.

Test (`tests/roi_homogeneity/run_test.sh`; synthetic RABIES tree, 28 mice in the 7 groups, white noise with planted signals): measures equal an independent `np.corrcoef`/`eigvalsh` computation (6.7e-16); one-signal ROI `r2_mean` 0.504 (expected 0.50); 75/25 two-signal ROI `r2_mean` 0.349 (0.35), `r2_pc1` 0.371 (0.38), `r2_pc2` 0.127 (0.13); constant voxels excluded (32 → 24); planted group effect found (q = 0.002), none in the other ROIs; eigenvariates match FSL-spec files (r = 1.0000000); a mouse with swapped columns and a mouse with another mouse's eigenvariates are flagged.

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
