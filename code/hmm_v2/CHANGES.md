# HMM code: changes to follow published code

**Rule.** Use published code and adapt our data to it. A deviation is allowed only to fix a documented flaw, and then it comes with the exact code and the evidence.

**Published code these files call:**
- **HMM-MAR** (Vidaurre et al.; GPL-3; github.com/OHBA-analysis/HMM-MAR). Tested here at commit `023905b18be11981c7549bbc6ff6e88bd3aaff1e` (2026-04-05). Record the commit used on HiPerGator, which `tests/test_hmm_v2_real_data.m` prints.
- **Brain Connectivity Toolbox** (BCT; Rubinov & Sporns 2010), release 2019-03-03.
- **MATLAB Statistics Toolbox** (`corr` with `'Type','Spearman'`, `prctile`, `iqr`) in run5B.
- **SciPy and statsmodels** (`scipy.stats.wilcoxon`, `multipletests(method='holm')`) in the drift test, as in the confirmatory analysis.

## Reported models on the ROI-mean signals (27 Oct 2026; D33)

- **Decision (PI, 27 Oct 2026).** Two models are reported. The primary (main text) is a Gaussian HMM on 25 principal components (`options.pca = 25`), K = 3, which passed step 6b. The secondary is HMM-PCA (`options.lowrank = 15`), K = 2, which passed step 6.
  - `hmm_null/MODEL_SELECTION_RULE.md` item 6 would have put the secondary in the main text (larger median c). The PI chose by precision instead, after the confirmation results and before any group result. That rule's amendment 1 records the deviation; `FIT_SELECTION_RULE.md` amendment 1 applies the fit rule to each model.
  - The section below ("HMM-PCA, rank 15, K = 4") is the choice on the eigenvariate signals and no longer decides anything.
- **One place for each model.** `MODEL_PRIMARY.env` and `MODEL_SECONDARY.env` are read by:
  - `hmm_config.m`: the primary model is the default; `HMM_K` / `HMM_LOWRANK` / `HMM_PCA` override it, and the file is required;
  - `hmm_py/hmm_model_selection.py`: same defaults; both models need `inference_scope.csv`;
  - `hmm_occupancy_drift.py`;
  - the job scripts, through `hipergator_tests/reported_model.sh primary|secondary`.

  Before this change, five job scripts and the Python defaults hard-coded K = 4 and rank 15.
- **`hmm_config.m`.** `confirm_gamma` now points to the model's own confirmation run:
  - step 6 (`hmm_confirm_lowrank/…/gamma_lowrank_<r>_K<K>.mat`) for HMM-PCA;
  - step 6b (`hmm_confirm_candidate/…/gamma_gaussian_pca_<d>_K<K>.mat`) for a Gaussian model.
- **`select_reported_fit.m`.**
  - **Default fits folder:** `hmm_k<K>_pca<d>_fits` or `hmm_k<K>_hmmpca<r>_fits`.
  - **Model check:** it stops unless the fits folder's `model.env` (copied by the array job before fitting) is the model in `hmm_config.m`. Each fit's `options.pca` is checked as well as `lowrank`.
  - **Previous refit:** the comparison with the 4 Oct 2026 refit is reported as not applicable (`PREVIOUS_REFIT = false`; amendment 1, item 6). Every earlier fit is of the eigenvariate signals.
  - **Messages:** no longer say "K = 4".
  - **Tests:** F11 updated; F14 (another model's `model.env` refused) and F15 (a Gaussian PCA model: default folder, selection, a fit with another `options.pca` refused) added. 15/15 in Octave 8.4 with HMM-MAR 023905b.
- **Fits array.** As well as the per-folder copy, the first fit of each role records the model file in `<signals>/hmm_fits_<role>_model.env`. Any later task stops if the file has changed, even when a new K or dimension would point to a new fits folder (tasks still queued after an edit).
  - Each array runs at most 8 tasks at once (`--array=1-10%8`), so the two arrays together stay within 16 cores.
  - Test: `hipergator_tests/tests/test_model_files.sh`, 8/8 (stubs for `matlab`, `sbatch`, `module`).
- **E50: K = 2 occupancy counted once** (found before running, 27 Oct 2026).
  - **Flaw.** Each mouse's occupancies sum to 1 over the states. With two states, FO(state 2) = 1 − FO(state 1) for every mouse, and each mouse's drift change in state 2 is minus that in state 1. So the per-state tests are one test counted twice: Welch's ANOVA, the Welch t contrasts with the sign reversed, the permutation p with the same permutations, and the two-sided Wilcoxon of the drift. The code, written for K = 4, would have put both copies in the Holm (and BH) family and doubled the adjusted p.
  - **Fix in `hmm_occupancy_drift.py`** (the same lines in `hmm_null/analyze_confirm_lowrank.py`):
    ```python
    if int(K) == 2 and len(rows) == 2 and complements(ok):
        # one test: state 2's change is minus state 1's (E50)
        rows["p_two_sided"] = rows.p_two_sided.iloc[0]
        rows["p_holm_within_K"] = rows.p_two_sided.iloc[0]
    ```
    Here `complements(ok)` checks on the data that the two occupancies sum to 1 within 1e-6. Otherwise both states are tested with Holm and the row is flagged.
  - **Fix for occupancy** (`hmm_py/hmm_model_selection.fo_test_plan`, used by `hmm_dynamic_stats.py` and the occupancy figure): the second state takes the first state's test (`same_test_as`), with the same check on the data.
  - **Evidence.**
    - `hmm_null/tests/hodges_lehmann/test_hodges_lehmann.py` H8 and H9: the K = 2 drift p is counted once in both scripts; the previous code gave 2 × p; K = 3 is unchanged (max diff 2.8e-17); non-complements fall back to Holm.
    - `hmm_py/tests/test_hmm_python_stats.py` S15: state 2's Welch test computed on its own equals state 1's (same F within 1e-9, same permutation p). The tables carry it once (Holm p = p; contrasts t(2) = −t(1); state 1's q; BH over state 1's contrasts only), and both figure panels show the one p.
  - **Effect on results already seen.** For step 6's rank 15, K = 2, drift Holm p 0.30 becomes p 0.15 (no drift either way). No pass or fail in step 6 or 6b changes.
- **New (descriptive): `hmm_state_descriptors.m`** (values in `hmm_state_descriptor_values.m`), run after run5 in the pipeline. It asks whether the states differ mainly in amplitude or in correlation pattern. For each mouse and state, and pooled over mice, it gives:
  - occupancy, mean ROI variance and global-signal variance, all from HMM-MAR's own state statistic (setxx.m) on the standardised data;
  - mean correlation;
  - for each pair of states, the FC pattern r, with r² as the share an affine shift of the correlations explains, and its noise-corrected value. The correction is Spearman's (1904) correction for attenuation, using odd- vs even-numbered mice.

  It uses no group labels and tests nothing. It was proposed because a two-state fit with about half the time in each state could separate high- from low-amplitude periods.
  - It also writes `state_fc_pooled_upper.csv` (the pooled state FC, equal to `R.state_fc`, upper triangle). `hmm_py/hmm_model_agreement.py` reads it to pair the two models' states (`hmm_null/MODEL_SELECTION_RULE.md` amendment 2, D35).
  - **Tests:** `tests/test_state_descriptors.m` D1–D7, 7/7 in Octave. It uses simulated amplitude-only, pattern-only and global-signal differences, a weak-FC case for the correction, the pooled FC against `R.state_fc`, and a mouse checked by hand. The workflow test W10 runs the script on a real run4e fit (MATLAB).

## Reported model on the eigenvariate signals: HMM-PCA, rank 15, K = 4 (your decision, 2 Oct 2026; superseded 27 Oct 2026, D33)

- **The model.** `options.lowrank = 15` (HMM-PCA), K = 4.
- **How K was chosen.** After the pre-specified confirmatory test (`run4c_confirm_lowrank.m`: rank 15/25 × K = 2–4, 30 split-halves, calibrated against stationary replicas, Holm across the six cells). Rank 15 is the rank the spin-echo hypothesis favoured (prediction 1 in `README_confirm_lowrank.md`); the time-stamped plan (`PREREGISTRATION.txt`, written by run4c) names both ranks and all K with no primary cell, so the choice of rank 15 was also made after the results. At rank 15, K = 4 is the only K that met the decision rule (`hmm_confirm_lowrank/analysis/confirm_cells.csv`):

  | Rank 15 | Calibrated median | 95% CI | Holm p | Weakest state above its null | Meets the rule |
  |---|---|---|---|---|---|
  | K = 2 | 0.025 | −0.036 to 0.092 | 0.12 | yes | no |
  | K = 3 | 0.007 | −0.013 to 0.096 | 0.12 | yes | no |
  | K = 4 | 0.051 | 0.037 to 0.073 | 0.0009 | yes | **yes** |

  Rank 25, K = 2 also met the rule (0.075, Holm p = 0.025). It goes to the Supplement because the paper reports rank 15. The Holm correction across the six cells covers the choice among them; the paper shows all six.
- **K = 3 before.** K = 3 was the reported model until this decision. It did not meet the rule: its stationary replicas show split-half excess of their own (replica p = 0.03), which the calibration removes. At K = 4 the replicas show none (p = 0.66).
- **How strong the evidence is.** The calibrated excess at K = 4 is small: 0.051, against 0.19 (event simulation) and 0.22 (switching simulation) at the same cell in the positive control. The weakest state replicates at 0.018 (null 0.005), and 2 of the 6 cells pass. This supports weak but replicable state structure.
- **Drift.** In the confirmatory full fit at K = 4, no state's occupancy changes between scan halves (Wilcoxon, Holm within K: p ≥ 0.54; 38–55% of mice increase; `confirm_occupancy_drift.csv`). At K = 3, state 1 did (62% of mice decrease, Holm p = 0.008). `hmm_occupancy_drift.py` repeats the test on the reported fit; the 4 Oct 2026 refit differs (see Refits).
- **Order.** Decided after the confirmatory results were known (K = 3 had been planned), and before any K = 4 group comparison was run. The paper must say the choice of cell was post hoc.
- **What it replaces.** The fit used before, `hmmmar_simple_k3_pca15`, was PCA to 15 components followed by full-covariance states (`pca = 15`, `lowrank = 0`; read from the fit on HiPerGator). That is a different model from the one the confirmatory test was run on. A K = 3 HMM-PCA folder (`hmmmar_simple_k3_hmmpca15`), if one was made, is not the reported model either.
- **Defaults.** `hmm_config.m` defaults to K = 4, `lowrank = 15`, `pca = 0`; the output folder is `hmmmar_simple_k4_hmmpca15`. The Python side (`hmm_py/hmm_model_selection.py`, `hmm_occupancy_drift.py`) has the same defaults. run4d's table grid now starts at K = 2 (`HMM_K_VALUES`, was 3–10), so a Supplement fit at rank 25, K = 2 is tabled; folders that do not exist are skipped, as before.
- **Data folder.** `HMM_OUT_BASE` defaults to `/blue/febo/share/Abeta_Age/rabies/signals` (was `../signals`, relative to the folder MATLAB was started in), and the signal files to its `hmm_signals/` (`HMM_DATA_DIR`). The steps now find the data wherever they are started from.
- **Fit settings.** run4e fits with the options of the confirmatory full fits (E7).
- **Refits.** No refit reproduces the confirmatory fit exactly (E8). `tests/check_against_confirmatory.m` measures how close the two are; with a weak fourth state, check it before using the refit.
  **Result, 4 Oct 2026 (HiPerGator, MATLAB R2026a).** The refit (fit `1783b9c3…`) is a different solution from the confirmatory run's full fit (`ae150f19…`, the fit its drift check used; the confirmatory test itself compared split-half fits): `getGammaSimilarity` 0.43 (1 = identical up to state order). One state is reproduced (state time course r = 0.63, pooled state FC r = 0.97); three are not (time course r = 0.05–0.15, FC r = 0.75–0.77). One refit state drifts between scan halves (median +0.037, 67% of mice increase, Holm p = 0.013). The 4 Oct refit is therefore not used for the paper: the reported fit is chosen by the rule below ("Fit selection"), fixed before any further fit was run (provenance workbook, D25).

## Fit selection (new, 4 Oct 2026; your decision D25)

The rule is in `FIT_SELECTION_RULE.md`, written before the fits it governs. In short: ten independent fits of the same model; the reported fit is the one with the lowest final free energy; group tests only for its states whose pooled state FC recurs (r ≥ 0.9 with the matched state in more than half of the other fits), and for switching rate and max FO only if their per-mouse values agree across fits by the same criterion.

| # | Before | Now | Why |
|---|---|---|---|
| S1 | One fit per run of run4e; the reported fit was whichever run came last | `hipergator_tests/run_hmm_fits_array.sbatch`: ten run4e fits, unchanged, each in its own folder (`<signals>/hmm_k4_hmmpca15_fits/fit_NN/`). The job records the rule's SHA-256 before fitting and never overwrites a finished fit. | HMM-MAR reseeds from the clock, so fits differ; one fit cannot show which states are stable. |
| S2 | No rule for choosing among fits | `select_reported_fit.m`: lowest final free energy (`fehist(end)`), the criterion HMM-MAR uses among its own initialisations (`train/hmmmar_init.m`). Matching with HMM-MAR's `getGammaSimilarity`. Per-mouse agreement is the statistic of HMM-MAR's `hmmtestretest.m`, computed on the saved fits. Writes `selection/` and `inference_scope.csv` in the chosen fit's folder. Only `fit_01`–`fit_10` are candidates. If no state recurs, nothing is tested (item 9). It refuses a changed rule file, fits made with other options (including `initrep`), fewer than 8 valid fits, and a second selection that differs from the first. The array and selection jobs ignore `HMM_*` settings other than the toolbox path, so a leaked `HMM_MODEL_DIR` or `HMM_INITREP` cannot change the fits. | The choice uses no group labels or results, so it cannot be fit-shopping. `hmmtestretest` itself refits inside one call and keeps no run4e outputs, so its statistic is applied to the run4e fits instead. |
| S3 | Every state tested | `hmm_py` reads `inference_scope.csv` (`hmm_model_selection.inference_scope`): tests, and the Holm and BH families, cover only recurring states and the whole-fit outcomes that pass (including the group model of bootstrap Table S-E). Descriptive tables keep every state. The drift test still covers every state. Without the file, the reported model is refused. Each script first deletes its own earlier test outputs. | Group differences in a state that changes from fit to fit would describe the fit, not the mice. |
| S4 | `run_hmm_pipeline.sbatch` always fitted | `SKIP_FIT=1` runs every later step on an existing fit; `run_selected_fit.sh` submits it for the chosen fit, with outputs inside `fit_NN/`. | One fit for every later step; the 4 Oct refit's outputs stay untouched. |

Tests: `tests/test_select_reported_fit.m` (F1–F13, simulated fits with a known answer; MATLAB or Octave; 13 of 13 passed here in Octave 8.4) and `tests/test_select_on_run4e.m` (R1–R4, eight real run4e fits of the simulated cohort; MATLAB, in the HiPerGator test job). `hmm_py`: S11–S14. An independent review (assistant sub-agent, before any fit) found nine issues in the first version; all were fixed and are covered by F12, F13, S13 and S14.

## All steps: one configuration and one fit ID

| # | Before | Now | Why |
|---|---|---|---|
| A1 | Each script had its own settings block. run5 and run5B used different effective-n thresholds (83 and 50) and different model folders. | `hmm_config.m` holds every setting, and run4e, run4d, run5 and run5B all read it. Environment variables override single fields for one run (`HMM_K`, `HMM_LOWRANK`, `HMM_PCA`, `HMM_MIN_NEFF`, `HMM_MODEL_DIR`, ...). run4e saves the configuration it used (`hmm_config_used.mat`, and `R.config`). | One place to set and report the analysis. run5 and run5B now share the density (0.15) and the threshold (Kish n ≥ 83; your decision). |
| A2 | Nothing linked an output file to the fit it came from | **Fit ID.** `hmm_fit_id.m` returns the SHA-256 of the fit's state probabilities (Gamma, as double). run4e writes it to `fit_id.txt`, `R.fit_id` and `hmmmar_results.mat`. run4d, run5, run5B and the drift script check it and write it as a `fit_id` column in every CSV. | Any table can be traced to its fit. A file from another fit stops the step with an error. Known-answer test: the same digest as Python `hashlib` (W8). |

**Fit IDs for files from the previous run4e.** These files have no stored ID. run4d computes it from their Gamma; run5 and run5B warn and compute it.

## `hmm_fit_raw_timeseries.m`

| # | Before | Now (published function) | Effect on results |
|---|---|---|---|
| H1 | Group state FC from our own estimator: Gamma-weighted scatter pooled over mice, on raw (not z-scored) data, centred within each mouse and state | `R.state_fc`: the Gamma-weighted scatter of z-scored ROI data pooled over mice (HMM-MAR's state statistic, `setxx.m`; your decision). `R.state_fc_model`: `getFuncConn(hmm, k, true)`, the model's own estimate. | Synthetic: r = 0.999 with the previous estimate. Real fit: see `r_previous_groupFC_vs_pooled_scatter` in the real-data test. `getFuncConn` vs previous on the real fit: r = 0.68 / 0.81 / 0.88. |
| H2 | Fractional occupancy: mean of Gamma per mouse | `getFractionalOccupancy(Gamma, T, hmm.train, 2)` | Identical (T2) |
| H3 | Not computed | `getMaxFractionalOccupancy`: the HMM-MAR diagnostic for states that describe whole subjects | New output only |
| H4 | Switching rate not computed | `getSwitchingRate` | New output only |
| H5 | State labels = most probable state per time point (stored as `R.vpath`); dwell = mean run length of those labels | Viterbi path returned by `hmmmar` (it calls `hmmdecode`); dwell = mean of `getStateLifeTimes(vpath, T, hmm.train, 0, [], false)` | Real mouse fit: labels agree at 91.7% of TRs; median `R.dwell_TR` rises from 9.9 / 8.6 / 5.9 to 15.2 / 14.0 / 8.0 TRs. The manuscript tables did not use `R.dwell_TR`: the previous run4d took dwell from the dual-estimation Viterbi paths (see run4d below). |
| H6 | Empirical transition counts and probabilities from the labels | `hmm.P` (`R.P_model`) and `getTransProbs(hmm)` (`R.P_offdiag`) | `R.transition_counts` and `R.P_emp` are removed |
| H7 | Custom renormalisation of Gamma rows | Removed: `hmmmar` returns normalised probabilities | None |
| H8 | Free energy not returned | `R.fehist` from `hmmmar` | New output only |

**Note on H1 for the reported HMM-PCA model.**
- **What `getFuncConn` returns.** It returns W·W′ + σ²I, a full-rank ROI-space matrix. That is the probabilistic-PCA fit to the same pooled scatter (`updatePCAparam.m`: `S = XXGXX{k}`).
- **Evidence (T4).** On a simulated HMM-PCA fit, r(`R.state_fc`, `getFuncConn`) = 0.997 and 0.998. `R.state_fc` (pooled scatter) and `R.state_fc_model` (`getFuncConn`) are both kept.

**Note on H1 for the pre-PCA folders (`_pca<p>`, including the previous `k3_pca15` fit).**
- **What `getFuncConn` returns for them.** For this model, `getFuncConn(hmm, k, true)` returns A·C·A′ (`getFuncConn.m`, the `hmm.train.A` branch). That is a rank-15 matrix in ROI space.
- **Proof of the relation (T7).** On a simulated pre-PCA fit, `getFuncConn` equals the pooled Gamma-weighted ROI scatter (the sum of `hmm_subject_state_fc` scatters) projected onto the PCA subspace, A·A′·S·A·A′. The agreement is 1 − r = 1.6e-6, and the remainder is HMM-MAR's Wishart prior.
  - The model's state FC is therefore the 15-component part of the same statistic the per-mouse FC uses.
  - The previous estimator, and the per-mouse FC, keep all 82 dimensions. That is why r is only 0.68–0.88 on the real fit.
- **Decision (applied).** For ROI-space group state FC (figures, group-level graphs), use the pooled scatter, the same statistic as the per-mouse FC. `getFuncConn` is kept as `R.state_fc_model`, the model's 15-component version. A rank-15 matrix is a poor basis for 82-node graphs.
- **What the old estimator added.** Its centring within mouse and state and its pooling of raw data were our additions. Pooling raw data gave mice with larger BOLD amplitude more weight.
- **`R.fo` is not added.** The finished confirmatory run's demeaning used equal state weights (`fo_source = equal`), because the previous function had no field that `local_get_fo` reads.

## `hmm_subject_state_fc.m` (new; replaces `compute_subject_state_fc_robust` in run5)

**The per-mouse state FC applies HMM-MAR's state statistic to one mouse's ROI data:**

```matlab
X  = standardisedata(X, size(X, 1), 1);   % HMM-MAR utils/preproc/standardisedata.m (line 21)
Sk = bsxfun(@times, X, w)' * X;           % HMM-MAR utils/macro/setxx.m line 56:
                                          %   XXGXX{k} = bsxfun(@times, XX, Gamma(:,k))' * XX;
FC = Sk ./ (sqrt(diag(Sk)) * sqrt(diag(Sk))');   % corrcov
```

- **Where HMM-MAR uses it.** HMM-MAR uses this scatter for state k's HMM-PCA update (`train/obs/updatePCAparam.m` line 13: `S = XXGXX{k}`) and for the full-covariance update (`train/obs/updateOmega.m`).
- **For the reported HMM-PCA model** the scatter is exactly the statistic of the state update. T1 was run on an HMM-PCA fit.
- **For pre-PCA fits (the previous `k3_pca15`).** HMM-MAR computes the scatter on the 15 PCA components. The ROI-space version here is the same weighting applied to all 82 ROIs, and its projection onto the components is what the model uses (T7).
- **Evidence (T1).** Summed over mice, our per-mouse scatter equals HMM-MAR's `XXGXX` computed by `setxx.m` on the same data and Gamma. Relative maximum difference: 2.8e-16.
- **What changed from the previous version.**
  - The data are z-scored per mouse, as `hmmmar` does. This changes nothing within one mouse, because a correlation doesn't depend on scale.
  - There is no centring within states (zero-mean states have none). This is the only difference (T3b: 2.2e-16 once the centring is removed). Synthetic maximum |Δr| was 0.019.
- **Kept as before.** The effective-n filter (Kish n ≥ 83) and its definition are unchanged.

## `run4e_hmm_mar.m` (fits the chosen model and builds its output folder)

| # | Before | Now | Why |
|---|---|---|---|
| E1 | **Two fits.** `hmmmar` was called directly (saved in `hmmmar_results.mat`, read by run4d), then again inside `hmm_fit_raw_timeseries` (saved in `raw_hmm_results.mat`, read by run5). `corrStaticFC`, called before the first fit with `use_pca = 1`, draws random numbers (`corrStaticFC.m`: `X = X + 1e-8 * randn(size(X))`). So the two fits started from different random states and need not agree, including in the order of the states. | **One fit,** through `hmm_fit_raw_timeseries`. Both files are written from it. `hmmmar_results.mat` keeps its variable names (`options`, `hmm`, `Gamma`, `vpath`, `GrpData`, `SubjData`, `groups`, `group_names`, `metadata_aligned`, `subject_ids`, `fehist`). | FO and dwell tables (run4d) and graph metrics (run5) must describe the same states. `tests/check_fit_consistency.m` checks whether your existing outputs came from one fit. |
| E2 | `explainedvar_PCA` and `corrStaticFC` were run before the fit, and their outputs were not saved. Two custom summaries (`MeanCorrFC`, `SumCorrFC`) were computed but not saved. | Both run after the fit, so they cannot change it. Saved as `R.pca_explained_group`, `R.pca_explained_subj` and `R.static_fc_similarity` (`corrStaticFC` is HMM-MAR's diagnostic for subject-specific states). The custom summaries are removed. | Published diagnostics are kept; unsaved custom code is dropped. |
| E3 | Dual estimation (`hmmdual`) per mouse, and its Viterbi paths used for dwell in run4d | Still computed (`R.dual`, and `SubjData` in `hmmmar_results.mat`) as an extra output, but not used for the temporal metrics (your decision) | Dual estimation re-estimates states per mouse, so its paths are not the group states used for FC and graph metrics |
| E4 | Data checks repeated in the script and in the fit function | Checks done once, in `hmm_fit_raw_timeseries`; metadata are aligned to the mice the fit kept | One place to audit |
| E5 | Settings hard-coded | `hmm_config.m` (A1); environment variables override the defaults, as in run4c | The same script runs every configuration and the test |
| E6 | No record of which fit a later table came from; occupancy by scan quarter only in the confirmatory run | Writes `fit_id.txt` (A2) and `occupancy_by_quarter.csv`: per mouse, state and quarter, the mean state probability. The quarters are defined as in `run4c_confirm_lowrank.m`. | The pre-specified drift test can be run on the reported fit (see `hmm_occupancy_drift.py`) |
| E7 | `options.useParallel` not set, so HMM-MAR's default applied: 1 for more than one subject (`checkoptions.m` lines 262–263). That opens a parallel pool (`hmmmar.m` line 101: `gcp`) and runs the 15 initialisations in `parfor` (`hmmmar_init.m` line 36). | `options.useParallel = 0`. Together with `lowrank = 15`, the options are now those of the confirmatory full fits (`run4c_confirm_lowrank.m`: base options with `initrep = 15`). | Same model and settings as the fit that was tested. On a 1-core job a pool brings no speed. |
| E8 | `rng(123)` before the fit, implying a reproducible fit | The seed is kept and recorded, but **it does not fix the fit.** HMM-MAR's initialisation reseeds from the clock: `train/initGamma_random.m` lines 16–17, `rng('default'); rng('shuffle') % make this "truly" random`. It is called by every initialisation (`hmmmar_init.m` → `run_initialization`). | Every run gives a new fit with a new fit ID. This applies to all HMM-MAR fits made so far, including the confirmatory full fits. Evidence: two fits seeded identically (Octave 8.4, this HMM-MAR commit) differed (max \|ΔGamma\| = 1). To confirm your copy has these lines: `grep -n shuffle /blue/febo/MATLAB/HMM-MAR-master/train/initGamma_random.m`. The record is the saved fit and its fit ID (A2). The split-halves and surrogates of run4b/run4c use their own `RandStream`s and are reproducible. HMM-MAR is not patched (see below). |

**No longer saved:** `Xi`, `GammaInit`, `residuals`, `feterms`, `rho`. If another script reads them, tell me.

**Why HMM-MAR is not patched for E8.** Deleting the two lines would make seeded refits reproducible. But it would change published code, and the patched fits would differ in kind from every fit made so far, including the confirmatory ones. Reproducibility here rests on the saved fit (A2) and on the 15 initialisations plus the split-half test, not on refitting.

## `run4d_hmm_mar_plots_analysis.m` (long-format tables across K and model dimension)

| # | Before | Now | Effect |
|---|---|---|---|
| D1 | Dwell from the dual-estimation Viterbi paths (`SubjData.vpath_subj`), with our own run-length code; FO, switching rate and max FO from the group fit | Every metric from the group fit: FO, switching rate and max FO from `GrpData` (HMM-MAR functions), dwell from the group Viterbi path (`GrpData.viterbipath`) with HMM-MAR `getStateLifeTimes` (your decision). New column `state_path = group_viterbi`. | Dwell values change. `tests/check_fit_consistency.m` reports the dual vs group medians on your data. Same CSV names and columns otherwise. |
| D2 | Each file loaded twice, with two different rules for subject IDs | One loop, one rule (`metadata_aligned`, then `subject_ids`, then row numbers with a warning) | None |
| D3 | Settings hard-coded | `hmm_config.m` (A1) | None |
| D4 | No fit record | `fit_id` column in all five tables. Stops if a folder's `hmmmar_results.mat` and `fit_id.txt` disagree. | None |
| D5 | One grid, `hmmmar_simple_k<K>_pca<p>` | Two families, each with its own set of files and a `model_type` column. HMM-PCA (`_hmmpca<r>`; r from `cfg.lowrank_values`, default 15, 25, 50) is written to `*_all_K_all_lowrank.csv` with a `lowrank_components` column. Pre-PCA keeps its file names (`*_all_K_all_PCA.csv`, `pca_components`). Missing folders are skipped. | The reported model is in the `_lowrank` files. Read those in the Python group analysis, filtered on `lowrank_components == 15` and the reported K (`K == 4`). Separate files keep a filter on component number from mixing the two model types. |

**Previous files are still readable.** Files from the previous run4e include `GrpData.viterbipath`, so run4d works on every existing K × PCA folder without refitting. Each of those files is internally one fit.

## `run5_analyze_raw_hmm_graph_metrics.m`

| # | Before | Now | Effect |
|---|---|---|---|
| G1 | Per-mouse FC from `compute_subject_state_fc_robust` | `hmm_subject_state_fc` (above) | Small change in r (the centring term) |
| G2 | Proportional threshold: our own exact-rank code | BCT `threshold_proportional` (keeps `round((n²−n)p/2)` edges, line 39) | Identical edges and metrics on simulated FC (T6: 36 matrices, max difference 0). The two can differ only for exactly tied weights, which continuous correlations don't produce. |
| G3 | `graph`/`conncomp` (MATLAB) for components | BCT `get_components` | Same counts |
| G4 | 1./W for lengths | BCT `weight_conversion(W, 'lengths')` (same formula) | None |
| G5 | An older copy of the graph function sat after the last `end` (lines 501–622) | Removed | MATLAB requires local functions to be the last code in a script, so that block could not have run |
| G6 | `FO`, `map_TR_count` from our helper | `getFractionalOccupancy` per mouse. The labels come from `R.vpath`: the Viterbi path in new fits, and the same most-probable-state labels as before in old fits. | FO identical (T2) |
| G7 | Paths hard-coded | `hmm_config.m` (A1); defaults unchanged | None |
| G8 | No fit record | Checks the fit ID of `raw_hmm_results.mat`. Adds `fit_id` and `min_effective_n` columns to the CSV. | None |
| G9 | Wrote `R_analysis_template.R` (an `lmer` model) | Removed: the group statistics are done in Python | None on the CSVs |

**Unchanged:** the effective-n filter, positive-edge graphs at 15% density, Louvain modularity (50 runs, seeded), characteristic path length only for connected graphs, the CSV columns and the preliminary plots.

## `hmm_occupancy_drift.py` (new): drift test on the reported fit

- **What it is.** The drift block of the confirmatory analysis (`analyze_confirm_lowrank.py`), unchanged, run on the fit that is reported. Per mouse and state, the change in occupancy between scan halves is (Q3 + Q4)/2 − (Q1 + Q2)/2. It is tested across mice with a two-sided Wilcoxon signed-rank test (`scipy.stats.wilcoxon`), with Holm correction across states (statsmodels `multipletests`).
- **Input and output.** It reads `occupancy_by_quarter.csv` from run4e and writes `occupancy_drift.csv` with the fit ID. It stops if the table mixes fits or disagrees with `fit_id.txt`.
- **Why.** The confirmatory run tested drift on its own full fit of HMM-PCA 15. The reported fit is a separate fit of the same model (E8), so the same test is repeated on it.
- **Evidence.** `tests/test_occupancy_drift.py` runs the confirmatory script's drift block verbatim on the same simulated table. Maximum difference: 5.55e-17.

## `run5B_hmm_block_bootstrap_stability.m` (block-bootstrap stability)

The MATLAB script now includes the Python v2 diagnostics. It runs on the same fit as run5, with the same networks and threshold. The Python port is superseded and kept in `tests/legacy/` for reference.

| # | Previous MATLAB run5B | Now | Evidence or effect |
|---|---|---|---|
| B-1 | Read `../signals/hmmmar_simple_k3_hmmpca50/raw_hmm_results.mat`. That is a different fit (HMM-PCA folder, 50 components) from the model reported at the time (`hmmmar_simple_k3_pca15`). | Reads the reported fit (`cfg.model_dir`) and checks its fit ID. To run on another folder, set `HMM_MODEL_DIR`. | **The previous stability results describe the HMM-PCA 50 fit, not the model the graph metrics come from.** |
| B-2 | State FC from the script's own weighted covariance: raw data, centred within each state | `hmm_subject_state_fc` (HMM-MAR's state statistic), the same function run5 uses | B6: removing the within-state centring from the previous formula gives the updated FC exactly (4.4e-16). Simulated max \|Δr\| = 0.012, similar to run5's change (T3a, 0.019). |
| B-3 | Its own exact-rank positive-edge threshold, with ties broken by edge position | BCT `threshold_proportional` on the positive edges, as run5. Strength from BCT `strengths_und`. | B2: identical edges and strengths on 100 matrices, 50 of them with exactly tied weights. BCT sorts with `sortrows`, which is stable, so it breaks ties by edge position too. |
| B-4 | Spearman ρ from `tiedrank` + `corrcoef` | `corr(x, y, 'Type', 'Spearman')` | B3 (MATLAB): identical, with and without ties |
| B-5 | Rank from its own eigenvalue count. Entropy effective rank of the raw-scale covariance. | MATLAB `rank`. Same entropy formula (Roy & Vetterli 2007), applied to the state statistic of z-scored data (the matrix HMM-MAR fits with `standardise = 1`, and the one the networks come from). | B4: both functions identical on the same matrix. **Effective-rank values change**, because effective rank depends on each ROI's scale: in simulation 15–17 before, 24–27 now. The rank itself is unchanged when T > 82. |
| B-6 | Kish n ≥ **50** | Kish n ≥ **83**, `cfg.min_effective_n`, the same as run5 (your decision) | Fewer mouse × state networks are analysable. run5B and run5 now include the same ones (W6). |
| B-7 | One seed for the whole run | One seed per mouse (`boot_seed` + subject ID) | Results no longer depend on the order mice are processed. Bootstrap summaries differ from before by Monte Carlo error only. B1: the circular block bootstrap (Politis & Romano 1992) draws identical indices for the same random state. |
| B-8 | `map_count` from the most probable state per TR. Gamma rows renormalised. | `map_count` counts the Viterbi labels (`R.vpath`), as run5's `map_TR_count`. No renormalisation (`hmmmar` returns normalised probabilities, as H7). | Same labels as run4d and run5 |
| B-9 | Not computed | The Python v2 diagnostics, ported with the same definitions (listed below the table). The switching descriptors now come from the group Viterbi path, through HMM-MAR `getStateLifeTimes` and `getMaxFractionalOccupancy` (`R.lifetimes_TR`, `R.maxFO`). | Dwell and max FO match run4d (W3). Python v2 took dwell as `map_count / n_visits` from the most probable state, so its column `mean_dwell_volumes` is now `mean_dwell_TR`. |
| B-10 | Outputs: two CSVs and a .mat | Same three files and the same v1 columns. Adds the v2 columns and `fit_id`; the .mat stores `cfg`. | run5B needs `raw_hmm_results.mat` from the updated run4e, and stops with a message otherwise. |

**Diagnostics ported from Python v2 (B-9):**
- resample-vs-resample Jaccard and ρ, from disjoint pairs of successful resamples (at most 500);
- split-half Jaccard and ρ, first vs second half of the scan;
- failure counts, split into resamples that fail the Neff criterion and those whose network fails;
- unfiltered summaries that keep every resample;
- posterior mass, with an optional mass criterion (`HMM_MIN_MASS`);
- switching descriptors (visits, mean dwell, transitions, max FO) and a near-static flag (FO ≥ 0.85).

**Unchanged:**
- Density 15%, positive edges only.
- 10,000 resamples per mouse and state.
- Circular blocks of ceil(T^(1/3)) TRs, which is 9 TRs for 570-TR scans. The previous script's comment says "T = 197 gives about 6", which does not match the mouse scans.
- An analysable network needs at least 80% successful resamples.
- Edge-set Jaccard and nodal-strength ρ against the full-data network, with median, mean and 2.5–97.5% interval.
- The state summary table.

## Tests

- **`tests/test_hmm_v2_synthetic.m`** (MATLAB or Octave): T1–T7 on simulated data.
  - Run here in Octave 8.4 with HMM-MAR at the commit above and BCT 2019-03-03.
  - All "must be equal" tests passed (T1 4e-16, T2 1e-16, T3b 2.2e-16, T6 0, T7 1 − r = 1.6e-6).
  - Octave lacks `corrcov`, `RandStream`, `graph`/`conncomp` and the Statistics Toolbox `pca`/`mnrnd`. These were supplied as follows (MATLAB uses its own functions):
    - Octave's statistics-package `pca.m`/`mnrnd.m`;
    - a stand-in `corrcov` (R = C./(s·s′));
    - `rand('seed', …)` in place of `RandStream`, inside the test copies only;
    - BCT `get_components` in place of `graph`/`conncomp`, inside the test copies only.
- **`tests/test_hmm_v2_real_data.m`** (MATLAB on HiPerGator; no refit): previous vs updated code on the existing mouse fit.
  - Per-mouse FC differences.
  - Graph metrics from the whole previous pipeline vs the updated one, plus the threshold change alone.
  - Group FC vs `getFuncConn`, FO, Viterbi vs most-probable-state dwell, and maxFO per mouse.
  - **Run 2026-10-01 on HiPerGator** (MATLAB R2024b; 95 mice, K = 3, pre-PCA 15):
    - per-mouse FC max |Δr| median 0.021, max 0.14 (largest in low-occupancy states);
    - the graph code alone (BCT threshold) changed nothing (maximum difference 0);
    - the whole pipeline correlated with the previous one at r ≥ 0.956 per metric and state (most > 0.99);
    - FO identical (3e-16);
    - Viterbi vs most-probable state agree at 91.7% of TRs;
    - maxFO median 0.52, with 2 of 95 mice above 0.9.
- **`tests/check_fit_consistency.m`** (MATLAB, real data): compares `hmmmar_results.mat` and `raw_hmm_results.mat` in a model folder written by the previous run4e. It uses HMM-MAR `getGammaSimilarity` and state matching, and also compares dual-estimation dwell with group-Viterbi dwell.
- **`tests/test_run5B_helpers.m`** (MATLAB or Octave): the updated run5B against the previous one, function by function (B1–B6 above).
  - Each script's local functions are extracted unchanged (`tests/extract_local_functions.m`) and called directly, so the test runs the code the scripts run.
  - **Run here in Octave 8.4.** Every "must be equal" check passed:
    - block indices: 0;
    - networks: 0 of 100 matrices differ;
    - rank: 0;
    - effective-rank formula: 0;
    - Kish n: 0;
    - FC without centring: 4.4e-16.
  - The Spearman check (B3) needs MATLAB and runs in the HiPerGator job.
- **`tests/check_against_confirmatory.m`** (MATLAB, real data; after run4e): compares the reported fit with the confirmatory full fit of the same model (`hmm_confirm_lowrank/observed/gamma_lowrank_15_K4.mat` for the reported model). It reports:
  - HMM-MAR `getGammaSimilarity` (0–1) and the state matching;
  - per matched state, the correlation of the state time courses, of FO across mice, and of the pooled state FC;
  - both fit IDs.
- **`tests/test_workflow_synthetic.m`** (MATLAB): runs the updated run4e → run4d → run5 → run5B chain on 12 simulated mice with an HMM-PCA model (40 resamples). It checks:
  - every output comes from one fit (W1);
  - FO and dwell in the tables equal the fit's (W2–W3);
  - the state FC file matches (W4);
  - run5's rows and FO match (W5);
  - run5B's Neff equals run5's, with the same inclusion rule (W6);
  - all six CSVs carry the fit ID in `fit_id.txt` (W7);
  - `hmm_fit_id` passes a known-answer test (W8);
  - the fit is HMM-PCA with `useParallel = 0` (W9).
- **`tests/test_occupancy_drift.py`** (Python): the drift script against the confirmatory code (5.55e-17, PASS here).
- **`tests/legacy/`**: verbatim copies of the previous code, used only by the tests. This includes the previous run5B and its Python v2 port.

## What needs rerunning

1. **The reported model (HMM-PCA 15, K = 4)**: ten fits (`run_hmm_fits_array.sbatch`), the selection (`run_select_fit.sbatch`), then, on the chosen fit only, these steps (`run_selected_fit.sh`, which runs `run_hmm_pipeline.sbatch` with `SKIP_FIT=1`):
   1. run4e (already done in the array: skipped);
   2. `tests/check_against_confirmatory.m`;
   3. `python hmm_occupancy_drift.py`;
   4. run4d;
   5. run5;
   6. run5B;
   7. the Python group analysis, on the `_lowrank` tables and the `hmmmar_simple_k4_hmmpca15` folder inside `fit_NN/`, restricted by `inference_scope.csv`.
2. **Results from `hmmmar_simple_k3_pca15` are superseded for the main text.** That includes the graph metrics, the FO and dwell tables, and any figures. They can stay as a Supplement comparison (pre-PCA vs HMM-PCA).
3. **Supplementary stability results.** The previous bootstrap stability numbers came from the HMM-PCA 50 fit, with Neff ≥ 50 (B-1, B-6). Replace them with the run5B output from the reported fit.
4. **Confirmatory test: no rerun.** It already tested HMM-PCA 15 and 25, and its results stand as run.
   - Its drift check was made on its own full fit.
   - `hmm_occupancy_drift.py` repeats the same test on the reported fit, and `check_against_confirmatory.m` shows how close the two fits are.
5. **The pre-PCA model gets no confirmatory test** (your decision).
