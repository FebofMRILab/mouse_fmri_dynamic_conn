# Updated HMM code (HMM-MAR and BCT calls)

## Workflow order

Every step reads its settings from `hmm_config.m` and works on one fit, identified by its fit ID.

**Reported models (27 Oct 2026, D33):** two models on the ROI-mean signals, each set in one file:

| | Primary (main text) | Secondary |
|---|---|---|
| Model file | `MODEL_PRIMARY.env` | `MODEL_SECONDARY.env` |
| Model | Gaussian HMM on 25 PCs (`options.pca = 25`), K = 3 | HMM-PCA (`options.lowrank = 15`), K = 2 |
| Confirmation | step 6b (`hmm_confirm_candidate`) | step 6 (`hmm_confirm_lowrank`) |
| Fits and every later output | `signals/hmm_k3_pca25_fits/fit_NN/` | `signals/hmm_k2_hmmpca15_fits/fit_NN/` |

How they were chosen: `hmm_null/MODEL_SELECTION_RULE.md` and its amendment 1. Each model is fitted ten times with the confirmatory test's options. Each model's reported fit, and the states that are tested, are chosen by the rule in `FIT_SELECTION_RULE.md` (amendment 1): lowest free energy; states that recur across fits.

`hmm_config.m` uses the primary model when no `HMM_K` / `HMM_LOWRANK` / `HMM_PCA` is set. The job scripts load either model with `hipergator_tests/reported_model.sh primary|secondary`.

The earlier choice on the eigenvariate signals (HMM-PCA rank 15, K = 4) is in `CHANGES.md`, "Reported model".

**Data folder:** `/blue/febo/share/Abeta_Age/rabies/signals` (`HMM_OUT_BASE`), with the signal files in its `hmm_signals/`. Every step reads and writes there, wherever it is started from.

| Step | File | Replaces |
|---|---|---|
| 1. Fit and output folder (ten times, one folder each) | `run4e_hmm_mar.m` | previous run4e (fitted twice; now fits once) |
| 1a. Choose the reported fit and the states to test | `select_reported_fit.m` (rule: `FIT_SELECTION_RULE.md`) | new |
| 1b. Compare with the confirmatory full fit | `tests/check_against_confirmatory.m` | new |
| 2. Drift test on that fit | `hmm_occupancy_drift.py` | drift block of the confirmatory analysis, now run on the reported fit |
| 3. Long-format tables, all K × model dimension | `run4d_hmm_mar_plots_analysis.m` | previous run4d |
| 4. Graph metrics per mouse and state | `run5_analyze_raw_hmm_graph_metrics.m` | previous run5 |
| 4a. What separates the states: amplitude (ROI and global-signal variance) or correlation pattern (descriptive) | `hmm_state_descriptors.m` (values: `hmm_state_descriptor_values.m`) | new (27 Oct 2026) |
| 5. Block-bootstrap stability of those networks | `run5B_hmm_block_bootstrap_stability.m` | previous run5B (MATLAB) and its Python v2 port |

These steps use four shared files:

| File | Purpose |
|---|---|
| `hmm_config.m` | all settings (K, HMM-PCA rank or PCA from `MODEL_PRIMARY.env` unless set, folders, density, min effective n = 83, bootstrap) |
| `MODEL_PRIMARY.env`, `MODEL_SECONDARY.env` | the two reported models (the only place they are set) |
| `hmm_fit_id.m` | fit ID: SHA-256 of the fit's Gamma |
| `hmm_fit_raw_timeseries.m` | the fit; replaces the previous file of the same name |
| `hmm_subject_state_fc.m` | per-mouse state FC; replaces `compute_subject_state_fc_robust` in run5 and the covariance code in run5B |

`CHANGES.md` lists every change with its published source and evidence.

## On HiPerGator

1. **Run the tests:** `sbatch hipergator_tests/run_hmm_tests.sbatch` (see `hipergator_tests/README.md`).
2. **The two reported models** from `code/` (the folder that holds `hmm_v2/`):

   ```bash
   sbatch hipergator_tests/run_hmm_fits_array.sbatch primary                        # step 1, ten times
   sbatch hipergator_tests/run_hmm_fits_array.sbatch secondary
   sbatch --dependency=afterany:<primary job ID> hipergator_tests/run_select_fit.sbatch primary     # step 1a
   sbatch --dependency=afterany:<secondary job ID> hipergator_tests/run_select_fit.sbatch secondary
   bash hipergator_tests/run_selected_fit.sh primary        # steps 1b-5 and hmm_py, chosen fit
   bash hipergator_tests/run_selected_fit.sh secondary
   ```

   For a single fit (an exploratory model), `sbatch hipergator_tests/run_hmm_pipeline.sbatch` runs steps 1–5 on one core, in order. To run them by hand instead:

   ```bash
   M='matlab -nodisplay -singleCompThread -batch'
   $M "addpath('hmm_v2'); run4e_hmm_mar; quit force"
   $M "addpath('hmm_v2'); addpath('hmm_v2/tests'); check_against_confirmatory(); quit force"
   python hmm_v2/hmm_occupancy_drift.py                            # conda env from setup_python_env.sh
   $M "addpath('hmm_v2'); run4d_hmm_mar_plots_analysis; quit force"
   $M "addpath('hmm_v2'); run5_analyze_raw_hmm_graph_metrics; quit force"
   $M "addpath('hmm_v2'); hmm_state_descriptors; quit force"
   $M "addpath('hmm_v2'); run5B_hmm_block_bootstrap_stability; quit force"
   ```

   - **Other models.** Override the setting first, e.g. `export HMM_K=2 HMM_LOWRANK=15 HMM_PCA=0` (the secondary model) or `export HMM_K=4 HMM_LOWRANK=25 HMM_PCA=0`. Set all three: unset ones come from `MODEL_PRIMARY.env`. Any field in `hmm_config.m` can be overridden the same way.
   - **BCT.** run5 and run5B need BCT. If it is not on your MATLAB path, give its folder with `export HMM_BCT=/path/to/BCT`.
   - **Confirmation fit location.** `check_against_confirmatory` and `select_reported_fit` look for the model's full fit from its confirmation: `<signals>/hmm_confirm_candidate/observed/gamma_gaussian_pca_25_K3.mat` (primary) or `<signals>/hmm_confirm_lowrank/observed/gamma_lowrank_15_K2.mat` (secondary). Set `HMM_CONFIRM_GAMMA` if it is elsewhere.
   - **Another data folder.** `export HMM_OUT_BASE=/path/to/signals` (MATLAB and Python read the same variable).
   - **Each run gives a new fit.** HMM-MAR reseeds from the clock (CHANGES.md, E8). Keep the saved folder; its fit ID identifies it.
3. **Then run the Python statistics and figures** in `hmm_py/`. They read run4d's tables (`hmm_*_all_K_all_PCA.csv` for the primary model, `..._lowrank.csv` for the secondary) and the model folder (`hmmmar_simple_k3_pca25` or `hmmmar_simple_k2_hmmpca15`). The pipeline job runs them at the end when `hmm_py/` is present.

## Tests

| File | Runs on | What it checks |
|---|---|---|
| `tests/test_hmm_v2_synthetic.m` | MATLAB or Octave | equivalence with HMM-MAR and BCT (T1–T7) |
| `tests/test_run5B_helpers.m` | MATLAB or Octave | the updated run5B against the previous one, function by function (B1–B6) |
| `tests/test_hmm_v2_real_data.m` | MATLAB | previous vs updated code on the existing mouse fit |
| `tests/check_fit_consistency.m` | MATLAB | whether the previous run4e outputs came from one fit or two |
| `tests/test_workflow_synthetic.m` | MATLAB | the run4e → run4d → run5 → run5B chain with an HMM-PCA model on simulated data, all from one fit, and the state descriptors on it (W1–W10) |
| `tests/test_state_descriptors.m` | MATLAB or Octave | the state descriptors on simulated states that differ in amplitude, pattern or global signal (D1–D7) |
| `tests/check_against_confirmatory.m` | MATLAB | the reported fit against the confirmatory full fit of the same model (real data, after run4e) |
| `tests/test_occupancy_drift.py` | Python | the drift script against the confirmatory code (K = 3; the K = 2 one-test fix is checked in `hmm_null/tests/hodges_lehmann/`) |
| `tests/test_select_reported_fit.m` | MATLAB or Octave | the fit-selection rule on simulated fits with a known answer, including the model record and a Gaussian PCA model (F1–F15) |
| `tests/test_select_on_run4e.m` | MATLAB | the selection on eight real run4e fits of the simulated cohort (R1–R4) |

`tests/legacy/` holds verbatim copies of the previous code.
