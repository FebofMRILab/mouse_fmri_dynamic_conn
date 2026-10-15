# Updated HMM code (HMM-MAR and BCT calls)

## Workflow order

Every step reads its settings from `hmm_config.m` and works on one fit, identified by its fit ID.

**Reported model:** HMM-PCA (`options.lowrank = 15`), K = 4, chosen after the pre-specified confirmatory test (rank 15/25 × K = 2–4): at rank 15, the rank the spin-echo hypothesis favoured, K = 4 is the only K that met its decision rule (calibrated median 0.051, Holm p = 0.0009; `hmm_confirm_lowrank/analysis/confirm_cells.csv`). K = 3, reported before, did not (0.007, Holm p = 0.12). Details in `CHANGES.md`, "Reported model". It is fitted with that test's options and saved in `/blue/febo/share/Abeta_Age/rabies/signals/hmmmar_simple_k4_hmmpca15/`.

**Data folder:** `/blue/febo/share/Abeta_Age/rabies/signals` (`HMM_OUT_BASE`), with the signal files in its `hmm_signals/`. Every step reads and writes there, wherever it is started from.

| Step | File | Replaces |
|---|---|---|
| 1. Fit and output folder | `run4e_hmm_mar.m` | previous run4e (fitted twice; now fits once) |
| 1b. Compare with the confirmatory full fit | `tests/check_against_confirmatory.m` | new |
| 2. Drift test on that fit | `hmm_occupancy_drift.py` | drift block of the confirmatory analysis, now run on the reported fit |
| 3. Long-format tables, all K × model dimension | `run4d_hmm_mar_plots_analysis.m` | previous run4d |
| 4. Graph metrics per mouse and state | `run5_analyze_raw_hmm_graph_metrics.m` | previous run5 |
| 5. Block-bootstrap stability of those networks | `run5B_hmm_block_bootstrap_stability.m` | previous run5B (MATLAB) and its Python v2 port |

These steps use four shared files:

| File | Purpose |
|---|---|
| `hmm_config.m` | all settings (K, HMM-PCA rank or PCA, folders, density, min effective n = 83, bootstrap) |
| `hmm_fit_id.m` | fit ID: SHA-256 of the fit's Gamma |
| `hmm_fit_raw_timeseries.m` | the fit; replaces the previous file of the same name |
| `hmm_subject_state_fc.m` | per-mouse state FC; replaces `compute_subject_state_fc_robust` in run5 and the covariance code in run5B |

`CHANGES.md` lists every change with its published source and evidence.

## On HiPerGator

1. **Run the tests:** `sbatch hipergator_tests/run_hmm_tests.sbatch` (see `hipergator_tests/README.md`).
2. **Run the pipeline for the reported model** from `code/` (the folder that holds `hmm_v2/`):

   ```bash
   sbatch hipergator_tests/run_hmm_pipeline.sbatch
   ```

   The job runs steps 1–5 on one core, in order. To run them by hand instead:

   ```bash
   M='matlab -nodisplay -singleCompThread -batch'
   $M "addpath('hmm_v2'); run4e_hmm_mar; quit force"
   $M "addpath('hmm_v2'); addpath('hmm_v2/tests'); check_against_confirmatory(); quit force"
   python hmm_v2/hmm_occupancy_drift.py                            # conda env from setup_python_env.sh
   $M "addpath('hmm_v2'); run4d_hmm_mar_plots_analysis; quit force"
   $M "addpath('hmm_v2'); run5_analyze_raw_hmm_graph_metrics; quit force"
   $M "addpath('hmm_v2'); run5B_hmm_block_bootstrap_stability; quit force"
   ```

   - **Other models.** For a Supplement model, override the setting first, e.g. `export HMM_LOWRANK=25`. For PCA before the HMM, use `export HMM_LOWRANK=0 HMM_PCA=15`. Any field in `hmm_config.m` can be overridden the same way.
   - **BCT.** run5 and run5B need BCT. If it is not on your MATLAB path, give its folder with `export HMM_BCT=/path/to/BCT`.
   - **Confirmatory fit location.** `check_against_confirmatory` looks for `<signals>/hmm_confirm_lowrank/observed/gamma_lowrank_15_K4.mat`. Set `HMM_CONFIRM_GAMMA` if it is elsewhere.
   - **Another data folder.** `export HMM_OUT_BASE=/path/to/signals` (MATLAB and Python read the same variable).
   - **Each run gives a new fit.** HMM-MAR reseeds from the clock (CHANGES.md, E8). Keep the saved folder; its fit ID identifies it.
3. **Then run the Python statistics and figures** in `hmm_py/` (from `hmm_python_stats.zip`). They read `hmm_*_all_K_all_lowrank.csv` and the `hmmmar_simple_k4_hmmpca15` folder. The pipeline job runs them at the end when `hmm_py/` is present.

## Tests

| File | Runs on | What it checks |
|---|---|---|
| `tests/test_hmm_v2_synthetic.m` | MATLAB or Octave | equivalence with HMM-MAR and BCT (T1–T7) |
| `tests/test_run5B_helpers.m` | MATLAB or Octave | the updated run5B against the previous one, function by function (B1–B6) |
| `tests/test_hmm_v2_real_data.m` | MATLAB | previous vs updated code on the existing mouse fit |
| `tests/check_fit_consistency.m` | MATLAB | whether the previous run4e outputs came from one fit or two |
| `tests/test_workflow_synthetic.m` | MATLAB | the run4e → run4d → run5 → run5B chain with an HMM-PCA model on simulated data, all from one fit (W1–W9) |
| `tests/check_against_confirmatory.m` | MATLAB | the reported fit against the confirmatory full fit of the same model (real data, after run4e) |
| `tests/test_occupancy_drift.py` | Python | the drift script against the confirmatory code |

`tests/legacy/` holds verbatim copies of the previous code.
