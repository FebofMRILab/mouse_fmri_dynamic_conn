# HiPerGator jobs: setup, tests, HMM pipeline

The run order for the whole project is in the top-level `README.md`. Submit these jobs from `code/` (`/blue/febo/share/Abeta_Age/rabies/code`). The data folder is `/blue/febo/share/Abeta_Age/rabies/signals`; set `HMM_OUT_BASE` at submission to use another.

- `link_into_signals.sh`: run once. It links the packages that run inside the data folder (`hmm_null`, `dfc`, `positive_control`) into `/blue/febo/share/Abeta_Age/rabies/signals`.

The scripts use the same account, QOS and limits as your other jobs: febo, 1 core, 8 GB.

## Layout

`code/` (`abeta_fc_code.zip` unpacks as `code/`) contains, among others:

```
hmm_v2/              from hmm_v2_code.zip
hmm_py/              from hmm_python_stats.zip
static_v2/           from static_fc_published.zip
dfc/                 from dfc_dcc_cpca_matlab.zip (optional here; its test runs if present)
hipergator_tests/    this folder
logs/                (mkdir -p logs)
```

## 1. HMM tests (MATLAB, then Python; about 1-1.5 h)

Set up the Python environment first (once; see section 2). The last step of the HMM job uses it for the drift test.

```bash
sbatch hipergator_tests/run_hmm_tests.sbatch
```

The defaults:
- the fit: `<signals>/hmmmar_simple_k3_pca15/raw_hmm_results.mat`. Steps 2 and 3 compare previous and updated code on a fit that already exists, so they use the previous fit (K = 3, pre-PCA 15), not the reported model. If that fit is gone, steps 2 and 3 are skipped (they passed on 4 Oct 2026) and every other step runs;
- the signals: `<signals>/hmm_signals`, where `<signals>` is `/blue/febo/share/Abeta_Age/rabies/signals`
- HMM-MAR: `/blue/febo/MATLAB/HMM-MAR-master`
- BCT: found automatically under `/blue/febo/MATLAB`

If any of these is elsewhere, override it:

```bash
sbatch --export=ALL,BCT_DIR=/path/to/BCT,HMM_RESULTS=/path/raw_hmm_results.mat hipergator_tests/run_hmm_tests.sbatch
```

Output goes to `hmm_v2_test_results/`:
- `synthetic_tests.csv` (T1–T7)
- `run5B_helper_tests.csv` (B1–B6)
- `real_data/`, containing the CSVs and `test_log.txt`
- `fit_consistency/`
- `workflow/workflow_tests.csv` (W1–W10), plus the simulated run in `workflow/work/`
- `select_fit_tests.csv` (F1–F15: the fit-selection rule on simulated fits with a known answer, including the model record and a Gaussian PCA model)
- `select_on_run4e_tests.csv` (R1–R4: the selection on eight real run4e fits of the simulated cohort)
- `state_descriptor_tests.csv` (D1–D7: amplitude vs correlation pattern on simulated states)
- `occupancy_drift_test.txt` and `occupancy_drift_workflow.txt`
- `hodges_lehmann_test.txt` (H1–H9: the intervals and the K = 2 drift fix of the confirmation analysis)
- `model_files_test.txt` (M1–M8: the job scripts read the two reported models from their files; stubs, nothing is fitted)

## 2. Static-FC tests (Python, about 15-30 min)

Set up the environment once:

```bash
bash hipergator_tests/setup_python_env.sh
```

If you created the environment before, add python-docx (needed for the Word tables): `conda install -p /blue/febo/$USER/envs/fc_static -c conda-forge python-docx`.

Then submit:

```bash
sbatch hipergator_tests/run_static_tests.sbatch
```

Output goes to `static_test_results/`. The same job also tests `hmm_py/` (about 6 min; includes runs for both reported models, and the agreement script, A1–A6) and `dfc/` if they are present.

The two jobs are independent and can run at the same time (2 cores in total).

## 3. The reported model: ten fits, the selection rule, then the pipeline

The paper reports one model, the **primary** (`hmm_v2/MODEL_PRIMARY.env`): Gaussian HMM on 25 PCs, K = 3, in `<signals>/hmm_k3_pca25_fits/`. It is set in one file and loaded by `reported_model.sh`.

The **secondary** model (`hmm_v2/MODEL_SECONDARY.env`: HMM-PCA rank 15, K = 2, in `<signals>/hmm_k2_hmmpca15_fits/`) was run during development and is not reported in the paper. Its commands below are optional. (The rule is `hmm_v2/FIT_SELECTION_RULE.md`, amendment 1 of 27 Oct 2026.)

**Do not edit the rule or the model files after step 3a is submitted.** The array job records the rule's SHA-256 and copies the model file, both into the fits folder (`rule_sha256.txt`, `model.env`) and per role into `<signals>/hmm_fits_<role>_model.env`. Every later task, the selection and `run_selected_fit.sh` stop if either has changed.

```bash
# 3a. ten independent fits per model, one core and 8 GB each (about 1-3 h each);
#     at most 8 tasks per array at once, so both arrays fit in 16 cores
sbatch hipergator_tests/run_hmm_fits_array.sbatch primary              # prints the job ID
sbatch hipergator_tests/run_hmm_fits_array.sbatch secondary
# 3b. each model's selection, once its ten fits have finished (minutes)
sbatch --dependency=afterany:<primary job ID> hipergator_tests/run_select_fit.sbatch primary
sbatch --dependency=afterany:<secondary job ID> hipergator_tests/run_select_fit.sbatch secondary
# 3c. read <fits>/selection/selection_report.txt of each model, then
bash hipergator_tests/run_selected_fit.sh primary                       # steps 2-8 below, on the chosen fit
bash hipergator_tests/run_selected_fit.sh secondary
```

- **Fits.** They go to `<fits>/fit_01` … `fit_10`. A task never overwrites a finished fit.
  - Rerun failed tasks with `sbatch --array=3,7 hipergator_tests/run_hmm_fits_array.sbatch primary`.
  - Tasks above 10 are refused.
  - Every `HMM_*` setting except `HMM_TOOLBOX` is ignored; the model comes only from its file.
- **Selection.** It needs at least 8 valid fits. It writes `selection/` (report and tables) and `inference_scope.csv` in the chosen fit's folder. It runs once: a different result on a rerun is refused.
- **Agreement.** When both pipelines have finished, `python hmm_py/hmm_model_agreement.py` (from `code/`, in the `fc_static` environment) applies `hmm_null/MODEL_SELECTION_RULE.md` amendment 2 and writes `<signals>/hmm_model_agreement/`.
- **Pipeline.** `run_selected_fit.sh` submits the pipeline below with `SKIP_FIT=1` (no refit) on the chosen fit, with the model's confirmation fit for the comparison (step 6b's for the primary, step 6's for the secondary). Every output goes inside `fit_NN/`. The Python statistics test only what `inference_scope.csv` allows.

### The pipeline job

```bash
sbatch hipergator_tests/run_hmm_pipeline.sbatch
```

On its own (without `SKIP_FIT=1`) it fits once and runs every step: the way to run an exploratory model (e.g. `--export=ALL,HMM_K=4,HMM_LOWRANK=15,HMM_PCA=0`; set all three, as unset ones come from `MODEL_PRIMARY.env`). For the reported models, use section 3 above. It runs on one core and stops at once if the signal folder is missing:
1. run4e;
2. the comparison with the confirmation's full fit;
3. the drift test (Python);
4. run4d;
5. run5;
6. the state descriptors (amplitude vs correlation pattern; descriptive);
7. run5B;
8. the Python statistics and figures in `hmm_py/`.

It takes several hours. It needs BCT (found automatically under `/blue/febo/MATLAB`, or set `HMM_BCT`) and the Python environment from section 2.

## 4. DCC cross-check against the published DCC toolbox

Once, get the toolbox: `git clone https://github.com/canlab/Lindquist_Dynamic_Correlation /blue/febo/MATLAB/Lindquist_Dynamic_Correlation`. Then:

```bash
sbatch hipergator_tests/run_dcc_crosscheck.sbatch
```

It needs MATLAB's Optimization Toolbox and stops with a message if it is not licensed. Output: `dcc_lindquist_check/` (`dcc_lindquist_comparison.csv`, `dcc_lindquist_summary.txt`). Set `DCC_CODE` if the toolbox is elsewhere.

## 5. Supplement figures in natural units (DCC/GARCH/complex PCA)

```bash
sbatch hipergator_tests/run_dcc_example.sbatch
```

It needs the Lindquist toolbox (section 4) and the Optimization Toolbox, and reads the main results in `<signals>/dfc_dcc_cpca`. Output: `dfc_dcc_cpca/figures/FigS_dfc_raw_units.*` with `table_raw_units.csv`, and `dfc_dcc_cpca/dcc_example/` (the figure, its caption, the CSVs and `dcc_example_info.txt`, which names the mouse drawn).

## What to send back

One file: from `code/`, run `bash hipergator_tests/collect_results.sh`. It bundles, without `.mat`, `.svg` or `.tiff` files:
- `hmm_v2_test_results/`, `static_test_results/` and `logs/`;
- `dcc_lindquist_check/`;
- `dfc_dcc_cpca/dcc_example/` and the raw-units figure and table;
- `hmm_model_agreement/` (step 10a);
- for each reported model after its selection: `<fits>/selection/`, `rule_sha256.txt`, `model.env` and the chosen fit's `inference_scope.csv`;
- from the chosen fit's model folder: `fit_id.txt`, `occupancy_drift.csv`, `check_against_confirmatory/`, `state_descriptors/`, the graph-metric and bootstrap tables, `hmm_graph_figures/` and `bootstrap_supplement/`;
- `hmm_dynamics_stats_<tag>/`, `hmm_dynamics_figures_<tag>/` and the three run4d tables of that fit.
