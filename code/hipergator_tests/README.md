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
- the fit: `<signals>/hmmmar_simple_k3_pca15/raw_hmm_results.mat`. These tests compare previous and updated code on a fit that already exists, so they use the previous fit (K = 3, pre-PCA 15), not the reported model;
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
- `workflow/workflow_tests.csv` (W1–W8), plus the simulated run in `workflow/work/`
- `select_fit_tests.csv` (F1–F13: the fit-selection rule on simulated fits with a known answer)
- `select_on_run4e_tests.csv` (R1–R4: the selection on eight real run4e fits of the simulated cohort)
- `occupancy_drift_test.txt` and `occupancy_drift_workflow.txt`

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

Output goes to `static_test_results/`. The same job also tests `hmm_py/` (about 4 min; includes the K = 4 run) and `dfc/` if they are present.

The two jobs are independent and can run at the same time (2 cores in total).

## 3. The reported K = 4 fit: ten fits, the selection rule, then the pipeline

The reported fit is chosen by the rule in `hmm_v2/FIT_SELECTION_RULE.md` (4 Oct 2026). **Do not edit that file after step 3a is submitted**: the array job records its SHA-256, and the selection stops if it has changed.

```bash
# 3a. ten independent fits, one core and 8 GB each (about 1-3 h each)
sbatch hipergator_tests/run_hmm_fits_array.sbatch                       # prints the job ID
# 3b. the selection, once all ten have finished (minutes)
sbatch --dependency=afterany:<job ID> hipergator_tests/run_select_fit.sbatch
# 3c. read <signals>/hmm_k4_hmmpca15_fits/selection/selection_report.txt, then
bash hipergator_tests/run_selected_fit.sh                              # steps 2-7 below, on the chosen fit
```

- The fits go to `<signals>/hmm_k4_hmmpca15_fits/fit_01` … `fit_10`. A task never overwrites a finished fit; rerun failed tasks with `sbatch --array=3,7 hipergator_tests/run_hmm_fits_array.sbatch`. Tasks above 10 are refused, and every `HMM_*` setting except `HMM_TOOLBOX` is ignored, so the fits always use `hmm_config.m`'s defaults.
- The selection needs at least 8 valid fits. It writes `selection/` (report and tables) and `inference_scope.csv` in the chosen fit's folder. It runs once: a different result on a rerun is refused.
- `run_selected_fit.sh` submits the pipeline below with `SKIP_FIT=1` (no refit) on the chosen fit. Every output goes inside `fit_NN/`, so the outputs of the 4 Oct 2026 refit in `<signals>/hmmmar_simple_k4_hmmpca15/` stay as they were. The Python statistics test only what `inference_scope.csv` allows.

### The pipeline job

```bash
sbatch hipergator_tests/run_hmm_pipeline.sbatch
```

On its own (without `SKIP_FIT=1`) it fits once and runs every step: the way to run a Supplement model (e.g. `--export=ALL,HMM_LOWRANK=25,HMM_K=2`). For the reported model, use section 3 above. It runs on one core and stops at once if the signal folder is missing:
1. run4e;
2. the comparison with the confirmatory full fit;
3. the drift test (Python);
4. run4d;
5. run5;
6. run5B;
7. the Python statistics and figures in `hmm_py/`.

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
- after the selection: `hmm_k4_hmmpca15_fits/selection/`, `rule_sha256.txt` and the chosen fit's `inference_scope.csv` (the K = 4 outputs below then come from the chosen fit);
- from the K = 4 model folder: `fit_id.txt`, `occupancy_drift.csv`, `check_against_confirmatory/`, the graph-metric and bootstrap tables, `hmm_graph_figures/` and `bootstrap_supplement/`;
- `hmm_dynamics_stats_hmmpca15_k4/`, `hmm_dynamics_figures_hmmpca15_k4/` and the three run4d tables.
