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

## 3. The HMM pipeline for the reported model (after the tests)

```bash
sbatch hipergator_tests/run_hmm_pipeline.sbatch
```

This runs, on one core, the reported model (HMM-PCA 15, K = 4, into `/blue/febo/share/Abeta_Age/rabies/signals/hmmmar_simple_k4_hmmpca15/`). It stops at once if `hmm_signals/` is not in the data folder:
1. run4e;
2. the comparison with the confirmatory full fit;
3. the drift test (Python);
4. run4d;
5. run5;
6. run5B;
7. the Python statistics and figures in `hmm_py/`.

It takes several hours. It needs BCT (found automatically under `/blue/febo/MATLAB`, or set `HMM_BCT`) and the Python environment from section 2.

## What to send back

- `hmm_v2_test_results/`
- `static_test_results/`
- the two log files in `logs/`
- after the pipeline job: `<signals>/hmmmar_simple_k4_hmmpca15/check_against_confirmatory/`, `occupancy_drift.csv` and its log
