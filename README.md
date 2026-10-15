# Abeta mouse functional connectivity: code

This is the code for the static and dynamic FC analyses of the Abeta mouse cohort:
- 95 mice in 7 groups;
- 82 ROIs;
- spin-echo EPI at 11.1 T, TR 2 s, 570 volumes.

It also covers the companion human Low/High Abeta static analysis.

**Main result:** HMM-PCA (rank 15), K = 4, chosen after the pre-specified confirmatory test (rank 15/25 × K = 2–4). Rank 15 is the rank the spin-echo hypothesis favoured, and at rank 15, K = 4 is the only K that met the test's decision rule (calibrated median 0.051, Holm p = 0.0009). Rank 25, K = 2 also met it; K = 3, the model reported before, did not (0.007, Holm p = 0.12). The evidence for recurring states is weak (2 of 6 cells), and no state of the confirmatory run's full K = 4 fit drifts over the scan. **4 Oct 2026:** a refit with the same options reproduced only one of the four states (`hmm_v2/CHANGES.md`, "Refits"). The reported fit and the states that are tested are now chosen by a rule fixed before fitting (`hmm_v2/FIT_SELECTION_RULE.md`; steps 7a–7c). Details: `hmm_v2/CHANGES.md`, "Reported model". Most of the other analyses explain this result and go to the Supplement.

Each package has its own `README.md` (how to run it) and `CHANGES.md` (every change to published code, with its evidence).

## 1. Folder layout

The code and the data sit in sibling folders on HiPerGator. Share only `code/`, plus the derived, de-identified matrices.

```
/blue/febo/share/Abeta_Age/rabies/
├── code/                           ← this repository (abeta_fc_code.zip unpacks as code/)
│   ├── README.md                   this file
│   ├── hipergator_tests/           environment setup, test jobs, HMM pipeline job, link helper
│   ├── static_v2/                  static FC: edgewise tests, graph metrics, NBS, ROI table
│   ├── hmm_null/                   HMM split-half null (K selection) and pre-specified confirmatory test
│   ├── hmm_v2/                     reported HMM fit and its network metrics (MATLAB)
│   ├── hmm_py/                     HMM statistics and figures (Python)
│   ├── dfc/                        window-free dynamic FC (DCC/GARCH, complex PCA), confound checks
│   └── positive_control/           simulated mice with a known mechanism
├── signals/                        ← mouse ROI data and every HMM / DFC result (not shared)
│   ├── hmm_signals/                signal1.txt … signal95.txt (570 × 82), meta_data.csv
│   ├── hmm_raw_timeseries_null/    split-half null (hmm_null, run4b)
│   ├── hmm_confirm_lowrank/        confirmatory test (hmm_null, run4c)
│   ├── hmmmar_simple_k4_hmmpca15/  reported fit: run4e, run5, run5B, hmm_py figures
│   ├── hmm_*_all_K_all_lowrank.csv run4d tables (all low-rank fits)
│   ├── hmm_dynamics_stats_hmmpca15_k4/    hmm_py statistics
│   ├── hmm_dynamics_figures_hmmpca15_k4/  hmm_py figures
│   └── dfc_dcc_cpca/               DFC results
└── static/                         ← static FC inputs and outputs (not shared)
    ├── merged.csv                  95 stacked 82 × 82 FC matrices
    ├── roi_names_order.csv, subject_group_order_mappings.csv
    └── fc_analysis_mice/           outputs (fc_analysis_humans/ for the human script)
```

**The data folder is an absolute path.** The HMM code (MATLAB and Python) and the job scripts default to `/blue/febo/share/Abeta_Age/rabies/signals`, so they find the data wherever they are started from. (From inside a package folder such as `code/hmm_v2/`, the same folder is `../../signals`.) To use another location, set `HMM_OUT_BASE`, e.g. `export HMM_OUT_BASE=/path/to/signals` or `sbatch --export=ALL,HMM_OUT_BASE=/path/to/signals ...`.

**Where each package runs from.** The packages follow two conventions:

| Packages | Run from | How |
|---|---|---|
| `hmm_v2`, `hmm_py`, `hipergator_tests` jobs | `code/` | The job scripts call `hmm_v2/…` and `hmm_py/…` by relative path, so submit them from `code/`. They read and write the data folder above (settings in `hmm_v2/hmm_config.m`). |
| `hmm_null`, `dfc`, `positive_control` | `signals/` | They were written to run inside the folder that holds `hmm_signals/`. Run `bash hipergator_tests/link_into_signals.sh` once from `code/`: it links their files into `signals/` instead of copying them. **If you copied these files into `signals/` before, move those copies away first** (e.g. into `signals/_old_code/`); the helper never overwrites a real file. |
| `static_v2` | `static/` | `cd /blue/febo/share/Abeta_Age/rabies/static`, then `python ../code/static_v2/<script>.py` |

## 2. Once: environment and toolboxes

| What | Command or version |
|---|---|
| Python environment (SciPy ≥ 1.16, statsmodels, networkx, bctpy 0.6.1, python-docx) | `bash hipergator_tests/setup_python_env.sh` (from `code/`). If the environment already exists, add `python-docx`. |
| MATLAB toolboxes | HMM-MAR at `/blue/febo/MATLAB/HMM-MAR-master` (local tests at commit `023905b`; the HiPerGator copy is a download with no commit recorded); Brain Connectivity Toolbox (2019-03-03 in local tests; 2017-01-15 on HiPerGator, where all tests passed). MATLAB R2026a on HiPerGator. |
| Link the data-folder packages | `bash hipergator_tests/link_into_signals.sh` (from `code/`) |

## 3. Run order

Status as of 4 Oct 2026:
- **run:** not yet run with the current code;
- **rerun:** code changed since the last run;
- **done:** finished, and the result stands as run.

| # | Step | Run from | Command | Status |
|---|---|---|---|---|
| **0** | **Tests** (synthetic + existing mouse fit) | `code/` | `sbatch hipergator_tests/run_hmm_tests.sbatch` · `sbatch hipergator_tests/run_static_tests.sbatch` | rerun (new: fit-selection tests 4b, 4c and hmm_py S11–S13; the rest passed 4 Oct) |
| | **Static FC** (main text; independent of the HMM) | | | |
| 1 | Mouse static FC, graph metrics, NBS | `static/` | `python ../code/static_v2/full_analysis_abeta_mice_permutation_transitivity.py` (about 4 h; submit as a 1-core job) | rerun |
| 2 | Human static FC (companion) | human static folder | `python ../code/static_v2/full_analysis_abeta_humans_permutations.py` (about 1 h) | rerun |
| 3 | ROI nodal table (replaces `statistical_table.r`) | `static/` | `python ../code/static_v2/roi_group_table.py mice_NodePosStrength_PCorr.csv` (wide file: subject index, group, one column per ROI) | run |
| | **HMM model selection** (Supplement) | | | |
| 4 | Signal rank (Marchenko–Pastur and parallel analysis) | `signals/` | `python mp_signal_rank.py --dataset mouse hmm_signals 95 2` | done |
| 5 | Split-half null across configurations | `signals/` | `sbatch slurm/hmm_listconfigs.sbatch` → `bash submit_hmm_chunks.sh all` → `sbatch slurm/hmm_collect.sbatch` → `python plot_hmm_splithalf_null.py` | done |
| 6 | Pre-specified confirmatory test (HMM-PCA 15 and 25, K = 2–4); chooses K = 4 | `signals/` | `bash submit_confirm_lowrank.sh`, then `python analyze_confirm_lowrank.py --root hmm_confirm_lowrank` | done |
| | **Reported HMM: rank 15, K = 4** (main text) | | | |
| 7a | Ten independent fits (rule: `hmm_v2/FIT_SELECTION_RULE.md`; do not edit it after this step) | `code/` | `sbatch hipergator_tests/run_hmm_fits_array.sbatch` | run |
| 7b | Choose the reported fit and the states to test (lowest free energy; states that recur across fits) | `code/` | `sbatch --dependency=afterany:<7a job ID> hipergator_tests/run_select_fit.sbatch` → read `signals/hmm_k4_hmmpca15_fits/selection/selection_report.txt` | run (after 7a) |
| 7c | Steps 8–10 on the chosen fit, plus its comparison with the confirmatory full fit | `code/` | `bash hipergator_tests/run_selected_fit.sh` (submits `run_hmm_pipeline.sbatch` with `SKIP_FIT=1`; outputs inside `fit_NN/`) | run (after 7b) |
| 8 | Drift test on the reported fit (every state) | `code/` | `hmm_v2/hmm_occupancy_drift.py` (in 7c) | run (in 7c) |
| 9 | Tables, graph metrics, bootstrap stability | `code/` | `hmm_v2/run4d…`, `run5…`, `run5B…` (in 7c) | run (in 7c) |
| 10 | Statistics and figures, only for what `inference_scope.csv` allows | `code/` | `hmm_py/hmm_dynamic_stats.py`, `dynamic_fc_plotting_RtoPythonmigration.py`, `brainnetwork_plotter_Rmigration_bin_transitivity.py`, `hmm_bootstrap_stability_plotter.py` (in 7c) | run (in 7c) |
| 10b | DCC cross-check against the published DCC toolbox (Lindquist) | `code/` | `sbatch hipergator_tests/run_dcc_crosscheck.sbatch` (once: clone the toolbox; see `hipergator_tests/README.md`) | done (4 Oct; matches the published estimator) |
| 10c | Bundle the results to send back | `code/` | `bash hipergator_tests/collect_results.sh` (picks up the chosen fit after 7b) | run (after 7c) |
| 11 | Supplement models | `code/` | the same job with other settings, e.g. the other passing cell, `sbatch --export=ALL,HMM_LOWRANK=25,HMM_K=2 hipergator_tests/run_hmm_pipeline.sbatch`, or the previous K, `--export=ALL,HMM_K=3` | optional |
| | **Window-free dynamic FC** (Supplement) | | | |
| 12 | DCC/GARCH and complex PCA vs own surrogates and replicas | `signals/` | `sbatch slurm/dfc_quicktest.sbatch` → `bash submit_dfc.sh` | done |
| 13 | Lag covariate | `signals/` | `python compute_lag_covariate.py --out lag_covariate.csv` | done |
| 14 | DFC figures and tables | `signals/` | `python plot_dfc_results.py --in_dir dfc_dcc_cpca --lag_covariate lag_covariate.csv` | rerun (Welch tests) |
| 14b | Supplement figures in natural units: each mouse vs its surrogates in raw units, and ρ(t) for one random NTG_Y mouse from the published DCC estimator | `code/` | `sbatch hipergator_tests/run_dcc_example.sbatch` (needs the Lindquist toolbox and the Optimization Toolbox) | run |
| 15 | Confound and transient checks | `signals/` | `python dfc_confound_checks.py --in_dir dfc_dcc_cpca --motion mouse_motion.csv` | done |
| 16 | Group tests on calibrated and transient measures | `signals/` | `python dfc_group_calibrated.py dfc_dcc_cpca` | rerun (after 14) |
| | **Positive control** (Supplement) | | | |
| 17 | Simulated cohorts through the same pipelines | `signals/` | `python sim_positive_controls.py --stage grid` → `bash submit_positive_controls.sh grid` → `python sim_positive_controls.py --stage final` → `bash submit_positive_controls.sh final` → `python summarize_positive_controls.py` (details in `positive_control/README_positive_control.md`) | done |

**Python steps outside the job scripts.** Run the long ones as 1-core jobs, e.g. step 1:

```bash
cd /blue/febo/share/Abeta_Age/rabies/static
sbatch -A febo --qos febo -c 1 --mem 8gb -t 08:00:00 -o static_%j.log --wrap \
  "module load conda; conda run --no-capture-output -p /blue/febo/$USER/envs/fc_static \
   python ../code/static_v2/full_analysis_abeta_mice_permutation_transitivity.py"
```

Short steps (3, 4, 13–16) can run in an interactive session (`srun -A febo --qos febo -c 1 --mem 8gb --pty bash`) with the same environment.

**Dependencies:**
- Step 7c reads step 6's confirmatory fit of the same model (`signals/hmm_confirm_lowrank/observed/gamma_lowrank_15_K4.mat`), for description only.
- Steps 8–10 read only the fit chosen in 7b.
- Step 14 reads steps 12–13.
- Step 16 reads steps 12 and 15.
- Step 17 reuses the step 6 and step 12 code.
- The static steps (1–3) are independent of everything else.

**Each HMM fit is new.** HMM-MAR reseeds from the clock at every initialisation, so a refit never reproduces an earlier fit exactly. The record is the saved model folder and its fit ID (`fit_id.txt`). Every table made from a fit carries that ID, and the scripts refuse to mix fits. On 4 Oct 2026 a refit reproduced one of the four states of the confirmatory run's full fit (`hmm_v2/CHANGES.md`, "Refits"). The reported fit is therefore chosen by the rule in `hmm_v2/FIT_SELECTION_RULE.md`, written before the ten fits of step 7a: the lowest final free energy among ten independent fits, with group tests only for the states that recur across fits. The 4 Oct refit's outputs (`signals/hmmmar_simple_k4_hmmpca15/`) are not used for the paper.

**The null and confirmatory runs (steps 5–6) used the previous `hmm_fit_raw_timeseries.m`.** That version is kept in `hmm_null/` so these steps can be repeated as run. The updated function (in `hmm_v2/`) is used only from step 7 on.

## 4. What to share, and what to leave out

- **Share:** `code/`, and the derived, de-identified matrices only (e.g. `merged.csv`, ROI time series if permitted, the result tables). Raw images and identifiers stay out.
- **Previous code.** `tests/legacy/` in each package holds verbatim copies of the previous code, used by the equivalence tests. Keep them, because they are the evidence for `CHANGES.md`.
- **Shared module.** `welch_tests_scipy.py` appears in `static_v2/`, `hmm_py/` and `dfc/` as identical copies. Keep them identical.
- **Paths.** The defaults point at `/blue/febo/share/Abeta_Age/rabies/signals`. Anyone running the code elsewhere sets `HMM_OUT_BASE`.
- **Remove before sharing:** `__pycache__/` folders and SLURM `logs/`.

## 5. Software versions to report

Record these from the test logs:
- MATLAB release;
- HMM-MAR commit;
- BCT release;
- Python, SciPy (≥ 1.16), statsmodels, networkx, bctpy 0.6.1.
