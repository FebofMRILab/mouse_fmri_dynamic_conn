# mouse_fmri_dynamic_conn

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24431472.svg)](https://doi.org/10.5281/zenodo.24431472)

Code for static and dynamic functional connectivity (FC) analysis of mouse resting-state fMRI. It accompanies:

> Grudny MM, Rodriguez N, et al. *Multi-method assessment of functional connectivity in TgCRND8 and 5XFAD mice reveals robust static and hidden Markov state effects of amyloidosis.* (eLife, submitted; preprint doi:10.64898/2026.04.26.720907)

**The study's data:**
- 95 mice in 7 groups (non-transgenic, TgCRND8 and 5XFAD; young, middle-aged and aged)
- 82 ROIs
- spin-echo EPI at 11.1 T, TR 2 s, 570 volumes after preprocessing
- the raw BIDS data are on OpenNeuro (ds008951)

**Analyses:**
- static FC with graph theory and the network-based statistic (NBS)
- hidden Markov model (HMM) states with HMM-MAR
- window-free dynamic FC: DCC-GARCH, and complex PCA for propagation

Each analysis is a self-contained package under `code/`. The steps below run them in the order used for the paper.

## Example data

`code/static_v2/example_files/` holds the static-FC inputs used in the paper for all 95 mice:
- `merged.csv`: 95 stacked 82 × 82 FC matrices from the RABIES-preprocessed OpenNeuro data
- `roi_names_order.csv`
- `subject_group_order_mappings.csv`: group of each mouse

To run the static analysis (step 4) on them:

```bash
mkdir -p static && cp code/static_v2/example_files/*.csv static/
cd static && python ../code/static_v2/full_analysis_abeta_mice_permutation_transitivity.py   # outputs in fc_analysis_mice/
```

The full run uses 10,000 permutations and takes about 4 h on one core.

`example_data/hmm_signals.zip` holds the ROI-mean time series of the same 95 mice:
- `signal1.txt` … `signal95.txt` (570 volumes × 82 ROIs each)
- `meta_data.csv` (subject_id, group)
- `fc_check.csv`: per mouse, the largest difference between the FC of these series and RABIES's FC matrix (about 2 × 10⁻⁷ everywhere)

These are the inputs to steps 6–13. Unzip them into your `signals/` folder (`unzip example_data/hmm_signals.zip -d <study>/signals`). They can also be regenerated from the OpenNeuro data with steps 1–2.

## Requirements

| What | Version used | Needed for |
|---|---|---|
| Python ≥ 3.10 with numpy, scipy (≥ 1.16), pandas, statsmodels, networkx, bctpy 0.6.1, matplotlib, seaborn, python-docx | SciPy 1.17.1, statsmodels 0.15.0, NetworkX 3.6.1 | all Python steps (`bash code/hipergator_tests/setup_python_env.sh` builds the environment) |
| MATLAB (base, no toolboxes; most functions also run in Octave ≥ 7) | R2026a | HMM, DCC-GARCH, complex PCA (all mice) |
| [HMM-MAR](https://github.com/OHBA-analysis/HMM-MAR) | see `code/hmm_v2/README.md` | HMM steps |
| [Brain Connectivity Toolbox](https://sites.google.com/site/bctnet/) (MATLAB) | 2019-03-03 | HMM state networks |
| [complex_pca](https://github.com/tsb46/complex_pca) (Bolt et al., 2022) | commit `d5ec42c`, used unchanged | group complex PCA (step 12) |
| FSL (`fslmeants`) | 6.0 | ROI time series (step 2) |
| Optional: [RABIES](https://github.com/CoBrALab/RABIES) | 0.6.1 | preprocessing (step 1) |
| Optional: the DCC toolbox of Lindquist et al. (2014) | — | example ρ(t) (step 11) and the cross-check against the published estimator |

## Folder layout

The code and the data sit in sibling folders:

```
<study>/
├── code/       this repository's code/ folder
├── signals/    hmm_signals/signal<N>.txt (volumes × ROIs) and meta_data.csv; HMM and dynamic-FC outputs
└── static/     merged.csv, roi_names_order.csv, subject_group_order_mappings.csv; static-FC outputs
```

- Set `export HMM_OUT_BASE=/path/to/<study>/signals`. The default is the HiPerGator path used for the paper.
- Run `bash code/hipergator_tests/link_into_signals.sh` once. The `hmm_null` and `dfc` scripts run from inside `signals/`, and this links them there.
- The `.sbatch` files are SLURM jobs written for HiPerGator (University of Florida). Change the account, partition and module lines for your cluster.

## Steps used in the paper

| Step | What | Run from | Command | Paper |
|---|---|---|---|---|
| 0 | Tests on synthetic data | `code/` | `sbatch hipergator_tests/run_static_tests.sbatch`; `sbatch hipergator_tests/run_hmm_tests.sbatch` | — |
| 1 | Preprocessing (optional): RABIES, with slice timing along the slice axis (Z, interleaved `alt+z`) | — | `optional_rabies_run/run_rabies_mice.sh` (see its README) | Methods, Image processing |
| 2 | ROI-mean time series per mouse (needs `meta_data.csv`) | RABIES output folder | `code/organize_signals/organize_signals.m` | Methods, Static FC |
| 3 | Stack the 95 FC matrices | `code/static_v2/build_matrices/` | `build_merged_matrices.m` | — |
| 4 | Static FC: edgewise Welch tests, graph metrics, NBS | `static/` | `python ../code/static_v2/full_analysis_abeta_mice_permutation_transitivity.py` (about 4 h; run as a 1-core job) | Figs 1–3 |
| 5 | Nodal tables: node strength, clustering and nodal efficiency at 15% density | `static/` | `python ../code/static_v2/roi_group_table.py` (reads `fc_analysis_mice/node_metrics_by_density.csv` from step 4) | Results, nodal metrics |
| 6 | Signal dimensionality (parallel analysis, Marchenko–Pastur) | `signals/` | `python mp_signal_rank.py --dataset mouse hmm_signals 95 2`; `python mp_signal_rank_group.py --dataset mouse_SE11T hmm_signals 95 2` | Supp Fig S1 |
| 7 | HMM model selection: split-half reproducibility against phase and IAAFT surrogates | `signals/` | `IAAFT_MODELS="all" IAAFT_K="2:10" bash run_nulls_in_sequence.sh split`, then `python plot_hmm_splithalf_null.py` and `python select_screen_candidate.py` | Supp Fig S2 |
| 8 | Reported HMM (Gaussian HMM on 25 PCs, K = 3): ten fits, choice of the reported fit, then metrics, state networks, bootstrap stability, statistics and figures | `code/` | `sbatch hipergator_tests/run_hmm_fits_array.sbatch primary`, then `sbatch hipergator_tests/run_select_fit.sbatch primary`, then `bash hipergator_tests/run_selected_fit.sh primary` | Figs 4–5; Supp Fig S3 |
| 9 | DCC-GARCH and complex PCA (all mice) against each mouse's surrogates and stationary replicas | `signals/` | `sbatch slurm/dfc_quicktest.sbatch`, then `bash submit_dfc.sh` | — |
| 10 | Dynamic-FC figures and group tests | `signals/` | `python plot_dfc_results.py --in_dir dfc_dcc_cpca` | Figs 6, 9; Supp Figs S5, S7 |
| 11 | Example ρ(t) with the published DCC estimator | `code/` | `sbatch hipergator_tests/run_dcc_example.sbatch` | Supp Fig S4 |
| 12 | Group complex PCA and traveling index (published code, unchanged; permutation tests with refits) | `<study>/` | `python code/cpca/python/cpca_group_lags.py --cpca_dir complex_pca --data_dir signals/hmm_signals --groups signals/dfc_dcc_cpca/dfc_subject_metrics.csv --roi_labels code/cpca/atlas/roi_labels.csv --n_perm 1000 --n_jobs 16 --out_dir signals/dfc_dcc_cpca/cpca_group_lags` | Figs 7–8 |
| 13 | Complex PCA figures | `code/cpca/` | `bash make_figures.sh` (needs the brain mesh in `atlas/`, see `code/cpca/README.md`) | Figs 7–8; Supp Fig S6 |

The fuller run order is in `code/hipergator_tests/README.md`, and each package's `README.md` gives its options. Each package's `CHANGES.md` lists every change to published code, with its evidence.

## Checks against published code

`code/dfc/tests/equivalence_published/` compares our implementations with the published ones:
- DCC-GARCH against the toolbox of Lindquist et al. (2014): DCC parameters within 0.0006
- complex PCA against Bolt et al. (2022): |cosine| ≥ 0.999 for components 1–3
- phase randomisation against the CBIG code (Liégeois et al., 2017): identical

`tests/` folders hold the unit and synthetic tests. `tests/legacy/` keeps earlier lab versions as reference implementations for those tests.

## Quality control (`code/qc/`)

These checks are not needed to reproduce the paper's results. See `code/qc/README.md`.
- parcellation and ROI-signal checks
- slice-timing axis and order check (`stc_axis_check.py`)
- slice timing against complex-PCA lag
- ROI signal homogeneity
- motion and subject-matrix QC

## Analyses not reported in the paper

These were run during development as internal tests and validations, and are not part of the paper. They stay inside their packages because the pipeline scripts share helper functions with them.

| Analysis | Files |
|---|---|
| Pre-specified confirmatory HMM tests (calibrated against stationary replicas) | `code/hmm_null/run4c_confirm_*.m`, `analyze_confirm_lowrank.py`, `submit_confirm_*.sh`, `slurm/confirm_*` |
| Second HMM (HMM-PCA rank 15, K = 2) and agreement between models | `code/hmm_v2/MODEL_SECONDARY.env`, `code/hmm_py/hmm_model_agreement.py` |
| State-occupancy drift and state descriptors | `code/hmm_v2/hmm_occupancy_drift.py`, `hmm_state_descriptors*.m` (run inside the step 8 pipeline) |
| Lag covariate, confound checks, calibrated group tests | `code/dfc/compute_lag_covariate.py`, `dfc_confound_checks.py`, `dfc_group_calibrated.py`, `plot_dfc_raw_units.py` |
| Positive control (simulated cohorts with known mechanisms) | `code/positive_control/` |

The confirmatory test and the drift test informed the choice of K = 3.

Code comments cite IDs such as D33 or E47. These refer to the authors' provenance record (decisions D, errors E, components R), which is available from the corresponding author on request.

## Citation and licence

Please cite the paper and this code (`CITATION.cff`). The code is archived at Zenodo:
- v1.0.0, the version used for the paper: [doi:10.5281/zenodo.24431472](https://doi.org/10.5281/zenodo.24431472)
- all versions: [doi:10.5281/zenodo.24431471](https://doi.org/10.5281/zenodo.24431471)

Licence: CC0 1.0 (`LICENSE`).
