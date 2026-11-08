# Abeta mouse functional connectivity: code

This is the code for the static and dynamic FC analyses of the Abeta mouse cohort:
- 95 mice in 7 groups;
- 82 ROIs;
- spin-echo EPI at 11.1 T, TR 2 s, 570 volumes.

The paper is mouse-only (D31, 26 Oct 2026: the human Low/High Abeta comparison was dropped after review). The human scripts stay in the repository but are not part of the run order.

**HMM model (27 Oct 2026, D33).** The ROI time series are now the ROI means (E47, D29), and the model was chosen again on them by a rule fixed before the results (`hmm_null/MODEL_SELECTION_RULE.md`, D32). Two models passed their calibrated confirmatory tests, and both are reported:

- **Primary (main text):** Gaussian HMM on the first 25 principal components (`options.pca = 25`), K = 3. It was the screen candidate and passed step 6b (median c 0.048, Holm p 1.2 × 10⁻⁴; Hodges–Lehmann c 0.061, 95% CI 0.030–0.099; rank-biserial 0.77).

- **Secondary:** HMM-PCA (`options.lowrank = 15`), K = 2. It passed step 6 (median c 0.086, Holm p 0.044; Hodges–Lehmann c 0.081, 95% CI 0.018–0.147; rank-biserial 0.51).
- The Hodges–Lehmann intervals and rank-biserial values were computed after the decision (rerun of 27 Oct 2026, E51) and do not change it. The intervals are unadjusted; Holm p decides.

Rule item 6 would have put the secondary in the main text (larger median c). The PI chose the primary by precision instead, after the confirmation results and before any group result: the rule's amendment 1 records this deviation.

Each model has its own ten fits, reported fit and recurring states (`hmm_v2/FIT_SELECTION_RULE.md`, amendment 1; steps 7a–7c). Both models' pre-specified results are reported in full. The agreement between them (`hmm_null/MODEL_SELECTION_RULE.md` amendment 2, D35; applied by `hmm_py/hmm_model_agreement.py`, step 10a) is a descriptive comparison (amendment 3, D37, 28 Oct 2026, after the results). Each tested primary state is paired with the secondary state of highest pooled-FC r, and each planned contrast is classed by whether BH q < 0.05 in both models with the same sign.

The earlier result on the eigenvariate signals (HMM-PCA rank 15, K = 4, chosen after the confirmatory results, D17) is history: `hmm_v2/CHANGES.md`, "Reported model". Most of the other analyses explain the HMM result and go to the Supplement.

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
│   ├── hmm_confirm_lowrank/        confirmatory test, HMM-PCA 15/25 (hmm_null, run4c; step 6)
│   ├── hmm_confirm_candidate/      confirmation of the screen candidate, Gaussian PCA-25 (step 6b)
│   ├── hmm_k3_pca25_fits/          primary model: ten fits (fit_01 … fit_10), selection/, model.env;
│   │                               the reported fit_NN/ holds every later output (run4e, run5,
│   │                               run5B, state descriptors, run4d tables, hmm_py statistics and figures)
│   ├── hmm_k2_hmmpca15_fits/       secondary model: the same
│   ├── hmm_fits_{primary,secondary}_model.env   model recorded when its first fit started
│   ├── hmm_model_agreement/        agreement between the two models (step 10a)
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

Status as of 27 Oct 2026:
- **run:** not yet run with the current code;
- **rerun:** code changed since the last run;
- **done:** finished, and the result stands as run.

| # | Step | Run from | Command | Status |
|---|---|---|---|---|

| **0** | **Tests** (synthetic + existing mouse fit) | `code/` | `sbatch hipergator_tests/run_hmm_tests.sbatch` · `sbatch hipergator_tests/run_static_tests.sbatch` | rerun (new: fit-selection tests 4b, 4c and hmm_py S11–S13; the rest passed 4 Oct) |

| | **Input: ROI time series** | | | |
| 0b | `signal<N>.txt` with the lab script `organize_signals.m`, edited 24 Oct 2026 (E47, D29): `fslmeants` without `--eig`, i.e. the ROI mean, the same series RABIES's FC matrices (and `merged.csv`) are computed from. Writes `signals/hmm_signals_mean/` and checks every mouse against its RABIES FC matrix (`fc_check.csv`; expected < 1e-4). Then keep the old folder and swap: `cd signals; mv hmm_signals hmm_signals_eig; mv hmm_signals_mean hmm_signals` | `code/organize_signals/` | `module load fsl matlab` → `matlab -nodisplay -r "run('organize_signals.m'); exit"` | run (first; D29 adopted 24 Oct) |

| | **Static FC** (main text; independent of the HMM) | | | |
| 1 | Mouse static FC, graph metrics, NBS | `static/` | `python ../code/static_v2/full_analysis_abeta_mice_permutation_transitivity.py` (about 4 h; submit as a 1-core job) | rerun |
| 2 | ~~Human static FC (companion)~~ | | not in this paper (D31) | dropped |
| 3 | ROI nodal table (replaces `statistical_table.r`) | `static/` | `python ../code/static_v2/roi_group_table.py mice_NodePosStrength_PCorr.csv` (wide file: subject index, group, one column per ROI) | run |

| | **HMM model selection** (Supplement) | | | |
| 4 | Signal rank (Marchenko–Pastur and parallel analysis) | `signals/` | `python mp_signal_rank.py --dataset mouse hmm_signals 95 2` | done (ROI mean, 26 Oct) |
| 4b | Group signal rank: the same two thresholds on the pooled 95 mice, plus split-half overlap of the group subspace | `signals/` | `python mp_signal_rank_group.py --dataset mouse_SE11T hmm_signals 95 2` (minutes) | done (ROI mean, 26 Oct) |
| 5 | Split-half null across configurations (phase null; then IAAFT, which reuses the phase run's observed fits) | `signals/` | Steps 5 and 6, phase then IAAFT, chained by SLURM dependencies: `bash run_nulls_in_sequence.sh both` (progress in `logs/null_sequence.log`), then `python plot_hmm_splithalf_null.py` → `python select_screen_candidate.py --table hmm_raw_timeseries_null/figures/table_hmm_splithalf_null.csv --metadata hmm_raw_timeseries_null/metadata_aligned.csv --rule MODEL_SELECTION_RULE.md` (the candidate and its confirmation, by `hmm_null/MODEL_SELECTION_RULE.md`, D32; record the rule's SHA-256 first). By hand: `sbatch slurm/hmm_listconfigs.sbatch` → `bash submit_hmm_chunks.sh all` → when finished, `HMM_NULL=iaaft HMM_K_VALUES="2:4" bash submit_hmm_chunks.sh 1` → `sbatch slurm/hmm_collect.sbatch` → `python plot_hmm_splithalf_null.py` | done (ROI mean, 95 mice, 26 Oct; candidate Gaussian PCA-25) |
| 6 | Pre-specified confirmatory test (HMM-PCA 15 and 25, K = 2–4); rank 15, K = 2 passes (secondary model) | `signals/` | In the chain above (`both`), or alone: `bash run_nulls_in_sequence.sh confirm`. By hand: `bash submit_confirm_lowrank.sh` → when finished, `HMM_NULL=iaaft bash submit_confirm_lowrank.sh` (descriptive). Then `python analyze_confirm_lowrank.py --root hmm_confirm_lowrank` and `... --null iaaft` | done (ROI mean, 26 Oct; analysis rerun for E50/E51: `python analyze_confirm_lowrank.py --root hmm_confirm_lowrank`) |
| 6b | Calibrated confirmation of the step 5 screen candidate (`MODEL_SELECTION_RULE.md` item 4, D32): step 6's design for the candidate's model and K cells, read from `screen_candidate.txt`; seeds 20261026/20261027 | `signals/` | after the selector: `bash submit_confirm_candidate.sh` → `python analyze_confirm_lowrank.py --root hmm_confirm_candidate`; then optionally `HMM_NULL=iaaft bash submit_confirm_candidate.sh` (descriptive) | done (27 Oct: Gaussian PCA-25, K = 3 passes; primary model). Rerun the analysis for E50/E51: `python analyze_confirm_lowrank.py --root hmm_confirm_candidate` |

| | **Reported HMMs** (`hmm_v2/MODEL_PRIMARY.env`: Gaussian PCA-25, K = 3, main text; `hmm_v2/MODEL_SECONDARY.env`: HMM-PCA 15, K = 2) | | | |
| 7a | Ten independent fits per model (rule: `hmm_v2/FIT_SELECTION_RULE.md`, amendment 1 of 27 Oct 2026; do not edit it or the model files after this step) | `code/` | `sbatch hipergator_tests/run_hmm_fits_array.sbatch primary` and `sbatch hipergator_tests/run_hmm_fits_array.sbatch secondary` (each array runs at most 8 fits at once, 16 cores together) | done (27 Oct; 10 of 10 fits per model) |
| 7b | Choose each model's reported fit and the states to test (lowest free energy; states that recur across fits) | `code/` | `sbatch --dependency=afterany:<7a primary job ID> hipergator_tests/run_select_fit.sbatch primary`, the same with `secondary` → read `signals/hmm_k3_pca25_fits/selection/selection_report.txt` and `signals/hmm_k2_hmmpca15_fits/selection/selection_report.txt` | done (27 Oct; primary fit_02, states 1-3 and both whole-fit outcomes tested; secondary fit_04, states 1-2 tested) |
| 7c | Steps 8–10 on each chosen fit, plus its comparison with its confirmation's full fit | `code/` | `bash hipergator_tests/run_selected_fit.sh primary`, then `... secondary` (each submits `run_hmm_pipeline.sbatch` with `SKIP_FIT=1`; outputs inside `fit_NN/`) | done (27 Oct) |
| 8 | Drift test on each reported fit (every state; K = 2 counted once, E50) | `code/` | `hmm_v2/hmm_occupancy_drift.py` (in 7c) | done (no state drifts after Holm) |
| 9 | Tables, graph metrics, state descriptors (amplitude vs correlation pattern; descriptive), bootstrap stability | `code/` | `hmm_v2/run4d…`, `run5…`, `hmm_state_descriptors.m`, `run5B…` (in 7c) | done |
| 10 | Statistics and figures, only for what `inference_scope.csv` allows (K = 2 occupancy tested once, E50) | `code/` | `hmm_py/hmm_dynamic_stats.py`, `dynamic_fc_plotting_RtoPythonmigration.py`, `brainnetwork_plotter_Rmigration_bin_transitivity.py`, `hmm_bootstrap_stability_plotter.py` (in 7c) | done |
| 10a | Agreement between the two reported models (`hmm_null/MODEL_SELECTION_RULE.md` amendment 2, D35; descriptive since amendment 3, D37): each tested primary state paired with the secondary state of highest pooled-FC r (written before any group table is read); each planned contrast classed robust / opposite / primary only / secondary only / neither / not assessable | `code/` | after 7c for both models: `python hmm_py/hmm_model_agreement.py` → `signals/hmm_model_agreement/agreement_report.txt` | done (28 Oct) |
| 10b | DCC cross-check against the published DCC toolbox (Lindquist) | `code/` | `sbatch hipergator_tests/run_dcc_crosscheck.sbatch` (once: clone the toolbox; see `hipergator_tests/README.md`) | done (4 Oct; matches the published estimator) |
| 10c | Bundle the results to send back | `code/` | `bash hipergator_tests/collect_results.sh` (picks up both models' chosen fits after 7b) | done (28 Oct) |
| 11 | Other models (exploratory, labelled as such) | `code/` | the same job with other settings, e.g. `sbatch --export=ALL,HMM_K=4,HMM_LOWRANK=15,HMM_PCA=0 hipergator_tests/run_hmm_pipeline.sbatch` (single fit, no fit-selection rule, every state tested with a warning) | optional |

| | **Window-free dynamic FC** (Supplement) | | | |
| 12 | DCC/GARCH and complex PCA vs own surrogates and replicas | `signals/` | `sbatch slurm/dfc_quicktest.sbatch` → `bash submit_dfc.sh` | rerun (E47: ROI mean) |
| 13 | Lag covariate | `signals/` | `python compute_lag_covariate.py --out lag_covariate.csv` | rerun (E47: ROI mean) |
| 14 | DFC figures and tables | `signals/` | `python plot_dfc_results.py --in_dir dfc_dcc_cpca --lag_covariate lag_covariate.csv` | rerun (Welch tests) |
| 14b | Supplement figures in natural units: each mouse vs its surrogates in raw units, and ρ(t) for one random NTG_Y mouse from the published DCC estimator | `code/` | `sbatch hipergator_tests/run_dcc_example.sbatch` (needs the Lindquist toolbox and the Optimization Toolbox) | run |
| 15 | Confound and transient checks | `signals/` | `python dfc_confound_checks.py --in_dir dfc_dcc_cpca --motion mouse_motion.csv` | rerun (E47: ROI mean) |
| 15a | Slice order and RABIES slice-timing check (RABIES ran STC along Y; slices are along Z; `dfc/README_dfc.md`) | `rabies/` | `python signals/stc_axis_check.py --inputs "preprocess_outputs/bold_datasink/input_bold/*.nii.gz" --TR 2 --order 0,2,4,6,8,10,12,1,3,5,7,9,11,13 --out_dir stc_check_raw` | run |

| 15b | Slice timing vs complex-PCA lag (diagnostic; `dfc/README_dfc.md`) | `signals/` | `module load ants` → `python make_slice_maps.py --meta meta_data_with_bids.csv --rabies_preproc ../preprocess_outputs --atlas /blue/febo/share/civm_atlas/BrainNetCoord.nii --signals_dir hmm_signals --out_dir slice_maps --qc_n 0 --cleaned_glob "../confound_correction_outputs/confound_correction_datasink/cleaned_timeseries/**/*sub-{id}_*.nii*"` (any time after RABIES) → after step 12: `python slice_timing_lag_check.py --loadings dfc_dcc_cpca/cpca_loadings_roi.csv --pairs slice_maps/slice_timing_pairs.csv --fractions slice_maps/slice_fractions.csv --strata group --TR 2 --rabies_tpattern alt+z --order 0,2,4,6,8,10,12,1,3,5,7,9,11,13` | run |

| 15c | ROI signal homogeneity (Supplement; D30): share of each ROI's voxel variance explained by the ROI mean, the first-component ceiling, mean pairwise voxel r, r(mean, first eigenvariate); group tests; `FigS_roi_homogeneity` (`dfc/README_dfc.md`) | `signals/` | after step 0b and the swap: `sbatch -A febo --qos febo -c 1 --mem 16gb -t 03:00:00 -o homog_%j.log --wrap "module load conda; pip install --upgrade pip; pip install nibabel; conda run --no-capture-output -p /blue/febo/$USER/envs/fc_static python roi_signal_homogeneity.py --meta meta_data_with_bids.csv --atlas_cs slice_maps/atlas_commonspace.nii.gz --cleaned_glob '../confound_correction_outputs/confound_correction_datasink/cleaned_timeseries/**/*sub-{id}_*.nii*' --roi_labels roi_labels.csv --eig_dir hmm_signals_eig --signals_dir hmm_signals --out_dir roi_homogeneity"` | run (after 0b) |
| 16 | Group tests on calibrated and transient measures | `signals/` | `python dfc_group_calibrated.py dfc_dcc_cpca` | rerun (after 14) |

| | **Positive control** (Supplement) | | | |
| 17 | Simulated cohorts through the same pipelines | `signals/` | `python sim_positive_controls.py --stage grid` → `bash submit_positive_controls.sh grid` → `python sim_positive_controls.py --stage final` → `bash submit_positive_controls.sh final` → `python summarize_positive_controls.py` (details in `positive_control/README_positive_control.md`) | rerun (E47: ROI mean) |

**24 Oct 2026: the ROI time series (E47).** Every step that reads `hmm_signals` (4–17, including 15b) was run on the first eigenvariate of each ROI (`fslmeants --eig`), while static FC (`merged.csv`) uses the ROI mean. The ROI mean is adopted for every analysis (D29, 24 Oct 2026), so steps 4–17 are rerun after step 0b (statuses "rerun (E47)" above). **Rename the earlier result folders first; this is required, not optional (E48).** Steps 5, 6, 12 and 17 keep per-job checkpoints keyed on the settings, not on the data, so with the folders in place they reload the eigenvariate results (logs: "reused checkpoint" / "checkpoint reused") while the summary files get new dates; step 7a never refits an existing `fit_NN`. Renaming also keeps the eigenvariate results, which have to be reported (the change came after the confirmatory test, step 6). From `signals/`:

```bash
for d in hmm_raw_timeseries_null* hmm_confirm_lowrank hmm_k4_hmmpca15_fits dfc_dcc_cpca dfc_dcc_cpca_* \
         sim_positive_control dfc_sim_* hmm_confirm_sim_*; do
  case "$d" in *_eig) continue ;; esac        # safe to run twice
  [ -e "$d" ] && mv "$d" "${d}_eig" && echo "renamed $d"
done
```

Steps 4, 4b, 13–16 and 15b–15c recompute and overwrite; 13–16 read step 12's folder, and step 17 reads `hmm_signals/` and `dfc_dcc_cpca/`, so run them after step 12.

**Python steps outside the job scripts.** Run the long ones as 1-core jobs, e.g. step 1:

```bash
cd /blue/febo/share/Abeta_Age/rabies/static
sbatch -A febo --qos febo -c 1 --mem 8gb -t 08:00:00 -o static_%j.log --wrap \
  "module load conda; conda run --no-capture-output -p /blue/febo/$USER/envs/fc_static \
   python ../code/static_v2/full_analysis_abeta_mice_permutation_transitivity.py"
```

Short steps (3, 4, 13–16) can run in an interactive session (`srun -A febo --qos febo -c 1 --mem 8gb --pty bash`) with the same environment.

**Dependencies:**
- Step 7c reads each model's full fit from its confirmation, for description only: `signals/hmm_confirm_candidate/observed/gamma_gaussian_pca_25_K3.mat` (primary, step 6b) and `signals/hmm_confirm_lowrank/observed/gamma_lowrank_15_K2.mat` (secondary, step 6).
- Steps 8–10 read only the fit chosen in 7b.
- Step 14 reads steps 12–13.
- Steps 4–17 read the ROI time series from step 0b (`signals/hmm_signals`).
- Step 15b reads step 12 and RABIES's preprocessing outputs (transforms, `input_bold`, `commonspace_bold`).
- Step 16 reads steps 12 and 15.
- Step 17 reuses the step 6 and step 12 code.
- The static steps (1–3) are independent of everything else.

**Each HMM fit is new.** HMM-MAR reseeds from the clock at every initialisation, so a refit never reproduces an earlier fit exactly. The record is the saved model folder and its fit ID (`fit_id.txt`). Every table made from a fit carries that ID, and the scripts refuse to mix fits. On 4 Oct 2026 a refit reproduced one of the four states of the confirmatory run's full fit (`hmm_v2/CHANGES.md`, "Refits"). Each reported fit is therefore chosen by the rule in `hmm_v2/FIT_SELECTION_RULE.md`, written before the ten fits of step 7a: the lowest final free energy among ten independent fits, with group tests only for the states that recur across fits. Amendment 1 (27 Oct 2026) applies it to the two models on the ROI-mean signals. Fits made on the eigenvariate signals (`signals/hmmmar_simple_k4_hmmpca15/`, `hmm_k4_hmmpca15_fits_eig/`) are not used for the paper's HMM results.

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