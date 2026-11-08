# HMM statistics and figures (Python)

These scripts read the outputs of the MATLAB steps in `hmm_v2/` for the chosen fit of one reported model: the primary (Gaussian HMM on 25 PCs, K = 3; the default, `hmm_v2/MODEL_PRIMARY.env`) or the secondary (HMM-PCA rank 15, K = 2; `hmm_v2/MODEL_SECONDARY.env`). They check that every table comes from that one fit. All group tests come from `welch_tests_scipy.py`. With K = 2, occupancy is tested once (state 2's FO is 1 minus state 1's; E50).

| Script | Reads | Writes |
|---|---|---|
| `hmm_dynamic_stats.py` | run4d `hmm_{switching_rates,maxFO,dwell_times}_all_K_all_PCA.csv` (primary) or `..._lowrank.csv` (secondary) | `<signals>/hmm_dynamics_stats_<tag>/` (tag `pca25_k3` or `hmmpca15_k2`) |
| `dynamic_fc_plotting_RtoPythonmigration.py` | the same tables | `<signals>/hmm_dynamics_figures_<tag>/` |
| `brainnetwork_plotter_Rmigration_bin_transitivity.py` | run5 `graph_metrics_per_subject_per_state_density15.csv` | `<model folder>/hmm_graph_figures/` |
| `hmm_bootstrap_stability_plotter.py` | run5B `bootstrap_stability_subject_state.csv` | `<model folder>/bootstrap_supplement/` |

After both reported models' pipelines, `hmm_model_agreement.py` applies the agreement rule (`hmm_null/MODEL_SELECTION_RULE.md` amendment 2) and writes `<signals>/hmm_model_agreement/` (`state_pairing.csv`, `agreement_contrasts.csv`, `agreement_counts.csv`, `agreement_report.txt`). Test: `python tests/test_model_agreement.py` (A1–A6).

Shared files:
- `hmm_model_selection.py`: which fit to read, and the fit-ID check.
- `welch_tests_scipy.py`: the group tests. It is identical to the copies in `static_v2/` and `dfc/`.

## Run (after the MATLAB pipeline)

From `code/`, with the `fc_static` environment (SciPy ≥ 1.16, statsmodels, python-docx). `<signals>` is `/blue/febo/share/Abeta_Age/rabies/signals` unless `HMM_OUT_BASE` says otherwise:

```bash
python hmm_py/hmm_dynamic_stats.py
python hmm_py/dynamic_fc_plotting_RtoPythonmigration.py
python hmm_py/brainnetwork_plotter_Rmigration_bin_transitivity.py
python hmm_py/hmm_bootstrap_stability_plotter.py
```

`hipergator_tests/run_hmm_pipeline.sbatch` runs all four after run5B, when `hmm_py/` is present; `hipergator_tests/run_selected_fit.sh primary|secondary` sets the model and `<signals>` (the chosen `fit_NN/`).

For the secondary model by hand, set all three variables: `export HMM_K=2 HMM_LOWRANK=15 HMM_PCA=0`. Unset ones come from `MODEL_PRIMARY.env`.

## Tests

```bash
cd hmm_py && python tests/test_hmm_python_stats.py
```

This compares the updated and previous scripts on simulated data (K = 3, the K the previous scripts were written for), runs every script for both reported models, and checks the fit-ID guard, the defaults, the fit-selection scope and the K = 2 occupancy fix. `CHANGES.md` lists every change and its evidence.

The ROI table that replaces `statistical_table.r` is `static_v2/roi_group_table.py`, with the static analysis.
