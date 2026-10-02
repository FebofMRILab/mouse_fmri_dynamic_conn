# HMM statistics and figures (Python)

These scripts read the outputs of the MATLAB steps in `hmm_v2/` for the reported fit (HMM-PCA rank 15, K = 4). They check that every table comes from that one fit. All group tests come from `welch_tests_scipy.py`.

| Script | Reads | Writes |
|---|---|---|
| `hmm_dynamic_stats.py` | run4d `hmm_{switching_rates,maxFO,dwell_times}_all_K_all_lowrank.csv` | `<signals>/hmm_dynamics_stats_hmmpca15_k4/` |
| `dynamic_fc_plotting_RtoPythonmigration.py` | the same tables | `<signals>/hmm_dynamics_figures_hmmpca15_k4/` |
| `brainnetwork_plotter_Rmigration_bin_transitivity.py` | run5 `graph_metrics_per_subject_per_state_density15.csv` | `<model folder>/hmm_graph_figures/` |
| `hmm_bootstrap_stability_plotter.py` | run5B `bootstrap_stability_subject_state.csv` | `<model folder>/bootstrap_supplement/` |

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

`hipergator_tests/run_hmm_pipeline.sbatch` runs all four after run5B, when `hmm_py/` is present.

For a Supplement model, set the same variables as for MATLAB, e.g. `export HMM_LOWRANK=25`. For PCA before the HMM, use `export HMM_LOWRANK=0 HMM_PCA=15`.

## Tests

```bash
cd hmm_py && python tests/test_hmm_python_stats.py
```

This compares the updated and previous scripts on simulated data (K = 3, the K the previous scripts were written for), runs every script for K = 4, and checks the fit-ID guard and the defaults. `CHANGES.md` lists every change and its evidence.

The ROI table that replaces `statistical_table.r` is `static_v2/roi_group_table.py`, with the static analysis.
