# code/

The packages used in the paper. The run order, requirements and data layout are in the repository's top-level `README.md`.

| Package | Contents | Paper |
|---|---|---|
| `organize_signals/` | ROI-mean time series per mouse from RABIES outputs | input to all steps |
| `static_v2/` | Static FC: edgewise Welch tests, graph metrics, NBS (bctpy with Welch t), ROI table, BrainNet files | Figs 1–3 |
| `hmm_null/` | Signal dimensionality and HMM model selection (split-half reproducibility against phase and IAAFT surrogates) | Supp Figs S1–S2 |
| `hmm_v2/` | Reported HMM: fits, choice of the reported fit, state FC, graph metrics, block-bootstrap stability (MATLAB) | Fig 5; Supp Fig S3 |
| `hmm_py/` | HMM statistics and figures (Python) | Figs 4–5 |
| `dfc/` | DCC-GARCH and complex PCA (all mice) against surrogates and stationary replicas; figures and group tests | Figs 6, 9; Supp Figs S4, S5, S7 |
| `cpca/` | Group complex PCA and traveling index with the published code of Bolt et al. (2022); complex-PCA figures | Figs 7–8; Supp Fig S6 |
| `rois/` | Helpers for BrainNet Viewer maps | figures |
| `hipergator_tests/` | SLURM jobs (HiPerGator), environment setup and the HMM pipeline job | — |
| `qc/` | Quality-control checks, not needed to reproduce the paper | — |
| `positive_control/` | Simulated cohorts with known mechanisms. Not reported in the paper | — |

Each package has a `README.md` (how to run it) and, where published code was changed, a `CHANGES.md` (each change, with its evidence). Some package READMEs keep their development history. Where they mention the second HMM, the confirmatory tests or the positive control, those analyses are not reported in the paper; see the top-level README.
