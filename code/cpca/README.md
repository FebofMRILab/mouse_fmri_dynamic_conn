# Complex PCA (cPCA): package for the paper

Packaged 11 November 2026. This package holds the scripts, result tables and figures behind
the complex PCA part of the dynamic functional connectivity paper (95 scans from 78 mice, 7 groups,
82 ROIs, TR = 2 s).

Complex PCA is applied as published by Bolt et al. (2022), in the same way as the HMM
analysis follows the HMM-MAR toolbox and DCC-GARCH follows Lindquist's toolbox. Groups
are compared the way Bolt et al. compared their independent sample: each group gets its
own fit. Propagation is measured with their traveling index and tested by permutation.

## Quick start: redraw the figures

```bash
bash make_figures.sh
```

This needs Python 3 with numpy, pandas, scipy, matplotlib and seaborn. It redraws
Figures 7, 8 and 9 into `figures/` from the tables in `results/`, in about 20 seconds.
On 11 November 2026 its output was checked against the figures delivered for the paper
and was pixel-identical.

## Figures

| Paper | File in `figures/` | Script | Input |
|---|---|---|---|
| Figure 7 | `Fig_cpca_patterns` (.png, .tiff at 600 dpi, .pdf) | `plot_cpca_patterns.py` | `results/cpca_group_lags/` |
| Supplementary Figure S6 | `Fig_cpca_brain_maps` (.png, .tif at 600 dpi) | `cpca_brainnet_files.py`, then `plot_cpca_brain_maps.py` | `results/cpca_loadings_roi.csv`, `atlas/` |
| Figure 8 | `Fig_cpca_group_phase_maps` (.png, .tif at 600 dpi) | `plot_cpca_group_phase_maps.py` | `results/cpca_group_lags/cpca_group_loadings.csv`, `atlas/` |
| Optional | `Fig_cpca_traveling_index.png` (Figure 7's right-hand panels on their own) | `plot_cpca_traveling_index.py` | `results/cpca_group_lags/` |
| Figures 6 and 9, Supplementary Figures S5 and S7 (DCC/GARCH; complex PCA by group) | not in this package | `../dfc/plot_dfc_results.py` | output folder of `../dfc/run_dcc_cpca_null.m` |

What each figure shows:

- **Figure 7:** one row per component. On the left are the all-mouse loadings, signed
  by phase; on the right is each group's traveling index. Once
  `cpca_ti_bootstrap_summary.csv` is in `results/cpca_group_lags/`, each group's dot gets
  a bar showing its 95% bootstrap interval; without that file the figure is drawn
  without bars, as delivered.
- **Supplementary Figure S6:** the three components on a dorsal view of the brain, with the largest
  regions named.
- **Figure 8:** the phase delay maps of each group fit side by side, in the manner of
  Bolt et al. Supplementary Fig. 1. It is drawn at its printed size (7.2 in wide).

## Layout

```
make_figures.sh        redraws Figures 7-9 from results/
python/                analysis and figure scripts (table below)
(../dfc/)              all-mouse complex PCA, run_dcc_cpca_null.m (step 1 of the pipeline)
results/
  cpca_loadings_roi.csv          all-mouse loadings, run_dcc_cpca_null.m (28 Oct 2026 run)
  cpca_group_lags/               cpca_group_lags.py outputs (11 Nov 2026 run, 1,000 permutations)
    cpca_group_loadings.csv        magnitude and phase per ROI, all mice and each group fit
    cpca_group_traveling_index.csv traveling index per fit and component
    cpca_lag_contrasts.csv         8 planned contrasts x 3 components: dTI, p, BH q
    cpca_ti_bootstrap_summary.csv  95% bootstrap interval per group and component
                                   (add after running cpca_ti_bootstrap.py; not here yet)
  cpca_brainnet/                 BrainNet node files written by make_figures.sh
atlas/
  brainnet_node_coords_highres.csv   ROI coordinates and names (CIVM atlas, atlas order)
  roi_labels.csv                     region, label (made from the coordinates file)
  civm.g                             CIVM brain surface (BYU mesh used in BrainNet Viewer); not included, see below
figures/               the figures as delivered (make_figures.sh rewrites them)
```

## Scripts

| File | Step | What it does | Code source |
|---|---|---|---|
| `../dfc/run_dcc_cpca_null.m` | 1 | Runs complex PCA on all 95 scans, with the DCC/GARCH tests in the same run. Writes the loadings (`cpca_loadings_roi.csv`), per-scan measures (`dfc_subject_metrics.csv`) and the cycle tables. | Ours; the method is Bolt et al. (2022) |
| `../dfc/dfc_cpca_fit.m` | 1 | Group complex PCA: eigendecomposition of the pooled analytic-signal covariance. Each component is rotated so its largest ROI has phase 0. | Ours; the method is Bolt et al. (2022) |
| `../dfc/dfc_cpca_subject_stats.m` | 1 | Per-mouse variance explained and frequency, computed on the group loadings (time courses = Z * V). | Ours |
| `../dfc/dfc_analytic_signal.m` | 1 | Analytic signal via FFT; the same result as `hilbert()` without the Signal Processing Toolbox. | Standard method |
| `../dfc/dfc_dcc_subject.m` | 1 | AR-GARCH(1,1)-DCC(1,1) for one mouse (DCC/GARCH part of the run). | Ours; the method is Engle (2002) and Lindquist et al. (2014) |
| `../dfc/hmm_phase_randomise.m` | 1 | Multivariate phase-randomised surrogate: the same random phase for every ROI. | Standard method (Prichard and Theiler, 1994) |
| `../dfc/hmm_iaaft_surrogate.m` | 1 | Multivariate IAAFT surrogate. | Standard method (Schreiber and Schmitz, 2000) |
| `../dfc/dfc_stationary_replica.m` | 1 | Stationary Gaussian replica of a mouse's data (tapered, zero-padded, phase-randomised, middle cut), used for the supplementary bias check and the grey band of Figure 6. | Ours |
| `../dfc/plot_dfc_results.py` | 2 | Figures and tables from step 1, including Figure 6 (`Fig_dfc_group_z`) and its raw-unit supplement (`FigS_dfc_group_raw`). Its cPCA figures are `Fig_cpca_propagation_roi`, `Fig_cpca_static_by_group`, `FigS_cpca_roi_lags`, `FigS_cpca_propagation_pca` and `FigS_pca_cpca_checks`. | Ours |
| `python/cpca_brainnet_files.py` | 3 | Writes BrainNet Viewer node files (`cpca_C1..3.node`) from the all-mouse loadings. Colours: red = phase near 0, blue = phase near pi, grey = weak ROI. Size = abs(loading). | Ours (display only) |
| `python/plot_cpca_brain_maps.py` | 4 | Supplementary Figure S6: dorsal glass-brain maps drawn from the node files. | Ours (display only) |
| `python/cpca_group_lags.py` | 5 | Fits each group separately with Bolt's own code and matches the components to the all-mouse fit. Computes the traveling index and tests the 8 planned contrasts by permutation with refits. | Bolt's `complex_pca`, called unchanged; matching and the permutation test are ours |
| `python/cpca_ti_bootstrap.py` | 5b | 95% bootstrap interval of each group's traveling index: resamples the group's mice with replacement (1,000 times), refits each resample with the same code as step 5 and takes the 2.5th and 97.5th percentiles. | Bolt's `complex_pca` through step 5's functions; the bootstrap is standard (Efron and Tibshirani, 1993) |
| `python/plot_cpca_patterns.py` | 6 | Figure 7. | Ours (display only) |
| `python/plot_cpca_traveling_index.py` | 6 | Optional: Figure 7's traveling-index panels on their own. | Ours (display only) |
| `python/plot_cpca_group_phase_maps.py` | 7 | Figure 8. | Ours (display only) |
| `../dfc/welch_tests_scipy.py` | 2 | The study's Welch tests (SciPy 1.16 or later), imported by `plot_dfc_results.py`. | Ours, on SciPy and statsmodels |

Steps 3, 4, 6 and 7 import nothing else from the project. Figure 7's loadings are the
'ALL' rows of `cpca_group_loadings.csv`, which come from Bolt's Python code run on all 95
mice. Supplementary Figure S6 uses the MATLAB loadings from step 1. The two fits agree (see the checks
below).

## Full pipeline (from the raw signals)

```bash
# 1. All-mouse complex PCA (+ DCC/GARCH), MATLAB; writes dfc_dcc_cpca/
matlab -batch run_dcc_cpca_null        # or split over a SLURM array (DFC_N_CHUNKS)

# 2. Figures and tables from step 1 (Figure 6 and the cPCA supplementary figures)
python ../dfc/plot_dfc_results.py --in_dir dfc_dcc_cpca --roi_labels atlas/roi_labels.csv

# 3-4. Supplementary Figure S6
python python/cpca_brainnet_files.py --loadings dfc_dcc_cpca/cpca_loadings_roi.csv \
    --node_template atlas/brainnet_node_coords_highres.csv --out_dir dfc_dcc_cpca/cpca_brainnet
python python/plot_cpca_brain_maps.py --mesh atlas/civm.g --bn_dir dfc_dcc_cpca/cpca_brainnet \
    --out Fig_cpca_brain_maps.tif --dpi 600

# 5. Per-group fits and the traveling-index permutation test
#    (about 16,000 fits for 1,000 permutations; use --n_jobs in a SLURM job)
python python/cpca_group_lags.py --cpca_dir complex_pca --data_dir hmm_signals \
    --groups dfc_dcc_cpca/dfc_subject_metrics.csv --roi_labels atlas/roi_labels.csv \
    --n_perm 1000 --n_jobs 16 --out_dir dfc_dcc_cpca/cpca_group_lags

# 5b. Bootstrap intervals of each group's traveling index
#     (7 groups x 1,000 resamples = 7,000 fits; imports cpca_group_lags.py from the same folder)
python python/cpca_ti_bootstrap.py --cpca_dir complex_pca --data_dir hmm_signals \
    --groups dfc_dcc_cpca/dfc_subject_metrics.csv \
    --n_boot 1000 --n_jobs 16 --out_dir dfc_dcc_cpca/cpca_group_lags

# 6. Figure 7 (draws the bootstrap bars when step 5b's summary is in --in_dir)
python python/plot_cpca_patterns.py --in_dir dfc_dcc_cpca/cpca_group_lags --formats pdf png tiff

# 7. Figure 8
python python/plot_cpca_group_phase_maps.py \
    --loadings dfc_dcc_cpca/cpca_group_lags/cpca_group_loadings.csv \
    --mesh atlas/civm.g --node_template atlas/brainnet_node_coords_highres.csv \
    --out Fig_cpca_group_phase_maps.tif --dpi 600
```

Steps 3-4 depend only on step 1, step 6 on steps 5 and 5b, and step 7 only on step 5.

### Needed for the full pipeline but not in this package

- **Bolt's code:** a clone of `github.com/tsb46/complex_pca`. The scripts were tested
  against commit `d5ec42ca6dfb2b43c93d124d535f0410439d9d7b`; that repository needs fbpca,
  xmca, nibabel and scipy.
- **MATLAB:** base MATLAB is enough (no toolboxes); the code also runs in Octave 7 or
  later. Every function `run_dcc_cpca_null.m` calls is in `../dfc/`.
- **The signals:**
  - one file per mouse (`hmm_signals/signal{N}.txt`, volumes x ROIs);
  - `meta_data.csv` (subject_id, group).
- **Python packages:** numpy, pandas, scipy, matplotlib, seaborn and statsmodels, plus
  python-docx for the tables.


## What the paper reports

- **All-mouse components (step 1):** loadings (magnitude and phase), variance explained,
  frequency, and the component brain maps. These are Bolt's outputs as they come.
- **Group comparison of propagation (step 5, Figure 7):** the traveling index of each
  group fit, with the difference tested for the 8 planned contrasts.
  - Each test permutes mice between the two groups, keeps the group sizes and refits both
    groups (1,000 permutations). p = (1 + b)/(k + 1).
  - BH correction is applied across the 24 tests (8 contrasts x 3 components).
  - Each group's index is a single value from one fit on its pooled mice. Its 95%
    bootstrap interval (step 5b) shows how much it would move with a different sample
    of mice of the same size. Group differences are judged by the permutation test, not
    by whether intervals overlap.
  - With 5 to 13 mice per group the intervals are approximate. Noise raises the index,
    and resamples repeat mice, so an interval can sit partly above its dot; the summary
    table gives the bootstrap median so this can be seen.
- **Group phase delay maps (step 7, Figure 8):** a descriptive display with no test.

The traveling index (Bolt et al., 2022, after Feeny, 2008) is the smaller singular value
of the ROI x 2 matrix [Re(v), Im(v)] divided by the larger: 0 is a standing pattern and 1
a traveling one.

Result of the 11 November 2026 run:

| Traveling index | Component 1 | Component 2 | Component 3 |
|---|---|---|---|
| All mice | 0.04 | 0.08 | 0.11 |
| Group fits | 0.06–0.14 | 0.15–0.47 | 0.19–0.42 |

- Component 1 matched in every group with abs(cosine) of 0.979 or more.
- No contrast differed: all q ≥ 0.84; the smallest p was 0.061 (NTG_Y vs NTG_A,
  component 2).
- Components 2 and 3 match less closely in small groups (abs(cosine) 0.69-0.96), so
  read them with caution.

## Computed but not reported

These outputs stay in the tables for the record, but the paper does not use them.

- **`cpca_group_lags.py`:**
  - the G statistic (loading-weighted mean absolute phase difference), with its `p_perm`
    and `q_BH` columns in `cpca_lag_contrasts.csv`;
  - the per-ROI phase tests (`cpca_lag_contrasts_roi.csv`);
  - the polar diagnostic `FigS_cpca_lags_by_group`.
- **`run_dcc_cpca_null.m`:** envelope CV (`cpca_*_env_cv`) and the lag index
  (`cpca_*_lag_index`). Both were removed from the paper because they are not published
  measures; `plot_dfc_results.py` leaves them out (`EXCLUDED_MEASURES`).
- **`plot_cpca_brain_maps.py`:** it can also draw group-contrast maps from `cpca_mag_*`
  node files. The paper pipeline does not make those files, so Supplementary Figure S6 shows only the
  three components.

Kept outside this package:

- `slice_timing_lag_check.py` is in `../qc/` (quality control; not reported in the paper).

## Notes for reviewers

**A correction to Bolt's published code that does not change our results.** At commit
d5ec42c, `cpca.pca` computes component time courses as
`pc_scores = input_data @ Va.T`. For complex data this should be
`input_data @ Va.conj().T`, which equals U * s.

- The loadings, phases and variance explained do not use that line and are unaffected.
- Our scripts take time courses as U * s (Python) or Z * V (MATLAB), and use them only for
  the component frequency.

**Checks run on the real data:**

- The all-mouse fit from Bolt's Python code (`cpca_group_lags.py`) matches the MATLAB
  loadings: abs(cosine) is 1.0000, 0.9999 and 0.9994 for components 1-3.
- The traveling index recomputed from `cpca_group_loadings.csv` equals
  `cpca_group_traveling_index.csv` (difference below 1e-15).

**Synthetic checks:**

- The header of `cpca_group_lags.py` describes the synthetic test of its permutation
  procedure.
- `cpca_ti_bootstrap.py` was run on 95 synthetic mice in the study's group sizes. Each
  group's own-fit index equalled `cpca_group_traveling_index.csv` exactly, and the
  resamples were identical with 1 or 4 worker processes.

## References

- Efron B, Tibshirani RJ (1993). An Introduction to the Bootstrap. Chapman & Hall, New York.
- Engle R (2002). Dynamic conditional correlation. J Bus Econ Stat 20:339-350.
- Lindquist MA, Xu Y, Nebel MB, Caffo BS (2014). Evaluating dynamic bivariate correlations in resting-state fMRI. NeuroImage 101:531-546.
- Prichard D, Theiler J (1994). Generating surrogate data for time series with several simultaneously measured variables. Phys Rev Lett 73:951-954.
- Schreiber T, Schmitz A (2000). Surrogate time series. Physica D 142:346-382.
- Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate. J R Stat Soc B 57:289-300.
- Bolt T, Nomi JS, Bzdok D, Salas JA, Chang C, Yeo BTT, Uddin LQ, Keilholz SD (2022). A parsimonious description of global functional brain organization in three spatiotemporal patterns. Nat Neurosci 25:1093-1103. doi:10.1038/s41593-022-01118-1
- Feeny BF (2008). A complex orthogonal decomposition for wave motion analysis. J Sound Vib 310:77-90.
- Nichols TE, Holmes AP (2002). Nonparametric permutation tests for functional neuroimaging. Hum Brain Mapp 15:1-25.
- Phipson B, Smyth GK (2010). Permutation p-values should never be zero. Stat Appl Genet Mol Biol 9:39.
- Xia M, Wang J, He Y (2013). BrainNet Viewer: a network visualization tool for human brain connectomics. PLoS One 8:e68910.

## Brain mesh

The brain surface `atlas/civm.g` is not included in this repository. It is a surface mesh in BrainNet Viewer's `.g` format, made in ITK-SNAP from the brain mask of the mouse template. That template matches the DSURQE atlas distributed with RABIES. To make your own, export a surface mesh from the template's brain mask in ITK-SNAP, convert it to `.g`, and place it in `atlas/` before running `make_figures.sh`. Supplementary Figure S6 and Figure 8 are drawn on it; Figure 7 and the traveling-index figure do not need it.
