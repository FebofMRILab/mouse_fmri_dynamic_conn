# qc/: quality-control checks

These checks are not needed to reproduce the paper's results. They were used to check the inputs: parcellation, ROI signals, slice timing and motion. Each script runs on its own. Run `python <script>.py --help` for options.

| Script | Checks | Tests |
|---|---|---|
| `check_parcellation.py` | the atlas parcellation in common space (ROI labels, coverage) | — |
| `check_roi_signals.py` | ROI time series against the RABIES outputs | — |
| `stc_axis_check.py` | slice-timing axis and slice order from the data (per-slice BOLD lag against candidate orders). It showed that this dataset's slices run along Z and are interleaved (`alt+z`) | `tests/stc_axis/` |
| `make_slice_maps.py`, `make_slice_timing_pairs.py`, `slice_timing_lag_check.py` | whether the slice acquisition order explains complex-PCA lags (each mouse sits differently in the slice package) | `tests/slice_maps/`, `tests/slice_timing/`; see `README_slicetiming_lag_check.md` |
| `roi_signal_homogeneity.py` | how well each ROI mean represents its voxels (variance explained, first-component ceiling, mean pairwise r) | `tests/roi_homogeneity/` |
| `mouse_motion_qc.py`, `subject_matrix_qc.py` | head motion per mouse, and subject FC matrices before stacking | — |

`welch_tests_scipy.py` is the study's shared Welch-test module, an identical copy of the one in the other packages. The slice-timing test uses the complex PCA functions in `../dfc/`.

To run a test: `bash tests/<name>/run_test.sh`. It simulates data, so no study data are needed.
