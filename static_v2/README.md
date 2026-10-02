# Static FC scripts (published-code version)

Put these files in the folder you run the static FC analysis from, next to `roi_names_order.csv`, `subject_group_order_mappings.csv` and `merged.csv`:

| File | Role |
|---|---|
| `full_analysis_abeta_mice_permutation_transitivity.py` | mouse analysis (updated) |
| `full_analysis_abeta_humans_permutations.py` | human analysis (updated) |
| `welch_tests_scipy.py` | replaces `welch_permutation.py`: SciPy Welch tests (omnibus: permutation; pairwise: ordinary Welch t-test) |
| `nbs_published.py` | `run_nbs` used by both scripts: bctpy NBS, same inputs and outputs as before |
| `bctpy_nbs_welch.py` | bctpy 0.6.1 `nbs.py` with fixes F01–F03 (`fixes/F01-F03_nbs.patch`) |
| `roi_group_table.py` | ROI-by-ROI nodal table (replaces `statistical_table.r`): Welch ANOVA per ROI + the 8 planned Welch t contrasts; CSV, Word and PDF |

## Requirements

- SciPy ≥ 1.16
- statsmodels
- networkx
- bctpy 0.6.1, installed with `pip install bctpy==0.6.1`
- python-docx (Word tables)
- the plotting packages the scripts already use

## Running

Run the scripts as before. The NBS now takes about 4 h for the mouse script (8 comparisons) and about 1 h for the human script, on one core.

## Changes and evidence

See `CHANGES.md`. The tests are in `tests/`:

```bash
python tests/test_static_published.py
python tests/test_roi_group_table.py
```
