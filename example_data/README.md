# Example data

These are derived data from the study's mice. The raw data are on OpenNeuro (ds008951). They were preprocessed with RABIES and reduced to 82 ROI-mean time series per mouse (steps 1–2 of the top-level README).

| File | Contents | Used by |
|---|---|---|
| `hmm_signals.zip` | `hmm_signals/signal1.txt` … `signal95.txt` (570 volumes × 82 ROIs, ROI order as in `code/static_v2/example_files/roi_names_order.csv`), `meta_data.csv` (subject_id, group), `fc_check.csv` (check against RABIES's FC matrices) | steps 6–13 (HMM, DCC-GARCH, complex PCA) |
| `../code/static_v2/example_files/` | `merged.csv` (95 stacked 82 × 82 FC matrices), `roi_names_order.csv`, `subject_group_order_mappings.csv` | steps 4–5 (static FC) |

Groups: NTG_Y, NTG_M, NTG_A (non-transgenic young, middle-aged, aged), CRND8_Y, CRND8_M (TgCRND8), 5XFAD_Y, 5XFAD_M. Group sizes and ages are in the paper's Methods.
