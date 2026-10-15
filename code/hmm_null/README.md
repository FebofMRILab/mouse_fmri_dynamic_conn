# HMM split-half null and pre-specified confirmatory test

| Part | Files | Details |
|---|---|---|
| Split-half null across HMM configurations (K selection) | `run4b_hmm_splithalf_null.m`, `hmm_select_k_splithalf_null.m`, `hmm_match_states_fc.m`, `hmm_phase_randomise.m`, `hmm_iaaft_surrogate.m`, `collect_sensitivity_null.m`, `plot_hmm_splithalf_null.py`, `submit_hmm_chunks.sh`, `slurm/hmm_*` | `README_hipergator.md` |
| Signal rank | `mp_signal_rank.py` | |
| Confirmatory test of HMM-PCA 15 and 25 | `run4c_confirm_lowrank.m`, `dfc_stationary_replica.m`, `analyze_confirm_lowrank.py`, `submit_confirm_lowrank.sh`, `slurm/confirm_*` | `README_confirm_lowrank.md` |

**Where it runs.** These scripts run inside the data folder (the one that holds `hmm_signals/`). From `code/`, run `bash hipergator_tests/link_into_signals.sh` once, then submit from `signals/`.

**Fit function.** `hmm_fit_raw_timeseries.m` here is the previous version, the one these runs used (the copy you sent on 1 Oct 2026). Check that it matches the file on HiPerGator. The updated fit function in `hmm_v2/` is used from the reported fit (run4e) on.
