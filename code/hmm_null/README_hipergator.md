# Split-half K selection with a null — running on HiPerGator

Resource budget: **at most 16 cores × 8 GB at a time.**

## Files (all in the folder with `hmm_signals/` and `hmm_fit_raw_timeseries.m`)

| File | Role |
|---|---|
| `run4b_hmm_splithalf_null.m` | Runner; configuration at the top (primary = `gaussian`, `pca_dim_primary = 25`) |
| `hmm_select_k_splithalf_null.m` | Split-half + null; pool, serial or chunked; checkpoints |
| `hmm_match_states_fc.m` | State matching (raw = original metric; demeaned = new) |
| `hmm_phase_randomise.m` | Surrogates |
| `collect_sensitivity_null.m` | Merges all models into one table + figure |
| `submit_hmm_chunks.sh` | **Main way to submit** (chunked arrays, 16-core cap) |
| `slurm/hmm_chunks.sbatch` | 32 chunks × 1 core × 8 GB, max 15 running |
| `slurm/hmm_finish.sbatch` | 1 core: summary, K choice, final fit for model 1 |
| `slurm/hmm_quicktest.sbatch` | Small serial test |
| `slurm/hmm_primary_pool.sbatch` | Alternative: 16-core pool (needs a free PCT seat) |
| `slurm/hmm_listconfigs.sbatch`, `slurm/hmm_collect.sbatch` | List models / merge models |

## Why chunks instead of a pool

`parpool` needs a Parallel Computing Toolbox seat, and UF's seats run out.
Chunk tasks are ordinary single-core MATLAB sessions, so they never touch that
toolbox. State matching no longer uses `corr`, so the Statistics Toolbox is not
checked out either. If the pool route is used and no seat is free, the code now
falls back to serial instead of crashing.

## Submit

```bash
cd /blue/febo/<analysis folder>
mkdir -p logs
sbatch slurm/hmm_listconfigs.sbatch        # writes model_configs.csv, prints the model list

bash submit_hmm_chunks.sh                  # primary model (row 1) only
bash submit_hmm_chunks.sh 1 3              # primary, then model 3
bash submit_hmm_chunks.sh all              # every model, one after another, then merge
```

For each model, the script submits:
1. The chunk array: 32 tasks, at most 15 at once. The next model's array waits
   for the previous array to finish.
2. A 1-core finish job, which starts when its own array ends (15 + 1 = 16 cores).
3. A merge job at the end, if more than one model was submitted.

## Re-using earlier work

Checkpoints are matched on K, seed, subject halves and options. With the
primary set to `gaussian_pca_25`, the earlier sensitivity run's fits (observed
reps 1–10, null reps 1–5, in `K_selection_gaussian_pca_25/jobs/`) are loaded,
not refitted. Only null reps 6–10 and the final fit are new.

## Monitoring and failures

```bash
squeue -u $USER
tail -f logs/hmm_chunks_<arrayid>_<task>.out
ls hmm_raw_timeseries_null/K_selection_gaussian_pca_25/jobs | wc -l
sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State
```

- **A chunk failed or ran out of time.** Resubmit only that chunk:
  `sbatch --array=7 --export=ALL,HMM_CONFIG=1 slurm/hmm_chunks.sbatch`.
  Keep `HMM_N_CHUNKS=32` in the script.
- **The finish job ran before every chunk finished.** It uses `afterany`, so it
  still runs, and it fits any missing jobs itself (serially, which is slower).
  Resubmitting it later is always safe.
- **A license error in a chunk** means the base MATLAB license, not PCT. Lower
  the throttle (`%15` to e.g. `%8`) in `hmm_chunks.sbatch`.

**Subjects and TR (E49, 26 Oct 2026).** `run4b_hmm_splithalf_null.m` had the human settings hard-coded (`n_subjects = 58`, `TR_sec = 3`), so on the mouse data it loaded signal1–58 only (with the warning "N more signal file(s) exist ... Check n_subjects"). It now reads `HMM_N_SUBJECTS` (default 95) and `HMM_TR` (default 2), like `run4c`. For the human data: `HMM_N_SUBJECTS=58 HMM_TR=3`. Check a run with `tail -n +2 hmm_raw_timeseries_null/metadata_aligned.csv | wc -l` (95 for the mice).

## Stricter null: IAAFT

`HMM_NULL=iaaft` swaps the phase-randomised surrogates for multivariate IAAFT
surrogates (`hmm_iaaft_surrogate.m`). These also keep each ROI's value
distribution, so heavy tails and amplitude bursts survive in the null. An
effect that beats the phase null but not IAAFT is explained by non-Gaussian
amplitude structure rather than FC switching.

- The observed-data fits are shared with the phase run (already checkpointed),
  so only the null fits are new.
- IAAFT outputs carry the suffix `_iaaft`: `split_half_K_selection_iaaft.csv`,
  `nulliaaft_*.mat` checkpoints. Phase results are never overwritten.
- No final fit is run: the final model comes from the phase run.
- Restrict K to keep it cheap. The `ALL` export passes your shell variables
  through to every job:

```bash
HMM_NULL=iaaft HMM_K_VALUES="2:4" bash submit_hmm_chunks.sh 1
```

With 10 null repetitions and K = 2–4 that is 30 null jobs (60 fits); the
observed jobs load from checkpoints. Afterwards `collect_sensitivity_null.m`
also writes `null_comparison_phase_vs_iaaft.csv`.

**Phase, then IAAFT, without waiting at the terminal** (`run_nulls_in_sequence.sh`,
25 Oct 2026). Each stage is submitted by a short 1-core launcher job that starts
when every job of the previous stage has ended (`afterany`), so stages never
overlap (16-core limit) and IAAFT starts only after the phase run has written
the observed fits it reuses. It calls the submit scripts unchanged and refuses
to start if `hmm_raw_timeseries_null/` or `hmm_confirm_lowrank/` already exists
(their checkpoints would be reused: rename them first, README run order, E48).

```bash
bash run_nulls_in_sequence.sh both      # list models -> split-half phase (all) -> confirmatory phase
                                        #   -> split-half IAAFT (model 1, K 2-4) + collect -> confirmatory IAAFT
bash run_nulls_in_sequence.sh split     # split-half only;  bash run_nulls_in_sequence.sh confirm  # confirmatory only
IAAFT_MODELS=all bash run_nulls_in_sequence.sh split   # IAAFT for every model (also IAAFT_K="2:6")
cat logs/null_sequence.log              # one line per stage with its job IDs
```

Tested with a mock `sbatch` (each stage depends on all job IDs of the previous
one; HMM_* settings from the calling shell do not reach the jobs, HMM_TOOLBOX
is kept; one collect job per null).

## Activation-state diagnostic: zeromean = 0

`HMM_ZEROMEAN=0` fits states that also have a mean (activation /
co-activation states). Everything goes to `<out_dir>_zeromean0`, so nothing
from the primary `zeromean = 1` analysis is touched.

**One change to `hmm_fit_raw_timeseries.m` first.** The state mean maps are
scored alongside FC, so the fitting function must return them. Add either of
these after the HMM is fitted (use your function's variable names):

```matlab
R.hmm = hmm;                                   % simplest: lets the pipeline call getMean
% or
for k = 1:options.K, R.state_mean{k} = getMean(hmm, k); end
```

Each map must have one value per ROI. PCA-space means (25 values) are
projected back to ROI space when `R.hmm` is available; otherwise they are
skipped with a warning. Without mean maps the run still works, but it only
scores FC, which misses the point of this diagnostic.

**Order (one after the other, never in parallel: they share observed fits):**

```bash
sbatch --export=ALL,HMM_ZEROMEAN=0 slurm/hmm_quicktest.sbatch   # log must NOT warn "no state mean maps"
HMM_ZEROMEAN=0 bash submit_hmm_chunks.sh 1                       # phase null + final fit
HMM_ZEROMEAN=0 HMM_NULL=iaaft bash submit_hmm_chunks.sh 1        # IAAFT null, after the first finishes
HMM_ZEROMEAN=0 sbatch slurm/hmm_collect.sbatch                   # phase vs IAAFT table
```

The IAAFT null matters most here. Activation events are non-Gaussian bursts,
which phase randomisation removes but IAAFT keeps, so an activation "state"
that beats only the phase null may just reflect heavy-tailed amplitudes.

## Robust columns (all runs)

`delta_median_*` (observed median − null median) and `p_mw_*` (one-sided
Mann–Whitney, observed > null, uncorrected across K) are now in every
summary, for demeaned FC and, when available, for mean maps (`*_meanmap`,
cosine similarity). K is chosen by the largest median delta, with a warning
when its p ≥ 0.05. Re-running a finish job adds these columns to existing
results from checkpoints.
