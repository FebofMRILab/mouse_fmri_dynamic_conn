# Positive control: simulated mice with a known mechanism

The schematic (`Fig_schematic_event_mechanism`) contrasts three ways to get
the same static FC:
- **A**, stationary Gaussian activity;
- **B**, sparse shared events;
- **C**, state switching.

This package makes those three mechanisms concrete. It builds simulated
cohorts that look like the mouse data to the pipelines. It then runs the
unchanged `run_dcc_cpca_null.m` and `run4c_confirm_lowrank.m` on them and sets
their outputs beside the mouse outputs. Whatever the mouse results turn out to
be, they can be read against what each mechanism actually produces, not what
we expect it to produce.

## What a simulated cohort copies from the mouse data

Each simulated cohort copies:
- the number of mice, ROIs, volumes, TR (2 s) and group labels;
- the group FC (pooled correlation), split into signal networks and noise. The
  networks are the top *k* eigen-components, with *k* the parallel-analysis
  signal rank, varimax-rotated into localised networks. The noise is the rest.
- each mouse's own network strengths. Simulated mouse *i* copies real mouse
  *i*, so between-mouse and between-group differences carry over.

All mechanisms use:
- the same haemodynamic response (peak 3 s, `--hrf_peak`);
- the same 0.01–0.1 Hz band-pass;
- the same spectra and, in expectation, the same static FC.

The pilot checked this: power spectra were identical across mechanisms.

| Cohort | Mechanism | Strength knob |
|---|---|---|
| A | Each network driven by Gaussian white noise. This is what the stationary replica is: the negative control. | none |
| B | Each network driven by Poisson events (lognormal amplitudes), each activating all of its ROIs at once. Stationary and non-Gaussian. | events per network per minute |
| C | 3 covariance states shared by all mice, differing in which networks dominate and which couple. Dwell times are 30–90 s; the states average to the static FC. | state separation (0–1) |
| C_strong | The same mice as C, with separation 0.9. | fixed |

## Matching: why two stages

B and C are each set to the strength at which they reproduce the mouse's
headline result: the calibrated DCC excess, median *d* = z_obs − z_replica
for `dcc_a`. At matched strength, every other outcome becomes a prediction
that can be compared with the mouse:
- the half-life of coupling changes;
- the HMM decision;
- bursts;
- FC change between scan halves.

The design went through one revision. Matching on cheap Python statistics
fails for C: the FC-change-between-halves statistic barely moves with 30–90 s
states, because each half averages over them.

## Run on HiPerGator

Put this folder's files in the folder you run the mouse pipelines from. It
must contain `hmm_signals/`, `dfc_dcc_cpca/`, both MATLAB packages and their
`slurm/` scripts.

```bash
# 1. grid: A (95 mice) + B at 1/4/16/64 events/min + C at separation 0.3/0.6/0.9 (32 mice each)
python sim_positive_controls.py --stage grid
AFTER=<confirm-run finish job ids> bash submit_positive_controls.sh grid

# 2. final: B and C matched to the mouse DCC excess, 95 mice each, plus C_strong
python sim_positive_controls.py --stage final
bash submit_positive_controls.sh final

# 3. compare
python summarize_positive_controls.py
```

**Where to run the Python steps.** `sim_positive_controls.py` takes a few
minutes for 95 template mice: it computes the parallel-analysis rank and 19
surrogates per mouse. Run it in an interactive `srun` session or a 1-core job
rather than on a login node.

**Wait for the confirmatory run.** Use `AFTER=` with the confirmatory run's
finish job ids (for example `AFTER=44083090:44083091`) so the jobs start after
it. The chain never uses more than 16 cores. Arrays run at most 12 tasks, and
each array waits for the previous array and for the finish jobs before that.

**Check a stage before submitting it.** `DRY_RUN=1 bash
submit_positive_controls.sh final` prints the sbatch commands without
submitting anything.

**Model choice.** `HMM_CONFIGS="1"` runs only lowrank 15. The default runs
both models, so the Holm family matches the mouse run.

**Cost, relative to your mouse runs:**
- **Grid:** DFC on 95 + 7 × 32 ≈ 320 mice, about 3.4× the mouse DFC run.
- **Final:** DFC on 3 × 95 mice (≈ 3×) plus the confirmatory HMM on 4 cohorts
  (≈ 4× the mouse confirmatory run).

The HMM part dominates. To halve it, use `HMM_CONFIGS="1"`.

**Order of checks.** The grid-stage `sim_positive_controls.py` prints the
mouse's burst and FC-change statistics next to the grid cohorts. That is
already a first, pipeline-free look at which mechanism is closer.

## Outputs

**`sim_positive_control/`:**
- `<cohort>/signal*.txt` and `meta_data.csv`;
- truth files: `truth_events.csv` for B, `truth_states.csv` for C;
- `sim_manifest_<stage>.json`, recording every parameter, seed and matched
  value;
- `sim_diagnostics_<stage>.csv`;
- `FigS_positive_control_design`.

**Pipeline outputs:** `dfc_sim_<cohort>/` and `hmm_confirm_sim_<cohort>/`,
next to the mouse outputs.

**`positive_control_summary/`:**
- `positive_control_schematic.csv`: the schematic's table rows, filled in for
  Mouse, A, B, C and C strong;
- `positive_control_table.csv`: every number;
- `positive_control_vs_mouse.csv`: mouse vs each cohort, per-mouse measures,
  Mann–Whitney;
- `FigS_positive_controls`:
  - calibrated excess in coupling (DCC), amplitude (GARCH) and the
    propagation envelope (cPCA);
  - half-life ratio;
  - bursts;
  - FC change;
  - HMM calibrated gain per cell;
  - state recovery against the simulated truth;
- `FigS_positive_control_matching`: grid strength vs DCC excess, with the mouse
  band.

## How to read it

- **A must come out null everywhere.** This covers the DCC excess, the HMM and
  bursts. If it doesn't, the pipeline is biased, and the problem is not in the
  mouse data.
- **C_strong must be detected by the HMM and recovered against the truth.** If
  it isn't, the HMM null in the mouse is uninformative about switching.
- **C at matched strength is the critical cell.**
  - If the HMM detects it, the mouse HMM result can rule out switching large
    enough to produce the observed DCC excess.
  - If the HMM misses it, the HMM row cannot discriminate B from C.
- **At matched DCC excess, B and C differ in their profile across measures**
  (see the pilot):
  - B brings shared bursts with little amplitude or envelope excess;
  - C brings amplitude and envelope excess without bursts;
  - the mouse's profile across panels A–C and E is the comparison.
- **`--c_variance_share 0`** makes states differ in coupling only, not in
  network amplitude. Run it as an extra cohort if the mouse profile sits
  between B and C.
- **Limits of the simulation:** the mechanisms are idealised, with one knob
  each and fixed dwell times. The noise is Gaussian, so the simulation has no
  motion spikes or physiological transients.
- **B is not specifically neural.** B models any sparse shared transient,
  neural or physiological. It cannot distinguish the two.

## Pilot (local, Octave; not results)

**Setup:** 10 simulated mice per cohort; a synthetic 40-ROI template with 12
signal networks; 9 surrogates. Values are medians of d = z_obs − z_replica.
Half-life is observed/surrogates, all mice.

| Cohort | DCC a | GARCH α | Envelope C1 | Bursts (z) | Half-life ratio |
|---|---|---|---|---|---|
| A Gaussian | 0.12 (p 0.65) | −0.26 | −0.59 | −0.7 | 1.01 |
| B 2 events/min | 4.44 (p < 0.001) | 2.92 | 2.61 | 6.4 | 0.98 |
| B 8 events/min | 1.93 (p 0.014) | 1.02 | −0.14 | 2.0 | 0.98 |
| B 32 events/min | 0.34 (p 0.39) | 0.50 | 0.25 | −0.2 | 1.00 |
| C separation 0.3 | 0.61 (p 0.12) | 1.35 | 0.99 | −0.1 | 1.05 |
| C separation 0.6 | 1.25 (p 0.03) | 1.83 | 1.54 | 0.0 | 1.05 |
| C separation 0.9 | 2.12 (p 0.005) | 1.77 | 1.23 | 0.2 | 1.02 |

What the pilot shows:
- **The calibration holds on simulated data:** A is null.
- **Both mechanisms produce a DCC excess, rising with strength.** The grid
  brackets a mouse-like d ≈ 1.4.
- **In the pilot, the DCC half-life did not separate the mechanisms.** It
  stayed within ±5% of the surrogates for events and for 30–90 s states alike,
  with no trend in switching strength. DCC(1,1) persistence (about 0.79 at the
  decimated TR of 4 s) reflects the band-limited signal, not how long states
  last. A half-life equal to the surrogates' (mouse: 12.7 vs 12.6 s) therefore
  does not show that coupling changes are brief.
- **At similar DCC excess, the discriminators are:**
  - bursts, which are high for B and ≈ 0 for C;
  - the amplitude/envelope excess, which is larger for C;
  - the HMM, which could not be run locally.
