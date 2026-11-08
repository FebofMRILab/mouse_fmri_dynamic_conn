# Pre-specified rule: which fit of the reported HMM is reported, and which states are tested

Written 4 Oct 2026, before any of the fits it governs were run. Decided by the PI (provenance workbook, D25). Nothing below is changed after the fits are run: the array job records this file's SHA-256 before fitting, and `select_reported_fit.m` stops if the file differs from the recorded version.

**Amended 27 Oct 2026, before any fit of the amended models was run (Amendment 1, end of file; D33).** The rule now applies to two models on the ROI-mean signals, each on its own: the primary, a Gaussian HMM on 25 principal components at K = 3, and the secondary, HMM-PCA rank 15 at K = 2. Where the original text below says K = 4 or HMM-PCA rank 15, read "the model the rule is applied to". Until this amendment the title read "which K = 4 fit is reported"; nothing else in the original text was changed.

## Why a rule is needed

- **Refits differ.** HMM-MAR reseeds MATLAB's generator from the clock at every initialisation (`train/initGamma_random.m`: `rng('shuffle')`), so each fit of the same model is a new solution.
- **The 4 Oct 2026 refit was a different solution.** It reproduced one of the four states of the confirmatory run's full fit (`getGammaSimilarity` 0.43; state FC r = 0.97 for that state, 0.75–0.77 for the other three).
- **Its group results have been seen.** Choosing among fits after seeing group results would be fit-shopping. The reported fit must be chosen by a criterion that does not use group labels or group results.
- **The model choice itself is not in question.** The confirmatory test compared split-half fits and supports K = 4 at rank 15 as a model; this rule decides which full-data solution is reported and which of its states support inference.

## The rule

1. **Model.** HMM-PCA, `options.lowrank` = 15, K = 4, with the options in `run4e_hmm_mar.m` (the options of the confirmatory full fits). Nothing about the model or the options changes.
2. **Fits.** Ten independent fits, `fit_01` … `fit_10`, each by `run4e_hmm_mar.m` unchanged with the defaults in `hmm_config.m` (`hipergator_tests/run_hmm_fits_array.sbatch`, which ignores any other `HMM_*` setting in the caller's environment). These ten folders are the only candidates; any other fit, whatever its free energy, is not. The 4 Oct 2026 refit and the confirmatory full fit are not candidates, because their results were seen before this rule was written; they are compared with the reported fit for description only.
3. **Valid fits.** A fit counts if run4e finished and saved its outputs. Fits with identical fit IDs count once. At least 8 valid fits are needed; with fewer, the missing tasks are rerun before selection.
4. **Reported fit.** The valid fit with the lowest final free energy (`fehist(end)` from `hmmmar`). This is HMM-MAR's default criterion for choosing among its own initialisations (`options.initcriterion = 'FreeEnergy'`, set in `utils/internal/checkoptions.m` and applied in `train/hmmmar_init.m`). Ties go to the lower fit number.
5. **Matching.** Every other valid fit is matched state by state to the reported fit with HMM-MAR's `getGammaSimilarity` (Hungarian assignment on the state time courses).
6. **Recurring states.** A state of the reported fit recurs if its pooled state FC (`R.state_fc`, upper triangle without the diagonal) correlates at Pearson r ≥ 0.9 with the matched state's in more than half of the other valid fits (at least 5 of 9 when all 10 fits are valid).
7. **Whole-fit outcomes.** Switching rate and maximum fractional occupancy depend on every state. Each is tested only if its per-mouse values correlate at Pearson r ≥ 0.9 with the reported fit's in more than half of the other valid fits, and at least one state recurs (item 9).
8. **Inference.**
   - Group tests (Welch ANOVA, planned Welch t contrasts, state presence, state-network graph metrics, and the group model of the bootstrap-stability table) are run only for recurring states and for the whole-fit outcomes that pass item 7. Within-group comparisons between states are run only between recurring states.
   - The multiple-comparison families (Holm across states or state x metric panels, BH across contrasts) contain only those states and outcomes.
   - Non-recurring states are described (occupancy, state FC, how often they recur) without group tests.
   - The occupancy drift test is run on every state of the reported fit, as pre-specified for the confirmatory analysis; a recurring state that drifts is reported as drifting.
9. **No recurring state.** If no state recurs, no K = 4 group tests are reported, including switching rate and maximum fractional occupancy. The paper reports the confirmatory test and the instability across fits.
10. **One fit for every later step.** The reported fit's folder is used by every later step (comparison with the confirmatory full fit, drift test, run4d, run5, run5B, `hmm_py`). No step refits.
11. **Reported whatever the outcome.** Final free energy of every fit; `getGammaSimilarity` between every pair of valid fits; for each state of the reported fit, the matched state FC r, time-course r and occupancy-across-mice r in every other fit and the recurrence count; per-mouse agreement between the reported fit and each other fit (the statistic of HMM-MAR's `hmmtestretest.m`); and the reported fit's agreement with the 4 Oct 2026 refit and with the confirmatory full fit.

## A known limitation

Recurrence is judged on state FC. Pooled state FC is dominated by the static FC that all states share, so a state can recur in FC while its time course and per-mouse occupancy do not (in the 4 Oct comparison, states that were not reproduced still had FC r = 0.75–0.77 with time-course r ≤ 0.15). The time-course and occupancy-across-mice agreement of every state are therefore reported next to the recurrence result (item 11), and the paper shows them for every tested state.

## Where the thresholds come from

- **r ≥ 0.9 and "more than half"** were proposed by the assistant and accepted by the PI on 4 Oct 2026. The only data seen when they were set was the one comparison above (0.97 for the reproduced state, 0.75–0.77 for the others), which uses no group information. No group result from any of the ten fits had been computed.
- **Item 7** applies the same threshold and majority to the two outcomes that are not tied to one state. It was added by the assistant while writing this rule, so that a test is never run on a quantity that changes from fit to fit.

## Amendment 1, 27 Oct 2026: two models on the ROI-mean signals (D33)

Written before any fit of either model was run, as `hmm_null/MODEL_SELECTION_RULE.md` item 9 requires. The models were chosen by the PI on 27 Oct 2026 (that rule's amendment 1). Before this amendment the file had SHA-256 `6735bc92b526058f83760cb160197a1375693b096403ba9b467ac8cb13eaa515`.

### The two models

| | Primary (main text) | Secondary |
|---|---|---|
| Observation model | Gaussian HMM on the first 25 principal components (`options.pca = 25`) | HMM-PCA (`options.lowrank = 15`) |
| K | 3 | 2 |
| Model file | `hmm_v2/MODEL_PRIMARY.env` | `hmm_v2/MODEL_SECONDARY.env` |
| Fits folder | `<signals>/hmm_k3_pca25_fits/` | `<signals>/hmm_k2_hmmpca15_fits/` |
| Confirmation | step 6b | step 6 |
| Full fit of that confirmation (description only) | `hmm_confirm_candidate/observed/gamma_gaussian_pca_25_K3.mat` | `hmm_confirm_lowrank/observed/gamma_lowrank_15_K2.mat` |

### What changes

1. **Model (item 1).** The rule is applied to each model separately, with the options in `run4e_hmm_mar.m` (those of the confirmatory full fits); only the observation model and K differ.
   - Each model is set in its own file, listed above, and nowhere else.
   - The files are read by `hmm_config.m` (the primary is its default), `hmm_py/hmm_model_selection.py`, `hmm_v2/hmm_occupancy_drift.py`, and the job scripts through `hipergator_tests/reported_model.sh primary|secondary`.
   - The array job copies the model file before fitting, into the fits folder (`model.env`) and into `<signals>/hmm_fits_<role>_model.env`. Every later task, `select_reported_fit.m` and `run_selected_fit.sh` stop if the file later differs, as for this rule.
2. **Data.** The ROI-mean signals (D29) in `<signals>/hmm_signals`, all 95 mice, for both models.
3. **Each model has its own ten fits, reported fit and recurring states.**
   - Items 2–11 apply within each model.
   - Item 9 applies per model: if no state of a model recurs, that model has no group tests.
4. **Wording.** "K = 4" and "rank 15" in the original text (title, items 1 and 9, the reasons) mean the model the rule is applied to. The reasons in "Why a rule is needed" hold for both: refits differ because HMM-MAR reseeds from the clock, and the reported fit must be chosen without group information.
5. **Not candidates (item 2).** As well as the fits named in item 2, these are not candidates:
   - each model's full fit from its confirmation (table above), whose occupancy and drift were seen;
   - any earlier run4e fit in `<signals>/hmmmar_simple_k3_pca25/` or `<signals>/hmmmar_simple_k2_hmmpca15/`;
   - the ten K = 4 fits of the original rule, on the eigenvariate signals (`hmm_k4_hmmpca15_fits_eig/`).
6. **Item 11.** Both fits named there (the 4 Oct 2026 refit and the confirmatory K = 4 full fit) are of the eigenvariate signals at K = 4, so neither is comparable.
   - There is no earlier fit of either model on the ROI-mean signals, and `select_reported_fit.m` reports the "previous refit" comparison as not applicable.
   - Each reported fit is compared, for description only, with its confirmation's full fit (table above), which is on the same signals.
7. **Compute.** Each array runs at most 8 fits at once (`--array=1-10%8`), so the two arrays together stay within 16 cores.

### What does not change

The criteria of items 2–11, within each model:
- ten fits, `fit_01` … `fit_10`, each by `run4e_hmm_mar.m` unchanged;
- at least 8 valid fits;
- the fit with the lowest final free energy is reported;
- other fits are matched with `getGammaSimilarity`;
- a state recurs if its state FC correlates at r ≥ 0.9 with the matched state in more than half of the other valid fits;
- whole-fit outcomes follow item 7;
- group tests only for recurring states and for whole-fit outcomes that pass;
- one fit for every later step;
- everything in item 11 is reported whatever the outcome.

### What K implies, written down before fitting

- **Primary, K = 3.** The occupancies of the three states sum to 1 for each mouse, but no state's occupancy is a function of another single state's. Each state is tested, with Holm across the recurring states as in the original rule. Holm is valid under any dependence between the tests.
- **Secondary, K = 2: occupancy is one test (E50).**
  - FO(state 2) = 1 − FO(state 1) for every mouse, so Welch's ANOVA, the Welch t contrasts (sign reversed) and the permutation p (same permutations) are identical for the two states.
  - If both states recur, occupancy is tested once (state 1), the result is shown for both states, and it counts once in the Holm family. The code written for K = 4 would have counted it twice and doubled its Holm p.
- **Secondary, K = 2: drift is one test (E50).**
  - Each mouse's occupancy change in state 2 is minus that in state 1, so the two-sided Wilcoxon p is the same.
  - It counts once at K = 2, in `hmm_occupancy_drift.py` and in step 6/6b's analysis (`analyze_confirm_lowrank.py`).
  - For rank 15, K = 2 in step 6, Holm p 0.30 becomes p 0.15: still no drift. No pass or fail in step 6 or 6b changes.
- **Secondary, K = 2: other consequences.**
  - Maximum FO = max(FO1, 1 − FO1). It remains a whole-fit outcome under item 7.
  - Comparing occupancy between the two states within a group tests FO(state 1) against 0.5.
- **Both models.** Dwell time, presence and the state-network graph metrics remain per-state tests, with Holm across the recurring states.

### What had been seen when this amendment was written

- **Step 6** (HMM-PCA rank 15 and 25, K 2–4; ROI-mean signals, 95 mice).
  - Rank 15, K = 2 passes: median c 0.086, bootstrap 95% CI 0.004–0.162, Holm p 0.044. Its weakest state is above the null, there is no drift (Holm p 0.30), and occupancy is about half in each state.
  - The other five cells fail (Holm p 0.086–0.096).
- **Step 6b** (Gaussian PCA-25, K 2–4).
  - K = 3 passes: median c 0.048, CI 0.026–0.088, Holm p 1.2 × 10⁻⁴.
  - Drift trends in states 2 and 3 (p 0.035 and 0.026; Holm 0.079).
  - K = 2 and 4 fail (Holm p 0.092).
- **Not computed:** any group result of any fit of either model.

### Run (from `code/`)

```bash
sbatch hipergator_tests/run_hmm_fits_array.sbatch primary
sbatch hipergator_tests/run_hmm_fits_array.sbatch secondary
sbatch --dependency=afterany:<primary array job ID> hipergator_tests/run_select_fit.sbatch primary
sbatch --dependency=afterany:<secondary array job ID> hipergator_tests/run_select_fit.sbatch secondary
bash hipergator_tests/run_selected_fit.sh primary      # after its selection
bash hipergator_tests/run_selected_fit.sh secondary
```
