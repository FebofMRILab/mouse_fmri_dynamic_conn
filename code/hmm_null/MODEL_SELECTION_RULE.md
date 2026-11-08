# Pre-specified rule: which HMM is confirmed and reported (ROI-mean signals)

Written 26 Oct 2026 and decided by the PI (provenance D32), before the results it governs were seen. Nothing below is changed after the step 5 results on all 95 mice or the step 6 analysis have been seen. Record this file's SHA-256 on HiPerGator before either is looked at (end of file). Any later change is a dated amendment at the end, with what had been seen when it was made.

## Why a new rule

- The ROI time series changed from the first eigenvariate to the ROI mean (E47, D29). Every HMM result before this date rests on the eigenvariate signals.
- The earlier choices of the reported model (HMM-PCA rank 15, D16; K = 4, D17) came from results on those signals, and K = 4 was chosen after the confirmatory results. They no longer decide anything.
- Pooled PCA signal rank does not set the HMM-PCA rank, so the model family, dimension and K are treated as open. They are chosen here by a mechanical rule rather than after seeing results.

## What had been seen when this was written

- **Signal rank on the ROI-mean signals** (steps 4, 4b).
  - Per mouse: parallel analysis median 12 (range 6–19); Marchenko–Pastur median 5 (3–6).
  - Pooled: Marchenko–Pastur 21; parallel analysis 24.
- **Step 5 on mice 1–58 only** (E49: no NTG_A mice, NTG_M 4 of 9, 5XFAD_M 6 of 12). Not used for anything here.
- **Step 5 on all 95 mice:** running; not seen.
- **Step 6** (pre-specified 30 Sep 2026; HMM-PCA rank 15 and 25, K 2–4): run on the 95 mice' ROI-mean signals; not analysed, no output viewed.

## The rule

1. **Screen.**
   - The data are step 5 (`run4b_hmm_splithalf_null.m`, every model configuration, K 2–10, 10 split-halves, 10 null fits per cell) on all 95 mice; `metadata_aligned.csv` must list 95.
   - The table is `table_hmm_splithalf_null.csv`, rows with `metric = demeaned` and `null = phase`. The phase-randomised null decides; IAAFT is descriptive only.
2. **Lead.** A model × K cell leads if its one-sided Mann–Whitney p (observed > null) is below 0.05/9: Bonferroni across the 9 K of that model.
3. **Candidate.**
   - The candidate is the leading cell with the largest `delta_median` (median observed minus median null, demeaned state-FC similarity).
   - Ties at three decimals go to the smaller K, then to the smaller dimension (PCA dimension or HMM-PCA rank; "all ROIs" = 82).
   - `select_screen_candidate.py` applies items 1–3 and writes the result; no other choice is made by hand.
4. **Confirmation of the candidate.**
   - **Already covered:** if the candidate is HMM-PCA rank 15 or 25 at K 2, 3 or 4, step 6 is its confirmation, and no new test is run.
   - **Otherwise,** the candidate's model gets a calibrated test with step 6's design (`run4c_confirm_lowrank.m`):
     - 30 split-halves and 30 phase-null fits per K;
     - a replica arm of stationary Gaussian replicas with the same auto- and cross-spectra;
     - per split, c_r = (obs_r − null_r) − (rep_r − repnull_r), with the demeaned state-FC similarity;
     - a one-sided Wilcoxon signed-rank test of c > 0, with Holm across this test's cells.
   - **Pass:** Holm p < 0.05 and the weakest matched state replicates better in the observed than in the null splits (medians). A passing cell whose occupancy drifts between scan halves is reported as drift. IAAFT is descriptive only.
   - **K cells:** 2, 3 and 4 if the candidate's K is 2, 3 or 4; otherwise the candidate's K − 1, K and K + 1.
   - **Options:** as in `run4c_confirm_lowrank.m`. The only change is the observation model: `options.pca = d` for Gaussian PCA-d, no reduction for "Gaussian, all ROIs", or `options.lowrank = r` for another HMM-PCA rank.
   - **Seeds:** seed 20261026 and replica seed 20261027; they differ from step 5 (123) and step 6 (20261001, 20261002).
   - The script and its settings are written and checked before it is run. Only the model and K lines differ from step 6.
5. **No lead.** If no cell leads, no new test is run, and step 6's result is reported under item 7.
6. **Step 6 is analysed as pre-specified and reported whatever its outcome.**
   - **Reported model:** across all passing cells (step 6 and, if run, item 4), the main text reports the cell with the largest median c. Ties go to the smaller K, then the smaller dimension. Every passing cell is reported in the Supplement.
   - **No passing cell anywhere:** item 7 applies.
7. **No passing cell** (PI, 26 Oct 2026).
   - No HMM state-level group tests; steps 7–10 are not run for inference.
   - The HMM section reports a negative confirmatory result, next to the eigenvariate run's result.
   - Dynamic-FC claims rest on the calibrated DCC/GARCH/complex PCA tests (steps 12 and 16).
   - No widening of the K range, ranks, model families or nulls after the fact; anything further is exploratory and labelled as such.
8. **Signal rank** is reported descriptively, as context for the chosen dimension. It does not choose anything.
9. **Fits.**
   - Step 7a waits until items 1–7 have decided.
   - `FIT_SELECTION_RULE.md` then gets a dated amendment naming the model (family, dimension, K) before any of the ten fits is run. Its criteria (free energy, recurrence at r ≥ 0.9 in more than half of the other fits, whole-fit outcomes, inference only for recurring states) stay unchanged.
   - `run_hmm_fits_array.sbatch` takes that model instead of the fixed rank 15, K = 4.
   - If nothing passes, step 7a is not run.

## To be stated in the paper

- The model was chosen from a screen and confirmed on the same 95 mice. Fresh randomness and the replica arm calibrate the test against the null, but they do not remove the optimism of choosing the best of many cells (winner's curse). Selecting on half of the mice and confirming on the other half was considered and rejected, because groups of 5 would leave both halves too weak.
- The history: the eigenvariate signals (E47), the earlier model choices (D16, D17), the 58-mouse screen (E49), and this rule (D32).

## Record the hash before looking (HiPerGator, from `code/hmm_null/`)

```bash
( set -o noclobber; { sha256sum MODEL_SELECTION_RULE.md; date -Iseconds; } > MODEL_SELECTION_RULE.sha256 )
```

## Amendment 1, 27 Oct 2026: two reported models, the primary chosen by precision (D33)

Decided by the PI on 27 Oct 2026 ("Gaussian PCA-25 K3 primary and the low rank 15, K2 secondary"). It was made after the step 6 and step 6b results had been seen, and before any group statistic of any fit of either model had been computed. Before this amendment the file had SHA-256 `a6dc76c24e1a1b9d0e4343d4321dc80582e3bab4f0344131a75736b59a6f8f7c`, the version recorded on HiPerGator before the results (`MODEL_SELECTION_RULE.sha256`).

### Deviation from item 6

Item 6 puts in the main text the passing cell with the largest median c: HMM-PCA rank 15, K = 2 (median c 0.086, against 0.048 for Gaussian PCA-25, K = 3). It is replaced by two reported models:

- **Primary (main text):** Gaussian HMM on the first 25 principal components (`options.pca = 25`), K = 3. It is the step 5 screen candidate and passed step 6b.
- **Secondary:** HMM-PCA (`options.lowrank = 15`), K = 2. It passed step 6.
- **Both** go through `hmm_v2/FIT_SELECTION_RULE.md` (ten fits each; group tests only for recurring states) and the same group tests, and both are reported. An amyloid effect is called robust where the two models agree; where they disagree, the paper says so.
- **How agreement is judged** is written into this file, as a dated amendment, before any group statistic of either model is computed.
- The models are set in `hmm_v2/MODEL_PRIMARY.env` and `hmm_v2/MODEL_SECONDARY.env`.

### Why (PI)

The question is whether amyloidosis changes dynamic connectivity, so the main-text model should be the one whose state structure replicates most consistently.

- **The item 6 criterion is not on a common scale.** Median c is compared across model families, but its scale depends on the model (dimension and K).
- **On scale-free measures, the Gaussian cell is stronger.**
  - One-sided Wilcoxon p is about 4 × 10⁻⁵, against about 0.007. These are back-calculated from Holm p 1.2 × 10⁻⁴ across 3 cells and 0.044 across 6.
  - The 95% CI of median c is 0.026–0.088, against 0.004–0.162.
- **The choice is blind to group.** Split-half reproducibility uses no group labels, and no group result of either model had been computed.
- **It is a change made after the results.** A scale-free comparison across families was considered when this rule was written and was not adopted. Adopting it now, after the results, is a deviation, and the paper states it.

### Against the primary model (stated in the paper)

- **Winner's curse.** It was chosen as the best cell of the screen and confirmed on the same 95 mice, so its p is optimistic. Rank 15, K = 2 was fixed on 30 Sep 2026, before the ROI-mean signals existed.
- **Drift trends.** In step 6b's full fit, occupancy drifts in states 2 and 3 at p 0.035 and 0.026 (Holm 0.079). Rank 15, K = 2 shows no drift (Holm p 0.30; p 0.15 counted once, E50).

### Added to the confirmation analysis (descriptive; computed after the decision, which it does not change)

`analyze_confirm_lowrank.py` also writes, for each cell:
- the Hodges–Lehmann estimate of c with its 95% CI (the estimator and interval that match the Wilcoxon signed-rank test);
- the matched-pairs rank-biserial correlation of c over the 30 splits (scale-free).

## Amendment 2, 27 Oct 2026: how agreement between the two models is judged (D35)

Decided by the PI on 27 Oct 2026 ("let's go with your proposed definitions of agreement"), on the assistant's proposal. It was made before any fit of either model was run and before any group statistic was computed, as amendment 1 requires. Before this amendment the file had SHA-256 `ee4832dd19a04bc048b09b1eef1139580438f4a77d81cf70b4d83480cf410978`.

1. **What is compared.** The 8 planned Welch t contrasts of every outcome that is tested in both models under `hmm_v2/FIT_SELECTION_RULE.md`:
   - **whole-fit outcomes:** switching rate and maximum FO;
   - **state outcomes:** fractional occupancy, dwell time (log10), and the state-network graph metrics (mean strength, binarised transitivity, binarised global efficiency).

   Outcomes without planned contrasts (state presence) and the within-group comparisons between states are reported for each model but not classified.
2. **Whole-fit outcomes** are compared directly: the same contrast in the two models.
3. **State pairing.**
   - Each tested state of the primary model is paired with the secondary state whose pooled state FC has the highest Pearson r with it. Pooled state FC is `R.state_fc` of the reported fit: the Gamma-weighted correlation in ROI space, upper triangle without the diagonal. All secondary states are candidates, tested or not.
   - The pairing uses no group information. It is computed and written before any group table is read.
   - Several primary states may pair with the same secondary state.
   - Every pair is reported with its r, and with the r against every other secondary state. No threshold is applied.
   - If the paired secondary state is not tested in the secondary model (it does not recur), the primary state's contrasts are classed "no tested counterpart".
4. **Classification of each contrast.**
   - **Definitions:** q is the BH q of each model's own analysis (pooled per outcome, or per graph metric); the sign is the sign of the Welch t (first group minus second).
   - **robust:** q < 0.05 in both models, same sign;
   - **opposite:** q < 0.05 in both, opposite signs;
   - **primary only / secondary only:** q < 0.05 in one model only;
   - **neither:** q ≥ 0.05 in both;
   - **not assessable:** untestable (no t or q) in either model, or no tested counterpart.

   Only "robust" effects are called robust in the paper. Every class is reported.
5. **K = 2 occupancy (E50).** The secondary model's state 2 occupancy contrasts are state 1's test with the sign reversed, and they carry state 1's q. The pairing decides which secondary state's row is used, and that row's own sign counts.
6. **Mechanics.** `hmm_py/hmm_model_agreement.py` applies items 1–5 after both pipelines and writes `<signals>/hmm_model_agreement/`. No choice is made by hand.

## Amendment 3, 28 Oct 2026: the agreement is descriptive (D37)

Decided by the PI on 28 Oct 2026: "I will use the agreement as internally descriptive. But the presentation of both model results will be done." Before this amendment the file had SHA-256 `a45d44d80ca7acd77eda25bbb2caafb5da7ac5c52bf8d69db8f8548c66f477b2`.

### What changes

- **The classification is descriptive.** Amendment 2's classification (robust, opposite, primary only, secondary only, neither, not assessable) is kept as a descriptive comparison of the two models. It no longer decides which effects are called robust (amendment 2, item 4; amendment 1, "an amyloid effect is called robust where the two models agree").
- **Both models are reported in full.** Every pre-specified group test of both reported models (`FIT_SELECTION_RULE.md`), with its own Holm and BH correction, is presented, as each model's own result.

### What had been seen when this amendment was written

All group results of both models, and the agreement output (`signals/hmm_model_agreement/`, 06:58 on 28 Oct 2026). The assistant had pointed out two things about those results:
- **The two "robust" effects** (mean network strength, NTG_M lower than CRND8_M and than 5XFAD_M, in primary state 1 / secondary state 2) lie in the state where effective N differs most between these groups. There, strength falls with log effective N within groups (r −0.78 in the primary, −0.46 in the secondary).
- **A Neff-matched sensitivity analysis** was proposed. The PI declined it.

### Stated in the paper

- The agreement rule was fixed before the results and made descriptive after them.
