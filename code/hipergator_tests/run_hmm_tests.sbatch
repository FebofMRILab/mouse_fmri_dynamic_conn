#!/bin/bash
# HMM code tests: (1) synthetic equivalence tests, (1b) updated vs previous
# run5B functions, (2) previous vs updated code on the existing mouse fit (no
# refit), (3) whether the previous run4e outputs come from one fit or two,
# (4) the updated run4e -> run4d -> run5 -> run5B chain on simulated data,
# (4b, 4c) the fit-selection rule on simulated fits and on eight real run4e fits
# of the simulated cohort, (5) the drift script (Python) on that simulated fit.
# One core; about 1.5 h.
# Step 5 uses the Python environment from setup_python_env.sh.
#
# Submit from code/ (the folder that contains hmm_v2/):
#   mkdir -p logs && sbatch hipergator_tests/run_hmm_tests.sbatch
# Paths can be overridden at submission, e.g.
#   sbatch --export=ALL,BCT_DIR=/blue/febo/MATLAB/BCT hipergator_tests/run_hmm_tests.sbatch
#SBATCH --job-name=hmm_v2_tests
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset DISPLAY      # no X11 in batch jobs (silences "Invalid MIT-MAGIC-COOKIE-1 key")

# ---- inputs: the existing (previous) mouse fit, K = 3 pre-PCA 15, no refit.
# These tests compare previous and updated code on the same fit, so they use
# the fit that already exists, not the reported model. Data folder as in hmm_config.m.
SIGNALS=${HMM_OUT_BASE:-/blue/febo/share/Abeta_Age/rabies/signals}
HMM_RESULTS=${HMM_RESULTS:-$SIGNALS/hmmmar_simple_k3_pca15/raw_hmm_results.mat}
HMM_SIGNALS=${HMM_SIGNALS:-$SIGNALS/hmm_signals}
HMMMAR_DIR=${HMMMAR_DIR:-/blue/febo/MATLAB/HMM-MAR-master}
if [[ -z "${BCT_DIR:-}" ]]; then        # find the folder holding BCT's threshold_proportional.m
    f=$(find /blue/febo/MATLAB -name threshold_proportional.m -print -quit 2>/dev/null)
    BCT_DIR=${f:+$(dirname "$f")}
fi
OUT=$PWD/hmm_v2_test_results

for p in "$HMM_RESULTS" "$HMM_SIGNALS" "$HMMMAR_DIR" "${BCT_DIR:-missing}" hmm_v2/tests; do
    [[ -e "$p" ]] || { echo "Not found: $p  (set it with --export=ALL,NAME=path)"; exit 1; }
done
mkdir -p "$OUT"
echo "Start $(date)"
echo "Fit: $HMM_RESULTS"; echo "Signals: $HMM_SIGNALS"
echo "HMM-MAR: $HMMMAR_DIR"; echo "BCT: $BCT_DIR"; echo "Results: $OUT"

module purge
module load matlab

echo "== 1. Synthetic equivalence tests"
matlab -nodisplay -nosplash -singleCompThread -batch "cd('hmm_v2/tests'); \
    r = test_hmm_v2_synthetic('$HMMMAR_DIR', '$BCT_DIR'); \
    writetable(struct2table(r), '$OUT/synthetic_tests.csv'); quit force" \
    || echo "Synthetic tests stopped with an error (see above)."

echo "== 1b. Updated vs previous run5B functions (block indices, network, rank, Neff, FC)"
matlab -nodisplay -nosplash -singleCompThread -batch "cd('hmm_v2/tests'); \
    r = test_run5B_helpers('$HMMMAR_DIR', '$BCT_DIR'); \
    writetable(struct2table(r), '$OUT/run5B_helper_tests.csv'); quit force" \
    || echo "run5B helper tests stopped with an error (see above)."

echo "== 2. Previous vs updated code on the existing mouse fit"
matlab -nodisplay -nosplash -singleCompThread -batch "cd('hmm_v2/tests'); \
    test_hmm_v2_real_data('$HMM_RESULTS', '$HMM_SIGNALS', '$OUT/real_data', \
    '$HMMMAR_DIR', '$BCT_DIR'); quit force" \
    || echo "Real-data test stopped with an error (see above)."

echo "== 3. Do the existing outputs of the previous run4e describe one fit or two?"
MODEL_DIR=$(dirname "$HMM_RESULTS")
matlab -nodisplay -nosplash -singleCompThread -batch "cd('hmm_v2/tests'); \
    check_fit_consistency('$MODEL_DIR', '$OUT/fit_consistency', '$HMMMAR_DIR'); quit force" \
    || echo "Fit-consistency check stopped with an error (see above)."

echo "== 4. Updated run4e -> run4d -> run5 -> run5B on a simulated cohort"
matlab -nodisplay -nosplash -singleCompThread -batch "cd('hmm_v2/tests'); \
    test_workflow_synthetic('$HMMMAR_DIR', '$BCT_DIR', '$OUT/workflow'); quit force" \
    || echo "Workflow test stopped with an error (see above)."

echo "== 4b. Fit-selection rule (select_reported_fit.m) on simulated fits with a known answer"
matlab -nodisplay -nosplash -singleCompThread -batch "cd('hmm_v2/tests'); \
    r = test_select_reported_fit('$HMMMAR_DIR', '$OUT/select_fit'); \
    writetable(struct2table(r), '$OUT/select_fit_tests.csv'); quit force" \
    || echo "Fit-selection test stopped with an error (see above)."

echo "== 4c. Fit selection on real run4e outputs (8 fits of the simulated cohort from step 4)"
matlab -nodisplay -nosplash -singleCompThread -batch "cd('hmm_v2/tests'); \
    r = test_select_on_run4e('$HMMMAR_DIR', '$OUT/workflow', '$OUT/select_on_run4e'); \
    writetable(struct2table(r), '$OUT/select_on_run4e_tests.csv'); quit force" \
    || echo "Selection on run4e outputs stopped with an error (see above)."

echo "== 5. Drift test (Python): equivalence with the confirmatory code, then on the simulated fit"
ENV=${ENV:-/blue/febo/$USER/envs/fc_static}
module purge
module load conda
PY="conda run --no-capture-output -p $ENV python"
(cd hmm_v2 && $PY tests/test_occupancy_drift.py) 2>&1 | tee "$OUT/occupancy_drift_test.txt"
$PY hmm_v2/hmm_occupancy_drift.py --model_dir "$OUT/workflow/work/signals/hmmmar_simple_k2_hmmpca5" \
    2>&1 | tee "$OUT/occupancy_drift_workflow.txt"

echo "End $(date). Send back the folder $OUT and this log."
