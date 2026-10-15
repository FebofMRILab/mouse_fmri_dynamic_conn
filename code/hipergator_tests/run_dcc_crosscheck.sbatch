#!/bin/bash
# DCC cross-check: our dfc_dcc_subject.m against the published DCC estimator in
# Lindquist's DCC toolbox (dcc_mvgarch.m), on simulated data with known
# parameters and on 5 mice. Needs MATLAB's Optimization Toolbox (fmincon) for
# the published code. One core; a few minutes.
#
# Once: get the toolbox (MATLAB code, GPL-3):
#   git clone https://github.com/canlab/Lindquist_Dynamic_Correlation /blue/febo/MATLAB/Lindquist_Dynamic_Correlation
# Then, from code/:
#   mkdir -p logs && sbatch hipergator_tests/run_dcc_crosscheck.sbatch
#SBATCH --job-name=dcc_crosscheck
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset DISPLAY      # no X11 in batch jobs (silences "Invalid MIT-MAGIC-COOKIE-1 key")
SIGNALS=${HMM_OUT_BASE:-/blue/febo/share/Abeta_Age/rabies/signals}
DCC_CODE=${DCC_CODE:-/blue/febo/MATLAB/Lindquist_Dynamic_Correlation/DCC_toolbox/DCCcode}
N_MICE=${N_MICE:-5}
for p in "$SIGNALS/hmm_signals" "$DCC_CODE/dcc_mvgarch.m" dfc/tests/equivalence_published; do
    [[ -e "$p" ]] || { echo "Not found: $p"; exit 1; }
done
module purge
module load matlab
matlab -nodisplay -nosplash -singleCompThread -batch "\
    if ~license('test', 'Optimization_Toolbox'), error('Optimization Toolbox not licensed: fmincon is needed.'); end; \
    addpath('dfc/tests/equivalence_published'); \
    compare_dcc_with_lindquist('$SIGNALS/hmm_signals', '$DCC_CODE', '$PWD/dcc_lindquist_check', $N_MICE); quit force" \
    || { echo "Cross-check failed (see above)."; exit 1; }
echo "Results: $PWD/dcc_lindquist_check/"
