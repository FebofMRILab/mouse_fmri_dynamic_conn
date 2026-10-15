#!/bin/bash
# Ten independent fits of the reported model (HMM-PCA rank 15, K = 4) for the
# pre-specified fit-selection rule (hmm_v2/FIT_SELECTION_RULE.md). Each array task
# runs hmm_v2/run4e_hmm_mar.m unchanged, with its own output folder:
#   <signals>/hmm_k4_hmmpca15_fits/fit_NN/hmmmar_simple_k4_hmmpca15/
# Then: sbatch hipergator_tests/run_select_fit.sbatch (picks the reported fit).
#
# Submit from code/ (/blue/febo/share/Abeta_Age/rabies/code):
#   mkdir -p logs && sbatch hipergator_tests/run_hmm_fits_array.sbatch
# Rerun single tasks that failed:   sbatch --array=3,7 hipergator_tests/run_hmm_fits_array.sbatch
# One core and 8 GB per task; 10 tasks at once stay within 16 cores.
#SBATCH --job-name=hmm_k4_fits
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=1-00:00:00
#SBATCH --array=1-10
#SBATCH --output=logs/%x_%A_%a.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset DISPLAY
SIGNALS=${HMM_OUT_BASE:-/blue/febo/share/Abeta_Age/rabies/signals}
FITS=${HMM_FITS_DIR:-$SIGNALS/hmm_k4_hmmpca15_fits}
DATA=${HMM_DATA_DIR:-$SIGNALS/hmm_signals}
RULE=hmm_v2/FIT_SELECTION_RULE.md
(( SLURM_ARRAY_TASK_ID >= 1 && SLURM_ARRAY_TASK_ID <= 10 )) \
    || { echo "The rule has ten fits (tasks 1-10); task $SLURM_ARRAY_TASK_ID refused."; exit 1; }
FIT=$(printf 'fit_%02d' "$SLURM_ARRAY_TASK_ID")
[[ -d "$DATA" ]] || { echo "No signal folder $DATA (set HMM_OUT_BASE or HMM_DATA_DIR)"; exit 1; }
[[ -f "$RULE" ]] || { echo "No $RULE: the rule must exist before fitting."; exit 1; }
mkdir -p "$FITS"
# Only the toolbox path may come from the caller; every other HMM_* setting
# (e.g. HMM_MODEL_DIR, HMM_INITREP) takes hmm_config.m's default, so each fit
# is the model in the rule and lands in its own folder.
for v in $(compgen -e | grep '^HMM_'); do
    [[ $v == HMM_TOOLBOX ]] || unset "$v"
done

# HMM-MAR seeds from the clock: stagger the starts so no two tasks begin together
# (select_reported_fit.m also counts identical fits once).
sleep $(( (SLURM_ARRAY_TASK_ID - 1) * 30 ))

# The rule is fixed before fitting: the first task records its SHA-256; every
# task (and select_reported_fit.m) stops if the file has changed since.
hash=$(sha256sum "$RULE" | cut -d' ' -f1)
( set -o noclobber; printf 'sha256: %s\nrecorded: %s\nfile: %s\n' "$hash" "$(date -Iseconds)" "$RULE" \
    > "$FITS/rule_sha256.txt" ) 2>/dev/null
recorded=$(sed -n 's/^sha256: //p' "$FITS/rule_sha256.txt")
[[ "$recorded" == "$hash" ]] || { echo "$RULE changed after it was recorded ($recorded); stop."; exit 1; }

# The fit belongs to this task only; an existing fit is never replaced.
export HMM_K=4 HMM_LOWRANK=15 HMM_PCA=0
export HMM_DATA_DIR=$DATA
export HMM_OUT_BASE=$FITS/$FIT
if [[ -f "$HMM_OUT_BASE/hmmmar_simple_k4_hmmpca15/raw_hmm_state_fc.mat" ]]; then   # run4e's last data file
    echo "$FIT already exists; not refitted."; exit 0
fi
echo "Start $(date); $FIT -> $HMM_OUT_BASE; rule $hash"

module purge
module load matlab
matlab -nodisplay -nosplash -singleCompThread -batch "addpath('hmm_v2'); run4e_hmm_mar; quit force" \
    || { echo "run4e failed for $FIT."; exit 1; }
echo "End $(date)"
