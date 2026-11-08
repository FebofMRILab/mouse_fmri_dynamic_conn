#!/bin/bash
# Ten independent fits of one reported model for the pre-specified
# fit-selection rule (hmm_v2/FIT_SELECTION_RULE.md, amendment 1 of 27 Oct 2026):
#   primary    hmm_v2/MODEL_PRIMARY.env    Gaussian HMM on 25 PCs, K = 3
#              -> <signals>/hmm_k3_pca25_fits/fit_NN/hmmmar_simple_k3_pca25/
#   secondary  hmm_v2/MODEL_SECONDARY.env  HMM-PCA rank 15, K = 2
#              -> <signals>/hmm_k2_hmmpca15_fits/fit_NN/hmmmar_simple_k2_hmmpca15/
# Each array task runs hmm_v2/run4e_hmm_mar.m unchanged, with its own output folder.
# Then: sbatch hipergator_tests/run_select_fit.sbatch <primary|secondary>.
#
# Submit from code/ (/blue/febo/share/Abeta_Age/rabies/code):
#   mkdir -p logs
#   sbatch hipergator_tests/run_hmm_fits_array.sbatch primary
#   sbatch hipergator_tests/run_hmm_fits_array.sbatch secondary
# Rerun single tasks that failed:   sbatch --array=3,7 hipergator_tests/run_hmm_fits_array.sbatch primary
# One core and 8 GB per task; at most 8 tasks of an array at once, so both
# arrays together stay within 16 cores.
#SBATCH --job-name=hmm_fits
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=1-00:00:00
#SBATCH --array=1-10%8
#SBATCH --output=logs/%x_%A_%a.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset DISPLAY
SIGNALS=${HMM_OUT_BASE:-/blue/febo/share/Abeta_Age/rabies/signals}
DATA=${HMM_DATA_DIR:-$SIGNALS/hmm_signals}
FITS_OVERRIDE=${HMM_FITS_DIR:-}
RULE=hmm_v2/FIT_SELECTION_RULE.md
(( SLURM_ARRAY_TASK_ID >= 1 && SLURM_ARRAY_TASK_ID <= 10 )) \
    || { echo "The rule has ten fits (tasks 1-10); task $SLURM_ARRAY_TASK_ID refused."; exit 1; }
FIT=$(printf 'fit_%02d' "$SLURM_ARRAY_TASK_ID")
[[ -d "$DATA" ]] || { echo "No signal folder $DATA (set HMM_OUT_BASE or HMM_DATA_DIR)"; exit 1; }
[[ -f "$RULE" ]] || { echo "No $RULE: the rule must exist before fitting."; exit 1; }
# Only the toolbox path may come from the caller; every other HMM_* setting
# (e.g. HMM_MODEL_DIR, HMM_INITREP) takes hmm_config.m's default and the model
# comes from its file, so each fit is the model in the rule and lands in its
# own folder.
for v in $(compgen -e | grep '^HMM_'); do
    [[ $v == HMM_TOOLBOX ]] || unset "$v"
done
# The model comes only from its file (exports HMM_K, HMM_LOWRANK, HMM_PCA).
. hipergator_tests/reported_model.sh "${1:-}" || exit 1
# Per role: the first fit of this role ever started records the model file in
# <signals>; any later task stops if the file has changed, even when a changed K
# or dimension would point to a new fits folder (e.g. tasks still queued after
# an edit).
ROLE_REC=$SIGNALS/hmm_fits_${MODEL_ROLE}_model.env
( set -o noclobber; cat "$MODEL_FILE" > "$ROLE_REC" ) 2>/dev/null
cmp -s "$MODEL_FILE" "$ROLE_REC" \
    || { echo "$MODEL_FILE differs from the $MODEL_ROLE model recorded when its first fit started ($ROLE_REC)."
         echo "The model changes only by a dated amendment to $RULE, after which $ROLE_REC is renamed by hand; stop."
         exit 1; }
FITS=${FITS_OVERRIDE:-$SIGNALS/$FITS_NAME}
mkdir -p "$FITS"

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
# The same for the model file, per fits folder: the copy that
# select_reported_fit.m and run_selected_fit.sh check.
( set -o noclobber; cat "$MODEL_FILE" > "$FITS/model.env" ) 2>/dev/null
cmp -s "$MODEL_FILE" "$FITS/model.env" \
    || { echo "$MODEL_FILE differs from the copy in $FITS/model.env; stop."; exit 1; }

# The fit belongs to this task only; an existing fit is never replaced.
export HMM_DATA_DIR=$DATA
export HMM_OUT_BASE=$FITS/$FIT
if [[ -f "$HMM_OUT_BASE/$MODEL_FOLDER/raw_hmm_state_fc.mat" ]]; then   # run4e's last data file
    echo "$FIT already exists; not refitted."; exit 0
fi
echo "Start $(date); $MODEL_LABEL; $FIT -> $HMM_OUT_BASE; rule $hash"

module purge
module load matlab
matlab -nodisplay -nosplash -singleCompThread -batch "addpath('hmm_v2'); run4e_hmm_mar; quit force" \
    || { echo "run4e failed for $FIT."; exit 1; }
echo "End $(date)"
