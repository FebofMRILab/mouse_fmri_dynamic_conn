#!/bin/bash
# Choose the reported fit of one model (primary or secondary) from its ten array
# fits, by the pre-specified rule (hmm_v2/FIT_SELECTION_RULE.md): lowest final
# free energy; states that recur across fits; writes <fits>/selection/ and
# inference_scope.csv in the chosen fit.
# Submit from code/ after that model's run_hmm_fits_array.sbatch has finished, e.g.
#   sbatch --dependency=afterany:<array job id> hipergator_tests/run_select_fit.sbatch primary
# Then: bash hipergator_tests/run_selected_fit.sh primary  (every later step on that fit)
#SBATCH --job-name=hmm_select_fit
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
unset DISPLAY
SIGNALS=${HMM_OUT_BASE:-/blue/febo/share/Abeta_Age/rabies/signals}
FITS_OVERRIDE=${HMM_FITS_DIR:-}
for v in $(compgen -e | grep '^HMM_'); do        # hmm_config.m defaults, as in the fits
    case $v in HMM_TOOLBOX|HMM_CONFIRM_GAMMA) ;; *) unset "$v" ;; esac
done
. hipergator_tests/reported_model.sh "${1:-}" || exit 1   # exports HMM_K, HMM_LOWRANK, HMM_PCA
FITS=${FITS_OVERRIDE:-$SIGNALS/$FITS_NAME}
[[ -f "$FITS/model.env" ]] || { echo "No $FITS/model.env: the array job copies the model file before fitting."; exit 1; }
cmp -s "$MODEL_FILE" "$FITS/model.env" \
    || { echo "$MODEL_FILE differs from the copy made before fitting ($FITS/model.env); stop."; exit 1; }
export HMM_OUT_BASE=$SIGNALS
echo "Start $(date); $MODEL_LABEL; fits: $FITS"
module purge
module load matlab
matlab -nodisplay -nosplash -singleCompThread -batch \
    "addpath('hmm_v2'); select_reported_fit('$FITS'); quit force" || { echo "Selection failed (see above)."; exit 1; }
echo "End $(date)"
