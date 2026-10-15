#!/bin/bash
# Choose the reported K = 4 fit from the ten array fits, by the pre-specified rule
# (hmm_v2/FIT_SELECTION_RULE.md): lowest final free energy; states that recur
# across fits; writes <fits>/selection/ and inference_scope.csv in the chosen fit.
# Submit from code/ after run_hmm_fits_array.sbatch has finished, e.g.
#   sbatch --dependency=afterany:<array job id> hipergator_tests/run_select_fit.sbatch
# Then: bash hipergator_tests/run_selected_fit.sh  (every later step on that fit)
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
FITS=${HMM_FITS_DIR:-$SIGNALS/hmm_k4_hmmpca15_fits}
for v in $(compgen -e | grep '^HMM_'); do        # hmm_config.m defaults, as in the fits
    case $v in HMM_TOOLBOX|HMM_CONFIRM_GAMMA) ;; *) unset "$v" ;; esac
done
export HMM_K=4 HMM_LOWRANK=15 HMM_PCA=0 HMM_OUT_BASE=$SIGNALS
echo "Start $(date); fits: $FITS"
module purge
module load matlab
matlab -nodisplay -nosplash -singleCompThread -batch \
    "addpath('hmm_v2'); select_reported_fit('$FITS'); quit force" || { echo "Selection failed (see above)."; exit 1; }
echo "End $(date)"
