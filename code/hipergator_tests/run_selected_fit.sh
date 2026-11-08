#!/bin/bash
# Submit every step after the fit (comparison with the confirmation's full fit,
# drift test, run4d, run5, run5B, state descriptors, hmm_py) for the fit of one
# reported model chosen by select_reported_fit.m. Outputs go inside that fit's
# folder (<fits>/fit_NN/...), so nothing outside it is touched.
# Run from code/:   bash hipergator_tests/run_selected_fit.sh primary
#                   bash hipergator_tests/run_selected_fit.sh secondary
set -euo pipefail
SIGNALS=${HMM_OUT_BASE:-/blue/febo/share/Abeta_Age/rabies/signals}
FITS_OVERRIDE=${HMM_FITS_DIR:-}
# Only the toolbox paths may come from this shell; every other HMM_* setting is
# hmm_config.m's default, set from the model file, or set here, so nothing leaks
# into the chosen fit's steps.
for v in $(compgen -e | grep '^HMM_'); do
    case $v in HMM_TOOLBOX|HMM_BCT) ;; *) unset "$v" ;; esac
done
. hipergator_tests/reported_model.sh "${1:-}" || exit 1   # exports HMM_K, HMM_LOWRANK, HMM_PCA
FITS=${FITS_OVERRIDE:-$SIGNALS/$FITS_NAME}
SEL="$FITS/selection/selected_fit.txt"
[[ -f "$SEL" ]] || { echo "No $SEL: run run_select_fit.sbatch first."; exit 1; }
BASE=$(sed -n 's/^selected_base: //p' "$SEL")
MODEL=$(sed -n 's/^model_dir: //p' "$SEL")
[[ -f "$MODEL/inference_scope.csv" ]] || { echo "No inference_scope.csv in $MODEL; stop."; exit 1; }
cmp -s "$MODEL_FILE" "$FITS/model.env" \
    || { echo "$MODEL_FILE differs from the copy made before fitting ($FITS/model.env); stop."; exit 1; }
[[ "$(basename "$MODEL")" == "$MODEL_FOLDER" ]] \
    || { echo "The chosen fit ($MODEL) is not a fit of $MODEL_LABEL; stop."; exit 1; }
echo "$MODEL_LABEL"
echo "Reported fit: $(sed -n 's/^reported_fit: //p' "$SEL") ($(sed -n 's/^fit_id: //p' "$SEL" | cut -c1-12))"
sed -n 's/^recurring_states/recurring_states/p' "$SEL"
mkdir -p logs
sbatch --job-name="hmm_pipeline_${MODEL_ROLE}" \
    --export=ALL,SKIP_FIT=1,HMM_K=$HMM_K,HMM_LOWRANK=$HMM_LOWRANK,HMM_PCA=$HMM_PCA,HMM_OUT_BASE="$BASE",HMM_MODEL_DIR="$MODEL",HMM_DATA_DIR="$SIGNALS/hmm_signals",HMM_CONFIRM_GAMMA="$SIGNALS/$CONFIRM_GAMMA_REL" \
    hipergator_tests/run_hmm_pipeline.sbatch
