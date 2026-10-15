#!/bin/bash
# Submit every step after the fit (comparison with the confirmatory full fit,
# drift test, run4d, run5, run5B, hmm_py) for the fit chosen by
# select_reported_fit.m. Outputs go inside that fit's folder
# (<fits>/fit_NN/...), so the 4 Oct 2026 refit's outputs are left as they are.
# Run from code/:   bash hipergator_tests/run_selected_fit.sh
set -euo pipefail
SIGNALS=${HMM_OUT_BASE:-/blue/febo/share/Abeta_Age/rabies/signals}
FITS=${HMM_FITS_DIR:-$SIGNALS/hmm_k4_hmmpca15_fits}
SEL="$FITS/selection/selected_fit.txt"
[[ -f "$SEL" ]] || { echo "No $SEL: run run_select_fit.sbatch first."; exit 1; }
BASE=$(sed -n 's/^selected_base: //p' "$SEL")
MODEL=$(sed -n 's/^model_dir: //p' "$SEL")
[[ -f "$MODEL/inference_scope.csv" ]] || { echo "No inference_scope.csv in $MODEL; stop."; exit 1; }
echo "Reported fit: $(sed -n 's/^reported_fit: //p' "$SEL") ($(sed -n 's/^fit_id: //p' "$SEL" | cut -c1-12))"
sed -n 's/^recurring_states/recurring_states/p' "$SEL"
mkdir -p logs
# Only the toolbox paths may come from this shell; every other HMM_* setting is
# hmm_config.m's default or set here, so nothing leaks into the chosen fit's steps.
for v in $(compgen -e | grep '^HMM_'); do
    case $v in HMM_TOOLBOX|HMM_BCT) ;; *) unset "$v" ;; esac
done
sbatch --export=ALL,SKIP_FIT=1,HMM_K=4,HMM_LOWRANK=15,HMM_PCA=0,HMM_OUT_BASE="$BASE",HMM_MODEL_DIR="$MODEL",HMM_DATA_DIR="$SIGNALS/hmm_signals",HMM_CONFIRM_GAMMA="$SIGNALS/hmm_confirm_lowrank/observed/gamma_lowrank_15_K4.mat" \
    hipergator_tests/run_hmm_pipeline.sbatch
