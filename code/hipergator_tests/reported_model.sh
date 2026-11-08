#!/bin/bash
# Load one of the two reported HMMs from its model file, for the job scripts.
# Source it from code/ after clearing the caller's HMM_* settings:
#   . hipergator_tests/reported_model.sh primary     # hmm_v2/MODEL_PRIMARY.env
#   . hipergator_tests/reported_model.sh secondary   # hmm_v2/MODEL_SECONDARY.env
# (FIT_SELECTION_RULE.md amendment 1, 27 Oct 2026; provenance D33)
#
# Exports HMM_K, HMM_LOWRANK, HMM_PCA (read by hmm_config.m and hmm_py) and sets
#   MODEL_ROLE     primary | secondary
#   MODEL_FILE     hmm_v2/MODEL_<ROLE>.env
#   MODEL_LABEL    text for logs
#   MODEL_FOLDER   run4e's folder, as hmm_config.m names it
#   MODEL_TAG      hmm_py's tag (hmm_dynamics_stats_<tag>/, hmm_dynamics_figures_<tag>/)
#   TABLE_SUFFIX   run4d's table suffix (all_K_all_PCA or all_K_all_lowrank)
#   FITS_NAME      folder of the ten fits under <signals>/
#   CONFIRM_GAMMA_REL  full fit of the model's confirmation (step 6 or 6b), under <signals>/
# Stops the caller (return 1) on a missing or malformed file.

reported_model_load() {
    local role=${1:-}
    case "$role" in
        primary|secondary) ;;
        *) echo "Model must be 'primary' or 'secondary', not '${role}'." >&2; return 1 ;;
    esac
    local file="hmm_v2/MODEL_${role^^}.env"
    [[ -f "$file" ]] || { echo "No $file (run from code/)." >&2; return 1; }
    local line name value k="" r="" p=""
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        [[ -z "${line//[[:space:]]/}" || "$line" =~ ^[[:space:]]*# ]] && continue
        if [[ "$line" =~ ^(HMM_K|HMM_LOWRANK|HMM_PCA)=([0-9]+)$ ]]; then
            name=${BASH_REMATCH[1]}; value=${BASH_REMATCH[2]}
            case $name in HMM_K) k=$value ;; HMM_LOWRANK) r=$value ;; HMM_PCA) p=$value ;; esac
        else
            echo "Unreadable line in $file: '$line'" >&2; return 1
        fi
    done < "$file"
    [[ -n "$k" && -n "$r" && -n "$p" ]] || { echo "$file must set HMM_K, HMM_LOWRANK and HMM_PCA." >&2; return 1; }
    (( r > 0 && p > 0 )) && { echo "$file sets both HMM_LOWRANK and HMM_PCA." >&2; return 1; }

    export HMM_K=$k HMM_LOWRANK=$r HMM_PCA=$p
    MODEL_ROLE=$role
    MODEL_FILE=$file
    if (( r > 0 )); then
        MODEL_FOLDER=hmmmar_simple_k${k}_hmmpca${r}
        MODEL_TAG=hmmpca${r}_k${k}
        TABLE_SUFFIX=all_K_all_lowrank
        MODEL_LABEL="HMM-PCA (lowrank ${r}), K = ${k}"
        CONFIRM_GAMMA_REL=hmm_confirm_lowrank/observed/gamma_lowrank_${r}_K${k}.mat
    else
        MODEL_FOLDER=hmmmar_simple_k${k}_pca${p}
        MODEL_TAG=pca${p}_k${k}
        TABLE_SUFFIX=all_K_all_PCA
        if (( p > 0 )); then
            MODEL_LABEL="Gaussian HMM on ${p} PCs (options.pca), K = ${k}"
            CONFIRM_GAMMA_REL=hmm_confirm_candidate/observed/gamma_gaussian_pca_${p}_K${k}.mat
        else
            MODEL_LABEL="Gaussian HMM, all ROIs, K = ${k}"
            CONFIRM_GAMMA_REL=hmm_confirm_candidate/observed/gamma_gaussian_full_K${k}.mat
        fi
    fi
    FITS_NAME=${MODEL_FOLDER/hmmmar_simple_/hmm_}_fits
    MODEL_LABEL="$MODEL_LABEL [$role model]"
}

reported_model_load "${1:-}"
