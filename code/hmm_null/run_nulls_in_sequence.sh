#!/bin/bash
# Phase null first, then IAAFT, one stage after another, without waiting at the terminal.
#
# Each stage is submitted by a short 1-core launcher job that starts only when every job of
# the previous stage has ended (afterany), so:
#   - an IAAFT run starts after its phase run has written the observed-data fits, which it
#     reuses: both nulls are compared against the same observed fits (README_hipergator.md);
#   - stages never overlap, so each submit script's 16-core limit holds.
# The submit scripts themselves are unchanged: this only calls them in order.
#
# Usage (from signals/, AFTER renaming the earlier result folders: README run order, E48):
#   bash run_nulls_in_sequence.sh split     # step 5: list models -> phase (all models) -> IAAFT -> collect
#   bash run_nulls_in_sequence.sh confirm   # step 6: phase -> IAAFT (descriptive)
#   bash run_nulls_in_sequence.sh both      # step 5 phase -> step 6 phase -> step 5 IAAFT -> step 6 IAAFT
# Step 5's IAAFT run follows README_hipergator.md (model 1, K 2-4); override with
#   IAAFT_MODELS="all" or "1 2 3", IAAFT_K="2:6".
# Progress: logs/null_sequence.log (one line per stage with its job IDs).
# Afterwards (by hand): python plot_hmm_splithalf_null.py;
#   python analyze_confirm_lowrank.py --root hmm_confirm_lowrank [--null iaaft]
set -euo pipefail
# The working folder is where the first call is made; the launcher jobs get it as an argument.
# (Not SLURM_SUBMIT_DIR: inside an Open OnDemand session that points to the OnDemand dashboard.)
WORK=$(cd "${3:-$PWD}" && pwd)
cd "$WORK"
mkdir -p logs
LOG=logs/null_sequence.log
plan=${1:?usage: bash run_nulls_in_sequence.sh split|confirm|both}
i=${2:-0}                                    # stage index (set by the launcher jobs)
IAAFT_MODELS=${IAAFT_MODELS:-1}
IAAFT_K=${IAAFT_K:-2:4}

case "$plan" in
    split)   stages=(split_list split_phase split_iaaft);  outs=(hmm_raw_timeseries_null) ;;
    confirm) stages=(confirm_phase confirm_iaaft);         outs=(hmm_confirm_lowrank) ;;
    both)    stages=(split_list split_phase confirm_phase split_iaaft confirm_iaaft)
             outs=(hmm_raw_timeseries_null hmm_confirm_lowrank) ;;
    *) echo "unknown plan '$plan': use split, confirm or both" >&2; exit 1 ;;
esac
if (( i == 0 )); then
    for f in submit_hmm_chunks.sh submit_confirm_lowrank.sh slurm/hmm_chunks.sbatch slurm/confirm_chunks.sbatch; do
        [[ -e $f ]] || { echo "Missing $f: run from signals/ (bash hipergator_tests/link_into_signals.sh)." >&2; exit 1; }
    done
    for d in "${outs[@]}"; do
        if [[ -e $d ]]; then
            echo "$d/ exists: its checkpoints would be reused (E48). Rename it first (README, run order)." >&2
            exit 1
        fi
    done
fi

run_stage() {
    # No HMM_* setting from the calling shell or an earlier stage reaches the jobs
    # (HMM_TOOLBOX, the toolbox path, is kept).
    local v out fin
    for v in $(compgen -e); do
        if [[ $v == HMM_* && $v != HMM_TOOLBOX ]]; then unset "$v"; fi
    done
    case "$1" in
        split_list)
            echo "listconfigs job $(sbatch --parsable slurm/hmm_listconfigs.sbatch)" ;;
        split_phase)
            HMM_NULL=phase bash submit_hmm_chunks.sh all ;;
        split_iaaft)
            out=$(HMM_NULL=iaaft HMM_K_VALUES="$IAAFT_K" bash submit_hmm_chunks.sh $IAAFT_MODELS)
            echo "$out"
            if ! grep -q "collect job" <<< "$out"; then        # one model: the script adds no collect
                fin=$(grep -oE 'finish job [0-9]+' <<< "$out" | awk '{print $3}' | paste -sd: -)
                echo "collect job $(sbatch --parsable --dependency=afterany:$fin slurm/hmm_collect.sbatch)"
            fi ;;
        confirm_phase)
            HMM_NULL=phase bash submit_confirm_lowrank.sh ;;
        confirm_iaaft)
            HMM_NULL=iaaft bash submit_confirm_lowrank.sh ;;
    esac
}

stage=${stages[$i]}
out=$(run_stage "$stage")
echo "$out"
ids=$(grep -oE 'job [0-9]+' <<< "$out" | awk '{print $2}' | paste -sd: -)
[[ -n $ids ]] || { echo "$(date '+%F %T')  $stage: no job IDs found; stop" | tee -a "$LOG" >&2; exit 1; }
echo "$(date '+%F %T')  stage $((i + 1))/${#stages[@]} $stage: jobs $ids" >> "$LOG"

next=$((i + 1))
if (( next < ${#stages[@]} )); then
    l=$(sbatch --parsable --account=febo --qos=febo --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1gb \
        --time=00:20:00 --job-name=null_seq --output=logs/null_seq_%j.out \
        --dependency=afterany:$ids \
        --export=ALL,IAAFT_MODELS="$IAAFT_MODELS",IAAFT_K="$IAAFT_K" \
        --chdir="$WORK" --wrap="bash run_nulls_in_sequence.sh $plan $next '$WORK'")
    echo "$(date '+%F %T')  launcher job $l submits ${stages[$next]} when these end" >> "$LOG"
    echo "Next stage (${stages[$next]}) is submitted by job $l when the jobs above have ended."
else
    echo "$(date '+%F %T')  last stage submitted" >> "$LOG"
fi
