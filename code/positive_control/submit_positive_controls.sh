#!/bin/bash
# Positive control: run the UNCHANGED DFC and PCA-HMM pipelines on the
# simulated cohorts written by sim_positive_controls.py.
#
# Run from the folder that holds both packages (run_dcc_cpca_null.m,
# run4c_confirm_lowrank.m, slurm/dfc_*.sbatch, slurm/confirm_*.sbatch), as for
# the mouse runs.
#
#   bash submit_positive_controls.sh grid    # DFC on every grid cohort (A + B/C grid)
#   bash submit_positive_controls.sh final   # DFC on B, C, C_strong; HMM on A, B, C, C_strong
#   bash submit_positive_controls.sh fixed   # DFC + HMM on the --stage fixed cohorts
#
# Environment options
#   SIM_DIR=sim_positive_control   where the cohorts are
#   AFTER=<jobid>[:<jobid>...]     wait for these jobs first (e.g. the confirm run's
#                                  finish jobs, so the core budget is not exceeded)
#   HMM_CONFIGS="1 2"              PCA-HMM models (1 = lowrank 15, 2 = lowrank 25);
#                                  both by default so the Holm family matches the mouse run
#   DRY_RUN=1                      print the sbatch commands, submit nothing
#
# Outputs: dfc_sim_<cohort>/ and hmm_confirm_sim_<cohort>/ next to the mouse outputs.
#
# Core budget (<= 16): the cohorts run one after another. Each array runs at most
# 12 tasks and starts only when the previous array has ended and the finish jobs
# of the array before that have ended, so at most 12 + 4 finish jobs = 16 cores.
set -euo pipefail
stage=${1:?usage: bash submit_positive_controls.sh grid|final|fixed}
SIM_DIR=${SIM_DIR:-sim_positive_control}
HMM_CONFIGS=${HMM_CONFIGS:-"1 2"}
MAXRUN=12
mkdir -p logs

list="$SIM_DIR/cohorts_${stage}.txt"
[[ -f "$list" ]] || { echo "No $list: run sim_positive_controls.py --stage $stage first."; exit 1; }
for f in run_dcc_cpca_null.m run4c_confirm_lowrank.m slurm/dfc_chunks.sbatch \
         slurm/dfc_finish.sbatch slurm/confirm_chunks.sbatch slurm/confirm_finish.sbatch; do
    [[ -f "$f" ]] || { echo "Missing $f: run from the folder with both packages."; exit 1; }
done
SIM_ABS=$(cd "$SIM_DIR" && pwd)
mapfile -t cohorts < <(grep -v '^\s*$' "$list")

n_mice() { find "$SIM_ABS/$1" -maxdepth 1 -name 'signal*.txt' 2>/dev/null | wc -l; }

fake_ctr=$(mktemp)
echo 100000 > "$fake_ctr"
trap 'rm -f "$fake_ctr"' EXIT
submit() {   # prints the job id (runs in a subshell, hence the counter file)
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        local id=$(( $(cat "$fake_ctr") + 1 ))
        echo $id > "$fake_ctr"
        echo "[$id] sbatch $*" >&2
        echo $id
    else
        sbatch --parsable "$@"
    fi
}
dep_opt() {  # joins job ids into an afterany dependency (or nothing)
    local ids
    ids=$(echo "$@" | tr ' ' '\n' | grep -v '^$' | paste -sd: -)
    [[ -n "$ids" ]] && echo "--dependency=afterany:$ids"
}

prev_array="${AFTER:-}"
prev_fin=""
prevprev_fin=""
step() {     # step <array job id> <finish job ids...>: advance the chain
    prevprev_fin="$prev_fin"
    prev_array="$1"
    shift
    prev_fin="$*"
}

dfc_cohort() {
    local c=$1 n
    n=$(n_mice "$c")
    [[ $n -ge 4 ]] || { echo "Cohort $c: only $n mice in $SIM_ABS/$c"; exit 1; }
    local env="ALL,DFC_DATA_DIR=$SIM_ABS/$c,DFC_N_SUBJECTS=$n,DFC_OUT_DIR=dfc_sim_$c"
    local arr fin
    arr=$(submit $(dep_opt $prev_array $prevprev_fin) --array=1-32%$MAXRUN \
          --job-name=pc_dfc_$c --export=$env slurm/dfc_chunks.sbatch)
    fin=$(submit --dependency=afterany:$arr --job-name=pc_dfcfin_$c --export=$env \
          slurm/dfc_finish.sbatch)
    echo "DFC  $c ($n mice): chunks $arr, finish $fin -> dfc_sim_$c/"
    step "$arr" "$fin"
}

hmm_cohort() {
    local c=$1 n
    n=$(n_mice "$c")
    local base="ALL,HMM_DATA_DIR=$SIM_ABS/$c,HMM_N_SUBJECTS=$n,HMM_OUT_DIR=hmm_confirm_sim_$c,HMM_NULL=phase"
    local ranges=() cfg
    for cfg in $HMM_CONFIGS; do
        case $cfg in 1) ranges+=("1-32") ;; 2) ranges+=("33-64") ;;
            *) echo "HMM_CONFIGS: 1 and/or 2"; exit 1 ;; esac
    done
    local arr fins="" f arm skip
    arr=$(submit $(dep_opt $prev_array $prevprev_fin) --array=$(IFS=,; echo "${ranges[*]}")%$MAXRUN \
          --job-name=pc_hmm_$c --export=$base slurm/confirm_chunks.sbatch)
    for cfg in $HMM_CONFIGS; do
        for arm in observed replica; do
            # full fits (state occupancy, and state recovery against the truth) only for
            # lowrank 15, observed arm
            skip=1
            if [[ $cfg == 1 && $arm == observed ]]; then skip=0; fi
            f=$(submit --dependency=afterany:$arr --job-name=pc_hmmfin_$c \
                --export=$base,HMM_CONFIG=$cfg,HMM_ARM=$arm,HMM_SKIP_FULLFIT=$skip \
                slurm/confirm_finish.sbatch)
            fins="$fins $f"
        done
    done
    echo "HMM  $c ($n mice): chunks $arr, finish$fins -> hmm_confirm_sim_$c/"
    step "$arr" $fins
}

case $stage in
    grid)
        for c in "${cohorts[@]}"; do dfc_cohort "$c"; done
        echo
        echo "When all jobs have ended:  python sim_positive_controls.py --stage final"
        ;;
    final)
        for c in "${cohorts[@]}"; do dfc_cohort "$c"; done
        [[ -d "$SIM_ABS/A" ]] || { echo "No cohort A in $SIM_ABS (grid stage)."; exit 1; }
        for c in A "${cohorts[@]}"; do hmm_cohort "$c"; done
        echo
        echo "When all jobs have ended:  python summarize_positive_controls.py"
        ;;
    fixed)
        for c in "${cohorts[@]}"; do dfc_cohort "$c"; done
        for c in "${cohorts[@]}"; do hmm_cohort "$c"; done
        echo
        echo "When all jobs have ended:  python summarize_positive_controls.py"
        ;;
    *) echo "stage must be grid, final or fixed"; exit 1 ;;
esac
