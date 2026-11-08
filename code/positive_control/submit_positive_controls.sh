#!/bin/bash
# Positive control: run the UNCHANGED DFC and HMM pipelines on the simulated
# cohorts written by sim_positive_controls.py.
#
# Run from the folder that holds both packages (run_dcc_cpca_null.m,
# run4c_confirm_lowrank.m, run4c_confirm_candidate.m, slurm/dfc_*.sbatch,
# slurm/confirm_*.sbatch), as for the mouse runs.
#
#   bash submit_positive_controls.sh grid    # DFC on every grid cohort (A + B/C grid)
#   bash submit_positive_controls.sh final   # DFC on B, C, C_strong; HMM on A, B, C, C_strong
#   bash submit_positive_controls.sh fixed   # DFC + HMM on the --stage fixed cohorts
#
# HMM on each cohort (stages final and fixed), each family with its own Holm
# correction, as in the mouse runs:
#   step 6 cells   HMM-PCA rank 15 / 25 at K 2-4 (run4c_confirm_lowrank.m)
#                  -> hmm_confirm_sim_<cohort>/  (rank 15, K = 2 is the secondary model)
#   step 6b cells  the primary reported model: the step 5 candidate at its K cells,
#                  read from screen_candidate.txt as submit_confirm_candidate.sh reads
#                  it (Gaussian PCA-25, K 2-4), run with run4c_confirm_candidate.m
#                  -> hmm_candidate_sim_<cohort>/  (added 28 Oct 2026, D38)
#
# Environment options
#   SIM_DIR=sim_positive_control   where the cohorts are
#   AFTER=<jobid>[:<jobid>...]     wait for these jobs first (e.g. the confirm run's
#                                  finish jobs, so the core budget is not exceeded)
#   HMM_CONFIGS="1 2"              step 6 models (1 = lowrank 15, 2 = lowrank 25);
#                                  both by default so the Holm family matches the mouse run
#   HMM_PRIMARY=1                  also the primary model's cells (default); 0 = step 6 cells only
#   SCREEN_CANDIDATE=<file>        default hmm_raw_timeseries_null/figures/screen_candidate.txt
#   ALLOW_EXISTING=1               submit into output folders that already exist, to resume
#                                  after a failed job (finished checkpoints are reused).
#                                  Otherwise an existing output folder stops the script:
#                                  checkpoints are keyed on settings, not data (E48).
#   DRY_RUN=1                      print the sbatch commands, submit nothing
# No other HMM_* or DFC_* variable of the calling shell reaches the jobs (HMM_TOOLBOX
# does), so an exported setting such as DFC_PCA_DIM=15 cannot change a cohort's run.
#
# Outputs: dfc_sim_<cohort>/, hmm_confirm_sim_<cohort>/ and hmm_candidate_sim_<cohort>/
# next to the mouse outputs.
#
# Core budget (<= 16): the cohorts run one after another. Each array runs at most
# 12 tasks and starts only when the previous array has ended and the finish jobs
# of the array before that have ended, so at most 12 + 4 finish jobs = 16 cores.
set -euo pipefail
stage=${1:?usage: bash submit_positive_controls.sh grid|final|fixed}
SIM_DIR=${SIM_DIR:-sim_positive_control}
configs=${HMM_CONFIGS:-"1 2"}
primary=${HMM_PRIMARY:-1}
SCREEN_CANDIDATE=${SCREEN_CANDIDATE:-hmm_raw_timeseries_null/figures/screen_candidate.txt}
MAXRUN=12
for v in $(compgen -e); do                       # nothing else from the calling shell
    case $v in HMM_TOOLBOX) ;; HMM_*|DFC_*) unset "$v" ;; esac
done
case $stage in grid|final|fixed) ;; *) echo "stage must be grid, final or fixed"; exit 1 ;; esac
[[ $primary == 0 || $primary == 1 ]] || { echo "HMM_PRIMARY: 0 or 1"; exit 1; }
mkdir -p logs

list="$SIM_DIR/cohorts_${stage}.txt"
[[ -f "$list" ]] || { echo "No $list: run sim_positive_controls.py --stage $stage first."; exit 1; }
need=(run_dcc_cpca_null.m run4c_confirm_lowrank.m slurm/dfc_chunks.sbatch
      slurm/dfc_finish.sbatch slurm/confirm_chunks.sbatch slurm/confirm_finish.sbatch)
if [[ $stage != grid && $primary == 1 ]]; then
    need+=(run4c_confirm_candidate.m hmm_candidate_options.m hmm_candidate_K.m
           slurm/confirm_candidate_chunks.sbatch slurm/confirm_candidate_finish.sbatch)
fi
for f in "${need[@]}"; do
    [[ -f "$f" ]] || { echo "Missing $f: run from the folder with both packages."; exit 1; }
done

# The primary model's cells: the step 5 candidate, as submit_confirm_candidate.sh reads it.
if [[ $stage != grid && $primary == 1 ]]; then
    [[ -f "$SCREEN_CANDIDATE" ]] || { echo "No $SCREEN_CANDIDATE (the step 5 candidate," \
        "i.e. the primary model). HMM_PRIMARY=0 runs step 6's cells only."; exit 1; }
    line=$(grep '^CONFIRMATION' "$SCREEN_CANDIDATE" || true)
    re='^CONFIRMATION  new calibrated test of ([a-z0-9_]+) at K \[([0-9, ]+)\]'
    [[ $line =~ $re ]] || { echo "No new candidate test in $SCREEN_CANDIDATE, so no" \
        "primary-model cells to run. HMM_PRIMARY=0 runs step 6's cells only."; exit 1; }
    export HMM_CAND_MODEL=${BASH_REMATCH[1]} HMM_CAND_K="${BASH_REMATCH[2]//,/}"
    echo "Primary model's cells: $HMM_CAND_MODEL at K $HMM_CAND_K (from $SCREEN_CANDIDATE)"
fi

SIM_ABS=$(cd "$SIM_DIR" && pwd)
mapfile -t cohorts < <(grep -v '^\s*$' "$list")
hmm_cohorts=("${cohorts[@]}")
if [[ $stage == final ]]; then
    [[ -d "$SIM_ABS/A" ]] || { echo "No cohort A in $SIM_ABS (grid stage)."; exit 1; }
    hmm_cohorts=(A "${cohorts[@]}")
fi

# Output folders of this stage must be new (E48), unless resuming.
outputs=()
for c in "${cohorts[@]}"; do outputs+=("dfc_sim_$c"); done
if [[ $stage != grid ]]; then
    for c in "${hmm_cohorts[@]}"; do
        outputs+=("hmm_confirm_sim_$c")
        if [[ $primary == 1 ]]; then outputs+=("hmm_candidate_sim_$c"); fi
    done
fi
if [[ "${ALLOW_EXISTING:-0}" != "1" ]]; then
    found=()
    for o in "${outputs[@]}"; do
        if [[ -e "$o" ]]; then found+=("$o"); fi
    done
    if (( ${#found[@]} )); then
        echo "These output folders already exist: ${found[*]}"
        echo "Their checkpoints are keyed on settings, not data, so this run would reuse them (E48)."
        echo "Rename them first, or set ALLOW_EXISTING=1 to resume a run that stopped."
        exit 1
    fi
fi

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
    for cfg in $configs; do
        case $cfg in 1) ranges+=("1-32") ;; 2) ranges+=("33-64") ;;
            *) echo "HMM_CONFIGS: 1 and/or 2"; exit 1 ;; esac
    done
    local arr fins="" f arm skip
    arr=$(submit $(dep_opt $prev_array $prevprev_fin) --array=$(IFS=,; echo "${ranges[*]}")%$MAXRUN \
          --job-name=pc_hmm_$c --export=$base slurm/confirm_chunks.sbatch)
    for cfg in $configs; do
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

hmm_primary_cohort() {   # the primary model's cells: step 6b's runner, unchanged
    local c=$1 n
    n=$(n_mice "$c")
    # HMM_CAND_MODEL and HMM_CAND_K reach the jobs through ALL (exported above)
    local base="ALL,HMM_DATA_DIR=$SIM_ABS/$c,HMM_N_SUBJECTS=$n,HMM_OUT_DIR=hmm_candidate_sim_$c,HMM_NULL=phase"
    local arr fins="" f arm skip
    arr=$(submit $(dep_opt $prev_array $prevprev_fin) --array=1-32%$MAXRUN \
          --job-name=pc_cand_$c --export=$base slurm/confirm_candidate_chunks.sbatch)
    for arm in observed replica; do
        # full fits (state occupancy, and state recovery against the truth): observed arm
        skip=1
        if [[ $arm == observed ]]; then skip=0; fi
        f=$(submit --dependency=afterany:$arr --job-name=pc_candfin_$c \
            --export=$base,HMM_CONFIG=1,HMM_ARM=$arm,HMM_SKIP_FULLFIT=$skip \
            slurm/confirm_candidate_finish.sbatch)
        fins="$fins $f"
    done
    echo "HMM  $c ($n mice), primary model ($HMM_CAND_MODEL, K $HMM_CAND_K): chunks $arr," \
         "finish$fins -> hmm_candidate_sim_$c/"
    step "$arr" $fins
}

case $stage in
    grid)
        for c in "${cohorts[@]}"; do dfc_cohort "$c"; done
        echo
        echo "When all jobs have ended:  python sim_positive_controls.py --stage final"
        ;;
    final|fixed)
        for c in "${cohorts[@]}"; do dfc_cohort "$c"; done
        for c in "${hmm_cohorts[@]}"; do hmm_cohort "$c"; done
        if [[ $primary == 1 ]]; then
            for c in "${hmm_cohorts[@]}"; do hmm_primary_cohort "$c"; done
        fi
        echo
        echo "When all jobs have ended:  python summarize_positive_controls.py"
        ;;
esac