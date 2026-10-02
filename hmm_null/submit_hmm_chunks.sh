#!/bin/bash
# Submit one or more model configurations as chunked arrays, never using more
# than 16 cores in total:
#   - each model's array runs at most 15 tasks at once (1 core, 8 GB each)
#   - the next model's array waits until the previous array has finished
#   - each model's 1-core finish job starts when its own array ends
#     (15 + 1 = 16 cores)
#   - if more than one model is given, collect_sensitivity_null runs at the end
#
# Usage (from the analysis folder):
#   bash submit_hmm_chunks.sh            # model 1 (primary) only
#   bash submit_hmm_chunks.sh 1 2 3      # models 1, 2, 3 in sequence
#   bash submit_hmm_chunks.sh all        # every model in model_configs.csv
set -euo pipefail
mkdir -p logs

if [[ "${1:-}" == "all" ]]; then
    if [[ ! -f hmm_raw_timeseries_null/model_configs.csv ]]; then
        echo "model_configs.csv not found: run  sbatch slurm/hmm_listconfigs.sbatch  first." >&2
        exit 1
    fi
    n=$(( $(wc -l < hmm_raw_timeseries_null/model_configs.csv) - 1 ))
    configs=$(seq 1 "$n")
else
    configs=${*:-1}
fi

prev_array=""
finish_ids=()
for c in $configs; do
    dep=""
    [[ -n "$prev_array" ]] && dep="--dependency=afterany:${prev_array}"
    arr=$(sbatch --parsable $dep --export=ALL,HMM_CONFIG=$c slurm/hmm_chunks.sbatch)
    fin=$(sbatch --parsable --dependency=afterany:${arr} --export=ALL,HMM_CONFIG=$c slurm/hmm_finish.sbatch)
    echo "model $c: chunks job $arr, finish job $fin"
    prev_array=$arr
    finish_ids+=("$fin")
done

if (( ${#finish_ids[@]} > 1 )); then
    deps=$(IFS=:; echo "${finish_ids[*]}")
    col=$(sbatch --parsable --dependency=afterany:${deps} slurm/hmm_collect.sbatch)
    echo "collect job $col (after all finish jobs)"
fi
