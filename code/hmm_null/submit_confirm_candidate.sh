#!/bin/bash
# Calibrated confirmation of the step 5 screen candidate (MODEL_SELECTION_RULE.md
# item 4, D32): step 6's design for the candidate's model at its K cells.
# The model and K come from screen_candidate.txt (select_screen_candidate.py);
# nothing is chosen by hand. 32 chunk tasks (max 15 at once), then 2 finish jobs.
#
# Usage (from signals/):
#   bash submit_confirm_candidate.sh [hmm_raw_timeseries_null/figures/screen_candidate.txt]
#   HMM_NULL=iaaft bash submit_confirm_candidate.sh [...]   # descriptive; after the phase run has ended
# Then: python analyze_confirm_lowrank.py --root hmm_confirm_candidate [--null iaaft]
set -euo pipefail
mkdir -p logs
f=${1:-hmm_raw_timeseries_null/figures/screen_candidate.txt}
[[ -f $f ]] || { echo "No $f: run select_screen_candidate.py first." >&2; exit 1; }
if grep -q '^NO LEAD' "$f"; then
    echo "The screen has no lead: no candidate test (rule item 5)." >&2; exit 1
fi
line=$(grep '^CONFIRMATION' "$f" || true)
if [[ $line == *"step 6 (already run"* ]]; then
    echo "The candidate is covered by step 6: no new test (rule item 4)." >&2; exit 1
fi
re='^CONFIRMATION  new calibrated test of ([a-z0-9_]+) at K \[([0-9, ]+)\]'
[[ $line =~ $re ]] || { echo "Could not read the candidate from $f" >&2; exit 1; }
model=${BASH_REMATCH[1]}
kcells=${BASH_REMATCH[2]//,/}
export HMM_NULL=${HMM_NULL:-phase}
out=hmm_confirm_candidate
if [[ $HMM_NULL == phase && -e $out ]]; then
    echo "$out/ exists: its checkpoints would be reused (E48). Rename it first." >&2; exit 1
fi
if [[ $HMM_NULL == iaaft && ! -f $out/PREREGISTRATION.txt ]]; then
    echo "Run the phase test first: IAAFT reuses its observed fits." >&2; exit 1
fi
for v in $(compgen -e); do                       # nothing else from the calling shell
    if [[ $v == HMM_* && $v != HMM_TOOLBOX && $v != HMM_NULL ]]; then unset "$v"; fi
done
export HMM_CAND_MODEL=$model HMM_CAND_K="$kcells" HMM_OUT_DIR=$out
echo "Candidate $model, K $kcells (from $f); null $HMM_NULL"
arr=$(sbatch --parsable --export=ALL slurm/confirm_candidate_chunks.sbatch)
echo "chunks: job $arr"
for arm in observed replica; do
    fin=$(sbatch --parsable --dependency=afterany:${arr} --export=ALL,HMM_CONFIG=1,HMM_ARM=$arm \
          slurm/confirm_candidate_finish.sbatch)
    echo "finish: $arm arm -> job $fin"
done
echo "When the finish jobs are done:  python analyze_confirm_lowrank.py --root $out$( [[ $HMM_NULL == iaaft ]] && echo ' --null iaaft')"
