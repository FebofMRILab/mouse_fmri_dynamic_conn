#!/bin/bash
# Confirmatory PCA-HMM run (see run4c_confirm_lowrank.m for the design).
# 64 chunk tasks (max 15 at once), then 4 finish jobs (1 core each): at most
# 16 cores in use at any time.
#
# Usage (from the folder that contains hmm_signals/):
#   sbatch slurm/confirm_quicktest.sbatch     # optional setup check first
#   bash submit_confirm_lowrank.sh            # phase null (the decision null)
#   HMM_NULL=iaaft bash submit_confirm_lowrank.sh   # optional, descriptive
set -euo pipefail
mkdir -p logs
export HMM_NULL=${HMM_NULL:-phase}
arr=$(sbatch --parsable --export=ALL,HMM_NULL=$HMM_NULL slurm/confirm_chunks.sbatch)
echo "chunks: job $arr (null: $HMM_NULL)"
for cfg in 1 2; do
    for arm in observed replica; do
        fin=$(sbatch --parsable --dependency=afterany:${arr} \
            --export=ALL,HMM_CONFIG=$cfg,HMM_ARM=$arm,HMM_NULL=$HMM_NULL \
            slurm/confirm_finish.sbatch)
        echo "finish: config $cfg, $arm arm -> job $fin"
    done
done
echo "When all finish jobs are done:"
echo "  python analyze_confirm_lowrank.py --root hmm_confirm_lowrank"
