#!/bin/bash
# After a model's chunks: load all checkpoints (refit any missing), write the
# summary and choose K. For model 1 (primary) also run the final fit.
# Serial, 1 core, no Parallel Computing Toolbox.
#SBATCH --job-name=hmm_finish
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load matlab

export HMM_CONFIG=${HMM_CONFIG:-1}
export HMM_NO_POOL=1
unset HMM_N_CHUNKS
echo "Start $(date)  finishing model ${HMM_CONFIG}"
matlab -nodisplay -nosplash -singleCompThread -batch "run4b_hmm_splithalf_null"
echo "End $(date)"
