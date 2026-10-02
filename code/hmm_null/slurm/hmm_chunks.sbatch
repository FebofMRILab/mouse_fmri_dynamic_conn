#!/bin/bash
# One model's split-half + null fits, spread over array tasks.
# Each task = 1 core, 8 GB, serial MATLAB: NO Parallel Computing Toolbox seat.
# 32 chunks, at most 15 running at once (+1 core kept free for the finish job
# = 16 cores in total). Submit via submit_hmm_chunks.sh, which sets HMM_CONFIG.
#SBATCH --job-name=hmm_chunks
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=2-00:00:00
#SBATCH --array=1-32%15
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load matlab

export HMM_CONFIG=${HMM_CONFIG:-1}
export HMM_N_CHUNKS=32     # MUST equal the top of the --array range (keep 32 even when
                            # resubmitting a few failed chunks, e.g. --array=5,9)
echo "Start $(date)  model ${HMM_CONFIG}  chunk ${SLURM_ARRAY_TASK_ID}/${HMM_N_CHUNKS}  node $(hostname)"
matlab -nodisplay -nosplash -singleCompThread -batch "run4b_hmm_splithalf_null"
echo "End $(date)"
