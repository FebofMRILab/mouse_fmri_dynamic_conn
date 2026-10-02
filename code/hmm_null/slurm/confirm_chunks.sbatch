#!/bin/bash
# Confirmatory PCA-HMM run: 2 models x 2 arms (observed, replica), 16 chunks
# each = 64 array tasks, at most 15 running at once (1 core, 8 GB each).
# Task t -> combo = (t-1)/16 (0..3), chunk = (t-1)%16 + 1.
#   combo 0: lowrank 15, observed   combo 1: lowrank 15, replica
#   combo 2: lowrank 25, observed   combo 3: lowrank 25, replica
# Submit with submit_confirm_lowrank.sh.
#SBATCH --job-name=confirm_chunks
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=1-00:00:00
#SBATCH --array=1-64%15
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load matlab

N_CHUNKS=16
t=$SLURM_ARRAY_TASK_ID
combo=$(( (t - 1) / N_CHUNKS ))
export HMM_CHUNK_IDX=$(( (t - 1) % N_CHUNKS + 1 ))
export HMM_N_CHUNKS=$N_CHUNKS
export HMM_CONFIG=$(( combo / 2 + 1 ))
if (( combo % 2 == 0 )); then export HMM_ARM=observed; else export HMM_ARM=replica; fi
echo "Start $(date)  config ${HMM_CONFIG}  arm ${HMM_ARM}  chunk ${HMM_CHUNK_IDX}/${N_CHUNKS}  node $(hostname)"
matlab -nodisplay -nosplash -singleCompThread -batch "run4c_confirm_lowrank; quit force"
echo "End $(date)"
