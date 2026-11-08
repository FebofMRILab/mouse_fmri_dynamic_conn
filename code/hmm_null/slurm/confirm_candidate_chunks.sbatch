#!/bin/bash
# Candidate confirmation (MODEL_SELECTION_RULE.md item 4): 1 model x 2 arms, 16 chunks
# each = 32 array tasks, at most 15 running at once (1 core, 8 GB each).
# Task t -> combo = (t-1)/16 (0 observed, 1 replica), chunk = (t-1)%16 + 1.
# Submit with submit_confirm_candidate.sh (it sets HMM_CAND_MODEL and HMM_CAND_K).
#SBATCH --job-name=cand_chunks
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=1-00:00:00
#SBATCH --array=1-32%15
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
export HMM_CONFIG=1
if (( combo == 0 )); then export HMM_ARM=observed; else export HMM_ARM=replica; fi
echo "Start $(date)  model ${HMM_CAND_MODEL}  K ${HMM_CAND_K}  arm ${HMM_ARM}  chunk ${HMM_CHUNK_IDX}/${N_CHUNKS}  node $(hostname)"
matlab -nodisplay -nosplash -singleCompThread -batch "run4c_confirm_candidate; quit force"
echo "End $(date)"
