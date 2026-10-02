#!/bin/bash
# After the chunks: builds the tables for one model x arm (fits anything
# missing). The observed arm also runs the full fits for occupancy by quarter.
#SBATCH --job-name=confirm_finish
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
unset HMM_N_CHUNKS HMM_CHUNK_IDX
echo "Start $(date)  finishing config ${HMM_CONFIG}  arm ${HMM_ARM}"
matlab -nodisplay -nosplash -singleCompThread -batch "run4c_confirm_lowrank; quit force"
echo "End $(date)"
