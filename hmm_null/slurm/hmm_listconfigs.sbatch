#!/bin/bash
# Print the model list (= array size) without fitting anything.
#SBATCH --job-name=hmm_listconfigs
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load matlab
export HMM_LIST_CONFIGS=1
matlab -nodisplay -nosplash -batch "run4b_hmm_splithalf_null"
