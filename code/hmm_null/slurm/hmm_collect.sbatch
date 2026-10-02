#!/bin/bash
# Merge all model configurations after the array finishes.
# Submit with a dependency:  sbatch --dependency=afterany:<ARRAY_JOB_ID> slurm/hmm_collect.sbatch
#SBATCH --job-name=hmm_null_collect
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load matlab
matlab -nodisplay -nosplash -batch "collect_sensitivity_null"
