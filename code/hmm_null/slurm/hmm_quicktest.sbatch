#!/bin/bash
# Quick end-to-end test: 2 splits, 2 null, K = 2:4, short fits. Serial, 1 core,
# no Parallel Computing Toolbox. Checks the setup and times one fit.
#SBATCH --job-name=hmm_quicktest
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load matlab

export HMM_QUICK_TEST=1
export HMM_NO_POOL=1
echo "Start $(date)  node $(hostname)"
matlab -nodisplay -nosplash -singleCompThread -batch "run4b_hmm_splithalf_null"
echo "End $(date)"
