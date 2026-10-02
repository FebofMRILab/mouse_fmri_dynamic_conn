#!/bin/bash
# Setup check: tiny run of lowrank 15, both arms, including a short full fit.
#SBATCH --job-name=confirm_quicktest
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load matlab
unset HMM_N_CHUNKS HMM_CHUNK_IDX
export HMM_QUICK_TEST=1 HMM_CONFIG=1
for arm in observed replica; do
    export HMM_ARM=$arm
    echo "Quick test, arm $arm, $(date)"
    matlab -nodisplay -nosplash -singleCompThread -batch "run4c_confirm_lowrank"
done
echo "End $(date)"
