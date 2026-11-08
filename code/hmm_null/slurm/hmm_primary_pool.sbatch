#!/bin/bash
# ALTERNATIVE to the chunked array: one job, 16 cores, parallel pool.
# Needs a free Parallel Computing Toolbox seat. If none is free the script
# falls back to SERIAL on 1 core (the other 15 sit idle) - so when licenses
# are tight, use submit_hmm_chunks.sh instead.
#SBATCH --job-name=hmm_primary_pool
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load matlab

export HMM_CONFIG=${HMM_CONFIG:-1}
echo "Start $(date)  node $(hostname)  cpus ${SLURM_CPUS_PER_TASK}"
matlab -nodisplay -nosplash -batch "run4b_hmm_splithalf_null"
echo "End $(date)"
