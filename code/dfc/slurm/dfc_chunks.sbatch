#!/bin/bash
# DCC + complex PCA vs surrogates: subjects split over 32 array tasks,
# at most 15 running (1 core, 8 GB each; +1 core for the finish job = 16).
# Serial MATLAB, no Parallel Computing Toolbox. Submit via submit_dfc.sh.
#SBATCH --job-name=dfc_chunks
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=12:00:00
#SBATCH --array=1-32%15
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load matlab

export DFC_N_CHUNKS=32      # MUST equal the top of the --array range (keep 32
                            # when resubmitting single chunks, e.g. --array=7)
echo "Start $(date)  chunk ${SLURM_ARRAY_TASK_ID}/${DFC_N_CHUNKS}  node $(hostname)"
matlab -nodisplay -nosplash -singleCompThread -batch "run_dcc_cpca_null; quit force"
echo "End $(date)"
