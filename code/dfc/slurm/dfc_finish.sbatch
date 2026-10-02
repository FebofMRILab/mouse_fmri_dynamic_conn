#!/bin/bash
# Loads all subject checkpoints (fits any missing subject), writes the outputs.
#SBATCH --job-name=dfc_finish
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
unset DFC_N_CHUNKS
echo "Start $(date)"
matlab -nodisplay -nosplash -singleCompThread -batch "run_dcc_cpca_null; quit force"
echo "End $(date)"
