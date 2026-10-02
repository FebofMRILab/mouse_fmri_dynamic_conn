#!/bin/bash
# End-to-end check on the first 6 subjects with 2 surrogates per null.
# (The loader will warn that more signal files exist: expected here.)
# Output: dfc_dcc_cpca_quicktest/ ; the log prints minutes per subject.
#SBATCH --job-name=dfc_quicktest
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load matlab
export DFC_QUICK_TEST=1
export DFC_N_SUBJECTS=6
unset DFC_N_CHUNKS
echo "Start $(date)"
matlab -nodisplay -nosplash -singleCompThread -batch "run_dcc_cpca_null"
echo "End $(date)"
