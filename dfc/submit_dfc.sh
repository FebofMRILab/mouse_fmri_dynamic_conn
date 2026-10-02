#!/bin/bash
# Submit the chunked DCC + complex PCA run, then the finish job (<= 16 cores).
# Any DFC_* variables set in your shell are passed through, e.g.
#   DFC_PCA_DIM=15 DFC_OUT_DIR=dfc_dcc_cpca_pca15 bash submit_dfc.sh
set -euo pipefail
mkdir -p logs
arr=$(sbatch --parsable --export=ALL slurm/dfc_chunks.sbatch)
fin=$(sbatch --parsable --export=ALL --dependency=afterany:${arr} slurm/dfc_finish.sbatch)
echo "chunks job $arr, finish job $fin"
