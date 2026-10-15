#!/bin/bash
# Static-FC code tests: equivalence checks S1-S9 and the type I error
# simulations; then, if present, the HMM Python scripts (hmm_py/) and the DFC
# plotting statistics (dfc/). One core; about 15-30 min. No data needed (all synthetic).
#
# Submit from the folder that contains static_v2/ (unzipped static_fc_published.zip),
# after running setup_python_env.sh once:
#   mkdir -p logs && sbatch hipergator_tests/run_static_tests.sbatch
#SBATCH --job-name=static_fc_tests
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV=${ENV:-/blue/febo/$USER/envs/fc_static}
OUT=$PWD/static_test_results
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

module purge
module load conda
PY="conda run --no-capture-output -p $ENV python"
echo "Start $(date)"
$PY -c "import scipy, statsmodels, networkx, bct, numpy; print('scipy', scipy.__version__, \
'| numpy', numpy.__version__, '| statsmodels', statsmodels.__version__, '| networkx', networkx.__version__)" \
    | tee "$OUT/versions.txt"

cd static_v2
echo "== 1. Equivalence checks S1-S9"
$PY tests/test_static_published.py 2>&1 | grep -v "^permutation \|^max component\|^estimating" \
    | tee "$OUT/test_static_published_output.txt"
echo "== 2. Type I error simulations"
$PY tests/simulate_type1_two_groups.py | tee "$OUT/simulate_type1_two_groups_output.txt"
$PY tests/simulate_type1_seven_groups.py | tee "$OUT/simulate_type1_seven_groups_output.txt"
echo "== 2b. ROI table (roi_group_table.py)"
$PY tests/test_roi_group_table.py | tee "$OUT/test_roi_group_table_output.txt"
cd ..
if [[ -d hmm_py ]]; then
    echo "== 3. HMM statistics and figure scripts (updated vs previous, simulated data)"
    (cd hmm_py && $PY tests/test_hmm_python_stats.py "$OUT/hmm_py_work") 2>&1 \
        | tail -40 | tee "$OUT/test_hmm_python_stats_output.txt"
fi
if [[ -d dfc ]]; then
    echo "== 4. DFC plotting group statistics (updated vs previous)"
    (cd dfc && $PY tests/test_dfc_group_stats.py) 2>&1 | tee "$OUT/test_dfc_group_stats_output.txt"
fi
echo "End $(date). Send back the folder $OUT."
