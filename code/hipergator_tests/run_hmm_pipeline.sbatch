#!/bin/bash
# HMM pipeline for the reported model (hmm_config.m: HMM-PCA, lowrank 15, K 4):
#   run4e (fit) -> comparison with the confirmatory full fit -> drift test
#   -> run4d (tables) -> run5 (graph metrics) -> run5B (bootstrap stability)
#   -> the Python statistics and figures in hmm_py/, if that folder is present
# One core, 8 GB; the fit takes about 1-3 h and run5B about 1 h.
#
# Submit from code/ (/blue/febo/share/Abeta_Age/rabies/code). The data folder is
# hmm_config.m's default, /blue/febo/share/Abeta_Age/rabies/signals
# (hmm_signals/ in, model folders out); set HMM_OUT_BASE to use another.
#   mkdir -p logs && sbatch hipergator_tests/run_hmm_pipeline.sbatch
# Any hmm_config.m setting can be overridden at submission, e.g.
#   sbatch --export=ALL,HMM_LOWRANK=25 hipergator_tests/run_hmm_pipeline.sbatch
#SBATCH --job-name=hmm_pipeline
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ -z "${HMM_BCT:-}" ]]; then        # BCT folder (threshold_proportional.m), for run5 and run5B
    f=$(find /blue/febo/MATLAB -name threshold_proportional.m -print -quit 2>/dev/null)
    export HMM_BCT=${f:+$(dirname "$f")}
fi
ENV=${ENV:-/blue/febo/$USER/envs/fc_static}
M="matlab -nodisplay -nosplash -singleCompThread -batch"
SIGNALS=${HMM_OUT_BASE:-/blue/febo/share/Abeta_Age/rabies/signals}
[[ -d "$SIGNALS/hmm_signals" ]] || { echo "No hmm_signals/ in $SIGNALS (set HMM_OUT_BASE)"; exit 1; }
echo "Start $(date); data: $SIGNALS; K = ${HMM_K:-4 (default)}; BCT: ${HMM_BCT:-not found}"

module purge
module load matlab
$M "addpath('hmm_v2'); run4e_hmm_mar; quit force" || { echo "run4e failed; stop."; exit 1; }
$M "addpath('hmm_v2'); addpath('hmm_v2/tests'); check_against_confirmatory(); quit force" \
    || echo "Comparison with the confirmatory fit not run (file missing?); continuing."

module purge
module load conda
conda run --no-capture-output -p "$ENV" python hmm_v2/hmm_occupancy_drift.py \
    || echo "Drift test failed (see above); continuing."

module purge
module load matlab
$M "addpath('hmm_v2'); run4d_hmm_mar_plots_analysis; quit force" || echo "run4d failed (see above)."
$M "addpath('hmm_v2'); run5_analyze_raw_hmm_graph_metrics; quit force" || { echo "run5 failed; stop."; exit 1; }
$M "addpath('hmm_v2'); run5B_hmm_block_bootstrap_stability; quit force" || echo "run5B failed (see above)."
if [[ -d hmm_py ]]; then                 # Python statistics and figures, same fit
    module purge
    module load conda
    for f in hmm_dynamic_stats.py dynamic_fc_plotting_RtoPythonmigration.py \
             brainnetwork_plotter_Rmigration_bin_transitivity.py hmm_bootstrap_stability_plotter.py; do
        MPLBACKEND=Agg conda run --no-capture-output -p "$ENV" python "hmm_py/$f" \
            || echo "$f failed (see above)."
    done
fi
echo "End $(date)"
