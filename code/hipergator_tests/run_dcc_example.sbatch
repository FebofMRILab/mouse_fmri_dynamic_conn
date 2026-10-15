#!/bin/bash
# Supplement figures in natural units for the DCC/GARCH/complex-PCA results:
#   1. FigS_dfc_raw_units: every mouse's observed value vs its phase-surrogate
#      mean, per pre-specified measure (from dfc_subject_metrics.csv; minutes)
#   2. FigS_dcc_example: rho(t) for one NTG_Y mouse drawn at random, with the
#      published DCC estimator (dcc_mvgarch.m, Lindquist toolbox) on the mouse
#      and its 19 phase surrogates (needs the Optimization Toolbox; about 1 h)
# Uses the main D = 25 results in <signals>/dfc_dcc_cpca. Submit from code/:
#   mkdir -p logs && sbatch hipergator_tests/run_dcc_example.sbatch
#SBATCH --job-name=dcc_example
#SBATCH --account=febo
#SBATCH --qos=febo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset DISPLAY
SIGNALS=${HMM_OUT_BASE:-/blue/febo/share/Abeta_Age/rabies/signals}
DFC=${DFC_RESULTS:-$SIGNALS/dfc_dcc_cpca_pca15}
DCC_CODE=${DCC_CODE:-/blue/febo/MATLAB/Lindquist_Dynamic_Correlation/DCC_toolbox/DCCcode}
ENV=${ENV:-/blue/febo/$USER/envs/fc_static}
for p in "$SIGNALS/hmm_signals/meta_data.csv" "$DFC/dfc_subject_metrics.csv" "$DCC_CODE/dcc_mvgarch.m" \
         dfc/dcc_example_traces.m dfc/plot_dcc_example.py dfc/plot_dfc_raw_units.py; do
    [[ -e "$p" ]] || { echo "Not found: $p"; exit 1; }
done
echo "Start $(date); results: $DFC"

module purge
module load conda
conda run --no-capture-output -p "$ENV" env MPLBACKEND=Agg python dfc/plot_dfc_raw_units.py --in_dir "$DFC" \
    || echo "Raw-units figure failed (see above); continuing."

module purge
module load matlab
matlab -nodisplay -nosplash -singleCompThread -batch "\
    if ~license('test', 'Optimization_Toolbox'), error('Optimization Toolbox not licensed: fmincon is needed.'); end; \
    addpath('dfc'); dcc_example_traces('$SIGNALS/hmm_signals', '$DCC_CODE', '$DFC', '$DFC/dcc_example'); quit force" \
    || { echo "dcc_example_traces failed (see above)."; exit 1; }

module purge
module load conda
conda run --no-capture-output -p "$ENV" env MPLBACKEND=Agg python dfc/plot_dcc_example.py --in_dir "$DFC/dcc_example" \
    || { echo "Example figure failed (see above)."; exit 1; }
echo "End $(date). Figures: $DFC/figures/FigS_dfc_raw_units.*, $DFC/dcc_example/FigS_dcc_example.*"
