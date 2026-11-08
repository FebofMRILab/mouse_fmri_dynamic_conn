#!/bin/bash
# Test of roi_signal_homogeneity.py on a synthetic RABIES-like tree with known answers.
# Run from dfc/tests/roi_homogeneity:  bash run_test.sh [python]
set -e
PY=${1:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(mktemp -d)
$PY "$HERE/sim.py" "$W"
cd "$W"
PYTHONPATH="$HERE/../.." $PY "$HERE/../../roi_signal_homogeneity.py" --meta meta_data_with_bids.csv \
  --atlas_cs slice_maps/atlas_commonspace.nii.gz \
  --cleaned_glob "confound_correction_outputs/confound_correction_datasink/cleaned_timeseries/**/*sub-{id}_*.nii*" \
  --roi_labels roi_labels.csv --eig_dir hmm_signals_eig --signals_dir hmm_signals \
  --n_perm 2000 --out_dir roi_homogeneity > run.log
$PY "$HERE/verify.py" "$W" "$W/roi_homogeneity"
echo "outputs: $W/roi_homogeneity"
