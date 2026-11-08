#!/bin/bash
# Test of make_slice_maps.py + slice_timing_lag_check.py --fractions on a
# synthetic RABIES 0.6.1 output tree (~6 min; needs antspyx: pip install antspyx).
# antsApplyTransforms here is a stand-in built on antspyx (same ANTs library and
# transform-list order); on HiPerGator use `module load ants`.
# Expected: 82 regions; registration r = 1.000 (0.18 with a wrong transform);
# ROI signal r = 1.000; lag slope (scanner order) no STC C1 about -0.6, C2 about
# -1.1, p = 0.0005; correct STC C1 and C2 CIs include 0.
set -euo pipefail
cd "$(dirname "$0")"
H=$(pwd); D=$(pwd)/work
rm -rf "$D"; mkdir -p "$D"
python3 sim_rabies.py "$D" 12
for s in none correct; do
  python3 cpca_loadings.py "$D/$s/hmm_signals" "$D/$s/cpca_loadings_roi.csv"
  (cd "$D/$s" && python3 "$H/../../make_slice_maps.py" --meta meta_data_with_bids.csv \
      --rabies_preproc preprocess_outputs --atlas ../atlas/atlas.nii.gz --signals_dir hmm_signals \
      --out_dir slice_maps --ants "$H/antsApplyTransforms" --qc_n 3 \
      --cleaned_glob "preprocess_outputs/bold_datasink/commonspace_bold/**/sub-{id}_*.nii.gz" 2>/dev/null \
      | grep -E "regions|Registration|ROI signal")
  echo "== lag check, $s"
  (cd "$D/$s" && python3 "$H/../../slice_timing_lag_check.py" --loadings cpca_loadings_roi.csv \
      --pairs slice_maps/slice_timing_pairs.csv --fractions slice_maps/slice_fractions.csv \
      --strata group --order 0,2,4,6,8,10,12,1,3,5,7,9,11,13 --rabies_tpattern alt+z \
      --components 1 2 --n_perm 2000 --n_boot 500 --out_dir lagcheck 2>/dev/null \
      | grep -E "^ +[12] +scanner ")
done
