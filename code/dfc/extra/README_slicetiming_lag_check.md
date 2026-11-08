### BASH code for running code step 1

cd /blue/febo/share/Abeta_Age/rabies/signals
find ../confound_correction_outputs -name "*sub-001_*.nii*" | head    # confirm the cleaned-data file name
module load ants
  python ../code/dfc/make_slice_maps.py --meta meta_data_with_bids.csv --rabies_preproc ../preprocess_outputs \
  --atlas /blue/febo/share/civm_atlas/BrainNetCoord.nii --signals_dir hmm_signals \
  --out_dir slice_maps --qc_n 0 \
  --cleaned_glob "../confound_correction_outputs/confound_correction_datasink/cleaned_timeseries/**/*sub-{id}_*.nii*"
  
  
  
  ### BASH code for running code step 2, after main step 12
  
  python slice_timing_lag_check.py --loadings dfc_dcc_cpca/cpca_loadings_roi.csv \
  --pairs slice_maps/slice_timing_pairs.csv --fractions slice_maps/slice_fractions.csv \
  --strata group --TR 2 --rabies_tpattern alt+z \
  --order 0,2,4,6,8,10,12,1,3,5,7,9,11,13 --out_dir slice_timing_check
  
  
