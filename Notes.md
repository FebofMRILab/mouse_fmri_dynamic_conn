### Run order 
•	Recommended order: individual code can be run independently. But see main README.md

### Step
0	Test jobs (run_hmm_tests.sbatch, run_static_tests.sbatch)
1	Signal rank
2	Split-half null
3	Confirmatory test
4–7	Reported HMM: one job, run_hmm_pipeline.sbatch (see below)
8	Supplement models, e.g. rank 25 (if want to run other models)
9	DFC: DCC/GARCH and complex PCA
10	DFC: lag covariate
11	DFC figures and tables
12	DFC: confound and transient checks
13	DFC: calibrated group tests
14	Positive control

### Notes:
1.	Begin with step 0 after organizing ‘signals/hmm_signals’ and ‘static’ FC directories.
2.	For ‘signals/hmm_signals’ use code in ‘organize_signals’. This extracts signals from cleaned timeseries images in ‘confound_correction_outputs’ directory output from RABIES: https://rabies.readthedocs.io/en/latest/. It also aligns meta_data with timeseries. Need meta_data.csv. Example is provided. 
3.	Static_v2 can go into static/ directory for running (not shown in table above). Static FC is analyzed independently and optionally. 
4.	After Static_V2/build_matrices, run_static_tests.sbatch can be run. 
5.	Before running run_hmm_tests.sbatch, run_static_tests.sbatch, it is recommended hipergator_tests/setup_python_env and then link_to_signals before step 0 in table. 
6.	All README files provide information on how to adjust to your data sets. 

### ISCLAIMER: This code was put together for analysis of our datasets and is freely distributable as an analysis workflow that others can use and test on our data (see information on how to obtain data sets used in this study). It adapts code already available (see publications list and code repositories list). There are no guarantees that the code will work on other data and it is not for clinical use or diagnostics, only for research purposes. 


