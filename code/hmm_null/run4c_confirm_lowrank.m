% run4c_confirm_lowrank.m
% PRE-SPECIFIED confirmatory test of state structure with the PCA-HMM
% (HMM-MAR options.lowrank), motivated by the spin-echo / low-signal-rank
% argument. Written BEFORE any of these fits were run.
%
% PRE-SPECIFIED DESIGN (do not edit after seeing results)
%   Models      lowrank 15 and lowrank 25 (HMM_CONFIG = 1 or 2)
%   K           2, 3, 4
%   Randomness  fresh: seed 20261001 for splits and surrogates (the
%               exploratory run used 123); 30 split-halves; 30 phase-null
%               fits per K (null repetition r reuses split r)
%   Arms        observed data, and stationary replicas of every mouse
%               (dfc_stationary_replica: same auto- and cross-spectra,
%               Gaussian, stationary) run through the identical procedure,
%               so the HMM is calibrated the way the DCC analysis was
%   Decision    for each of the 6 model x K cells:
%               c_r = (obs_r - null_r) - (rep_r - repnull_r), demeaned
%               state-FC similarity, paired over split r;
%               one-sided Wilcoxon signed-rank c > 0; Holm across 6 cells.
%               State structure is claimed for a cell only if Holm p < 0.05
%               AND the weakest matched state replicates better in the
%               observed splits than in the null (median).
%               IAAFT (HMM_NULL=iaaft) is descriptive only.
%   Drift check full fits at K = 2-4 (observed data); fractional occupancy
%               of every state by scan quarter, per mouse. A state whose
%               occupancy changes between scan halves consistently across
%               mice is a drift (e.g. sedation) rather than a recurring state.
%   Analysis    analyze_confirm_lowrank.py (Python) after all finish jobs.
%
% HOW IT RUNS (same machinery as run4b)
%   HMM_CONFIG=1|2        lowrank 15 | lowrank 25
%   HMM_ARM=observed|replica
%   HMM_N_CHUNKS=N and HMM_CHUNK_IDX (or SLURM_ARRAY_TASK_ID): chunk mode,
%                         fits this chunk's share, saves checkpoints, exits
%   no HMM_N_CHUNKS       finish: loads checkpoints (fits any missing), writes
%                         the tables; for the observed arm also the full fits
%   HMM_SKIP_FULLFIT=1    finish without the occupancy full fits
%   HMM_NULL=iaaft        descriptive IAAFT null (outputs carry _iaaft)
%   HMM_QUICK_TEST=1      tiny run to check the setup (separate folder)
%   HMM_DATA_DIR, HMM_N_SUBJECTS, HMM_TR, HMM_OUT_DIR, HMM_TOOLBOX: paths and
%                         cohort (defaults: mouse, 95 subjects, TR 2 s)
%
% Submit everything with:  bash submit_confirm_lowrank.sh
%
% Requires on the path: hmm_fit_raw_timeseries.m, hmm_select_k_splithalf_null.m,
% hmm_match_states_fc.m, hmm_phase_randomise.m, hmm_iaaft_surrogate.m,
% dfc_stationary_replica.m, HMM-MAR.

clear;
close all;

%% ========== Configuration ==========
data_dir = local_env_str('HMM_DATA_DIR', 'hmm_signals');
file_pattern = 'signal%d.txt';
n_subjects = local_env_num('HMM_N_SUBJECTS', 95);
TR_sec = local_env_num('HMM_TR', 2);
toolbox_dir = local_env_str('HMM_TOOLBOX', '/blue/febo/MATLAB/HMM-MAR-master');
out_root = local_env_str('HMM_OUT_DIR', 'hmm_confirm_lowrank');

% ---- pre-specified settings ----
lowrank_dims = [15, 25];
K_values = 2:4;
n_split_half = 30;
n_null = 30;
seed = 20261001;
replica_seed = 20261002;
initrep_selection = 5;
initrep_final = 15;

base_options = struct();
base_options.standardise = 1;
base_options.zeromean = 1;
base_options.covtype = 'full';
base_options.order = 0;
base_options.inittype = 'hmmmar';
base_options.initrep = initrep_selection;
base_options.initcyc = 25;
base_options.cyc = 1000;
base_options.verbose = 0;
base_options.useParallel = 0;
base_options.Fs = 1 / TR_sec;
base_options.DirichletDiag = 10;

%% ========== Environment switches ==========
cfg_id = local_env_num('HMM_CONFIG', 1);
if ~any(cfg_id == 1:numel(lowrank_dims))
    error('HMM_CONFIG must be 1 (lowrank %d) or 2 (lowrank %d).', lowrank_dims);
end
arm = lower(local_env_str('HMM_ARM', 'observed'));
if ~any(strcmp(arm, {'observed', 'replica'}))
    error('HMM_ARM must be "observed" or "replica" (got "%s").', arm);
end
null_type = lower(local_env_str('HMM_NULL', 'phase'));
if ~any(strcmp(null_type, {'phase', 'iaaft'}))
    error('HMM_NULL must be "phase" or "iaaft".');
end
n_chunks = local_env_num('HMM_N_CHUNKS', 1);
chunk_mode = n_chunks > 1;
chunk_idx = 1;
if chunk_mode
    chunk_idx = local_env_num('HMM_CHUNK_IDX', local_env_num('SLURM_ARRAY_TASK_ID', NaN));
    if isnan(chunk_idx)
        error('HMM_N_CHUNKS is set but neither HMM_CHUNK_IDX nor SLURM_ARRAY_TASK_ID is.');
    end
end
skip_fullfit = local_env_flag('HMM_SKIP_FULLFIT');

if local_env_flag('HMM_QUICK_TEST')
    fprintf('*** QUICK TEST MODE ***\n');
    K_values = 2:3;
    n_split_half = 2;
    n_null = 2;
    base_options.initrep = 2;
    base_options.cyc = 100;
    initrep_final = 2;
    out_root = [out_root '_quicktest'];
end

label = sprintf('lowrank_%d', lowrank_dims(cfg_id));
opts = base_options;
opts.lowrank = lowrank_dims(cfg_id);
arm_dir = fullfile(out_root, arm);
sel_dir = fullfile(arm_dir, ['K_selection_' label]);
local_mkdir(sel_dir);
fprintf('Model %s | arm %s | null %s | K %s | %d splits, %d null fits | seed %d\n', ...
    label, arm, null_type, mat2str(K_values), n_split_half, n_null, seed);

%% ========== HMM-MAR ==========
if ~exist(toolbox_dir, 'dir')
    error('HMM-MAR toolbox not found: %s', toolbox_dir);
end
addpath(genpath(toolbox_dir));

%% ========== Load metadata and time series (as run4b) ==========
fprintf('Working directory: %s\n', pwd);
meta_path = fullfile(data_dir, 'meta_data.csv');
groups = categorical(repmat("All", n_subjects, 1));
if exist(meta_path, 'file')
    meta = readtable(meta_path, 'TextType', 'string');
    if all(ismember({'subject_id', 'group'}, meta.Properties.VariableNames))
        [found, loc] = ismember((1:n_subjects).', double(meta.subject_id));
        groups = categorical(repmat("NA", n_subjects, 1));
        groups(found) = categorical(meta.group(loc(found)));
    else
        warning('meta_data.csv lacks subject_id and/or group; using one group.');
    end
else
    warning('No meta_data.csv; using one group.');
end

Ts_list = cell(n_subjects, 1);
valid = false(n_subjects, 1);
n_roi = [];
for s = 1:n_subjects
    fn = fullfile(data_dir, sprintf(file_pattern, s));
    if ~exist(fn, 'file')
        warning('Missing subject file: %s', fn);
        continue;
    end
    X = readmatrix(fn);
    if isempty(X) || size(X, 1) < 50 || any(~isfinite(X(:)))
        warning('Invalid time series for subject %d.', s);
        continue;
    end
    if isempty(n_roi)
        n_roi = size(X, 2);
    elseif size(X, 2) ~= n_roi
        warning('Subject %d has %d ROIs (expected %d); skipped.', s, size(X, 2), n_roi);
        continue;
    end
    Ts_list{s} = double(X);
    valid(s) = true;
end
if nnz(valid) < 4
    error(['Only %d subjects loaded from "%s" (working directory %s). Set HMM_DATA_DIR ' ...
        'to the folder that contains the signal files.'], nnz(valid), data_dir, pwd);
end
Ts_list = Ts_list(valid);
subject_ids = find(valid);
groups = removecats(groups(valid));
if any(string(groups) == "NA")
    error('Missing group for subject(s) %s.', mat2str(subject_ids(string(groups) == "NA").'));
end
T_sub = cellfun(@(x) size(x, 1), Ts_list);
fprintf('Loaded %d subjects, %d ROIs, %d-%d volumes (TR %.2f s).\n', numel(Ts_list), ...
    n_roi, min(T_sub), max(T_sub), TR_sec);
if opts.lowrank >= n_roi
    error('lowrank %d must be smaller than the number of ROIs (%d).', opts.lowrank, n_roi);
end

%% ========== Replica arm: stationary Gaussian replicas ==========
if strcmp(arm, 'replica')
    Ts_use = cell(size(Ts_list));
    for s = 1:numel(Ts_list)
        rng(replica_seed + subject_ids(s), 'twister');   % deterministic per mouse
        Ts_use{s} = dfc_stationary_replica(Ts_list{s});
    end
    fprintf('Replica arm: %d stationary replicas generated (seed %d + subject id).\n', ...
        numel(Ts_use), replica_seed);
else
    Ts_use = Ts_list;
end

%% ========== Record the pre-specified design before any fitting ==========
local_write_prereg(fullfile(out_root, 'PREREGISTRATION.txt'), lowrank_dims, K_values, ...
    n_split_half, n_null, seed, replica_seed, data_dir, n_subjects, TR_sec);

%% ========== Split-half selection against the null ==========
run_opts = struct('use_pool', false, 'n_chunks', n_chunks, 'chunk_idx', chunk_idx, ...
    'null_type', null_type);
selection_tbl = hmm_select_k_splithalf_null(Ts_use, groups, K_values, n_split_half, ...
    opts, sel_dir, seed, n_null, run_opts);

if chunk_mode
    fprintf('\nChunk %d of %d finished (%s, %s).\n', chunk_idx, n_chunks, label, arm);
    return;
end

writetable(table(subject_ids(:), groups(:), 'VariableNames', {'subject_id', 'group'}), ...
    fullfile(out_root, 'metadata_aligned.csv'));

%% ========== Full fits: occupancy by scan quarter (observed, phase run) ==========
if strcmp(arm, 'observed') && strcmp(null_type, 'phase') && ~skip_fullfit
    X_all = vertcat(Ts_list{:});
    offs = [0; cumsum(T_sub(:))];
    rows = cell(0, 7);
    for K = K_values
        o = opts;
        o.K = K;
        o.initrep = initrep_final;
        rng(seed + K, 'twister');
        fprintf('Full fit %s, K = %d ...\n', label, K);
        try
            [~, Gamma] = hmmmar(X_all, T_sub(:), o);
        catch ME
            warning('Full fit K = %d failed (%s); occupancy not computed.', K, ME.message);
            continue;
        end
        if size(Gamma, 1) ~= sum(T_sub)
            warning(['Gamma has %d rows, expected %d (lags/embedding?). Occupancy ' ...
                'for K = %d skipped.'], size(Gamma, 1), sum(T_sub), K);
            continue;
        end
        for s = 1:numel(Ts_list)
            G = Gamma(offs(s) + 1:offs(s + 1), :);
            Ts_n = size(G, 1);
            q = min(floor((0:Ts_n - 1).' / Ts_n * 4) + 1, 4);
            for qq = 1:4
                fo = mean(G(q == qq, :), 1);
                for k = 1:K
                    rows(end + 1, :) = {subject_ids(s), char(groups(s)), K, k, qq, ...
                        fo(k), mean(G(:, k))}; %#ok<AGROW>
                end
            end
        end
        save(fullfile(arm_dir, sprintf('gamma_%s_K%d.mat', label, K)), 'Gamma', ...
            'T_sub', 'subject_ids', '-v7.3');
    end
    if ~isempty(rows)
        occ = cell2table(rows, 'VariableNames', {'subject_id', 'group', 'K', 'state', ...
            'quarter', 'fo_quarter', 'fo_total'});
        writetable(occ, fullfile(arm_dir, ['occupancy_' label '.csv']));
        fprintf('Occupancy by quarter written: %s\n', ...
            fullfile(arm_dir, ['occupancy_' label '.csv']));
    end
end
fprintf('\nFinished %s, %s arm. Tables in %s\n', label, arm, sel_dir);


%% ========================================================================
function local_write_prereg(fn, dims, K_values, n_split, n_null, seed, rseed, data_dir, ...
    n_subjects, TR)
if exist(fn, 'file')
    return;       % written once, by the first job that starts
end
fid = fopen(fn, 'w');
fprintf(fid, 'Confirmatory PCA-HMM test, pre-specified before fitting.\n');
fprintf(fid, 'Written: %s\n\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')));
fprintf(fid, 'Data: %s (%d subjects, TR %.2f s)\n', data_dir, n_subjects, TR);
fprintf(fid, 'Models: lowrank %s\nK: %s\n', mat2str(dims), mat2str(K_values));
fprintf(fid, 'Split-halves: %d; phase-null fits per K: %d; seed %d; replica seed %d\n', ...
    n_split, n_null, seed, rseed);
fprintf(fid, ['Decision: per model x K cell, c_r = (obs_r - null_r) - (rep_r - repnull_r)\n' ...
    '  (demeaned state-FC split-half similarity, paired over split r);\n' ...
    '  one-sided Wilcoxon signed-rank c > 0, Holm across all cells;\n' ...
    '  state structure claimed only if Holm p < 0.05 AND the weakest matched\n' ...
    '  state replicates better in observed than in null splits (medians).\n' ...
    'IAAFT null: descriptive only.\n' ...
    'Drift: full fits K = 2-4; per-state occupancy 2nd minus 1st scan half,\n' ...
    '  Wilcoxon across mice, Holm across states within K.\n']);
fclose(fid);
end


function local_mkdir(d)
if ~exist(d, 'dir')
    try
        mkdir(d);
    catch
        % another array task created it first
    end
end
end


function v = local_env_num(name, default)
s = strtrim(getenv(name));
if isempty(s)
    v = default;
    return;
end
v = str2double(s);
if isnan(v)
    error('Environment variable %s = "%s" is not a number.', name, s);
end
end


function v = local_env_str(name, default)
v = strtrim(getenv(name));
if isempty(v)
    v = default;
end
end


function tf = local_env_flag(name)
v = strtrim(lower(getenv(name)));
tf = any(strcmp(v, {'1', 'true', 'yes', 'on'}));
end
