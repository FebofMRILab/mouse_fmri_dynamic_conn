% run4b_hmm_splithalf_null.m
% Raw-time-series HMM-MAR: split-half K selection against a phase-randomised
% null. Runs in a parallel pool, serially, or split across a SLURM array.
%
% HOW IT RUNS (model chosen by HMM_CONFIG; default 1 = primary)
%   - CHUNK task (HMM_N_CHUNKS > 1, SLURM_ARRAY_TASK_ID = chunk): fits its
%     share of that model's jobs serially, one core, NO Parallel Computing
%     Toolbox license, saves checkpoints and exits.
%   - FINISH run (no HMM_N_CHUNKS): loads every checkpoint (fitting any that
%     are missing), writes the summary, chooses K and, for model 1, does the
%     final fit. Uses a parallel pool only if a license is free, otherwise
%     runs serially.
%   - run_sensitivity = true in a FINISH run of model 1 runs all the other
%     models afterwards in the same job (slow; the array route is better).
%   Merge all models afterwards with collect_sensitivity_null.m.
%
% ENVIRONMENT SWITCHES (set in the SLURM scripts)
%   HMM_CONFIG=i         model i of model_configs.csv (1 = primary)
%   HMM_N_CHUNKS=N       chunk mode; chunk = SLURM_ARRAY_TASK_ID (1..N)
%   HMM_NO_POOL=1        never try to start a parallel pool
%   HMM_NULL=iaaft       use the multivariate IAAFT null instead of phase
%                        randomisation (default HMM_NULL=phase). IAAFT
%                        outputs get the suffix _iaaft; observed-data fits
%                        are shared with the phase run; no final fit.
%   HMM_K_VALUES="2:4"   override the K grid (e.g. restrict IAAFT to the
%                        K values that matter)
%   HMM_QUICK_TEST=1     tiny run (2 splits, 2 null, K=2:4, short fits)
%   HMM_LIST_CONFIGS=1   print the model list and exit (no fitting)
%
% Changes from run4_hmm_fit_raw_timeseries.m
%   - model_type switch: 'lowrank' (HMM-PCA, options.lowrank) or 'gaussian'
%   - K grid starts at 2 so the smallest K is not chosen for being the edge
%   - K is selected by delta_demeaned (observed - null, demeaned state FC);
%     the legacy choice (max raw median) is printed for comparison
%   - sensitivity models compare pre-PCA and lowrank at the SAME dimensions
%   - fewer initialisations during selection (initrep_selection), full
%     initrep for the final fit
%
% Requires on the path: hmm_fit_raw_timeseries.m, hmm_select_k_splithalf_null.m,
% hmm_match_states_fc.m, hmm_phase_randomise.m, hmm_iaaft_surrogate.m, HMM-MAR.

clc;
clear;
close all;

%% ========== Configuration ==========
data_dir = 'hmm_signals';
file_pattern = 'signal%d.txt';
n_subjects = 95;
TR_sec = 2;

toolbox_dir = '/blue/febo/MATLAB/HMM-MAR-master';
out_dir = 'hmm_raw_timeseries_null';

% Primary model (set to the null-supported choice from the sensitivity run)
model_type = 'lowrank';     % 'lowrank' (HMM-PCA) or 'gaussian'
lowrank_dim = 10;            % used when model_type = 'lowrank'
pca_dim_primary = 0;        % used when model_type = 'gaussian' (0 = no PCA)

% 1 = covariance (FC) states only (primary analysis).
% 0 = states also have a mean: activation / co-activation states. Diagnostic
%     only; results go to <out_dir>_zeromean0 and state mean maps are scored.
%     Can also be set per job with HMM_ZEROMEAN=0.
zeromean = 1;

K_values = 2:10;
n_split_half = 10;
n_null = 10;                 % phase-randomised null repetitions (<= n_split_half)
final_seed = 123;

% Initialisations: fewer during selection (only consistency across halves
% matters there), the full number for the final model.
initrep_selection = 5;
initrep_final = 15;

% Sensitivity models (matched dimensions)
run_sensitivity = false;     % true = in the finish run of model 1, also run
                             % every other model serially (slow; prefer
                             % submit_hmm_chunks.sh with each HMM_CONFIG)
sensitivity_dims = [15, 25, 50];
n_null_sensitivity = 5;

% Common HMM-MAR options
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
quick_test = local_env_flag('HMM_QUICK_TEST');
list_only = local_env_flag('HMM_LIST_CONFIGS');
no_pool = local_env_flag('HMM_NO_POOL');

cfg_id = local_env_num('HMM_CONFIG', 1);
n_chunks = local_env_num('HMM_N_CHUNKS', 1);
chunk_mode = n_chunks > 1;
chunk_idx = 1;
if chunk_mode
    chunk_idx = local_env_num('SLURM_ARRAY_TASK_ID', NaN);
    if isnan(chunk_idx)
        error('HMM_N_CHUNKS is set but SLURM_ARRAY_TASK_ID is not (not an array task).');
    end
end

zeromean = local_env_num('HMM_ZEROMEAN', zeromean);
if ~any(zeromean == [0 1])
    error('zeromean / HMM_ZEROMEAN must be 0 or 1.');
end
base_options.zeromean = zeromean;
if zeromean == 0
    % Separate folder: labels and checkpoint names would otherwise collide
    % with (and overwrite) the zeromean = 1 results.
    out_dir = [out_dir '_zeromean0'];
    select_metric = 'meanmap';
    fprintf('*** zeromean = 0: activation-state diagnostic -> %s ***\n', out_dir);
else
    select_metric = 'demeaned';
end

if quick_test
    fprintf('*** QUICK TEST MODE ***\n');
    K_values = 2:4;
    n_split_half = 2;
    n_null = 2;
    n_null_sensitivity = 2;
    base_options.initrep = 2;
    base_options.cyc = 100;
    initrep_final = 2;
    out_dir = [out_dir '_quicktest'];
end

null_type = lower(strtrim(getenv('HMM_NULL')));
if isempty(null_type)
    null_type = 'phase';
end
if ~any(strcmp(null_type, {'phase', 'iaaft'}))
    error('HMM_NULL must be "phase" or "iaaft" (got "%s").', null_type);
end

k_override = strtrim(getenv('HMM_K_VALUES'));
if ~isempty(k_override)
    K_values = local_parse_k(k_override);
end
fprintf('Null model: %s.  K values: %s\n', null_type, mat2str(K_values));

%% ========== Model list ==========
configs = local_build_configs(model_type, lowrank_dim, pca_dim_primary, ...
    sensitivity_dims);
fprintf('\nModel configurations (HMM_CONFIG = row number):\n');
for c = 1:numel(configs)
    fprintf('  %d  %s\n', c, configs(c).label);
end

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
if ~chunk_mode || chunk_idx == 1
    writetable(struct2table(configs(:)), fullfile(out_dir, 'model_configs.csv'));
end

if list_only
    fprintf('\nHMM_LIST_CONFIGS set: %d configurations. Exiting.\n', numel(configs));
    return;
end

if cfg_id < 1 || cfg_id > numel(configs) || cfg_id ~= floor(cfg_id)
    error('HMM_CONFIG = %g is not a valid row (1..%d).', cfg_id, numel(configs));
end
if chunk_mode
    run_idx = cfg_id;
    fprintf('\nCHUNK MODE: model %d (%s), chunk %d of %d.\n', cfg_id, ...
        configs(cfg_id).label, chunk_idx, n_chunks);
elseif cfg_id == 1 && run_sensitivity
    run_idx = 1:numel(configs);
else
    run_idx = cfg_id;
end

rng(final_seed, 'twister');

%% ========== Add HMM-MAR ==========
if ~exist(toolbox_dir, 'dir')
    error('HMM-MAR toolbox not found: %s', toolbox_dir);
end
addpath(genpath(toolbox_dir));

%% ========== Load metadata ==========
meta_path = fullfile(data_dir, 'meta_data.csv');
groups = categorical(repmat("All", n_subjects, 1));

if exist(meta_path, 'file')
    meta = readtable(meta_path, 'TextType', 'string');
    if all(ismember({'subject_id', 'group'}, meta.Properties.VariableNames))
        assert(numel(unique(meta.subject_id)) == height(meta), ...
            'meta_data.csv contains duplicate subject_id values.');
        [found, loc] = ismember((1:n_subjects).', double(meta.subject_id));
        groups = categorical(repmat("NA", n_subjects, 1));
        groups(found) = categorical(meta.group(loc(found)));
    else
        warning('Metadata file lacks subject_id and/or group; using a single group.');
    end
else
    warning('No metadata file found; using a single group.');
end

%% ========== Load raw ROI time series ==========
Ts_list = cell(n_subjects, 1);
valid_subject = false(n_subjects, 1);
n_roi = [];

for s = 1:n_subjects
    filename = fullfile(data_dir, sprintf(file_pattern, s));
    if ~exist(filename, 'file')
        warning('Missing subject file: %s', filename);
        continue;
    end
    try
        Ts = readmatrix(filename);
    catch ME
        warning('Could not read subject %d: %s', s, ME.message);
        continue;
    end
    if ~isnumeric(Ts) || ~ismatrix(Ts) || isempty(Ts) || ...
            size(Ts, 1) < 2 || size(Ts, 2) < 2 || any(~isfinite(Ts(:)))
        warning('Invalid raw time series for subject %d.', s);
        continue;
    end
    if isempty(n_roi)
        n_roi = size(Ts, 2);
    elseif size(Ts, 2) ~= n_roi
        warning('Subject %d has %d ROIs; expected %d.', s, size(Ts, 2), n_roi);
        continue;
    end
    Ts_list{s} = double(Ts);
    valid_subject(s) = true;
end

if nnz(valid_subject) < 4
    error('Only %d valid subjects were loaded; at least four are required.', ...
        nnz(valid_subject));
end

Ts_list = Ts_list(valid_subject);
subject_ids = find(valid_subject);
groups = groups(valid_subject);

missing_group = ismissing(groups) | string(groups) == "NA" | ...
    strlength(strtrim(string(groups))) == 0;
if any(missing_group)
    bad_ids = subject_ids(missing_group);
    error('Missing group metadata for subject ID(s): %s', mat2str(bad_ids(:).'));
end
groups = removecats(groups);

group_names = categories(groups);
for g = 1:numel(group_names)
    n_group = nnz(groups == group_names{g});
    if n_group < 2
        error(['Group "%s" has %d valid subject(s). Stratified split-half ' ...
            'analysis requires at least two.'], group_names{g}, n_group);
    end
end

fprintf('Loaded %d valid subjects with %d ROIs.\n', numel(Ts_list), n_roi);

% Sanity checks on subject count, scan length and groups
n_extra = 0;
while exist(fullfile(data_dir, sprintf(file_pattern, n_subjects + n_extra + 1)), 'file')
    n_extra = n_extra + 1;
end
if n_extra > 0
    warning(['%d more signal file(s) exist after %s: only subjects 1-%d are ' ...
        'loaded. Check n_subjects.'], n_extra, sprintf(file_pattern, n_subjects), ...
        n_subjects);
end
T_sub = cellfun(@(x) size(x, 1), Ts_list);
fprintf('Volumes per subject: %d-%d at TR %.2f s (%.1f-%.1f min).\n', ...
    min(T_sub), max(T_sub), TR_sec, min(T_sub) * TR_sec / 60, max(T_sub) * TR_sec / 60);
if min(T_sub) ~= max(T_sub)
    warning('Scan lengths differ across subjects (%d-%d volumes).', min(T_sub), max(T_sub));
end
if any(T_sub < n_roi)
    warning('Some subjects have fewer time points than ROIs: are files ROI x time?');
end
fprintf('Subjects per group:\n');
for g = 1:numel(group_names)
    fprintf('  %-12s %3d\n', group_names{g}, nnz(groups == group_names{g}));
end

metadata_aligned = table(subject_ids(:), groups(:), ...
    'VariableNames', {'subject_id', 'group'});
if ~chunk_mode || chunk_idx == 1
    writetable(metadata_aligned, fullfile(out_dir, 'metadata_aligned.csv'));
end

%% ========== Parallel pool (never in chunk mode) ==========
pool = [];
if ~chunk_mode && ~no_pool
    pool = local_start_pool();
end
run_opts = struct('use_pool', ~isempty(pool), 'n_chunks', n_chunks, ...
    'chunk_idx', chunk_idx, 'null_type', null_type);

%% ========== Run the selected model configuration(s) ==========
for c = run_idx
    cfg = configs(c);
    fprintf('\n================ Model %d of %d: %s ================\n', ...
        c, numel(configs), cfg.label);

    if strcmp(cfg.type, 'lowrank') && cfg.lowrank >= n_roi
        warning('Skipping %s: lowrank (%d) must be < n_roi (%d).', ...
            cfg.label, cfg.lowrank, n_roi);
        continue;
    end
    if strcmp(cfg.type, 'gaussian') && cfg.pca >= n_roi
        warning('Skipping %s: pca (%d) must be < n_roi (%d).', ...
            cfg.label, cfg.pca, n_roi);
        continue;
    end
    if strcmp(cfg.type, 'lowrank') && cfg.lowrank > 0.75 * n_roi
        warning(['lowrank = %d is %.0f%% of n_roi = %d; per-state covariances ' ...
            'are close to full rank, so HMM-PCA is doing little reduction.'], ...
            cfg.lowrank, 100 * cfg.lowrank / n_roi, n_roi);
    end

    opts = local_model_options(base_options, cfg);
    if c == 1
        this_n_null = n_null;
    else
        this_n_null = min(n_null_sensitivity, n_split_half);
    end

    selection_tbl = hmm_select_k_splithalf_null( ...
        Ts_list, groups, K_values, n_split_half, opts, ...
        fullfile(out_dir, ['K_selection_' cfg.label]), final_seed, ...
        this_n_null, run_opts);

    if chunk_mode
        continue;       % summary, K choice and final fit happen in the finish run
    end

    [selected_K, legacy_K] = local_choose_K(selection_tbl, select_metric, n_split_half, ...
        this_n_null, K_values);

    %% ---------- Final fit (primary model, phase-null run only) ----------
    if c == 1 && ~strcmp(null_type, 'phase')
        fprintf(['IAAFT run: no final fit (the final model comes from the ' ...
            'phase-null run). Compare split_half_K_selection_iaaft.csv ' ...
            'with split_half_K_selection.csv.\n']);
    end
    if c == 1 && strcmp(null_type, 'phase')
        final_options = opts;
        final_options.K = selected_K;
        final_options.initrep = initrep_final;
        final_options.verbose = 1;

        rng(final_seed, 'twister');
        R = hmm_fit_raw_timeseries(Ts_list, final_options, subject_ids);

        R.TR_sec = TR_sec;
        if isfield(R, 'dwell_TR')
            R.dwell_sec = R.dwell_TR * TR_sec;
        end
        R.groups = groups;
        R.model_selection = selection_tbl;
        R.model_label = cfg.label;

        save(fullfile(out_dir, 'raw_hmm_results.mat'), 'R', '-v7.3');
        writetable(selection_tbl, fullfile(out_dir, 'K_selection_summary.csv'));

        state_fc = R.state_fc;
        save(fullfile(out_dir, 'raw_hmm_state_fc.mat'), 'state_fc', '-v7.3');

        finite_max = cellfun(@(A) max(abs(A(:)), [], 'omitnan'), state_fc);
        max_abs_r = max(finite_max, [], 'omitnan');
        if isempty(max_abs_r) || ~isfinite(max_abs_r) || max_abs_r <= 0
            max_abs_r = 1;
        end
        for k = 1:R.K
            f = figure('Visible', 'off');
            imagesc(state_fc{k});
            axis square; colorbar; colormap(parula);
            clim([-max_abs_r, max_abs_r]);
            title(sprintf('%s state %d FC', cfg.label, k), 'Interpreter', 'none');
            xlabel('ROI'); ylabel('ROI');
            exportgraphics(f, fullfile(out_dir, ...
                sprintf('state_%02d_raw_HMM_FC.png', k)), 'Resolution', 300);
            close(f);
        end

        save(fullfile(out_dir, 'raw_HMM_analysis_configuration.mat'), ...
            'base_options', 'final_options', 'configs', 'K_values', ...
            'n_split_half', 'n_null', 'selected_K', 'legacy_K', 'TR_sec', ...
            'subject_ids', 'groups', 'selection_tbl', '-v7.3');
    end
end

if ~isempty(pool)
    delete(pool);
end
if chunk_mode
    fprintf('\nChunk %d of %d finished. Results are checkpoints in: %s\n', ...
        chunk_idx, n_chunks, out_dir);
else
    fprintf('\nCompleted. Results saved to: %s\n', out_dir);
end


%% ========================================================================
function configs = local_build_configs(model_type, lowrank_dim, pca_dim, dims)
% Row 1 = primary; then pre-PCA / lowrank pairs at matched dimensions, then
% the full-dimensional Gaussian. Duplicates of the primary are dropped.
configs = struct('type', {}, 'lowrank', {}, 'pca', {}, 'label', {});
configs(1) = local_cfg(model_type, lowrank_dim, pca_dim);
for dd = dims
    configs(end + 1) = local_cfg('gaussian', 0, dd); %#ok<AGROW>
    configs(end + 1) = local_cfg('lowrank', dd, 0);  %#ok<AGROW>
end
configs(end + 1) = local_cfg('gaussian', 0, 0);
[~, keep] = unique({configs.label}, 'stable');
configs = configs(sort(keep));
end


function cfg = local_cfg(model_type, lowrank_dim, pca_dim)
cfg.type = model_type;
if strcmp(model_type, 'lowrank')
    cfg.lowrank = lowrank_dim;
    cfg.pca = 0;
    cfg.label = sprintf('lowrank_%d', lowrank_dim);
elseif strcmp(model_type, 'gaussian')
    cfg.lowrank = 0;
    cfg.pca = pca_dim;
    if pca_dim > 0
        cfg.label = sprintf('gaussian_pca_%d', pca_dim);
    else
        cfg.label = 'gaussian_full';
    end
else
    error('Unknown model_type "%s".', model_type);
end
end


function opts = local_model_options(base_options, cfg)
opts = base_options;
if isfield(opts, 'lowrank'), opts = rmfield(opts, 'lowrank'); end
if isfield(opts, 'pca'), opts = rmfield(opts, 'pca'); end
if strcmp(cfg.type, 'lowrank')
    opts.lowrank = cfg.lowrank;
elseif cfg.pca > 0
    opts.pca = cfg.pca;
end
end


function [selected_K, legacy_K] = local_choose_K(tbl, metric, n_splits, n_null, K_values)
% metric: 'demeaned' (FC states) or 'meanmap' (activation states).
% K = largest MEDIAN observed-minus-null gap: the mean gap is fragile to a
% few null fits that land on a different solution.
min_ok = ceil(0.80 * n_splits);
min_null_ok = ceil(0.80 * n_null);
eligible = tbl.n_successful_splits >= min_ok & ...
    isfinite(tbl.median_reproducibility);
if ~any(eligible)
    error('No K completed at least %d of %d split-half repetitions.', ...
        min_ok, n_splits);
end

legacy_score = tbl.median_reproducibility;
legacy_score(~eligible) = -Inf;
[~, i_legacy] = max(legacy_score);
legacy_K = tbl.K(i_legacy);

if strcmp(metric, 'meanmap') && ~any(isfinite(tbl.delta_median_meanmap))
    warning('No state mean maps were scored; choosing K on demeaned FC instead.');
    metric = 'demeaned';
end
dcol = ['delta_median_' metric];
pcol = ['p_mw_' metric];

if n_null > 0
    eligible_null = eligible & tbl.n_successful_null >= min_null_ok & ...
        isfinite(tbl.(dcol));
    if ~any(eligible_null)
        error('No K has enough successful observed and null repetitions.');
    end
    score = tbl.(dcol);
    score(~eligible_null) = -Inf;
    [~, i_best] = max(score);
    selected_K = tbl.K(i_best);
    fprintf(['\nSelected K=%d: largest median observed-minus-null gap in %s ' ...
        '(delta = %.3f; one-sided Mann-Whitney p = %.3g, uncorrected).\n'], ...
        selected_K, metric, tbl.(dcol)(i_best), tbl.(pcol)(i_best));
    if ~(tbl.(pcol)(i_best) < 0.05)
        warning(['No evidence that K=%d exceeds the null (p = %.3g): it is the ' ...
            'least-bad K, not a supported one.'], selected_K, tbl.(pcol)(i_best));
    end
else
    selected_K = legacy_K;
    fprintf('\nNo null run; selected K=%d by the legacy rule.\n', selected_K);
end
fprintf('Legacy rule (max raw median reproducibility) would give K=%d.\n', legacy_K);

if selected_K == max(K_values)
    warning(['Selected K=%d is the largest K tested; extend K_values ' ...
        'before trusting this choice.'], selected_K);
elseif selected_K == min(K_values)
    fprintf(['Note: K=%d is the smallest K tested. K=1 cannot be fitted as ' ...
        'an HMM; the null already provides the "no switching" comparison.\n'], selected_K);
end
end


function pool = local_start_pool()
% Try to start a pool sized to the SLURM allocation. ANY failure (toolbox not
% installed, no free license seat, cluster error) returns [] so the analysis
% continues serially instead of crashing.
pool = [];
try
    pool = gcp('nocreate');
catch
    pool = [];
end
if ~isempty(pool)
    return;
end

n = str2double(getenv('SLURM_CPUS_PER_TASK'));
if isnan(n) || n < 1
    n = feature('numcores');
end
if n < 2
    fprintf('Only %d core allocated; running serially.\n', n);
    return;
end

if ~license('test', 'Distrib_Computing_Toolbox')
    warning('No Parallel Computing Toolbox license on this system; running serially.');
    return;
end
[ok, lic_msg] = license('checkout', 'Distrib_Computing_Toolbox');
if ~ok
    warning(['Parallel Computing Toolbox license seat not available (%s). ' ...
        'Running serially on 1 core - consider the chunked array instead.'], ...
        strtrim(lic_msg));
    return;
end

try
    scratch = getenv('TMPDIR');
    if isempty(scratch) || ~exist(scratch, 'dir')
        scratch = tempdir;
    end
    job_tag = getenv('SLURM_JOB_ID');
    if isempty(job_tag)
        job_tag = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    end
    jsl = fullfile(scratch, ['matlab_pool_' job_tag]);
    if ~exist(jsl, 'dir')
        mkdir(jsl);
    end
    c = parcluster('local');
    c.JobStorageLocation = jsl;
    c.NumWorkers = n;
    pool = parpool(c, n, 'IdleTimeout', Inf);
    fprintf('Started parallel pool with %d workers.\n', pool.NumWorkers);
catch ME
    warning('Could not start a parallel pool (%s). Running serially.', ME.message);
    pool = [];
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


function tf = local_env_flag(name)
v = strtrim(lower(getenv(name)));
tf = any(strcmp(v, {'1', 'true', 'yes', 'on'}));
end


function K = local_parse_k(s)
% Parse "2:4", "2 3 4" or "2,3,4" without eval.
s = strtrim(s);
tok = regexp(s, '^(\d+)\s*:\s*(\d+)$', 'tokens', 'once');
if ~isempty(tok)
    K = str2double(tok{1}):str2double(tok{2});
else
    K = str2double(regexp(s, '[\s,]+', 'split'));
end
if isempty(K) || any(isnan(K)) || any(K < 2) || any(K ~= floor(K))
    error('HMM_K_VALUES = "%s" is not a valid list of integers >= 2.', s);
end
K = unique(K(:).', 'stable');
end
