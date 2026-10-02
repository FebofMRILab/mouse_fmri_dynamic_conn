function selection_tbl = hmm_select_k_splithalf_null(Ts_list, groups, K_values, ...
    n_splits, base_options, out_dir, seed, n_null, run_opts)
% HMM_SELECT_K_SPLITHALF_NULL  Split-half K selection with a null at every K.
% Runs in a parallel pool if one is open, serially otherwise, or as one chunk
% of a SLURM array; all three give identical results.
%
%   selection_tbl = hmm_select_k_splithalf_null(Ts_list, groups, K_values, ...
%       n_splits, base_options, out_dir, seed, n_null, run_opts)
%
% run_opts (optional struct):
%   .chunk_idx, .n_chunks  run only chunk chunk_idx of n_chunks (round-robin
%                          over the job list), save checkpoints and return an
%                          EMPTY table. A later call without chunking loads
%                          all checkpoints and builds the summary.
%   .use_pool              false = never use a parallel pool (default true:
%                          use one if it is already open; a pool is never
%                          started here)
%   .null_type             'phase' (default): multivariate phase
%                          randomisation (hmm_phase_randomise).
%                          'iaaft': multivariate IAAFT (hmm_iaaft_surrogate),
%                          which also keeps each ROI's value distribution.
%                          IAAFT null checkpoints are 'nulliaaft_*.mat' and
%                          summary files carry the suffix '_iaaft', so both
%                          nulls can live in the same folder; observed-data
%                          checkpoints are shared between them.
%
% Extends hmm_select_k_splithalf.m. Fitting still uses hmm_fit_raw_timeseries
% (HMM-MAR). The subject partitions, seeds and raw matched-state similarity
% are the same as the original, so the raw columns should reproduce previous
% runs with the same options.
%
%   1. Demeaned similarity: the occupancy-weighted mean FC across states is
%      removed before matching/correlating, so the score reflects
%      state-SPECIFIC connectivity rather than static FC shared by all states.
%   2. Phase-randomised null: every subject's data are replaced by a surrogate
%      with identical random phases across ROIs (static FC and autocorrelation
%      kept, real state switching removed) and the split-half procedure is
%      repeated. Interpret observed - null (delta_*).
%
% PARALLELISM. Every (condition, repetition, K) pair is one independent job
% that fits both halves and scores them. If a parallel pool is already open,
% jobs run in parfor; otherwise a plain for loop runs them (no Parallel
% Computing Toolbox license needed). In chunk mode each SLURM array task runs
% its share serially. HMM-MAR's own useParallel is forced to 0.
%
% CHECKPOINTS. Each finished job is saved to <out_dir>/jobs/. On a rerun with
% the same inputs, finished jobs are loaded instead of refitted, so a job that
% hits its SLURM time limit can simply be resubmitted. A checkpoint is reused
% only if K, seed, subject halves and options all match.
%
% n_null : number of null repetitions (default n_splits; 0 skips the null).
%          Must be <= n_splits because null repetition r reuses partition r.
%
% Requires: hmm_fit_raw_timeseries.m, hmm_match_states_fc.m,
%           hmm_phase_randomise.m, hmm_iaaft_surrogate.m, HMM-MAR
%           (matchpairs is base MATLAB R2019a+).
%           Parallel Computing Toolbox only for pool mode.

if nargin < 7 || isempty(seed)
    seed = 123;
end
if nargin < 6 || isempty(out_dir)
    out_dir = 'hmm_raw_model_selection';
end
if nargin < 8 || isempty(n_null)
    n_null = n_splits;
end
if nargin < 9 || isempty(run_opts)
    run_opts = struct();
end
if ~isfield(run_opts, 'use_pool'), run_opts.use_pool = true; end
if ~isfield(run_opts, 'n_chunks') || isempty(run_opts.n_chunks)
    run_opts.n_chunks = 1;
end
if ~isfield(run_opts, 'chunk_idx') || isempty(run_opts.chunk_idx)
    run_opts.chunk_idx = 1;
end
chunk_mode = run_opts.n_chunks > 1;
if chunk_mode && (run_opts.chunk_idx < 1 || run_opts.chunk_idx > run_opts.n_chunks)
    error('chunk_idx (%d) must be between 1 and n_chunks (%d).', ...
        run_opts.chunk_idx, run_opts.n_chunks);
end
if ~isfield(run_opts, 'null_type') || isempty(run_opts.null_type)
    run_opts.null_type = 'phase';
end
null_type = lower(char(run_opts.null_type));
switch null_type
    case 'phase'
        null_tag = 'null';        % unchanged, so earlier checkpoints still match
        file_sfx = '';
        null_label = 'phase-randomised null';
    case 'iaaft'
        null_tag = 'nulliaaft';
        file_sfx = '_iaaft';
        null_label = 'IAAFT null';
    otherwise
        error('run_opts.null_type must be ''phase'' or ''iaaft''.');
end

%% ---------- Input validation (as original) ----------
if ~iscell(Ts_list) || numel(Ts_list) < 4
    error('Ts_list must contain at least four subjects.');
end
Ts_list = Ts_list(:);
S = numel(Ts_list);

if isempty(groups)
    groups = categorical(repmat("All", S, 1));
else
    groups = categorical(groups(:));
end
if numel(groups) ~= S
    error('groups must have one entry per subject.');
end
missing_group = ismissing(groups) | string(groups) == "NA" | ...
    strlength(strtrim(string(groups))) == 0;
if any(missing_group)
    error('groups contains missing or NA assignments.');
end
groups = removecats(groups);

K_values = unique(K_values(:).', 'stable');
if isempty(K_values) || any(K_values < 1) || any(K_values ~= floor(K_values))
    error('K_values must contain positive integers.');
end
if ~isscalar(n_splits) || n_splits < 1 || n_splits ~= floor(n_splits)
    error('n_splits must be a positive integer.');
end
if ~isscalar(n_null) || n_null < 0 || n_null ~= floor(n_null) || n_null > n_splits
    error('n_null must be an integer between 0 and n_splits.');
end

group_names = categories(groups);
for g = 1:numel(group_names)
    n_group = nnz(groups == group_names{g});
    if n_group < 2
        error('Group "%s" has only %d subject(s); at least two are required.', ...
            group_names{g}, n_group);
    end
end

job_dir = fullfile(out_dir, 'jobs');
if ~exist(job_dir, 'dir')
    mkdir(job_dir);
end

base_options.useParallel = 0;   % no nested parallelism inside workers

%% ---------- Partitions (identical to original for the same seed) ----------
split_stream = RandStream('mt19937ar', 'Seed', seed);
split_a = cell(n_splits, 1);
split_b = cell(n_splits, 1);
for split_idx = 1:n_splits
    [split_a{split_idx}, split_b{split_idx}] = ...
        local_stratified_half_split(groups, split_stream);
    if numel(split_a{split_idx}) < 2 || numel(split_b{split_idx}) < 2
        error('Split %d produced a half with fewer than two subjects.', split_idx);
    end
end

% Surrogates for null repetition r are generated on the worker from their own
% seed (seed + 999 + r), so they are deterministic and never broadcast
% (10 copies of the data per worker would be wasteful on large cohorts).

%% ---------- Job list ----------
nK = numel(K_values);
n_jobs = (n_splits + n_null) * nK;
job_cond = zeros(n_jobs, 1);    % 1 = observed, 2 = null
job_rep = zeros(n_jobs, 1);
job_kidx = zeros(n_jobs, 1);
job_seed = zeros(n_jobs, 1);

j = 0;
for r = 1:n_splits
    for k_idx = 1:nK
        j = j + 1;
        job_cond(j) = 1;
        job_rep(j) = r;
        job_kidx(j) = k_idx;
        job_seed(j) = seed + r * 100000 + K_values(k_idx) * 100;   % as original
    end
end
for r = 1:n_null
    for k_idx = 1:nK
        j = j + 1;
        job_cond(j) = 2;
        job_rep(j) = r;
        job_kidx(j) = k_idx;
        job_seed(j) = seed + r * 100000 + K_values(k_idx) * 100 + 50000;
    end
end

% Largest K first: slowest fits start early, which balances the pool and,
% with round-robin chunking, spreads large-K fits evenly over array tasks.
[~, order] = sort(K_values(job_kidx), 'descend');
job_cond = job_cond(order);
job_rep = job_rep(order);
job_kidx = job_kidx(order);
job_seed = job_seed(order);

if chunk_mode
    run_jobs = find(mod((1:n_jobs).' - 1, run_opts.n_chunks) == run_opts.chunk_idx - 1);
else
    run_jobs = (1:n_jobs).';
end

n_done = 0;
for jj = run_jobs(:).'
    if job_cond(jj) == 1, tag = 'obs'; else, tag = null_tag; end
    if exist(fullfile(job_dir, sprintf('%s_rep%02d_K%02d.mat', tag, ...
            job_rep(jj), K_values(job_kidx(jj)))), 'file')
        n_done = n_done + 1;
    end
end

% Use a pool only if one is already open (the runner decides whether to
% start one). gcp itself needs the toolbox, so guard it.
pool = [];
if run_opts.use_pool && ~chunk_mode
    try
        pool = gcp('nocreate');
    catch
        pool = [];
    end
end
use_pool = ~isempty(pool);
if use_pool
    n_workers = pool.NumWorkers;
    mode_str = sprintf('parallel pool, %d workers', n_workers);
elseif chunk_mode
    n_workers = 0;
    mode_str = sprintf('serial, chunk %d of %d', run_opts.chunk_idx, run_opts.n_chunks);
else
    n_workers = 0;
    mode_str = 'serial';
end
fprintf(['\n%d of %d jobs assigned here (%s, %s); %d already checkpointed. ' ...
    'Each job fits both halves.\n'], numel(run_jobs), n_jobs, mode_str, ...
    null_label, n_done);

% Progress messages from pool workers
q = [];
if use_pool
    try
        q = parallel.pool.DataQueue;
        afterEach(q, @(msg) fprintf('%s\n', msg));
    catch
        q = [];
    end
end

%% ---------- Run jobs ----------
t_start = tic;
results = cell(n_jobs, 1);
n_run = numel(run_jobs);
run_results = cell(n_run, 1);

if use_pool
    parfor ii = 1:n_run
        jj = run_jobs(ii);
        [run_results{ii}, msg] = local_run_job(Ts_list, split_a, split_b, ...
            job_cond(jj), job_rep(jj), K_values(job_kidx(jj)), job_seed(jj), ...
            base_options, job_dir, seed, null_type, null_tag);
        if ~isempty(q)
            send(q, msg);
        end
    end
else
    for ii = 1:n_run
        jj = run_jobs(ii);
        [run_results{ii}, msg] = local_run_job(Ts_list, split_a, split_b, ...
            job_cond(jj), job_rep(jj), K_values(job_kidx(jj)), job_seed(jj), ...
            base_options, job_dir, seed, null_type, null_tag);
        fprintf('%s  (%d/%d)\n', msg, ii, n_run);
    end
end
results(run_jobs) = run_results;

fprintf('\nJobs finished in %.1f h.\n', toc(t_start) / 3600);

if chunk_mode
    fprintf(['Chunk %d of %d done. Run again without chunking to build the ' ...
        'summary from the checkpoints.\n'], run_opts.chunk_idx, run_opts.n_chunks);
    selection_tbl = table();
    return;
end

%% ---------- Collect ----------
obs_raw = NaN(n_splits, nK);
obs_dm = NaN(n_splits, nK);
obs_weak_dm = NaN(n_splits, nK);
null_raw = NaN(n_null, nK);
null_dm = NaN(n_null, nK);
null_weak_dm = NaN(n_null, nK);
obs_mm = NaN(n_splits, nK);        % state-mean (activation) map similarity
obs_weak_mm = NaN(n_splits, nK);
null_mm = NaN(n_null, nK);
obs_details = cell(n_splits, nK);
null_details = cell(n_null, nK);
state_rows = cell(0, 10);
timing_rows = cell(0, 5);

[~, back] = sort(order);   % restore canonical job order for tidy outputs
for jj = back(:).'
    d = results{jj};
    rep = job_rep(jj);
    k_idx = job_kidx(jj);
    K = K_values(k_idx);
    if job_cond(jj) == 1
        cond = 'observed';
        obs_details{rep, k_idx} = d;
    else
        cond = 'null';
        null_details{rep, k_idx} = d;
    end
    timing_rows(end + 1, :) = {cond, rep, K, d.seconds, ~isempty(d.error_message)}; %#ok<AGROW>

    if ~isempty(d.error_message)
        continue;
    end
    has_mm = isfield(d, 'matched_mean') && all(isfinite(d.matched_mean));
    if job_cond(jj) == 1
        obs_raw(rep, k_idx) = mean(d.matched_raw);
        obs_dm(rep, k_idx) = mean(d.matched_dm);
        obs_weak_dm(rep, k_idx) = min(d.matched_dm);
        if has_mm
            obs_mm(rep, k_idx) = mean(d.matched_mean);
            obs_weak_mm(rep, k_idx) = min(d.matched_mean);
        end
    else
        null_raw(rep, k_idx) = mean(d.matched_raw);
        null_dm(rep, k_idx) = mean(d.matched_dm);
        null_weak_dm(rep, k_idx) = min(d.matched_dm);
        if has_mm
            null_mm(rep, k_idx) = mean(d.matched_mean);
        end
    end
    state_rows = [state_rows; local_state_rows(cond, rep, K, d)]; %#ok<AGROW>
end

%% ---------- Summary table ----------
null_mean_raw = local_colmean(null_raw, nK);
null_mean_dm = local_colmean(null_dm, nK);

selection_tbl = table( ...
    K_values(:), ...
    mean(obs_raw, 1, 'omitnan').', ...
    median(obs_raw, 1, 'omitnan').', ...
    std(obs_raw, 0, 1, 'omitnan').', ...
    mean(obs_dm, 1, 'omitnan').', ...
    median(obs_dm, 1, 'omitnan').', ...
    std(obs_dm, 0, 1, 'omitnan').', ...
    mean(obs_weak_dm, 1, 'omitnan').', ...
    null_mean_raw.', ...
    local_colsd(null_raw, nK).', ...
    null_mean_dm.', ...
    local_colsd(null_dm, nK).', ...
    (mean(obs_raw, 1, 'omitnan') - null_mean_raw).', ...
    (mean(obs_dm, 1, 'omitnan') - null_mean_dm).', ...
    sum(isfinite(obs_raw), 1).', ...
    sum(isfinite(null_raw), 1).', ...
    'VariableNames', {'K', ...
    'mean_reproducibility', 'median_reproducibility', 'sd_reproducibility', ...
    'mean_demeaned', 'median_demeaned', 'sd_demeaned', ...
    'weakest_state_demeaned', ...
    'null_mean_raw', 'null_sd_raw', 'null_mean_demeaned', 'null_sd_demeaned', ...
    'delta_raw', 'delta_demeaned', ...
    'n_successful_splits', 'n_successful_null'});

% Robust comparison. A mean-based delta is fragile: a few null fits that
% land on a different solution drag the null mean down (seen in the mouse
% K = 2 IAAFT run). Median delta + one-sided Mann-Whitney p (observed >
% null; uncorrected across K) are the columns to interpret.
selection_tbl.delta_median_demeaned = (median(obs_dm, 1, 'omitnan') - ...
    local_colmedian(null_dm, nK)).';
selection_tbl.p_mw_demeaned = arrayfun(@(k) ...
    local_mw_greater(obs_dm(:, k), null_dm(:, k)), (1:nK).');

% Activation (state-mean) maps: NaN unless zeromean = 0 and means available
null_mean_mm = local_colmean(null_mm, nK);
selection_tbl.mean_meanmap = mean(obs_mm, 1, 'omitnan').';
selection_tbl.median_meanmap = median(obs_mm, 1, 'omitnan').';
selection_tbl.sd_meanmap = std(obs_mm, 0, 1, 'omitnan').';
selection_tbl.weakest_state_meanmap = mean(obs_weak_mm, 1, 'omitnan').';
selection_tbl.null_mean_meanmap = null_mean_mm.';
selection_tbl.null_sd_meanmap = local_colsd(null_mm, nK).';
selection_tbl.delta_meanmap = (mean(obs_mm, 1, 'omitnan') - null_mean_mm).';
selection_tbl.delta_median_meanmap = (median(obs_mm, 1, 'omitnan') - ...
    local_colmedian(null_mm, nK)).';
selection_tbl.p_mw_meanmap = arrayfun(@(k) ...
    local_mw_greater(obs_mm(:, k), null_mm(:, k)), (1:nK).');
has_meanmap = any(isfinite(obs_mm(:)));

if isfield(base_options, 'zeromean') && base_options.zeromean == 0 && ~has_meanmap
    warning(['zeromean = 0 but no state mean maps were found in the output of ' ...
        'hmm_fit_raw_timeseries, so activation states were NOT scored. ' ...
        'Return R.state_mean{k} = getMean(hmm, k) (ROI space) or R.hmm = hmm.']);
end

writetable(selection_tbl, fullfile(out_dir, ['split_half_K_selection' file_sfx '.csv']));

if ~isempty(state_rows)
    state_tbl = cell2table(state_rows, 'VariableNames', ...
        {'condition', 'repetition', 'K', 'metric', 'state_a', 'state_b', ...
        'similarity', 'fo_a', 'fo_b', 'fo_source'});
    writetable(state_tbl, fullfile(out_dir, ['split_half_per_state' file_sfx '.csv']));
else
    state_tbl = table();
end

timing_tbl = cell2table(timing_rows, 'VariableNames', ...
    {'condition', 'repetition', 'K', 'seconds_both_halves', 'failed'});
writetable(timing_tbl, fullfile(out_dir, ['split_half_timing' file_sfx '.csv']));

save(fullfile(out_dir, ['split_half_K_selection' file_sfx '.mat']), ...
    'selection_tbl', 'state_tbl', 'timing_tbl', 'obs_raw', 'obs_dm', ...
    'obs_weak_dm', 'null_raw', 'null_dm', 'null_weak_dm', 'obs_mm', ...
    'obs_weak_mm', 'null_mm', 'obs_details', ...
    'null_details', 'split_a', 'split_b', 'K_values', 'n_splits', 'n_null', ...
    'base_options', 'seed', 'null_type', '-v7.3');

%% ---------- Figure ----------
panels = {'raw state FC', 'demeaned state FC'};
if has_meanmap
    panels{end + 1} = 'state mean (activation) maps';
end
n_pan = numel(panels);
f = figure('Visible', 'off', 'Position', [100 100 500 * n_pan 400]);
for p = 1:n_pan
    subplot(1, n_pan, p);
    hold on;
    if p == 1
        m = selection_tbl.mean_reproducibility;  sd = selection_tbl.sd_reproducibility;
        nm = selection_tbl.null_mean_raw;         nsd = selection_tbl.null_sd_raw;
    elseif p == 2
        m = selection_tbl.mean_demeaned;          sd = selection_tbl.sd_demeaned;
        nm = selection_tbl.null_mean_demeaned;    nsd = selection_tbl.null_sd_demeaned;
    else
        m = selection_tbl.mean_meanmap;           sd = selection_tbl.sd_meanmap;
        nm = selection_tbl.null_mean_meanmap;     nsd = selection_tbl.null_sd_meanmap;
    end
    errorbar(selection_tbl.K, m, sd, '-o', 'LineWidth', 1.5, ...
        'DisplayName', 'observed');
    if n_null > 0
        errorbar(selection_tbl.K, nm, nsd, '--s', 'LineWidth', 1.2, ...
            'DisplayName', null_label);
    end
    xlabel('Number of states (K)');
    ylabel('Split-half matched-state similarity');
    title(panels{p});
    legend('Location', 'best');
    grid on;
    hold off;
end
exportgraphics(f, fullfile(out_dir, ['split_half_K_selection' file_sfx '.png']), ...
    'Resolution', 300);
close(f);

%% ---------- Console summary ----------
disp(selection_tbl(:, {'K', 'mean_reproducibility', 'null_mean_raw', ...
    'mean_demeaned', 'null_mean_demeaned', 'delta_demeaned', ...
    'delta_median_demeaned', 'p_mw_demeaned', 'weakest_state_demeaned'}));
if has_meanmap
    disp(selection_tbl(:, {'K', 'mean_meanmap', 'null_mean_meanmap', ...
        'delta_median_meanmap', 'p_mw_meanmap', 'weakest_state_meanmap'}));
end

ok = ~timing_tbl.failed & timing_tbl.seconds_both_halves > 0;
if any(ok)
    fprintf('Median time per job (both halves): %.1f min; slowest K=%d.\n', ...
        median(timing_tbl.seconds_both_halves(ok)) / 60, ...
        timing_tbl.K(find(timing_tbl.seconds_both_halves == ...
        max(timing_tbl.seconds_both_halves), 1)));
end
if any(timing_tbl.failed)
    warning('%d jobs failed; see split_half_timing.csv and the .mat details.', ...
        nnz(timing_tbl.failed));
end
if ~isempty(state_rows) && any(strcmp(state_rows(:, 10), 'equal'))
    warning(['Fractional occupancy was not found in the output of ' ...
        'hmm_fit_raw_timeseries for some fits; demeaning used equal ' ...
        'weights across states. Add R.fo (1 x K) to use occupancy weights.']);
end
end


%% ========================================================================
function [d, msg] = local_run_job(Ts_list, split_a, split_b, cond, rep, K, ...
    job_seed, base_options, job_dir, seed, null_type, null_tag)
% One job = fit both halves for one (condition, repetition, K) and score.
% Reuses a matching checkpoint if present; saves one after a successful fit.
if cond == 1
    Ts = Ts_list;
    tag = 'obs';
else
    Ts = local_make_surrogates(Ts_list, seed, rep, null_type);
    tag = null_tag;
end
idx_a = split_a{rep};
idx_b = split_b{rep};

sig = struct('K', K, 'seed', job_seed, 'idx_a', idx_a(:).', ...
    'idx_b', idx_b(:).', 'cond', tag, 'options', base_options);
ckpt = fullfile(job_dir, sprintf('%s_rep%02d_K%02d.mat', tag, rep, K));

d = local_load_job(ckpt, sig);
reused = ~isempty(d);
if ~reused
    d = local_fit_and_score(Ts, idx_a, idx_b, base_options, K, job_seed, rep);
    if isempty(d.error_message)
        local_save_job(ckpt, d, sig);
    end
end

if reused
    status = 'reused checkpoint';
elseif isempty(d.error_message)
    status = sprintf('%.1f min', d.seconds / 60);
else
    status = ['FAILED: ' d.error_message];
end
msg = sprintf('[%s] %-4s rep %2d  K=%2d  %s', ...
    char(datetime('now', 'Format', 'HH:mm')), tag, rep, K, status);
end


function d = local_fit_and_score(Ts, idx_a, idx_b, base_options, K, ...
    seed_base, rep_idx)

options = base_options;
options.K = K;

d = struct();
d.repetition = rep_idx;
d.K = K;
d.subject_idx_a = idx_a;
d.subject_idx_b = idx_b;
d.seed_a = seed_base + 1;
d.seed_b = seed_base + 2;
d.error_message = '';
t0 = tic;

try
    rng(seed_base + 1, 'twister');
    Ra = hmm_fit_raw_timeseries(Ts(idx_a), options, idx_a);

    rng(seed_base + 2, 'twister');
    Rb = hmm_fit_raw_timeseries(Ts(idx_b), options, idx_b);

    [fo_a, src_a] = local_get_fo(Ra, K);
    [fo_b, src_b] = local_get_fo(Rb, K);

    [sim_raw, pairs_raw, matched_raw] = ...
        hmm_match_states_fc(Ra.state_fc, Rb.state_fc, [], [], false);
    [sim_dm, pairs_dm, matched_dm] = ...
        hmm_match_states_fc(Ra.state_fc, Rb.state_fc, fo_a, fo_b, true);

    d.similarity_raw = sim_raw;
    d.pairs_raw = pairs_raw;
    d.matched_raw = matched_raw;
    d.similarity_dm = sim_dm;
    d.pairs_dm = pairs_dm;
    d.matched_dm = matched_dm;
    d.fo_a = fo_a;
    d.fo_b = fo_b;
    if strcmp(src_a, 'equal') || strcmp(src_b, 'equal')
        d.fo_source = 'equal';
    else
        d.fo_source = 'fitted';
    end

    % Activation (state-mean) maps, only when means are modelled
    % (zeromean = 0). Matched separately from FC: an activation state can
    % share its FC with another state and differ only in its mean pattern.
    d.matched_mean = NaN(K, 1);
    d.pairs_mean = [];
    d.mean_source = 'not modelled';
    if isfield(options, 'zeromean') && options.zeromean == 0
        n_roi = size(Ts{idx_a(1)}, 2);
        [Ma, msrc_a] = local_get_means(Ra, K, n_roi);
        [Mb, ~] = local_get_means(Rb, K, n_roi);
        if ~isempty(Ma) && ~isempty(Mb)
            [sim_m, pairs_m, matched_m] = local_match_vectors(Ma, Mb);
            d.similarity_mean = sim_m;
            d.pairs_mean = pairs_m;
            d.matched_mean = matched_m;
            d.mean_source = msrc_a;
        else
            d.mean_source = 'unavailable';
        end
    end

catch ME
    warning('Repetition %d, K=%d failed: %s', rep_idx, K, ...
        getReport(ME, 'basic', 'hyperlinks', 'off'));
    d.error_identifier = ME.identifier;
    d.error_message = ME.message;
    if isempty(d.error_message)
        d.error_message = 'unknown error';
    end
end
d.seconds = toc(t0);
end


function Ts_null = local_make_surrogates(Ts_list, seed, rep, null_type)
% Deterministic surrogates for null repetition rep. The phase null keeps its
% original seed (seed + 999 + rep) so earlier checkpoints remain valid.
if strcmp(null_type, 'iaaft')
    stream = RandStream('mt19937ar', 'Seed', seed + 1999 + rep);
else
    stream = RandStream('mt19937ar', 'Seed', seed + 999 + rep);
end
Ts_null = cell(size(Ts_list));
for s = 1:numel(Ts_list)
    if strcmp(null_type, 'iaaft')
        Ts_null{s} = hmm_iaaft_surrogate(Ts_list{s}, stream);
    else
        Ts_null{s} = hmm_phase_randomise(Ts_list{s}, stream);
    end
end
end


function d = local_load_job(ckpt, sig)
d = [];
if ~exist(ckpt, 'file')
    return;
end
try
    L = load(ckpt, 'd', 'sig');
    if isfield(L, 'sig') && isequaln(L.sig, sig) && isfield(L, 'd')
        d = L.d;
    end
catch
    d = [];
end
end


function local_save_job(ckpt, d, sig) %#ok<INUSD>
% save() cannot be called directly inside parfor. Write to a temporary file,
% then rename, so a job killed mid-write leaves no corrupt checkpoint.
tmp = [ckpt '.tmp.mat'];
save(tmp, 'd', 'sig', '-v7');
movefile(tmp, ckpt, 'f');
end


function [fo, source] = local_get_fo(R, K)
% Fractional occupancy of each state, from whichever field the fitting
% function provides. Falls back to equal weights.
fo = [];
source = 'equal';
candidates = {'fo', 'FO', 'fractional_occupancy', 'FractionalOccupancy'};
for c = 1:numel(candidates)
    if isfield(R, candidates{c})
        v = R.(candidates{c});
        if isnumeric(v) && ~isempty(v)
            if isvector(v) && numel(v) == K
                fo = v(:).';
            elseif size(v, 2) == K
                fo = mean(v, 1, 'omitnan');     % subjects x K
            end
        end
        if ~isempty(fo)
            break;
        end
    end
end
if isempty(fo) && isfield(R, 'Gamma') && size(R.Gamma, 2) == K
    fo = mean(R.Gamma, 1);
end
if ~isempty(fo) && all(isfinite(fo)) && sum(fo) > 0
    fo = fo / sum(fo);
    source = 'fitted';
else
    fo = ones(1, K) / K;
end
end


function [M, source] = local_get_means(R, K, n_roi)
% State mean (activation) maps as a K x n_roi matrix, from whichever field
% the fitting function provides, or from HMM-MAR's getMean if R.hmm exists.
% Maps must be in ROI space: PCA is fitted separately in each half, so
% PCA-space means are not comparable between halves.
M = [];
source = 'unavailable';
candidates = {'state_mean', 'state_means', 'means', 'mean_state'};
for c = 1:numel(candidates)
    if isfield(R, candidates{c})
        v = R.(candidates{c});
        if iscell(v) && numel(v) == K
            M = cell2mat(cellfun(@(x) x(:).', v(:), 'UniformOutput', false));
        elseif isnumeric(v) && size(v, 1) == K && size(v, 2) ~= K
            M = v;
        elseif isnumeric(v) && size(v, 2) == K
            M = v.';
        end
        if ~isempty(M)
            source = candidates{c};
            break;
        end
    end
end
if isempty(M) && isfield(R, 'hmm') && exist('getMean', 'file') == 2
    try
        rows = cell(K, 1);
        for k = 1:K
            m = getMean(R.hmm, k);
            rows{k} = m(:).';
        end
        M = vertcat(rows{:});
        source = 'getMean';
    catch
        M = [];
    end
end
if ~isempty(M) && size(M, 2) ~= n_roi
    A = [];
    if isfield(R, 'hmm') && isfield(R.hmm, 'train') && isfield(R.hmm.train, 'A')
        A = R.hmm.train.A;
    end
    if ~isempty(A) && size(A, 1) == n_roi && size(A, 2) == size(M, 2)
        M = M * A.';                          % PCA space -> ROI space
        source = [source '+projected'];
    else
        M = [];
        source = 'wrong dimension';
    end
end
end


function [sim, pairs, matched] = local_match_vectors(Ma, Mb)
% Match state mean maps between halves (Hungarian) by COSINE similarity.
% Uncentred on purpose: Pearson across ROIs would remove a near-uniform
% global co-activation map and score a real global state as noise.
K = size(Ma, 1);
na = sqrt(sum(Ma .^ 2, 2));
nb = sqrt(sum(Mb .^ 2, 2));
sim = (Ma * Mb.') ./ max(na * nb.', eps);
sim(na <= eps, :) = -1;
sim(:, nb <= eps) = -1;
sim(~isfinite(sim)) = -1;
sim = max(min(sim, 1), -1);
pairs = matchpairs(1 - sim, 3);
if size(pairs, 1) ~= K
    error('Mean-map matching returned %d pairs; expected %d.', size(pairs, 1), K);
end
pairs = sortrows(pairs, 1);
matched = sim(sub2ind([K, K], pairs(:, 1), pairs(:, 2)));
end


function p = local_mw_greater(x, y)
% One-sided Mann-Whitney U test (x > y): normal approximation with tie and
% continuity corrections, base MATLAB only. NaN if a sample has < 2 values.
x = x(isfinite(x));
y = y(isfinite(y));
n1 = numel(x);
n2 = numel(y);
if n1 < 2 || n2 < 2
    p = NaN;
    return;
end
v = [x(:); y(:)];
N = n1 + n2;
[s, idx] = sort(v);
r = zeros(N, 1);
tie_term = 0;
i = 1;
while i <= N
    j = i;
    while j < N && s(j + 1) == s(i)
        j = j + 1;
    end
    r(idx(i:j)) = (i + j) / 2;
    t = j - i + 1;
    tie_term = tie_term + t ^ 3 - t;
    i = j + 1;
end
U = sum(r(1:n1)) - n1 * (n1 + 1) / 2;
sigma = sqrt(n1 * n2 / 12 * ((N + 1) - tie_term / (N * (N - 1))));
if sigma <= 0
    p = NaN;
    return;
end
z = (U - n1 * n2 / 2 - 0.5) / sigma;
p = 0.5 * erfc(z / sqrt(2));
end


function m = local_colmedian(X, nK)
if isempty(X)
    m = NaN(1, nK);
else
    m = median(X, 1, 'omitnan');
end
end


function rows = local_state_rows(condition, rep_idx, K, d)
metrics = {'raw', 'demeaned'};
pairs = {d.pairs_raw, d.pairs_dm};
sims = {d.matched_raw, d.matched_dm};
if isfield(d, 'pairs_mean') && ~isempty(d.pairs_mean)
    metrics{end + 1} = 'meanmap';
    pairs{end + 1} = d.pairs_mean;
    sims{end + 1} = d.matched_mean;
end
rows = cell(numel(metrics) * K, 10);
r = 0;
for m = 1:numel(metrics)
    for k = 1:K
        r = r + 1;
        a = pairs{m}(k, 1);
        b = pairs{m}(k, 2);
        rows(r, :) = {condition, rep_idx, K, metrics{m}, a, b, ...
            sims{m}(k), d.fo_a(a), d.fo_b(b), d.fo_source};
    end
end
end


function m = local_colmean(X, nK)
if isempty(X)
    m = NaN(1, nK);
else
    m = mean(X, 1, 'omitnan');
end
end


function s = local_colsd(X, nK)
if size(X, 1) < 2
    s = NaN(1, nK);
else
    s = std(X, 0, 1, 'omitnan');
end
end


function [idx_a, idx_b] = local_stratified_half_split(groups, stream)
% Identical to the original so the same seed gives the same partitions.
group_names = categories(groups);
idx_a = zeros(0, 1);
idx_b = zeros(0, 1);

for g = 1:numel(group_names)
    idx = find(groups == group_names{g});
    idx = idx(randperm(stream, numel(idx)));

    n_group = numel(idx);
    n_a = floor(n_group / 2);
    if mod(n_group, 2) == 1 && rand(stream) < 0.5
        n_a = n_a + 1;
    end

    idx_a = [idx_a; idx(1:n_a)]; %#ok<AGROW>
    idx_b = [idx_b; idx(n_a + 1:end)]; %#ok<AGROW>
end

idx_a = idx_a(randperm(stream, numel(idx_a)));
idx_b = idx_b(randperm(stream, numel(idx_b)));
end
