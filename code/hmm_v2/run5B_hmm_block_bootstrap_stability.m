%% ==============================================================
%% run5B_hmm_block_bootstrap_stability.m
%% ==============================================================
% Block-bootstrap stability of the subject x state FC networks that run5
% analyses, on the same fit, with the same network definition and the same
% inclusion threshold.
%
% Same fit       reads raw_hmm_results.mat from run4e and checks its fit ID
% Same networks  per-mouse state FC from hmm_subject_state_fc.m (HMM-MAR's
%                state statistic) and BCT threshold_proportional on the
%                positive edges, exactly as run5
% Same threshold cfg.min_effective_n (hmm_config.m), used by run5 and here
%
% Outcomes per mouse x state
%   Neff, posterior mass, p/Neff, rank and entropy effective rank of the state
%   scatter; edge-set Jaccard and nodal-strength Spearman rho between each
%   resample's network and the full-data network (median, mean, 2.5-97.5%);
%   bootstrap success fraction and the analyzable flag (as before).
% Diagnostics (from the Python v2 port)
%   resample vs resample   Jaccard and rho between disjoint pairs of
%                          successful resamples (less optimistic than vs-full)
%   split half             first vs second half of the scan
%   failure accounting     resamples failing the Neff criterion or the network
%   unfiltered             the same summaries keeping all resamples
%   switching descriptors  from the group Viterbi path (R.vpath, as run4d):
%                          visits, mean dwell, transitions, max FO, near-static
%
% Resampling: circular block bootstrap (Politis & Romano 1992) of paired rows
% of [time series, state probabilities]; block length ceil(T^(1/3)) by default.
% Each mouse has its own seed (cfg.boot_seed + subject ID), so results do not
% depend on the order mice are processed.
%
% Published code called: HMM-MAR (standardisedata, through
% hmm_subject_state_fc.m); BCT (threshold_proportional, strengths_und);
% MATLAB Statistics Toolbox (corr 'Spearman', prctile, iqr).
% Changes from the previous version are listed in CHANGES.md.

clc;
clear;
close all;

%% -------------------- Configuration (hmm_config.m) --------------------
cfg = hmm_config();
model_dir = cfg.model_dir;
hmm_results_path = fullfile(model_dir, 'raw_hmm_results.mat');
out_dir = fullfile(model_dir, 'hmm_bootstrap_stability');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

addpath(genpath(cfg.hmmmar_dir));
if ~isempty(cfg.bct_dir), addpath(cfg.bct_dir); end
assert(exist('threshold_proportional', 'file') == 2 && exist('strengths_und', 'file') == 2, ...
    'BCT (threshold_proportional, strengths_und) is not on the path; set HMM_BCT.');

loaded = load(hmm_results_path, 'R');
R = loaded.R;
assert(all(isfield(R, {'vpath', 'lifetimes_TR', 'maxFO', 'hmm'})), ...
    '%s was written by the previous run4e (no Viterbi lifetimes or max FO); rerun run4e first.', ...
    hmm_results_path);
fit_id = hmm_fit_id(R.Gamma);
if isfield(R, 'fit_id')
    assert(strcmp(R.fit_id, fit_id), 'raw_hmm_results.mat: Gamma does not match its stored fit ID.');
else
    warning('raw_hmm_results.mat has no stored fit ID (previous run4e). Computed: %s', fit_id);
end

K = R.K;
S = numel(R.subject_ids);
p = R.n_roi;
if isfield(R.hmm.train, 'A') && ~isempty(R.hmm.train.A)
    fprintf(['Model fitted after PCA: networks use the ROI-space state statistic ' ...
        '(see hmm_subject_state_fc.m).\n']);
    warning('off', 'hmm_subject_state_fc:prePCA');   % otherwise repeated for every resample
end
fprintf('Fit %s: %d mice, %d states, %d ROIs; %d resamples; density %.2f; min Neff %g\n', ...
    fit_id(1:12), S, K, p, cfg.n_bootstrap, cfg.graph_density, cfg.min_effective_n);

%% -------------------- Per mouse --------------------
rows = cell(S, 1);
for s = 1:S
    X = double(readmatrix(fullfile(cfg.data_dir, sprintf(cfg.file_pattern, R.subject_ids(s)))));
    assert(isequal(size(X, 1), size(R.Gamma{s}, 1)) && size(X, 2) == p, ...
        'Time series / Gamma mismatch for subject %d.', R.subject_ids(s));
    rows{s} = local_subject(s, X, R, cfg);
    fprintf('  mouse %d/%d (ID %d) done\n', s, S, R.subject_ids(s));
end
subject_state_table = vertcat(rows{:});
subject_state_table.fit_id = repmat(string(fit_id), height(subject_state_table), 1);
writetable(subject_state_table, fullfile(out_dir, 'bootstrap_stability_subject_state.csv'));

%% -------------------- State summary --------------------
state_summary_table = local_state_summary(subject_state_table, K);
state_summary_table.fit_id = repmat(string(fit_id), height(state_summary_table), 1);
writetable(state_summary_table, fullfile(out_dir, 'bootstrap_stability_by_state.csv'));

save(fullfile(out_dir, 'hmm_block_bootstrap_stability.mat'), 'subject_state_table', ...
    'state_summary_table', 'cfg', 'fit_id', '-v7.3');
disp(state_summary_table);
fprintf('Written to %s\n', out_dir);


%% ==============================================================
%% Local functions
%% ==============================================================
function T = local_subject(s, X, R, cfg)
K = R.K;
p = size(X, 2);
G = double(R.Gamma{s});
Tn = size(X, 1);
if isnan(cfg.block_length), L = max(2, ceil(Tn^(1/3))); else, L = min(Tn, cfg.block_length); end
min_neff = cfg.min_effective_n;
B = cfg.n_bootstrap;
rng(cfg.boot_seed + double(R.subject_ids(s)), 'twister');
pair_stream = RandStream('mt19937ar', 'Seed', cfg.boot_seed + 1e6 + double(R.subject_ids(s)));

[fc0, neff0, sc0] = hmm_subject_state_fc({X}, {G}, R.hmm.train, 0);

% switching descriptors from the group Viterbi path (same states as run4d)
vp = double(R.vpath{s}(:));
n_trans = nnz(diff(vp));

v = struct();
for k = 1:K
    w = G(:, k);
    r = struct();
    r.subject_id = R.subject_ids(s);
    r.state = k;
    r.group = string(R.groups(s));
    r.n_timepoints = Tn;
    r.block_length = L;
    r.FO = mean(w);
    r.map_count = nnz(vp == k);                 % R.vpath labels, as run5's map_TR_count
    r.n_visits = numel(R.lifetimes_TR{s, k});
    r.mean_dwell_TR = mean_or_nan(R.lifetimes_TR{s, k});
    r.subject_n_transitions = n_trans;
    r.subject_max_FO = R.maxFO(s);
    r.near_static = r.FO >= cfg.near_static_fo;
    r.Neff = neff0(1, k);
    r.posterior_mass = sum(w);
    r.p_over_Neff = p / max(r.Neff, eps);
    r.passes_Neff = r.Neff >= min_neff && ...
        (isnan(cfg.min_posterior_mass) || r.posterior_mass >= cfg.min_posterior_mass);
    r.covariance_rank = rank(sc0{1, k});
    r.covariance_effective_rank = entropy_rank(sc0{1, k});
    [mask0, str0] = state_network(fc0{1, k}, cfg.graph_density);
    r.valid_original_network = ~isempty(mask0);
    r.original_positive_edges = NaN;
    r.original_graph_density = NaN;
    if r.valid_original_network
        r.original_positive_edges = nnz(mask0);
        r.original_graph_density = nnz(mask0) / numel(mask0);
    end
    v.r{k} = r;
    v.mask0{k} = mask0;
    v.str0{k} = str0;
end

% ---- split half (no shared time points) ----
h = floor(Tn / 2);
parts = {1:h, h + 1:Tn};
for k = 1:K
    v.r{k}.split_half_Neff_1 = NaN; v.r{k}.split_half_Neff_2 = NaN;
    v.r{k}.split_half_valid = false;
    v.r{k}.split_half_edge_jaccard = NaN; v.r{k}.split_half_nodal_rank_correlation = NaN;
end
half = cell(2, 1);
for hh = 1:2
    [fch, neffh] = hmm_subject_state_fc({X(parts{hh}, :)}, {G(parts{hh}, :)}, R.hmm.train, 0);
    half{hh} = struct('fc', {fch}, 'neff', neffh);
end
for k = 1:K
    r = v.r{k};
    r.split_half_Neff_1 = half{1}.neff(1, k);
    r.split_half_Neff_2 = half{2}.neff(1, k);
    if r.passes_Neff && r.valid_original_network && all([r.split_half_Neff_1, r.split_half_Neff_2] >= min_neff)
        [m1, s1] = state_network(half{1}.fc{1, k}, cfg.graph_density);
        [m2, s2] = state_network(half{2}.fc{1, k}, cfg.graph_density);
        if ~isempty(m1) && ~isempty(m2)
            r.split_half_valid = true;
            r.split_half_edge_jaccard = jac(m1, m2);
            r.split_half_nodal_rank_correlation = corr(s1, s2, 'Type', 'Spearman');
        end
    end
    v.r{k} = r;
end

% ---- bootstrap ----
todo = cellfun(@(r) r.passes_Neff && r.valid_original_network, v.r);
Jf = NaN(B, K); Rf = NaN(B, K); Ja = NaN(B, K); Ra = NaN(B, K); Nb = NaN(B, K);
masks = cell(1, K); strengths = cell(1, K);
n_upper = p * (p - 1) / 2;
for k = find(todo)
    masks{k} = false(B, n_upper);
    strengths{k} = NaN(B, p);
end
if any(todo)
    for b = 1:B
        idx = circular_block_indices(Tn, L);
        [fcb, neffb] = hmm_subject_state_fc({X(idx, :)}, {G(idx, :)}, R.hmm.train, 0);
        for k = find(todo)
            Nb(b, k) = neffb(1, k);
            [mb, sb] = state_network(fcb{1, k}, cfg.graph_density);
            if isempty(mb), continue; end
            Ja(b, k) = jac(v.mask0{k}, mb);
            Ra(b, k) = corr(v.str0{k}, sb, 'Type', 'Spearman');
            masks{k}(b, :) = mb';
            strengths{k}(b, :) = sb';
            if neffb(1, k) >= min_neff
                Jf(b, k) = Ja(b, k);
                Rf(b, k) = Ra(b, k);
            end
        end
    end
end

% ---- summaries ----
for k = 1:K
    r = v.r{k};
    ok = isfinite(Jf(:, k)) & isfinite(Rf(:, k));
    ok_all = isfinite(Ja(:, k)) & isfinite(Ra(:, k));
    r.bootstrap_success_count = nnz(ok);
    r.bootstrap_success_fraction = nnz(ok) / B;
    r.n_fail_boot_Neff = nnz(Nb(:, k) < min_neff);
    r.n_fail_boot_network = nnz(~(Nb(:, k) < min_neff) & ~ok & isfinite(Nb(:, k)));
    r.median_boot_Neff = median(Nb(isfinite(Nb(:, k)), k));
    [r.median_edge_jaccard, r.mean_edge_jaccard, r.edge_jaccard_CI_low, r.edge_jaccard_CI_high] = ...
        summ(Jf(ok, k));
    [r.median_nodal_strength_rank_correlation, r.mean_nodal_strength_rank_correlation, ...
        r.nodal_rank_correlation_CI_low, r.nodal_rank_correlation_CI_high] = summ(Rf(ok, k));
    r.n_unfiltered = nnz(ok_all);
    r.median_edge_jaccard_unfiltered = median_or_nan(Ja(ok_all, k));
    r.median_nodal_rank_unfiltered = median_or_nan(Ra(ok_all, k));
    % resample vs resample (disjoint pairs of successful resamples)
    r.n_resample_pairs = 0;
    r.median_edge_jaccard_rr = NaN; r.edge_jaccard_rr_CI_low = NaN; r.edge_jaccard_rr_CI_high = NaN;
    r.median_nodal_rank_rr = NaN; r.nodal_rank_rr_CI_low = NaN; r.nodal_rank_rr_CI_high = NaN;
    succ = find(ok);
    if numel(succ) >= 2
        succ = succ(randperm(pair_stream, numel(succ)));
        n_pairs = min(cfg.max_resample_pairs, floor(numel(succ) / 2));
        a = succ(1:2:2 * n_pairs); c = succ(2:2:2 * n_pairs);
        Jrr = sum(masks{k}(a, :) & masks{k}(c, :), 2) ./ max(sum(masks{k}(a, :) | masks{k}(c, :), 2), 1);
        Rrr = arrayfun(@(i) corr(strengths{k}(a(i), :)', strengths{k}(c(i), :)', 'Type', 'Spearman'), ...
            (1:n_pairs)');
        r.n_resample_pairs = n_pairs;
        [r.median_edge_jaccard_rr, ~, r.edge_jaccard_rr_CI_low, r.edge_jaccard_rr_CI_high] = summ(Jrr);
        [r.median_nodal_rank_rr, ~, r.nodal_rank_rr_CI_low, r.nodal_rank_rr_CI_high] = summ(Rrr);
    end
    r.analyzable = r.passes_Neff && r.valid_original_network && ...
        r.bootstrap_success_fraction >= cfg.min_bootstrap_success_fraction;
    v.r{k} = r;
end
T = struct2table([v.r{:}]);
end

function [mask, strength] = state_network(fc, density)
% Positive-edge proportional-threshold network, as in run5 (BCT).
mask = []; strength = [];
if any(~isfinite(fc(:))), return; end
A = (fc + fc') / 2;
A(1:size(A, 1) + 1:end) = 0;
A = max(A, 0);
if ~any(A(:) > 0), return; end
W = threshold_proportional(A, density);
mask = W(triu(true(size(W)), 1)) > 0;
strength = strengths_und(W)';
end

function j = jac(a, b)
u = nnz(a | b);
if u == 0, j = NaN; else, j = nnz(a & b) / u; end
end

function idx = circular_block_indices(T, L)
% Circular block bootstrap (Politis & Romano 1992), as in the previous run5B.
nb = ceil(T / L);
starts = randi(T, nb, 1);
idx = mod((starts - 1) + (0:L - 1), T) + 1;
idx = reshape(idx', [], 1);
idx = idx(1:T);
end

function er = entropy_rank(C)
% Entropy effective rank, exp(-sum q log q) of normalised eigenvalues (Roy & Vetterli 2007).
ev = max(real(eig((C + C') / 2)), 0);
if sum(ev) <= eps, er = NaN; return; end
q = ev(ev > 0) / sum(ev);
er = exp(-sum(q .* log(q)));
end

function [md, mn, lo, hi] = summ(x)
x = x(isfinite(x));
if isempty(x), md = NaN; mn = NaN; lo = NaN; hi = NaN; return; end
md = median(x); mn = mean(x);
ci = prctile(x, [2.5, 97.5]); lo = ci(1); hi = ci(2);
end

function m = median_or_nan(x)
x = x(isfinite(x));
if isempty(x), m = NaN; else, m = median(x); end
end

function m = mean_or_nan(x)
if isempty(x), m = NaN; else, m = mean(x); end
end

function S = local_state_summary(t, K)
rows = cell(K, 1);
for k = 1:K
    s = t(t.state == k, :);
    a = s(s.analyzable, :);
    sh = a(a.split_half_valid, :);
    r = struct();
    r.state = k;
    r.n_subject_states_total = height(s);
    r.n_passing_Neff = nnz(s.passes_Neff);
    r.n_analyzable = nnz(s.analyzable);
    r.n_near_static = nnz(s.near_static);
    r.median_Neff = median_or_nan(s.Neff);
    r.IQR_Neff = iqr_or_nan(s.Neff);
    r.median_p_over_Neff = median_or_nan(s.p_over_Neff);
    r.median_covariance_rank = median_or_nan(s.covariance_rank);
    r.median_covariance_effective_rank = median_or_nan(s.covariance_effective_rank);
    r.median_edge_jaccard = median_or_nan(a.median_edge_jaccard);
    r.IQR_edge_jaccard = iqr_or_nan(a.median_edge_jaccard);
    r.median_nodal_strength_rank_correlation = median_or_nan(a.median_nodal_strength_rank_correlation);
    r.IQR_nodal_strength_rank_correlation = iqr_or_nan(a.median_nodal_strength_rank_correlation);
    r.median_edge_jaccard_rr = median_or_nan(a.median_edge_jaccard_rr);
    r.median_nodal_rank_rr = median_or_nan(a.median_nodal_rank_rr);
    r.n_split_half_valid = height(sh);
    r.median_split_half_edge_jaccard = median_or_nan(sh.split_half_edge_jaccard);
    r.median_split_half_nodal_rank = median_or_nan(sh.split_half_nodal_rank_correlation);
    r.median_edge_jaccard_unfiltered = median_or_nan(a.median_edge_jaccard_unfiltered);
    r.median_nodal_rank_unfiltered = median_or_nan(a.median_nodal_rank_unfiltered);
    r.percentage_passing_Neff = 100 * mean(s.passes_Neff);
    r.percentage_analyzable_subject_states = 100 * mean(s.analyzable);
    rows{k} = r;
end
S = struct2table([rows{:}]);
end

function q = iqr_or_nan(x)
x = x(isfinite(x));
if isempty(x), q = NaN; else, q = iqr(x); end
end
