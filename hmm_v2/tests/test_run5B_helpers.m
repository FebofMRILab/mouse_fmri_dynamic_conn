function results = test_run5B_helpers(hmmmar_dir, bct_dir)
% TEST_RUN5B_HELPERS  The updated run5B against the previous run5B, function by
% function, on simulated data. MATLAB or Octave; about 1 min.
%
%   r = test_run5B_helpers('/blue/febo/MATLAB/HMM-MAR-master', '/blue/febo/MATLAB/BCT')
%
% Both scripts' local functions are extracted unchanged (extract_local_functions.m)
% and called directly: the previous version from tests/legacy/, the updated one
% from run5B_hmm_block_bootstrap_stability.m.
%
%   B1  circular block bootstrap indices: identical for the same random state
%   B2  network: BCT threshold_proportional (updated) vs the previous exact-rank
%       code; same edges and nodal strengths, including exactly tied weights
%   B3  nodal-strength Spearman: corr(...,'Type','Spearman') vs the previous
%       tiedrank + corrcoef (MATLAB only; Octave has no corr 'Type')
%   B4  covariance rank (MATLAB rank vs the previous eigenvalue count) and
%       entropy effective rank (same formula), on rank-deficient matrices
%   B5  Kish effective n: hmm_subject_state_fc vs the previous function
%   B6  state FC: previous estimator vs hmm_subject_state_fc (report), and the
%       previous formula without within-state centring (must be equal)

here = fileparts(mfilename('fullpath'));
v2 = fileparts(here);
addpath(v2);
addpath(genpath(hmmmar_dir));
addpath(bct_dir);
tmp = tempname();
old_dir = fullfile(tmp, 'previous');
new_dir = fullfile(tmp, 'updated');
extract_local_functions(fullfile(here, 'legacy', 'run5B_hmm_block_bootstrap_stability_previous.m'), old_dir);
extract_local_functions(fullfile(v2, 'run5B_hmm_block_bootstrap_stability.m'), new_dir);
addpath(old_dir);
addpath(new_dir);
cleanup = onCleanup(@() local_cleanup(tmp, old_dir, new_dir));
is_octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
results = struct('check', {}, 'value', {}, 'pass', {});

% ---- B1 block indices ----
d = 0;
for T = [197, 300, 570]
    for L = [2, 6, 9, 13]
        set_seed(T + L);
        a = local_circular_block_indices(T, L);
        set_seed(T + L);
        b = circular_block_indices(T, L);
        d = max(d, max(abs(a(:) - b(:))));
    end
end
results(end + 1) = struct('check', 'B1 circular block indices, previous vs updated (max diff)', ...
    'value', d, 'pass', d == 0);

% ---- B2 networks ----
set_seed(1);
p = 82;
n_mat = 100;
n_bad = 0;
max_ds = 0;
for i = 1:n_mat
    X = randn(300, p) + 0.6 * randn(300, 1) * randn(1, p);
    C = corrcoef(X);
    if i > n_mat / 2                   % exactly tied weights
        C = round(C * 20) / 20;
        C(1:p + 1:end) = 1;
    end
    [W_old, ids_old, ok_old] = local_positive_proportional_graph(C, 0.15);
    [mask_new, str_new] = state_network(C, 0.15);
    U = find(triu(true(p), 1));
    ids_new = U(mask_new);
    if ~ok_old || isempty(mask_new) || ~isequal(sort(ids_old(:)), sort(ids_new(:)))
        n_bad = n_bad + 1;
    else
        max_ds = max(max_ds, max(abs(sum(W_old, 2) - str_new)));
    end
end
results(end + 1) = struct('check', sprintf(['B2 network, previous vs BCT: matrices with other edges ' ...
    '(of %d; %d with ties); max strength diff %.0g'], n_mat, n_mat / 2, max_ds), ...
    'value', n_bad, 'pass', n_bad == 0 && max_ds < 1e-12);

% ---- B3 Spearman ----
if is_octave
    results(end + 1) = struct('check', 'B3 Spearman corr Type (MATLAB only; skipped in Octave)', ...
        'value', NaN, 'pass', true);
else
    d = 0;
    for i = 1:200
        x = randn(p, 1); y = x + randn(p, 1);
        if mod(i, 2) == 0, x = round(x); y = round(y); end   % ties
        d = max(d, abs(local_spearman_rank_correlation(x, y) - corr(x, y, 'Type', 'Spearman')));
    end
    results(end + 1) = struct('check', 'B3 Spearman, previous vs corr Spearman (max diff, half with ties)', ...
        'value', d, 'pass', d < 1e-12);
end

% ---- B4 rank and effective rank ----
d_rank = 0;
d_er = 0;
for r_true = [5, 20, 60, 82]
    Z = randn(400, r_true) * randn(r_true, p);
    w = rand(400, 1);
    Sk = bsxfun(@times, Z, w)' * Z;
    d_rank = max(d_rank, abs(rank(Sk) - local_covariance_rank(Sk)));
    d_er = max(d_er, abs(entropy_rank(Sk) - local_effective_rank(Sk)));
end
results(end + 1) = struct('check', 'B4 rank: MATLAB rank vs previous count (max diff, ranks 5-82)', ...
    'value', d_rank, 'pass', d_rank == 0);
results(end + 1) = struct('check', 'B4 entropy effective rank, previous vs updated (max diff)', ...
    'value', d_er, 'pass', d_er < 1e-10);

% ---- B5 and B6 on one simulated mouse ----
T = 570;
X = randn(T, p) * diag(0.5 + rand(p, 1)) + 3;      % unequal scales, nonzero means
X = X + randn(T, 1) * randn(1, p);
G = rand(T, 3) .^ 4;
G = bsxfun(@rdivide, G, sum(G, 2));
train = struct('standardise', 1, 'zeromean', 1);
[fc, neff, sc] = hmm_subject_state_fc({X}, {G}, train, 0);
Xz = standardisedata(X, T, 1);
d_n = 0; d_r = 0; d_nc = 0; d_scale = 0; er = zeros(3, 2);
for k = 1:3
    w = G(:, k);
    d_n = max(d_n, abs(neff(1, k) - local_effective_n(w)) / neff(1, k));
    [C_old, R_old] = local_weighted_covariance(X, w);
    er(k, :) = [local_effective_rank(C_old), entropy_rank(sc{1, k})];
    [~, R_old_z] = local_weighted_covariance(Xz, w);
    d_scale = max(d_scale, max(abs(R_old(:) - R_old_z(:))));
    d_r = max(d_r, max(abs(R_old(:) - fc{1, k}(:))));
    % previous formula with the within-state centring removed
    Cnc = Xz' * bsxfun(@times, Xz, w) / (sum(w) - sum(w .^ 2) / sum(w));
    s = sqrt(diag(Cnc));
    Rnc = Cnc ./ (s * s');
    Rnc(1:p + 1:end) = 1;
    d_nc = max(d_nc, max(abs(Rnc(:) - fc{1, k}(:))));
end
results(end + 1) = struct('check', 'B5 Kish effective n, previous vs hmm_subject_state_fc (rel. max diff)', ...
    'value', d_n, 'pass', d_n < 1e-12);
results(end + 1) = struct('check', 'B6 previous FC on raw vs z-scored data (max |dr|; scale-free)', ...
    'value', d_scale, 'pass', d_scale < 1e-10);
results(end + 1) = struct('check', 'B6 state FC, previous run5B vs hmm_subject_state_fc (max |dr|)', ...
    'value', d_r, 'pass', true);
results(end + 1) = struct('check', sprintf(['B6 effective rank, previous (raw-scale covariance) vs ' ...
    'updated (z-scored scatter): %s vs %s'], mat2str(er(:, 1)', 3), mat2str(er(:, 2)', 3)), ...
    'value', max(abs(er(:, 1) - er(:, 2))), 'pass', true);
results(end + 1) = struct('check', 'B6 previous formula without within-state centring vs updated (max |dr|)', ...
    'value', d_nc, 'pass', d_nc < 1e-12);

% ---- report ----
fprintf('\n%-92s %12s  %s\n', 'Test', 'Value', 'Pass');
for i = 1:numel(results)
    if ~isempty(strfind(results(i).check, 'previous run5B vs')) || ...
            ~isempty(strfind(results(i).check, 'B6 effective rank')), tag = 'report';
    elseif isnan(results(i).value), tag = 'skipped';
    elseif results(i).pass, tag = 'PASS';
    else, tag = 'FAIL';
    end
    results(i).pass = double(results(i).pass);
    fprintf('%-92s %12.4g  %s\n', results(i).check, results(i).value, tag);
end
end

function set_seed(s)
if exist('OCTAVE_VERSION', 'builtin')
    rand('state', s); randn('state', s); %#ok<RAND>
else
    rng(s, 'twister');
end
end

function local_cleanup(tmp, a, b)
rmpath(a); rmpath(b);
if exist(tmp, 'dir'), rmdir(tmp, 's'); end
end
