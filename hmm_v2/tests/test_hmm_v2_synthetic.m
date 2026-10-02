function results = test_hmm_v2_synthetic(hmmmar_dir, bct_dir)
% TEST_HMM_V2_SYNTHETIC  Equivalence tests for the updated HMM code, on
% simulated data. Runs in MATLAB or Octave (>= 8 with the statistics package
% functions pca/mnrnd on the path).
%
%   results = test_hmm_v2_synthetic('/path/to/HMM-MAR', '/path/to/BCT')
%
% Tests
%   T1  per-subject state scatter (hmm_subject_state_fc), summed over subjects,
%       equals HMM-MAR's own XXGXX computed by setxx.m       -> must be equal
%   T2  fractional occupancy: getFractionalOccupancy equals the previous
%       mean-of-Gamma per subject                             -> must be equal
%   T3  new per-subject FC vs previous per-subject FC; the previous one is
%       recomputed on z-scored data without within-state centring to show
%       the difference is the centring only                   -> report / equal
%   T4  group state FC: getFuncConn vs previous pooled estimator (report)
%   T5  dwell times: Viterbi visits (getStateLifeTimes) vs previous
%       most-probable-state runs (report)
%   T6  graph metrics: BCT threshold_proportional vs previous rank threshold,
%       and every metric, on the same matrices                -> must be equal
% A test marked "must be equal" fails if the maximum absolute difference
% exceeds 1e-8.

here = fileparts(mfilename('fullpath'));
addpath(genpath(hmmmar_dir));
addpath(bct_dir);
addpath(fileparts(here));                 % hmm_v2 functions
addpath(fullfile(here, 'legacy'));        % previous versions (verbatim)
is_octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
tol = 1e-8;
results = struct('test', {}, 'value', {}, 'pass', {}, 'note', {});

% ---- simulated data: 6 subjects, 30 ROIs, 2 covariance states ----
if is_octave
    rand('seed', 7); randn('seed', 7);
else
    rng(7, 'twister');
end
S = 6; T = 400; n = 30; dwell = 40;
L1 = randn(n, 4); L2 = randn(n, 4);
Ts = cell(S, 1);
for s = 1:S
    gain = 0.5 + rand;                    % subject-specific amplitude
    x = zeros(T, n);
    for t = 1:T
        if mod(floor((t - 1) / dwell), 2) == 0, L = L1; else, L = L2; end
        x(t, :) = gain * ((L * randn(4, 1))' + 0.7 * randn(1, n));
    end
    Ts{s} = x;
end
opts = struct('K', 2, 'lowrank', 5, 'standardise', 1, 'zeromean', 1, ...
    'covtype', 'full', 'order', 0, 'inittype', 'hmmmar', 'initrep', 2, ...
    'initcyc', 10, 'cyc', 100, 'verbose', 0, 'useParallel', 0, 'Fs', 0.5, ...
    'DirichletDiag', 10);

R = hmm_fit_raw_timeseries(Ts, opts, (1:S)');
K = R.K;

% ---- T1: summed per-subject scatter == HMM-MAR setxx XXGXX ----
[fc_new, eff_n, scat] = hmm_subject_state_fc(Ts, R.Gamma, R.hmm.train, 0);
XXGXX = hmmmar_xxgxx(R.hmm, vertcat(Ts{:}), R.Tvec, vertcat(R.Gamma{:}));
d = 0;
for k = 1:K
    Ssum = zeros(n);
    for s = 1:S, Ssum = Ssum + scat{s, k}; end
    d = max(d, max(abs(Ssum(:) - XXGXX{k}(:))) / max(abs(XXGXX{k}(:))));
end
results = add(results, 'T1 sum of per-subject scatter vs HMM-MAR XXGXX (rel. max diff)', ...
    d, d < tol, 'setxx.m line: XXGXX{k} = bsxfun(@times, XX, Gamma(:,k))'' * XX');

% ---- T2: FO ----
FO_old = cell2mat(cellfun(@(G) mean(G, 1), R.Gamma, 'UniformOutput', false));
d = max(abs(FO_old(:) - R.FO_subj(:)));
results = add(results, 'T2 getFractionalOccupancy vs mean of Gamma (max diff)', d, d < tol, '');

% ---- T3: per-subject FC, new vs previous ----
fc_old = compute_subject_state_fc_robust(Ts, R.Gamma, 0);
d_old = 0; d_nocent = 0;
for s = 1:S
    Z = standardisedata(Ts{s}, T, 1);
    for k = 1:K
        a = fc_new{s, k}; b = fc_old{s, k};
        d_old = max(d_old, max(abs(a(:) - b(:))));
        w = R.Gamma{s}(:, k);
        C = (Z .* w)' * Z; C = C ./ (sqrt(diag(C)) * sqrt(diag(C))');   % previous formula, no centring
        d_nocent = max(d_nocent, max(abs(a(:) - C(:))));
    end
end
results = add(results, 'T3a per-subject FC new vs previous (max |dr|)', d_old, NaN, ...
    'difference = within-state centring of the previous version');
results = add(results, 'T3b new vs previous formula without centring (max |dr|)', d_nocent, ...
    d_nocent < tol, 'confirms centring is the only difference');

% ---- T4: group FC ----
R_old = legacy_hmm_fit_raw_timeseries_fc_only(Ts, R.Gamma);
ut = triu(true(n), 1);
d4 = 0;
for k = 1:K
    Ssum = zeros(n);
    for s = 1:S, Ssum = Ssum + scat{s, k}; end
    pooled_z = Ssum ./ (sqrt(diag(Ssum)) * sqrt(diag(Ssum))');
    d4 = max(d4, max(abs(pooled_z(ut) - R.state_fc{k}(ut))));
    results = add(results, sprintf('T4 state %d: r(R.state_fc, previous group FC)', k), ...
        corr_(R.state_fc{k}(ut), R_old{k}(ut)), NaN, sprintf('r(R.state_fc, getFuncConn) = %.4f', ...
        corr_(R.state_fc{k}(ut), R.state_fc_model{k}(ut))));
end
results = add(results, 'T4 R.state_fc equals the pooled per-subject scatter (max diff)', d4, d4 < tol, '');

% ---- T5: dwell ----
dw_old = NaN(S, K);
for s = 1:S
    [~, lab] = max(R.Gamma{s}, [], 2);
    for k = 1:K, dw_old(s, k) = run_len(lab == k); end
end
agree = mean(cellfun(@(G, v) mean(argmax_(G) == double(v)), R.Gamma, R.vpath));
results = add(results, 'T5 mean dwell (TR): Viterbi minus most-probable-state', ...
    mean(R.dwell_TR(:) - dw_old(:), 'omitnan'), NaN, ...
    sprintf('labels agree at %.1f%% of time points', 100 * agree));

% ---- T6: graph metrics, previous vs BCT-based function ----
v2_metrics = extract_v2_graph_function(fileparts(here), is_octave);
if is_octave
    legacy_fn = octave_legacy_graph_function(fullfile(here, 'legacy'));
else
    legacy_fn = @legacy_graph_metrics;
end
fields = {'density_actual', 'n_components', 'strength_mean', 'C_bin', 'C_wei', 'L_bin', ...
    'L_wei', 'Eglob_bin', 'Eglob_wei', 'assort_bin', 'assort_wei', 'Q_mod', ...
    'transitivity_bin'};
maxd = 0; nmat = 0;
for s = 1:S
    for k = 1:K
        for dens = [0.10 0.15 0.20]
            m_old = legacy_fn(fc_new{s, k}, dens, 100 * s + k);
            m_new = v2_metrics(fc_new{s, k}, dens, 100 * s + k);
            nmat = nmat + 1;
            for f = 1:numel(fields)
                a = m_old.(fields{f}); b = m_new.(fields{f});
                if ~(isnan(a) && isnan(b)), maxd = max(maxd, abs(a - b)); end
            end
        end
    end
end
results = add(results, sprintf('T6 graph metrics previous vs BCT threshold (%d matrices, max diff)', nmat), ...
    maxd, maxd < tol, 'same thresholded edges and metrics');

% ---- T7: pre-PCA model (options.pca): getFuncConn = pooled ROI scatter
% projected onto the PCA subspace ----
opts_pca = rmfield(opts, 'lowrank'); opts_pca.pca = 6;
Rp = hmm_fit_raw_timeseries(Ts, opts_pca, (1:S)');
[~, ~, scat_p] = hmm_subject_state_fc(Ts, Rp.Gamma, Rp.hmm.train, 0);
P = Rp.hmm.train.A * Rp.hmm.train.A';
r_proj = 1; r_full = 1;
for k = 1:Rp.K
    Sk = zeros(n);
    for s = 1:S, Sk = Sk + scat_p{s, k}; end
    C = getFuncConn(Rp.hmm, k, true);
    Cp = P * Sk * P;
    r_proj = min(r_proj, corr_(C(ut), Cp(ut)));
    r_full = min(r_full, corr_(C(ut), Sk(ut)));
end
results = add(results, 'T7 pre-PCA: 1 - r(getFuncConn, pooled ROI scatter projected on PCs)', ...
    1 - r_proj, 1 - r_proj < 1e-4, sprintf(['equal up to HMM-MAR''s Wishart prior; getFuncConn has ' ...
    'rank %d in ROI space; r with the full pooled scatter = %.3f'], ...
    rank(getFuncConn(Rp.hmm, 1, true)), r_full));

% ---- report ----
fprintf('\n%-75s %12s  %s\n', 'Test', 'Value', 'Pass');
for i = 1:numel(results)
    p = results(i).pass;
    if isnan(p), ps = 'report'; elseif p, ps = 'PASS'; else, ps = 'FAIL'; end
    fprintf('%-75s %12.4g  %s  %s\n', results(i).test, results(i).value, ps, results(i).note);
end
end

% ======================================================================
function r = add(r, name, value, pass, note)
r(end + 1) = struct('test', name, 'value', value, 'pass', double(pass), 'note', note);
end

function c = corr_(a, b)
a = a - mean(a); b = b - mean(b);
c = (a' * b) / sqrt((a' * a) * (b' * b));
end

function m = run_len(x)
x = x(:);
if ~any(x), m = NaN; return; end
d = diff([0; x; 0]);
m = mean(find(d == -1) - find(d == 1));
end

function l = argmax_(G)
[~, l] = max(G, [], 2);
end

function XXGXX = hmmmar_xxgxx(hmm, X, T, Gamma)
% Runs HMM-MAR's own setxx.m macro on the data as hmmmar sees them.
data = struct('X', standardisedata(X, T, hmm.train.standardise));
XX = [];  %#ok<NASGU> setxx builds XX from data when XX is empty
setxx;
end

function fc = legacy_hmm_fit_raw_timeseries_fc_only(Ts, Gamma)
% The previous group estimator, taken verbatim from the legacy fit function.
txt = fileread(which('legacy_hmm_fit_raw_timeseries'));
i0 = strfind(txt, 'function [state_fc, effective_n] = ...');
i1 = strfind(txt, 'function dwell = local_mean_run_length');
body = txt(i0:i1 - 1);
body = strrep(body, 'local_within_subject_weighted_state_fc', 'legacy_group_fc_tmp');
d = tempname; mkdir(d);
fid = fopen(fullfile(d, 'legacy_group_fc_tmp.m'), 'w'); fwrite(fid, body); fclose(fid);
addpath(d);
fc = legacy_group_fc_tmp(Ts, Gamma);
rmpath(d);
end

function fn = extract_v2_graph_function(v2_dir, is_octave)
% Copies compute_graph_metrics_single_robust out of the run5 script unchanged
% (Octave only: RandStream is replaced by rand('seed', ...), which Octave has).
txt = fileread(fullfile(v2_dir, 'run5_analyze_raw_hmm_graph_metrics.m'));
i0 = strfind(txt, 'function metrics = compute_graph_metrics_single_robust');
body = strrep(txt(i0:end), 'compute_graph_metrics_single_robust', 'v2_graph_metrics_tmp');
if is_octave, body = octave_randstream(body); end
d = tempname; mkdir(d);
fid = fopen(fullfile(d, 'v2_graph_metrics_tmp.m'), 'w'); fwrite(fid, body); fclose(fid);
addpath(d);
fn = @v2_graph_metrics_tmp;
end

function fn = octave_legacy_graph_function(legacy_dir)
% Octave lacks graph/conncomp and RandStream: components are counted with BCT
% get_components and the seed is set with rand('seed', ...). MATLAB runs the
% legacy file unchanged.
body = fileread(fullfile(legacy_dir, 'legacy_graph_metrics.m'));
body = strrep(body, 'function metrics = legacy_graph_metrics', 'function metrics = legacy_graph_metrics_oct');
body = strrep(body, sprintf('Gbin = graph(B);\ncomponent_labels = conncomp(Gbin);\ncomponent_sizes = accumarray(component_labels(:), 1);'), ...
    '[~, component_sizes] = get_components(B);');
body = octave_randstream(body);
d = tempname; mkdir(d);
fid = fopen(fullfile(d, 'legacy_graph_metrics_oct.m'), 'w'); fwrite(fid, body); fclose(fid);
addpath(d);
fn = @legacy_graph_metrics_oct;
end

function body = octave_randstream(body)
body = regexprep(body, 'old_stream = RandStream\.getGlobalStream;[^\n]*\n', '');
body = regexprep(body, 'cleanup = onCleanup[^\n]*\n', '');
body = regexprep(body, 'new_stream = RandStream\([^\n]*\n', '');
body = regexprep(body, 'RandStream\.setGlobalStream\([^\n]*\n', 'rand(''seed'', seed);\n');
end
