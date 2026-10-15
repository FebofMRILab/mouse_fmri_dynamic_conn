function info = dcc_example_traces(data_dir, dcc_code_dir, dfc_dir, out_dir, mouse_id)
% DCC_EXAMPLE_TRACES  Conditional correlations rho_ij(t) for one mouse and its
% phase-randomised surrogates, from the published DCC estimator (dcc_mvgarch.m,
% Lindquist's DCC toolbox; K. Sheppard). For a Supplement figure in the units of
% Lindquist et al. (2014): correlations between -1 and 1 over time.
% Display only: the inference is the calibrated test in run_dcc_cpca_null.m.
%
%   info = dcc_example_traces(data_dir, dcc_code_dir, dfc_dir, out_dir, mouse_id)
%
%   data_dir      hmm_signals/ (signal<id>.txt, meta_data.csv)
%   dcc_code_dir  .../Lindquist_Dynamic_Correlation/DCC_toolbox/DCCcode
%   dfc_dir       results of run_dcc_cpca_null.m (default 'dfc_dcc_cpca'): its
%                 dfc_config.txt gives the settings, pca_explained.csv checks
%                 the group PCA, dfc_subject_metrics.csv gives this mouse's
%                 pipeline values for the caption
%   out_dir       default <dfc_dir>/dcc_example
%   mouse_id      optional; default: one NTG_Y mouse drawn at random with
%                 rng(20261014) (PI, 14 Oct 2026: "a random mouse from NTG_Y")
%
% Input, identical to the pipeline's for this mouse (run_dcc_cpca_null.m):
%   z-scored ROI series -> group PCA (25 components, all mice) -> odd-volume
%   polyphase component (decimation 2; the pipeline averages odd and even, the
%   display uses odd) -> AR(2) residuals. The 19 phase-randomised surrogates
%   use the pipeline's seeds (seed + id*1000 + r), so they are the same
%   surrogates the mouse's z-scores were computed from.
% Pairs shown (fixed in advance, not chosen from the results): components 1-2,
% 1-3 and 2-3, the three largest. The SD over time of rho_ij(t) is saved for
% all pairs (its mean over pairs is the pipeline's corr_fluct).
%
% Needs MATLAB's Optimization Toolbox (fmincon, used by dcc_mvgarch.m). The rest
% is base MATLAB and also runs in Octave (plumbing test with a stand-in estimator).
% Writes: dcc_example_traces.csv, dcc_example_pair_sd.csv,
% dcc_example_params.csv, dcc_example_info.txt.

if nargin < 3 || isempty(dfc_dir), dfc_dir = 'dfc_dcc_cpca_pca15'; end
if nargin < 4 || isempty(out_dir), out_dir = fullfile(dfc_dir, 'dcc_example'); end
if nargin < 5, mouse_id = []; end
here = fileparts(mfilename('fullpath'));
addpath(here);                                   % hmm_phase_randomise.m
addpath(dcc_code_dir);
start_dir = pwd;
out_dir = local_abs(out_dir, start_dir);
dfc_dir = local_abs(dfc_dir, start_dir);
data_dir = local_abs(data_dir, start_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ---- settings: the pipeline's, from its config file when present ----
cfg = struct('seed', 123, 'pca_dim', 25, 'dcc_decimate', 2, 'ar_order', 2, 'TR', 2, 'n_surr', 19);
cfg_file = fullfile(dfc_dir, 'dfc_config.txt');
if exist(cfg_file, 'file')
    txt = fileread(cfg_file);
    for f = fieldnames(cfg)'
        tok = regexp(txt, ['(?m)^' f{1} '\s*=\s*([0-9.]+)'], 'tokens', 'once');
        if ~isempty(tok), cfg.(f{1}) = str2double(tok{1}); end
    end
else
    warning('No %s: pipeline defaults used (seed 123, 25 components, decimation 2).', cfg_file);
end
PAIRS = [1 2; 1 3; 2 3];
SELECT_SEED = 20261014;

% ---- data and groups (all mice: the group PCA uses all of them) ----
[meta_id, meta_grp] = local_read_meta(fullfile(data_dir, 'meta_data.csv'));   % the pipeline's reader
ids = []; Xraw = {};
s = 1;
while exist(fullfile(data_dir, sprintf('signal%d.txt', s)), 'file')
    Xraw{end + 1} = double(dlmread(fullfile(data_dir, sprintf('signal%d.txt', s)))); %#ok<AGROW>
    ids(end + 1) = s; %#ok<AGROW>
    s = s + 1;
end
grp = repmat({''}, numel(ids), 1);
for i = 1:numel(ids)
    j = find(meta_id == ids(i), 1);
    if ~isempty(j), grp{i} = strtrim(meta_grp{j}); end
end
cands = sort(ids(strcmp(grp, 'NTG_Y')));
assert(~isempty(cands), 'No NTG_Y mice in meta_data.csv.');
if isempty(mouse_id)
    rng(SELECT_SEED, 'twister');
    mouse_id = cands(randi(numel(cands)));
    how = sprintf('drawn at random from the %d NTG_Y mice with rng(%d)', numel(cands), SELECT_SEED);
else
    how = 'given as an argument';
end
im = find(ids == mouse_id, 1);
assert(~isempty(im), 'Mouse %d not found.', mouse_id);
fprintf('Mouse %d (%s), %s.\n', mouse_id, grp{im}, how);

Xs = cellfun(@local_standardise, Xraw, 'UniformOutput', false);
[A, expl] = local_group_pca(Xs, cfg.pca_dim);
pca_check = 'not checked (no pca_explained.csv)';
pe_file = fullfile(dfc_dir, 'pca_explained.csv');
if exist(pe_file, 'file')
    pe = local_read_csv(pe_file);
    d = max(abs(str2double(pe.explained(:)) - expl(:)));
    pca_check = sprintf('max |difference| from the pipeline''s pca_explained.csv = %.2g', d);
    if d > 1e-6
        warning('Group PCA differs from the pipeline''s (%g): different mice or data?', d);
    end
end

% ---- observed and surrogates through the published estimator ----
work = tempname; mkdir(work);                 % dcc_mvgarch writes tempHt.mat in pwd
cd(work); back = onCleanup(@() cd(start_dir));
X = Xraw{im};
[Rd, E] = local_rho(X, A, cfg);              % observed: T x pairs, and residuals
[T, N] = size(E);
iu = find(triu(true(N), 1));
[pi_, pj_] = ind2sub([N N], iu);
par = local_params(Rd.p, N);
sd_obs = std(Rd.R, 0, 1);
nS = cfg.n_surr;
Rs = zeros(T, numel(iu), nS);
sd_sur = zeros(nS, numel(iu));
par_s = repmat(par, nS, 1);
for r = 1:nS
    rng(cfg.seed + mouse_id * 1000 + r, 'twister');   % = local_surrogate(.., 'phase', id, r)
    Xsur = hmm_phase_randomise(X, []);
    Ds = local_rho(Xsur, A, cfg);
    Rs(:, :, r) = Ds.R;
    sd_sur(r, :) = std(Ds.R, 0, 1);
    par_s(r) = local_params(Ds.p, N);
    fprintf('surrogate %d of %d: DCC a %.4f\n', r, nS, par_s(r).a);
end
cd(start_dir);

% ---- write ----
q = cfg.ar_order;
t_s = ((q:q + T - 1)' * cfg.dcc_decimate) * cfg.TR;     % time of each residual (s)
C = corrcoef(E);
fid = fopen(fullfile(out_dir, 'dcc_example_traces.csv'), 'w');
fprintf(fid, 'time_s,pair,observed_r,surr_p025,surr_p50,surr_p975,static_r\n');
for k = 1:size(PAIRS, 1)
    c = find(pi_ == PAIRS(k, 1) & pj_ == PAIRS(k, 2));
    S = squeeze(Rs(:, c, :));
    qs = local_quantiles(S, [0.025 0.5 0.975]);
    for t = 1:T
        fprintf(fid, '%.1f,%d-%d,%.6f,%.6f,%.6f,%.6f,%.6f\n', t_s(t), PAIRS(k, 1), PAIRS(k, 2), ...
            Rd.R(t, c), qs(t, 1), qs(t, 2), qs(t, 3), C(PAIRS(k, 1), PAIRS(k, 2)));
    end
end
fclose(fid);

fid = fopen(fullfile(out_dir, 'dcc_example_pair_sd.csv'), 'w');
fprintf(fid, 'comp_i,comp_j,sd_observed,sd_surr_mean,sd_surr_sd,static_r\n');
for c = 1:numel(iu)
    fprintf(fid, '%d,%d,%.6f,%.6f,%.6f,%.6f\n', pi_(c), pj_(c), sd_obs(c), mean(sd_sur(:, c)), ...
        std(sd_sur(:, c)), C(pi_(c), pj_(c)));
end
fclose(fid);

fid = fopen(fullfile(out_dir, 'dcc_example_params.csv'), 'w');
fprintf(fid, 'source,dcc_a,dcc_b,garch_alpha_mean,corr_fluct\n');
fprintf(fid, 'observed,%.6f,%.6f,%.6f,%.6f\n', par.a, par.b, par.alpha, mean(sd_obs));
for r = 1:nS
    fprintf(fid, 'phase_surrogate_%02d,%.6f,%.6f,%.6f,%.6f\n', r, par_s(r).a, par_s(r).b, ...
        par_s(r).alpha, mean(sd_sur(r, :)));
end
fclose(fid);

pipe = local_pipeline_values(fullfile(dfc_dir, 'dfc_subject_metrics.csv'), mouse_id);
fid = fopen(fullfile(out_dir, 'dcc_example_info.txt'), 'w');
c = onCleanup(@() fclose(fid));
say = @(varargin) local_say(fid, varargin{:});
say('DCC example traces (published estimator: dcc_mvgarch.m, Lindquist DCC toolbox)\n');
say('Run %s, %s\n', datestr(now, 'yyyy-mm-dd HH:MM'), version);
say('mouse_id: %d\ngroup: %s\nselection: %s\nNTG_Y candidates: %s\n', mouse_id, grp{im}, how, mat2str(cands));
say('Input: %d group-PCA components (%.1f%% of variance), odd-volume polyphase component (TR %g s), AR(%d) residuals, T = %d\n', ...
    N, 100 * sum(expl), cfg.TR * cfg.dcc_decimate, q, T);
say('Group PCA check: %s\n', pca_check);
say('Surrogates: %d phase-randomised, pipeline seeds (%d + id*1000 + r)\n', nS, cfg.seed);
say('Pairs shown: %s (fixed in advance)\n', mat2str(PAIRS));
say('Published estimator, observed: DCC a = %.4f, b = %.4f; mean GARCH alpha = %.4f; corr_fluct = %.4f\n', ...
    par.a, par.b, par.alpha, mean(sd_obs));
say('Published estimator, surrogates: DCC a mean %.4f (SD %.4f); corr_fluct mean %.4f (SD %.4f)\n', ...
    mean([par_s.a]), std([par_s.a]), mean(mean(sd_sur, 2)), std(mean(sd_sur, 2)));
say('Pairs whose SD over time exceeds their surrogate mean: %d of %d\n', sum(sd_obs(:) > mean(sd_sur, 1)'), numel(iu));
say('%s\n', pipe);
info = struct('mouse_id', mouse_id, 'group', grp{im}, 'dcc_a', par.a, 'dcc_b', par.b, ...
    'corr_fluct', mean(sd_obs), 'surr_dcc_a', [par_s.a], 'surr_corr_fluct', mean(sd_sur, 2)');
end


function [D, E] = local_rho(X, A, cfg)
% Pipeline input for one series set, then the published estimator.
Y = local_standardise(X) * A;
Y = Y(1:cfg.dcc_decimate:end, :);                    % odd-volume polyphase component
E = local_ar_resid(Y, cfg.ar_order);
[p, ~, Ht] = dcc_mvgarch(E, 1, 1, 1, 1);
N = size(E, 2);
iu = find(triu(true(N), 1));
T = size(Ht, 3);
R = zeros(T, numel(iu));
for t = 1:T
    H = Ht(:, :, t);
    s = sqrt(diag(H));
    Rt = H ./ (s * s');
    R(t, :) = Rt(iu)';
end
D = struct('R', R, 'p', p);
end


function s = local_params(p, N)
g = reshape(p(1:3 * N), 3, N);                       % [omega; alpha; beta] per series
s = struct('a', p(3 * N + 1), 'b', p(3 * N + 2), 'alpha', mean(g(2, :)));
end


function txt = local_pipeline_values(f, id)
if ~exist(f, 'file')
    txt = 'Pipeline values: dfc_subject_metrics.csv not found.';
    return;
end
T = local_read_csv(f);
rows = find(str2double(T.subject_id) == id);
txt = 'Pipeline values for this mouse (our estimator, odd and even polyphase averaged):';
for m = {'dcc_a', 'dcc_LR', 'corr_fluct', 'garch_alpha'}
    r = rows(strcmp(T.metric(rows), m{1}));
    if ~isempty(r)
        v = @(c) str2double(T.(c){r(1)});
        txt = sprintf('%s\n  %s: observed %.4f, phase-surrogate mean %.4f (SD %.4f), z %.2f, replica z %.2f', ...
            txt, m{1}, v('observed'), v('phase_mean'), v('phase_sd'), v('phase_z'), v('phase_replica_z'));
    end
end
end


function S = local_read_csv(f)
% Comma-separated file written by the pipeline (header row, no quoted commas)
% -> struct of cell columns (text). Runs in Octave too. A UTF-8 byte-order
% mark is removed, as in local_read_meta.
txt = fileread(f);
if numel(txt) >= 3 && isequal(double(txt(1:3)), [239 187 191]), txt = txt(4:end); end
if ~isempty(txt) && double(txt(1)) == 65279, txt = txt(2:end); end
L = strsplit(strtrim(txt), {'\r\n', '\n'});
h = strtrim(strsplit(L{1}, ',', 'CollapseDelimiters', false));
S = struct();
for k = 1:numel(h), S.(h{k}) = cell(numel(L) - 1, 1); end      % headers are valid names
fn = fieldnames(S);
for i = 2:numel(L)
    c = strsplit(L{i}, ',', 'CollapseDelimiters', false);
    for k = 1:min(numel(c), numel(fn)), S.(fn{k}){i - 1} = strtrim(c{k}); end
end
end


% As local_read_meta in run_dcc_cpca_null.m (copied unchanged): robust to a
% UTF-8 byte-order mark, line endings, delimiters and quoted fields.
function [sid, grp] = local_read_meta(fn)
% Robust to a UTF-8 byte-order mark (Excel "CSV UTF-8"), Windows/Mac line
% endings, comma/semicolon/tab delimiters, quoted fields, header case and
% extra columns. Column names are matched after lower-casing and removing
% everything except letters, digits and underscores.
txt = fileread(fn);
if numel(txt) >= 3 && isequal(double(txt(1:3)), [239 187 191])
    txt = txt(4:end);                       % BOM read as three bytes
end
if ~isempty(txt) && double(txt(1)) == 65279
    txt = txt(2:end);                       % BOM decoded as U+FEFF
end
lines = regexp(txt, '\r\n|\n|\r', 'split');
lines = lines(~cellfun(@(l) isempty(strtrim(l)), lines));
if numel(lines) < 2
    error('meta_data.csv has no data rows.');
end
cands = {',', ';', sprintf('\t')};
[~, di] = max(cellfun(@(d) sum(lines{1} == d), cands));
delim = cands{di};
raw_header = strsplit(lines{1}, delim);
header = lower(regexprep(raw_header, '[^A-Za-z0-9_]', ''));
ci = find(strcmp(header, 'subject_id') | strcmp(header, 'subjectid'), 1);
cg = find(strcmp(header, 'group'), 1);
if isempty(ci) || isempty(cg)
    error(['meta_data.csv needs columns subject_id and group; header read as: ' ...
        '%s (delimiter "%s").'], strjoin(raw_header, ' | '), ...
        strrep(delim, sprintf('\t'), '\t'));
end
n = numel(lines) - 1;
sid = NaN(n, 1);
grp = cell(n, 1);
for i = 1:n
    parts = strsplit(lines{i + 1}, delim);
    parts = strtrim(regexprep(parts, '"', ''));
    if numel(parts) < max(ci, cg)
        error('meta_data.csv row %d has %d fields; expected at least %d.', ...
            i + 1, numel(parts), max(ci, cg));
    end
    sid(i) = str2double(parts{ci});
    grp{i} = parts{cg};
end
if any(isnan(sid))
    error('Non-numeric subject_id in meta_data.csv rows: %s', mat2str(find(isnan(sid)).' + 1));
end
end


function Q = local_quantiles(S, p)
% Pointwise quantiles over surrogates (rows = time), linear interpolation.
S = sort(S, 2);
n = size(S, 2);
Q = zeros(size(S, 1), numel(p));
for k = 1:numel(p)
    h = (n - 1) * p(k) + 1;
    lo = floor(h); hi = min(lo + 1, n);
    Q(:, k) = S(:, lo) + (h - lo) * (S(:, hi) - S(:, lo));
end
end


function E = local_ar_resid(Y, p)
% Step 1 of dfc_dcc_subject.m, unchanged: demean, AR(p) by least squares.
[T, N] = size(Y);
E = zeros(T - p, N);
for i = 1:N
    x = Y(:, i) - mean(Y(:, i));
    Xl = zeros(T - p, p);
    for L = 1:p
        Xl(:, L) = x(p + 1 - L:T - L);
    end
    E(:, i) = x(p + 1:T) - Xl * (Xl \ x(p + 1:T));
end
end


function [A, expl] = local_group_pca(Xs, d)
% As local_group_pca in run_dcc_cpca_null.m.
N = size(Xs{1}, 2); C = zeros(N); n = 0;
for s = 1:numel(Xs), C = C + Xs{s}' * Xs{s}; n = n + size(Xs{s}, 1); end
C = (C / (n - 1) + (C / (n - 1))') / 2;
[V, D] = eig(C);
[lam, o] = sort(diag(D), 'descend');
A = V(:, o(1:d));
for k = 1:d
    [~, im] = max(abs(A(:, k)));
    A(:, k) = A(:, k) * sign(A(im, k));
end
expl = lam(1:d) / sum(lam);
end


function Xs = local_standardise(X)
% As local_standardise in run_dcc_cpca_null.m.
Xs = X - mean(X, 1);
sd = std(Xs, 0, 1);
sd(sd <= 0) = 1;
Xs = Xs ./ sd;
end


function local_say(fid, varargin)
fprintf(1, varargin{:});
fprintf(fid, varargin{:});
end


function p = local_abs(p, base)
if ~(strncmp(p, '/', 1) || (numel(p) > 1 && p(2) == ':'))
    p = fullfile(base, p);
end
end