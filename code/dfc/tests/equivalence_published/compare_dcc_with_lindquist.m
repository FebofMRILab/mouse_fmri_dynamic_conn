function T = compare_dcc_with_lindquist(data_dir, dcc_code_dir, out_dir, n_mice)
% COMPARE_DCC_WITH_LINDQUIST  dfc_dcc_subject.m (our GARCH(1,1)-DCC(1,1), base
% MATLAB) against the published DCC estimator in Lindquist's DCC toolbox
% (github.com/canlab/Lindquist_Dynamic_Correlation, DCC_toolbox/DCCcode:
% dcc_mvgarch.m by K. Sheppard, used by Lindquist et al. 2014). MATLAB with the
% Optimization Toolbox (fmincon) is required for the published code.
%
%   T = compare_dcc_with_lindquist(data_dir, dcc_code_dir, out_dir, n_mice)
%
%   data_dir      folder with signal1.txt ... (570 x 82), e.g.
%                 /blue/febo/share/Abeta_Age/rabies/signals/hmm_signals
%   dcc_code_dir  .../Lindquist_Dynamic_Correlation/DCC_toolbox/DCCcode
%   out_dir       results folder (default 'dcc_lindquist_check')
%   n_mice        mice to compare (default 5: signal1 ... signal5)
%
% Both estimators get IDENTICAL input, so any difference is the estimator:
%   1. Simulated GARCH(1,1)-DCC(1,1) data with known parameters (5 series,
%      T = 2000; a = 0.05, b = 0.90; GARCH alpha = 0.10, beta = 0.80).
%   2. Real mice, prepared as in run_dcc_cpca_null.m: z-scored ROI signals,
%      group PCA to 25 components (over the mice loaded here), the odd-volume
%      polyphase component (decimation by 2), AR(2) residuals by least squares.
%      Each estimator then fits GARCH(1,1) per series and DCC(1,1) on the
%      residuals (ours with ar_order = 0, because the AR step is done here).
%
% Known differences in the estimators, both standard:
%   - ours uses variance targeting for GARCH (omega = (1-alpha-beta) var(e));
%     the published code estimates omega freely;
%   - ours optimises with fminsearch, the published code with fmincon.
% Agreement is therefore expected to optimiser tolerance, not to 1e-12.
%
% Returns a struct of the rows. Writes out_dir/dcc_lindquist_comparison.csv (one row per case x quantity)
% and out_dir/dcc_lindquist_summary.txt.

if nargin < 3 || isempty(out_dir), out_dir = 'dcc_lindquist_check'; end
if nargin < 4 || isempty(n_mice), n_mice = 5; end
here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(here)));                 % dfc/ (dfc_dcc_subject.m)
addpath(dcc_code_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
work = tempname; mkdir(work);                        % dcc_mvgarch saves tempHt.mat in pwd
old = cd(work); restore = onCleanup(@() cd(old));
out_dir = local_abs(out_dir, old);

rows = {};
% ---------- 1. simulated data with known parameters ----------
rng(20261004);
truth = struct('a', 0.05, 'b', 0.90, 'alpha', 0.10, 'beta', 0.80, 'omega', 0.10);
E = local_simulate_dcc(5, 2000, truth);
rows = [rows; local_compare('simulated', E, truth)];

% ---------- 2. real mice ----------
Xs = cell(1, n_mice);
for s = 1:n_mice
    X = load(fullfile(data_dir, sprintf('signal%d.txt', s)));
    Xs{s} = (X - mean(X)) ./ std(X);
end
A = local_group_pca(Xs, 25);
for s = 1:n_mice
    Y = Xs{s} * A;
    Y = Y(1:2:end, :);                               % odd-volume polyphase component
    rows = [rows; local_compare(sprintf('mouse%d', s), local_ar_resid(Y, 2), [])]; %#ok<AGROW>
end

% plain arrays and fprintf, so the script also runs in Octave (tests)
cases = rows(:, 1); qty = rows(:, 2);
ours = cell2mat(rows(:, 3)); pub = cell2mat(rows(:, 4)); tru = cell2mat(rows(:, 5));
d = abs(ours - pub);
fid = fopen(fullfile(out_dir, 'dcc_lindquist_comparison.csv'), 'w');
fprintf(fid, 'case,quantity,ours,lindquist,truth,abs_diff\n');
for r = 1:numel(d)
    fprintf(fid, '%s,%s,%.6g,%.6g,%.6g,%.6g\n', cases{r}, qty{r}, ours(r), pub(r), tru(r), d(r));
end
fclose(fid);
T = struct('case', {cases}, 'quantity', {qty}, 'ours', ours, 'lindquist', pub, 'truth', tru, 'abs_diff', d);

fid = fopen(fullfile(out_dir, 'dcc_lindquist_summary.txt'), 'w');
c = onCleanup(@() fclose(fid));
say = @(varargin) local_say(fid, varargin{:});   % to the screen and the file
say('DCC cross-check: dfc_dcc_subject.m vs dcc_mvgarch.m (Lindquist DCC toolbox)\n');
say('Run %s, %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM'), version);
for q = {'dcc_a', 'dcc_b', 'garch_alpha', 'garch_beta'}
    k = strcmp(qty, q{1});
    say('%-12s max |diff| %.4f   median |diff| %.4f   (n = %d)\n', q{1}, max(d(k)), median(d(k)), sum(k));
end
m = strncmp(cases, 'mouse', 5) & strcmp(qty, 'dcc_a');
if sum(m) > 2
    cc = corrcoef(ours(m), pub(m));
    say('dcc_a across mice: Pearson r (ours vs published) = %.3f\n', cc(1, 2));
end
s = strcmp(cases, 'simulated');
say('\nSimulated (truth a = %.2f, b = %.2f, GARCH alpha = %.2f, beta = %.2f):\n', ...
    truth.a, truth.b, truth.alpha, truth.beta);
for q = {'dcc_a', 'dcc_b', 'garch_alpha', 'garch_beta'}
    k = s & strcmp(qty, q{1});
    say('  %-12s ours %.3f   published %.3f   (mean over series)\n', q{1}, mean(ours(k)), mean(pub(k)));
end
say('\nRead with the known differences in the header (variance targeting, optimiser).\n');
end


function rows = local_compare(name, E, truth)
o = dfc_dcc_subject(E, 0, true);
[p, ~] = dcc_mvgarch(E, 1, 1, 1, 1);
N = size(E, 2);
g = reshape(p(1:3 * N), 3, N);                       % [omega; alpha; beta] per series
ab = p(3 * N + 1:end);
tr = @(f) local_truth(truth, f);
rows = {name, 'dcc_a', o.dcc_a, ab(1), tr('a'); name, 'dcc_b', o.dcc_b, ab(2), tr('b')};
for i = 1:N
    rows = [rows; {name, 'garch_alpha', o.garch_alpha(i), g(2, i), tr('alpha'); ...
                   name, 'garch_beta', o.garch_beta(i), g(3, i), tr('beta')}]; %#ok<AGROW>
end
end


function local_say(fid, varargin)
% Print to the job log and to the summary file. (A cellfun over {1, fid}
% failed in MATLAB R2024b: 'Unexpected output ... Set UniformOutput to false'.)
fprintf(1, varargin{:});
fprintf(fid, varargin{:});
end


function v = local_truth(truth, f)
if isempty(truth), v = NaN; else, v = truth.(f); end
end


function E = local_simulate_dcc(N, T, p)
R0 = 0.4 * ones(N) + 0.6 * eye(N);                   % unconditional correlation
Q = R0; h = ones(1, N) * p.omega / (1 - p.alpha - p.beta);
E = zeros(T, N); e = zeros(1, N);
for t = 1:T + 200                                    % 200 burn-in samples
    R = Q ./ sqrt(diag(Q) * diag(Q)');
    z = (chol(R, 'lower') * randn(N, 1))';
    h = p.omega + p.alpha * e.^2 + p.beta * h;
    e = sqrt(h) .* z;
    Q = (1 - p.a - p.b) * R0 + p.a * (z' * z) + p.b * Q;
    if t > 200, E(t - 200, :) = e; end
end
E = E - mean(E);
end


function e = local_ar_resid(Y, p)
[T, N] = size(Y);
e = zeros(T - p, N);
for i = 1:N
    y = Y(p + 1:end, i);
    X = ones(T - p, 1);
    for l = 1:p, X = [X, Y(p + 1 - l:end - l, i)]; end %#ok<AGROW>
    e(:, i) = y - X * (X \ y);
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


function p = local_abs(p, base)
if ~(startsWith(p, '/') || (numel(p) > 1 && p(2) == ':'))
    p = fullfile(base, p);
end
end
