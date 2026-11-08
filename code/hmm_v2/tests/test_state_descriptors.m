function results = test_state_descriptors(hmmmar_dir)
% TEST_STATE_DESCRIPTORS  hmm_state_descriptor_values.m on simulated two-state
% data whose answer is known. MATLAB or Octave; seconds.
%
%   test_state_descriptors('/blue/febo/MATLAB/HMM-MAR-master')
%
% 20 mice, 12 ROIs, 600 time points; states alternate in blocks of 30 time
% points (Gamma 0.95 / 0.05). State 1 has correlation pattern C1 in every
% scenario; state 2 differs by:
%   amplitude   covariance 4 * C1: same pattern, 4 x the variance
%   pattern     another correlation matrix C2 with the same mean correlation
%   global      C1 plus a shared (global) component g = 0.5 on every ROI pair
%
% Checks
%   D1 amplitude: pooled variance ratio 2/1 near 4 (the soft Gamma and the
%      per-mouse standardisation make the exact value 4 * mix / mix', computed
%      here from the simulation), FC pattern r > 0.98, mean correlation equal
%      (|diff| < 0.03)
%   D2 pattern: variance ratio within 0.9-1.1, FC pattern r^2 < 0.5, mean
%      correlation equal (|diff| < 0.03)
%   D3 global: global-signal variance ratio 2/1 within 8% of its expected value
%      (1'C1/N^2 and 1'C3/N^2 mixed by the soft Gamma), mean correlation higher
%      in state 2 (> 0.2), FC pattern r > 0.95 (a uniform shift of correlations)
%   D4 pooled values = per-mouse scatter summed: pooled FC equals the FC
%      built as hmm_fit_raw_timeseries.m builds R.state_fc (max diff)
%   D5 per-mouse values by hand for one mouse (soft weights, standardised data)
%   D6 the effective-N filter blanks a state's per-mouse values below it
%   D7 noise correction: amplitude-only states with weak FC structure (true
%      correlations spread by about 0.03, so each state's FC split-half r is
%      well below 1) have a raw pattern r clearly below 1 but a disattenuated
%      r near 1 (0.85-1.15); the pattern scenario's disattenuated r stays low
%      (< 0.5) and the structured amplitude scenario's is near 1 (> 0.95).
%      (With no FC structure at all, the split-half r is near 0 and the
%      correction is undefined; the report gives each state's split-half r.)

here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));
addpath(genpath(hmmmar_dir));
rand('seed', 11); randn('seed', 11); %#ok<RAND>
S = 20; T = 600; N = 12; L = 30;
up = triu(true(N), 1);
C1 = local_corr_target(N, 0.3);
C2 = local_corr_target(N, 0.3);
g = 0.5;
C3 = (C1 + g) / (1 + g); C3(1:N + 1:end) = 1;
train = struct('standardise', 1, 'zeromean', 1);
res = {};

lab = repmat([ones(L, 1); 2 * ones(L, 1)], T / (2 * L), 1);
Gm = 0.05 * ones(T, 2); Gm(sub2ind([T 2], (1:T)', lab)) = 0.95;
Gamma = repmat({Gm}, S, 1);

scen = {'amplitude', 4 * C1; 'pattern', C2; 'global', C3};
D = cell(1, 3);
for c = 1:3
    Ts = cell(S, 1);
    for s = 1:S
        X = zeros(T, N);
        X(lab == 1, :) = randn(sum(lab == 1), N) * chol(C1);
        X(lab == 2, :) = randn(sum(lab == 2), N) * chol(scen{c, 2});
        Ts{s} = X;
    end
    D{c} = hmm_state_descriptor_values(Ts, Gamma, train, 0);
    if c == 1, Ts_amp = Ts; end
end

% D1: expected ratio with soft weights: E[x^2 | state] mixes 1 and 4 by Gamma
w1 = 0.95 * 1 + 0.05 * 4; w2 = 0.05 * 1 + 0.95 * 4;       % per unit of raw variance
p = D{1}.pairs(1);
r_exp = (1 / p.mean_variance_ratio);                       % state 1 / state 2
ok1 = abs(p.mean_variance_ratio - w1 / w2) / (w1 / w2) < 0.08 && p.fc_pattern_r > 0.98 ...
    && abs(p.mean_correlation_diff) < 0.03;
res(end + 1, :) = {sprintf('D1 amplitude: ratio 1/2 %.3f (expected %.3f), r %.3f, d mean r %.3f', ...
    p.mean_variance_ratio, w1 / w2, p.fc_pattern_r, p.mean_correlation_diff), r_exp, ok1};

p = D{2}.pairs(1);
ok2 = p.mean_variance_ratio > 0.9 && p.mean_variance_ratio < 1.1 && p.fc_pattern_r2 < 0.5 ...
    && abs(p.mean_correlation_diff) < 0.03;
res(end + 1, :) = {sprintf('D2 pattern: ratio %.3f, r^2 %.3f, d mean r %.3f', p.mean_variance_ratio, ...
    p.fc_pattern_r2, p.mean_correlation_diff), p.fc_pattern_r2, ok2};

p = D{3}.pairs(1);
g1 = sum(C1(:)) / N ^ 2; g3 = sum(C3(:)) / N ^ 2;
gs_exp = (0.05 * g1 + 0.95 * g3) / (0.95 * g1 + 0.05 * g3);
ok3 = abs(1 / p.gs_variance_ratio - gs_exp) / gs_exp < 0.08 && -p.mean_correlation_diff > 0.2 ...
    && p.fc_pattern_r > 0.95;
res(end + 1, :) = {sprintf('D3 global: GS ratio 2/1 %.2f (expected %.2f), mean r %.3f vs %.3f, pattern r %.3f', ...
    1 / p.gs_variance_ratio, gs_exp, D{3}.pooled.mean_correlation(1), D{3}.pooled.mean_correlation(2), ...
    p.fc_pattern_r), p.fc_pattern_r, ok3};

% D4: as hmm_fit_raw_timeseries.m builds R.state_fc
[~, ~, sc] = hmm_subject_state_fc(Ts_amp, Gamma, train, 0);
dmax = 0;
for k = 1:2
    Sk = zeros(N);
    for s = 1:S, Sk = Sk + sc{s, k}; end
    sd = sqrt(diag(Sk));
    F = Sk ./ (sd * sd'); F(1:N + 1:end) = 1;
    dmax = max(dmax, max(abs(F(:) - D{1}.pooled.fc{k}(:))));
end
res(end + 1, :) = {'D4 pooled FC = R.state_fc construction (max diff)', dmax, dmax < 1e-12};

% D5: one mouse by hand
X = standardisedata(Ts_amp{3}, T, 1);
w = Gm(:, 2);
Sk = (X .* w)' * X;
R2 = corrcov((Sk + Sk') / 2);
hand = [trace(Sk) / (N * sum(w)), sum(Sk(:)) / (N ^ 2 * sum(w)), mean(R2(up))];
got = [D{1}.subject.mean_variance(3, 2), D{1}.subject.gs_variance(3, 2), D{1}.subject.mean_correlation(3, 2)];
d5 = max(abs(hand - got));
res(end + 1, :) = {'D5 mouse 3, state 2 by hand (max diff)', d5, d5 < 1e-12};

% D6: effective-N filter
neff = D{1}.subject.effective_n(1, 1);
Df = hmm_state_descriptor_values(Ts_amp, Gamma, train, neff + 1);
ok6 = all(isnan(Df.subject.mean_variance(:))) && all(isnan(Df.subject.mean_correlation(:))) ...
    && isfinite(Df.pooled.mean_variance(1)) && Df.pairs(1).n_mice == 0;
res(end + 1, :) = {sprintf('D6 effective N %.1f below the filter: per-mouse values blank', neff), ...
    double(ok6), ok6};

% D7
Cw = local_corr_target(N, 0);
Cw = eye(N) + 0.05 * (Cw - eye(N));
Ts = cell(S, 1);
for s = 1:S
    X = randn(T, N) * chol(Cw);
    X(lab == 2, :) = 2 * X(lab == 2, :);
    Ts{s} = X;
end
Dn = hmm_state_descriptor_values(Ts, Gamma, train, 0);
pn = Dn.pairs(1);
ok7 = pn.fc_pattern_r < 0.9 && abs(pn.fc_pattern_r_disattenuated - 1) < 0.15 ...
    && D{2}.pairs(1).fc_pattern_r_disattenuated < 0.5 && D{1}.pairs(1).fc_pattern_r_disattenuated > 0.95;
res(end + 1, :) = {sprintf(['D7 weak FC (split-half r %.2f/%.2f): raw r %.2f, disattenuated %.2f; ' ...
    'pattern %.2f; structured amplitude %.2f'], Dn.pooled.fc_split_half_r(1), ...
    Dn.pooled.fc_split_half_r(2), pn.fc_pattern_r, pn.fc_pattern_r_disattenuated, ...
    D{2}.pairs(1).fc_pattern_r_disattenuated, D{1}.pairs(1).fc_pattern_r_disattenuated), ...
    pn.fc_pattern_r_disattenuated, ok7};

results = cell2struct(res, {'check', 'value', 'pass'}, 2);
fprintf('\n');
for i = 1:size(res, 1)
    fprintf('%-90s %10.4g  %s\n', res{i, 1}, res{i, 2}, local_yesno(res{i, 3}));
end
fprintf('\n%d passed, %d failed.\n', sum([res{:, 3}]), sum(~[res{:, 3}]));
end


function C = local_corr_target(N, m)
% a random correlation matrix with mean off-diagonal correlation near m
A = randn(N, 3);
B = A * A';
d = sqrt(diag(B));
B = B ./ (d * d');
up = triu(true(N), 1);
C = m + 0.6 * (B - mean(B(up)));
C(1:N + 1:end) = 1;
[V, E] = eig((C + C') / 2);
C = V * diag(max(diag(E), 0.05)) * V';
d = sqrt(diag(C));
C = C ./ (d * d');
C = (C + C') / 2;
end


function s = local_yesno(tf)
if tf, s = 'PASS'; else, s = 'FAIL'; end
end
