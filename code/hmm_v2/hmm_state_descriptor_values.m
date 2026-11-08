function D = hmm_state_descriptor_values(Ts_list, Gamma, hmm_train, min_effective_n)
% HMM_STATE_DESCRIPTOR_VALUES  What separates the states: overall amplitude
% (variance, global signal) or the correlation pattern. Descriptive only.
%
%   D = hmm_state_descriptor_values(Ts_list, Gamma, hmm_train, min_effective_n)
%
% Inputs as hmm_subject_state_fc.m (which computes the state statistic HMM-MAR
% itself uses; setxx.m: XXGXX{k} = (X .* Gamma(:,k))' * X on the standardised,
% zero-mean data). Each ROI has unit variance over each mouse's whole scan, so
% the values below are relative to that mouse's average.
%
% Per mouse s and state k (weights w = Gamma{s}(:,k), scatter S = X' diag(w) X,
% N ROIs); NaN when the effective number of time points (sum w)^2 / sum(w.^2)
% is below min_effective_n (the same filter as run5 and run5B):
%   fo                 mean(w)
%   mean_variance      trace(S) / (N * sum(w))          mean ROI variance in the state
%   gs_variance        1' S 1 / (N^2 * sum(w))          variance of the ROI-mean
%                                                       (global) signal in the state
%   mean_correlation   mean of the upper triangle of the state's correlation matrix
% Pooled over mice (scatter summed, as R.state_fc): the same four, and fc{k}.
% Pairs of states (a < b):
%   fc_pattern_r       Pearson r between the upper triangles of the pooled FC;
%                      r^2 is the share of one pattern's variance that an affine
%                      function of the other (a uniform shift and scaling of all
%                      correlations, as a global signal produces) explains
%   fc_pattern_r_disattenuated  the same comparison corrected for estimation
%                      noise (Spearman 1904): FC pooled over odd- and over
%                      even-numbered mice; between-state r across halves
%                      (mean of r(a odd, b even) and r(a even, b odd)) divided
%                      by sqrt(r(a odd, a even) * r(b odd, b even)). Near 1:
%                      the two patterns agree as well as each agrees with
%                      itself, so they differ only by a shift and scaling.
%                      Needed because weak or noisy FC lowers the raw r even
%                      when the states share one pattern.
%   ratios a/b of the pooled mean and global-signal variance, the difference in
%   mean correlation, and per mouse the median log2 ratio and the fraction of
%   mice with the higher value in state a (mice with both states defined)
%
% Fields of D: subject (struct of S x K arrays: fo, effective_n, mean_variance,
% gs_variance, mean_correlation), pooled (struct of 1 x K arrays and fc, 1 x K
% cell), pairs (struct array). No group information is used.

if nargin < 4 || isempty(min_effective_n), min_effective_n = 0; end
[fc, neff, sc] = hmm_subject_state_fc(Ts_list, Gamma, hmm_train, 0);
S = numel(Ts_list);
K = size(Gamma{1}, 2);
N = size(Ts_list{1}, 2);
up = triu(true(N), 1);
one = ones(N, 1);

sub = struct('fo', NaN(S, K), 'effective_n', neff, 'mean_variance', NaN(S, K), ...
    'gs_variance', NaN(S, K), 'mean_correlation', NaN(S, K));
Spool = repmat({zeros(N)}, 1, K);
wsum = zeros(1, K);
for s = 1:S
    G = double(Gamma{s});
    for k = 1:K
        w = G(:, k);
        sub.fo(s, k) = mean(w);
        Spool{k} = Spool{k} + sc{s, k};
        wsum(k) = wsum(k) + sum(w);
        if neff(s, k) < min_effective_n || sum(w) <= eps, continue; end
        Sk = (sc{s, k} + sc{s, k}') / 2;
        sub.mean_variance(s, k) = trace(Sk) / (N * sum(w));
        sub.gs_variance(s, k) = (one' * Sk * one) / (N ^ 2 * sum(w));
        sub.mean_correlation(s, k) = mean(fc{s, k}(up));
    end
end

pool = struct('fo', mean(sub.fo, 1), 'mean_variance', NaN(1, K), 'gs_variance', NaN(1, K), ...
    'mean_correlation', NaN(1, K), 'fc_split_half_r', NaN(1, K));
pool.fc = cell(1, K);
half = {1:2:S, 2:2:S};                       % odd- and even-numbered mice
Fh = cell(2, K);
for h = 1:2
    for k = 1:K
        Sk = zeros(N);
        for s = half{h}, Sk = Sk + sc{s, k}; end
        Fh{h, k} = local_corr_from_scatter(Sk);
    end
end
for k = 1:K
    Sk = (Spool{k} + Spool{k}') / 2;
    sd = sqrt(diag(Sk));
    Rk = Sk ./ (sd * sd');
    Rk(1:N + 1:end) = 1;
    pool.fc{k} = Rk;
    pool.mean_variance(k) = trace(Sk) / (N * wsum(k));
    pool.gs_variance(k) = (one' * Sk * one) / (N ^ 2 * wsum(k));
    pool.mean_correlation(k) = mean(Rk(up));
    pool.fc_split_half_r(k) = local_r(Fh{1, k}(up), Fh{2, k}(up));
end

pairs = struct('state_a', {}, 'state_b', {}, 'fc_pattern_r', {}, 'fc_pattern_r2', {}, ...
    'fc_pattern_r_across_halves', {}, 'fc_pattern_r_disattenuated', {}, ...
    'mean_correlation_diff', {}, 'mean_variance_ratio', {}, 'gs_variance_ratio', {}, ...
    'n_mice', {}, 'median_log2_mean_variance_ratio', {}, 'frac_mice_higher_mean_variance', {}, ...
    'median_log2_gs_variance_ratio', {}, 'frac_mice_higher_gs_variance', {}, ...
    'median_mean_correlation_diff', {});
for a = 1:K - 1
    for b = a + 1:K
        c = corrcoef(pool.fc{a}(up), pool.fc{b}(up));
        rx = mean([local_r(Fh{1, a}(up), Fh{2, b}(up)), local_r(Fh{2, a}(up), Fh{1, b}(up))]);
        rel = pool.fc_split_half_r(a) * pool.fc_split_half_r(b);
        if rel > 0, rdis = rx / sqrt(rel); else, rdis = NaN; end
        ok = isfinite(sub.mean_variance(:, a)) & isfinite(sub.mean_variance(:, b));
        lv = log2(sub.mean_variance(ok, a) ./ sub.mean_variance(ok, b));
        lg = log2(sub.gs_variance(ok, a) ./ sub.gs_variance(ok, b));
        dc = sub.mean_correlation(ok, a) - sub.mean_correlation(ok, b);
        pairs(end + 1) = struct('state_a', a, 'state_b', b, 'fc_pattern_r', c(1, 2), ...
            'fc_pattern_r2', c(1, 2) ^ 2, 'fc_pattern_r_across_halves', rx, ...
            'fc_pattern_r_disattenuated', rdis, ...
            'mean_correlation_diff', pool.mean_correlation(a) - pool.mean_correlation(b), ...
            'mean_variance_ratio', pool.mean_variance(a) / pool.mean_variance(b), ...
            'gs_variance_ratio', pool.gs_variance(a) / pool.gs_variance(b), ...
            'n_mice', sum(ok), 'median_log2_mean_variance_ratio', local_median(lv), ...
            'frac_mice_higher_mean_variance', local_mean(lv > 0, ok), ...
            'median_log2_gs_variance_ratio', local_median(lg), ...
            'frac_mice_higher_gs_variance', local_mean(lg > 0, ok), ...
            'median_mean_correlation_diff', local_median(dc)); %#ok<AGROW>
    end
end
D = struct('subject', sub, 'pooled', pool, 'pairs', pairs, 'min_effective_n', min_effective_n);
end


function F = local_corr_from_scatter(Sk)
Sk = (Sk + Sk') / 2;
sd = sqrt(diag(Sk));
F = Sk ./ (sd * sd');
F(1:size(F, 1) + 1:end) = 1;
end


function r = local_r(x, y)
if numel(x) < 3 || std(x) == 0 || std(y) == 0, r = NaN; return; end
c = corrcoef(x, y);
r = c(1, 2);
end


function m = local_median(x)
if isempty(x), m = NaN; else, m = median(x); end
end


function m = local_mean(x, ok)
if ~any(ok), m = NaN; else, m = mean(x); end
end
