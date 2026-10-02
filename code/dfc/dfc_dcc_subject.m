function out = dfc_dcc_subject(Y, ar_order, do_dcc)
% DFC_DCC_SUBJECT  AR-GARCH(1,1)-DCC(1,1) for one subject's multivariate series.
%
%   out = dfc_dcc_subject(Y, ar_order)
%
% Y         T x N time series (e.g. one subject's 25 group-PCA components)
% ar_order  order of the AR prewhitening model (default 2; 0 = none)
%
% Pipeline (Engle 2002; applied to fMRI by Lindquist et al. 2014):
%   1. AR(p) mean model per series (least squares) -> residuals e
%   2. GARCH(1,1) per series, variance targeting -> sigma_t;  u = e ./ sigma_t
%   3. DCC(1,1), correlation targeting, on u:
%        Q_t = (1-a-b) Qbar + a u_{t-1} u_{t-1}' + b Q_{t-1}
%        R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)
%      Constant conditional correlation (CCC) is the special case a = b = 0.
%
% Output fields (scalars)
%   dcc_a          DCC news parameter: 0 = constant correlation
%   dcc_b, dcc_persist  diagnostics only: b is not identified when a ~ 0
%   dcc_LR         2*(LL_DCC - LL_CCC): evidence for time-varying correlation.
%                  Its null distribution is non-standard (b unidentified at
%                  a = 0), so calibrate against surrogates, never chi-square.
%   corr_fluct     mean over pairs of the SD over time of R_t(i,j)
%   garch_alpha_mean  mean over series of GARCH alpha (0 = constant variance)
%   garch_persist  mean of alpha + beta: diagnostic only, beta is arbitrary
%                  when alpha ~ 0 (checked on white noise)
%   garch_LR       mean over series of 2*(LL_GARCH - LL_constant-variance)
%   vol_cv         mean over series of the CV of sigma_t
%   garch_alpha, garch_beta (N x 1)
%
% do_dcc (default true): false skips step 3 (DCC fields returned as NaN);
% used for IAAFT surrogates, which are not a valid null for DCC on
% band-limited data (see README).
%
% Base MATLAB only (fminsearch): no Econometrics or Optimization Toolbox.
%
% IMPORTANT: band-pass-filtered data sampled well above twice the low-pass
% edge cannot be whitened by any AR model (no power above the cut-off), and
% DCC/GARCH then fit the smoothness as "dynamics" (a ~ 0.4 on stationary
% data). Decimate to just above twice the cut-off first; the runner does.

if nargin < 2 || isempty(ar_order)
    ar_order = 2;
end
if nargin < 3 || isempty(do_dcc)
    do_dcc = true;
end
[T, N] = size(Y);
p = ar_order;
if T <= p + 20
    error('Time series too short (%d points) for AR(%d).', T, p);
end

%% 1. AR(p) prewhitening
E = zeros(T - p, N);
for i = 1:N
    x = Y(:, i) - mean(Y(:, i));
    if p > 0
        Xl = zeros(T - p, p);
        for L = 1:p
            Xl(:, L) = x(p + 1 - L:T - L);
        end
        E(:, i) = x(p + 1:T) - Xl * (Xl \ x(p + 1:T));
    else
        E(:, i) = x;
    end
end
Te = size(E, 1);

%% 2. GARCH(1,1) per series
alpha = zeros(N, 1);
beta = zeros(N, 1);
lr_g = zeros(N, 1);
vcv = zeros(N, 1);
U = zeros(Te, N);
for i = 1:N
    [alpha(i), beta(i), sig2, nll_g, nll_c] = local_garch_fit(E(:, i));
    U(:, i) = E(:, i) ./ sqrt(sig2);
    lr_g(i) = max(2 * (nll_c - nll_g), 0);
    s = sqrt(sig2);
    vcv(i) = std(s) / mean(s);
end

out.garch_alpha_mean = mean(alpha);
out.garch_persist = mean(alpha + beta);   % diagnostic only: beta is arbitrary when alpha ~ 0
out.garch_LR = mean(lr_g);
out.vol_cv = mean(vcv);
out.garch_alpha = alpha;
out.garch_beta = beta;
if ~do_dcc
    out.dcc_a = NaN;
    out.dcc_b = NaN;
    out.dcc_persist = NaN;
    out.dcc_LR = NaN;
    out.corr_fluct = NaN;
    return;
end

%% 3. DCC(1,1)
Uc = U - mean(U, 1);
Qbar = (Uc' * Uc) / (Te - 1);
d = sqrt(diag(Qbar));
Qbar = Qbar ./ (d * d');
Qbar = (Qbar + Qbar') / 2;

nll_ccc = local_dcc_nll(0, 0, U, Qbar);
fopt = optimset('Display', 'off', 'TolX', 1e-5, 'TolFun', 1e-6, ...
    'MaxFunEvals', 400, 'MaxIter', 400);
starts = [0.02 0.95; 0.05 0.50];
best_nll = Inf;
best_ab = [0 0];
for s = 1:size(starts, 1)
    th0 = local_ab_to_theta(starts(s, 1), starts(s, 2));
    [th, nll] = fminsearch(@(th) local_dcc_obj(th, U, Qbar), th0, fopt);
    if nll < best_nll
        best_nll = nll;
        [a, b] = local_theta_to_ab(th);
        best_ab = [a b];
    end
end
if ~(best_nll < nll_ccc)            % nothing better than constant correlation
    best_nll = nll_ccc;
    best_ab = [0 0];
end
[~, Rup] = local_dcc_nll(best_ab(1), best_ab(2), U, Qbar);

out.dcc_a = best_ab(1);
out.dcc_b = best_ab(2);
out.dcc_persist = sum(best_ab);
out.dcc_LR = max(2 * (nll_ccc - best_nll), 0);
out.corr_fluct = mean(std(Rup, 0, 1));
end


%% ------------------------------------------------------------------------
function [al, be, sig2, nll, nll_c] = local_garch_fit(e)
e = e(:);
s2 = mean(e .^ 2);
nll_c = 0.5 * numel(e) * (log(s2) + 1);        % constant variance
fopt = optimset('Display', 'off', 'TolX', 1e-5, 'TolFun', 1e-6, ...
    'MaxFunEvals', 300, 'MaxIter', 300);
starts = [0.05 0.85; 0.15 0.40];
nll = Inf;
al = 0;
be = 0;
for s = 1:size(starts, 1)
    th0 = local_ab_to_theta(starts(s, 1), starts(s, 2));
    [th, f] = fminsearch(@(th) local_garch_obj(th, e, s2), th0, fopt);
    if f < nll
        nll = f;
        [al, be] = local_theta_to_ab(th);
    end
end
if ~(nll < nll_c)
    al = 0;
    be = 0;
    nll = nll_c;
    sig2 = s2 * ones(size(e));
else
    [nll, sig2] = local_garch_nll(al, be, e, s2);
end
end


function f = local_garch_obj(th, e, s2)
[al, be] = local_theta_to_ab(th);
f = local_garch_nll(al, be, e, s2);
end


function [nll, sig2] = local_garch_nll(al, be, e, s2)
% sigma2_t = omega + al e_{t-1}^2 + be sigma2_{t-1}, omega = (1-al-be) s2
omega = (1 - al - be) * s2;
sig2 = [s2; filter(1, [1 -be], omega + al * e(1:end - 1) .^ 2, be * s2)];
sig2 = max(sig2, 1e-12);
nll = 0.5 * sum(log(sig2) + e .^ 2 ./ sig2);
end


function f = local_dcc_obj(th, U, Qbar)
[a, b] = local_theta_to_ab(th);
f = local_dcc_nll(a, b, U, Qbar);
end


function [nll, Rup] = local_dcc_nll(a, b, U, Qbar)
% Correlation part of the DCC negative log-likelihood (terms that do not
% depend on a, b dropped). Optional Rup: T x pairs, upper triangle of R_t.
[T, N] = size(U);
want_R = nargout > 1;
if want_R
    iu = find(triu(true(N), 1));
    Rup = zeros(T, numel(iu));
end
Q = Qbar;
nll = 0;
c0 = (1 - a - b) * Qbar;
for t = 1:T
    if t > 1
        up = U(t - 1, :)';
        Q = c0 + a * (up * up') + b * Q;
    end
    d = sqrt(diag(Q));
    R = Q ./ (d * d');
    [L, flag] = chol(R, 'lower');
    if flag
        nll = Inf;
        return;
    end
    z = L \ U(t, :)';
    nll = nll + sum(log(diag(L))) + 0.5 * (z' * z);
    if want_R
        Rup(t, :) = R(iu).';
    end
end
end


function th = local_ab_to_theta(a, b)
% Unconstrained parameters for (a, b) with a, b >= 0 and a + b < 0.999
pers = a + b;
share = a / pers;
th = [log(pers / (0.999 - pers)), log(share / (1 - share))];
end


function [a, b] = local_theta_to_ab(th)
pers = 0.999 / (1 + exp(-th(1)));
share = 1 / (1 + exp(-th(2)));
a = pers * share;
b = pers * (1 - share);
end
