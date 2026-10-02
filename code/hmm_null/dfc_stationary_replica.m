function R = dfc_stationary_replica(X, frac, factor)
% DFC_STATIONARY_REPLICA  Stationary Gaussian replica of X for calibration.
%
%   R = dfc_stationary_replica(X)            % frac = 0.05, factor = 3
%
% Returns a T x N series that is stationary by construction and has (nearly)
% the same auto- and cross-spectra as X (lags included), but is cut from a
% longer series, so, like trimmed real data, its first and last samples do
% not join up. Running the observed-vs-surrogate procedure on replicas gives
% the distribution of the test statistics when there is NO non-stationarity,
% which calibrates any bias of the procedure itself.
%
% Method: cosine-taper both ends (frac of T), zero-pad to factor*T, apply
% multivariate phase randomisation (same random phases for every channel),
% keep the middle T samples, restore each channel's mean and SD.
% Uses the global random stream: seed it before calling.

if nargin < 2 || isempty(frac)
    frac = 0.05;
end
if nargin < 3 || isempty(factor)
    factor = 3;
end
[T, N] = size(X);
L = max(2, round(frac * T));
ramp = 0.5 * (1 - cos(pi * (0:L - 1)' / L));
w = ones(T, 1);
w(1:L) = ramp;
w(end - L + 1:end) = flipud(ramp);
mu = mean(X, 1);
sd = std(X, 0, 1);
S = hmm_phase_randomise([(X - mu) .* w; zeros((factor - 1) * T, N)], []);
k0 = floor((factor - 1) * T / 2);
R = S(k0 + 1:k0 + T, :);
R = (R - mean(R, 1)) ./ std(R, 0, 1) .* sd + mu;
end
