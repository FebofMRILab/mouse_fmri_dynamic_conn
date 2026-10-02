function Xs = hmm_iaaft_surrogate(X, stream, max_iter)
% HMM_IAAFT_SURROGATE  Multivariate IAAFT surrogate of a (T x n_roi) time series.
%
%   Xs = hmm_iaaft_surrogate(X, stream, max_iter)
%
% Multivariate iterative amplitude-adjusted Fourier transform (Schreiber &
% Schmitz 2000). A stricter null than plain phase randomisation:
%
%   preserved EXACTLY   each ROI's distribution of values (the same numbers,
%                       re-ordered in time), so skewness, heavy tails and
%                       amplitude outliers survive
%   preserved closely   each ROI's power spectrum and the cross-spectra
%                       between ROIs, hence autocorrelation and static FC
%   removed             non-stationarity (e.g. covariance switching) and
%                       non-linear temporal structure beyond what the
%                       marginals and spectra imply
%
% If an HMM effect beats the phase-randomised null but NOT this one, it is
% explained by non-Gaussian amplitude structure rather than by FC switching.
%
% Algorithm
%   start  phase-randomised surrogate (same random phase for all ROIs),
%          rank-remapped to the original values
%   repeat
%     spectral step  original Fourier amplitudes; at each frequency one
%                    common phase rotation for all ROIs (so the original
%                    relative phases, i.e. cross-spectral phases, are kept),
%                    chosen as the least-squares closest to the current
%                    iterate
%     rank step      each ROI takes its original values in the rank order
%                    of the spectral-step result
%   until the rank order stops changing or max_iter (default 200).
%   The result ends on a rank step, so marginals are exact and spectra are
%   approximate (the usual IAAFT convention).
%
% stream (optional) is a RandStream for reproducibility.

if nargin < 3 || isempty(max_iter)
    max_iter = 200;
end
if nargin < 2
    stream = [];
end

[T, N] = size(X);
X_sorted = sort(X, 1);
F0 = fft(X);
A = abs(F0);
phi0 = angle(F0);

nh = floor((T - 1) / 2);
pos = (2:nh + 1).';          % strictly positive frequencies
neg = T - pos + 2;           % their conjugate partners
col_offset = (0:N - 1) * T;  % for vectorised rank remapping

% Random start: phase-randomised surrogate, then rank remap
Y = hmm_phase_randomise(X, stream);
[~, idx] = sort(Y, 1);
Xs = zeros(T, N);
Xs(idx + col_offset) = X_sorted;
prev_idx = idx;

for it = 1:max_iter
    % Spectral step: least-squares common rotation alpha(f),
    % alpha = angle( sum_j S_j * A_j * exp(-i*phi0_j) )
    S = fft(Xs);
    w = sum(S(pos, :) .* A(pos, :) .* exp(-1i * phi0(pos, :)), 2);
    alpha = zeros(T, 1);
    alpha(pos) = angle(w);
    alpha(neg) = -alpha(pos);
    Y = real(ifft(A .* exp(1i * (phi0 + alpha))));

    % Rank step: original values in the new rank order
    [~, idx] = sort(Y, 1);
    Xs(idx + col_offset) = X_sorted;

    if isequal(idx, prev_idx)
        break;
    end
    prev_idx = idx;
end
end
