function Xs = hmm_phase_randomise(X, stream)
% HMM_PHASE_RANDOMISE  Phase-randomised surrogate of a (T x n_roi) time series.
%
%   Xs = hmm_phase_randomise(X, stream)
%
% The SAME random phase is added to every ROI at each frequency. This keeps
% each ROI's power spectrum (autocorrelation) and the cross-spectrum between
% ROIs, so the static covariance/FC is preserved exactly. What it removes is
% non-stationarity: in the surrogate, no epoch has a covariance different from
% any other, beyond chance. An HMM fitted to surrogates therefore shows how
% reproducible "states" are when there are no real state transitions.
%
% stream (optional) is a RandStream, so the surrogates can be reproduced.

T = size(X, 1);
F = fft(X);                         % column-wise FFT, T x n_roi
nh = floor((T - 1) / 2);            % number of strictly positive frequencies

if nargin < 2 || isempty(stream)
    phi = 2 * pi * rand(nh, 1);
else
    phi = 2 * pi * rand(stream, nh, 1);
end

ph = zeros(T, 1);                   % DC (and Nyquist when T is even) unchanged
ph(2:nh + 1) = phi;                 % positive frequencies
ph(T - nh + 1:T) = -flipud(phi);    % matching negative frequencies (conjugate)

Xs = real(ifft(F .* exp(1i * ph)));
end
