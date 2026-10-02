function Z = dfc_analytic_signal(X)
% DFC_ANALYTIC_SIGNAL  Analytic signal of each column of X (T x N), via FFT.
%
%   Z = dfc_analytic_signal(X)
%
% Same result as hilbert() from the Signal Processing Toolbox, without the
% toolbox: real(Z) = X, imag(Z) = Hilbert transform of X.
T = size(X, 1);
h = zeros(T, 1);
h(1) = 1;
if mod(T, 2) == 0
    h(2:T / 2) = 2;
    h(T / 2 + 1) = 1;
else
    h(2:(T + 1) / 2) = 2;
end
Z = ifft(fft(X) .* h);
end
