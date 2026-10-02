function st = dfc_cpca_subject_stats(Z, V, TR)
% DFC_CPCA_SUBJECT_STATS  One subject's complex-PCA measures on fixed loadings.
%
%   st = dfc_cpca_subject_stats(Z, V, TR)
%
% Z   T x N analytic signal of the subject's standardised data
% V   N x K group complex loadings (from dfc_cpca_fit)
% TR  repetition time in seconds
%
% Outputs (K x 1 unless noted)
%   var_expl   share of the subject's analytic power carried by each component
%              (static: preserved exactly by phase randomisation)
%   freq_hz    mean rotation frequency of the component time course
%              (static)
%   env_cv     coefficient of variation of the envelope |c_k(t)|: episodic,
%              bursty propagation raises it. For a stationary Gaussian process
%              the envelope is Rayleigh (CV ~ 0.52). This is the NON-
%              STATIONARITY measure; the null removes bursts.
%   lag_index  (scalar) share of coupling that is lagged rather than zero-lag:
%              sqrt(|Im C|^2 / |C|^2) over off-diagonal entries of the subject's
%              complex covariance, each estimated as a split-half cross-product
%              so that noise does not inflate it (static)

c = Z * V;       % C = Z'Z (C_ij = sum conj(z_i) z_j), so v'Cv = ||Z v||^2
P = mean(sum(abs(Z) .^ 2, 2));
st.var_expl = (mean(abs(c) .^ 2, 1) / P).';
env = abs(c);
st.env_cv = (std(env, 0, 1) ./ mean(env, 1)).';
dphi = angle(c(2:end, :) .* conj(c(1:end - 1, :)));
st.freq_hz = (mean(dphi, 1) / (2 * pi * TR)).';
% Split-half estimate: noise is independent between the two halves, so the
% cross-products estimate |Im C|^2 and |C|^2 without the noise floor that
% inflates the naive ratio (~0.71 for pure noise).
T = size(Z, 1);
h = floor(T / 2);
C1 = (Z(1:h, :)' * Z(1:h, :)) / h;
C2 = (Z(h + 1:T, :)' * Z(h + 1:T, :)) / (T - h);
off = ~eye(size(C1, 1));
num = sum(imag(C1(off)) .* imag(C2(off)));
den = sum(real(C1(off) .* conj(C2(off))));
st.lag_index = sqrt(max(num, 0) / max(den, eps));
end
