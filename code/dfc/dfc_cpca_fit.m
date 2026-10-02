function G = dfc_cpca_fit(Zs, n_comp)
% DFC_CPCA_FIT  Group complex PCA of analytic signals (Bolt et al. 2022).
%
%   G = dfc_cpca_fit(Zs, n_comp)
%
% Zs      cell array of T_s x N analytic signals (from standardised data)
% n_comp  number of complex components to keep
%
% The Hermitian covariance C = Z'Z / n (C_ij = mean conj(z_i) z_j) is pooled
% over subjects and eigendecomposed. Each complex loading has a magnitude
% (how strongly a region takes part) and a phase (when, within the cycle, it
% peaks), so a component with spread-out phases describes a propagating
% pattern. Phase convention: a region lagging by tau seconds has loading
% phase +2*pi*f*tau, so lag_sec = angle(v) / (2*pi*f), positive = later
% (verified on a simulated travelling wave). Component time course: Z*V.
%
% Note: C is fixed by the cross-spectra of the data. Multivariate phase
% randomisation preserves the cross-spectra exactly, so the loadings and
% explained variance are IDENTICAL on phase-randomised surrogates. They
% describe stationary lag structure, not non-stationarity.
%
% Output: G.V (N x n_comp, unit-norm; phase fixed so the largest-magnitude
% loading has phase 0), G.lambda, G.explained, G.C

N = size(Zs{1}, 2);
C = zeros(N);
n = 0;
for s = 1:numel(Zs)
    C = C + Zs{s}' * Zs{s};                 % ' = conjugate transpose
    n = n + size(Zs{s}, 1);
end
C = C / n;
C = (C + C') / 2;
[V, D] = eig(C);
[lam, o] = sort(real(diag(D)), 'descend');
V = V(:, o(1:n_comp));
for k = 1:n_comp
    [~, im] = max(abs(V(:, k)));
    V(:, k) = V(:, k) * exp(-1i * angle(V(im, k)));
end
G.V = V;
G.lambda = lam(1:n_comp);
G.explained = lam(1:n_comp) / sum(lam);
G.C = C;
end
