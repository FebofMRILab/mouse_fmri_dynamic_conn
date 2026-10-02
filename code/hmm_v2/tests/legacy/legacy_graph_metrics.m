% Verbatim copy of compute_graph_metrics_single_robust (lines 394-499) from the previous run5 script, for the equivalence tests only.
function metrics = legacy_graph_metrics(A, density, seed)
% Positive-edge graph analysis.
%
% IMPORTANT: Negative correlations are intentionally excluded. If negative
% edges are scientifically meaningful, replace this with a signed-network
% analysis rather than interpreting these as whole-correlation graphs.

if nargin < 3 || isempty(seed)
    seed = 1;
end
if ~isscalar(density) || density <= 0 || density > 1
    error('density must be in (0,1].');
end
if ~ismatrix(A) || size(A, 1) ~= size(A, 2) || ...
        any(~isfinite(A(:)))
    error('A must be a finite square matrix.');
end

p = size(A, 1);
A = (A + A.') ./ 2;
A(1:p + 1:end) = 0;
A = max(A, 0);

% Exact rank-based thresholding avoids tie-dependent over-retention.
upper_mask = triu(true(p), 1);
upper_idx = find(upper_mask);
upper_weights = A(upper_idx);

n_possible = numel(upper_idx);
n_keep_target = max(1, round(density * n_possible));

positive_idx = find(upper_weights > 0);
if isempty(positive_idx)
    error('The matrix has no positive off-diagonal edges.');
end

[~, order] = sort(upper_weights(positive_idx), 'descend');
n_keep = min(n_keep_target, numel(positive_idx));
selected_upper_idx = upper_idx(positive_idx(order(1:n_keep)));

W = zeros(p);
W(selected_upper_idx) = A(selected_upper_idx);
W = W + W.';
B = double(W > 0);

metrics = struct();
metrics.density_target = density;
metrics.density_actual = nnz(triu(B, 1)) / n_possible;

Gbin = graph(B);
component_labels = conncomp(Gbin);
component_sizes = accumarray(component_labels(:), 1);

metrics.n_components = numel(component_sizes);
metrics.largest_component_fraction = max(component_sizes) / p;
metrics.is_connected = double(metrics.n_components == 1);

metrics.strength_mean = mean(strengths_und(W));
metrics.C_bin = mean(clustering_coef_bu(B), 'omitnan');
metrics.C_wei = mean(clustering_coef_wu(W), 'omitnan');

% Characteristic path length is reported only for connected graphs.
% Global efficiency remains meaningful when graphs are disconnected.
if metrics.is_connected
    Dbin = distance_bin(B);
    metrics.L_bin = charpath(Dbin, 0, 1);

    Wlen = zeros(size(W));
    nz = W > 0;
    Wlen(nz) = 1 ./ W(nz);
    Dwei = distance_wei(Wlen);
    metrics.L_wei = charpath(Dwei, 0, 1);
else
    metrics.L_bin = NaN;
    metrics.L_wei = NaN;
end

metrics.Eglob_bin = efficiency_bin(B);
metrics.Eglob_wei = efficiency_wei(W);

if nnz(B) > 0
    metrics.assort_bin = assortativity_bin(B, 0);
    metrics.assort_wei = assortativity_wei(W, 0);
else
    metrics.assort_bin = NaN;
    metrics.assort_wei = NaN;
end

% Deterministic stream for reproducible stochastic Louvain optimization.
old_stream = RandStream.getGlobalStream;
cleanup = onCleanup(@() RandStream.setGlobalStream(old_stream));
new_stream = RandStream('mt19937ar', 'Seed', seed);
RandStream.setGlobalStream(new_stream);

n_louvain = 50;
Q_values = NaN(n_louvain, 1);

for rep = 1:n_louvain
    [~, Q_values(rep)] = community_louvain(W, 1.0, [], 'modularity');
end

metrics.Q_mod = max(Q_values);
metrics.Q_mod_mean = mean(Q_values, 'omitnan');
metrics.Q_mod_sd = std(Q_values, 0, 'omitnan');
metrics.transitivity_bin = transitivity_bu(B);
end
