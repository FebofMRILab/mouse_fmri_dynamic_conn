%% ==============================================================
%% run5B_hmm_block_bootstrap_stability.m
%% ==============================================================
% Block-bootstrap stability analysis for subject-specific,
% state-specific HMM functional-connectivity networks.
%
% Outputs:
%   1. bootstrap_stability_subject_state.csv
%   2. bootstrap_stability_by_state.csv
%   3. hmm_block_bootstrap_stability.mat
%
% Stability outcomes:
%   - Effective sample size (Neff)
%   - p / Neff
%   - Covariance effective rank
%   - Edge-set Jaccard similarity
%   - Spearman correlation of nodal-strength rankings
%   - Percentage of analyzable subject-state observations
%
% The moving-block bootstrap resamples paired rows of:
%       [time series, posterior state probabilities]
%
% Graph construction follows a positive-edge proportional-threshold
% approach. Nodal-rank stability refers to weighted node strength.

clc;
clear;
close all;

%% -------------------- Configuration ----------------------------

K = 3;
Comp = 50;

hmm_results_path = sprintf( ...
    '../signals/hmmmar_simple_k%d_hmmpca%d/raw_hmm_results.mat', ...
    K, Comp);

data_dir = '../signals/hmm_signals';
file_pattern = 'signal%d.txt';

out_dir = sprintf( ...
    '../signals/hmmmar_simple_k%d_hmmpca%d/hmm_bootstrap_stability', ...
    K, Comp);

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% Number of bootstrap repetitions.
n_bootstrap = 10000;

% Graph density. This retains the strongest 15% positive edges.
graph_density = 0.15;

% Occupancy criterion, not a covariance-identifiability criterion.
min_effective_n = 50;

% A subject-state must have this fraction of successful bootstrap
% repetitions to be classified as analyzable.
min_bootstrap_success_fraction = 0.80;

% Set [] to choose ceil(T^(1/3)) separately for each subject.
% With T = 197 this gives approximately 6 time points.
block_length = [];

random_seed = 20260902;

fprintf('Loading HMM results...\n');
loaded = load(hmm_results_path, 'R');
R = loaded.R;

K = R.K;
S = numel(R.subject_ids);
p = R.n_roi;

subject_ids = R.subject_ids;
groups = R.groups;

assert(iscell(R.Gamma), ...
    'R.Gamma must be a subject-level cell array.');

assert(numel(R.Gamma) == S, ...
    'Number of Gamma matrices does not match number of subjects.');

fprintf('Subjects: %d\n', S);
fprintf('States: %d\n', K);
fprintf('ROIs: %d\n', p);
fprintf('Bootstrap repetitions: %d\n', n_bootstrap);
fprintf('Graph density: %.3f\n', graph_density);
fprintf('Minimum Neff: %.2f\n', min_effective_n);

%% -------------------- Load time series --------------------------

Ts_list = cell(S, 1);

for s = 1:S
    subject_id = subject_ids(s);
    filename = fullfile( ...
        data_dir, sprintf(file_pattern, subject_id));

    if ~exist(filename, 'file')
        error('Missing time-series file: %s', filename);
    end

    X = double(readmatrix(filename));
    G = double(R.Gamma{s});

    if size(X, 1) ~= size(G, 1)
        error(['Time/Gamma mismatch for subject %d: ' ...
            '%d time points versus %d Gamma rows.'], ...
            subject_id, size(X, 1), size(G, 1));
    end

    if size(X, 2) ~= p
        error(['ROI mismatch for subject %d: expected %d, ' ...
            'found %d.'], ...
            subject_id, p, size(X, 2));
    end

    if size(G, 2) ~= K
        error(['State mismatch for subject %d: expected %d, ' ...
            'found %d.'], ...
            subject_id, K, size(G, 2));
    end

    if any(~isfinite(X(:)))
        error('Nonfinite time-series value for subject %d.', ...
            subject_id);
    end

    if any(~isfinite(G(:))) || any(G(:) < -1e-10)
        error('Invalid Gamma values for subject %d.', ...
            subject_id);
    end

    G(G < 0) = 0;
    row_sums = sum(G, 2);

    if any(row_sums <= eps)
        error('Zero-mass Gamma row for subject %d.', ...
            subject_id);
    end

    R.Gamma{s} = G ./ row_sums;
    Ts_list{s} = X;
end

fprintf('All time-series dimensions validated.\n');

%% -------------------- Allocate results --------------------------

n_rows = S * K;

subject_id_col = nan(n_rows, 1);
state_col = nan(n_rows, 1);
group_col = strings(n_rows, 1);

n_timepoints_col = nan(n_rows, 1);
block_length_col = nan(n_rows, 1);

FO_col = nan(n_rows, 1);
map_count_col = nan(n_rows, 1);

Neff_col = nan(n_rows, 1);
p_over_Neff_col = nan(n_rows, 1);

cov_rank_col = nan(n_rows, 1);
cov_effective_rank_col = nan(n_rows, 1);

original_positive_edges_col = nan(n_rows, 1);
original_graph_density_col = nan(n_rows, 1);

median_edge_jaccard_col = nan(n_rows, 1);
mean_edge_jaccard_col = nan(n_rows, 1);
edge_jaccard_ci_low_col = nan(n_rows, 1);
edge_jaccard_ci_high_col = nan(n_rows, 1);

median_nodal_rank_corr_col = nan(n_rows, 1);
mean_nodal_rank_corr_col = nan(n_rows, 1);
nodal_rank_corr_ci_low_col = nan(n_rows, 1);
nodal_rank_corr_ci_high_col = nan(n_rows, 1);

bootstrap_success_count_col = zeros(n_rows, 1);
bootstrap_success_fraction_col = zeros(n_rows, 1);

passes_Neff_col = false(n_rows, 1);
valid_original_network_col = false(n_rows, 1);
analyzable_col = false(n_rows, 1);

%% -------------------- Bootstrap analysis ------------------------

rng(random_seed, 'twister');

row = 0;

for s = 1:S

    X = Ts_list{s};
    G = R.Gamma{s};

    T = size(X, 1);

    if isempty(block_length)
        subject_block_length = max(2, ceil(T^(1/3)));
    else
        subject_block_length = min(T, block_length);
    end

    [~, map_path] = max(G, [], 2);

    fprintf('Subject %d/%d, ID=%d, T=%d, block=%d\n', ...
        s, S, subject_ids(s), T, subject_block_length);

    for k = 1:K

        row = row + 1;

        subject_id_col(row) = subject_ids(s);
        state_col(row) = k;
        group_col(row) = string(groups(s));

        n_timepoints_col(row) = T;
        block_length_col(row) = subject_block_length;

        weights = G(:, k);

        FO_col(row) = mean(weights);
        map_count_col(row) = nnz(map_path == k);

        Neff = local_effective_n(weights);

        Neff_col(row) = Neff;
        p_over_Neff_col(row) = p / max(Neff, eps);
        passes_Neff_col(row) = Neff >= min_effective_n;

        % Compute the original weighted covariance and correlation.
        [C_original, R_original, covariance_valid] = ...
            local_weighted_covariance(X, weights);

        if ~covariance_valid
            continue;
        end

        cov_rank_col(row) = local_covariance_rank(C_original);
        cov_effective_rank_col(row) = ...
            local_effective_rank(C_original);

        % Construct the original graph.
        [W_original, original_edge_ids, graph_valid] = ...
            local_positive_proportional_graph( ...
            R_original, graph_density);

        if ~graph_valid
            continue;
        end

        original_strength = sum(W_original, 2);

        original_positive_edges_col(row) = ...
            numel(original_edge_ids);

        original_graph_density_col(row) = ...
            numel(original_edge_ids) / (p * (p - 1) / 2);

        valid_original_network_col(row) = true;

        % Do not bootstrap subject-states that fail the prespecified
        % occupancy criterion.
        if ~passes_Neff_col(row)
            continue;
        end

        jaccard_values = nan(n_bootstrap, 1);
        nodal_rank_values = nan(n_bootstrap, 1);

        for b = 1:n_bootstrap

            bootstrap_indices = local_circular_block_indices( ...
                T, subject_block_length);

            X_boot = X(bootstrap_indices, :);
            G_boot = G(bootstrap_indices, :);

            % Preserve valid posterior rows after resampling.
            G_boot = G_boot ./ sum(G_boot, 2);
            weights_boot = G_boot(:, k);

            Neff_boot = local_effective_n(weights_boot);

            if Neff_boot < min_effective_n
                continue;
            end

            [~, R_boot, bootstrap_covariance_valid] = ...
                local_weighted_covariance( ...
                X_boot, weights_boot);

            if ~bootstrap_covariance_valid
                continue;
            end

            [W_boot, bootstrap_edge_ids, ...
                bootstrap_graph_valid] = ...
                local_positive_proportional_graph( ...
                R_boot, graph_density);

            if ~bootstrap_graph_valid
                continue;
            end

            % Edge-set Jaccard similarity.
            edge_union = union( ...
                original_edge_ids, bootstrap_edge_ids);

            edge_intersection = intersect( ...
                original_edge_ids, bootstrap_edge_ids);

            if ~isempty(edge_union)
                jaccard_values(b) = ...
                    numel(edge_intersection) / numel(edge_union);
            end

            % Spearman correlation of weighted nodal strengths.
            bootstrap_strength = sum(W_boot, 2);

            nodal_rank_values(b) = ...
                local_spearman_rank_correlation( ...
                original_strength, bootstrap_strength);
        end

        successful = ...
            isfinite(jaccard_values) & ...
            isfinite(nodal_rank_values);

        n_successful = nnz(successful);
        success_fraction = n_successful / n_bootstrap;

        bootstrap_success_count_col(row) = n_successful;
        bootstrap_success_fraction_col(row) = success_fraction;

        if n_successful > 0

            valid_jaccard = jaccard_values(successful);
            valid_rank = nodal_rank_values(successful);

            median_edge_jaccard_col(row) = ...
                median(valid_jaccard, 'omitnan');

            mean_edge_jaccard_col(row) = ...
                mean(valid_jaccard, 'omitnan');

            jaccard_ci = local_percentile_interval( ...
                valid_jaccard, [2.5, 97.5]);

            edge_jaccard_ci_low_col(row) = jaccard_ci(1);
            edge_jaccard_ci_high_col(row) = jaccard_ci(2);

            median_nodal_rank_corr_col(row) = ...
                median(valid_rank, 'omitnan');

            mean_nodal_rank_corr_col(row) = ...
                mean(valid_rank, 'omitnan');

            rank_ci = local_percentile_interval( ...
                valid_rank, [2.5, 97.5]);

            nodal_rank_corr_ci_low_col(row) = rank_ci(1);
            nodal_rank_corr_ci_high_col(row) = rank_ci(2);
        end

        analyzable_col(row) = ...
            passes_Neff_col(row) && ...
            valid_original_network_col(row) && ...
            success_fraction >= min_bootstrap_success_fraction;
    end
end

%% -------------------- Subject-state table -----------------------

subject_state_table = table( ...
    subject_id_col, ...
    state_col, ...
    group_col, ...
    n_timepoints_col, ...
    block_length_col, ...
    FO_col, ...
    map_count_col, ...
    Neff_col, ...
    p_over_Neff_col, ...
    cov_rank_col, ...
    cov_effective_rank_col, ...
    original_positive_edges_col, ...
    original_graph_density_col, ...
    median_edge_jaccard_col, ...
    mean_edge_jaccard_col, ...
    edge_jaccard_ci_low_col, ...
    edge_jaccard_ci_high_col, ...
    median_nodal_rank_corr_col, ...
    mean_nodal_rank_corr_col, ...
    nodal_rank_corr_ci_low_col, ...
    nodal_rank_corr_ci_high_col, ...
    bootstrap_success_count_col, ...
    bootstrap_success_fraction_col, ...
    passes_Neff_col, ...
    valid_original_network_col, ...
    analyzable_col, ...
    'VariableNames', { ...
    'subject_id', ...
    'state', ...
    'group', ...
    'n_timepoints', ...
    'block_length', ...
    'FO', ...
    'map_count', ...
    'Neff', ...
    'p_over_Neff', ...
    'covariance_rank', ...
    'covariance_effective_rank', ...
    'original_positive_edges', ...
    'original_graph_density', ...
    'median_edge_jaccard', ...
    'mean_edge_jaccard', ...
    'edge_jaccard_CI_low', ...
    'edge_jaccard_CI_high', ...
    'median_nodal_strength_rank_correlation', ...
    'mean_nodal_strength_rank_correlation', ...
    'nodal_rank_correlation_CI_low', ...
    'nodal_rank_correlation_CI_high', ...
    'bootstrap_success_count', ...
    'bootstrap_success_fraction', ...
    'passes_Neff', ...
    'valid_original_network', ...
    'analyzable'});

subject_output_path = fullfile( ...
    out_dir, 'bootstrap_stability_subject_state.csv');

writetable(subject_state_table, subject_output_path);

%% -------------------- State summary table -----------------------

state_col_summary = (1:K).';

n_subjects_total = zeros(K, 1);
n_passing_Neff = zeros(K, 1);
n_analyzable = zeros(K, 1);

median_Neff = nan(K, 1);
IQR_Neff = nan(K, 1);
median_p_over_Neff = nan(K, 1);

median_covariance_rank = nan(K, 1);
median_covariance_effective_rank = nan(K, 1);

median_edge_jaccard = nan(K, 1);
IQR_edge_jaccard = nan(K, 1);

median_nodal_rank_correlation = nan(K, 1);
IQR_nodal_rank_correlation = nan(K, 1);

percentage_passing_Neff = nan(K, 1);
percentage_analyzable_subject_states = nan(K, 1);

for k = 1:K

    state_rows = subject_state_table.state == k;
    analyzable_rows = state_rows & subject_state_table.analyzable;

    n_subjects_total(k) = nnz(state_rows);

    n_passing_Neff(k) = nnz( ...
        state_rows & subject_state_table.passes_Neff);

    n_analyzable(k) = nnz(analyzable_rows);

    median_Neff(k) = local_nanmedian( ...
        subject_state_table.Neff(state_rows));

    IQR_Neff(k) = local_naniqr( ...
        subject_state_table.Neff(state_rows));

    median_p_over_Neff(k) = local_nanmedian( ...
        subject_state_table.p_over_Neff(state_rows));

    median_covariance_rank(k) = local_nanmedian( ...
        subject_state_table.covariance_rank(state_rows));

    median_covariance_effective_rank(k) = local_nanmedian( ...
        subject_state_table.covariance_effective_rank(state_rows));

    % Stability summaries include only subject-states meeting the
    % complete analyzability definition.
    median_edge_jaccard(k) = local_nanmedian( ...
        subject_state_table.median_edge_jaccard( ...
        analyzable_rows));

    IQR_edge_jaccard(k) = local_naniqr( ...
        subject_state_table.median_edge_jaccard( ...
        analyzable_rows));

    median_nodal_rank_correlation(k) = local_nanmedian( ...
        subject_state_table. ...
        median_nodal_strength_rank_correlation( ...
        analyzable_rows));

    IQR_nodal_rank_correlation(k) = local_naniqr( ...
        subject_state_table. ...
        median_nodal_strength_rank_correlation( ...
        analyzable_rows));

    percentage_passing_Neff(k) = ...
        100 * n_passing_Neff(k) / n_subjects_total(k);

    percentage_analyzable_subject_states(k) = ...
        100 * n_analyzable(k) / n_subjects_total(k);
end

state_summary_table = table( ...
    state_col_summary, ...
    n_subjects_total, ...
    n_passing_Neff, ...
    n_analyzable, ...
    median_Neff, ...
    IQR_Neff, ...
    median_p_over_Neff, ...
    median_covariance_rank, ...
    median_covariance_effective_rank, ...
    median_edge_jaccard, ...
    IQR_edge_jaccard, ...
    median_nodal_rank_correlation, ...
    IQR_nodal_rank_correlation, ...
    percentage_passing_Neff, ...
    percentage_analyzable_subject_states, ...
    'VariableNames', { ...
    'state', ...
    'n_subject_states_total', ...
    'n_passing_Neff', ...
    'n_analyzable', ...
    'median_Neff', ...
    'IQR_Neff', ...
    'median_p_over_Neff', ...
    'median_covariance_rank', ...
    'median_covariance_effective_rank', ...
    'median_edge_jaccard', ...
    'IQR_edge_jaccard', ...
    'median_nodal_strength_rank_correlation', ...
    'IQR_nodal_strength_rank_correlation', ...
    'percentage_passing_Neff', ...
    'percentage_analyzable_subject_states'});

state_output_path = fullfile( ...
    out_dir, 'bootstrap_stability_by_state.csv');

writetable(state_summary_table, state_output_path);

%% -------------------- Save complete results ---------------------

settings = struct();
settings.n_bootstrap = n_bootstrap;
settings.graph_density = graph_density;
settings.min_effective_n = min_effective_n;
settings.min_bootstrap_success_fraction = ...
    min_bootstrap_success_fraction;
settings.block_length_requested = block_length;
settings.random_seed = random_seed;
settings.n_roi = p;
settings.K = K;

mat_output_path = fullfile( ...
    out_dir, 'hmm_block_bootstrap_stability.mat');

save(mat_output_path, ...
    'subject_state_table', ...
    'state_summary_table', ...
    'settings', ...
    '-v7.3');

fprintf('\n====================================================\n');
fprintf('Block-bootstrap stability analysis complete.\n');
fprintf('Subject-state table:\n%s\n', subject_output_path);
fprintf('State summary table:\n%s\n', state_output_path);
fprintf('MATLAB results:\n%s\n', mat_output_path);
fprintf('====================================================\n\n');

disp(state_summary_table);

%% ==============================================================
%% Local functions
%% ==============================================================

function Neff = local_effective_n(weights)

weights = double(weights(:));
weights(weights < 0) = 0;

sum_weights = sum(weights);
sum_squared_weights = sum(weights .^ 2);

if sum_weights <= eps || sum_squared_weights <= eps
    Neff = 0;
else
    Neff = sum_weights^2 / sum_squared_weights;
end
end

function [C, R, valid] = ...
    local_weighted_covariance(X, weights)

X = double(X);
weights = double(weights(:));

p = size(X, 2);

C = nan(p);
R = nan(p);
valid = false;

if size(X, 1) ~= numel(weights)
    return;
end

if any(~isfinite(X(:))) || any(~isfinite(weights))
    return;
end

weights(weights < 0) = 0;

total_weight = sum(weights);
sum_squared_weights = sum(weights .^ 2);

if total_weight <= eps
    return;
end

% Unbiased reliability-weighted covariance denominator.
denominator = ...
    total_weight - sum_squared_weights / total_weight;

if denominator <= eps
    return;
end

weighted_mean = (X.' * weights) ./ total_weight;
X_centered = X - weighted_mean.';

C = X_centered.' * ...
    (X_centered .* weights) ./ denominator;

C = (C + C.') ./ 2;

variances = diag(C);

if any(~isfinite(variances)) || any(variances <= eps)
    C = nan(p);
    return;
end

standard_deviations = sqrt(variances);

R = C ./ ...
    (standard_deviations * standard_deviations.');

R = max(min(R, 1), -1);
R = (R + R.') ./ 2;
R(1:p + 1:end) = 1;

if any(~isfinite(C(:))) || any(~isfinite(R(:)))
    C = nan(p);
    R = nan(p);
    return;
end

valid = true;
end

function covariance_rank = local_covariance_rank(C)

eigenvalues = real(eig((C + C.') ./ 2));
maximum_eigenvalue = max(eigenvalues);

if isempty(eigenvalues) || ...
        ~isfinite(maximum_eigenvalue) || ...
        maximum_eigenvalue <= eps
    covariance_rank = NaN;
    return;
end

tolerance = ...
    size(C, 1) * eps(maximum_eigenvalue);

covariance_rank = nnz(eigenvalues > tolerance);
end

function effective_rank = local_effective_rank(C)
% Entropy-based effective rank:
% exp(-sum(q_i * log(q_i))), where q_i are normalized eigenvalues.

eigenvalues = real(eig((C + C.') ./ 2));
eigenvalues(eigenvalues < 0) = 0;

total_variance = sum(eigenvalues);

if total_variance <= eps
    effective_rank = NaN;
    return;
end

probabilities = eigenvalues ./ total_variance;
probabilities = probabilities(probabilities > 0);

effective_rank = exp( ...
    -sum(probabilities .* log(probabilities)));
end

function [W, edge_ids, valid] = ...
    local_positive_proportional_graph(R, density)

p = size(R, 1);

W = zeros(p);
edge_ids = [];
valid = false;

if size(R, 2) ~= p || any(~isfinite(R(:)))
    return;
end

R = (R + R.') ./ 2;
R(1:p + 1:end) = 0;

% Positive-edge analysis.
R = max(R, 0);

upper_mask = triu(true(p), 1);
upper_indices = find(upper_mask);
upper_weights = R(upper_indices);

positive_positions = find(upper_weights > 0);

if isempty(positive_positions)
    return;
end

number_possible = numel(upper_indices);
number_to_keep = max(1, round(density * number_possible));
number_to_keep = min(number_to_keep, numel(positive_positions));

positive_weights = upper_weights(positive_positions);

% Secondary sorting by edge position provides deterministic handling
% of exact ties.
sorting_matrix = [ ...
    -positive_weights(:), positive_positions(:)];

[~, sorting_order] = sortrows(sorting_matrix, [1, 2]);

selected_positions = positive_positions( ...
    sorting_order(1:number_to_keep));

selected_linear_indices = ...
    upper_indices(selected_positions);

W(selected_linear_indices) = ...
    R(selected_linear_indices);

W = W + W.';

% The upper-triangle linear indices identify the selected edge set.
edge_ids = sort(selected_linear_indices(:));

valid = ...
    ~isempty(edge_ids) && ...
    all(isfinite(W(:)));
end

function indices = ...
    local_circular_block_indices(T, block_length)

number_blocks = ceil(T / block_length);
block_starts = randi(T, number_blocks, 1);

indices = zeros(numberBlocksTimesLength( ...
    number_blocks, block_length), 1);

position = 1;

for block = 1:number_blocks

    current_indices = mod( ...
        (block_starts(block) - 1) + ...
        (0:block_length - 1), T) + 1;

    final_position = position + block_length - 1;

    indices(position:final_position) = ...
        current_indices(:);

    position = final_position + 1;
end

indices = indices(1:T);
end

function n = numberBlocksTimesLength( ...
    number_blocks, block_length)

n = number_blocks * block_length;
end

function rho = local_spearman_rank_correlation(x, y)

x = double(x(:));
y = double(y(:));

valid_rows = isfinite(x) & isfinite(y);
x = x(valid_rows);
y = y(valid_rows);

if numel(x) < 3
    rho = NaN;
    return;
end

rank_x = tiedrank(x);
rank_y = tiedrank(y);

if std(rank_x) <= eps || std(rank_y) <= eps
    rho = NaN;
    return;
end

correlation_matrix = corrcoef(rank_x, rank_y);
rho = correlation_matrix(1, 2);
end

function interval = ...
    local_percentile_interval(values, percentiles)

values = values(isfinite(values));

if isempty(values)
    interval = [NaN, NaN];
    return;
end

interval = prctile(values, percentiles);
end

function value = local_nanmedian(x)

x = x(isfinite(x));

if isempty(x)
    value = NaN;
else
    value = median(x);
end
end

function value = local_naniqr(x)

x = x(isfinite(x));

if isempty(x)
    value = NaN;
else
    value = iqr(x);
end
end