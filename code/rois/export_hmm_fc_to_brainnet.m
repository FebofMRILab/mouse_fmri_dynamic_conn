function output = export_hmm_fc_to_brainnet( ...
    state_fc, centroid_csv, out_dir, options)
% EXPORT_HMM_FC_TO_BRAINNET
%
% Export HMM state FC matrices as BrainNet Viewer .node and .edge files.
%
% INPUTS
%   state_fc:
%       K-by-1 cell array of N-by-N FC matrices, or N-by-N-by-K array.
%
%   centroid_csv:
%       CSV generated from the mouse atlas. Required columns:
%           roi_index
%           roi_name
%           plot_x
%           plot_y
%           plot_z
%
%       Rows must be in the same order as the ROI columns used for HMM.
%
%   out_dir:
%       Output directory for BrainNet files.
%
%   options:
%       Optional structure:
%
%       .retain_fraction
%           Fraction of eligible edges retained. Default = 0.05.
%
%       .edge_mode
%           'signed', 'positive', or 'negative'. Default = 'signed'.
%
%       .center_across_states
%           Subtract across-state mean FC before thresholding.
%           Default = false.
%
%       .node_size_mode
%           'strength' or 'constant'. Default = 'strength'.
%
%       .node_size_range
%           Minimum and maximum BrainNet node sizes. Default = [2 8].
%
%       .node_color_mode
%           'strength' or 'state'. Default = 'strength'.
%
%       .minimum_abs_correlation
%           Exclude edges below this absolute value before proportional
%           thresholding. Default = 0.
%
% OUTPUT
%   output:
%       Structure containing exported matrices and filenames.

if nargin < 4 || isempty(options)
    options = struct();
end

options = local_defaults(options);

if ~exist(centroid_csv, 'file')
    error('Centroid CSV was not found: %s', centroid_csv);
end

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%% Convert FC input to cell array
if isnumeric(state_fc)
    if ismatrix(state_fc)
        state_fc = {double(state_fc)};
    elseif ndims(state_fc) == 3
        K = size(state_fc, 3);
        state_fc = arrayfun( ...
            @(k) double(state_fc(:, :, k)), ...
            (1:K).', ...
            'UniformOutput', false);
    else
        error('Numeric state_fc must be N-by-N or N-by-N-by-K.');
    end
elseif iscell(state_fc)
    state_fc = state_fc(:);
    state_fc = cellfun(@double, state_fc, ...
        'UniformOutput', false);
else
    error('state_fc must be numeric or a cell array.');
end

K = numel(state_fc);
n_roi = size(state_fc{1}, 1);

%% Validate FC matrices
for k = 1:K
    C = state_fc{k};

    if ~isequal(size(C), [n_roi, n_roi])
        error('State %d does not contain an %d-by-%d matrix.', ...
            k, n_roi, n_roi);
    end

    if any(~isfinite(C(:)))
        error('State %d FC matrix contains nonfinite values.', k);
    end

    C = (C + C.') ./ 2;
    C(1:n_roi + 1:end) = 0;
    state_fc{k} = C;
end

%% Load atlas-derived ROI coordinates
roi_tbl = readtable(centroid_csv, 'TextType', 'string');

required_variables = { ...
    'roi_index', 'roi_name', ...
    'plot_x', 'plot_y', 'plot_z'};

missing_variables = setdiff( ...
    required_variables, roi_tbl.Properties.VariableNames);

if ~isempty(missing_variables)
    error('Centroid table is missing: %s', ...
        strjoin(missing_variables, ', '));
end

if height(roi_tbl) ~= n_roi
    error(['Centroid table contains %d ROIs, but FC matrices contain ' ...
        '%d ROIs.'], height(roi_tbl), n_roi);
end

if ~isequal(double(roi_tbl.roi_index(:)), (1:n_roi).')
    error(['roi_index must run from 1 to %d in the same order as the ' ...
        'HMM input columns.'], n_roi);
end

coordinates = [ ...
    double(roi_tbl.plot_x), ...
    double(roi_tbl.plot_y), ...
    double(roi_tbl.plot_z)];

if any(~isfinite(coordinates(:)))
    error('ROI coordinates contain nonfinite values.');
end

roi_names = string(roi_tbl.roi_name);
roi_names = regexprep(strtrim(roi_names), '\s+', '_');
roi_names(strlength(roi_names) == 0) = ...
    "ROI_" + string(find(strlength(roi_names) == 0));

%% Optionally calculate state deviations from mean FC
if options.center_across_states
    fc_stack = cat(3, state_fc{:});
    mean_fc = mean(fc_stack, 3);

    for k = 1:K
        state_fc{k} = state_fc{k} - mean_fc;
        state_fc{k}(1:n_roi + 1:end) = 0;
    end
end

%% Export each state
node_files = strings(K, 1);
edge_files = strings(K, 1);
edge_csv_files = strings(K, 1);

thresholded_fc = cell(K, 1);
node_strength = NaN(n_roi, K);

for k = 1:K
    C = state_fc{k};

    [W, edge_tbl] = local_threshold_fc( ...
        C, ...
        options.retain_fraction, ...
        options.edge_mode, ...
        options.minimum_abs_correlation);

    thresholded_fc{k} = W;
    node_strength(:, k) = sum(abs(W), 2);

    node_size = local_scale_values( ...
        node_strength(:, k), ...
        options.node_size_range(1), ...
        options.node_size_range(2));

    switch lower(options.node_size_mode)
        case 'strength'
            % Use scaled strength values calculated above.

        case 'constant'
            node_size(:) = mean(options.node_size_range);

        otherwise
            error('Unknown node_size_mode.');
    end

    switch lower(options.node_color_mode)
        case 'strength'
            node_color = node_strength(:, k);

        case 'state'
            node_color = repmat(k, n_roi, 1);

        otherwise
            error('Unknown node_color_mode.');
    end

    node_files(k) = fullfile(out_dir, sprintf( ...
        'HMM_state_%02d.node', k));

    edge_files(k) = fullfile(out_dir, sprintf( ...
        'HMM_state_%02d.edge', k));

    edge_csv_files(k) = fullfile(out_dir, sprintf( ...
        'HMM_state_%02d_edges.csv', k));

    local_write_brainnet_node( ...
        node_files(k), ...
        coordinates, ...
        node_color, ...
        node_size, ...
        roi_names);

    writematrix(W, edge_files(k), ...
        'Delimiter', 'tab', ...
        'FileType', 'text');

    edge_tbl.state = repmat(k, height(edge_tbl), 1);
    writetable(edge_tbl, edge_csv_files(k));

    fprintf('State %d:\n', k);
    fprintf('  Node file: %s\n', node_files(k));
    fprintf('  Edge file: %s\n', edge_files(k));
    fprintf('  Retained edges: %d\n', height(edge_tbl));
end

%% Save state-level node metrics
node_tbl = table();

for k = 1:K
    tmp = table( ...
        repmat(k, n_roi, 1), ...
        (1:n_roi).', ...
        roi_names, ...
        coordinates(:, 1), ...
        coordinates(:, 2), ...
        coordinates(:, 3), ...
        node_strength(:, k), ...
        'VariableNames', { ...
        'state', ...
        'roi_index', ...
        'roi_name', ...
        'x', ...
        'y', ...
        'z', ...
        'retained_absolute_strength'});

    node_tbl = [node_tbl; tmp]; %#ok<AGROW>
end

node_metric_csv = fullfile(out_dir, ...
    'HMM_BrainNet_node_metrics.csv');

writetable(node_tbl, node_metric_csv);

%% Return outputs
output = struct();
output.node_files = node_files;
output.edge_files = edge_files;
output.edge_csv_files = edge_csv_files;
output.node_metric_csv = node_metric_csv;
output.coordinates = coordinates;
output.roi_names = roi_names;
output.thresholded_fc = thresholded_fc;
output.node_strength = node_strength;
output.options = options;

save(fullfile(out_dir, 'HMM_BrainNet_export.mat'), ...
    'output', '-v7.3');

fprintf('\nBrainNet export completed: %s\n', out_dir);
end


function options = local_defaults(options)

defaults = struct();
defaults.retain_fraction = 0.05;
defaults.edge_mode = 'signed';
defaults.center_across_states = false;
defaults.node_size_mode = 'strength';
defaults.node_size_range = [2, 8];
defaults.node_color_mode = 'strength';
defaults.minimum_abs_correlation = 0;

names = fieldnames(defaults);

for i = 1:numel(names)
    name = names{i};

    if ~isfield(options, name) || isempty(options.(name))
        options.(name) = defaults.(name);
    end
end

if ~isscalar(options.retain_fraction) || ...
        options.retain_fraction <= 0 || ...
        options.retain_fraction > 1
    error('retain_fraction must be in (0,1].');
end

if numel(options.node_size_range) ~= 2 || ...
        options.node_size_range(1) <= 0 || ...
        options.node_size_range(2) < options.node_size_range(1)
    error('node_size_range must be [minimum maximum].');
end

if options.minimum_abs_correlation < 0 || ...
        options.minimum_abs_correlation > 1
    error('minimum_abs_correlation must be between 0 and 1.');
end
end


function [W, edge_tbl] = local_threshold_fc( ...
    C, retain_fraction, edge_mode, minimum_abs_correlation)

n_roi = size(C, 1);
upper_mask = triu(true(n_roi), 1);

[i_all, j_all] = find(upper_mask);
weights_all = C(upper_mask);

switch lower(edge_mode)
    case 'signed'
        eligible = abs(weights_all) >= minimum_abs_correlation & ...
            weights_all ~= 0;
        ranking = abs(weights_all);

    case 'positive'
        eligible = weights_all > 0 & ...
            weights_all >= minimum_abs_correlation;
        ranking = weights_all;

    case 'negative'
        eligible = weights_all < 0 & ...
            abs(weights_all) >= minimum_abs_correlation;
        ranking = abs(weights_all);

    otherwise
        error('edge_mode must be signed, positive, or negative.');
end

eligible_idx = find(eligible);

if isempty(eligible_idx)
    W = zeros(n_roi);

    edge_tbl = table( ...
        zeros(0, 1), zeros(0, 1), ...
        zeros(0, 1), zeros(0, 1), ...
        'VariableNames', { ...
        'roi_i', 'roi_j', ...
        'correlation', 'absolute_correlation'});
    return;
end

n_keep = max(1, round( ...
    retain_fraction * numel(eligible_idx)));

[~, order] = sort( ...
    ranking(eligible_idx), 'descend');

selected = eligible_idx(order(1:n_keep));

i = i_all(selected);
j = j_all(selected);
weights = weights_all(selected);

W = zeros(n_roi);
upper_linear = sub2ind([n_roi, n_roi], i, j);
W(upper_linear) = weights;
W = W + W.';

edge_tbl = table( ...
    i, j, weights, abs(weights), ...
    'VariableNames', { ...
    'roi_i', 'roi_j', ...
    'correlation', 'absolute_correlation'});

edge_tbl = sortrows( ...
    edge_tbl, 'absolute_correlation', 'descend');
end


function values_scaled = local_scale_values( ...
    values, output_minimum, output_maximum)

values = double(values(:));
finite_values = values(isfinite(values));

if isempty(finite_values) || ...
        max(finite_values) - min(finite_values) <= eps
    values_scaled = repmat( ...
        mean([output_minimum, output_maximum]), ...
        size(values));
    return;
end

values_scaled = output_minimum + ...
    (values - min(finite_values)) ./ ...
    (max(finite_values) - min(finite_values)) .* ...
    (output_maximum - output_minimum);

values_scaled(~isfinite(values_scaled)) = output_minimum;
end


function local_write_brainnet_node( ...
    filename, coordinates, node_color, node_size, labels)

fid = fopen(filename, 'w');

if fid < 0
    error('Could not open node file for writing: %s', filename);
end

cleanup = onCleanup(@() fclose(fid));

for r = 1:size(coordinates, 1)
    fprintf(fid, ...
        '%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%s\n', ...
        coordinates(r, 1), ...
        coordinates(r, 2), ...
        coordinates(r, 3), ...
        node_color(r), ...
        node_size(r), ...
        char(labels(r)));
end
end