function output = plot_mouse_hmm_fc_on_atlas( ...
    state_fc, atlas_path, roi_table_path, out_dir, options)
% PLOT_MOUSE_HMM_FC_ON_ATLAS
%
% Plot HMM state FC networks using ROI centroids calculated directly from
% a labeled mouse-brain parcellation in common space.
%
% This function does not require OSL or MNI coordinates.
%
% INPUTS
%   state_fc:
%       Either:
%         1) K-by-1 cell array, with each cell containing an N-by-N
%            correlation matrix; or
%         2) N-by-N-by-K numeric array.
%
%   atlas_path:
%       Path to a labeled 3-D NIfTI atlas. Zero is treated as background.
%
%   roi_table_path:
%       CSV containing:
%           atlas_value : integer label in the atlas
%           roi_name    : optional ROI name
%
%       The row order must match the ROI-column order used for HMM fitting.
%
%   out_dir:
%       Directory for figures, coordinates, edge tables, and NIfTI maps.
%
%   options:
%       Optional structure with fields:
%
%       .retain_fraction
%           Proportion of strongest absolute edges to display.
%           Default = 0.05, meaning the strongest 5%.
%
%       .edge_mode
%           'signed'   : retain positive and negative connections
%           'positive' : retain only positive connections
%           'negative' : retain only negative connections
%           Default = 'signed'
%
%       .center_across_states
%           If true, subtract the across-state mean FC matrix before
%           plotting. Default = false.
%
%       .node_metric
%           'strength' : node size reflects absolute retained strength
%           'constant' : all nodes have equal size
%           Default = 'strength'
%
%       .show_labels
%           Display ROI names next to nodes. Default = false.
%
%       .use_world_coordinates
%           Convert voxel centroids using the NIfTI affine transform.
%           Default = true.
%
%       .write_node_maps
%           Write state-level node-strength NIfTI maps.
%           Default = true.
%
%       .figure_visible
%           'on' or 'off'. Default = 'off'.
%
% OUTPUT
%   output:
%       Structure containing ROI coordinates, labels, thresholded
%       matrices, edge tables, and node-strength values.

if nargin < 5 || isempty(options)
    options = struct();
end

options = local_set_defaults(options);

if ~exist(atlas_path, 'file')
    error('Atlas file not found: %s', atlas_path);
end

if ~exist(roi_table_path, 'file')
    error('ROI-order table not found: %s', roi_table_path);
end

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%% ------------------------------------------------------------------------
% Convert FC input to a cell array
%% ------------------------------------------------------------------------

if isnumeric(state_fc)
    if ndims(state_fc) == 2
        state_fc = {double(state_fc)};
    elseif ndims(state_fc) == 3
        K = size(state_fc, 3);
        state_fc = arrayfun( ...
            @(k) double(state_fc(:, :, k)), ...
            1:K, ...
            'UniformOutput', false);
        state_fc = state_fc(:);
    else
        error('Numeric state_fc must be N-by-N or N-by-N-by-K.');
    end
elseif iscell(state_fc)
    state_fc = state_fc(:);
    state_fc = cellfun(@double, state_fc, ...
        'UniformOutput', false);
else
    error('state_fc must be a numeric array or cell array.');
end

K = numel(state_fc);

if K < 1
    error('No state FC matrices were supplied.');
end

n_roi = size(state_fc{1}, 1);

for k = 1:K
    C = state_fc{k};

    if ~ismatrix(C) || size(C, 1) ~= size(C, 2)
        error('State %d FC matrix is not square.', k);
    end

    if size(C, 1) ~= n_roi
        error('State %d has a different number of ROIs.', k);
    end

    if any(~isfinite(C(:)))
        error('State %d FC matrix contains nonfinite values.', k);
    end

    C = (C + C.') ./ 2;
    C(1:n_roi + 1:end) = 0;
    state_fc{k} = C;
end

%% ------------------------------------------------------------------------
% Read atlas and explicit ROI mapping
%% ------------------------------------------------------------------------

atlas_info = niftiinfo(atlas_path);
atlas = double(niftiread(atlas_info));

if ndims(atlas) ~= 3
    error('Atlas must be a three-dimensional labeled volume.');
end

roi_table = readtable(roi_table_path, 'TextType', 'string');

if ~ismember('atlas_value', roi_table.Properties.VariableNames)
    error('ROI table must contain an atlas_value column.');
end

atlas_values = double(roi_table.atlas_value(:));

if ismember('roi_name', roi_table.Properties.VariableNames)
    roi_names = string(roi_table.roi_name(:));
else
    roi_names = "ROI_" + string((1:height(roi_table)).');
end

if height(roi_table) ~= n_roi
    error(['ROI table has %d rows, but the FC matrices contain %d ROIs. ' ...
        'The atlas mapping must match the HMM signal-column order.'], ...
        height(roi_table), n_roi);
end

if numel(unique(atlas_values)) ~= n_roi
    error('atlas_value entries must be unique.');
end

if any(atlas_values == 0)
    error('atlas_value cannot be zero because zero is treated as background.');
end

atlas_labels_present = unique(atlas(:));
missing_labels = atlas_values(~ismember(atlas_values, atlas_labels_present));

if ~isempty(missing_labels)
    error('Atlas label(s) missing from the NIfTI volume: %s', ...
        mat2str(missing_labels(:).'));
end

unexpected_labels = setdiff( ...
    atlas_labels_present(atlas_labels_present ~= 0), ...
    atlas_values);

if ~isempty(unexpected_labels)
    warning(['The atlas contains labels not listed in the ROI-order table: ' ...
        '%s. These labels will not be used.'], ...
        mat2str(unexpected_labels(:).'));
end

%% ------------------------------------------------------------------------
% Calculate each ROI centroid
%% ------------------------------------------------------------------------

centroids_voxel = NaN(n_roi, 3);
voxel_count = zeros(n_roi, 1);

for r = 1:n_roi
    voxel_indices = find(atlas == atlas_values(r));
    voxel_count(r) = numel(voxel_indices);

    if isempty(voxel_indices)
        error('Atlas ROI %g contains no voxels.', atlas_values(r));
    end

    [i, j, k] = ind2sub(size(atlas), voxel_indices);

    centroids_voxel(r, :) = [ ...
        mean(i), ...
        mean(j), ...
        mean(k)];
end

if options.use_world_coordinates
    try
        centroids_plot = transformPointsForward( ...
            atlas_info.Transform, centroids_voxel);
        coordinate_space = "atlas_world";
    catch ME
        warning(['Could not apply the NIfTI affine transform (%s). ' ...
            'Using voxel coordinates instead.'], ME.message);

        centroids_plot = centroids_voxel;
        coordinate_space = "voxel";
    end
else
    centroids_plot = centroids_voxel;
    coordinate_space = "voxel";
end

coordinate_table = table( ...
    (1:n_roi).', ...
    atlas_values, ...
    roi_names, ...
    voxel_count, ...
    centroids_voxel(:, 1), ...
    centroids_voxel(:, 2), ...
    centroids_voxel(:, 3), ...
    centroids_plot(:, 1), ...
    centroids_plot(:, 2), ...
    centroids_plot(:, 3), ...
    repmat(coordinate_space, n_roi, 1), ...
    'VariableNames', { ...
    'roi_index', ...
    'atlas_value', ...
    'roi_name', ...
    'voxel_count', ...
    'voxel_i', ...
    'voxel_j', ...
    'voxel_k', ...
    'plot_x', ...
    'plot_y', ...
    'plot_z', ...
    'coordinate_space'});

coordinate_csv = fullfile(out_dir, ...
    'mouse_atlas_ROI_centroids.csv');

writetable(coordinate_table, coordinate_csv);

%% ------------------------------------------------------------------------
% Optionally center FC matrices across states
%% ------------------------------------------------------------------------

if options.center_across_states
    FC_stack = cat(3, state_fc{:});
    across_state_mean = mean(FC_stack, 3);

    for k = 1:K
        state_fc{k} = state_fc{k} - across_state_mean;
        state_fc{k}(1:n_roi + 1:end) = 0;
    end
end

%% ------------------------------------------------------------------------
% Plot each state
%% ------------------------------------------------------------------------

thresholded_fc = cell(K, 1);
edge_tables = cell(K, 1);
node_strength = NaN(n_roi, K);

view_names = ["dorsal", "left", "right", "anterior", "posterior"];
view_angles = [ ...
      0,  90; ...
     90,   0; ...
    -90,   0; ...
      0,   0; ...
    180,   0];

for state = 1:K
    C = state_fc{state};

    [W, edge_table] = local_threshold_edges( ...
        C, ...
        atlas_values, ...
        roi_names, ...
        options.retain_fraction, ...
        options.edge_mode);

    thresholded_fc{state} = W;
    edge_tables{state} = edge_table;

    node_strength(:, state) = sum(abs(W), 2);

    edge_csv = fullfile(out_dir, sprintf( ...
        'state_%02d_mouse_atlas_edges.csv', state));

    writetable(edge_table, edge_csv);

    f = figure( ...
        'Visible', options.figure_visible, ...
        'Color', 'w', ...
        'Position', [100, 100, 1100, 900]);

    ax = axes(f);
    hold(ax, 'on');

    local_plot_edges(ax, W, centroids_plot);
    local_plot_nodes( ...
        ax, ...
        centroids_plot, ...
        node_strength(:, state), ...
        roi_names, ...
        options);

    axis(ax, 'equal');
    axis(ax, 'off');
    grid(ax, 'off');

    title(ax, sprintf( ...
        'Mouse HMM state %d: strongest %.1f%% %s FC edges', ...
        state, ...
        100 * options.retain_fraction, ...
        options.edge_mode), ...
        'Interpreter', 'none');

    for v = 1:size(view_angles, 1)
        view(ax, view_angles(v, 1), view_angles(v, 2));

        figure_path = fullfile(out_dir, sprintf( ...
            'state_%02d_mouse_FC_%s.png', ...
            state, view_names(v)));

        exportgraphics(f, figure_path, 'Resolution', 300);
    end

    close(f);

    if options.write_node_maps
        local_write_node_metric_nifti( ...
            atlas, ...
            atlas_info, ...
            atlas_values, ...
            node_strength(:, state), ...
            fullfile(out_dir, sprintf( ...
            'state_%02d_node_strength.nii', state)));
    end
end

%% ------------------------------------------------------------------------
% Save all state-level node metrics
%% ------------------------------------------------------------------------

node_table = table();

for state = 1:K
    tmp = table( ...
        (1:n_roi).', ...
        atlas_values, ...
        roi_names, ...
        repmat(state, n_roi, 1), ...
        node_strength(:, state), ...
        'VariableNames', { ...
        'roi_index', ...
        'atlas_value', ...
        'roi_name', ...
        'state', ...
        'absolute_node_strength'});

    node_table = [node_table; tmp]; %#ok<AGROW>
end

writetable(node_table, fullfile(out_dir, ...
    'mouse_HMM_state_node_strength.csv'));

%% ------------------------------------------------------------------------
% Return and save output
%% ------------------------------------------------------------------------

output = struct();
output.atlas_path = atlas_path;
output.roi_table_path = roi_table_path;
output.coordinate_table = coordinate_table;
output.centroids_voxel = centroids_voxel;
output.centroids_plot = centroids_plot;
output.coordinate_space = coordinate_space;
output.state_fc = state_fc;
output.thresholded_fc = thresholded_fc;
output.edge_tables = edge_tables;
output.node_strength = node_strength;
output.options = options;

save(fullfile(out_dir, 'mouse_HMM_atlas_graph_results.mat'), ...
    'output', '-v7.3');

fprintf('\nMouse-atlas HMM FC mapping completed.\n');
fprintf('States: %d\n', K);
fprintf('ROIs: %d\n', n_roi);
fprintf('Coordinate system: %s\n', coordinate_space);
fprintf('Results directory: %s\n', out_dir);
end


function options = local_set_defaults(options)

defaults = struct();
defaults.retain_fraction = 0.05;
defaults.edge_mode = 'signed';
defaults.center_across_states = false;
defaults.node_metric = 'strength';
defaults.show_labels = false;
defaults.use_world_coordinates = true;
defaults.write_node_maps = true;
defaults.figure_visible = 'off';

field_names = fieldnames(defaults);

for i = 1:numel(field_names)
    field_name = field_names{i};

    if ~isfield(options, field_name) || isempty(options.(field_name))
        options.(field_name) = defaults.(field_name);
    end
end

if ~isscalar(options.retain_fraction) || ...
        options.retain_fraction <= 0 || ...
        options.retain_fraction > 1
    error('options.retain_fraction must be in the interval (0,1].');
end

valid_edge_modes = {'signed', 'positive', 'negative'};

if ~ismember(lower(options.edge_mode), valid_edge_modes)
    error('options.edge_mode must be signed, positive, or negative.');
end

valid_node_metrics = {'strength', 'constant'};

if ~ismember(lower(options.node_metric), valid_node_metrics)
    error('options.node_metric must be strength or constant.');
end

if ~ismember(lower(options.figure_visible), {'on', 'off'})
    error('options.figure_visible must be on or off.');
end
end


function [W, edge_table] = local_threshold_edges( ...
    C, atlas_values, roi_names, retain_fraction, edge_mode)

n_roi = size(C, 1);
upper_mask = triu(true(n_roi), 1);

[row_idx, col_idx] = find(upper_mask);
edge_weights = C(upper_mask);

switch lower(edge_mode)
    case 'signed'
        eligible = isfinite(edge_weights) & edge_weights ~= 0;
        ranking_value = abs(edge_weights);

    case 'positive'
        eligible = isfinite(edge_weights) & edge_weights > 0;
        ranking_value = edge_weights;

    case 'negative'
        eligible = isfinite(edge_weights) & edge_weights < 0;
        ranking_value = abs(edge_weights);

    otherwise
        error('Unknown edge mode: %s', edge_mode);
end

eligible_idx = find(eligible);

if isempty(eligible_idx)
    warning('No eligible FC edges were found.');

    W = zeros(n_roi);

    edge_table = table( ...
        zeros(0, 1), zeros(0, 1), ...
        zeros(0, 1), zeros(0, 1), ...
        strings(0, 1), strings(0, 1), ...
        zeros(0, 1), zeros(0, 1), ...
        'VariableNames', { ...
        'roi_i', 'roi_j', ...
        'atlas_value_i', 'atlas_value_j', ...
        'roi_name_i', 'roi_name_j', ...
        'correlation', 'absolute_correlation'});
    return;
end

n_keep = max(1, round(retain_fraction * numel(eligible_idx)));

[~, sort_order] = sort( ...
    ranking_value(eligible_idx), 'descend');

selected = eligible_idx(sort_order(1:n_keep));

selected_rows = row_idx(selected);
selected_cols = col_idx(selected);
selected_weights = edge_weights(selected);

W = zeros(n_roi);

linear_upper = sub2ind( ...
    [n_roi, n_roi], selected_rows, selected_cols);

W(linear_upper) = selected_weights;
W = W + W.';

edge_table = table( ...
    selected_rows, ...
    selected_cols, ...
    atlas_values(selected_rows), ...
    atlas_values(selected_cols), ...
    roi_names(selected_rows), ...
    roi_names(selected_cols), ...
    selected_weights, ...
    abs(selected_weights), ...
    'VariableNames', { ...
    'roi_i', ...
    'roi_j', ...
    'atlas_value_i', ...
    'atlas_value_j', ...
    'roi_name_i', ...
    'roi_name_j', ...
    'correlation', ...
    'absolute_correlation'});

edge_table = sortrows(edge_table, ...
    'absolute_correlation', 'descend');
end


function local_plot_edges(ax, W, coordinates)

n_roi = size(W, 1);
upper_mask = triu(true(n_roi), 1);
[row_idx, col_idx] = find(upper_mask & W ~= 0);

if isempty(row_idx)
    return;
end

weights = W(sub2ind( ...
    [n_roi, n_roi], row_idx, col_idx));

max_weight = max(abs(weights));

if max_weight <= eps
    max_weight = 1;
end

for e = 1:numel(weights)
    p1 = coordinates(row_idx(e), :);
    p2 = coordinates(col_idx(e), :);

    normalized_weight = abs(weights(e)) / max_weight;
    line_width = 0.5 + 4.0 * normalized_weight;

    if weights(e) >= 0
        edge_color = [0.85, 0.15, 0.10];
    else
        edge_color = [0.10, 0.30, 0.85];
    end

    plot3(ax, ...
        [p1(1), p2(1)], ...
        [p1(2), p2(2)], ...
        [p1(3), p2(3)], ...
        '-', ...
        'Color', [edge_color, 0.65], ...
        'LineWidth', line_width);
end
end


function local_plot_nodes( ...
    ax, coordinates, strength, roi_names, options)

switch lower(options.node_metric)
    case 'strength'
        finite_strength = strength(isfinite(strength));

        if isempty(finite_strength) || max(finite_strength) <= eps
            node_size = repmat(35, size(strength));
        else
            normalized_strength = strength ./ max(finite_strength);
            node_size = 25 + 150 * normalized_strength;
        end

    case 'constant'
        node_size = repmat(50, size(strength));

    otherwise
        error('Unknown node metric.');
end

scatter3(ax, ...
    coordinates(:, 1), ...
    coordinates(:, 2), ...
    coordinates(:, 3), ...
    node_size, ...
    strength, ...
    'filled', ...
    'MarkerEdgeColor', [0.15, 0.15, 0.15], ...
    'LineWidth', 0.5);

colormap(ax, turbo);
colorbar(ax);

if options.show_labels
    for r = 1:size(coordinates, 1)
        text(ax, ...
            coordinates(r, 1), ...
            coordinates(r, 2), ...
            coordinates(r, 3), ...
            " " + roi_names(r), ...
            'FontSize', 7, ...
            'Interpreter', 'none');
    end
end
end


function local_write_node_metric_nifti( ...
    atlas, atlas_info, atlas_values, node_values, output_path)

metric_volume = NaN(size(atlas), 'single');
metric_volume(atlas == 0) = 0;

for r = 1:numel(atlas_values)
    metric_volume(atlas == atlas_values(r)) = ...
        single(node_values(r));
end

output_info = atlas_info;
output_info.Datatype = 'single';
output_info.BitsPerPixel = 32;

[output_dir, output_name, ~] = fileparts(output_path);

if isempty(output_dir)
    output_dir = pwd;
end

niftiwrite( ...
    metric_volume, ...
    fullfile(output_dir, output_name), ...
    output_info, ...
    'Compressed', true);
end