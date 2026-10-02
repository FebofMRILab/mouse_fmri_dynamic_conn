%% ==============================================================
%% run5_analyze_raw_hmm_graph_metrics.m
%% ==============================================================
% Post-hoc graph-theoretic analysis of raw-timeseries HMM output.
% Computes per-subject, per-state metrics for 7 groups and exports long-format CSVs
% (group statistics are done in Python).
%
% Published code called:
%   HMM-MAR (Vidaurre et al.; github.com/OHBA-analysis/HMM-MAR)
%     standardisedata, and the state statistic of setxx.m, via
%     hmm_subject_state_fc.m (per-subject state FC; see that file)
%   Brain Connectivity Toolbox (Rubinov & Sporns 2010; release 2019-03-03)
%     threshold_proportional, weight_conversion, get_components,
%     strengths_und, clustering_coef_bu/wu, distance_bin/wei, charpath,
%     efficiency_bin/wei, assortativity_bin/wei, community_louvain,
%     transitivity_bu
% Changes from the previous version are listed in CHANGES.md.
%
% Groups in meta_data.csv:
%   NTG_Y, NTG_M, NTG_A (Non-Transgenic: Young, Middle, Aged)
%   5XFAD_Y, 5XFAD_M (5XFAD mice)
%   CRND8_Y, CRND8_M (TgCRND8 mice)

clc;
clear;
close all;

%% ---------- Configuration (hmm_config.m) ----------
cfg = hmm_config();
model_dir = cfg.model_dir;
hmm_results_path = fullfile(model_dir, 'raw_hmm_results.mat');
data_dir = cfg.data_dir;
file_pattern = cfg.file_pattern;
out_dir = fullfile(model_dir, 'hmm_raw_posthoc_graph_metrics');

% HMM-MAR (standardisedata, getFractionalOccupancy) and, if given, BCT
if ~exist(cfg.hmmmar_dir, 'dir')
    error('HMM-MAR toolbox not found: %s', cfg.hmmmar_dir);
end
addpath(genpath(cfg.hmmmar_dir));
if ~isempty(cfg.bct_dir), addpath(cfg.bct_dir); end
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

density = cfg.graph_density;              % 0.15 by default
min_effective_n = cfg.min_effective_n;    % 83 by default (82 ROIs + 1), same as run5B

%% ---------- Check required BCT functions at startup ----------
required_bct = {
    'threshold_proportional', 'weight_conversion', 'get_components', 'strengths_und', ...
    'clustering_coef_bu', 'clustering_coef_wu', ...
    'distance_bin', 'distance_wei', 'charpath', ...
    'efficiency_bin', 'efficiency_wei', ...
    'assortativity_bin', 'assortativity_wei', ...
    'community_louvain', 'transitivity_bu'
};

missing_bct = required_bct( ...
    cellfun(@(f) exist(f, 'file') ~= 2, required_bct));

if ~isempty(missing_bct)
    error('Missing required BCT functions: %s', strjoin(missing_bct, ', '));
end
fprintf('All required BCT functions found.\n');

%% ---------- Load HMM results ----------
fprintf('Loading raw HMM results...\n');
load(hmm_results_path, 'R');

K = R.K;

% Fit ID: the fit this script reads (hmm_fit_id.m). Fits made by the previous
% run4e have no stored ID; it is computed here and reported.
fit_id = hmm_fit_id(R.Gamma);
if isfield(R, 'fit_id')
    assert(strcmp(R.fit_id, fit_id), 'raw_hmm_results.mat: Gamma does not match its stored fit ID.');
else
    warning('raw_hmm_results.mat has no stored fit ID (previous run4e). Computed: %s', fit_id);
end
fprintf('Fit ID: %s\n', fit_id);
S = numel(R.subject_ids);
n_roi = R.n_roi;
subject_ids = R.subject_ids;
groups = R.groups;

fprintf('Loaded: %d subjects, %d states, %d ROIs\n', S, K, n_roi);
fprintf('Groups present: %s\n', strjoin(string(categories(groups)), ', '));

% Validate metadata alignment
assert(numel(unique(subject_ids)) == S, ...
    'R.subject_ids contains duplicate IDs.');
assert(numel(groups) == S, ...
    'R.groups is not aligned with R.subject_ids.');
assert(~any(ismissing(groups) | string(groups) == "NA"), ...
    'Missing or NA group assignments found in R.groups.');

%% ---------- Load raw time series ----------
fprintf('Loading raw time series...\n');
Ts_list = cell(S, 1);

for i = 1:S
    subj_id = subject_ids(i);
    filename = fullfile(data_dir, sprintf(file_pattern, subj_id));
    
    if ~exist(filename, 'file')
        error('Missing file: %s', filename);
    end
    
    Ts_list{i} = readmatrix(filename);
end

% Validate alignment dimensions
for s = 1:S
    assert(size(Ts_list{s}, 1) == size(R.Gamma{s}, 1), ...
        'Time/Gamma mismatch for subject ID %d.', subject_ids(s));
    assert(size(Ts_list{s}, 2) == n_roi, ...
        'ROI mismatch for subject ID %d: expected %d, got %d.', ...
        subject_ids(s), n_roi, size(Ts_list{s}, 2));
    assert(size(R.Gamma{s}, 2) == K, ...
        'Gamma-state mismatch for subject ID %d.', subject_ids(s));
end
fprintf('All dimensions validated.\n');

%% ---------- Subject-specific state FC (HMM-MAR's state statistic, per subject) ----------
fprintf('Computing per-subject, per-state FC matrices with effective N filter (min_eff_n=%d)...\n', min_effective_n);
[subj_state_fc, effective_n] = hmm_subject_state_fc(Ts_list, R.Gamma, R.hmm.train, min_effective_n);

% Fractional occupancy (HMM-MAR's getFractionalOccupancy) and time points
% labelled with each state. R.vpath is the Viterbi path in fits made with the
% current hmm_fit_raw_timeseries.m, and the most probable state per time point
% in fits made with the previous version (same field name).
FO_subj = NaN(S, K);
map_count = zeros(S, K);
for s = 1:S
    FO_subj(s, :) = getFractionalOccupancy(R.Gamma{s}, size(R.Gamma{s}, 1), R.hmm.train, 2);
    if isfield(R, 'vpath') && ~isempty(R.vpath)
        labels = double(R.vpath{s});
    else
        [~, labels] = max(R.Gamma{s}, [], 2);
    end
    map_count(s, :) = accumarray(labels(:), 1, [K, 1]).';
end

%% ---------- Compute graph metrics per subject per state ----------
fprintf('Computing graph metrics (density=%.2f)...\n', density);

n_rows = S * K;
subj_id_col = nan(n_rows, 1);
state_col = nan(n_rows, 1);
group_col = strings(n_rows, 1);
FO_col = nan(n_rows, 1);
effective_n_col = nan(n_rows, 1);
map_TR_count_col = nan(n_rows, 1);

% Graph metrics columns
strength_mean = nan(n_rows, 1);
C_bin = nan(n_rows, 1);
C_wei = nan(n_rows, 1);
L_bin = nan(n_rows, 1);
L_wei = nan(n_rows, 1);
Eglob_bin = nan(n_rows, 1);
Eglob_wei = nan(n_rows, 1);
assort_bin = nan(n_rows, 1);
assort_wei = nan(n_rows, 1);
Q_mod = nan(n_rows, 1);
Q_mod_mean = nan(n_rows, 1);
Q_mod_sd = nan(n_rows, 1);
transitivity_bin = nan(n_rows, 1);
density_actual = nan(n_rows, 1);
n_components = nan(n_rows, 1);
largest_component_fraction = nan(n_rows, 1);
is_connected = nan(n_rows, 1);

row = 0;
for s = 1:S
    for k = 1:K
        row = row + 1;
        
        subj_id_col(row) = subject_ids(s);
        state_col(row) = k;
        group_col(row) = string(groups(s));
        
        FO_col(row) = FO_subj(s, k);
        effective_n_col(row) = effective_n(s, k);
        map_TR_count_col(row) = map_count(s, k);
        
        A = subj_state_fc{s, k};
        
        %if all(isnan(A(:)))
        %continue;
        %end
        
		if isempty(A) || any(~isfinite(A(:)))
			continue;
		end
		
        graph_seed = 1000000 + double(subject_ids(s)) * 100 + k;
		m = compute_graph_metrics_single_robust(A, density, graph_seed);
        
        strength_mean(row) = m.strength_mean;
        C_bin(row) = m.C_bin;
        C_wei(row) = m.C_wei;
        L_bin(row) = m.L_bin;
        L_wei(row) = m.L_wei;
        Eglob_bin(row) = m.Eglob_bin;
        Eglob_wei(row) = m.Eglob_wei;
        assort_bin(row) = m.assort_bin;
        assort_wei(row) = m.assort_wei;
        Q_mod(row) = m.Q_mod;
        Q_mod_mean(row) = m.Q_mod_mean;
        Q_mod_sd(row) = m.Q_mod_sd;
        transitivity_bin(row) = m.transitivity_bin;
        density_actual(row) = m.density_actual;
        n_components(row) = m.n_components;
        largest_component_fraction(row) = m.largest_component_fraction;
        is_connected(row) = m.is_connected;
    end
    
    if mod(s, 10) == 0
        fprintf('  Processed %d/%d subjects\n', s, S);
    end
end

%% ---------- Create long-format CSV ----------
fprintf('Creating output table...\n');

metrics_tbl = table(subj_id_col, state_col, group_col, ...
    FO_col, effective_n_col, map_TR_count_col, ...
    strength_mean, C_bin, C_wei, L_bin, L_wei, ...
    Eglob_bin, Eglob_wei, assort_bin, assort_wei, ...
    Q_mod, Q_mod_mean, Q_mod_sd, transitivity_bin, ...
    density_actual, n_components, largest_component_fraction, is_connected, ...
    'VariableNames', {'subject_id', 'state', 'group', ...
    'FO', 'effective_n', 'map_TR_count', ...
    'strength_mean', 'C_bin', 'C_wei', 'L_bin', 'L_wei', ...
    'Eglob_bin', 'Eglob_wei', 'assort_bin', 'assort_wei', ...
    'Q_mod', 'Q_mod_mean', 'Q_mod_sd', 'transitivity_bin', ...
    'density_actual', 'n_components', 'largest_component_fraction', 'is_connected'});
metrics_tbl.fit_id = repmat(string(fit_id), height(metrics_tbl), 1);
metrics_tbl.min_effective_n = repmat(min_effective_n, height(metrics_tbl), 1);

csv_path = fullfile(out_dir, sprintf('graph_metrics_per_subject_per_state_density%02d.csv', round(density * 100)));
writetable(metrics_tbl, csv_path);
fprintf('Saved: %s\n', csv_path);

save(fullfile(out_dir, 'graph_metrics_per_subject_per_state.mat'), ...
    'metrics_tbl', 'density', 'min_effective_n', 'fit_id', 'cfg', 'R', '-v7.3');

%% ---------- Summary statistics by group ----------
fprintf('\n=== Summary Statistics by Group ===\n');

valid_rows = isfinite(strength_mean) | isfinite(C_bin) | isfinite(Eglob_bin);
plot_tbl = metrics_tbl(valid_rows, :);

unique_groups = categories(categorical(plot_tbl.group));
unique_states = unique(plot_tbl.state);

fprintf('Total valid observations: %d (out of %d possible)\n', sum(valid_rows), n_rows);
fprintf('Missing/NaN observations (due to eff_N < %d): %d\n', min_effective_n, n_rows - sum(valid_rows));
fprintf('Groups: %s\n', strjoin(unique_groups, ', '));
fprintf('States: %s\n', mat2str(unique_states));

% Print sample sizes per group
fprintf('\nSample sizes per group:\n');
for g = 1:numel(unique_groups)
    grp = unique_groups{g};
    n_subj = sum(groups == grp);
    n_obs = sum(plot_tbl.group == grp);
    fprintf('  %s: %d subjects, %d valid observations\n', grp, n_subj, n_obs);
end

%% ---------- Preliminary group comparison plots ----------
if numel(unique_groups) > 1
    fprintf('\nGenerating preliminary group comparison plots...\n');
    
    % Plot 1: Global efficiency by state
    f1 = figure('Position', [100 100 1400 800], 'Visible', 'off');
    
    for k = 1:numel(unique_states)
        subplot(2, ceil(numel(unique_states) / 2), k);
        state_data = plot_tbl(plot_tbl.state == unique_states(k), :);
        [sorted_groups, sort_idx] = sort(categories(categorical(state_data.group)));
        
        boxplot(state_data.Eglob_bin, state_data.group, ...
            'GroupOrder', sorted_groups, 'Colors', 'k');
        ylabel('Global Efficiency (binary)', 'FontSize', 10);
        title(sprintf('State %d', unique_states(k)), 'FontSize', 12);
        xtickangle(45);
        grid on;
        
        hold on;
        for gg = 1:numel(sorted_groups)
            n = sum(state_data.group == sorted_groups{gg});
            text(gg, min(state_data.Eglob_bin) * 0.95, sprintf('n=%d', n), ...
                'HorizontalAlignment', 'center', 'FontSize', 8);
        end
        hold off;
    end
    sgtitle('Global Efficiency by State and Group (7 groups)', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f1, fullfile(out_dir, 'preliminary_Eglob_by_state_group.png'));
    close(f1);
    
    % Plot 2: Transitivity by state and group
    f2 = figure('Position', [100 100 1400 800], 'Visible', 'off');
    
    for k = 1:numel(unique_states)
        subplot(2, ceil(numel(unique_states) / 2), k);
        state_data = plot_tbl(plot_tbl.state == unique_states(k), :);
        [sorted_groups, ~] = sort(categories(categorical(state_data.group)));
        
        boxplot(state_data.transitivity_bin, state_data.group, ...
            'GroupOrder', sorted_groups, 'Colors', 'k');
        ylabel('Transitivity (binary)', 'FontSize', 10);
        title(sprintf('State %d', unique_states(k)), 'FontSize', 12);
        xtickangle(45);
        grid on;
    end
    sgtitle('Transitivity by State and Group', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f2, fullfile(out_dir, 'preliminary_transitivity_by_state_group.png'));
    close(f2);
end

fprintf('\n=== Post-hoc analysis complete! ===\n');
fprintf('CSV: %s\n', csv_path);


%% ==============================================================
%% Helper function
%% ==============================================================
function metrics = compute_graph_metrics_single_robust(A, density, seed)
% Positive-edge graph analysis with Brain Connectivity Toolbox functions.
%
% IMPORTANT: Negative correlations are intentionally excluded (set to 0 before
% thresholding). If negative edges are scientifically meaningful, replace this
% with a signed-network analysis rather than interpreting these as
% whole-correlation graphs.

if nargin < 3 || isempty(seed)
    seed = 1;
end
if ~isscalar(density) || density <= 0 || density > 1
    error('density must be in (0,1].');
end
if ~ismatrix(A) || size(A, 1) ~= size(A, 2) || any(~isfinite(A(:)))
    error('A must be a finite square matrix.');
end

p = size(A, 1);
A = (A + A.') ./ 2;
A(1:p + 1:end) = 0;
A = max(A, 0);                                  % positive edges only (adaptation)
if ~any(A(:) > 0)
    error('The matrix has no positive off-diagonal edges.');
end

W = threshold_proportional(A, density);         % BCT
B = weight_conversion(W, 'binarize');           % BCT

metrics = struct();
metrics.density_target = density;
metrics.density_actual = nnz(triu(B, 1)) / (p * (p - 1) / 2);

[~, component_sizes] = get_components(B);       % BCT
metrics.n_components = numel(component_sizes);
metrics.largest_component_fraction = max(component_sizes) / p;
metrics.is_connected = double(metrics.n_components == 1);

metrics.strength_mean = mean(strengths_und(W));
metrics.C_bin = mean(clustering_coef_bu(B), 'omitnan');
metrics.C_wei = mean(clustering_coef_wu(W), 'omitnan');

% Characteristic path length is reported only for connected graphs (it is
% infinite otherwise; Rubinov & Sporns 2010 recommend global efficiency for
% disconnected networks). Global efficiency is reported for all graphs.
if metrics.is_connected
    metrics.L_bin = charpath(distance_bin(B), 0, 1);
    metrics.L_wei = charpath(distance_wei(weight_conversion(W, 'lengths')), 0, 1);
else
    metrics.L_bin = NaN;
    metrics.L_wei = NaN;
end

metrics.Eglob_bin = efficiency_bin(B);
metrics.Eglob_wei = efficiency_wei(W);
metrics.assort_bin = assortativity_bin(B, 0);
metrics.assort_wei = assortativity_wei(W, 0);

% Deterministic stream for reproducible Louvain optimisation (50 runs).
old_stream = RandStream.getGlobalStream;
cleanup = onCleanup(@() RandStream.setGlobalStream(old_stream));
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', seed));
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
