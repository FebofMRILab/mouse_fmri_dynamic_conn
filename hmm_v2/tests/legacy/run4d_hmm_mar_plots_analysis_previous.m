%% run4C_hmm_mar_plots_analysis.m
% Consolidate HMM-MAR results across K and PCA configurations into
% R-friendly long-format CSV files.

clc;
clear;

%% ========== Configuration ==========
K_values = 3:10;
pca_values = [0, 15, 25, 50];
TR_sec = 2;
file_pattern = 'hmmmar_results.mat';
base_data_dir = '../signals';
out_dir = '../signals';

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%% ========== Add HMM-MAR ==========
toolbox_dir = '/blue/febo/MATLAB/HMM-MAR-master';

if ~exist(toolbox_dir, 'dir')
    error('HMM-MAR toolbox not found: %s', toolbox_dir);
end

addpath(genpath(toolbox_dir));

%% ========== Load and consolidate results ==========
% Subject-model table:
% one row per subject, K, and PCA configuration
model_tables = cell(numel(pca_values), numel(K_values));

% Subject-state table:
% one row per subject, K, PCA configuration, and state
state_tables = cell(numel(pca_values), numel(K_values));

for pca_idx = 1:numel(pca_values)
    pca_current = pca_values(pca_idx);

    for k_idx = 1:numel(K_values)
        K_current = K_values(k_idx);

        data_dir = fullfile(base_data_dir, sprintf( ...
            'hmmmar_simple_k%d_pca%d', K_current, pca_current));

        result_path = fullfile(data_dir, file_pattern);

        if ~exist(result_path, 'file')
            warning('Missing result file: %s', result_path);
            continue;
        end

        fprintf('Loading K=%d, PCA=%d: %s\n', ...
            K_current, pca_current, result_path);

        result = load(result_path);

        %% ----- Validate required fields -----
        if ~isfield(result, 'groups')
            warning('No groups field found in: %s', result_path);
            continue;
        end

        if ~isfield(result, 'GrpData')
            warning('No GrpData field found in: %s', result_path);
            continue;
        end

        required_fields = {'FO', 'switchingRate', 'maxFO'};

        if ~all(isfield(result.GrpData, required_fields))
            missing_fields = required_fields( ...
                ~isfield(result.GrpData, required_fields));

            warning('Missing GrpData field(s) in %s: %s', ...
                result_path, strjoin(missing_fields, ', '));
            continue;
        end

        %% ----- Extract subject-level data -----
        groups = string(result.groups(:));
        FO = double(result.GrpData.FO);
        switching_rate = double(result.GrpData.switchingRate(:));
        max_FO = double(result.GrpData.maxFO(:));

        n_subjects = size(FO, 1);
        n_returned_states = size(FO, 2);

        if numel(groups) ~= n_subjects
            error(['Group/FO dimension mismatch for K=%d, PCA=%d: ' ...
                '%d groups versus %d subjects.'], ...
                K_current, pca_current, numel(groups), n_subjects);
        end

        if numel(switching_rate) ~= n_subjects
            error(['Switching-rate/FO dimension mismatch for K=%d, ' ...
                'PCA=%d.'], K_current, pca_current);
        end

        if numel(max_FO) ~= n_subjects
            error('MaxFO/FO dimension mismatch for K=%d, PCA=%d.', ...
                K_current, pca_current);
        end

        % This detects state dropping or an unexpected output state count.
        if n_returned_states ~= K_current
            warning(['Requested K=%d with PCA=%d, but FO contains %d ' ...
                'states. Exporting the returned states.'], ...
                K_current, pca_current, n_returned_states);
        end

        if any(~isfinite(FO(:)))
            warning('FO contains nonfinite values for K=%d, PCA=%d.', ...
                K_current, pca_current);
        end

        %% ----- Recover subject IDs if available -----
        if isfield(result, 'subject_ids') && ...
                numel(result.subject_ids) == n_subjects

            subject_id = double(result.subject_ids(:));

        elseif isfield(result.GrpData, 'subject_ids') && ...
                numel(result.GrpData.subject_ids) == n_subjects

            subject_id = double(result.GrpData.subject_ids(:));

        else
            % This is safe only if all result files retain exactly the same
            % subject order. Replace with actual IDs when available.
            subject_id = (1:n_subjects).';

            warning(['No subject ID field found for K=%d, PCA=%d. ' ...
                'Using row numbers as subject IDs.'], ...
                K_current, pca_current);
        end

        %% ----- Subject-model table -----
        % One row per subject for each fitted model.
        model_tables{pca_idx, k_idx} = table( ...
            subject_id, ...
            groups, ...
            repmat(K_current, n_subjects, 1), ...
            repmat(pca_current, n_subjects, 1), ...
            switching_rate, ...
            max_FO, ...
            'VariableNames', { ...
            'subject_id', ...
            'group', ...
            'K', ...
            'pca_components', ...
            'switching_rate', ...
            'max_FO'});

        %% ----- Subject-state FO table -----
        % Convert the n_subjects-by-K FO matrix into long format.
        %
        % FO(:) stacks all subjects from state 1, followed by all subjects
        % from state 2, etc. The accompanying columns use the same order.
        n_state_rows = n_subjects * n_returned_states;

        state_tables{pca_idx, k_idx} = table( ...
            repmat(subject_id, n_returned_states, 1), ...
            repmat(groups, n_returned_states, 1), ...
            repmat(K_current, n_state_rows, 1), ...
            repmat(pca_current, n_state_rows, 1), ...
            repelem((1:n_returned_states).', n_subjects), ...
            FO(:), ...
            'VariableNames', { ...
            'subject_id', ...
            'group', ...
            'K', ...
            'pca_components', ...
            'state', ...
            'fractional_occupancy'});
    end
end

%% ========== Remove missing cells and concatenate ==========
valid_model = ~cellfun(@isempty, model_tables);
valid_state = ~cellfun(@isempty, state_tables);

if ~any(valid_model(:))
    error('No valid subject-model results were found.');
end

if ~any(valid_state(:))
    error('No valid subject-state results were found.');
end

hmm_subject_model = vertcat(model_tables{valid_model});
hmm_subject_state = vertcat(state_tables{valid_state});

%% ========== Sort output ==========
hmm_subject_model = sortrows(hmm_subject_model, ...
    {'pca_components', 'K', 'subject_id'});

hmm_subject_state = sortrows(hmm_subject_state, ...
    {'pca_components', 'K', 'subject_id', 'state'});

%% ========== Export CSV files ==========
model_csv_path = fullfile(out_dir, ...
    'hmm_subject_model_metrics_all_K_all_PCA.csv');

state_csv_path = fullfile(out_dir, ...
    'hmm_subject_state_FO_all_K_all_PCA.csv');

writetable(hmm_subject_model, model_csv_path);
writetable(hmm_subject_state, state_csv_path);

fprintf('\nSaved subject-model table:\n  %s\n', model_csv_path);
fprintf('Rows: %d\n', height(hmm_subject_model));

fprintf('\nSaved subject-state FO table:\n  %s\n', state_csv_path);
fprintf('Rows: %d\n', height(hmm_subject_state));

%% ========== Save consolidated MATLAB results ==========
mat_path = fullfile(out_dir, ...
    'hmm_metrics_all_K_all_PCA.mat');

save(mat_path, ...
    'hmm_subject_model', ...
    'hmm_subject_state', ...
    'K_values', ...
    'pca_values', ...
    '-v7.3');

fprintf('\nSaved consolidated MATLAB file:\n  %s\n', mat_path);

%% ========== Export switching rate, Max FO, and dwell time ==========
fprintf('Creating R-friendly HMM metric tables...\n');

pca_values = [0, 15, 25, 50];

switching_tables = cell(numel(pca_values), numel(K_values));
maxfo_tables = cell(numel(pca_values), numel(K_values));
dwell_tables = cell(numel(pca_values), numel(K_values));

for pca_idx = 1:numel(pca_values)
    pca_current = pca_values(pca_idx);

    for k_idx = 1:numel(K_values)
        K_requested = K_values(k_idx);

        result_path = fullfile(base_data_dir, sprintf( ...
            'hmmmar_simple_k%d_pca%d', ...
            K_requested, pca_current), file_pattern);

        if ~exist(result_path, 'file')
            warning('Missing result file: %s', result_path);
            continue;
        end

        result = load(result_path);

        if isempty(result)
            warning('No result found for K=%d, PCA=%d.', ...
                K_requested, pca_current);
            continue;
        end

        %% ----- Extract saved variables -----
        groups = string(result.groups(:));
        FO = double(result.GrpData.FO);
        switching_rate = double(result.GrpData.switchingRate(:));
        max_FO = double(result.GrpData.maxFO(:));

        n_subjects_result = size(FO, 1);
        K_returned = size(FO, 2);

        %% ----- Validate dimensions -----
        assert(numel(groups) == n_subjects_result, ...
            'Group/FO mismatch for K=%d, PCA=%d.', ...
            K_requested, pca_current);

        assert(numel(switching_rate) == n_subjects_result, ...
            'Switching-rate/FO mismatch for K=%d, PCA=%d.', ...
            K_requested, pca_current);

        assert(numel(max_FO) == n_subjects_result, ...
            'MaxFO/FO mismatch for K=%d, PCA=%d.', ...
            K_requested, pca_current);

        if K_returned ~= K_requested
            warning(['Requested K=%d with PCA=%d, but FO contains %d ' ...
                'states. This may indicate state dropping.'], ...
                K_requested, pca_current, K_returned);
        end

        %% ----- Recover subject IDs -----
        if isfield(result, 'metadata_aligned') && ...
                istable(result.metadata_aligned) && ...
                ismember('subject_id', ...
                result.metadata_aligned.Properties.VariableNames) && ...
                height(result.metadata_aligned) == n_subjects_result

            subject_id = double( ...
                result.metadata_aligned.subject_id(:));

        elseif isfield(result, 'subject_ids') && ...
                numel(result.subject_ids) == n_subjects_result

            subject_id = double(result.subject_ids(:));

        else
            subject_id = (1:n_subjects_result).';

            warning(['No aligned subject IDs found for K=%d, PCA=%d. ' ...
                'Using row numbers.'], K_requested, pca_current);
        end

        %% ----- Switching-rate table -----
        switching_tables{pca_idx, k_idx} = table( ...
            subject_id, ...
            groups, ...
            repmat(K_requested, n_subjects_result, 1), ...
            repmat(K_returned, n_subjects_result, 1), ...
            repmat(pca_current, n_subjects_result, 1), ...
            switching_rate, ...
            'VariableNames', { ...
            'subject_id', ...
            'group', ...
            'K_requested', ...
            'K_returned', ...
            'pca_components', ...
            'switching_rate'});

        %% ----- Maximum-FO table -----
        maxfo_tables{pca_idx, k_idx} = table( ...
            subject_id, ...
            groups, ...
            repmat(K_requested, n_subjects_result, 1), ...
            repmat(K_returned, n_subjects_result, 1), ...
            repmat(pca_current, n_subjects_result, 1), ...
            max_FO, ...
            'VariableNames', { ...
            'subject_id', ...
            'group', ...
            'K_requested', ...
            'K_returned', ...
            'pca_components', ...
            'max_FO'});

        %% ----- Calculate subject-state dwell measures -----
        mean_dwell_TR = NaN(n_subjects_result, K_returned);
        median_dwell_TR = NaN(n_subjects_result, K_returned);
        n_episodes = zeros(n_subjects_result, K_returned);
        map_TR_count = zeros(n_subjects_result, K_returned);

        if ~isfield(result, 'SubjData') || ...
                ~isfield(result.SubjData, 'vpath_subj')

            warning(['No SubjData.vpath_subj found for K=%d, PCA=%d. ' ...
                'Dwell measures cannot be calculated.'], ...
                K_requested, pca_current);
            continue;
        end

        if numel(result.SubjData.vpath_subj) ~= n_subjects_result
            error(['The number of subject Viterbi paths does not match ' ...
                'the FO rows for K=%d, PCA=%d.'], ...
                K_requested, pca_current);
        end

        for s = 1:n_subjects_result
            labels = local_get_state_labels( ...
                result.SubjData.vpath_subj{s}, K_returned);

            for state = 1:K_returned
                run_lengths = local_run_lengths(labels == state);

                map_TR_count(s, state) = sum(labels == state);
                n_episodes(s, state) = numel(run_lengths);

                if ~isempty(run_lengths)
                    mean_dwell_TR(s, state) = mean(run_lengths);
                    median_dwell_TR(s, state) = median(run_lengths);
                end
            end
        end

        %% ----- Long-format subject-state dwell table -----
        n_dwell_rows = n_subjects_result * K_returned;

        % These columns are ordered subject 1/state 1...K,
        % subject 2/state 1...K, etc.
        dwell_tables{pca_idx, k_idx} = table( ...
            repelem(subject_id, K_returned), ...
            repelem(groups, K_returned), ...
            repmat(K_requested, n_dwell_rows, 1), ...
            repmat(K_returned, n_dwell_rows, 1), ...
            repmat(pca_current, n_dwell_rows, 1), ...
            repmat((1:K_returned).', n_subjects_result, 1), ...
            reshape(FO.', [], 1), ...
            reshape(map_TR_count.', [], 1), ...
            reshape(n_episodes.', [], 1), ...
            reshape(mean_dwell_TR.', [], 1), ...
            reshape(median_dwell_TR.', [], 1), ...
            reshape((mean_dwell_TR * TR_sec).', [], 1), ...
            reshape((median_dwell_TR * TR_sec).', [], 1), ...
            'VariableNames', { ...
            'subject_id', ...
            'group', ...
            'K_requested', ...
            'K_returned', ...
            'pca_components', ...
            'state', ...
            'fractional_occupancy', ...
            'map_TR_count', ...
            'n_dwell_episodes', ...
            'mean_dwell_TR', ...
            'median_dwell_TR', ...
            'mean_dwell_seconds', ...
            'median_dwell_seconds'});
    end
end

%% ========== Concatenate valid tables ==========
valid_switching = ~cellfun(@isempty, switching_tables);
valid_maxfo = ~cellfun(@isempty, maxfo_tables);
valid_dwell = ~cellfun(@isempty, dwell_tables);

if ~any(valid_switching(:))
    error('No valid switching-rate results were found.');
end

if ~any(valid_maxfo(:))
    error('No valid Max-FO results were found.');
end

if ~any(valid_dwell(:))
    error('No valid Viterbi paths were found for dwell analysis.');
end

switching_tbl = vertcat(switching_tables{valid_switching});
maxfo_tbl = vertcat(maxfo_tables{valid_maxfo});
dwell_tbl = vertcat(dwell_tables{valid_dwell});

%% ========== Sort tables ==========
switching_tbl = sortrows(switching_tbl, ...
    {'pca_components', 'K_requested', 'subject_id'});

maxfo_tbl = sortrows(maxfo_tbl, ...
    {'pca_components', 'K_requested', 'subject_id'});

dwell_tbl = sortrows(dwell_tbl, ...
    {'pca_components', 'K_requested', 'subject_id', 'state'});

%% ========== Write CSV files ==========
switching_csv = fullfile(out_dir, ...
    'hmm_switching_rates_all_K_all_PCA.csv');

maxfo_csv = fullfile(out_dir, ...
    'hmm_maxFO_all_K_all_PCA.csv');

dwell_csv = fullfile(out_dir, ...
    'hmm_dwell_times_all_K_all_PCA.csv');

writetable(switching_tbl, switching_csv);
writetable(maxfo_tbl, maxfo_csv);
writetable(dwell_tbl, dwell_csv);

fprintf('\nSaved switching rates:\n  %s\n', switching_csv);
fprintf('Saved maximum FO:\n  %s\n', maxfo_csv);
fprintf('Saved dwell measures:\n  %s\n', dwell_csv);

%% ========== Save consolidated MATLAB tables ==========
save(fullfile(out_dir, 'hmm_temporal_metrics_all_K_all_PCA.mat'), ...
    'switching_tbl', 'maxfo_tbl', 'dwell_tbl', ...
    'K_values', 'pca_values', 'TR_sec', '-v7.3');


%% ========================================================================
% Local helper functions
%% ========================================================================

function labels = local_get_state_labels(vpath, K)
% Convert a Viterbi output to a column vector of integer state labels.

vpath = double(vpath);

if isvector(vpath)
    labels = vpath(:);

elseif size(vpath, 2) == K
    % Handles one-hot/probability-like T-by-K output.
    [~, labels] = max(vpath, [], 2);

elseif size(vpath, 1) == K
    % Handles transposed K-by-T output.
    [~, labels] = max(vpath, [], 1);
    labels = labels(:);

else
    error('Unrecognized Viterbi-path dimensions: %s.', ...
        mat2str(size(vpath)));
end

if any(~isfinite(labels)) || ...
        any(labels < 1) || any(labels > K) || ...
        any(labels ~= round(labels))

    error('Viterbi path contains invalid state labels.');
end
end


function run_lengths = local_run_lengths(in_state)
% Return the length of every contiguous state visit.

in_state = logical(in_state(:));

if ~any(in_state)
    run_lengths = zeros(0, 1);
    return;
end

change = diff([false; in_state; false]);
run_starts = find(change == 1);
run_ends = find(change == -1) - 1;

run_lengths = run_ends - run_starts + 1;
end

