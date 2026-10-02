%% run4d_hmm_mar_plots_analysis.m
% Consolidate HMM-MAR results across K and model dimension into long-format
% CSV files (read by the Python group analysis), one set per model family:
%   HMM-PCA (options.lowrank; folders hmmmar_simple_k<K>_hmmpca<r>)
%       -> *_all_K_all_lowrank.csv, column lowrank_components
%       (the reported model, lowrank 15, K 3, is in this set)
%   PCA before the HMM (options.pca; folders hmmmar_simple_k<K>_pca<p>)
%       -> *_all_K_all_PCA.csv, column pca_components (as before)
% Folders that do not exist are skipped. Both sets carry a model_type column.
%
% Every temporal metric comes from the GROUP fit, so all of them describe the
% same states as the state FC and graph metrics (run5):
%   fractional occupancy, switching rate, max FO   group Gamma (HMM-MAR
%       getFractionalOccupancy, getSwitchingRate, getMaxFractionalOccupancy,
%       as stored by run4e in GrpData)
%   dwell times, episodes, time points per state    group Viterbi path
%       (GrpData.viterbipath) with HMM-MAR getStateLifeTimes
% Reads hmmmar_results.mat in each folder. Files written by the previous run4e
% also contain GrpData.viterbipath, so they can be read without refitting.
%
% Settings: hmm_config.m (K_values, lowrank_values, pca_values, TR, folders).
% Every row carries the fit ID of the model it came from (fit_id.txt /
% hmm_fit_id.m); for files from the previous run4e the ID is computed from
% their Gamma. Changes from the previous version are listed in CHANGES.md.

clc;
clear;

%% ========== Configuration (hmm_config.m) ==========
cfg = hmm_config();
K_values = cfg.K_values;
TR_sec = cfg.TR_sec;
base_data_dir = cfg.out_base;
out_dir = base_data_dir;
file_pattern = 'hmmmar_results.mat';
toolbox_dir = cfg.hmmmar_dir;

if ~exist(toolbox_dir, 'dir')
    error('HMM-MAR toolbox not found: %s', toolbox_dir);
end
addpath(genpath(toolbox_dir));

families = struct( ...
    'type', {'lowrank', 'pca'}, ...
    'values', {cfg.lowrank_values, cfg.pca_values}, ...
    'folder', {'hmmmar_simple_k%d_hmmpca%d', 'hmmmar_simple_k%d_pca%d'}, ...
    'suffix', {'all_K_all_lowrank', 'all_K_all_PCA'}, ...
    'column', {'lowrank_components', 'pca_components'});
n_written = 0;
for f = 1:numel(families)
    n_written = n_written + local_family(families(f), K_values, TR_sec, base_data_dir, ...
        out_dir, file_pattern);
end
if n_written == 0
    error('No valid results were found.');
end


%% ========== Local functions ==========
function n_models = local_family(fam, K_values, TR_sec, base_data_dir, out_dir, file_pattern)
% Load each model folder of one family once and write that family's tables.
model_tables = {};
state_tables = {};
switching_tables = {};
maxfo_tables = {};
dwell_tables = {};

for pca_current = fam.values
    for K_requested = K_values
        result_path = fullfile(base_data_dir, sprintf(fam.folder, K_requested, pca_current), ...
            file_pattern);
        if ~exist(result_path, 'file')
            fprintf('Not found (skipped): %s\n', result_path);
            continue;
        end
        fprintf('Loading %s, K=%d, %d components: %s\n', fam.type, K_requested, ...
            pca_current, result_path);
        result = load(result_path);

        % ----- required fields -----
        if ~isfield(result, 'groups') || ~isfield(result, 'GrpData') || ...
                ~isfield(result, 'hmm') || ...
                ~all(isfield(result.GrpData, {'FO', 'switchingRate', 'maxFO', 'viterbipath'}))
            warning('Missing groups, hmm or GrpData fields in %s; skipped.', result_path);
            continue;
        end

        % ----- fit ID: stored by run4e, or computed for previous files -----
        if isfield(result, 'fit_id')
            fit_id = string(result.fit_id);
        else
            fit_id = string(hmm_fit_id(double(result.Gamma)));
        end
        id_file = fullfile(fileparts(result_path), 'fit_id.txt');
        if exist(id_file, 'file') && ~strcmp(strtrim(fileread(id_file)), fit_id)
            error('%s does not come from the fit in %s.', result_path, id_file);
        end

        groups = string(result.groups(:));
        FO = double(result.GrpData.FO);
        switching_rate = double(result.GrpData.switchingRate(:));
        max_FO = double(result.GrpData.maxFO(:));
        vpath = double(result.GrpData.viterbipath(:));
        [n_subj, K_returned] = size(FO);

        assert(numel(groups) == n_subj, 'Group/FO mismatch for K=%d, PCA=%d.', K_requested, pca_current);
        assert(numel(switching_rate) == n_subj && numel(max_FO) == n_subj, ...
            'Switching-rate or MaxFO/FO mismatch for K=%d, PCA=%d.', K_requested, pca_current);
        if K_returned ~= K_requested
            warning('Requested K=%d with PCA=%d, but FO contains %d states.', ...
                K_requested, pca_current, K_returned);
        end

        % ----- time points per subject -----
        if isfield(result.GrpData, 'Tvec')
            Tvec = double(result.GrpData.Tvec(:));
        else
            assert(mod(numel(vpath), n_subj) == 0, ...
                'Viterbi path length is not a multiple of the subject count (K=%d, PCA=%d).', ...
                K_requested, pca_current);
            Tvec = repmat(numel(vpath) / n_subj, n_subj, 1);   % equal-length scans
        end
        assert(sum(Tvec) == numel(vpath), 'Tvec does not match the Viterbi path length.');

        % ----- subject IDs (one rule for every table) -----
        if isfield(result, 'metadata_aligned') && istable(result.metadata_aligned) && ...
                ismember('subject_id', result.metadata_aligned.Properties.VariableNames) && ...
                height(result.metadata_aligned) == n_subj
            subject_id = double(result.metadata_aligned.subject_id(:));
        elseif isfield(result, 'subject_ids') && numel(result.subject_ids) == n_subj
            subject_id = double(result.subject_ids(:));
        else
            subject_id = (1:n_subj).';
            warning('No aligned subject IDs for K=%d, PCA=%d; using row numbers.', ...
                K_requested, pca_current);
        end

        % ----- dwell from the group Viterbi path (HMM-MAR getStateLifeTimes) -----
        lifetimes = getStateLifeTimes(vpath, Tvec, result.hmm.train, 0, [], false);
        mean_dwell_TR = NaN(n_subj, K_returned);
        median_dwell_TR = NaN(n_subj, K_returned);
        n_episodes = zeros(n_subj, K_returned);
        map_TR_count = zeros(n_subj, K_returned);
        offs = [0; cumsum(Tvec)];
        for s = 1:n_subj
            labels = vpath(offs(s) + 1:offs(s + 1));
            for k = 1:K_returned
                map_TR_count(s, k) = sum(labels == k);
                n_episodes(s, k) = numel(lifetimes{s, k});
                if ~isempty(lifetimes{s, k})
                    mean_dwell_TR(s, k) = mean(lifetimes{s, k});
                    median_dwell_TR(s, k) = median(lifetimes{s, k});
                end
            end
        end

        Kc = repmat(K_requested, n_subj, 1);
        Fid = repmat(fit_id, n_subj, 1);
        Kr = repmat(K_returned, n_subj, 1);
        Pc = repmat(pca_current, n_subj, 1);

        model_tables{end + 1} = table(subject_id, groups, Kc, Pc, switching_rate, max_FO, Fid, ...
            'VariableNames', {'subject_id', 'group', 'K', 'pca_components', ...
            'switching_rate', 'max_FO', 'fit_id'}); %#ok<SAGROW>

        n_rows = n_subj * K_returned;
        state_tables{end + 1} = table(repmat(subject_id, K_returned, 1), ...
            repmat(groups, K_returned, 1), repmat(K_requested, n_rows, 1), ...
            repmat(pca_current, n_rows, 1), repelem((1:K_returned).', n_subj), FO(:), ...
            repmat(fit_id, n_rows, 1), ...
            'VariableNames', {'subject_id', 'group', 'K', 'pca_components', 'state', ...
            'fractional_occupancy', 'fit_id'}); %#ok<SAGROW>

        switching_tables{end + 1} = table(subject_id, groups, Kc, Kr, Pc, switching_rate, Fid, ...
            'VariableNames', {'subject_id', 'group', 'K_requested', 'K_returned', ...
            'pca_components', 'switching_rate', 'fit_id'}); %#ok<SAGROW>

        maxfo_tables{end + 1} = table(subject_id, groups, Kc, Kr, Pc, max_FO, Fid, ...
            'VariableNames', {'subject_id', 'group', 'K_requested', 'K_returned', ...
            'pca_components', 'max_FO', 'fit_id'}); %#ok<SAGROW>

        dwell_tables{end + 1} = table( ...
            repelem(subject_id, K_returned), repelem(groups, K_returned), ...
            repmat(K_requested, n_rows, 1), repmat(K_returned, n_rows, 1), ...
            repmat(pca_current, n_rows, 1), repmat((1:K_returned).', n_subj, 1), ...
            reshape(FO.', [], 1), reshape(map_TR_count.', [], 1), ...
            reshape(n_episodes.', [], 1), reshape(mean_dwell_TR.', [], 1), ...
            reshape(median_dwell_TR.', [], 1), reshape((mean_dwell_TR * TR_sec).', [], 1), ...
            reshape((median_dwell_TR * TR_sec).', [], 1), ...
            repmat("group_viterbi", n_rows, 1), repmat(fit_id, n_rows, 1), ...
            'VariableNames', {'subject_id', 'group', 'K_requested', 'K_returned', ...
            'pca_components', 'state', 'fractional_occupancy', 'map_TR_count', ...
            'n_dwell_episodes', 'mean_dwell_TR', 'median_dwell_TR', ...
            'mean_dwell_seconds', 'median_dwell_seconds', 'state_path', 'fit_id'}); %#ok<SAGROW>
    end
end

n_models = numel(model_tables);
if n_models == 0
    return;
end

%% ----- concatenate, label the family, sort and write -----
hmm_subject_model = local_label(vertcat(model_tables{:}), fam);
hmm_subject_state = local_label(vertcat(state_tables{:}), fam);
switching_tbl = local_label(vertcat(switching_tables{:}), fam);
maxfo_tbl = local_label(vertcat(maxfo_tables{:}), fam);
dwell_tbl = local_label(vertcat(dwell_tables{:}), fam);
c = fam.column;
hmm_subject_model = sortrows(hmm_subject_model, {c, 'K', 'subject_id'});
hmm_subject_state = sortrows(hmm_subject_state, {c, 'K', 'subject_id', 'state'});
switching_tbl = sortrows(switching_tbl, {c, 'K_requested', 'subject_id'});
maxfo_tbl = sortrows(maxfo_tbl, {c, 'K_requested', 'subject_id'});
dwell_tbl = sortrows(dwell_tbl, {c, 'K_requested', 'subject_id', 'state'});

sfx = fam.suffix;
writetable(hmm_subject_model, fullfile(out_dir, ['hmm_subject_model_metrics_' sfx '.csv']));
writetable(hmm_subject_state, fullfile(out_dir, ['hmm_subject_state_FO_' sfx '.csv']));
writetable(switching_tbl, fullfile(out_dir, ['hmm_switching_rates_' sfx '.csv']));
writetable(maxfo_tbl, fullfile(out_dir, ['hmm_maxFO_' sfx '.csv']));
writetable(dwell_tbl, fullfile(out_dir, ['hmm_dwell_times_' sfx '.csv']));

component_values = fam.values; %#ok<NASGU>
model_type = fam.type; %#ok<NASGU>
save(fullfile(out_dir, ['hmm_metrics_' sfx '.mat']), 'hmm_subject_model', ...
    'hmm_subject_state', 'K_values', 'component_values', 'model_type', '-v7.3');
save(fullfile(out_dir, ['hmm_temporal_metrics_' sfx '.mat']), 'switching_tbl', ...
    'maxfo_tbl', 'dwell_tbl', 'K_values', 'component_values', 'model_type', 'TR_sec', '-v7.3');

fprintf('\n%s: wrote %d models, %d subject-model rows, %d subject-state rows, to %s\n', ...
    fam.type, n_models, height(hmm_subject_model), height(hmm_subject_state), out_dir);
end

function T = local_label(T, fam)
% Name the component column after the family and add model_type.
T.Properties.VariableNames{strcmp(T.Properties.VariableNames, 'pca_components')} = fam.column;
T.model_type = repmat(string(fam.type), height(T), 1);
end
