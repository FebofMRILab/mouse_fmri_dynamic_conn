%% run4e_hmm_mar.m
% Fit the chosen HMM-MAR model ONCE and save everything the later steps read.
% Output folder (hmm_config.m): <HMM_OUT_BASE>/hmmmar_simple_k<K>_hmmpca<r>/
% for HMM-PCA (options.lowrank = r; the reported model, r = 15), or
% .../hmmmar_simple_k<K>_pca<p>/ for PCA before the HMM.
%   raw_hmm_results.mat   R: the fit and all summaries (read by run5, run5B)
%   hmmmar_results.mat    the same fit under the variable names run4d reads
%   raw_hmm_state_fc.mat  group state FC (state_fc) and the model's own
%                         estimate (state_fc_model)
%   occupancy_by_quarter.csv  per mouse x state x scan quarter (for the
%                         pre-specified drift test: hmm_occupancy_drift.py)
%   fit_id.txt, hmm_config_used.mat, metadata_aligned.csv, state_##_raw_HMM_FC.png
% Every file carries the fit ID (hmm_fit_id.m); later steps check it.
%
% The fit uses the options and number of initialisations of the
% pre-specified confirmatory full fits (run4c_confirm_lowrank.m), including
% options.useParallel = 0. HMM-MAR reseeds from the clock at each
% initialisation (initGamma_random.m), so every run gives a new fit with a new
% fit ID; tests/check_against_confirmatory.m compares it with the confirmatory
% full fit of the same model.
%
% Published code called (HMM-MAR, github.com/OHBA-analysis/HMM-MAR):
%   hmmmar (through hmm_fit_raw_timeseries.m), getFractionalOccupancy,
%   getMaxFractionalOccupancy, getSwitchingRate, getStateLifeTimes,
%   getTransProbs, getFuncConn, explainedvar_PCA, corrStaticFC, hmmdual
%
% Settings: hmm_config.m (shared by run4e, run4d, run5, run5B).
% Changes from the previous version are listed in CHANGES.md.

clc;
clear;
close all;

%% ========== Configuration (hmm_config.m) ==========
cfg = hmm_config();
K = cfg.K;
TR_sec = cfg.TR_sec;
n_subjects = cfg.n_subjects;
final_seed = cfg.seed;
do_dual = cfg.dual;
data_dir = cfg.data_dir;
toolbox_dir = cfg.hmmmar_dir;
file_pattern = cfg.file_pattern;
out_dir = cfg.model_dir;
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% HMM specification: base_options of run4c_confirm_lowrank.m, with
% initrep = initrep_final (15)
options = struct();
options.K = K;
options.standardise = 1;
options.zeromean = 1;
options.covtype = 'full';
options.order = 0;
options.inittype = 'hmmmar';
options.initrep = cfg.initrep;
options.initcyc = 25;
options.cyc = 1000;
options.verbose = 1;              % printing only
options.useParallel = 0;          % as run4c; HMM-MAR's default (1) opens a parallel
                                  % pool for the initialisations (CHANGES.md, E7)
options.Fs = 1 / TR_sec;
options.DirichletDiag = 10;
if cfg.lowrank > 0
    options.lowrank = cfg.lowrank;    % HMM-PCA (reported model: 15)
end
if cfg.pca > 0
    options.pca = cfg.pca;            % PCA before the HMM
end

%% ========== Add HMM-MAR ==========
if ~exist(toolbox_dir, 'dir')
    error('HMM-MAR toolbox not found: %s', toolbox_dir);
end
addpath(genpath(toolbox_dir));

%% ========== Load metadata ==========
meta_path = fullfile(data_dir, 'meta_data.csv');
groups = categorical(repmat("All", n_subjects, 1));

if exist(meta_path, 'file')
    meta = readtable(meta_path, 'TextType', 'string');
    if all(ismember({'subject_id', 'group'}, meta.Properties.VariableNames))
        assert(numel(unique(meta.subject_id)) == height(meta), ...
            'meta_data.csv contains duplicate subject_id values.');
        [found, loc] = ismember((1:n_subjects).', double(meta.subject_id));
        groups = categorical(repmat("NA", n_subjects, 1));
        groups(found) = categorical(meta.group(loc(found)));
    else
        warning('Metadata file lacks subject_id and/or group; using a single group.');
    end
else
    warning('No metadata file found; using a single group.');
end

%% ========== Load raw ROI time series ==========
Ts_list = cell(n_subjects, 1);
valid_subject = false(n_subjects, 1);
for s = 1:n_subjects
    filename = fullfile(data_dir, sprintf(file_pattern, s));
    if ~exist(filename, 'file')
        warning('Missing subject file: %s', filename);
        continue;
    end
    try
        Ts_list{s} = double(readmatrix(filename));
        valid_subject(s) = true;
    catch ME
        warning('Could not read subject %d: %s', s, ME.message);
    end
end

Ts_list = Ts_list(valid_subject);
subject_ids = find(valid_subject);
groups = groups(valid_subject);

missing_group = ismissing(groups) | string(groups) == "NA" | ...
    strlength(strtrim(string(groups))) == 0;
if any(missing_group)
    error('Missing group metadata for subject ID(s): %s', ...
        mat2str(subject_ids(missing_group).'));
end
groups = removecats(groups);

%% ========== Fit once (hmm_fit_raw_timeseries checks the data) ==========
rng(final_seed, 'twister');      % as run4c; HMM-MAR reseeds inside (see header)
R = hmm_fit_raw_timeseries(Ts_list, options, subject_ids);

% keep metadata aligned with the subjects the fit kept
groups = groups(R.valid_subject);
Ts_list = Ts_list(R.valid_subject);
metadata_aligned = table(R.subject_ids(:), groups(:), ...
    'VariableNames', {'subject_id', 'group'});
writetable(metadata_aligned, fullfile(out_dir, 'metadata_aligned.csv'));

R.TR_sec = TR_sec;
R.dwell_sec = R.dwell_TR * TR_sec;
R.groups = groups;
R.metadata_aligned = metadata_aligned;
R.seed = final_seed;
R.config = cfg;
R.fit_id = hmm_fit_id(R.Gamma);
fprintf('Fit ID: %s\n', R.fit_id);

%% ========== Diagnostics (after the fit, so they cannot change it) ==========
Xcat = vertcat(Ts_list{:});
[R.pca_explained_group, R.pca_explained_subj] = explainedvar_PCA(Xcat, R.Tvec, options);
CorrFC = struct();
[CorrFC.C, CorrFC.A, CorrFC.e, CorrFC.Av] = corrStaticFC(Xcat, R.Tvec, options, 1);
R.static_fc_similarity = CorrFC;   % subject x subject static FC similarity (HMM-MAR)

%% ========== Subject-level dual estimation (extra output) ==========
% Not used for the temporal metrics, which come from the group fit (same
% states as the graph metrics). Kept for subject-specific state parameters.
SubjData = struct();
if do_dual
    S = numel(Ts_list);
    SubjData.hmm_subj = cell(1, S);
    SubjData.Gamma_subj = cell(1, S);
    SubjData.vpath_subj = cell(1, S);
    SubjData.TP_subj = cell(1, S);
    for s = 1:S
        [SubjData.hmm_subj{s}, SubjData.Gamma_subj{s}, SubjData.vpath_subj{s}] = ...
            hmmdual(Ts_list{s}, R.Tvec(s), R.hmm);
        SubjData.TP_subj{s} = getTransProbs(SubjData.hmm_subj{s});
    end
end
R.dual = SubjData;

%% ========== Save ==========
save(fullfile(out_dir, 'raw_hmm_results.mat'), 'R', '-v7.3');

% Same fit under the variable names of the previous hmmmar_results.mat
hmm = R.hmm;
Gamma = vertcat(R.Gamma{:});
vpath = vertcat(R.vpath{:});
GrpData = struct();
GrpData.TP = R.P_offdiag;
GrpData.FO = R.FO_subj;
GrpData.switchingRate = R.switching_rate;
GrpData.maxFO = R.maxFO;
GrpData.viterbipath = vpath;
GrpData.Tvec = R.Tvec;
group_names = categories(groups);
fehist = R.fehist;
fit_id = R.fit_id;
subject_ids = R.subject_ids;
save(fullfile(out_dir, 'hmmmar_results.mat'), 'options', 'SubjData', 'hmm', ...
    'Gamma', 'GrpData', 'vpath', 'groups', 'group_names', 'metadata_aligned', ...
    'subject_ids', 'fehist', 'fit_id', '-v7.3');
fid = fopen(fullfile(out_dir, 'fit_id.txt'), 'w');
fprintf(fid, '%s\n', R.fit_id);
fclose(fid);
save(fullfile(out_dir, 'hmm_config_used.mat'), 'cfg');

%% ========== Occupancy by scan quarter (pre-specified drift test) ==========
% Same definition as run4c_confirm_lowrank.m: per mouse, the mean of each
% state's probability in each quarter of the scan. Tested by
% hmm_occupancy_drift.py with the same code as the confirmatory analysis.
rows = cell(0, 8);
for s = 1:numel(R.Gamma)
    G = R.Gamma{s};
    Ts_n = size(G, 1);
    q = min(floor((0:Ts_n - 1).' / Ts_n * 4) + 1, 4);
    for qq = 1:4
        fo = mean(G(q == qq, :), 1);
        for k = 1:R.K
            rows(end + 1, :) = {R.subject_ids(s), char(groups(s)), R.K, k, qq, ...
                fo(k), R.FO_subj(s, k), R.fit_id}; %#ok<SAGROW>
        end
    end
end
occ = cell2table(rows, 'VariableNames', {'subject_id', 'group', 'K', 'state', ...
    'quarter', 'fo_quarter', 'fo_total', 'fit_id'});
writetable(occ, fullfile(out_dir, 'occupancy_by_quarter.csv'));

%% ========== Save state FC matrices and figures ==========
state_fc = R.state_fc;
state_fc_model = R.state_fc_model;
save(fullfile(out_dir, 'raw_hmm_state_fc.mat'), 'state_fc', 'state_fc_model', 'fit_id', '-v7.3');

finite_max = cellfun(@(A) max(abs(A(:)), [], 'omitnan'), state_fc);
max_abs_r = max(finite_max, [], 'omitnan');
if isempty(max_abs_r) || ~isfinite(max_abs_r) || max_abs_r <= 0
    max_abs_r = 1;
end
for k = 1:R.K
    f = figure('Visible', 'off');
    imagesc(state_fc{k});
    axis square;
    colorbar;
    colormap(parula);
    clim([-max_abs_r, max_abs_r]);
    title(sprintf('Raw-time-series HMM state %d FC', k));
    xlabel('ROI');
    ylabel('ROI');
    exportgraphics(f, fullfile(out_dir, sprintf('state_%02d_raw_HMM_FC.png', k)), ...
        'Resolution', 300);
    close(f);
end
fprintf('Saved to %s\n', out_dir);
