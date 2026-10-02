% HMM-MAR zero order workflow.

clc;
clear;

%% ========== Configuration ==========
data_dir = '../signals/hmm_signals';
file_pattern = 'signal%d.txt';
n_subjects = 95;
TR_sec = 2;
K = 3; 
Comp = 15;
final_seed = 123;

toolbox_dir = '/blue/febo/MATLAB/HMM-MAR-master';
out_dir = sprintf('../signals/hmmmar_simple_k%d_pca%d', K,Comp);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% HMM specification
options = struct();
options.K = K;
options.standardise = 1;
options.zeromean = 1;
options.covtype = 'full';
options.order = 0;
options.inittype = 'hmmmar';
options.initrep = 15;
options.initcyc = 25;
options.cyc = 1000;
options.verbose = 1;
options.Fs = 1 / TR_sec;
options.DirichletDiag = 10; 
options.pca=Comp;
%options.filter=[0 0.1];
%options.lowrank = Comp;

rng(final_seed, 'twister');

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
n_roi = [];

for s = 1:n_subjects
    filename = fullfile(data_dir, sprintf(file_pattern, s));

    if ~exist(filename, 'file')
        warning('Missing subject file: %s', filename);
        continue;
    end

    try
        Ts = readmatrix(filename);
    catch ME
        warning('Could not read subject %d: %s', s, ME.message);
        continue;
    end

    if ~isnumeric(Ts) || ~ismatrix(Ts) || isempty(Ts) || ...
            size(Ts, 1) < 2 || size(Ts, 2) < 2 || ...
            any(~isfinite(Ts(:)))
        warning('Invalid raw time series for subject %d.', s);
        continue;
    end

    if isempty(n_roi)
        n_roi = size(Ts, 2);
    elseif size(Ts, 2) ~= n_roi
        warning('Subject %d has %d ROIs; expected %d.', ...
            s, size(Ts, 2), n_roi);
        continue;
    end

    Ts_list{s} = double(Ts);
    valid_subject(s) = true;
end

if nnz(valid_subject) < 4
    error('Only %d valid subjects were loaded; at least four are required.', ...
        nnz(valid_subject));
end

Ts_list = Ts_list(valid_subject);
subject_ids = find(valid_subject);
groups = groups(valid_subject);

missing_group = ismissing(groups) | string(groups) == "NA" | ...
    strlength(strtrim(string(groups))) == 0;

if any(missing_group)
    bad_ids = subject_ids(missing_group);
    error('Missing group metadata for subject ID(s): %s', ...
        mat2str(bad_ids(:).'));
end

groups = removecats(groups);

group_names = categories(groups);
for g = 1:numel(group_names)
    n_group = nnz(groups == group_names{g});

    if n_group < 2
        error(['Group "%s" has %d valid subject(s). Stratified split-half ' ...
            'analysis requires at least two.'], group_names{g}, n_group);
    end
end

fprintf('Loaded %d valid subjects with %d ROIs.\n', ...
    numel(Ts_list), n_roi);

metadata_aligned = table(subject_ids(:), groups(:), ...
    'VariableNames', {'subject_id', 'group'});
writetable(metadata_aligned, fullfile(out_dir, 'metadata_aligned.csv'));

Ts_list = Ts_list(:);
n_input_subjects = numel(Ts_list);

if numel(subject_ids) ~= n_input_subjects
    error('subject_ids must have one entry per input subject.');
end
if numel(unique(subject_ids)) ~= n_input_subjects
    error('subject_ids must be unique.');
end
if ~isstruct(options) || ~isfield(options, 'K')
    error('options must be a structure containing options.K.');
end
if ~isscalar(options.K) || options.K < 1 || options.K ~= floor(options.K)
    error('options.K must be a positive integer.');
end
if exist('hmmmar', 'file') ~= 2
    error('hmmmar was not found on the MATLAB path.');
end

first_nonempty = find(~cellfun(@isempty, Ts_list), 1);
if isempty(first_nonempty)
    error('Every entry in Ts_list is empty.');
end

n_roi = size(Ts_list{first_nonempty}, 2);
if n_roi < 2
    error('At least two ROIs are required.');
end

valid_subject = true(n_input_subjects, 1);

for s = 1:n_input_subjects
    Xs = Ts_list{s};

    if ~isnumeric(Xs) || ~ismatrix(Xs) || isempty(Xs)
        warning('Subject ID %g has an empty or nonnumeric matrix and will be removed.', ...
            subject_ids(s));
        valid_subject(s) = false;
        continue;
    end

    Xs = double(Xs);

    if size(Xs, 2) ~= n_roi
        error('Subject ID %g has %d ROIs; expected %d.', ...
            subject_ids(s), size(Xs, 2), n_roi);
    end

    if size(Xs, 1) < 2
        warning('Subject ID %g has fewer than two time points and will be removed.', ...
            subject_ids(s));
        valid_subject(s) = false;
        continue;
    end

    if any(~isfinite(Xs(:)))
        warning('Subject ID %g contains nonfinite values and will be removed.', ...
            subject_ids(s));
        valid_subject(s) = false;
        continue;
    end

    Ts_list{s} = Xs;
end

Ts_list = Ts_list(valid_subject);
subject_ids = subject_ids(valid_subject);
S = numel(Ts_list);

if S < 2
    error('Fewer than two valid subjects remain.');
end

Tvec = cellfun(@(x) size(x, 1), Ts_list);
Tvec = Tvec(:);
Xcat = vertcat(Ts_list{:});

covtype_label = '<unspecified>';
if isfield(options, 'covtype')
    covtype_label = char(string(options.covtype));
end

fprintf(['Fitting raw-time-series HMM-MAR: %d subjects, %d ROIs, ' ...
    '%d total observations.\n'], S, n_roi, sum(Tvec));
fprintf('Model: K=%d, covtype=%s\n', ...
    options.K, covtype_label);

[e_group,e_subj] = explainedvar_PCA(Xcat,Tvec,options);

CorrFC=struct();
[CorrFC.C,CorrFC.A,CorrFC.e,CorrFC.Av] = corrStaticFC(Xcat,Tvec,options,1);

MeanCorrFC(1:n_subjects,1:size(CorrFC.C,3))=0;

for i=1:size(CorrFC.C,3)
CorrFCtmp = ~eye(size(CorrFC.C(:,:,i))) & isfinite(CorrFC.C(:,:,i));
MeanCorrFC(:,i) = sum(abs(CorrFC.C(:,:,i)) .* CorrFCtmp, 2, 'omitnan');
SumCorrFC(:,i)  = sum(CorrFC.C(:,:,i) > 0.3 | CorrFC.C(:,:,i) < -0.3, 'all');
end

MeanCorrFC(isnan(MeanCorrFC))=0;
MeanCorrFCscaled  = (MeanCorrFC - min(MeanCorrFC(:))) ./ (max(MeanCorrFC(:)) - min(MeanCorrFC(:)));
SumCorrFC(isnan(SumCorrFC))=0;


[hmm, Gamma, Xi, vpath, GammaInit, residuals, fehist, feterms, rho] = hmmmar(Xcat, Tvec, options);

SubjData=struct();
for i = 1:size(Tvec,1)
data_subj = Ts_list{i,1};
T_subj = Tvec(i,1);
[SubjData.hmm_subj{i},SubjData.Gamma_subj{i},SubjData.vpath_subj{i}] = hmmdual(data_subj,T_subj,hmm);
SubjData.TP_subj{i} = getTransProbs(SubjData.hmm_subj{i});
end

GrpData=struct();
GrpData.TP = getTransProbs(hmm);
GrpData.FO = getFractionalOccupancy (Gamma,Tvec,options,2);
GrpData.switchingRate = getSwitchingRate(Gamma,Tvec,options);
GrpData.maxFO = getMaxFractionalOccupancy(Gamma,Tvec,options);
GrpData.viterbipath = hmmdecode(Xcat,Tvec,hmm,1);

save(fullfile(out_dir, 'hmmmar_results.mat'), 'options',...
    'SubjData','hmm','Gamma','GrpData','Xi', 'vpath', 'groups',...
    'GammaInit', 'residuals', 'fehist', 'feterms', 'rho',...
    'group_names','metadata_aligned','-v7.3');
	
%% ========== Final fit using all valid subjects ==========
final_options = options;
final_options.K = K;

rng(final_seed, 'twister');
R = hmm_fit_raw_timeseries(Ts_list, final_options, subject_ids);

R.TR_sec = TR_sec;
R.dwell_sec = R.dwell_TR * TR_sec;
R.groups = groups;
%R.model_selection = selection_tbl;

save(fullfile(out_dir, 'raw_hmm_results.mat'), 'R', '-v7.3');
%writetable(selection_tbl, fullfile(out_dir, 'K_selection_summary.csv'));

%% ========== Save state FC matrices and figures ==========
state_fc = R.state_fc;
save(fullfile(out_dir, 'raw_hmm_state_fc.mat'), 'state_fc', '-v7.3');

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
    
    exportgraphics(f, ...
        fullfile(out_dir, sprintf('state_%02d_raw_HMM_FC.png', k)), ...
        'Resolution', 300);
    close(f);
end
