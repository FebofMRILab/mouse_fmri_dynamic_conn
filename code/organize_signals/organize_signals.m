%% organize_signals.m
% Location: /blue/febo/share/Abeta_Age/rabies/code/organize_signals/organize_signals.m
% Expected siblings: /blue/febo/share/Abeta_Age/rabies/code, /blue/febo/share/Abeta_Age/rabies/confound_correction_outputs,
%                    /blue/febo/share/Abeta_Age/rabies/signals
%
% On the cluster, load FSL before starting MATLAB.

clear; clc;

%% Configuration
numRoi = 82;
atlasFile = '/blue/febo/share/civm_atlas/BrainNetCoord.nii';

scriptDir = fileparts(mfilename('fullpath'));
codeDir = fileparts(scriptDir);
projectDir = fileparts(codeDir);

cleanedDir = fullfile(projectDir, 'confound_correction_outputs', ...
    'confound_correction_datasink', 'cleaned_timeseries');
signalsDir = fullfile(projectDir, 'signals');
hmmDir = fullfile(signalsDir, 'hmm_signals');
metaFile = fullfile(hmmDir, 'meta_data.csv');

%% Check dependencies and metadata before extracting anything
if ~isfile(atlasFile)
    error('Atlas not found: %s', atlasFile);
end
if ~isfolder(cleanedDir)
    error('Cleaned-timeseries directory not found: %s', cleanedDir);
end
if system('command -v fslmaths >/dev/null 2>&1 && ' + ...
          "command -v fslmeants >/dev/null 2>&1") ~= 0
    error('FSL commands are unavailable. Load FSL before starting MATLAB.');
end

% Equivalent to mkdir -p: existing directories are left in place.
if ~isfolder(signalsDir), mkdir(signalsDir); end
if ~isfolder(hmmDir), mkdir(hmmDir); end

if ~isfile(metaFile)
    error('Required metadata file not found: %s', metaFile);
end

meta = readtable(metaFile, 'TextType', 'string');
if ~all(ismember({'subject_id', 'group'}, meta.Properties.VariableNames))
    error('meta_data.csv must contain subject_id and group columns.');
end

if isnumeric(meta.subject_id)
    meta.subject_id = double(meta.subject_id);
else
    meta.subject_id = str2double(string(meta.subject_id));
end
meta.group = categorical(string(meta.group));

if any(~isfinite(meta.subject_id) | meta.subject_id < 0 | ...
       meta.subject_id ~= floor(meta.subject_id))
    error('Metadata subject_id values must be nonnegative integers.');
end
if numel(unique(meta.subject_id)) ~= height(meta)
    error('Duplicate subject_id values in meta_data.csv.');
end
if any(ismissing(meta.group))
    error('Missing group values in meta_data.csv.');
end

%% Locate and check the input images
subjectDirs = dir(fullfile(cleanedDir, '*split*'));
subjectDirs = subjectDirs([subjectDirs.isdir]);
if isempty(subjectDirs)
    error('No *split* subject directories found in %s', cleanedDir);
end

nSubjects = numel(subjectDirs);
subjectIds = zeros(nSubjects, 1);
imageFiles = cell(nSubjects, 1);

for k = 1:nSubjects
    folderName = subjectDirs(k).name;
    token = regexp(folderName, 'sub-(\d+)', 'tokens', 'once');
    if isempty(token)
        error('Cannot infer subject_id from folder name: %s', folderName);
    end
    subjectIds(k) = str2double(token{1});

    matches = dir(fullfile(cleanedDir, folderName, '*cleaned.nii.gz'));
    if numel(matches) ~= 1
        error('Expected one *cleaned.nii.gz in %s; found %d.', ...
            folderName, numel(matches));
    end
    imageFiles{k} = fullfile(matches(1).folder, matches(1).name);
end

if numel(unique(subjectIds)) ~= nSubjects
    error('Multiple input folders resolve to the same numeric subject_id.');
end

% Process in subject-ID order, independent of filesystem listing order.
[subjectIds, order] = sort(subjectIds);
subjectDirs = subjectDirs(order);
imageFiles = imageFiles(order);

%% Create each atlas ROI mask once, not once per subject
setenv('FSLOUTPUTTYPE', 'NIFTI_GZ');
maskDir = tempname;
mkdir(maskDir);
maskCleanup = onCleanup(@() rmdir(maskDir, 's'));
maskFiles = cell(numRoi, 1);

fprintf('Creating %d ROI masks...\n', numRoi);
for roi = 1:numRoi
    maskFiles{roi} = fullfile(maskDir, sprintf('roi%d.nii.gz', roi));
    cmd = sprintf('fslmaths %s -thr %d -uthr %d -bin -fillh %s -odt int', ...
        shellQuote(atlasFile), roi, roi, shellQuote(maskFiles{roi}));
    runCommand(cmd, sprintf('Creating ROI %d mask', roi));
end

%% Extract ROI vectors and assemble each subject's signal matrix
matrices = cell(nSubjects, 1);
signalFiles = strings(nSubjects, 1);

for k = 1:nSubjects
    folderName = subjectDirs(k).name;
    roiDir = fullfile(signalsDir, folderName, 'roi_signals');
    if ~isfolder(roiDir), mkdir(roiDir); end

    info = niftiinfo(imageFiles{k});
    if numel(info.ImageSize) < 4
        error('Expected a 4-D cleaned image: %s', imageFiles{k});
    end
    nTimepoints = info.ImageSize(4);
    Ts = zeros(nTimepoints, numRoi);

    fprintf('Extracting subject %d (%d time points)...\n', ...
        subjectIds(k), nTimepoints);

    for roi = 1:numRoi
        roiFile = fullfile(roiDir, sprintf('roi%d.txt', roi));
        cmd = sprintf('fslmeants -i %s --eig -o %s -m %s', ...
            shellQuote(imageFiles{k}), shellQuote(roiFile), ...
            shellQuote(maskFiles{roi}));
        runCommand(cmd, sprintf('Subject %d, ROI %d', subjectIds(k), roi));

        values = readmatrix(roiFile);
        if ~isnumeric(values) || ~isvector(values) || ...
                numel(values) ~= nTimepoints || any(~isfinite(values(:)))
            error('Invalid ROI vector for subject %d, ROI %d: %s', ...
                subjectIds(k), roi, roiFile);
        end
        Ts(:, roi) = values(:);
    end

    matrices{k} = Ts;
    signalFiles(k) = string(fullfile(hmmDir, ...
        sprintf('signal%d.txt', subjectIds(k))));

    % Write completely before replacing an existing signal file.
    temporaryFile = [tempname(hmmDir) '.txt'];
    writematrix(Ts, temporaryFile, 'Delimiter', 'space');
    [ok, message] = movefile(temporaryFile, char(signalFiles(k)), 'f');
    if ~ok
        error('Could not save %s: %s', signalFiles(k), message);
    end
end

%% Align outputs to metadata row order and save HMM-MAR inputs
nMeta = height(meta);
valid_subject = false(nMeta, 1);
Ts_list = cell(nMeta, 1);
source_files = strings(nMeta, 1);

for i = 1:nMeta
    k = find(subjectIds == meta.subject_id(i), 1);
    if isempty(k)
        warning('No cleaned image for metadata subject_id %d; excluding it.', ...
            meta.subject_id(i));
        continue;
    end
    valid_subject(i) = true;
    Ts_list{i} = matrices{k};
    source_files(i) = signalFiles(k);
end

if ~any(valid_subject)
    error('No extracted subjects match subject_id values in meta_data.csv.');
end

extraIds = setdiff(subjectIds, meta.subject_id);
if ~isempty(extraIds)
    warning('%d extracted subject(s) are absent from metadata and excluded from the aligned outputs.', ...
        numel(extraIds));
end

Ts_list = Ts_list(valid_subject);
subject_ids = meta.subject_id(valid_subject);
group_aligned = meta.group(valid_subject);
source_files = source_files(valid_subject);
n_roi_expected = numRoi;

aligned_tbl = table( ...
    subject_ids(:), group_aligned(:), source_files(:), ...
    cellfun(@(x) size(x, 1), Ts_list), ...
    cellfun(@(x) size(x, 2), Ts_list), ...
    'VariableNames', ...
    {'subject_id', 'group', 'source_file', 'n_timepoints', 'n_rois'});

writetable(aligned_tbl, ...
    fullfile(hmmDir, 'metadata_timeseries_aligned.csv'));
save(fullfile(hmmDir, 'raw_timeseries_aligned.mat'), ...
    'Ts_list', 'subject_ids', 'group_aligned', 'aligned_tbl', ...
    'valid_subject', 'n_roi_expected', '-v7.3');

fprintf('Complete: aligned %d/%d metadata subjects; %d ROIs each.\n', ...
    height(aligned_tbl), nMeta, numRoi);

%% Local helpers
function runCommand(command, description)
    [status, output] = system(command);
    if status ~= 0
        error('%s failed (exit %d):\n%s', description, status, output);
    end
end

function quoted = shellQuote(value)
    apostrophe = char(39);
    replacement = [apostrophe '"' apostrophe '"' apostrophe];
    quoted = [apostrophe ...
        strrep(char(value), apostrophe, replacement) apostrophe];
end
