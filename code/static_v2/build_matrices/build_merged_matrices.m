%% build_merged_matrices.m
% Build merged.csv: one Pearson correlation matrix per subject, stacked in
% subject order, in the format read by the Python analysis scripts.
%
% Each block is (n_roi + 1) x (n_roi + 1):
%   top-left cell   empty (written from NaN)  <- REQUIRED: pandas names the
%                   first column "Unnamed: 0", which the loaders drop
%   first row       ROI numbers 1..n_roi      <- header row, detected and removed
%   first column    ROI numbers 1..n_roi      <- row labels
%   remainder       n_roi x n_roi correlation matrix
%
% Block k is the k-th signal file in NATURAL numeric order (signal1, signal2,
% ..., signal10), i.e. the same order as `ls -v`. The order actually used is
% written to merged_block_order.txt so it can be checked against the mapping
% file.

clc; clear;

addpath('/blue/febo/MATLAB/BCT');   % no savepath: do not modify the shared path

%% ---------- Configuration ----------
data_dir     = '../../../signals/hmm_signals';
out_dir      = '../../../static';
mapping_csv  = fullfile(out_dir, 'subject_group_order_mappings.csv');  % optional check
expected_timepoints = 570;          % 600 volumes minus 15 at each end; [] to skip
if ~isempty(getenv('EXPECTED_TIMEPOINTS'))   % other studies, e.g. 560 = 600 - 2 x 20 at TR 1.5 s
    expected_timepoints = str2double(getenv('EXPECTED_TIMEPOINTS'));
end

%% ---------- Subjects and their order ----------
% With signals/hmm_signals/meta_data.csv (written for organize_signals.m) the blocks follow its rows:
% signal<subject_id>.txt for each row, in row order, and only those (2026-11-08). For a metadata file
% listing 1..N in order this is the same as the natural numeric order used before. Without it, the
% previous behaviour: every signal*.txt in natural numeric order.
meta_csv = fullfile(data_dir, 'meta_data.csv');
meta_groups = strings(0, 1);
if isfile(meta_csv)
    meta = readtable(meta_csv, 'TextType', 'string');
    ids = meta.subject_id;
    if ~isnumeric(ids), ids = str2double(ids); end
    names = arrayfun(@(i) sprintf('signal%d.txt', i), ids(:)', 'UniformOutput', false);
    missing = names(~cellfun(@(f) isfile(fullfile(data_dir, f)), names));
    if ~isempty(missing)
        error('Signal files missing for meta_data.csv subjects: %s', strjoin(missing, ', '));
    end
    meta_groups = string(meta.group(:));
    n_subjects = numel(names);
    fprintf('Using meta_data.csv: %d subjects in its row order (%s ... %s).\n', ...
        n_subjects, names{1}, names{end});
else
files = dir(fullfile(data_dir, 'signal*.txt'));
if isempty(files)
    error('No signal*.txt files found in %s', data_dir);
end

names = {files.name};
numbers = str2double(regexp(names, '\d+', 'match', 'once'));
if any(isnan(numbers))
    error('Could not read a subject number from: %s', ...
        strjoin(names(isnan(numbers)), ', '));
end
[numbers, order] = sort(numbers);
names = names(order);
n_subjects = numel(names);

fprintf('Found %d signal files (%s ... %s).\n', n_subjects, names{1}, names{end});
end

%% ---------- Build the stacked matrix ----------
first = readmatrix(fullfile(data_dir, names{1}));
n_roi = size(first, 2);
fprintf('Number of ROIs is %d.\n', n_roi);

headers = 1:n_roi;
block_rows = n_roi + 1;
fullmat = zeros(block_rows * n_subjects, block_rows);

for k = 1:n_subjects
    signal = readmatrix(fullfile(data_dir, names{k}));

    if size(signal, 2) ~= n_roi
        error('%s has %d ROIs; expected %d.', names{k}, size(signal, 2), n_roi);
    end
    if ~isempty(expected_timepoints) && size(signal, 1) ~= expected_timepoints
        error('%s has %d time points; expected %d.', ...
            names{k}, size(signal, 1), expected_timepoints);
    end
    if any(~isfinite(signal(:)))
        error('%s contains non-finite values.', names{k});
    end

    block = [NaN, headers; headers', corr(signal)];   % NaN corner -> empty cell
    rows = (k - 1) * block_rows + (1:block_rows);
    fullmat(rows, :) = block;
end

%% ---------- Checks before writing ----------
n_blocks = size(fullmat, 1) / block_rows;
if n_blocks ~= n_subjects
    error('Built %d blocks from %d files.', n_blocks, n_subjects);
end

if exist(mapping_csv, 'file')
    opts = delimitedTextImportOptions('NumVariables', 3, ...
        'Delimiter', ',', 'VariableNamesLine', 1, 'DataLines', [2 Inf], ...
        'VariableNames', {'index', 'genotype', 'matrix_list'}, ...
        'VariableTypes', {'double', 'string', 'string'});
    opts.SelectedVariableNames = {'index', 'genotype'};
    mapping = readtable(mapping_csv, opts);

    if height(mapping) ~= n_subjects
        error(['%d signal files but %d subjects in %s. Block order is by file ' ...
            'position, so every subject after a mismatch would be mislabeled.'], ...
            n_subjects, height(mapping), mapping_csv);
    end
    fprintf('Block count matches the %d subjects in the mapping file.\n', height(mapping));
    if ~isempty(meta_groups)
        bad = find(strtrim(string(mapping.genotype)) ~= strtrim(meta_groups));
        if ~isempty(bad)
            error('Group differs between %s and meta_data.csv at block(s) %s.', mapping_csv, mat2str(bad'));
        end
        fprintf('Groups in the mapping file match meta_data.csv block by block.\n');
    end
end

%% ---------- Write (NaN corner cells as empty fields, as before) ----------
fprintf('Saving merged matrices...\n');

fid = fopen(fullfile(out_dir, 'merged.csv'), 'w');
if fid < 0
    error('Could not open %s for writing.', fullfile(out_dir, 'merged.csv'));
end
for r = 1:size(fullmat, 1)
    row = fullmat(r, :);
    fields = compose("%.15g", row);
    fields(isnan(row)) = "";
    fprintf(fid, '%s\n', strjoin(fields, ','));
end
fclose(fid);

fprintf('Wrote %d matrices to %s\n', n_blocks, fullfile(out_dir, 'merged.csv'));
writelines(string(names(:)), fullfile(out_dir, 'merged_block_order.txt'));   % was announced but never written
fprintf('Block order written to merged_block_order.txt\n');