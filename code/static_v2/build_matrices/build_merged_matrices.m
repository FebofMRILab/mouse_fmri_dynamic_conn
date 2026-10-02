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
data_dir     = '../../signals/hmm_signals';
out_dir      = '../../static';
mapping_csv  = fullfile(out_dir, 'subject_group_order_mappings.csv');  % optional check
expected_timepoints = 570;          % 600 volumes minus 15 at each end; [] to skip

%% ---------- Find signal files in natural numeric order ----------
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
fprintf('Block order written to merged_block_order.txt\n');