function names = extract_local_functions(script_path, out_dir)
% EXTRACT_LOCAL_FUNCTIONS  Write each local function of a script to its own
% file in out_dir, unchanged, so a test can call the exact code the script
% runs. Local functions start with "function" in the first column and run to
% the next one (or the end of the file).
%
%   names = extract_local_functions('../run5B_hmm_block_bootstrap_stability.m', tmp)

txt = fileread(script_path);
lines = regexp(txt, '\r?\n', 'split');
starts = find(~cellfun(@isempty, regexp(lines, '^function\s', 'once')));
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
names = cell(1, numel(starts));
for i = 1:numel(starts)
    a = starts(i);
    if i < numel(starts), b = starts(i + 1) - 1; else, b = numel(lines); end
    % the signature may continue over lines ending in "..."
    sig = lines{a};
    j = a;
    while ~isempty(regexp(strtrim(sig), '\.\.\.$', 'once'))
        j = j + 1;
        sig = [regexprep(strtrim(sig), '\.\.\.$', ' '), strtrim(lines{j})];
    end
    tok = regexp(sig, '^function\s+(?:\[[^\]]*\]\s*=\s*|\w+\s*=\s*)?(\w+)', 'tokens', 'once');
    names{i} = tok{1};
    fid = fopen(fullfile(out_dir, [names{i}, '.m']), 'w');
    fprintf(fid, '%s\n', lines{a:b});
    fclose(fid);
end
end
