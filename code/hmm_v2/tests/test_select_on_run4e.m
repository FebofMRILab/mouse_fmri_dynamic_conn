function results = test_select_on_run4e(hmmmar_dir, workflow_dir, out_dir)
% TEST_SELECT_ON_RUN4E  The fit-selection step on real run4e outputs: eight
% run4e fits of the simulated cohort from test_workflow_synthetic (run that
% first), each with its own HMM_OUT_BASE as run_hmm_fits_array.sbatch sets it,
% then select_reported_fit. MATLAB only (run4e uses tables); a few minutes.
%
%   test_select_on_run4e('/blue/febo/MATLAB/HMM-MAR-master', 'hmm_v2_test_results/workflow', ...
%       'hmm_v2_test_results/select_on_run4e')
%
% Checks
%   R1  eight fits, eight different fit IDs (HMM-MAR reseeds from the clock)
%   R2  select_reported_fit reads run4e's files and reports the fit with the
%       lowest fehist(end)
%   R3  inference_scope.csv is in that fit's folder only, with its fit ID
%   R4  the simulated two-state structure is found again: both states recur and
%       getGammaSimilarity to the reported fit is > 0.9 in every other fit

here = fileparts(mfilename('fullpath'));
v2 = fileparts(here);
addpath(v2);
addpath(genpath(hmmmar_dir));
sig = fullfile(workflow_dir, 'work', 'signals', 'hmm_signals');
if ~exist(fullfile(sig, 'signal1.txt'), 'file')
    error('Run test_workflow_synthetic first (no simulated cohort in %s).', sig);
end
if exist(out_dir, 'dir'), rmdir(out_dir, 's'); end
fits = fullfile(out_dir, 'hmm_k2_hmmpca5_fits');
mkdir(fits);
folder = 'hmmmar_simple_k2_hmmpca5';
env = {'HMM_K', '2'; 'HMM_LOWRANK', '5'; 'HMM_PCA', '0'; 'HMM_TR', '2'; 'HMM_N_SUBJECTS', '12'; ...
    'HMM_DATA_DIR', sig; 'HMM_TOOLBOX', hmmmar_dir; 'HMM_INITREP', '3'; 'HMM_DUAL', '0'; ...
    'HMM_SEED', ''; 'HMM_MODEL_DIR', ''; 'HMM_CONFIRM_GAMMA', ''; 'HMM_OUT_BASE', ''};
for i = 1:size(env, 1), setenv(env{i, 1}, env{i, 2}); end
cleanup = onCleanup(@() cellfun(@(v) setenv(v, ''), env(:, 1)));

[st, out] = system(sprintf('sha256sum "%s"', fullfile(v2, 'FIT_SELECTION_RULE.md')));
assert(st == 0, 'sha256sum failed');
fid = fopen(fullfile(fits, 'rule_sha256.txt'), 'w');
fprintf(fid, 'sha256: %s\nrecorded: test\n', strtok(out));
fclose(fid);
fid = fopen(fullfile(fits, 'model.env'), 'w');      % as the array job copies the model file
fprintf(fid, 'HMM_K=2\nHMM_LOWRANK=5\nHMM_PCA=0\n');
fclose(fid);

n = 8;
ids = cell(1, n); fe = NaN(1, n);
for i = 1:n                                  % as the array job: own HMM_OUT_BASE per fit
    setenv('HMM_OUT_BASE', fullfile(fits, sprintf('fit_%02d', i)));
    run_script(fullfile(v2, 'run4e_hmm_mar.m'));
    A = load(fullfile(fits, sprintf('fit_%02d', i), folder, 'hmmmar_results.mat'), 'fit_id', 'fehist');
    ids{i} = A.fit_id; fe(i) = A.fehist(end);
end
results = struct('check', {}, 'value', {}, 'pass', {});
nu = numel(unique(ids));
results(end + 1) = struct('check', 'R1 eight run4e fits, distinct fit IDs', 'value', nu, 'pass', nu == n);

setenv('HMM_OUT_BASE', fullfile(workflow_dir, 'work', 'signals'));   % the workflow fit = "previous refit"
sel = select_reported_fit(fits);
[~, best] = min(fe);
ok = strcmp(sel.reported_fit, sprintf('fit_%02d', best)) && strcmp(sel.fit_id, ids{best});
results(end + 1) = struct('check', sprintf('R2 reported %s = lowest fehist(end) (fit_%02d)', ...
    sel.reported_fit, best), 'value', min(fe), 'pass', ok);

n_scope = 0;
for i = 1:n
    n_scope = n_scope + (exist(fullfile(fits, sprintf('fit_%02d', i), folder, 'inference_scope.csv'), 'file') == 2);
end
txt = fileread(fullfile(sel.model_dir, 'inference_scope.csv'));
ok = n_scope == 1 && contains(txt, sel.fit_id);
results(end + 1) = struct('check', 'R3 inference_scope.csv in the reported fit only, with its fit ID', ...
    'value', n_scope, 'pass', ok);

others = setdiff(1:n, find(strcmp(sel.fit_names, sel.reported_fit)));
ok = all(sel.recurs) && all(sel.similarity(others) > 0.9);
results(end + 1) = struct('check', sprintf('R4 both simulated states recur; min getGammaSimilarity %.3f', ...
    min(sel.similarity(others))), 'value', min(sel.similarity(others)), 'pass', ok);

disp(struct2table(results));
fprintf('%d of %d checks passed.\n', sum([results.pass]), numel(results));
end

function run_script(p)
% Runs a script in its own workspace (run4e starts with "clear").
run(p);
end
