function results = test_workflow_synthetic(hmmmar_dir, bct_dir, out_dir)
% TEST_WORKFLOW_SYNTHETIC  Run the updated run4e -> run4d -> run5 -> run5B
% chain on a small simulated cohort, with an HMM-PCA model (options.lowrank, as
% the reported model), and check that every output describes the same fit. MATLAB only (tables, categorical). About 5-15 min. The simulated
% workflow is kept in <out_dir>/work, so the drift script can be run on it.
%
%   test_workflow_synthetic('/blue/febo/MATLAB/HMM-MAR-master', '/blue/febo/MATLAB/BCT', 'workflow_test')
%
% Checks
%   W1  hmmmar_results.mat and raw_hmm_results.mat hold the same Gamma (one fit)
%   W2  run4d fractional occupancy equals R.FO_subj
%   W3  run4d mean dwell equals the mean of R.lifetimes_TR (group Viterbi)
%   W4  raw_hmm_state_fc.mat equals R.state_fc
%   W5  run5 wrote one row per mouse and state, and its FO equals R.FO_subj
%   W6  run5B Neff equals run5 effective_n; inclusion uses the same threshold
%   W7  every CSV carries the fit ID in fit_id.txt
%   W8  hmm_fit_id: known-answer SHA-256 (same value as Python hashlib)
%   W9  the fit is HMM-PCA (lowrank 5, no pre-PCA) with useParallel = 0

here = fileparts(mfilename('fullpath'));
v2 = fileparts(here);
addpath(v2);
addpath(genpath(hmmmar_dir));
addpath(bct_dir);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
work = fullfile(out_dir, 'work');
if exist(work, 'dir'), rmdir(work, 's'); end
sig = fullfile(work, 'signals', 'hmm_signals');
mkdir(sig);

% ---- simulated cohort: 12 mice, 2 groups, 20 ROIs, 300 TRs, 2 states ----
rng(11, 'twister');
S = 12; T = 300; n = 20;
L1 = randn(n, 3); L2 = randn(n, 3);
for s = 1:S
    x = zeros(T, n);
    for t = 1:T
        if mod(floor((t - 1) / 30), 2) == 0, L = L1; else, L = L2; end
        x(t, :) = (L * randn(3, 1))' + 0.7 * randn(1, n);
    end
    writematrix(x, fullfile(sig, sprintf('signal%d.txt', s)));
end
grp = repmat(["G1"; "G2"], S / 2, 1);
writetable(table((1:S)', grp, 'VariableNames', {'subject_id', 'group'}), ...
    fullfile(sig, 'meta_data.csv'));

env = {'HMM_K', '2'; 'HMM_LOWRANK', '5'; 'HMM_PCA', '0'; 'HMM_TR', '2'; 'HMM_N_SUBJECTS', num2str(S); ...
    'HMM_DATA_DIR', sig; 'HMM_OUT_BASE', fullfile(work, 'signals'); ...
    'HMM_TOOLBOX', hmmmar_dir; 'HMM_BCT', bct_dir; 'HMM_K_VALUES', '2'; 'HMM_LOWRANK_VALUES', '5'; ...
    'HMM_PCA_VALUES', '5'; 'HMM_INITREP', '3'; 'HMM_SEED', ''; 'HMM_CONFIRM_GAMMA', ''; ...
    'HMM_DUAL', '1'; 'HMM_N_BOOT', '40'; 'HMM_MIN_NEFF', '83'; 'HMM_DENSITY', '0.15'; ...
    'HMM_MODEL_DIR', ''; 'HMM_BLOCK', ''; 'HMM_MIN_MASS', ''};   % '' = default, whatever the shell has
for i = 1:size(env, 1), setenv(env{i, 1}, env{i, 2}); end
cleanup = onCleanup(@() cellfun(@(v) setenv(v, ''), env(:, 1)));

run_script(fullfile(v2, 'run4e_hmm_mar.m'));
run_script(fullfile(v2, 'run4d_hmm_mar_plots_analysis.m'));
run_script(fullfile(v2, 'run5_analyze_raw_hmm_graph_metrics.m'));
run_script(fullfile(v2, 'run5B_hmm_block_bootstrap_stability.m'));

model_dir = fullfile(work, 'signals', 'hmmmar_simple_k2_hmmpca5');
A = load(fullfile(model_dir, 'hmmmar_results.mat'));
B = load(fullfile(model_dir, 'raw_hmm_results.mat'));
R = B.R;
results = struct('check', {}, 'value', {}, 'pass', {});

d = max(abs(double(A.Gamma(:)) - reshape(vertcat(R.Gamma{:}), [], 1)));
results(end + 1) = struct('check', 'W1 one fit: max |Gamma difference| between the two files', 'value', d, 'pass', d < 1e-12);

st = readtable(fullfile(work, 'signals', 'hmm_subject_state_FO_all_K_all_lowrank.csv'));
st = sortrows(st, {'state', 'subject_id'});
d = max(abs(st.fractional_occupancy - R.FO_subj(:)));
results(end + 1) = struct('check', 'W2 run4d FO equals R.FO_subj', 'value', d, 'pass', d < 1e-12);

dw = readtable(fullfile(work, 'signals', 'hmm_dwell_times_all_K_all_lowrank.csv'));
dw = sortrows(dw, {'subject_id', 'state'});
ref = cellfun(@(x) mean_or_nan(x), R.lifetimes_TR)';      % K x S -> subject-major
d = max(abs(dw.mean_dwell_TR - ref(:)), [], 'omitnan');
results(end + 1) = struct('check', 'W3 run4d mean dwell equals mean(R.lifetimes_TR)', 'value', d, 'pass', d < 1e-12);

F = load(fullfile(model_dir, 'raw_hmm_state_fc.mat'));
d = max(cellfun(@(a, b) max(abs(a(:) - b(:))), F.state_fc, R.state_fc));
results(end + 1) = struct('check', 'W4 raw_hmm_state_fc.mat equals R.state_fc', 'value', d, 'pass', d < 1e-12);

gm = readtable(fullfile(model_dir, 'hmm_raw_posthoc_graph_metrics', ...
    'graph_metrics_per_subject_per_state_density15.csv'));
gm = sortrows(gm, {'state', 'subject_id'});
d = max(abs(gm.FO - R.FO_subj(:)));
results(end + 1) = struct('check', sprintf('W5 run5 rows = %d (expected %d); FO equals R.FO_subj', ...
    height(gm), S * R.K), 'value', d, 'pass', height(gm) == S * R.K && d < 1e-12);

bs = readtable(fullfile(model_dir, 'hmm_bootstrap_stability', 'bootstrap_stability_subject_state.csv'));
bs = sortrows(bs, {'state', 'subject_id'});
d = max(abs(bs.Neff - gm.effective_n));
same_rule = isequal(logical(bs.passes_Neff), gm.effective_n >= 83);
results(end + 1) = struct('check', 'W6 run5B Neff equals run5 effective_n; same inclusion rule (83)', ...
    'value', d, 'pass', d < 1e-9 && same_rule);

fid = strtrim(fileread(fullfile(model_dir, 'fit_id.txt')));
files = {fullfile(model_dir, 'occupancy_by_quarter.csv'), ...
    fullfile(work, 'signals', 'hmm_subject_model_metrics_all_K_all_lowrank.csv'), ...
    fullfile(work, 'signals', 'hmm_dwell_times_all_K_all_lowrank.csv'), ...
    fullfile(model_dir, 'hmm_raw_posthoc_graph_metrics', 'graph_metrics_per_subject_per_state_density15.csv'), ...
    fullfile(model_dir, 'hmm_bootstrap_stability', 'bootstrap_stability_subject_state.csv'), ...
    fullfile(model_dir, 'hmm_bootstrap_stability', 'bootstrap_stability_by_state.csv')};
n_ok = 0;
for i = 1:numel(files)
    t = readtable(files{i}, 'TextType', 'string');
    n_ok = n_ok + all(t.fit_id == string(fid));
end
results(end + 1) = struct('check', sprintf('W7 fit ID in all %d CSVs equals fit_id.txt and R.fit_id', numel(files)), ...
    'value', n_ok, 'pass', n_ok == numel(files) && strcmp(fid, R.fit_id) && strcmp(A.fit_id, fid));

kat = hmm_fit_id({[0.1 0.9; 0.75 0.25], [1/3 2/3]});
results(end + 1) = struct('check', 'W8 hmm_fit_id known-answer test (SHA-256)', 'value', NaN, ...
    'pass', strcmp(kat, 'e953b926b446c001a42465001bc5eda970d0da098efaffc7eee1c1ff1ab654f5'));

ok = R.hmm.train.lowrank == 5 && R.hmm.train.pca == 0 && R.options.useParallel == 0;
results(end + 1) = struct('check', 'W9 fit is HMM-PCA (lowrank 5, pca 0), useParallel 0', 'value', NaN, ...
    'pass', ok);

Tres = struct2table(results);
disp(Tres);
writetable(Tres, fullfile(out_dir, 'workflow_tests.csv'));
fprintf('%d of %d workflow checks passed.\n', sum([results.pass]), numel(results));
end

function run_script(p)
% Runs a script in its own workspace (the scripts start with "clear").
run(p);
end

function m = mean_or_nan(x)
if isempty(x), m = NaN; else, m = mean(x); end
end
