function test_hmm_v2_real_data(hmm_results_path, data_dir, out_dir, hmmmar_dir, bct_dir)
% TEST_HMM_V2_REAL_DATA  Previous vs updated HMM code on the existing mouse fit.
% MATLAB on HiPerGator; no refit. About 20-40 min on one core (Louvain x 50
% is run for every mouse, state and version).
%
%   S = '/blue/febo/share/Abeta_Age/rabies/signals';
%   test_hmm_v2_real_data(fullfile(S, 'hmmmar_simple_k3_pca15/raw_hmm_results.mat'), ...
%       fullfile(S, 'hmm_signals'), 'hmm_v2_real_data_test', ...
%       '/blue/febo/MATLAB/HMM-MAR-master', '/path/to/BCT')
%
% Writes to out_dir:
%   per_subject_fc_diff.csv     per mouse x state: effective n, max and mean |dr|
%                               between previous and updated per-mouse FC
%   graph_metrics_old_vs_new.csv per mouse x state x metric: previous pipeline,
%                               updated pipeline, and updated FC with the
%                               previous graph function (isolates the threshold)
%   graph_metrics_summary.csv   per metric: correlation across mice and the
%                               median absolute difference, for both contrasts
%   group_fc_and_dynamics.csv   per state: r(previous group FC, getFuncConn),
%                               FO (previous vs getFractionalOccupancy), dwell
%                               (most-probable-state runs vs Viterbi visits)
%   test_log.txt                versions, settings and a one-line verdict per test

here = fileparts(mfilename('fullpath'));
addpath(genpath(hmmmar_dir));
addpath(bct_dir);
addpath(fileparts(here));
addpath(fullfile(here, 'legacy'));
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
logf = fopen(fullfile(out_dir, 'test_log.txt'), 'w');
cleanup = onCleanup(@() fclose(logf));
say = @(varargin) fprintf_both(logf, varargin{:});

say('test_hmm_v2_real_data  %s\n', datestr(now));
say('MATLAB %s\n', version);
say('HMM-MAR: %s (git commit: %s)\n', hmmmar_dir, git_commit(hmmmar_dir));
say('BCT: %s\n', bct_dir);
say('Fit: %s\n', hmm_results_path);

load(hmm_results_path, 'R');
S = numel(R.subject_ids); K = R.K; n_roi = R.n_roi;
say('%d mice, K = %d, %d ROIs; lowrank = %s, pca = %s\n', S, K, n_roi, ...
    mat2str(getf(R.hmm.train, 'lowrank')), mat2str(getf(R.hmm.train, 'pca')));

Ts = cell(S, 1);
for s = 1:S
    Ts{s} = readmatrix(fullfile(data_dir, sprintf('signal%d.txt', R.subject_ids(s))));
end
min_eff = 83; density = 0.15;

% ---- per-mouse FC ----
[fc_new, eff_new, sc_new] = hmm_subject_state_fc(Ts, R.Gamma, R.hmm.train, min_eff);
[fc_old, eff_old] = compute_subject_state_fc_robust(Ts, R.Gamma, min_eff);
rows = {};
ut = triu(true(n_roi), 1);
for s = 1:S
    for k = 1:K
        a = fc_new{s, k}; b = fc_old{s, k};
        if all(isfinite(a(:))) && all(isfinite(b(:)))
            d = abs(a(ut) - b(ut));
            rows(end + 1, :) = {R.subject_ids(s), k, eff_new(s, k), max(d), mean(d)}; %#ok<AGROW>
        else
            rows(end + 1, :) = {R.subject_ids(s), k, eff_new(s, k), NaN, NaN}; %#ok<AGROW>
        end
    end
end
T1 = cell2table(rows, 'VariableNames', {'subject_id', 'state', 'effective_n', 'max_abs_dr', 'mean_abs_dr'});
writetable(T1, fullfile(out_dir, 'per_subject_fc_diff.csv'));
excl_new = cellfun(@(A) any(~isfinite(A(:))), fc_new);
excl_old = cellfun(@(A) any(~isfinite(A(:))), fc_old);
say(['T1 per-mouse FC: effective n max |diff| %.2g (previous code renormalised Gamma rows); ' ...
    'exclusions (n < %d) differ for %d mouse-states; max |dr| median %.4f, max %.4f\n'], ...
    max(abs(eff_new(:) - eff_old(:))), min_eff, nnz(excl_new ~= excl_old), ...
    median(T1.max_abs_dr, 'omitnan'), max(T1.max_abs_dr));

% ---- graph metrics ----
v2_metrics = extract_v2_graph_function(fileparts(here));
fields = {'strength_mean', 'C_bin', 'C_wei', 'L_bin', 'L_wei', 'Eglob_bin', 'Eglob_wei', ...
    'assort_bin', 'assort_wei', 'Q_mod', 'transitivity_bin', 'n_components'};
rows = {};
for s = 1:S
    for k = 1:K
        if any(~isfinite(fc_new{s, k}(:))) || any(~isfinite(fc_old{s, k}(:))), continue; end
        seed = 1000000 + double(R.subject_ids(s)) * 100 + k;
        m_old = legacy_graph_metrics(fc_old{s, k}, density, seed);      % previous pipeline
        m_new = v2_metrics(fc_new{s, k}, density, seed);                 % updated pipeline
        m_thr = legacy_graph_metrics(fc_new{s, k}, density, seed);      % updated FC, previous graph code
        for f = 1:numel(fields)
            rows(end + 1, :) = {R.subject_ids(s), k, fields{f}, m_old.(fields{f}), ...
                m_new.(fields{f}), m_thr.(fields{f})}; %#ok<AGROW>
        end
    end
    if mod(s, 10) == 0, fprintf('  graph metrics: %d/%d mice\n', s, S); end
end
T2 = cell2table(rows, 'VariableNames', {'subject_id', 'state', 'metric', 'previous', ...
    'updated', 'updated_fc_previous_graph_code'});
writetable(T2, fullfile(out_dir, 'graph_metrics_old_vs_new.csv'));

srows = {};
for f = 1:numel(fields)
    for k = 1:K
        q = T2(strcmp(T2.metric, fields{f}) & T2.state == k, :);
        ok = isfinite(q.previous) & isfinite(q.updated);
        r_pipe = corr_(q.previous(ok), q.updated(ok));
        d_pipe = median(abs(q.previous(ok) - q.updated(ok)));
        ok2 = isfinite(q.updated) & isfinite(q.updated_fc_previous_graph_code);
        d_thr = max(abs(q.updated(ok2) - q.updated_fc_previous_graph_code(ok2)));
        srows(end + 1, :) = {fields{f}, k, sum(ok), r_pipe, d_pipe, d_thr}; %#ok<AGROW>
    end
end
T3 = cell2table(srows, 'VariableNames', {'metric', 'state', 'n_mice', ...
    'r_previous_vs_updated', 'median_abs_diff', 'max_abs_diff_graph_code_only'});
writetable(T3, fullfile(out_dir, 'graph_metrics_summary.csv'));
say('T2 graph code only (BCT threshold vs previous): max diff over all metrics %.3g\n', ...
    max(T3.max_abs_diff_graph_code_only));
say('T2 whole pipeline: lowest r(previous, updated) across metrics x states %.4f\n', ...
    min(T3.r_previous_vs_updated));

% ---- group FC, FO, dwell ----
Xcat = vertcat(Ts{:});
vpath = hmmdecode(Xcat, R.Tvec, R.hmm, 1);
vpath = double(vpath(:));
offs = [0; cumsum(R.Tvec(:))];
FO_hmm = getFractionalOccupancy(vertcat(R.Gamma{:}), R.Tvec, R.hmm.train, 2);
maxFO = getMaxFractionalOccupancy(vertcat(R.Gamma{:}), R.Tvec, R.hmm.train);
life = getStateLifeTimes(vpath, R.Tvec, R.hmm.train, 0, [], false);
grows = {};
for k = 1:K
    [~, C] = getFuncConn(R.hmm, k, true);
    r_fc = corr_(C(ut), R.state_fc{k}(ut));
    Sk = zeros(n_roi);
    for s = 1:S, Sk = Sk + sc_new{s, k}; end
    Pk = Sk ./ (sqrt(diag(Sk)) * sqrt(diag(Sk))');
    r_pooled = corr_(Pk(ut), R.state_fc{k}(ut));
    dw_v = NaN(S, 1);
    for s = 1:S
        if ~isempty(life{s, k}), dw_v(s) = mean(life{s, k}); end
    end
    grows(end + 1, :) = {k, r_fc, r_pooled, mean(R.FO_subj(:, k)), mean(FO_hmm(:, k)), ...
        max(abs(R.FO_subj(:, k) - FO_hmm(:, k))), median(R.dwell_TR(:, k), 'omitnan'), ...
        median(dw_v, 'omitnan')}; %#ok<AGROW>
end
T4 = cell2table(grows, 'VariableNames', {'state', 'r_previous_groupFC_vs_getFuncConn', ...
    'r_previous_groupFC_vs_pooled_scatter', ...
    'FO_previous', 'FO_getFractionalOccupancy', 'FO_max_abs_diff', ...
    'dwell_TR_previous_median', 'dwell_TR_viterbi_median'});
writetable(T4, fullfile(out_dir, 'group_fc_and_dynamics.csv'));
writetable(table(R.subject_ids(:), maxFO(:), 'VariableNames', {'subject_id', 'maxFO'}), ...
    fullfile(out_dir, 'maxFO_per_mouse.csv'));
lab = cellfun(@(G) argmax_(G), R.Gamma, 'UniformOutput', false);
agree = mean(vertcat(lab{:}) == vpath(:));
say('T3 group FC r(previous, getFuncConn): %s; r(previous, pooled scatter = new R.state_fc): %s\n', ...
    mat2str(T4.r_previous_groupFC_vs_getFuncConn', 4), mat2str(T4.r_previous_groupFC_vs_pooled_scatter', 4));
say('T3 FO max |diff|: %.2g; Viterbi vs most-probable state agree at %.1f%% of TRs\n', ...
    max(T4.FO_max_abs_diff), 100 * agree);
say('T3 maxFO: median %.3f, %d of %d mice above 0.9\n', median(maxFO), sum(maxFO > 0.9), S);
say('Done.\n');
end

% ======================================================================
function fprintf_both(fid, varargin)
fprintf(varargin{:});
fprintf(fid, varargin{:});
end

function v = getf(s, f)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = 0; end
end

function c = git_commit(d)
[st, out] = system(sprintf('git -C "%s" rev-parse HEAD', d));
if st == 0, c = strtrim(out); else, c = 'not a git checkout: record the download date'; end
end

function c = corr_(a, b)
a = a(:) - mean(a); b = b(:) - mean(b);
c = (a' * b) / sqrt((a' * a) * (b' * b));
end

function l = argmax_(G)
[~, l] = max(G, [], 2);
end

function fn = extract_v2_graph_function(v2_dir)
txt = fileread(fullfile(v2_dir, 'run5_analyze_raw_hmm_graph_metrics.m'));
i0 = strfind(txt, 'function metrics = compute_graph_metrics_single_robust');
body = strrep(txt(i0:end), 'compute_graph_metrics_single_robust', 'v2_graph_metrics_tmp');
d = tempname; mkdir(d);
fid = fopen(fullfile(d, 'v2_graph_metrics_tmp.m'), 'w'); fwrite(fid, body); fclose(fid);
addpath(d);
fn = @v2_graph_metrics_tmp;
end
