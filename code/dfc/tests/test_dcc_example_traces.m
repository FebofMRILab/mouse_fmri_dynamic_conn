function results = test_dcc_example_traces(out_dir)
% TEST_DCC_EXAMPLE_TRACES  Plumbing and equivalence checks for
% dcc_example_traces.m on simulated mice, with a stand-in for dcc_mvgarch.m
% (an EWMA covariance), so it runs without the Optimization Toolbox, in MATLAB
% or Octave. The published estimator itself was checked against ours in
% equivalence_published/compare_dcc_with_lindquist.m.
%
%   test_dcc_example_traces('dcc_example_test')
%
% Checks
%   X1  the mouse is drawn from NTG_Y, and the same mouse is drawn again
%   X2  traces: 3 fixed pairs x T rows; |r| <= 1; surrogate quantiles ordered
%   X3  one SD row per pair (N(N-1)/2); their mean equals corr_fluct in params
%   X4  group PCA equals run_dcc_cpca_null.m's local_group_pca (its local
%       function, extracted unchanged)
%   X5  surrogate r of the example equals run_dcc_cpca_null.m's
%       local_surrogate(X, 'phase', id, r, cfg) (same seeds, same surrogates)
%   X6  the mouse's pipeline values are read from dfc_subject_metrics.csv

here = fileparts(mfilename('fullpath'));
dfc = fileparts(here);
addpath(dfc);
if exist(out_dir, 'dir'), rmdir(out_dir, 's'); end
mkdir(out_dir);
sig = fullfile(out_dir, 'hmm_signals'); mkdir(sig);
stub = fullfile(out_dir, 'stub'); mkdir(stub);
fid = fopen(fullfile(stub, 'dcc_mvgarch.m'), 'w');
fprintf(fid, ['function [p, ll, Ht] = dcc_mvgarch(E, varargin)\n' ...
    '%% Stand-in for tests: EWMA covariance (lambda 0.94); fixed parameters.\n' ...
    '[T, N] = size(E); Ht = zeros(N, N, T); H = cov(E);\n' ...
    'for t = 1:T, if t > 1, H = 0.94 * H + 0.06 * E(t - 1, :)'' * E(t - 1, :); end; Ht(:, :, t) = H; end\n' ...
    'p = [repmat([0.1; 0.1; 0.8], N, 1); 0.05; 0.9]; ll = 0;\nend\n']);
fclose(fid);

% ---- simulated cohort: 10 mice, 82 ROIs, 570 volumes, 4 NTG_Y ----
rand('seed', 3); randn('seed', 3); %#ok<RAND>
S = 10; T = 570; n = 82;
L = randn(n, 6);
for s = 1:S
    x = filter(1, [1 -0.6], randn(T, 6)) * L' + randn(T, n);
    dlmwrite(fullfile(sig, sprintf('signal%d.txt', s)), x, ' ');
end
groups = {'NTG_Y', 'CRND8_Y', 'NTG_Y', 'NTG_M', 'NTG_Y', '5XFAD_Y', 'NTG_A', 'NTG_Y', 'CRND8_M', '5XFAD_M'};
fid = fopen(fullfile(sig, 'meta_data.csv'), 'w');
fprintf(fid, 'subject_id,group\n');
for s = 1:S, fprintf(fid, '%d,%s\n', s, groups{s}); end
fclose(fid);
dfcdir = fullfile(out_dir, 'dfc_dcc_cpca'); mkdir(dfcdir);
fid = fopen(fullfile(dfcdir, 'dfc_config.txt'), 'w');
fprintf(fid, 'TR = 2\npca_dim = 25\nar_order = 2\nn_surr = 3\nseed = 123\ndcc_decimate = 2\n');
fclose(fid);

% the pipeline's own local functions, unchanged
v2tests = fullfile(fileparts(dfc), 'hmm_v2', 'tests');
ext = fullfile(out_dir, 'pipeline_local');
addpath(v2tests);
extract_local_functions(fullfile(dfc, 'run_dcc_cpca_null.m'), ext);
addpath(ext);
Xs = cell(1, S); Xraw = cell(1, S);
for s = 1:S
    Xraw{s} = dlmread(fullfile(sig, sprintf('signal%d.txt', s)));
    Xs{s} = local_standardise(Xraw{s});
end
[A_pipe, expl_pipe] = local_group_pca(Xs, 25);        % pipeline's function
fid = fopen(fullfile(dfcdir, 'pca_explained.csv'), 'w');
fprintf(fid, 'component,explained\n');
fprintf(fid, '%d,%.15g\n', [(1:25); expl_pipe(:)']);
fclose(fid);
rmpath(ext);                                          % the example uses its own copies
fid = fopen(fullfile(dfcdir, 'dfc_subject_metrics.csv'), 'w');   % pipeline table format
fprintf(fid, ['subject_id,group,metric,role,observed,phase_mean,phase_sd,phase_z,phase_p,' ...
    'iaaft_mean,iaaft_sd,iaaft_z,iaaft_p,phase_replica_z\n']);
mets = {'dcc_a', 'dcc_LR', 'corr_fluct', 'garch_alpha'};
for s = 1:S
    for m = 1:numel(mets)
        fprintf(fid, '%d,%s,%s,test,%.4f,%.4f,0.0060,%.2f,0.05,,,,,%.2f\n', s, groups{s}, mets{m}, ...
            s / 100 + m, s / 100 + m - 0.001, 1 + s / 10, 0.1 * m);
    end
end
fclose(fid);

% ---- run twice ----
info = dcc_example_traces(sig, stub, dfcdir, fullfile(dfcdir, 'dcc_example'), []);
info2 = dcc_example_traces(sig, stub, dfcdir, fullfile(dfcdir, 'dcc_example_2'), []);
res = {};
ok = any(strcmp(groups{info.mouse_id}, 'NTG_Y')) && info.mouse_id == info2.mouse_id;
res(end + 1, :) = {sprintf('X1 mouse %d is NTG_Y and drawn again', info.mouse_id), info.mouse_id, ok};

tr = local_csv(fullfile(dfcdir, 'dcc_example', 'dcc_example_traces.csv'));
Tres = 570 / 2 - 2;
ok = numel(tr.pair) == 3 * Tres && all(abs(tr.observed_r) <= 1 + 1e-12) ...
    && all(tr.surr_p025 <= tr.surr_p50 + 1e-12) && all(tr.surr_p50 <= tr.surr_p975 + 1e-12) ...
    && isequal(unique(tr.pair)', {'1-2', '1-3', '2-3'});
res(end + 1, :) = {sprintf('X2 traces: %d rows (3 x %d), |r| <= 1, quantiles ordered', numel(tr.pair), Tres), ...
    numel(tr.pair), ok};

sd = local_csv(fullfile(dfcdir, 'dcc_example', 'dcc_example_pair_sd.csv'));
pr = local_csv(fullfile(dfcdir, 'dcc_example', 'dcc_example_params.csv'));
d = abs(mean(sd.sd_observed) - pr.corr_fluct(1));
ok = numel(sd.sd_observed) == 25 * 24 / 2 && d < 1e-5;
res(end + 1, :) = {'X3 300 pair rows; mean SD = corr_fluct in params (|diff|)', d, ok};

txt = fileread(fullfile(dfcdir, 'dcc_example', 'dcc_example_info.txt'));
tok = regexp(txt, 'max \|difference\| from the pipeline''s pca_explained.csv = ([0-9.eE+-]+)', 'tokens', 'once');
d = str2double(tok{1});
res(end + 1, :) = {'X4 group PCA = pipeline local_group_pca (max |diff| in explained)', d, d < 1e-10};

% X5: surrogate 1 of the example = the pipeline's surrogate 1 for this mouse
cfg = struct('seed', 123, 'null_types', {{'phase', 'iaaft'}});
addpath(ext);
Xp = local_surrogate(Xraw{info.mouse_id}, 'phase', info.mouse_id, 1, cfg);   % pipeline
rmpath(ext);
rng(123 + info.mouse_id * 1000 + 1, 'twister');                                % example's line
Xe = hmm_phase_randomise(Xraw{info.mouse_id}, []);
d = max(abs(Xp(:) - Xe(:)));
res(end + 1, :) = {'X5 example surrogate = pipeline local_surrogate (max |diff|)', d, d == 0};

want = sprintf('dcc_a: observed %.4f, phase-surrogate mean %.4f (SD 0.0060), z %.2f, replica z 0.10', ...
    info.mouse_id / 100 + 1, info.mouse_id / 100 + 1 - 0.001, 1 + info.mouse_id / 10);
ok = ~isempty(strfind(txt, want)); %#ok<STREMP>
res(end + 1, :) = {'X6 pipeline values for the mouse read from dfc_subject_metrics.csv', double(ok), ok};

results = cell2struct(res, {'check', 'value', 'pass'}, 2);
fprintf('\n');
for i = 1:size(res, 1)
    fprintf('%-72s %10.3g  %s\n', res{i, 1}, res{i, 2}, local_pf(res{i, 3}));
end
fprintf('\n%d passed, %d failed.\n', sum([res{:, 3}]), sum(~[res{:, 3}]));
end


function S = local_csv(f)
L = strsplit(strtrim(fileread(f)), {'\r\n', '\n'});
h = strsplit(L{1}, ',', 'CollapseDelimiters', false);
C = cellfun(@(x) strsplit(x, ',', 'CollapseDelimiters', false), L(2:end), 'UniformOutput', false);
C = vertcat(C{:});
S = struct();
for k = 1:numel(h)
    v = str2double(C(:, k));
    if all(isnan(v)), S.(h{k}) = C(:, k); else, S.(h{k}) = v; end
end
end


function s = local_pf(tf)
if tf, s = 'PASS'; else, s = 'FAIL'; end
end
