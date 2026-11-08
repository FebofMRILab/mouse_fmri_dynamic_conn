function results = test_select_reported_fit(hmmmar_dir, out_dir)
% TEST_SELECT_REPORTED_FIT  select_reported_fit.m on simulated fits whose
% answer is known. MATLAB or Octave; a few seconds.
%
%   test_select_reported_fit('/blue/febo/MATLAB/HMM-MAR-master', 'select_fit_test')
%
% Simulated: 6 mice, 12 ROIs, K = 3; ten fits of one "true" solution, each with
% its own state order and noise; fit_10 is an exact copy of fit_02; fit_11 is a
% finished fit with the lowest free energy of all but is not a candidate (the
% rule has fit_01-fit_10). The lowest free energy among candidates is fit_04. True states 1 and 2 have
% the same state FC in every fit; true state 3 keeps its FC in fits 1-4 only.
% Switching rate agrees across fits; max FO does not.
%
% Checks
%   F1  reported fit = lowest final free energy (fit_04)
%   F2  getGammaSimilarity recovers every fit's known state order
%   F3  recurring states: true states 1 and 2 recur (8 of 8), state 3 does
%       not (3 of 8; 5 needed)
%   F4  the copy of fit_02 counts once (9 valid fits); fit_11 is not used
%   F5  switching rate tested, max FO not
%   F6  inference_scope.csv written in the reported fit only (stale copy removed)
%   F7  per-mouse agreement (hmmtestretest statistic): its T-weighted mean
%       equals getGammaSimilarity
%   F8  rerun: the same selection is accepted; a different one is refused
%   F9  a changed rule file is refused
%   F10 fewer than 8 valid fits are refused
%   F11 the confirmation's full fit is compared and its state order recovered;
%       the 4 Oct refit comparison is reported as not applicable
%       (FIT_SELECTION_RULE.md amendment 1), although a fit sits in its folder
%   F12 a finished fit outside fit_01-fit_10 is not a candidate, even with the
%       lowest free energy; an unfinished candidate is not used
%   F13 no recurring state: switching rate and max FO are not tested either
%       (rule item 9), even when they agree across fits
%   F14 a fits folder whose model.env (copied before fitting) is another model
%       is refused
%   F15 a Gaussian PCA model (options.pca, no lowrank): the default fits folder
%       is hmm_k3_pca5_fits, the fits are selected, and a fit with another
%       options.pca is refused

here = fileparts(mfilename('fullpath'));
v2 = fileparts(here);
addpath(v2);
addpath(genpath(hmmmar_dir));
if exist(out_dir, 'dir'), rmdir(out_dir, 's'); end
mkdir(out_dir);
sig = fullfile(out_dir, 'signals');
fits = fullfile(sig, 'hmm_k3_hmmpca5_fits');
folder = 'hmmmar_simple_k3_hmmpca5';
mkdir(fits);
env = {'HMM_K', '3'; 'HMM_LOWRANK', '5'; 'HMM_PCA', '0'; 'HMM_INITREP', '3'; 'HMM_OUT_BASE', sig; ...
    'HMM_TOOLBOX', hmmmar_dir; 'HMM_CONFIRM_GAMMA', fullfile(sig, 'confirm_gamma.mat'); 'HMM_MODEL_DIR', ''};
for i = 1:size(env, 1), setenv(env{i, 1}, env{i, 2}); end
cleanup = onCleanup(@() cellfun(@(v) setenv(v, ''), env(:, 1)));

% ---- the true solution ----
rand('seed', 7); randn('seed', 7); %#ok<RAND>
S = 6; T = 120; n = 12; K = 3;
Tvec = T * ones(S, 1);
lab = zeros(S * T, 1);
for s = 1:S
    t = 0; x = zeros(T, 1);
    while t < T
        d = 6 + floor(rand * 10); x(t + 1:min(T, t + d)) = 1 + floor(rand * K); t = t + d;
    end
    lab((s - 1) * T + (1:T)) = x;
end
Gb = 0.05 * ones(S * T, K);
Gb(sub2ind(size(Gb), (1:S * T)', lab)) = 0.9;
fcb = cell(1, K);
for k = 1:K, fcb{k} = local_randcorr(n); end
sw0 = 0.05 + 0.1 * rand(S, 1);

perms = zeros(10, K);
fe = [-50 -52 -49 -60 -55 -51 -53 -54 -58];
keep3 = [1 2 3 4];                       % fits in which true state 3 keeps its FC
for i = 1:9
    perms(i, :) = local_randperm(K);
    G = zeros(S * T, K);
    G(:, perms(i, :)) = max(Gb + 0.02 * randn(size(Gb)), 0.001);
    G = G ./ sum(G, 2);
    fc = cell(1, K);
    for k = 1:K
        if k < 3 || any(keep3 == i)
            fc{perms(i, k)} = local_noisy(fcb{k}, 0.02);
        else
            fc{perms(i, k)} = local_randcorr(n);
        end
    end
    local_write_fit(fullfile(fits, sprintf('fit_%02d', i), folder), G, fc, fe(i), ...
        sw0 + 0.002 * randn(S, 1), 0.3 + 0.6 * rand(S, 1), Tvec, K);
end
perms(10, :) = perms(2, :);
copyfile(fullfile(fits, 'fit_02'), fullfile(fits, 'fit_10'));          % identical fit
G = zeros(S * T, K); G(:, 1:K) = max(Gb + 0.02 * randn(size(Gb)), 0.001); G = G ./ sum(G, 2);
local_write_fit(fullfile(fits, 'fit_11', folder), G, fcb, -100, sw0, rand(S, 1), Tvec, K);  % not a candidate
fid = fopen(fullfile(fits, 'fit_01', folder, 'inference_scope.csv'), 'w'); fprintf(fid, 'stale\n'); fclose(fid);

% earlier fits: 4 Oct refit (same truth, own order) and the confirmatory Gamma
pp = local_randperm(K);
G = zeros(S * T, K); G(:, pp) = Gb; fc = cell(1, K);
for k = 1:K, fc{pp(k)} = local_noisy(fcb{k}, 0.02); end
local_write_fit(fullfile(sig, folder), G, fc, -40, sw0, rand(S, 1), Tvec, K);
pc = local_randperm(K);
Gamma = zeros(S * T, K); Gamma(:, pc) = Gb; T_sub = Tvec; subject_ids = (1:S)'; %#ok<NASGU>
save('-v7', fullfile(sig, 'confirm_gamma.mat'), 'Gamma', 'T_sub', 'subject_ids');

% the rule as recorded before fitting (as run_hmm_fits_array.sbatch writes it)
rule = fullfile(v2, 'FIT_SELECTION_RULE.md');
local_record_rule(rule, fits);

% ---- run ----
sel = select_reported_fit(fits);
res = {};
ref = 4;
inv_ref = zeros(1, K); inv_ref(perms(ref, :)) = 1:K;    % reported state a = true state inv_ref(a)
res(end + 1, :) = {'F1 reported fit = lowest free energy (fit_04)', double(strcmp(sel.reported_fit, 'fit_04')), ...
    strcmp(sel.reported_fit, 'fit_04')};
others = setdiff(1:9, ref);
ok2 = true;
for j = others
    v = find(strcmp(sel.fit_names, sprintf('fit_%02d', j)));
    for a = 1:K
        ok2 = ok2 && sel.matched(a, v) == perms(j, inv_ref(a));
    end
end
res(end + 1, :) = {'F2 known state order recovered in all 8 other fits', double(ok2), ok2};
true_state = inv_ref;                                   % per reported state
exp_rec = true_state < 3;
exp_pass = 8 * ones(1, K); exp_pass(true_state == 3) = 3;
ok3 = isequal(sel.recurs, exp_rec) && isequal(sel.n_pass, exp_pass) && sel.need == 5;
res(end + 1, :) = {sprintf('F3 recurrence: n_pass %s (expected %s), need %d', mat2str(sel.n_pass), ...
    mat2str(exp_pass), sel.need), double(ok3), ok3};
csv = fileread(fullfile(fits, 'selection', 'fits.csv'));
ok4 = sel.n_valid == 9 && ~isempty(strfind(csv, 'fit_10,')) && ~isempty(regexp(csv, 'fit_10,[0-9a-f]+,[^,]+,0,fit_02', 'once')); %#ok<STREMP>
res(end + 1, :) = {'F4 copy of fit_02 counted once (9 valid)', sel.n_valid, ok4};
ok5 = sel.switching_rate_tested == 1 && sel.max_FO_tested == 0;
res(end + 1, :) = {'F5 switching rate tested, max FO not', double(ok5), ok5};
sc_ref = fullfile(fits, 'fit_04', folder, 'inference_scope.csv');
others_scope = 0;
for i = [1:3 5:10]
    others_scope = others_scope + exist(fullfile(fits, sprintf('fit_%02d', i), folder, 'inference_scope.csv'), 'file');
end
txt = fileread(sc_ref);
ok6 = exist(sc_ref, 'file') == 2 && others_scope == 0 && ~isempty(strfind(txt, sel.fit_id)); %#ok<STREMP>
res(end + 1, :) = {'F6 inference_scope.csv only in the reported fit (stale copy removed)', others_scope, ok6};
dmax = 0;
for j = others
    v = find(strcmp(sel.fit_names, sprintf('fit_%02d', j)));
    dmax = max(dmax, abs(sum(Tvec .* sel.agreement(:, v)) / sum(Tvec) - sel.similarity(v)));
end
res(end + 1, :) = {'F7 T-weighted per-mouse agreement = getGammaSimilarity (max diff)', dmax, dmax < 1e-12};

sel2 = select_reported_fit(fits);                       % same selection: accepted
f = fullfile(fits, 'selection', 'selected_fit.txt');
t0 = fileread(f);
fid = fopen(f, 'w'); fprintf(fid, '%s', strrep(t0, 'max_FO_tested: 0', 'max_FO_tested: 1')); fclose(fid);
ok8 = strcmp(sel2.fit_id, sel.fit_id) && local_fails(@() select_reported_fit(fits), 'already records');
fid = fopen(f, 'w'); fprintf(fid, '%s', t0); fclose(fid);
res(end + 1, :) = {'F8 same selection accepted on rerun; a different one refused', double(ok8), ok8};

rec = fullfile(fits, 'rule_sha256.txt');
r0 = fileread(rec);
fid = fopen(rec, 'w'); fprintf(fid, 'sha256: %s\n', repmat('0', 1, 64)); fclose(fid);
ok9 = local_fails(@() select_reported_fit(fits), 'differs from the version recorded');
fid = fopen(rec, 'w'); fprintf(fid, '%s', r0); fclose(fid);
res(end + 1, :) = {'F9 changed rule file refused', double(ok9), ok9};

few = fullfile(sig, 'few_fits');
mkdir(few);
for i = 1:7, copyfile(fullfile(fits, sprintf('fit_%02d', i)), fullfile(few, sprintf('fit_%02d', i))); end
mkdir(fullfile(few, 'fit_08', folder));                                 % unfinished candidate
copyfile(fullfile(fits, 'fit_08', folder, 'hmmmar_results.mat'), fullfile(few, 'fit_08', folder));
local_record_rule(rule, few);
ok10 = local_fails(@() select_reported_fit(few), '7 valid fits; the rule needs at least 8');
res(end + 1, :) = {'F10 fewer than 8 valid fits refused (7 + one unfinished)', double(ok10), ok10};

P = sel.previous;
inv_pp = zeros(1, K); inv_pp(pp) = 1:K; inv_pc = zeros(1, K); inv_pc(pc) = 1:K;
ok11 = numel(P) == 2 && strcmp(P(1).label, 'previous_refit') && strcmp(P(2).label, 'confirmatory_full_fit') ...
    && isnan(P(1).sim) && ~isempty(strfind(P(1).fit_id, 'not applicable')) ... %#ok<STREMP>
    && isequal(P(2).assig, pc(true_state)) && P(2).sim > 0.9;
res(end + 1, :) = {'F11 confirmation fit compared (order recovered); 4 Oct refit: not applicable', numel(P), ok11};

rep = fileread(fullfile(fits, 'selection', 'selection_report.txt'));
ok12 = strcmp(sel.reported_fit, 'fit_04') && ~any(strcmp(sel.fit_names, 'fit_11')) ...
    && ~isempty(strfind(rep, 'Not a candidate (rule item 2: fit_01-fit_10 only): fit_11')) ...
    && isempty(strfind(csv, 'fit_11')); %#ok<STREMP>
res(end + 1, :) = {'F12 fit_11 (lowest free energy, outside fit_01-10) not a candidate', double(ok12), ok12};

norec = fullfile(sig, 'no_recurring_fits');
mkdir(norec);
for i = 1:9
    G = zeros(S * T, K); G(:, perms(i, :)) = max(Gb + 0.02 * randn(size(Gb)), 0.001); G = G ./ sum(G, 2);
    fc = cell(1, K);
    for k = 1:K, fc{k} = local_randcorr(n); end                       % no state keeps its FC
    local_write_fit(fullfile(norec, sprintf('fit_%02d', i), folder), G, fc, fe(i), ...
        sw0 + 0.002 * randn(S, 1), 0.3 + 0.6 * rand(S, 1), Tvec, K);
end
local_record_rule(rule, norec);
s13 = select_reported_fit(norec);
ok13 = ~any(s13.recurs) && s13.switching_rate_tested == 0 && s13.max_FO_tested == 0;
res(end + 1, :) = {'F13 no recurring state: switching rate not tested although it agrees', ...
    double(ok13), ok13};

% F14: model.env of another model
wrong = fullfile(sig, 'wrong_model_fits');
mkdir(wrong);
for i = 1:9, copyfile(fullfile(fits, sprintf('fit_%02d', i)), fullfile(wrong, sprintf('fit_%02d', i))); end
local_record_rule(rule, wrong);
fid = fopen(fullfile(wrong, 'model.env'), 'w'); fprintf(fid, 'HMM_K=3\nHMM_LOWRANK=0\nHMM_PCA=5\n'); fclose(fid);
ok14 = local_fails(@() select_reported_fit(wrong), 'is not the model in hmm_config.m');
res(end + 1, :) = {'F14 model.env of another model refused', double(ok14), ok14};

% F15: Gaussian PCA model, default fits folder
setenv('HMM_LOWRANK', '0'); setenv('HMM_PCA', '5');
pfolder = 'hmmmar_simple_k3_pca5';
pfits = fullfile(sig, 'hmm_k3_pca5_fits');
mkdir(pfits);
for i = 1:9
    G = zeros(S * T, K); G(:, perms(i, :)) = max(Gb + 0.02 * randn(size(Gb)), 0.001); G = G ./ sum(G, 2);
    fc = cell(1, K);
    for k = 1:K, fc{perms(i, k)} = local_noisy(fcb{k}, 0.02); end
    local_write_fit(fullfile(pfits, sprintf('fit_%02d', i), pfolder), G, fc, fe(i), ...
        sw0 + 0.002 * randn(S, 1), 0.3 + 0.6 * rand(S, 1), Tvec, K, 5);
end
local_record_rule(rule, pfits);
s15 = select_reported_fit();
pbad = fullfile(sig, 'pca_bad');
copyfile(pfits, pbad);
rmdir(fullfile(pbad, 'selection'), 's');
local_write_fit(fullfile(pbad, 'fit_03', pfolder), G, fc, -59, sw0, rand(S, 1), Tvec, K, 6);
ok15 = strcmp(s15.reported_fit, 'fit_04') && all(s15.recurs) && s15.n_valid == 9 ...
    && exist(fullfile(pfits, 'selection', 'selected_fit.txt'), 'file') == 2 ...
    && local_fails(@() select_reported_fit(pbad), 'options.pca');
res(end + 1, :) = {'F15 Gaussian PCA model: default hmm_k3_pca5_fits, selected; other pca refused', ...
    double(ok15), ok15};
setenv('HMM_LOWRANK', '5'); setenv('HMM_PCA', '0');

results = cell2struct(res, {'check', 'value', 'pass'}, 2);
fprintf('\n');
for i = 1:size(res, 1)
    fprintf('%-80s %10.4g  %s\n', res{i, 1}, res{i, 2}, local_yesno(res{i, 3}));
end
fprintf('\n%d passed, %d failed. Work folder: %s\n', sum([res{:, 3}]), sum(~[res{:, 3}]), out_dir);
end


function local_write_fit(md, G, fc, fe, sw, mfo, Tvec, K, pca)
% pca > 0: a Gaussian model on PCA-reduced data (options.pca); else HMM-PCA rank 5
if nargin < 9, pca = 0; end
if ~exist(md, 'dir'), mkdir(md); end
S = numel(Tvec); offs = [0; cumsum(Tvec)];
FO = zeros(S, K);
for s = 1:S, FO(s, :) = mean(G(offs(s) + 1:offs(s + 1), :), 1); end
Gamma = G; fehist = [fe + 10, fe + 2, fe]; fit_id = hmm_fit_id(G); %#ok<NASGU>
GrpData = struct('FO', FO, 'switchingRate', sw, 'maxFO', mfo, 'Tvec', Tvec); %#ok<NASGU>
subject_ids = (1:S)'; %#ok<NASGU>
options = struct('K', K, 'zeromean', 1, 'covtype', 'full', 'order', 0, ...
    'DirichletDiag', 10, 'initrep', 3, 'initcyc', 25, 'cyc', 1000, 'useParallel', 0, ...
    'standardise', 1, 'inittype', 'hmmmar');
if pca > 0, options.pca = pca; else, options.lowrank = 5; end %#ok<STRNU>
save('-v7', fullfile(md, 'hmmmar_results.mat'), 'Gamma', 'fehist', 'fit_id', 'GrpData', ...
    'subject_ids', 'options');
state_fc = fc; %#ok<NASGU>
save('-v7', fullfile(md, 'raw_hmm_state_fc.mat'), 'state_fc', 'fit_id');
fid = fopen(fullfile(md, 'fit_id.txt'), 'w'); fprintf(fid, '%s\n', fit_id); fclose(fid);
end


function local_record_rule(rule, fits)
% as run_hmm_fits_array.sbatch: the rule's SHA-256 and a copy of the model file
% (here the model set in the environment)
[st, out] = system(sprintf('sha256sum "%s"', rule));
assert(st == 0, 'sha256sum failed');
fid = fopen(fullfile(fits, 'rule_sha256.txt'), 'w');
fprintf(fid, 'sha256: %s\nrecorded: test\n', strtok(out));
fclose(fid);
fid = fopen(fullfile(fits, 'model.env'), 'w');
fprintf(fid, '# test model\nHMM_K=%s\nHMM_LOWRANK=%s\nHMM_PCA=%s\n', getenv('HMM_K'), ...
    getenv('HMM_LOWRANK'), getenv('HMM_PCA'));
fclose(fid);
end


function C = local_randcorr(n)
A = randn(n, 2 * n);
C = A * A';
d = sqrt(diag(C));
C = C ./ (d * d');
end


function C = local_noisy(C0, sd)
E = sd * randn(size(C0)); E = (E + E') / 2;
C = C0 + E;
C(1:size(C, 1) + 1:end) = 1;
end


function p = local_randperm(K)
[~, p] = sort(rand(1, K));
end


function tf = local_fails(f, text)
tf = false;
try
    f();
catch err
    tf = ~isempty(strfind(err.message, text)); %#ok<STREMP>
    if ~tf, fprintf('Unexpected error: %s\n', err.message); end
end
end


function s = local_yesno(tf)
if tf, s = 'PASS'; else, s = 'FAIL'; end
end
