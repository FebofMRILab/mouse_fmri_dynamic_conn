function sel = select_reported_fit(fits_dir, out_dir)
% SELECT_REPORTED_FIT  Apply the pre-specified rule (FIT_SELECTION_RULE.md) to
% the independent fits made by hipergator_tests/run_hmm_fits_array.sbatch.
%
%   sel = select_reported_fit()        % <HMM_OUT_BASE>/hmm_k<K>_<pca<p>|hmmpca<r>>_fits
%                                      % (model: hmm_config.m, i.e. HMM_K / HMM_LOWRANK /
%                                      % HMM_PCA, else MODEL_PRIMARY.env)
%   sel = select_reported_fit(fits_dir, out_dir)
%
% What it does (rule items in brackets):
%   - candidates are fit_01 ... fit_10 only                             [2]
%   - reads every candidate's <model folder> that run4e finished; identical
%     fit IDs count once; stops with fewer than 8 valid fits            [3]
%   - reported fit = lowest final free energy, fehist(end); ties to the lower
%     fit number                                                        [4]
%   - matches every other valid fit to it with HMM-MAR's getGammaSimilarity [5]
%   - a state recurs if its pooled state FC (R.state_fc) correlates r >= 0.9
%     with the matched state in more than half of the other valid fits  [6]
%   - switching rate and max FO are tested only if their per-mouse values
%     correlate r >= 0.9 with the reported fit's in more than half of the
%     other valid fits                                                  [7]
%   - if no state recurs, nothing is tested                             [9]
%   (r is Pearson's correlation throughout)
%   - writes inference_scope.csv into the reported fit's model folder (read by
%     hmm_py), and the tables of item 11 into out_dir
%
% The rule constants are fixed here and in FIT_SELECTION_RULE.md; there is no
% environment override. The rule file must match the SHA-256 recorded by the
% array job before fitting. A selection already written is not replaced by a
% different one (delete out_dir by hand only if no group statistics were run).
%
% Published code called (HMM-MAR, github.com/OHBA-analysis/HMM-MAR):
%   getGammaSimilarity (with munkres): similarity and state assignment.
% Per-mouse agreement is the statistic of HMM-MAR's hmmtestretest.m (final
% loop: for each subject, the sum over states of sum_t min(Gamma_a, Gamma_b)
% divided by T), computed on the saved fits instead of refitting inside
% hmmtestretest. Its T-weighted mean over mice equals getGammaSimilarity.

R_MIN = 0.9;          % rule items 6 and 7
MIN_VALID = 8;        % rule item 3
% Rule item 11 compared the reported fit with the 4 Oct 2026 refit in
% <HMM_OUT_BASE>/<model folder>. Amendment 1 (27 Oct 2026): every earlier fit is
% of the eigenvariate signals, so that comparison is reported as not applicable.
PREVIOUS_REFIT = false;

here = fileparts(mfilename('fullpath'));
addpath(here);
cfg = hmm_config();
folder = local_model_folder(cfg);
if nargin < 1 || isempty(fits_dir)
    fits_dir = fullfile(cfg.out_base, [strrep(folder, 'hmmmar_simple_', 'hmm_') '_fits']);
end
if nargin < 2 || isempty(out_dir), out_dir = fullfile(fits_dir, 'selection'); end
if exist('getGammaSimilarity', 'file') ~= 2, addpath(genpath(cfg.hmmmar_dir)); end
if exist('getGammaSimilarity', 'file') ~= 2
    error('HMM-MAR (getGammaSimilarity) not found: %s', cfg.hmmmar_dir);
end

% ---- the rule has not changed since the fits were started ----
rule_file = fullfile(here, 'FIT_SELECTION_RULE.md');
rule_hash = local_file_sha256(rule_file);
rec_file = fullfile(fits_dir, 'rule_sha256.txt');
if ~exist(rec_file, 'file')
    error('No %s: the array job records the rule before fitting.', rec_file);
end
tok = regexp(fileread(rec_file), 'sha256: ([0-9a-f]{64})', 'tokens', 'once');
if isempty(tok) || ~strcmp(tok{1}, rule_hash)
    error('FIT_SELECTION_RULE.md differs from the version recorded before fitting (%s).', rec_file);
end
% ---- the fits are of the model in hmm_config.m, as copied before fitting ----
model_rec = fullfile(fits_dir, 'model.env');
if ~exist(model_rec, 'file')
    error('No %s: the array job copies the model file before fitting.', model_rec);
end
mrec = local_read_model_file(model_rec);
if mrec.HMM_K ~= cfg.K || mrec.HMM_LOWRANK ~= cfg.lowrank || mrec.HMM_PCA ~= cfg.pca
    error(['%s (K %d, lowrank %d, pca %d) is not the model in hmm_config.m ' ...
        '(K %d, lowrank %d, pca %d).'], model_rec, mrec.HMM_K, mrec.HMM_LOWRANK, mrec.HMM_PCA, ...
        cfg.K, cfg.lowrank, cfg.pca);
end

% ---- read the fits ----
d = dir(fullfile(fits_dir, 'fit_*'));
d = d([d.isdir]);
names = sort({d.name});
extra = names(cellfun('isempty', regexp(names, '^fit_(0[1-9]|10)$', 'once')));
names = setdiff(names, extra);                 % candidates: fit_01 ... fit_10 only (rule item 2)
F = struct('name', {}, 'dir', {}, 'fit_id', {}, 'fe', {}, 'G', {}, 'fc', {}, ...
    'FO', {}, 'sw', {}, 'mfo', {});
skipped = {};
for i = 1:numel(names)
    md = fullfile(fits_dir, names{i}, folder);
    need = {'hmmmar_results.mat', 'raw_hmm_state_fc.mat', 'fit_id.txt'};
    if ~all(cellfun(@(f) exist(fullfile(md, f), 'file') == 2, need))
        skipped{end + 1} = sprintf('%s: run4e did not finish', names{i}); %#ok<AGROW>
        continue;
    end
    A = load(fullfile(md, 'hmmmar_results.mat'), 'Gamma', 'fehist', 'fit_id', 'GrpData', ...
        'subject_ids', 'options');
    B = load(fullfile(md, 'raw_hmm_state_fc.mat'), 'state_fc', 'fit_id');
    G = double(A.Gamma);
    fid = hmm_fit_id(G);
    if ~strcmp(fid, strtrim(fileread(fullfile(md, 'fit_id.txt')))) || ~strcmp(fid, A.fit_id) ...
            || ~strcmp(fid, B.fit_id)
        error('%s: the files in %s do not carry one fit ID.', names{i}, md);
    end
    f = struct();
    f.name = names{i};
    f.dir = md;
    f.fit_id = fid;
    f.fe = A.fehist(end);
    f.G = G;
    f.fc = B.state_fc;
    f.FO = A.GrpData.FO;
    f.sw = A.GrpData.switchingRate(:);
    f.mfo = A.GrpData.maxFO(:);
    if isempty(F)
        Tvec = A.GrpData.Tvec(:);
        sid = double(A.subject_ids(:));
        opt0 = A.options;
    else
        if ~isequal(A.GrpData.Tvec(:), Tvec) || ~isequal(double(A.subject_ids(:)), sid)
            error('%s covers different mice or scan lengths from %s.', names{i}, F(1).name);
        end
        local_same_options(opt0, A.options, names{i});
    end
    if A.options.K ~= cfg.K || local_field(A.options, 'lowrank', 0) ~= cfg.lowrank ...
            || local_field(A.options, 'pca', 0) ~= cfg.pca ...
            || local_field(A.options, 'initrep', NaN) ~= cfg.initrep
        error('%s is not the model in hmm_config.m (K %d, lowrank %d, pca %d, initrep %d).', ...
            names{i}, cfg.K, cfg.lowrank, cfg.pca, cfg.initrep);
    end
    F(end + 1) = f; %#ok<AGROW>
end
if isempty(F), error('No finished fits in %s.', fits_dir); end

% identical fits count once (rule item 3)
ids = {F.fit_id};
dup = false(1, numel(F));
dup_of = repmat({''}, 1, numel(F));
for i = 2:numel(F)
    j = find(strcmp(ids(1:i - 1), ids{i}) & ~dup(1:i - 1), 1);
    if ~isempty(j), dup(i) = true; dup_of{i} = F(j).name; end
end
V = F(~dup);
nV = numel(V);
if nV < MIN_VALID
    error('%d valid fits; the rule needs at least %d. Rerun the missing array tasks.', nV, MIN_VALID);
end

% ---- reported fit: lowest final free energy (rule item 4) ----
fe = [V.fe];
[~, order] = sortrows([fe(:), (1:nV)']);
ref = order(1);
Rf = V(ref);
K = size(Rf.G, 2);
others = setdiff(1:nV, ref);
nO = numel(others);
need_n = floor(nO / 2) + 1;            % more than half of the other valid fits
offs = [0; cumsum(Tvec)];
S = numel(Tvec);
up = triu(true(size(Rf.fc{1}, 1)), 1);

% ---- match every other fit to the reported fit (rule items 5-7) ----
sim = NaN(1, nV); sim(ref) = 1;
r_fc = NaN(K, nV); r_tc = NaN(K, nV); r_fo = NaN(K, nV); matched = NaN(K, nV);
r_sw = NaN(1, nV); r_mfo = NaN(1, nV);
agree = NaN(S, nV);
for j = others
    [sim(j), assig] = getGammaSimilarity(Rf.G, V(j).G);
    assig = assig(:)';
    matched(:, j) = assig(:);
    for k = 1:K
        kk = assig(k);
        r_fc(k, j) = local_corr(Rf.fc{k}(up), V(j).fc{kk}(up));
        r_tc(k, j) = local_corr(Rf.G(:, k), V(j).G(:, kk));
        r_fo(k, j) = local_corr(Rf.FO(:, k), V(j).FO(:, kk));
    end
    for m = 1:S                         % hmmtestretest.m statistic, per mouse
        idx = offs(m) + 1:offs(m + 1);
        agree(m, j) = sum(sum(min(Rf.G(idx, :), V(j).G(idx, assig)))) / Tvec(m);
    end
    r_sw(j) = local_corr(Rf.sw, V(j).sw);
    r_mfo(j) = local_corr(Rf.mfo, V(j).mfo);
end
n_pass = sum(r_fc(:, others) >= R_MIN, 2);
recurs = n_pass >= need_n;
pass_sw = sum(r_sw(others) >= R_MIN) >= need_n && any(recurs);    % item 9: no recurring
pass_mfo = sum(r_mfo(others) >= R_MIN) >= need_n && any(recurs);   % state, no tests at all

% every pair of valid fits (descriptive, rule item 11)
pair = NaN(nV);
for a = 1:nV
    pair(a, a) = 1;
    for b = a + 1:nV
        pair(a, b) = getGammaSimilarity(V(a).G, V(b).G);
        pair(b, a) = pair(a, b);
    end
end

% ---- earlier fits, for description only (rule items 2 and 11) ----
prev = local_previous_fits(cfg, folder, Rf, Tvec, sid, up, PREVIOUS_REFIT);

% ---- a selection already made is not replaced by a different one ----
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
sel_file = fullfile(out_dir, 'selected_fit.txt');
scope_line = sprintf('recurring_states: %s; switching_rate_tested: %d; max_FO_tested: %d', ...
    strtrim(sprintf('%d ', find(recurs))), pass_sw, pass_mfo);
if exist(sel_file, 'file')
    old = fileread(sel_file);
    if isempty(strfind(old, ['fit_id: ' Rf.fit_id])) || isempty(strfind(old, scope_line)) %#ok<STREMP>
        error(['%s already records a different selection. The rule allows one selection; ' ...
            'delete %s by hand only if no group statistics have been run.'], sel_file, out_dir);
    end
end

% ---- write ----
logf = fopen(fullfile(out_dir, 'selection_report.txt'), 'w');
cleanup = onCleanup(@() fclose(logf));
say = @(varargin) local_say(logf, varargin{:});
say('Fit selection (FIT_SELECTION_RULE.md, sha256 %s)\n', rule_hash);
say('Fits folder: %s\nModel: %s (K %d, lowrank %d, pca %d)\n', fits_dir, folder, cfg.K, ...
    cfg.lowrank, cfg.pca);
say('Valid fits: %d', nV);
if any(dup), say(' (identical fits counted once: %s)', strjoin({F(dup).name}, ', ')); end
say('\n');
for i = 1:numel(skipped), say('Not used: %s\n', skipped{i}); end
for i = 1:numel(extra), say('Not a candidate (rule item 2: fit_01-fit_10 only): %s\n', extra{i}); end
say('Reported fit: %s (lowest final free energy %.6g; fit %s)\n', Rf.name, Rf.fe, Rf.fit_id);
say('getGammaSimilarity to the reported fit: median %.3f (range %.3f-%.3f) over %d fits\n', ...
    median(sim(others)), min(sim(others)), max(sim(others)), nO);
pv = pair(triu(true(nV), 1));
say('getGammaSimilarity between every pair of valid fits: median %.3f (range %.3f-%.3f)\n', ...
    median(pv), min(pv), max(pv));
say('A state recurs if state FC r >= %.2f in at least %d of %d other fits.\n', R_MIN, need_n, nO);
for k = 1:K
    say(['  state %d: FO %.3f; FC r >= %.2f in %d of %d (median r %.3f); median r of time course %.3f, ' ...
        'of FO across mice %.3f; %s\n'], k, mean(Rf.FO(:, k)), R_MIN, n_pass(k), nO, ...
        median(r_fc(k, others)), median(r_tc(k, others)), median(r_fo(k, others)), ...
        local_yesno(recurs(k), 'RECURS', 'does not recur'));
end
say('Switching rate: per-mouse r >= %.2f in %d of %d -> %s\n', R_MIN, sum(r_sw(others) >= R_MIN), ...
    nO, local_yesno(pass_sw, 'tested', 'not tested'));
say('Max FO: per-mouse r >= %.2f in %d of %d -> %s\n', R_MIN, sum(r_mfo(others) >= R_MIN), ...
    nO, local_yesno(pass_mfo, 'tested', 'not tested'));
if ~any(recurs), say('No state recurs: no group tests for this model (rule item 9).\n'); end
for p = 1:numel(prev)
    if isnan(prev(p).sim)
        say('%s: %s\n', prev(p).label, prev(p).fit_id);
    else
        say('%s (fit %s): getGammaSimilarity %.3f to the reported fit\n', prev(p).label, ...
            prev(p).fit_id, prev(p).sim);
    end
end

% fits.csv
fid = fopen(fullfile(out_dir, 'fits.csv'), 'w');
fprintf(fid, ['fit,fit_id,final_free_energy,valid,identical_to,reported,gamma_similarity_to_reported,' ...
    'r_switching_rate,r_max_FO,median_mouse_agreement\n']);
for i = 1:numel(F)
    v = find(strcmp({V.name}, F(i).name));
    if isempty(v) || dup(i)
        fprintf(fid, '%s,%s,%.10g,0,%s,0,,,,\n', F(i).name, F(i).fit_id, F(i).fe, dup_of{i});
    else
        fprintf(fid, '%s,%s,%.10g,1,,%d,%.6f,%s,%s,%s\n', F(i).name, F(i).fit_id, F(i).fe, v == ref, ...
            sim(v), local_num(r_sw(v)), local_num(r_mfo(v)), local_num(median(agree(:, v))));
    end
end
fclose(fid);

% state_recurrence.csv (long) and state_summary.csv
fid = fopen(fullfile(out_dir, 'state_recurrence.csv'), 'w');
fprintf(fid, ['reported_state,other_fit,matched_state,r_state_FC,r_state_time_course,' ...
    'r_FO_across_mice,passes\n']);
for k = 1:K
    for j = others
        fprintf(fid, '%d,%s,%d,%s,%s,%s,%d\n', k, V(j).name, matched(k, j), local_num(r_fc(k, j)), ...
            local_num(r_tc(k, j)), local_num(r_fo(k, j)), r_fc(k, j) >= R_MIN);
    end
end
fclose(fid);
fid = fopen(fullfile(out_dir, 'state_summary.csv'), 'w');
fprintf(fid, ['reported_state,FO_mean,n_other_fits,n_pass,n_needed,median_r_state_FC,min_r_state_FC,' ...
    'median_r_time_course,median_r_FO_across_mice,recurs,fit_id\n']);
for k = 1:K
    fprintf(fid, '%d,%.6f,%d,%d,%d,%s,%s,%s,%s,%d,%s\n', k, mean(Rf.FO(:, k)), nO, n_pass(k), need_n, ...
        local_num(median(r_fc(k, others))), local_num(min(r_fc(k, others))), ...
        local_num(median(r_tc(k, others))), local_num(median(r_fo(k, others))), recurs(k), Rf.fit_id);
end
fclose(fid);

% pairwise_gamma_similarity.csv, per_mouse_agreement.csv, previous_fits.csv
fid = fopen(fullfile(out_dir, 'pairwise_gamma_similarity.csv'), 'w');
fprintf(fid, 'fit_a,fit_b,gamma_similarity\n');
for a = 1:nV
    for b = a + 1:nV
        fprintf(fid, '%s,%s,%.6f\n', V(a).name, V(b).name, pair(a, b));
    end
end
fclose(fid);
fid = fopen(fullfile(out_dir, 'per_mouse_agreement.csv'), 'w');
fprintf(fid, 'subject_id,other_fit,agreement_with_reported\n');
for j = others
    for m = 1:S
        fprintf(fid, '%d,%s,%.6f\n', sid(m), V(j).name, agree(m, j));
    end
end
fclose(fid);
fid = fopen(fullfile(out_dir, 'previous_fits.csv'), 'w');
fprintf(fid, 'fit,fit_id,gamma_similarity,reported_state,matched_state,r_state_FC,r_state_time_course,r_FO_across_mice\n');
for p = 1:numel(prev)
    for k = 1:K
        fprintf(fid, '%s,"%s",%s,%d,%s,%s,%s,%s\n', prev(p).label, prev(p).fit_id, local_num(prev(p).sim), k, ...
            local_int(prev(p).assig(k)), local_num(prev(p).r_fc(k)), local_num(prev(p).r_tc(k)), local_num(prev(p).r_fo(k)));
    end
end
fclose(fid);

% inference_scope.csv: in out_dir and in the reported fit's folder only
scope_rows = cell(K + 2, 5);
for k = 1:K
    scope_rows(k, :) = {'state', sprintf('%d', k), recurs(k), n_pass(k), nO};
end
scope_rows(K + 1, :) = {'whole_fit', 'switching_rate', pass_sw, sum(r_sw(others) >= R_MIN), nO};
scope_rows(K + 2, :) = {'whole_fit', 'max_FO', pass_mfo, sum(r_mfo(others) >= R_MIN), nO};
for target = {out_dir, Rf.dir}
    fid = fopen(fullfile(target{1}, 'inference_scope.csv'), 'w');
    fprintf(fid, 'kind,item,tested,n_pass,n_other_fits,r_min,fit_id,rule_sha256\n');
    for r = 1:size(scope_rows, 1)
        fprintf(fid, '%s,%s,%d,%d,%d,%.2f,%s,%s\n', scope_rows{r, :}, R_MIN, Rf.fit_id, rule_hash);
    end
    fclose(fid);
end
for i = 1:numel(F)                      % no stale scope file in any other fit
    if ~strcmp(F(i).dir, Rf.dir) && exist(fullfile(F(i).dir, 'inference_scope.csv'), 'file')
        delete(fullfile(F(i).dir, 'inference_scope.csv'));
    end
end

fid = fopen(sel_file, 'w');
fprintf(fid, 'reported_fit: %s\nselected_base: %s\nmodel_dir: %s\nfit_id: %s\n%s\nrule_sha256: %s\n', ...
    Rf.name, fileparts(Rf.dir), Rf.dir, Rf.fit_id, scope_line, rule_hash);
fclose(fid);
say('Written to %s; inference_scope.csv also in %s\n', out_dir, Rf.dir);

sel = struct('reported_fit', Rf.name, 'model_dir', Rf.dir, 'fit_id', Rf.fit_id, ...
    'recurs', recurs(:)', 'n_pass', n_pass(:)', 'need', need_n, 'n_valid', nV, ...
    'switching_rate_tested', pass_sw, 'max_FO_tested', pass_mfo, 'similarity', sim, ...
    'pairwise', pair, 'r_fc', r_fc, 'matched', matched, 'agreement', agree, ...
    'fit_names', {{V.name}}, 'free_energy', fe, 'previous', prev);
end


function prev = local_previous_fits(cfg, folder, Rf, Tvec, sid, up, PREVIOUS_REFIT)
% The confirmation's full fit of the same model, if present, and (only while
% PREVIOUS_REFIT is true) the 4 Oct 2026 refit (description only).
prev = struct('label', {}, 'fit_id', {}, 'sim', {}, 'assig', {}, 'r_fc', {}, 'r_tc', {}, 'r_fo', {});
K = size(Rf.G, 2);
% Item 11: reported whatever the outcome, so a missing or mismatched file is said.
md = fullfile(cfg.out_base, folder);
if ~PREVIOUS_REFIT
    prev(end + 1) = local_missing('previous_refit', K, ['not applicable (FIT_SELECTION_RULE.md ' ...
        'amendment 1: no earlier fit of this model on the ROI-mean signals)']);
elseif exist(fullfile(md, 'hmmmar_results.mat'), 'file') && exist(fullfile(md, 'raw_hmm_state_fc.mat'), 'file')
    A = load(fullfile(md, 'hmmmar_results.mat'), 'Gamma', 'GrpData', 'subject_ids');
    B = load(fullfile(md, 'raw_hmm_state_fc.mat'), 'state_fc');
    if isequal(size(A.Gamma), size(Rf.G)) && isequal(double(A.subject_ids(:)), sid)
        prev(end + 1) = local_compare_prev('previous_refit', double(A.Gamma), A.GrpData.FO, ...
            B.state_fc, Rf, up, K);
    else
        prev(end + 1) = local_missing('previous_refit', K, ['not comparable (other mice or size): ' md]);
    end
else
    prev(end + 1) = local_missing('previous_refit', K, ['not found: ' md]);
end
if ~isempty(cfg.confirm_gamma) && exist(cfg.confirm_gamma, 'file')
    C = load(cfg.confirm_gamma);
    if isequal(size(C.Gamma), size(Rf.G)) && isequal(double(C.subject_ids(:)), sid) ...
            && isequal(double(C.T_sub(:)), Tvec)
        G = double(C.Gamma);
        FO = zeros(numel(Tvec), K);
        offs = [0; cumsum(Tvec)];
        for m = 1:numel(Tvec), FO(m, :) = mean(G(offs(m) + 1:offs(m + 1), :), 1); end
        prev(end + 1) = local_compare_prev('confirmatory_full_fit', G, FO, {}, Rf, up, K);
    else
        prev(end + 1) = local_missing('confirmatory_full_fit', K, ...
            ['not comparable (other mice or size): ' cfg.confirm_gamma]);
    end
else
    prev(end + 1) = local_missing('confirmatory_full_fit', K, ['not found: ' cfg.confirm_gamma]);
end
end


function p = local_missing(label, note_k, note)
p = struct('label', label, 'fit_id', note, 'sim', NaN, 'assig', NaN(1, note_k), ...
    'r_fc', NaN(1, note_k), 'r_tc', NaN(1, note_k), 'r_fo', NaN(1, note_k));
end


function p = local_compare_prev(label, G, FO, fc, Rf, up, K)
[s, assig] = getGammaSimilarity(Rf.G, G);
assig = assig(:)';
p = struct('label', label, 'fit_id', hmm_fit_id(G), 'sim', s, 'assig', assig, ...
    'r_fc', NaN(1, K), 'r_tc', NaN(1, K), 'r_fo', NaN(1, K));
for k = 1:K
    if ~isempty(fc), p.r_fc(k) = local_corr(Rf.fc{k}(up), fc{assig(k)}(up)); end
    p.r_tc(k) = local_corr(Rf.G(:, k), G(:, assig(k)));
    p.r_fo(k) = local_corr(Rf.FO(:, k), FO(:, assig(k)));
end
end


function local_same_options(a, b, name)
% The fits must differ only by their random start.
fields = {'K', 'lowrank', 'pca', 'zeromean', 'covtype', 'order', 'DirichletDiag', 'initrep', ...
    'initcyc', 'cyc', 'useParallel', 'standardise', 'inittype'};
for i = 1:numel(fields)
    if ~isequal(local_field(a, fields{i}, []), local_field(b, fields{i}, []))
        error('%s was fitted with a different options.%s.', name, fields{i});
    end
end
end


function v = local_field(s, f, default)
if isfield(s, f), v = s.(f); else, v = default; end
end


function r = local_corr(x, y)
x = double(x(:)); y = double(y(:));
ok = isfinite(x) & isfinite(y);
if sum(ok) < 3 || std(x(ok)) == 0 || std(y(ok)) == 0, r = NaN; return; end
c = corrcoef(x(ok), y(ok));
r = c(1, 2);
end


function m = local_read_model_file(path)
% NAME=integer lines (HMM_K, HMM_LOWRANK, HMM_PCA); same format and checks as
% hmm_config.m's read_model_file.
m = struct();
lines = regexp(fileread(path), '\r?\n', 'split');
for i = 1:numel(lines)
    t = strtrim(lines{i});
    if isempty(t) || t(1) == '#', continue; end
    tok = regexp(t, '^(HMM_K|HMM_LOWRANK|HMM_PCA)=(\d+)$', 'tokens', 'once');
    if isempty(tok), error('Unreadable line in %s: "%s"', path, t); end
    m.(tok{1}) = str2double(tok{2});
end
if ~all(isfield(m, {'HMM_K', 'HMM_LOWRANK', 'HMM_PCA'}))
    error('%s must set HMM_K, HMM_LOWRANK and HMM_PCA.', path);
end
end


function f = local_model_folder(cfg)
if cfg.lowrank > 0
    f = sprintf('hmmmar_simple_k%d_hmmpca%d', cfg.K, cfg.lowrank);
else
    f = sprintf('hmmmar_simple_k%d_pca%d', cfg.K, cfg.pca);
end
end


function h = local_file_sha256(path)
% SHA-256 of the file's bytes (same method as hmm_fit_id.m).
fid = fopen(path, 'r');
bytes = fread(fid, Inf, '*uint8');
fclose(fid);
if exist('OCTAVE_VERSION', 'builtin')
    h = hash('sha256', char(bytes(:)'));
else
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(bytes(:));
    dgst = typecast(md.digest(), 'uint8');
    h = lower(reshape(dec2hex(dgst, 2)', 1, []));
end
end


function s = local_num(x)
if isfinite(x), s = sprintf('%.6f', x); else, s = 'NaN'; end
end


function s = local_int(x)
if isfinite(x), s = sprintf('%d', x); else, s = 'NaN'; end
end


function s = local_yesno(tf, yes, no)
if tf, s = yes; else, s = no; end
end


function local_say(fid, varargin)
fprintf(1, varargin{:});
fprintf(fid, varargin{:});
end
