function run_dcc_cpca_null()
% RUN_DCC_CPCA_NULL  Window-free dynamic-FC tests against stationary surrogates.
%
%   (1) DCC / GARCH on group-PCA components (default 25):
%       does correlation (DCC) or amplitude (GARCH) vary over time beyond
%       what a stationary process with the same spectra produces?
%   (2) Complex PCA (propagation; Bolt et al. 2022), in ROI space (primary)
%       and in group-PCA space (to check what the 25-D reduction keeps).
%
% Every subject is compared with its OWN surrogates (default 19 per null):
%   phase  multivariate phase randomisation (same phases for all regions):
%          keeps every auto- and cross-spectrum, so static FC and lag
%          structure are preserved exactly; removes non-stationarity
%   iaaft  multivariate IAAFT: also keeps each region's value distribution.
%          NOT a valid null for DCC on band-limited data (its rank step
%          leaks power into the empty band: stationary data gave z = 1.6-2
%          even after decimation), so DCC is tested against phase only;
%          IAAFT is used for GARCH and complex-PCA measures.
%
% Band-limited data: BOLD low-passed at 0.1 Hz but sampled at TR = 2 s has
% no power between 0.1 and 0.25 Hz, cannot be whitened, and DCC/GARCH then
% fit the smoothness as dynamics (a ~ 0.4 on stationary data). DCC/GARCH are
% therefore run on data decimated to just above twice the low-pass edge
% (TR 2 s -> 4 s), averaged over the polyphase components (odd and even
% volumes), so no data are discarded. Complex PCA uses the full rate.
%
% Which measures test what (column "role" in the outputs):
%   test        non-stationarity: dcc_a, dcc_LR, corr_fluct, garch_alpha,
%               garch_LR, vol_cv, cpca_*_env_cv (episodic propagation)
%   static      preserved EXACTLY by the phase null: cpca_*_var_expl,
%               cpca_*_freq_hz, cpca_*_lag_index. These describe (lagged)
%               connectivity and are for group comparisons, not for
%               non-stationarity claims.
%   diagnostic  dcc_b, dcc_persist, garch_persist (unidentified when the
%               corresponding a/alpha ~ 0)
%
% Runs serially, or split over a SLURM array (DFC_N_CHUNKS + array task id);
% each subject is checkpointed. A run without DFC_N_CHUNKS loads all
% checkpoints, fits any missing subject and writes the outputs.
%
% Environment overrides: DFC_DATA_DIR, DFC_N_SUBJECTS, DFC_TR, DFC_PCA_DIM,
% DFC_N_SURR, DFC_OUT_DIR, DFC_QUICK_TEST=1, DFC_N_CHUNKS.
%
% Requires (same folder): dfc_dcc_subject.m, dfc_analytic_signal.m,
% dfc_cpca_fit.m, dfc_cpca_subject_stats.m, hmm_phase_randomise.m,
% hmm_iaaft_surrogate.m. Base MATLAB only; also runs in Octave >= 7.

%% ========== Configuration ==========
cfg.data_dir = 'hmm_signals';
cfg.file_pattern = 'signal%d.txt';
cfg.n_subjects = 95;
cfg.TR = 2;                      % seconds (mouse); 3 for the human data
cfg.expected_T = 570;            % volumes per subject (warning only); [] = skip
cfg.out_dir = 'dfc_dcc_cpca';
cfg.pca_dim = 25;                % DCC space = the HMM primary representation
cfg.ar_order = 2;                % AR prewhitening before GARCH/DCC
cfg.cpca_spaces = {'roi', 'pca'};% complex PCA: ROI space (primary) + PCA check
cfg.n_comp_cpca = 3;
cfg.n_cycle_bins = 20;
cfg.null_types = {'phase', 'iaaft'};
cfg.n_surr = 19;                 % per subject and null (p resolution 0.05)
cfg.n_replica = 1;               % stationary replicas per subject for the
                                 % calibrated phase-null test (0 = off)
cfg.seed = 123;
cfg.lowpass_hz = 0.1;            % upper edge of the preprocessing band-pass
cfg.dcc_decimate = [];           % [] = auto: largest factor keeping the new
                                 % Nyquist >= 1.1 x lowpass_hz (mouse: 2)

cfg.data_dir = local_env_str('DFC_DATA_DIR', cfg.data_dir);
cfg.n_subjects = local_env_num('DFC_N_SUBJECTS', cfg.n_subjects);
cfg.TR = local_env_num('DFC_TR', cfg.TR);
cfg.pca_dim = local_env_num('DFC_PCA_DIM', cfg.pca_dim);
cfg.n_surr = local_env_num('DFC_N_SURR', cfg.n_surr);
cfg.n_replica = local_env_num('DFC_N_REPLICA', cfg.n_replica);
cfg.out_dir = local_env_str('DFC_OUT_DIR', cfg.out_dir);
cfg.lowpass_hz = local_env_num('DFC_LOWPASS', cfg.lowpass_hz);
cfg.dcc_decimate = local_env_num('DFC_DECIMATE', cfg.dcc_decimate);
if local_env_flag('DFC_QUICK_TEST')
    cfg.n_surr = 2;
    cfg.out_dir = [cfg.out_dir '_quicktest'];
    fprintf('*** QUICK TEST MODE (2 surrogates per null) ***\n');
end
n_chunks = local_env_num('DFC_N_CHUNKS', 1);
chunk_mode = n_chunks > 1;
chunk_idx = 1;
if chunk_mode
    chunk_idx = local_env_num('SLURM_ARRAY_TASK_ID', NaN);
    if isnan(chunk_idx)
        error('DFC_N_CHUNKS is set but SLURM_ARRAY_TASK_ID is not.');
    end
end

%% ========== Data ==========
[Xraw, ids, groups, n_roi] = local_load(cfg);
S = numel(Xraw);
if cfg.pca_dim >= n_roi
    error('pca_dim (%d) must be smaller than the number of ROIs (%d).', cfg.pca_dim, n_roi);
end
local_mkdir(cfg.out_dir);
subj_dir = fullfile(cfg.out_dir, 'subjects');
local_mkdir(subj_dir);

%% ========== Group PCA and group complex PCA (deterministic) ==========
Xs = cellfun(@local_standardise, Xraw, 'UniformOutput', false);
if isempty(cfg.dcc_decimate)
    cfg.dcc_decimate = max(1, floor((1 / (2 * cfg.TR)) / (1.1 * cfg.lowpass_hz)));
end
local_spectral_check(Xs, cfg);
[A, pca_expl] = local_group_pca(Xs, cfg.pca_dim);
fprintf('Group PCA: %d components explain %.1f%% of variance.\n', ...
    cfg.pca_dim, 100 * sum(pca_expl));
G = struct();
for q = 1:numel(cfg.cpca_spaces)
    sp = cfg.cpca_spaces{q};
    Zs = cellfun(@(x) dfc_analytic_signal(local_space(x, sp, A)), Xs, ...
        'UniformOutput', false);
    G.(sp) = dfc_cpca_fit(Zs, cfg.n_comp_cpca);
    fprintf('Complex PCA (%s space): first %d components explain %s%%.\n', ...
        sp, cfg.n_comp_cpca, mat2str(round(1000 * G.(sp).explained.') / 10));
end
[names, roles] = local_metric_names(cfg);
M = numel(names);

sig.ids = ids(:).';
sig.n_roi = n_roi;
sig.pca_dim = cfg.pca_dim;
sig.ar_order = cfg.ar_order;
sig.spaces = cfg.cpca_spaces;
sig.n_comp = cfg.n_comp_cpca;
sig.nulls = cfg.null_types;
sig.n_surr = cfg.n_surr;
sig.seed = cfg.seed;
sig.TR = cfg.TR;
sig.dcc_decimate = cfg.dcc_decimate;
sig.code_version = '2026-09-29c';   % bump when per-subject computations change
sig_rep = sig;                        % replicas live in separate checkpoints
sig_rep.n_replica = cfg.n_replica;
sig_rep.replica_version = 'r1';
sig.names = names;

%% ========== Per-subject analyses (this task's share) ==========
if chunk_mode
    mine = find(mod((1:S) - 1, n_chunks) == chunk_idx - 1);
    fprintf('\nChunk %d of %d: %d subjects.\n', chunk_idx, n_chunks, numel(mine));
else
    mine = 1:S;
end
results = cell(S, 1);
results_rep = cell(S, 1);
for s = mine(:).'
    ck = fullfile(subj_dir, sprintf('sub%03d.mat', ids(s)));
    res = local_load_ck(ck, sig);
    if isempty(res)
        t0 = tic;
        res = local_subject(Xraw{s}, ids(s), A, G, cfg, M);
        local_save_ck(ck, res, sig);
        fprintf('subject %3d (%s): %.1f min\n', ids(s), groups{s}, toc(t0) / 60);
    else
        fprintf('subject %3d (%s): checkpoint reused\n', ids(s), groups{s});
    end
    results{s} = res;
    if cfg.n_replica > 0
        ckr = fullfile(subj_dir, sprintf('sub%03d_replica.mat', ids(s)));
        rr = local_load_ck(ckr, sig_rep);
        if isempty(rr)
            t0 = tic;
            rr = local_replicas(Xraw{s}, ids(s), A, G, cfg, M);
            local_save_ck(ckr, rr, sig_rep);
            fprintf('subject %3d (%s): replicas %.1f min\n', ids(s), groups{s}, toc(t0) / 60);
        end
        results_rep{s} = rr;
    end
end
if chunk_mode
    fprintf('Chunk %d done. Run without DFC_N_CHUNKS to write the outputs.\n', chunk_idx);
    return;
end

%% ========== Collect ==========
n_null = numel(cfg.null_types);
Obs = NaN(S, M);
Sur = NaN(S, cfg.n_surr, M, n_null);
for s = 1:S
    Obs(s, :) = results{s}.obs;
    for n = 1:n_null
        Sur(s, :, :, n) = results{s}.surr{n};
    end
end
% Replica z: each replica scored against its own phase surrogates exactly as
% the observed data are; averaged over replicas.
Zrep = NaN(S, M);
if cfg.n_replica > 0
    for s = 1:S
        zr = NaN(cfg.n_replica, M);
        for q = 1:cfg.n_replica
            v = results_rep{s}.surr{q};
            mu = mean(v, 1);
            sd = std(v, 0, 1);
            ok = sd > 1e-12 * max(1, abs(mu));
            zr(q, ok) = (results_rep{s}.obs(q, ok) - mu(ok)) ./ sd(ok);
        end
        Zrep(s, :) = mean(zr, 1);
    end
end

%% ========== Subject-level table (long format) ==========
rows = S * M;
col_sid = zeros(rows, 1);
col_grp = cell(rows, 1);
col_met = cell(rows, 1);
col_role = cell(rows, 1);
col_obs = zeros(rows, 1);
null_cols = cell(1, 4 * n_null);
null_hdr = cell(1, 4 * n_null);
for n = 1:n_null
    null_hdr(4 * n - 3:4 * n) = strcat(cfg.null_types{n}, {'_mean', '_sd', '_z', '_p'});
    for j = 1:4
        null_cols{4 * (n - 1) + j} = zeros(rows, 1);
    end
end
r = 0;
Z = NaN(S, M, n_null);
P = NaN(S, M, n_null);
for s = 1:S
    for m = 1:M
        r = r + 1;
        col_sid(r) = ids(s);
        col_grp{r} = groups{s};
        col_met{r} = names{m};
        col_role{r} = roles{m};
        col_obs(r) = Obs(s, m);
        for n = 1:n_null
            v = Sur(s, :, m, n);
            mu = mean(v);
            sd = std(v);
            if sd > 1e-12 * max(1, abs(mu))
                z = (Obs(s, m) - mu) / sd;
            else
                z = NaN;           % surrogates identical to observed (static measure)
            end
            if all(isnan(v))
                p = NaN;               % not computed for this null (e.g. DCC vs IAAFT)
            else
                p = (1 + sum(v >= Obs(s, m) - 1e-12 * max(1, abs(Obs(s, m))))) / (numel(v) + 1);
            end
            Z(s, m, n) = z;
            P(s, m, n) = p;
            null_cols{4 * (n - 1) + 1}(r) = mu;
            null_cols{4 * (n - 1) + 2}(r) = sd;
            null_cols{4 * (n - 1) + 3}(r) = z;
            null_cols{4 * (n - 1) + 4}(r) = p;
        end
    end
end
col_zrep = reshape(Zrep.', [], 1);        % same subject-major order as rows
local_write_csv(fullfile(cfg.out_dir, 'dfc_subject_metrics.csv'), ...
    [{'subject_id', 'group', 'metric', 'role', 'observed'}, null_hdr, {'phase_replica_z'}], ...
    [{col_sid, col_grp, col_met, col_role, col_obs}, null_cols, {col_zrep}]);

%% ========== Calibrated test: observed z vs replica z (phase null) ==========
% For stationary data the observed z and the replica z have the same
% distribution, whatever bias the procedure has; so a one-sided Wilcoxon
% on (z_observed - z_replica) across subjects is a calibrated test.
if cfg.n_replica > 0
    ip = find(strcmp(cfg.null_types, 'phase'), 1);
    tm = find(strcmp(roles, 'test'));
    ugr = unique(groups);
    c_met = names(tm).';
    c_n = zeros(numel(tm), 1); c_zo = c_n; c_zr = c_n; c_d = c_n; c_p = c_n;
    c_g = cell(1, numel(ugr));
    for g = 1:numel(ugr)
        c_g{g} = zeros(numel(tm), 1);
    end
    for i = 1:numel(tm)
        m = tm(i);
        d = Z(:, m, ip) - Zrep(:, m);
        c_n(i) = sum(isfinite(d));
        c_zo(i) = local_median(Z(:, m, ip));
        c_zr(i) = local_median(Zrep(:, m));
        c_d(i) = local_median(d);
        c_p(i) = local_wilcoxon_greater(d);
        for g = 1:numel(ugr)
            c_g{g}(i) = local_median(d(strcmp(groups, ugr{g})));
        end
    end
    local_write_csv(fullfile(cfg.out_dir, 'dfc_group_tests_calibrated.csv'), ...
        [{'metric', 'n_subjects', 'median_z_observed', 'median_z_replica', ...
        'median_z_difference', 'wilcoxon_p_greater'}, strcat('median_zdiff_', ugr(:).')], ...
        [{c_met, c_n, c_zo, c_zr, c_d, c_p}, c_g]);
end

%% ========== Group-level tests ==========
ugroups = unique(groups);
gr = numel(ugroups);
tr = M * n_null;
t_met = cell(tr, 1); t_role = cell(tr, 1); t_null = cell(tr, 1);
t_n = zeros(tr, 1); t_mobs = zeros(tr, 1); t_msur = zeros(tr, 1);
t_mdiff = zeros(tr, 1); t_p = zeros(tr, 1); t_frac = zeros(tr, 1);
t_gz = cell(1, gr);
for g = 1:gr
    t_gz{g} = zeros(tr, 1);
end
r = 0;
for m = 1:M
    for n = 1:n_null
        r = r + 1;
        sm = mean(Sur(:, :, m, n), 2);
        d = Obs(:, m) - sm;
        d(abs(d) <= 1e-9 * max(1, abs(Obs(:, m)))) = 0;
        t_met{r} = names{m};
        t_role{r} = roles{m};
        t_null{r} = cfg.null_types{n};
        t_n(r) = sum(isfinite(d));
        t_mobs(r) = local_median(Obs(:, m));
        t_msur(r) = local_median(sm);
        t_mdiff(r) = local_median(d);
        t_p(r) = local_wilcoxon_greater(d);
        if all(isnan(P(:, m, n)))
            t_frac(r) = NaN;
        else
            % p = (1 + #surr >= obs)/(n_surr + 1): with 19 surrogates the
            % smallest attainable p is exactly 0.05, so the level-0.05 test
            % is p <= 0.05 (observed above every surrogate).
            pv = P(:, m, n);
            pv = pv(isfinite(pv));
            t_frac(r) = mean(pv <= 0.05 + 1e-12);
        end
        for g = 1:gr
            t_gz{g}(r) = local_median(Z(strcmp(groups, ugroups{g}), m, n));
        end
    end
end
local_write_csv(fullfile(cfg.out_dir, 'dfc_group_tests.csv'), ...
    [{'metric', 'role', 'null', 'n_subjects', 'median_observed', ...
    'median_surrogate', 'median_difference', 'wilcoxon_p_greater', ...
    'frac_subjects_p05'}, strcat('median_z_', ugroups(:).')], ...
    [{t_met, t_role, t_null, t_n, t_mobs, t_msur, t_mdiff, t_p, t_frac}, t_gz]);

%% ========== Group PCA and complex PCA descriptions ==========
local_write_csv(fullfile(cfg.out_dir, 'pca_explained.csv'), ...
    {'component', 'explained'}, {(1:cfg.pca_dim).', pca_expl(:)});

for q = 1:numel(cfg.cpca_spaces)
    sp = cfg.cpca_spaces{q};
    V = G.(sp).V;
    K = cfg.n_comp_cpca;
    fk = zeros(K, 1);
    for k = 1:K
        fk(k) = local_median(Obs(:, strcmp(names, sprintf('cpca_%s_c%d_freq_hz', sp, k))));
    end

    % Sanity check: refit on the first surrogate of every subject
    sim = NaN(K, n_null);
    expl_s = NaN(K, n_null);
    for n = 1:n_null
        Zsur = cell(S, 1);
        for s = 1:S
            Xsur = local_surrogate(Xraw{s}, cfg.null_types{n}, ids(s), 1, cfg);
            Zsur{s} = dfc_analytic_signal(local_space(local_standardise(Xsur), sp, A));
        end
        Gs = dfc_cpca_fit(Zsur, K);
        for k = 1:K
            sim(k, n) = max(abs(V(:, k)' * Gs.V));
            expl_s(k, n) = Gs.explained(k);
        end
    end
    hdr = {'component', 'explained', 'median_freq_hz'};
    cols = {(1:K).', G.(sp).explained(:), fk};
    for n = 1:n_null
        hdr = [hdr, {['refit_similarity_' cfg.null_types{n}], ...
            ['refit_explained_' cfg.null_types{n}]}]; %#ok<AGROW>
        cols = [cols, {sim(:, n), expl_s(:, n)}]; %#ok<AGROW>
    end
    local_write_csv(fullfile(cfg.out_dir, ['cpca_components_' sp '.csv']), hdr, cols);

    % Loadings (and ROI-space projection for the PCA space)
    local_write_loadings(fullfile(cfg.out_dir, ['cpca_loadings_' sp '.csv']), V, fk);
    if strcmp(sp, 'pca')
        Vr = A * V;
        Vr = Vr ./ sqrt(sum(abs(Vr) .^ 2, 1));
        local_write_loadings(fullfile(cfg.out_dir, 'cpca_loadings_pca_in_roi.csv'), Vr, fk);
    end

    % One cycle of each component: phase-binned average of the real signal
    for k = 1:K
        edges = linspace(-pi, pi, cfg.n_cycle_bins + 1);
        acc = zeros(cfg.n_cycle_bins, size(V, 1));
        cnt = zeros(cfg.n_cycle_bins, 1);
        for s = 1:S
            Xk = local_space(Xs{s}, sp, A);
            ph = angle(dfc_analytic_signal(Xk) * V(:, k));
            b = min(max(floor((ph + pi) / (2 * pi) * cfg.n_cycle_bins) + 1, 1), cfg.n_cycle_bins);
            for bb = 1:cfg.n_cycle_bins
                acc(bb, :) = acc(bb, :) + sum(Xk(b == bb, :), 1);
                cnt(bb) = cnt(bb) + sum(b == bb);
            end
        end
        cyc = acc ./ max(cnt, 1);
        if strcmp(sp, 'pca')
            cyc = cyc * A.';                 % back to ROI space
        end
        centres = (edges(1:end - 1) + edges(2:end)).' / 2;
        roi_hdr = arrayfun(@(i) sprintf('roi_%d', i), 1:size(cyc, 2), 'UniformOutput', false);
        local_write_csv(fullfile(cfg.out_dir, sprintf('cpca_cycle_%s_c%d.csv', sp, k)), ...
            [{'phase_bin_centre_rad'}, roi_hdr], ...
            [{centres}, num2cell(cyc, 1)]);
    end
end

%% ========== Console summary and config ==========
if cfg.n_replica > 0
    fprintf('\nCalibrated (phase null): observed z vs stationary-replica z\n');
    fprintf('%-26s %9s %9s %9s\n', 'metric', 'z obs', 'z repl', 'p (W+)');
    for i = 1:numel(tm)
        fprintf('%-26s %9.2f %9.2f %9.3g\n', c_met{i}, c_zo(i), c_zr(i), c_p(i));
    end
end
fprintf('\n%-26s %-10s %-6s %9s %9s\n', 'metric', 'role', 'null', 'med diff', 'p (W+)');
for rr = 1:tr
    if ~strcmp(t_role{rr}, 'diagnostic') && t_n(rr) > 0
        fprintf('%-26s %-10s %-6s %9.4f %9.3g\n', t_met{rr}, t_role{rr}, ...
            t_null{rr}, t_mdiff(rr), t_p(rr));
    end
end
fid = fopen(fullfile(cfg.out_dir, 'dfc_config.txt'), 'w');
fn = fieldnames(cfg);
for i = 1:numel(fn)
    v = cfg.(fn{i});
    if iscell(v), v = strjoin(v, ','); elseif isnumeric(v), v = mat2str(v); end
    fprintf(fid, '%s = %s\n', fn{i}, v);
end
fprintf(fid, 'subjects = %d\nregions = %d\n', S, n_roi);
fclose(fid);
fprintf('\nOutputs written to %s\n', cfg.out_dir);
end


%% ========================================================================
function rr = local_replicas(X, id, A, G, cfg, M)
% n_replica stationary replicas of X, each scored against its own phase
% surrogates with exactly the observed-data procedure.
rr.id = id;
rr.obs = NaN(cfg.n_replica, M);
rr.surr = cell(1, cfg.n_replica);
for q = 1:cfg.n_replica
    rng(cfg.seed + 5e6 + id * 100 + q, 'twister');
    R = dfc_stationary_replica(X);
    rr.obs(q, :) = local_stats(R, A, G, cfg, true);
    v = NaN(cfg.n_surr, M);
    for r = 1:cfg.n_surr
        rng(cfg.seed + 6e6 + id * 10000 + q * 100 + r, 'twister');
        v(r, :) = local_stats(hmm_phase_randomise(R, []), A, G, cfg, true);
    end
    rr.surr{q} = v;
end
end


function res = local_subject(X, id, A, G, cfg, M)
res.id = id;
res.obs = local_stats(X, A, G, cfg);
res.surr = cell(1, numel(cfg.null_types));
for n = 1:numel(cfg.null_types)
    v = NaN(cfg.n_surr, M);
    for r = 1:cfg.n_surr
        v(r, :) = local_stats(local_surrogate(X, cfg.null_types{n}, id, r, cfg), ...
            A, G, cfg, ~strcmp(cfg.null_types{n}, 'iaaft'));
    end
    res.surr{n} = v;
end
end


function Xsur = local_surrogate(X, type, id, r, cfg)
% Deterministic: depends only on subject id, null type and repetition.
n = find(strcmp(cfg.null_types, type));
rng(cfg.seed + id * 1000 + (n - 1) * 500 + r, 'twister');
if strcmp(type, 'iaaft')
    Xsur = hmm_iaaft_surrogate(X, []);
else
    Xsur = hmm_phase_randomise(X, []);
end
end


function v = local_stats(X, A, G, cfg, do_dcc)
if nargin < 5
    do_dcc = true;
end
Xs = local_standardise(X);
Y = Xs * A;
q = cfg.dcc_decimate;
v = zeros(1, 9);
for ph = 1:q                              % average over polyphase components
    o = dfc_dcc_subject(Y(ph:q:end, :), cfg.ar_order, do_dcc);
    v = v + [o.dcc_a, o.dcc_LR, o.corr_fluct, o.garch_alpha_mean, o.garch_LR, ...
        o.vol_cv, o.dcc_b, o.dcc_persist, o.garch_persist] / q;
end
for q = 1:numel(cfg.cpca_spaces)
    sp = cfg.cpca_spaces{q};
    st = dfc_cpca_subject_stats(dfc_analytic_signal(local_space(Xs, sp, A)), ...
        G.(sp).V, cfg.TR);
    for k = 1:cfg.n_comp_cpca
        v = [v, st.env_cv(k), st.var_expl(k), st.freq_hz(k)]; %#ok<AGROW>
    end
    v = [v, st.lag_index]; %#ok<AGROW>
end
end


function local_spectral_check(Xs, cfg)
% Share of spectral power above the low-pass edge and above the Nyquist
% frequency after decimation (the latter would alias into the band).
q = cfg.dcc_decimate;
nyq_new = 1 / (2 * cfg.TR * q);
above_lp = 0;
above_nyq = 0;
for s = 1:numel(Xs)
    T = size(Xs{s}, 1);
    P = abs(fft(Xs{s})) .^ 2;
    f = (0:T - 1).' / (T * cfg.TR);
    f = min(f, 1 / cfg.TR - f);          % fold to [0, Nyquist]
    tot = sum(P(:));
    above_lp = above_lp + sum(sum(P(f > cfg.lowpass_hz, :))) / tot / numel(Xs);
    above_nyq = above_nyq + sum(sum(P(f > nyq_new, :))) / tot / numel(Xs);
end
fprintf(['DCC/GARCH decimation x%d (TR %.1f s -> %.1f s). Power above %.3f Hz: ' ...
    '%.2f%%; above new Nyquist %.3f Hz: %.2f%%.\n'], q, cfg.TR, cfg.TR * q, ...
    cfg.lowpass_hz, 100 * above_lp, nyq_new, 100 * above_nyq);
if q > 1 && above_nyq > 0.02
    warning(['%.1f%% of power lies above the decimated Nyquist frequency and ' ...
        'would alias. Check the band-pass settings (lowpass_hz).'], 100 * above_nyq);
end
if above_lp > 0.10
    warning(['%.0f%% of power lies above lowpass_hz = %.3f Hz: the data do not look ' ...
        'low-passed there. Set DFC_LOWPASS to the actual filter edge.'], ...
        100 * above_lp, cfg.lowpass_hz);
end
end


function [names, roles] = local_metric_names(cfg)
names = {'dcc_a', 'dcc_LR', 'corr_fluct', 'garch_alpha', 'garch_LR', 'vol_cv', ...
    'dcc_b', 'dcc_persist', 'garch_persist'};
roles = [repmat({'test'}, 1, 6), repmat({'diagnostic'}, 1, 3)];
for q = 1:numel(cfg.cpca_spaces)
    sp = cfg.cpca_spaces{q};
    for k = 1:cfg.n_comp_cpca
        names = [names, {sprintf('cpca_%s_c%d_env_cv', sp, k), ...
            sprintf('cpca_%s_c%d_var_expl', sp, k), ...
            sprintf('cpca_%s_c%d_freq_hz', sp, k)}]; %#ok<AGROW>
        roles = [roles, {'test', 'static', 'static'}]; %#ok<AGROW>
    end
    names = [names, {sprintf('cpca_%s_lag_index', sp)}]; %#ok<AGROW>
    roles = [roles, {'static'}]; %#ok<AGROW>
end
end


function Y = local_space(Xs, sp, A)
if strcmp(sp, 'pca')
    Y = Xs * A;
else
    Y = Xs;
end
end


function Xs = local_standardise(X)
Xs = X - mean(X, 1);
sd = std(Xs, 0, 1);
sd(sd <= 0) = 1;
Xs = Xs ./ sd;
end


function [A, expl] = local_group_pca(Xs, d)
% Group PCA on concatenated standardised data (as HMM-MAR options.pca).
% Sign fixed so the largest-magnitude loading of each component is positive.
N = size(Xs{1}, 2);
C = zeros(N);
n = 0;
for s = 1:numel(Xs)
    C = C + Xs{s}' * Xs{s};
    n = n + size(Xs{s}, 1);
end
C = C / (n - 1);
C = (C + C') / 2;
[V, D] = eig(C);
[lam, o] = sort(diag(D), 'descend');
A = V(:, o(1:d));
for k = 1:d
    [~, im] = max(abs(A(:, k)));
    A(:, k) = A(:, k) * sign(A(im, k));
end
expl = lam(1:d) / sum(lam);
end


function [Xraw, ids, groups, n_roi] = local_load(cfg)
fprintf('Working directory: %s\n', pwd);
if ~exist(cfg.data_dir, 'dir')
    error(['Data folder "%s" not found from the working directory above. ' ...
        'Run sbatch from the folder that contains it, or set DFC_DATA_DIR ' ...
        'to its absolute path.'], cfg.data_dir);
end
if ~exist(fullfile(cfg.data_dir, sprintf(cfg.file_pattern, 1)), 'file')
    d = dir(fullfile(cfg.data_dir, '*.txt'));
    ex = strjoin({d(1:min(3, end)).name}, ', ');
    error(['%s not found in %s. File pattern is "%s"; .txt files present: %s'], ...
        sprintf(cfg.file_pattern, 1), cfg.data_dir, cfg.file_pattern, ex);
end
Xraw = {};
ids = [];
n_roi = [];
T_all = [];
for s = 1:cfg.n_subjects
    fn = fullfile(cfg.data_dir, sprintf(cfg.file_pattern, s));
    if ~exist(fn, 'file')
        warning('Missing subject file: %s', fn);
        continue;
    end
    X = local_read_matrix(fn);
    if isempty(X) || size(X, 1) < 50 || any(~isfinite(X(:)))
        warning('Invalid time series for subject %d; skipped.', s);
        continue;
    end
    if isempty(n_roi)
        n_roi = size(X, 2);
    elseif size(X, 2) ~= n_roi
        warning('Subject %d has %d ROIs (expected %d); skipped.', s, size(X, 2), n_roi);
        continue;
    end
    Xraw{end + 1, 1} = double(X); %#ok<AGROW>
    ids(end + 1, 1) = s; %#ok<AGROW>
    T_all(end + 1, 1) = size(X, 1); %#ok<AGROW>
end
if numel(Xraw) < 4
    error('Only %d valid subjects loaded.', numel(Xraw));
end
n_extra = 0;
while exist(fullfile(cfg.data_dir, sprintf(cfg.file_pattern, cfg.n_subjects + n_extra + 1)), 'file')
    n_extra = n_extra + 1;
end
if n_extra > 0
    warning('%d more signal file(s) exist beyond n_subjects = %d: check n_subjects.', ...
        n_extra, cfg.n_subjects);
end
fprintf('Loaded %d subjects, %d ROIs, %d-%d volumes (TR %.2f s = %.1f-%.1f min).\n', ...
    numel(Xraw), n_roi, min(T_all), max(T_all), cfg.TR, ...
    min(T_all) * cfg.TR / 60, max(T_all) * cfg.TR / 60);
if ~isempty(cfg.expected_T) && any(T_all ~= cfg.expected_T)
    warning('%d subject(s) do not have %d volumes.', sum(T_all ~= cfg.expected_T), cfg.expected_T);
end
if any(T_all < n_roi)
    warning('Some subjects have fewer volumes than ROIs: are files ROI x time?');
end

groups = repmat({'All'}, numel(ids), 1);
meta = fullfile(cfg.data_dir, 'meta_data.csv');
if exist(meta, 'file')
    [sid, grp] = local_read_meta(meta);
    for i = 1:numel(ids)
        j = find(sid == ids(i), 1);
        if isempty(j)
            error('No group in meta_data.csv for subject %d.', ids(i));
        end
        groups{i} = grp{j};
    end
else
    warning('No meta_data.csv: all subjects in one group.');
end
ug = unique(groups);
fprintf('Groups:');
for g = 1:numel(ug)
    fprintf(' %s (%d)', ug{g}, sum(strcmp(groups, ug{g})));
end
fprintf('\n');
end


function X = local_read_matrix(fn)
if exist('readmatrix', 'file') || exist('readmatrix', 'builtin')
    X = readmatrix(fn);
else
    X = dlmread(fn);
end
end


function [sid, grp] = local_read_meta(fn)
% Robust to a UTF-8 byte-order mark (Excel "CSV UTF-8"), Windows/Mac line
% endings, comma/semicolon/tab delimiters, quoted fields, header case and
% extra columns. Column names are matched after lower-casing and removing
% everything except letters, digits and underscores.
txt = fileread(fn);
if numel(txt) >= 3 && isequal(double(txt(1:3)), [239 187 191])
    txt = txt(4:end);                       % BOM read as three bytes
end
if ~isempty(txt) && double(txt(1)) == 65279
    txt = txt(2:end);                       % BOM decoded as U+FEFF
end
lines = regexp(txt, '\r\n|\n|\r', 'split');
lines = lines(~cellfun(@(l) isempty(strtrim(l)), lines));
if numel(lines) < 2
    error('meta_data.csv has no data rows.');
end
cands = {',', ';', sprintf('\t')};
[~, di] = max(cellfun(@(d) sum(lines{1} == d), cands));
delim = cands{di};
raw_header = strsplit(lines{1}, delim);
header = lower(regexprep(raw_header, '[^A-Za-z0-9_]', ''));
ci = find(strcmp(header, 'subject_id') | strcmp(header, 'subjectid'), 1);
cg = find(strcmp(header, 'group'), 1);
if isempty(ci) || isempty(cg)
    error(['meta_data.csv needs columns subject_id and group; header read as: ' ...
        '%s (delimiter "%s").'], strjoin(raw_header, ' | '), ...
        strrep(delim, sprintf('\t'), '\t'));
end
n = numel(lines) - 1;
sid = NaN(n, 1);
grp = cell(n, 1);
for i = 1:n
    parts = strsplit(lines{i + 1}, delim);
    parts = strtrim(regexprep(parts, '"', ''));
    if numel(parts) < max(ci, cg)
        error('meta_data.csv row %d has %d fields; expected at least %d.', ...
            i + 1, numel(parts), max(ci, cg));
    end
    sid(i) = str2double(parts{ci});
    grp{i} = parts{cg};
end
if any(isnan(sid))
    error('Non-numeric subject_id in meta_data.csv rows: %s', mat2str(find(isnan(sid)).' + 1));
end
end


function res = local_load_ck(ck, sig)
res = [];
if exist(ck, 'file')
    try
        L = load(ck);
        if isfield(L, 'sig') && isequal(L.sig, sig) && isfield(L, 'res')
            res = L.res;
        end
    catch
        res = [];
    end
end
end


function local_save_ck(ck, res, sig) %#ok<INUSD>
tmp = [ck '.tmp.mat'];
save(tmp, 'res', 'sig', '-v7');
movefile(tmp, ck, 'f');
end


function local_write_loadings(fn, V, fk)
[N, K] = size(V);
region = repmat((1:N).', K, 1);
comp = kron((1:K).', ones(N, 1));
mag = abs(V(:));
ph = angle(V(:));
lag = ph ./ (2 * pi * fk(comp));        % seconds, positive = later
local_write_csv(fn, {'region', 'component', 'magnitude', 'phase_rad', 'lag_sec'}, ...
    {region, comp, mag, ph, lag});
end


function p = local_wilcoxon_greater(d)
% One-sided Wilcoxon signed-rank test (median of d > 0): normal
% approximation with tie and continuity corrections; zeros dropped.
d = d(isfinite(d));
d = d(d ~= 0);
n = numel(d);
if n < 5
    p = NaN;
    return;
end
[s, idx] = sort(abs(d));
r = zeros(n, 1);
tie = 0;
i = 1;
while i <= n
    j = i;
    while j < n && s(j + 1) == s(i)
        j = j + 1;
    end
    r(idx(i:j)) = (i + j) / 2;
    t = j - i + 1;
    tie = tie + t ^ 3 - t;
    i = j + 1;
end
W = sum(r(d > 0));
mu = n * (n + 1) / 4;
sigma = sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie / 48);
z = (W - mu - 0.5) / sigma;
p = 0.5 * erfc(z / sqrt(2));
end


function m = local_median(x)
x = x(isfinite(x));
if isempty(x)
    m = NaN;
else
    m = median(x);
end
end


function local_write_csv(fn, header, cols)
fid = fopen(fn, 'w');
fprintf(fid, '%s\n', strjoin(header, ','));
n = numel(cols{1});
parts = cell(1, numel(cols));
for r = 1:n
    for c = 1:numel(cols)
        v = cols{c};
        if iscell(v)
            parts{c} = v{r};
        else
            parts{c} = sprintf('%.10g', v(r));
        end
    end
    fprintf(fid, '%s\n', strjoin(parts, ','));
end
fclose(fid);
end


function local_mkdir(d)
% Tolerates the race when many array tasks create the same folder at once.
if exist(d, 'dir')
    return;
end
try
    mkdir(d);
catch
end
if ~exist(d, 'dir')
    pause(1 + rand);
    try
        mkdir(d);
    catch
    end
end
if ~exist(d, 'dir')
    error('Could not create folder %s.', d);
end
end


function v = local_env_num(name, default)
s = strtrim(getenv(name));
if isempty(s)
    v = default;
else
    v = str2double(s);
    if isnan(v)
        error('Environment variable %s = "%s" is not a number.', name, s);
    end
end
end


function v = local_env_str(name, default)
v = strtrim(getenv(name));
if isempty(v)
    v = default;
end
end


function tf = local_env_flag(name)
tf = any(strcmpi(strtrim(getenv(name)), {'1', 'true', 'yes', 'on'}));
end
