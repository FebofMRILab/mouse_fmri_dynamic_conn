function cfg = hmm_config()
% HMM_CONFIG  One configuration for every step of the HMM workflow
% (run4e, run4d, run5, run5B). Edit the defaults here, or override any field
% for one run with an environment variable (name in brackets), e.g.
%   export HMM_K=2 HMM_LOWRANK=15 HMM_PCA=0
%
% Every step saves the configuration it used next to its outputs.
%
% Model: read from MODEL_PRIMARY.env in this folder (K, HMM-PCA rank, PCA
% dimension) when HMM_K / HMM_LOWRANK / HMM_PCA are not set. Two models are
% reported (FIT_SELECTION_RULE.md amendment 1, 27 Oct 2026; provenance D33):
%   primary    MODEL_PRIMARY.env    Gaussian HMM on 25 PCs (options.pca = 25), K = 3
%   secondary  MODEL_SECONDARY.env  HMM-PCA (options.lowrank = 15), K = 2
% The job scripts set the secondary through hipergator_tests/reported_model.sh,
% which exports HMM_K, HMM_LOWRANK and HMM_PCA from its file. Both use the
% options of the confirmatory full fits (run4c: 15 initialisations, no
% parallel pool). Until 27 Oct 2026 the default was HMM-PCA rank 15, K = 4,
% chosen on the eigenvariate signals (D16, D17), which no longer decides anything.
%
% Data: /blue/febo/share/Abeta_Age/rabies/signals holds hmm_signals/ and every
% HMM output folder (static/ sits next to it). HMM_OUT_BASE points elsewhere.
%
% A refit is not identical to an earlier fit: HMM-MAR reseeds MATLAB's random
% generator from the clock at every initialisation (train/initGamma_random.m:
% rng('default'); rng('shuffle')), so rng(seed) before hmmmar does not fix the
% result. The saved fit and its fit ID are the record;
% tests/check_against_confirmatory.m compares a new fit with the confirmatory one.

% ---- model ----
model = read_model_file(fullfile(fileparts(mfilename('fullpath')), 'MODEL_PRIMARY.env'));
cfg.K = env_num('HMM_K', model.HMM_K);                  % number of states
cfg.lowrank = env_num('HMM_LOWRANK', model.HMM_LOWRANK); % HMM-PCA rank (options.lowrank; 0 = off)
cfg.pca = env_num('HMM_PCA', model.HMM_PCA);            % PCA before the HMM (options.pca; 0 = off)
cfg.TR_sec = env_num('HMM_TR', 2);
cfg.seed = env_num('HMM_SEED', 20261001 + cfg.K);   % as run4c; recorded, but see above
cfg.initrep = env_num('HMM_INITREP', 15);           % = run4c initrep_final
cfg.dual = env_num('HMM_DUAL', 1) == 1;         % subject-level dual estimation (extra output)
if cfg.lowrank > 0 && cfg.pca > 0
    error('Set either HMM_LOWRANK (HMM-PCA) or HMM_PCA (PCA before the HMM), not both.');
end

% ---- data and folders ----
cfg.n_subjects = env_num('HMM_N_SUBJECTS', 95);
cfg.out_base = env_str('HMM_OUT_BASE', '/blue/febo/share/Abeta_Age/rabies/signals');
cfg.data_dir = env_str('HMM_DATA_DIR', fullfile(cfg.out_base, 'hmm_signals'));
cfg.file_pattern = 'signal%d.txt';
cfg.model_dir = env_str('HMM_MODEL_DIR', fullfile(cfg.out_base, model_folder(cfg.K, cfg.lowrank, cfg.pca)));
% full fit of the same model from its confirmation (step 6: run4c_confirm_lowrank.m
% for HMM-PCA; step 6b: run4c_confirm_candidate.m for a Gaussian model), for
% tests/check_against_confirmatory.m and select_reported_fit.m; empty = not checked
cfg.confirm_gamma = env_str('HMM_CONFIRM_GAMMA', fullfile(cfg.out_base, confirm_gamma_file(cfg.K, cfg.lowrank, cfg.pca)));

% ---- toolboxes ----
cfg.hmmmar_dir = env_str('HMM_TOOLBOX', '/blue/febo/MATLAB/HMM-MAR-master');
cfg.bct_dir = env_str('HMM_BCT', '');           % empty = BCT already on the path

% ---- model comparison tables (run4d): every folder that exists ----
cfg.K_values = env_vec('HMM_K_VALUES', 2:10);   % 2 so a Supplement K = 2 fit is tabled
cfg.lowrank_values = env_vec('HMM_LOWRANK_VALUES', [15, 25, 50]);   % <out_base>/hmmmar_simple_k<K>_hmmpca<r>
cfg.pca_values = env_vec('HMM_PCA_VALUES', [0, 15, 25, 50]);        % <out_base>/hmmmar_simple_k<K>_pca<p>

% ---- subject x state networks (run5 and run5B use the same values) ----
cfg.graph_density = env_num('HMM_DENSITY', 0.15);
cfg.min_effective_n = env_num('HMM_MIN_NEFF', 83);   % 82 ROIs + 1; same for run5 and run5B

% ---- block bootstrap (run5B) ----
cfg.n_bootstrap = env_num('HMM_N_BOOT', 10000);
cfg.boot_seed = env_num('HMM_BOOT_SEED', 20260902);
cfg.block_length = env_num('HMM_BLOCK', NaN);        % NaN = ceil(T^(1/3)) per subject
cfg.min_bootstrap_success_fraction = 0.80;
cfg.max_resample_pairs = 500;
cfg.min_posterior_mass = env_num('HMM_MIN_MASS', NaN);   % NaN = not used
cfg.near_static_fo = 0.85;
end

function f = model_folder(K, lowrank, pca)
% Folder names as used so far: _hmmpca<r> for HMM-PCA fits, _pca<p> otherwise.
if lowrank > 0
    f = sprintf('hmmmar_simple_k%d_hmmpca%d', K, lowrank);
else
    f = sprintf('hmmmar_simple_k%d_pca%d', K, pca);
end
end

function f = confirm_gamma_file(K, lowrank, pca)
% Where run4c_confirm_lowrank.m (step 6) and run4c_confirm_candidate.m (step 6b)
% save their full fits: <root>/observed/gamma_<step 5 label>_K<K>.mat
if lowrank > 0
    f = fullfile('hmm_confirm_lowrank', 'observed', sprintf('gamma_lowrank_%d_K%d.mat', lowrank, K));
elseif pca > 0
    f = fullfile('hmm_confirm_candidate', 'observed', sprintf('gamma_gaussian_pca_%d_K%d.mat', pca, K));
else
    f = fullfile('hmm_confirm_candidate', 'observed', sprintf('gamma_gaussian_full_K%d.mat', K));
end
end

function m = read_model_file(path)
% NAME=integer lines (HMM_K, HMM_LOWRANK, HMM_PCA); '#' starts a comment line.
% The file is required: it is the one place the reported model is set.
if ~exist(path, 'file')
    error('hmm_config:model_file', 'No model file %s (the reported model is set there).', path);
end
m = struct();
lines = regexp(fileread(path), '\r?\n', 'split');
for i = 1:numel(lines)
    t = strtrim(lines{i});
    if isempty(t) || t(1) == '#', continue; end
    tok = regexp(t, '^(HMM_K|HMM_LOWRANK|HMM_PCA)=(\d+)$', 'tokens', 'once');
    if isempty(tok)
        error('hmm_config:model_file', 'Unreadable line in %s: "%s"', path, t);
    end
    m.(tok{1}) = str2double(tok{2});
end
need = {'HMM_K', 'HMM_LOWRANK', 'HMM_PCA'};
if ~all(isfield(m, need))
    error('hmm_config:model_file', '%s must set HMM_K, HMM_LOWRANK and HMM_PCA.', path);
end
end

function v = env_num(name, default)
s = getenv(name);
if isempty(s), v = default; else, v = str2double(s); end
end

function v = env_vec(name, default)
s = getenv(name);
if isempty(s), v = default; else, v = str2num(s); end %#ok<ST2NM>
end

function v = env_str(name, default)
s = getenv(name);
if isempty(s), v = default; else, v = s; end
end
