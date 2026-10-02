function cfg = hmm_config()
% HMM_CONFIG  One configuration for every step of the HMM workflow
% (run4e, run4d, run5, run5B). Edit the defaults here, or override any field
% for one run with an environment variable (name in brackets), e.g.
%   export HMM_K=4 HMM_LOWRANK=15
%
% Every step saves the configuration it used next to its outputs.
%
% Reported model: HMM-PCA (options.lowrank = 15), K = 4, with the options of
% the pre-registered confirmatory full fits (run4c_confirm_lowrank.m: same
% options, 15 initialisations, no parallel pool). Rank 15 was fixed before the
% confirmatory test; K = 4 is the only K at rank 15 that met its decision rule
% (CHANGES.md, "Reported model").
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
cfg.K = env_num('HMM_K', 4);                    % number of states
cfg.lowrank = env_num('HMM_LOWRANK', 15);       % HMM-PCA rank (options.lowrank; 0 = off)
cfg.pca = env_num('HMM_PCA', 0);                % PCA before the HMM (options.pca; 0 = off)
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
% confirmatory full fit of the same model (run4c_confirm_lowrank.m), for
% tests/check_against_confirmatory.m; empty = not checked
cfg.confirm_gamma = env_str('HMM_CONFIRM_GAMMA', fullfile(cfg.out_base, 'hmm_confirm_lowrank', 'observed', ...
    sprintf('gamma_lowrank_%d_K%d.mat', cfg.lowrank, cfg.K)));

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
