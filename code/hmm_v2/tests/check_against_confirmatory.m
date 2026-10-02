function T = check_against_confirmatory(model_dir, confirm_gamma, data_dir, hmmmar_dir, out_dir)
% CHECK_AGAINST_CONFIRMATORY  Does the fit run4e saved find the same states as
% the confirmatory full fit of the same model (run4c_confirm_lowrank.m)?
%
% The two are separate fits with the same options. HMM-MAR reseeds from the
% clock at every initialisation (train/initGamma_random.m), so they are not
% identical; this measures how close they are, with HMM-MAR's own state
% matching (getGammaSimilarity), and compares the matched states' occupancy and
% pooled state FC (hmm_subject_state_fc, as R.state_fc).
%
%   S = '/blue/febo/share/Abeta_Age/rabies/signals';
%   T = check_against_confirmatory(fullfile(S, 'hmmmar_simple_k4_hmmpca15'), ...
%       fullfile(S, 'hmm_confirm_lowrank/observed/gamma_lowrank_15_K4.mat'), ...
%       fullfile(S, 'hmm_signals'), '/blue/febo/MATLAB/HMM-MAR-master', 'confirm_check')
%
% Any argument left empty is taken from hmm_config.m.

here = fileparts(mfilename('fullpath'));
addpath(fileparts(here));
cfg = hmm_config();
if nargin < 1 || isempty(model_dir), model_dir = cfg.model_dir; end
if nargin < 2 || isempty(confirm_gamma), confirm_gamma = cfg.confirm_gamma; end
if nargin < 3 || isempty(data_dir), data_dir = cfg.data_dir; end
if nargin < 4 || isempty(hmmmar_dir), hmmmar_dir = cfg.hmmmar_dir; end
if nargin < 5 || isempty(out_dir), out_dir = fullfile(model_dir, 'check_against_confirmatory'); end
addpath(genpath(hmmmar_dir));
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
logf = fopen(fullfile(out_dir, 'check_against_confirmatory.txt'), 'w');
cleanup = onCleanup(@() fclose(logf));
say = @(varargin) fprintf_both(logf, varargin{:});

B = load(fullfile(model_dir, 'raw_hmm_results.mat'), 'R');
R = B.R;
C = load(confirm_gamma);                          % Gamma, T_sub, subject_ids (run4c)
G_new = vertcat(R.Gamma{:});
G_conf = double(C.Gamma);
say('Reported fit: %s (fit %s)\n', model_dir, hmm_fit_id(R.Gamma));
say('Confirmatory fit: %s (fit %s)\n', confirm_gamma, hmm_fit_id(G_conf));
say('Model of the reported fit: lowrank %d, pca %d, K %d\n', R.hmm.train.lowrank, ...
    R.hmm.train.pca, R.K);

assert(isequal(double(C.subject_ids(:)), double(R.subject_ids(:))), ...
    'The two fits include different mice.');
assert(isequal(double(C.T_sub(:)), double(R.Tvec(:))), 'The two fits have different scan lengths.');
assert(isequal(size(G_conf), size(G_new)), 'Gamma sizes differ.');

% ---- HMM-MAR state matching ----
[sim, assig] = getGammaSimilarity(G_conf, G_new);
say('getGammaSimilarity: %.4f (1 = identical up to state order)\n', sim);
say('Confirmatory state k = reported state assig(k): %s\n', mat2str(assig(:)'));

% ---- matched states: time courses, occupancy per mouse, pooled state FC ----
K = R.K;
S = numel(R.subject_ids);
Ts = cell(S, 1);
for s = 1:S
    Ts{s} = double(readmatrix(fullfile(data_dir, sprintf(cfg.file_pattern, R.subject_ids(s)))));
end
offs = [0; cumsum(R.Tvec(:))];
Gc = cell(S, 1);
for s = 1:S
    Gc{s} = G_conf(offs(s) + 1:offs(s + 1), :);
end
[~, ~, sc_conf] = hmm_subject_state_fc(Ts, Gc, R.hmm.train, 0);
fo_conf = getFractionalOccupancy(G_conf, R.Tvec, R.hmm.train, 2);
up = triu(true(R.n_roi), 1);
rows = cell(K, 7);
for k = 1:K
    kk = assig(k);
    Sk = zeros(R.n_roi);
    for s = 1:S, Sk = Sk + sc_conf{s, k}; end
    sd = sqrt(diag(Sk));
    fc_conf = Sk ./ (sd * sd');
    rows(k, :) = {k, kk, mean(fo_conf(:, k)), mean(R.FO_subj(:, kk)), ...
        corr(G_conf(:, k), G_new(:, kk)), corr(fo_conf(:, k), R.FO_subj(:, kk)), ...
        corr(fc_conf(up), R.state_fc{kk}(up))};
end
T = cell2table(rows, 'VariableNames', {'state_confirmatory', 'state_reported', ...
    'FO_confirmatory', 'FO_reported', 'r_state_time_course', 'r_FO_across_mice', ...
    'r_pooled_state_FC'});
T.fit_id_reported = repmat(string(hmm_fit_id(R.Gamma)), K, 1);
T.fit_id_confirmatory = repmat(string(hmm_fit_id(G_conf)), K, 1);
T.gamma_similarity = repmat(sim, K, 1);
writetable(T, fullfile(out_dir, 'state_matching.csv'));
disp(T(:, 1:7));
end

function fprintf_both(fid, varargin)
fprintf(varargin{:});
fprintf(fid, varargin{:});
end
