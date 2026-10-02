function R = hmm_fit_raw_timeseries(Ts_list, options, subject_ids)
% HMM_FIT_RAW_TIMESERIES  Fit a covariance-state HMM with HMM-MAR and summarise
% the fit with HMM-MAR's own analysis functions.
%
%   R = hmm_fit_raw_timeseries(Ts_list, options, subject_ids)
%
%   Ts_list      cell, one (time x ROI) matrix per subject
%   options      HMM-MAR options (must contain K)
%   subject_ids  optional, one id per subject
%
% Published code called (HMM-MAR, Vidaurre et al.; GPL-3;
% github.com/OHBA-analysis/HMM-MAR; record the commit used in the run log):
%   hmmmar                     fit; returns the state probabilities (Gamma)
%                              and the Viterbi path
%   getFuncConn                state FC, the model's own estimate (stored as
%                              R.state_fc_model)
%   getFractionalOccupancy     fractional occupancy per subject
%   getMaxFractionalOccupancy  maximum FO per subject: diagnoses states that
%                              describe whole subjects (Ahrends et al. 2022)
%   getSwitchingRate           switching rate per subject
%   getStateLifeTimes          visit durations, from the Viterbi path
%   getTransProbs              model transition probabilities between states
%
% Group state FC in ROI space (R.state_fc) is the Gamma-weighted scatter of the
% z-scored data pooled over subjects, the statistic HMM-MAR's state updates use
% (setxx.m; see hmm_subject_state_fc.m). For a model fitted after PCA
% (options.pca), getFuncConn returns this scatter projected onto the PCA
% components (rank = number of components); for a model fitted in ROI space
% the two agree up to HMM-MAR's prior. Both are stored.
%
% Our code only checks the inputs, splits the concatenated outputs per subject
% and stores them. Changes from the previous version are listed in CHANGES.md.

if nargin < 2
    error('Ts_list and options are required.');
end
if ~iscell(Ts_list) || isempty(Ts_list)
    error('Ts_list must be a nonempty cell array.');
end
if ~isstruct(options) || ~isfield(options, 'K')
    error('options must be a structure containing options.K.');
end
if exist('hmmmar', 'file') ~= 2
    error('hmmmar was not found on the MATLAB path.');
end

Ts_list = Ts_list(:);
n_input = numel(Ts_list);
if nargin < 3 || isempty(subject_ids)
    subject_ids = (1:n_input).';
end
subject_ids = subject_ids(:);
if numel(subject_ids) ~= n_input || numel(unique(subject_ids)) ~= n_input
    error('subject_ids must have one unique entry per subject.');
end

% ---- input checks: drop unusable subjects, as before ----
n_roi = [];
valid_subject = true(n_input, 1);
for s = 1:n_input
    X = Ts_list{s};
    if ~isnumeric(X) || ~ismatrix(X) || size(X, 1) < 2 || any(~isfinite(X(:)))
        warning('Subject ID %g has an empty, short or nonfinite matrix and is removed.', ...
            subject_ids(s));
        valid_subject(s) = false;
        continue;
    end
    if isempty(n_roi)
        n_roi = size(X, 2);
    elseif size(X, 2) ~= n_roi
        error('Subject ID %g has %d ROIs; expected %d.', subject_ids(s), size(X, 2), n_roi);
    end
    Ts_list{s} = double(X);
end
Ts_list = Ts_list(valid_subject);
subject_ids = subject_ids(valid_subject);
S = numel(Ts_list);
if S < 2
    error('Fewer than two valid subjects remain.');
end

Tvec = cellfun(@(x) size(x, 1), Ts_list);
Tvec = Tvec(:);
Xcat = vertcat(Ts_list{:});

fprintf('Fitting HMM-MAR: %d subjects, %d ROIs, %d observations, K = %d.\n', ...
    S, n_roi, sum(Tvec), options.K);

% ---- fit (published) ----
[hmm, Gamma, ~, vpath, ~, ~, fehist] = hmmmar(Xcat, Tvec, options);

if size(Gamma, 1) ~= sum(Tvec) || numel(vpath) ~= sum(Tvec)
    error(['Gamma/Viterbi path have %d/%d rows; expected %d. Lags or embedding ' ...
        'would shorten them; this function assumes order = 0 without embedding.'], ...
        size(Gamma, 1), numel(vpath), sum(Tvec));
end
K = size(Gamma, 2);
if K ~= options.K
    error('hmmmar returned %d states; options.K requested %d.', K, options.K);
end

% ---- summaries (published HMM-MAR functions) ----
FO_subj = getFractionalOccupancy(Gamma, Tvec, hmm.train, 2);          % S x K
maxFO = getMaxFractionalOccupancy(Gamma, Tvec, hmm.train);            % S x 1
switching_rate = getSwitchingRate(Gamma, Tvec, hmm.train);            % S x 1
lifetimes = getStateLifeTimes(vpath, Tvec, hmm.train, 0, [], false);  % S x K cell (TRs)

dwell_TR = NaN(S, K);
for s = 1:S
    for k = 1:K
        if ~isempty(lifetimes{s, k})
            dwell_TR(s, k) = mean(lifetimes{s, k});
        end
    end
end


% ---- split concatenated outputs per subject (bookkeeping) ----
offsets = [0; cumsum(Tvec)];
Gamma_s = cell(S, 1);
vpath_s = cell(S, 1);
for s = 1:S
    idx = offsets(s) + 1:offsets(s + 1);
    Gamma_s{s} = Gamma(idx, :);
    vpath_s{s} = vpath(idx);
end

R = struct();
R.hmm = hmm;
R.Gamma = Gamma_s;
R.vpath = vpath_s;
R.Tvec = Tvec;
R.subject_ids = subject_ids;
R.valid_subject = valid_subject;
R.K = K;
R.n_roi = n_roi;
R.options = options;

R.FO_subj = FO_subj;
R.FO_subject_mean = mean(FO_subj, 1);
R.maxFO = maxFO(:);
R.switching_rate = switching_rate(:);
R.lifetimes_TR = lifetimes;
R.dwell_TR = dwell_TR;

R.P_model = hmm.P;
R.diagP_model = diag(hmm.P).';
R.P_offdiag = getTransProbs(hmm);

% ---- group state FC ----
[~, ~, sc] = hmm_subject_state_fc(Ts_list, Gamma_s, hmm.train, 0);
state_fc = cell(K, 1);
state_fc_model = cell(K, 1);
for k = 1:K
    Sk = zeros(n_roi);
    for s = 1:S
        Sk = Sk + sc{s, k};
    end
    sd = sqrt(diag(Sk));
    state_fc{k} = Sk ./ (sd * sd');
    state_fc{k}(1:n_roi + 1:end) = 1;
    [~, state_fc_model{k}] = getFuncConn(hmm, k, true);
end
R.state_fc = state_fc;
R.state_fc_source = ['pooled Gamma-weighted scatter of z-scored ROI data ' ...
    '(hmm_subject_state_fc, summed over subjects)'];
R.state_fc_model = state_fc_model;
R.fehist = fehist;

fprintf('Subject-mean FO: %s\n', mat2str(R.FO_subject_mean, 4));
fprintf('Max FO per subject: median %.3f (range %.3f-%.3f)\n', ...
    median(R.maxFO), min(R.maxFO), max(R.maxFO));
fprintf('Model diag(P): %s\n', mat2str(R.diagP_model, 4));
end
