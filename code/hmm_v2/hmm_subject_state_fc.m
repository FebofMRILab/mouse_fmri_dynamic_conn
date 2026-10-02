function [fc, effective_n, scatter] = hmm_subject_state_fc(Ts_list, Gamma, hmm_train, min_effective_n)
% HMM_SUBJECT_STATE_FC  Subject-specific state FC from the statistic HMM-MAR
% itself uses to estimate state covariances.
%
%   [fc, effective_n, scatter] = hmm_subject_state_fc(Ts_list, Gamma, hmm_train, min_effective_n)
%
%   Ts_list          cell (S x 1) of (time x ROI) matrices, as passed to hmmmar
%   Gamma            cell (S x 1) of (time x K) state probabilities from the group fit
%   hmm_train        hmm.train from the fit (standardisation setting, model type)
%   min_effective_n  optional; states with fewer effective time points,
%                    (sum w)^2 / sum(w.^2), get NaN FC (default 0 = keep all)
%
%   fc{s,k}          ROI x ROI correlation matrix
%   effective_n(s,k) effective number of time points behind fc{s,k}
%   scatter{s,k}     the weighted scatter matrix itself (used by the tests)
%
% What HMM-MAR does (lines quoted from the published code):
%   utils/preproc/standardisedata.m  each subject's data are centred and scaled
%                                    to unit SD per ROI (options.standardise = 1)
%   utils/macro/setxx.m              XXGXX{k} = bsxfun(@times, XX, Gamma(:,k))' * XX;
%                                    with XX = data.X for HMM-PCA (lowrank > 0) and
%                                    for zero-mean Gaussian states
%   train/obs/updatePCAparam.m       S = XXGXX{k};  (state k's HMM-PCA update)
%   train/obs/updateOmega.m          e = (bsxfun(@times,e,Gamma(:,k)))' * e;
%                                    (state k's full-covariance update)
% Here the same scatter is computed with one subject's rows only and turned
% into a correlation matrix. Summed over subjects it equals HMM-MAR's XXGXX{k}
% (checked in tests/test_hmm_v2_synthetic.m). No centring is added within
% states, because HMM-MAR's zero-mean states have none.

if nargin < 4 || isempty(min_effective_n)
    min_effective_n = 0;
end
if exist('standardisedata', 'file') ~= 2
    error('HMM-MAR (standardisedata) is not on the MATLAB path.');
end
if isfield(hmm_train, 'A') && ~isempty(hmm_train.A)
    warning('hmm_subject_state_fc:prePCA', ['The model was fitted after PCA (options.pca): HMM-MAR''s statistic is in ' ...
        'PCA space. ROI-space FC computed here extends it to the original data.']);
end
if ~isfield(hmm_train, 'zeromean') || hmm_train.zeromean ~= 1
    error('This function reproduces zero-mean states only (options.zeromean = 1).');
end
do_standardise = isfield(hmm_train, 'standardise') && hmm_train.standardise;

S = numel(Ts_list);
K = size(Gamma{1}, 2);
n_roi = size(Ts_list{1}, 2);
fc = cell(S, K);
scatter = cell(S, K);
effective_n = NaN(S, K);

for s = 1:S
    X = double(Ts_list{s});
    G = double(Gamma{s});
    if size(G, 1) ~= size(X, 1) || size(G, 2) ~= K
        error('Subject %d: Gamma is %d x %d, data are %d x %d.', s, size(G, 1), ...
            size(G, 2), size(X, 1), size(X, 2));
    end
    if do_standardise
        X = standardisedata(X, size(X, 1), 1);      % HMM-MAR's own function
    end
    for k = 1:K
        w = G(:, k);
        effective_n(s, k) = sum(w)^2 / max(sum(w.^2), eps);
        Sk = bsxfun(@times, X, w)' * X;              % = setxx.m, one subject
        scatter{s, k} = Sk;
        if effective_n(s, k) < min_effective_n || any(diag(Sk) <= eps)
            fc{s, k} = NaN(n_roi);
        else
            Sk = (Sk + Sk') / 2;
            sd = sqrt(diag(Sk));
            Rk = Sk ./ (sd * sd');                     % as corrcov
            Rk(1:n_roi + 1:end) = 1;
            fc{s, k} = Rk;
        end
    end
end
end
