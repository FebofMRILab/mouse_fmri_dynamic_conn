function R = hmm_fit_raw_timeseries(Ts_list, options, subject_ids)
% Fit a covariance-state HMM-MAR model to raw ROI time series.
%
% Group state FC is estimated by pooling subject-centered, posterior-
% weighted within-subject covariance matrices. This avoids introducing
% between-subject mean differences into group-level FC.

if nargin < 2
    error('Ts_list and options are required.');
end
if ~iscell(Ts_list) || isempty(Ts_list)
    error('Ts_list must be a nonempty cell array.');
end

Ts_list = Ts_list(:);
n_input_subjects = numel(Ts_list);

if nargin < 3 || isempty(subject_ids)
    subject_ids = (1:n_input_subjects).';
else
    subject_ids = subject_ids(:);
end

if numel(subject_ids) ~= n_input_subjects
    error('subject_ids must have one entry per input subject.');
end
if numel(unique(subject_ids)) ~= n_input_subjects
    error('subject_ids must be unique.');
end
if ~isstruct(options) || ~isfield(options, 'K')
    error('options must be a structure containing options.K.');
end
if ~isscalar(options.K) || options.K < 1 || options.K ~= floor(options.K)
    error('options.K must be a positive integer.');
end
if exist('hmmmar', 'file') ~= 2
    error('hmmmar was not found on the MATLAB path.');
end

first_nonempty = find(~cellfun(@isempty, Ts_list), 1);
if isempty(first_nonempty)
    error('Every entry in Ts_list is empty.');
end

n_roi = size(Ts_list{first_nonempty}, 2);
if n_roi < 2
    error('At least two ROIs are required.');
end

valid_subject = true(n_input_subjects, 1);

for s = 1:n_input_subjects
    Xs = Ts_list{s};

    if ~isnumeric(Xs) || ~ismatrix(Xs) || isempty(Xs)
        warning('Subject ID %g has an empty or nonnumeric matrix and will be removed.', ...
            subject_ids(s));
        valid_subject(s) = false;
        continue;
    end

    Xs = double(Xs);

    if size(Xs, 2) ~= n_roi
        error('Subject ID %g has %d ROIs; expected %d.', ...
            subject_ids(s), size(Xs, 2), n_roi);
    end

    if size(Xs, 1) < 2
        warning('Subject ID %g has fewer than two time points and will be removed.', ...
            subject_ids(s));
        valid_subject(s) = false;
        continue;
    end

    if any(~isfinite(Xs(:)))
        warning('Subject ID %g contains nonfinite values and will be removed.', ...
            subject_ids(s));
        valid_subject(s) = false;
        continue;
    end

    Ts_list{s} = Xs;
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

covtype_label = '<unspecified>';
if isfield(options, 'covtype')
    covtype_label = char(string(options.covtype));
end

fprintf(['Fitting raw-time-series HMM-MAR: %d subjects, %d ROIs, ' ...
    '%d total observations.\n'], S, n_roi, sum(Tvec));
fprintf('Model: K=%d, covtype=%s, pca=%s\n', ...
    options.K, covtype_label, local_pca_label(options));

[hmm, Gamma_raw] = hmmmar(Xcat, Tvec, options);

Gamma = local_split_gamma(Gamma_raw, Tvec);
Gamma = local_validate_normalize_gamma(Gamma, Tvec);

K = size(Gamma{1}, 2);
if K ~= options.K
    error('hmmmar returned %d posterior states; options.K requested %d.', ...
        K, options.K);
end

FO_subj = zeros(S, K);
for s = 1:S
    FO_subj(s, :) = mean(Gamma{s}, 1);
end

% Time-point-weighted and subject-weighted summaries are both retained.
FO_overall = mean(vertcat(Gamma{:}), 1);
FO_subject_mean = mean(FO_subj, 1);

vpath = cell(S, 1);
dwell_TR = NaN(S, K);

for s = 1:S
    [~, labels] = max(Gamma{s}, [], 2);
    vpath{s} = uint16(labels);

    for k = 1:K
        dwell_TR(s, k) = local_mean_run_length(labels == k);
    end
end

transition_counts = zeros(K);
for s = 1:S
    labels = double(vpath{s});
    if numel(labels) < 2
        continue;
    end

    linear_idx = sub2ind([K, K], labels(1:end-1), labels(2:end));
    transition_counts = transition_counts + ...
        reshape(accumarray(linear_idx, 1, [K * K, 1]), K, K);
end

P_emp = NaN(K);
row_sums = sum(transition_counts, 2);
has_outgoing = row_sums > 0;
P_emp(has_outgoing, :) = ...
    transition_counts(has_outgoing, :) ./ row_sums(has_outgoing);

[state_fc, state_fc_effective_n] = ...
    local_within_subject_weighted_state_fc(Ts_list, Gamma);

R = struct();
R.hmm = hmm;
R.Gamma = Gamma;
R.Tvec = Tvec;
R.subject_ids = subject_ids;
R.valid_subject = valid_subject;
R.K = K;
R.n_roi = n_roi;

R.FO_overall = FO_overall;
R.FO_subject_mean = FO_subject_mean;
R.FO_subj = FO_subj;
R.vpath = vpath;
R.dwell_TR = dwell_TR;

R.transition_counts = transition_counts;
R.P_emp = P_emp;

R.state_fc = state_fc;
R.state_fc_effective_n = state_fc_effective_n;
R.options = options;

if isfield(hmm, 'P') && isequal(size(hmm.P), [K, K])
    R.diagP_model = diag(hmm.P).';
else
    R.diagP_model = NaN(1, K);
end

fprintf('Time-weighted FO: %s\n', mat2str(R.FO_overall, 4));
fprintf('Subject-mean FO: %s\n', mat2str(R.FO_subject_mean, 4));
fprintf('Model diag(P): %s\n', mat2str(R.diagP_model, 4));
fprintf('Empirical diag(P): %s\n', mat2str(diag(R.P_emp).', 4));
end


function Gamma = local_split_gamma(Gamma_raw, Tvec)
n_subjects = numel(Tvec);
expected_rows = sum(Tvec);

if iscell(Gamma_raw)
    if numel(Gamma_raw) == n_subjects
        Gamma = Gamma_raw(:);
        return;
    end
    Gamma_raw = vertcat(Gamma_raw{:});
end

if ~isnumeric(Gamma_raw) || size(Gamma_raw, 1) ~= expected_rows
    error('Gamma has %d rows; expected %d.', ...
        size(Gamma_raw, 1), expected_rows);
end

offsets = [0; cumsum(Tvec(:))];
Gamma = cell(n_subjects, 1);

for s = 1:n_subjects
    Gamma{s} = Gamma_raw(offsets(s) + 1:offsets(s + 1), :);
end
end


function Gamma = local_validate_normalize_gamma(Gamma, Tvec)
if numel(Gamma) ~= numel(Tvec)
    error('Gamma contains %d subjects; expected %d.', ...
        numel(Gamma), numel(Tvec));
end

K = [];
tolerance = 1e-10;

for s = 1:numel(Gamma)
    Gs = double(Gamma{s});

    if ~ismatrix(Gs) || size(Gs, 1) ~= Tvec(s)
        error('Gamma{%d} has incorrect dimensions.', s);
    end

    if isempty(K)
        K = size(Gs, 2);
    elseif size(Gs, 2) ~= K
        error('Gamma state count differs between subjects.');
    end

    if any(~isfinite(Gs(:)))
        error('Gamma{%d} contains nonfinite posterior values.', s);
    end

    if any(Gs(:) < -tolerance)
        error('Gamma{%d} contains negative posterior values.', s);
    end

    % Remove only tiny numerical negatives.
    Gs(Gs < 0) = 0;
    row_sums = sum(Gs, 2);

    bad_rows = ~isfinite(row_sums) | row_sums <= tolerance;
    if any(bad_rows)
        error(['Gamma{%d} contains %d row(s) with zero or invalid ' ...
            'posterior mass.'], s, nnz(bad_rows));
    end

    Gamma{s} = Gs ./ row_sums;
end
end


function [state_fc, effective_n] = ...
    local_within_subject_weighted_state_fc(Ts_list, Gamma)
% Pool within-subject weighted covariance scatters. Subject-specific
% centering prevents between-subject offsets from generating covariance.

S = numel(Ts_list);
K = size(Gamma{1}, 2);
n_roi = size(Ts_list{1}, 2);

state_fc = cell(K, 1);
effective_n = NaN(K, 1);

for k = 1:K
    pooled_scatter = zeros(n_roi);
    pooled_df = 0;
    sum_w_all = 0;
    sum_w2_all = 0;

    for s = 1:S
        Xs = double(Ts_list{s});
        wk = double(Gamma{s}(:, k));

        sum_w = sum(wk);
        sum_w2 = sum(wk .^ 2);

        if sum_w <= eps
            continue;
        end

        mu_s = (Xs.' * wk) ./ sum_w;
        Xc = Xs - mu_s.';

        denom_s = sum_w - sum_w2 / sum_w;
        if denom_s <= eps
            continue;
        end

        pooled_scatter = pooled_scatter + Xc.' * (Xc .* wk);
        pooled_df = pooled_df + denom_s;
        sum_w_all = sum_w_all + sum_w;
        sum_w2_all = sum_w2_all + sum_w2;
    end

    if pooled_df <= eps || sum_w_all <= eps
        state_fc{k} = NaN(n_roi);
        continue;
    end

    covariance = pooled_scatter ./ pooled_df;
    covariance = (covariance + covariance.') ./ 2;

    variances = diag(covariance);
    valid_roi = isfinite(variances) & variances > eps;

    Rk = NaN(n_roi);
    if nnz(valid_roi) >= 2
        sd = sqrt(variances(valid_roi));
        Rvalid = covariance(valid_roi, valid_roi) ./ (sd * sd.');
        Rvalid = max(min(Rvalid, 1), -1);
        Rvalid(1:size(Rvalid, 1) + 1:end) = 1;
        Rk(valid_roi, valid_roi) = Rvalid;
    end

    state_fc{k} = Rk;
    effective_n(k) = sum_w_all ^ 2 / max(sum_w2_all, eps);
end
end


function dwell = local_mean_run_length(logical_state)
logical_state = logical(logical_state(:));

if ~any(logical_state)
    dwell = NaN;
    return;
end

d = diff([false; logical_state; false]);
run_starts = find(d == 1);
run_ends = find(d == -1) - 1;
dwell = mean(run_ends - run_starts + 1);
end


function label = local_pca_label(options)
if isfield(options, 'pca') && ~isempty(options.pca) && ...
        isnumeric(options.pca) && isscalar(options.pca) && options.pca > 0
    label = num2str(options.pca);
else
    label = 'none';
end
end