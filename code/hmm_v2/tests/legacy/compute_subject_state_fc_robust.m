% Verbatim copy of compute_subject_state_fc_robust from the previous run5 script, for the equivalence tests only.
function [subj_state_fc, effective_n, FO_subj, map_count] = ...
    compute_subject_state_fc_robust(Ts_list, Gamma, min_effective_n)

S = numel(Ts_list);
K = size(Gamma{1}, 2);
n_roi = size(Ts_list{1}, 2);

subj_state_fc = cell(S, K);
effective_n = NaN(S, K);
FO_subj = NaN(S, K);
map_count = zeros(S, K);

for s = 1:S
    Xs = double(Ts_list{s});
    Gs = double(Gamma{s});

    if any(~isfinite(Xs(:)))
        error('Subject %d time series contains nonfinite values.', s);
    end
    if size(Gs, 1) ~= size(Xs, 1) || size(Gs, 2) ~= K
        error('Subject %d has incompatible Gamma dimensions.', s);
    end
    if any(~isfinite(Gs(:))) || any(Gs(:) < -1e-10)
        error('Subject %d has invalid posterior probabilities.', s);
    end

    Gs(Gs < 0) = 0;
    row_sum = sum(Gs, 2);

    if any(row_sum <= eps)
        error('Subject %d has zero-mass Gamma rows.', s);
    end

    Gs = Gs ./ row_sum;
    [~, map_state] = max(Gs, [], 2);

    for k = 1:K
        wk = Gs(:, k);
        total_w = sum(wk);
        sum_w2 = sum(wk .^ 2);

        FO_subj(s, k) = mean(wk);
        map_count(s, k) = nnz(map_state == k);
        effective_n(s, k) = total_w ^ 2 / max(sum_w2, eps);

        if effective_n(s, k) < min_effective_n || total_w <= eps
            subj_state_fc{s, k} = NaN(n_roi);
            continue;
        end

        denom_w = total_w - sum_w2 / total_w;
        if denom_w <= eps
            subj_state_fc{s, k} = NaN(n_roi);
            continue;
        end

        mu = (Xs.' * wk) ./ total_w;
        Xc = Xs - mu.';
        cov_k = Xc.' * (Xc .* wk) ./ denom_w;
        cov_k = (cov_k + cov_k.') ./ 2;

        variances = diag(cov_k);
        if any(~isfinite(variances)) || any(variances <= eps)
            subj_state_fc{s, k} = NaN(n_roi);
            continue;
        end

        sd = sqrt(variances);
        Rk = cov_k ./ (sd * sd.');
        Rk = max(min(Rk, 1), -1);
        Rk(1:n_roi + 1:end) = 1;

        if any(~isfinite(Rk(:)))
            subj_state_fc{s, k} = NaN(n_roi);
        else
            subj_state_fc{s, k} = Rk;
        end
    end
end
end
