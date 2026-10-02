function [similarity, pairs, matched_similarity] = ...
    hmm_match_states_fc(state_fc_a, state_fc_b, fo_a, fo_b, demean)
% HMM_MATCH_STATES_FC  Match states between two HMM fits by FC similarity.
%
%   [S, pairs, matched] = hmm_match_states_fc(fcA, fcB, foA, foB, demean)
%
% fcA, fcB : 1 x K cell arrays of n_roi x n_roi FC matrices (a 3-D
%            n_roi x n_roi x K array is also accepted)
% foA, foB : fractional occupancy for each state (1 x K). Used only when
%            demean = true; pass [] for equal weights.
% demean   : false -> identical to local_match_states in the original
%                     hmm_select_k_splithalf.m
%            true  -> the occupancy-weighted mean FC across states is
%                     subtracted from each state before correlating, so the
%                     score reflects state-specific structure rather than
%                     the static FC every state shares.
%
% States are matched one-to-one with matchpairs (Hungarian), using cost
% 1 - Pearson r on the upper triangle, as in the original.

if nargin < 5 || isempty(demean)
    demean = false;
end
if nargin < 4
    fo_b = [];
end
if nargin < 3
    fo_a = [];
end

state_fc_a = local_as_cell(state_fc_a);
state_fc_b = local_as_cell(state_fc_b);

K = numel(state_fc_a);
if numel(state_fc_b) ~= K
    error('The two fits have different numbers of states.');
end
if exist('matchpairs', 'file') ~= 2 && exist('matchpairs', 'builtin') ~= 5
    error('matchpairs was not found (part of base MATLAB since R2019a).');
end

if demean
    state_fc_a = local_demean(state_fc_a, fo_a);
    state_fc_b = local_demean(state_fc_b, fo_b);
end

n_roi = size(state_fc_a{1}, 1);
mask = triu(true(n_roi), 1);
similarity = NaN(K);

for i = 1:K
    A = state_fc_a{i};
    if ~isequal(size(A), [n_roi, n_roi])
        error('Invalid FC dimensions for state %d in the first fit.', i);
    end
    va = A(mask);

    for j = 1:K
        B = state_fc_b{j};
        if ~isequal(size(B), [n_roi, n_roi])
            error('Invalid FC dimensions for state %d in the second fit.', j);
        end
        vb = B(mask);
        valid = isfinite(va) & isfinite(vb);

        if nnz(valid) < 3 || std(va(valid)) <= eps || std(vb(valid)) <= eps
            similarity(i, j) = -1;
        else
            % Pearson r in base MATLAB (identical to corr(...,'Type','Pearson')
            % but needs no Statistics Toolbox license)
            x = va(valid) - mean(va(valid));
            y = vb(valid) - mean(vb(valid));
            similarity(i, j) = (x' * y) / sqrt((x' * x) * (y' * y));
        end
    end
end

similarity(~isfinite(similarity)) = -1;
similarity = max(min(similarity, 1), -1);

% Every real match has cost in [0,2]; an unmatched cost of 3 forces a
% complete one-to-one assignment.
pairs = matchpairs(1 - similarity, 3);

if size(pairs, 1) ~= K
    error('State matching returned %d pairs; expected %d.', size(pairs, 1), K);
end

pairs = sortrows(pairs, 1);
matched_similarity = similarity(sub2ind([K, K], pairs(:, 1), pairs(:, 2)));
end


function fc = local_as_cell(fc)
if iscell(fc)
    fc = fc(:).';
    return;
end
if isnumeric(fc) && ndims(fc) == 3
    K = size(fc, 3);
    c = cell(1, K);
    for k = 1:K
        c{k} = fc(:, :, k);
    end
    fc = c;
    return;
end
error('state_fc must be a cell array or an n_roi x n_roi x K array.');
end


function fc = local_demean(fc, fo)
K = numel(fc);
if isempty(fo) || numel(fo) ~= K || any(~isfinite(fo(:))) || sum(fo(:)) <= 0
    w = ones(K, 1) / K;
else
    w = fo(:) / sum(fo(:));
end
M = zeros(size(fc{1}));
for k = 1:K
    M = M + w(k) * fc{k};
end
for k = 1:K
    fc{k} = fc{k} - M;
end
end
