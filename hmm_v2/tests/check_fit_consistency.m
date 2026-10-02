function check_fit_consistency(model_dir, out_dir, hmmmar_dir)
% CHECK_FIT_CONSISTENCY  Do the existing outputs of the PREVIOUS run4e describe
% one HMM fit or two?
%
% The previous run4e fitted hmmmar twice: once directly (saved in
% hmmmar_results.mat, read by run4d for FO, switching, max FO and dwell) and
% once through hmm_fit_raw_timeseries (saved in raw_hmm_results.mat, read by
% run5 for the graph metrics). corrStaticFC, called between the two, draws
% random numbers, so the two fits started from different random states and may
% differ, including in the order of the states. This checks it with HMM-MAR's
% getGammaSimilarity, and compares dwell times from the subject-level dual
% estimation (previous run4d) with dwell times from the group Viterbi path
% (updated run4d).
%
%   check_fit_consistency('/blue/febo/share/Abeta_Age/rabies/signals/hmmmar_simple_k3_pca15', 'fit_check', ...
%       '/blue/febo/MATLAB/HMM-MAR-master')

addpath(genpath(hmmmar_dir));
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
logf = fopen(fullfile(out_dir, 'fit_consistency.txt'), 'w');
cleanup = onCleanup(@() fclose(logf));
say = @(varargin) fprintf_both(logf, varargin{:});

A = load(fullfile(model_dir, 'hmmmar_results.mat'));
B = load(fullfile(model_dir, 'raw_hmm_results.mat'), 'R');
R = B.R;
G1 = double(A.Gamma);
G2 = vertcat(R.Gamma{:});
say('Model folder: %s\n', model_dir);
say('hmmmar_results.mat Gamma %d x %d; raw_hmm_results.mat Gamma %d x %d\n', size(G1), size(G2));
if ~isequal(size(G1), size(G2))
    say('Different sizes: the two files cover different data. Stop.\n');
    return;
end

say('Max |Gamma difference| without reordering: %.3g\n', max(abs(G1(:) - G2(:))));
[sim, assig] = getGammaSimilarity(G1, G2);          % HMM-MAR; optimal state matching
say('getGammaSimilarity: %.4f (1 = identical up to state order)\n', sim);
say('State k in hmmmar_results.mat matches state assig(k) in raw_hmm_results.mat: %s\n', ...
    mat2str(assig(:)'));

K = size(G1, 2);
fo1 = mean(G1, 1);
fo2 = mean(G2, 1);
rows = cell(K, 6);
for k = 1:K
    c = corr(G1(:, k), G2(:, assig(k)));
    rows(k, :) = {k, assig(k), fo1(k), fo2(assig(k)), c, mean(abs(G1(:, k) - G2(:, assig(k))))};
end
T = cell2table(rows, 'VariableNames', {'state_hmmmar_results', 'matched_state_raw_hmm_results', ...
    'FO_hmmmar_results', 'FO_raw_hmm_results', 'r_state_time_course', 'mean_abs_diff'});
writetable(T, fullfile(out_dir, 'state_matching.csv'));
disp(T);
if isequal(assig(:)', 1:K)
    say('State labels agree between the two files.\n');
else
    say('STATE LABELS DIFFER: graph metrics (run5) and FO/dwell tables (run4d) use different numbering.\n');
end

% dwell: dual-estimation Viterbi (previous run4d) vs group Viterbi (updated run4d)
if isfield(A, 'SubjData') && isfield(A.SubjData, 'vpath_subj') && isfield(A, 'GrpData')
    vg = double(A.GrpData.viterbipath(:));
    S = numel(A.SubjData.vpath_subj);
    Tlen = numel(vg) / S;
    d = NaN(S, K, 2);
    agree = NaN(S, 1);
    for s = 1:S
        lg = vg((s - 1) * Tlen + 1:s * Tlen);
        ld = double(A.SubjData.vpath_subj{s}(:));
        agree(s) = mean(lg == ld);
        for k = 1:K
            d(s, k, 1) = mean_run(ld == k);
            d(s, k, 2) = mean_run(lg == k);
        end
    end
    say('Dual-estimation vs group Viterbi labels agree at %.1f%% of time points (median over mice)\n', ...
        100 * median(agree));
    for k = 1:K
        say('State %d median dwell (TR): dual estimation %.1f, group Viterbi %.1f\n', k, ...
            median(d(:, k, 1), 'omitnan'), median(d(:, k, 2), 'omitnan'));
    end
end
end

function m = mean_run(x)
x = x(:);
if ~any(x), m = NaN; return; end
c = diff([0; x; 0]);
m = mean(find(c == -1) - find(c == 1));
end

function fprintf_both(fid, varargin)
fprintf(varargin{:});
fprintf(fid, varargin{:});
end
