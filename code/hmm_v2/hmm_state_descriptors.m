%% hmm_state_descriptors.m
% Descriptive check of what separates the states of the reported fit: overall
% amplitude (mean ROI variance, global-signal variance) or the correlation
% pattern. No group information is used and nothing is tested; it describes
% the states that the group tests (if any) are about. Proposed 26-27 Oct 2026
% and accepted by the PI with the reported-model decision (D33): a two-state
% fit with half the time in each state could separate high- from low-amplitude
% periods (e.g. global-signal fluctuations) rather than distinct networks.
%
% Reads (hmm_config.m; the model in HMM_K / HMM_LOWRANK / HMM_PCA, else
% MODEL_PRIMARY.env)
%   <model folder>/raw_hmm_results.mat     R (run4e)
%   <data_dir>/signal<id>.txt              the ROI time series the fit used
% Writes <model folder>/state_descriptors/
%   state_descriptors_per_mouse.csv   subject_id, state, fo, effective_n,
%                                     mean_variance, gs_variance, mean_correlation
%   state_descriptors_pooled.csv      the same per state, pooled over mice
%   state_pairs.csv                   per pair of states: FC pattern r and r^2,
%                                     variance and global-signal ratios,
%                                     per-mouse medians
%   state_descriptors_report.txt      the numbers in words
%   state_fc_pooled_upper.csv         the pooled state FC (= R.state_fc), upper
%                                     triangle: roi_i, roi_j, state_1 ... state_K;
%                                     read by hmm_py/hmm_model_agreement.py to pair
%                                     the two reported models' states
%                                     (MODEL_SELECTION_RULE.md amendment 2)
% Every file carries the fit ID. Definitions: hmm_state_descriptor_values.m.
%
% Published code called: HMM-MAR standardisedata (through
% hmm_subject_state_fc.m, the state statistic of setxx.m).

clc;
clear;
close all;

cfg = hmm_config();
model_dir = cfg.model_dir;
out_dir = fullfile(model_dir, 'state_descriptors');
if ~exist(cfg.hmmmar_dir, 'dir'), error('HMM-MAR toolbox not found: %s', cfg.hmmmar_dir); end
addpath(genpath(cfg.hmmmar_dir));
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

load(fullfile(model_dir, 'raw_hmm_results.mat'), 'R');
fit_id = hmm_fit_id(R.Gamma);
if isfield(R, 'fit_id'), assert(strcmp(R.fit_id, fit_id), 'Gamma does not match its stored fit ID.'); end
id_file = fullfile(model_dir, 'fit_id.txt');
if exist(id_file, 'file'), assert(strcmp(strtrim(fileread(id_file)), fit_id), ...
        'raw_hmm_results.mat is not the fit in fit_id.txt.'); end
S = numel(R.subject_ids);
K = R.K;
Ts_list = cell(S, 1);
for i = 1:S
    f = fullfile(cfg.data_dir, sprintf(cfg.file_pattern, R.subject_ids(i)));
    if ~exist(f, 'file'), error('Missing file: %s', f); end
    Ts_list{i} = readmatrix(f);
    assert(isequal(size(Ts_list{i}), [size(R.Gamma{i}, 1), R.n_roi]), ...
        'Data and Gamma do not match for subject ID %d.', R.subject_ids(i));
end

D = hmm_state_descriptor_values(Ts_list, R.Gamma, R.hmm.train, cfg.min_effective_n);

% the pooled FC is the fit's own state FC (R.state_fc)
if isfield(R, 'state_fc')
    dfc = max(cellfun(@(a, b) max(abs(a(:) - b(:))), D.pooled.fc(:), R.state_fc(:)));
    assert(dfc < 1e-8, 'Pooled FC differs from R.state_fc (max %.3g).', dfc);
end

fid = fopen(fullfile(out_dir, 'state_descriptors_per_mouse.csv'), 'w');
fprintf(fid, 'subject_id,state,fo,effective_n,mean_variance,gs_variance,mean_correlation,fit_id\n');
for s = 1:S
    for k = 1:K
        fprintf(fid, '%d,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n', R.subject_ids(s), k, ...
            D.subject.fo(s, k), D.subject.effective_n(s, k), D.subject.mean_variance(s, k), ...
            D.subject.gs_variance(s, k), D.subject.mean_correlation(s, k), fit_id);
    end
end
fclose(fid);

fid = fopen(fullfile(out_dir, 'state_descriptors_pooled.csv'), 'w');
fprintf(fid, 'state,fo,mean_variance,gs_variance,mean_correlation,fc_split_half_r,fit_id\n');
for k = 1:K
    fprintf(fid, '%d,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n', k, D.pooled.fo(k), D.pooled.mean_variance(k), ...
        D.pooled.gs_variance(k), D.pooled.mean_correlation(k), D.pooled.fc_split_half_r(k), fit_id);
end
fclose(fid);

[ii, jj] = find(triu(true(R.n_roi), 1));
U = zeros(numel(ii), K);
for k = 1:K, U(:, k) = D.pooled.fc{k}(sub2ind([R.n_roi, R.n_roi], ii, jj)); end
fid = fopen(fullfile(out_dir, 'state_fc_pooled_upper.csv'), 'w');
fprintf(fid, 'roi_i,roi_j,%s,fit_id\n', strjoin(arrayfun(@(k) sprintf('state_%d', k), 1:K, ...
    'UniformOutput', false), ','));
fmt = ['%d,%d', repmat(',%.10g', 1, K), ',%s\n'];
for r = 1:numel(ii)
    fprintf(fid, fmt, ii(r), jj(r), U(r, :), fit_id);
end
fclose(fid);

P = D.pairs;
fid = fopen(fullfile(out_dir, 'state_pairs.csv'), 'w');
names = fieldnames(P);
fprintf(fid, '%s,fit_id\n', strjoin(names', ','));
for i = 1:numel(P)
    v = cellfun(@(n) sprintf('%.10g', P(i).(n)), names, 'UniformOutput', false);
    fprintf(fid, '%s,%s\n', strjoin(v', ','), fit_id);
end
fclose(fid);

fid = fopen(fullfile(out_dir, 'state_descriptors_report.txt'), 'w');
rep = {};      % report lines, printed and written at the end (plain fprintf: MATLAB and Octave)
rep{end + 1} = sprintf('State descriptors (descriptive; no group information): %s, fit %s\n', ...
    model_dir, fit_id(1:12));
rep{end + 1} = sprintf(['Each ROI has unit variance over each mouse''s scan, so a state''s mean variance above 1 ' ...
    'means larger signals than that mouse''s average.\n']);
rep{end + 1} = sprintf('Mice with effective N below %g in a state are left out of that state''s per-mouse values.\n\n', ...
    cfg.min_effective_n);
for k = 1:K
    rep{end + 1} = sprintf(['State %d: occupancy %.3f; mean ROI variance %.3f; global-signal variance %.3f; mean correlation ' ...
        '%.3f; FC split-half r (odd vs even mice) %.3f\n'], k, D.pooled.fo(k), D.pooled.mean_variance(k), ...
        D.pooled.gs_variance(k), D.pooled.mean_correlation(k), D.pooled.fc_split_half_r(k));
end
rep{end + 1} = sprintf('\n');
for i = 1:numel(P)
    p = P(i);
    rep{end + 1} = sprintf(['States %d vs %d: FC pattern r = %.3f (an affine function of one pattern, i.e. a uniform ' ...
        'shift and scaling of all correlations, explains r^2 = %.3f of the other); mean correlation ' ...
        '%.3f vs %.3f.\n'], p.state_a, p.state_b, p.fc_pattern_r, p.fc_pattern_r2, ...
        D.pooled.mean_correlation(p.state_a), D.pooled.mean_correlation(p.state_b));
    rep{end + 1} = sprintf(['  Corrected for estimation noise (odd vs even mice): r across halves %.3f, disattenuated ' ...
        '%.3f (1 = the patterns agree as well as each agrees with itself).\n'], ...
        p.fc_pattern_r_across_halves, p.fc_pattern_r_disattenuated);
    rep{end + 1} = sprintf(['  Mean ROI variance ratio %d/%d = %.2f (per mouse: median %.2f, higher in state %d in %.0f%% ' ...
        'of %d mice); global-signal variance ratio = %.2f (per mouse: median %.2f, higher in %.0f%%).\n'], ...
        p.state_a, p.state_b, p.mean_variance_ratio, 2 ^ p.median_log2_mean_variance_ratio, p.state_a, ...
        100 * p.frac_mice_higher_mean_variance, p.n_mice, p.gs_variance_ratio, ...
        2 ^ p.median_log2_gs_variance_ratio, 100 * p.frac_mice_higher_gs_variance);
end
rep{end + 1} = sprintf(['\nReading: states that differ mainly in amplitude have a disattenuated pattern r near 1 ' ...
    'with variance or global-signal ratios away from 1; states that differ in network pattern ' ...
    'have it well below 1. The raw r is also lowered by estimation noise; the correction is ' ...
    'unstable when a state''s FC split-half r is low. The numbers are descriptive; no ' ...
    'threshold is applied.\n']);
fprintf('%s', rep{:});
fprintf(fid, '%s', rep{:});
fclose(fid);
fprintf('Written to %s\n', out_dir);
