function [label, opts] = hmm_candidate_options(model)
% Step 5 model label (select_screen_candidate.py) -> HMM-MAR observation-model
% options, set exactly as run4b_hmm_splithalf_null.m does (local_cfg /
% local_model_options): lowrank_<r> -> options.lowrank = r;
% gaussian_pca_<d> -> options.pca = d; gaussian_full -> neither.
model = strtrim(char(model));
opts = struct();
tok = regexp(model, '^(lowrank|gaussian_pca)_(\d+)$', 'tokens', 'once');
if strcmp(model, 'gaussian_full')
    % full-covariance Gaussian states on all ROIs, no reduction
elseif ~isempty(tok) && strcmp(tok{1}, 'lowrank')
    opts.lowrank = str2double(tok{2});
elseif ~isempty(tok)
    opts.pca = str2double(tok{2});
else
    error(['HMM_CAND_MODEL "%s" is not a step 5 model label ' ...
           '(gaussian_full, gaussian_pca_<d>, lowrank_<r>).'], model);
end
label = model;
end
