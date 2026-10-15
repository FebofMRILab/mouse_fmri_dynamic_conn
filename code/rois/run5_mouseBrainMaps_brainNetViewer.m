load('../../signals/hmmmar_simple_k4_hmmpca15/hmmmar_results.mat', 'hmm');

K = numel(hmm.state);
state_fc = cell(K, 1);

for k = 1:K
    [~, state_fc{k}] = getFuncConn(hmm, k, 1);
end

brainnet_options = struct();
brainnet_options.retain_fraction = 0.15;
brainnet_options.edge_mode = 'positive';
brainnet_options.center_across_states = false;
brainnet_options.node_size_mode = 'strength';
brainnet_options.node_color_mode = 'strength';
brainnet_options.minimum_abs_correlation = 0.05;

brainnet_output = export_hmm_fc_to_brainnet( ...
    state_fc, ...
    '../../signals/hmmmar_simple_k4_hmmpca15/mouse_HMM_atlas_figures/mouse_atlas_ROI_centroids.csv', ...
    '../../signals/hmmmar_simple_k4_hmmpca15/brainnet_HMM_K4_PCA15', ...
    brainnet_options);