%use matlab -nodisplay -r run_mouseBrainMaps_hmm

load('../../signals/hmmmar_simple_k4_hmmpca15/hmmmar_results.mat', 'hmm');

K = numel(hmm.state);
state_fc = cell(K, 1);

for k = 1:K
    [~, state_fc{k}] = getFuncConn(hmm, k, 1);
end

plot_options = struct();
plot_options.retain_fraction = 0.15;
plot_options.edge_mode = 'positive';
plot_options.center_across_states = false;
plot_options.show_labels = false;

mouse_graph = plot_mouse_hmm_fc_on_atlas( ...
    state_fc, ...
    'example/BrainNetCoord.nii', ...
    'example/mouse_atlas_roi_order.csv', ...
    '../../signals/hmmmar_simple_k4_hmmpca15/mouse_HMM_atlas_figures', ...
    plot_options);