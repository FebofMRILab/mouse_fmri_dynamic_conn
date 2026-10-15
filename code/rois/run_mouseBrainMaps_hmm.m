%use matlab -nodisplay -r run_mouseBrainMaps_hmm

load('../../signals/hmmmar_simple_k3_hmmpca25/hmmmar_results.mat', 'hmm');

K = numel(hmm.state);
state_fc = cell(K, 1);

for k = 1:K
    [~, state_fc{k}] = getFuncConn(hmm, k, 1);
end

plot_options = struct();
plot_options.retain_fraction = 0.15;
plot_options.edge_mode = 'signed';
plot_options.center_across_states = true;
plot_options.show_labels = false;
plot_options.figure_visible = 'off'

mouse_graph = plot_mouse_hmm_fc_on_atlas( ...
    state_fc, ...
    'example/BrainNetCoord.nii', ...
    'example/mouse_atlas_roi_order.csv', ...
    '../../signals/hmmmar_simple_k3_hmmpca25/mouse_HMM_atlas_figures', ...
    plot_options);