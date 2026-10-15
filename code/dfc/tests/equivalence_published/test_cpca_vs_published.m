function test_cpca_vs_published(ref_dir)
% TEST_CPCA_VS_PUBLISHED  dfc_analytic_signal + dfc_cpca_fit against the
% published complex PCA (Bolt et al. 2022; tsb46/complex_pca), on the files
% written by make_cpca_reference.py.
%
%   test_cpca_vs_published('cpca_reference')      % MATLAB or Octave
%
% Run here (Octave 8.4, 4 Oct 2026): analytic signal vs scipy.signal.hilbert
% 1.8e-15; per-scan Hilbert: 1 - |<v, v'>| <= 4e-16 for components 1-3;
% concatenated Hilbert (as the published script): 0.99996, 0.994, 0.978.
here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(here)));             % dfc/
f = dir(fullfile(ref_dir, 'scan*.txt'));
Zs = cell(1, numel(f));
for s = 1:numel(f)
    Zs{s} = dfc_analytic_signal(load(fullfile(ref_dir, f(s).name)));
end
H = load(fullfile(ref_dir, 'scipy_hilbert_scan01_imag.txt'));
fprintf('analytic signal vs scipy.signal.hilbert: max |diff| = %.2e\n', max(abs(imag(Zs{1}(:)) - H(:))));
G = dfc_cpca_fit(Zs, 3);
for tag = {'per_scan', 'concatenated'}
    Va = load(fullfile(ref_dir, ['Va_' tag{1} '_real.txt'])) + 1i * load(fullfile(ref_dir, ['Va_' tag{1} '_imag.txt']));
    for k = 1:3
        vb = Va(k, :).';
        v = G.V(:, k);
        fprintf('%-12s component %d: 1 - |<ours, published>| = %.2e\n', tag{1}, k, ...
            1 - abs(v' * vb) / (norm(v) * norm(vb)));
    end
end
end
