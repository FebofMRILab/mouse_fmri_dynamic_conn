function test_phase_randomise_vs_cbig(cbig_dir)
% TEST_PHASE_RANDOMISE_VS_CBIG  hmm_phase_randomise.m against the published
% CBIG_RL2017_get_PR_surrogate.m (Liegeois et al. 2017; github.com/ThomasYeoLab/CBIG,
% stable_projects/fMRI_dynamics/Liegeois2017_Surrogates), same random draws.
%
%   test_phase_randomise_vs_cbig('/path/to/Liegeois2017_Surrogates')
%
% Run here (Octave 8.4, 4 Oct 2026): max |difference| = 0 for odd T (569) and
% for even T (570), where the CBIG function drops the last sample and ours,
% given the same 569 samples, matches exactly.
here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(here)));             % dfc/ (hmm_phase_randomise.m)
addpath(cbig_dir);
X = randn(569, 82);
rand('state', 7); A = CBIG_RL2017_get_PR_surrogate(X, 1);
rand('state', 7); B = hmm_phase_randomise(X);
fprintf('odd T = 569: max |diff| = %g\n', max(abs(A(:) - B(:))));
X2 = randn(570, 82);
rand('state', 9); A2 = CBIG_RL2017_get_PR_surrogate(X2, 1);
rand('state', 9); B2 = hmm_phase_randomise(X2(1:569, :));
fprintf('even T = 570 (CBIG drops the last sample): max |diff| = %g\n', max(abs(A2(:) - B2(:))));
end
