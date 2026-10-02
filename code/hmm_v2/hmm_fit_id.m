function id = hmm_fit_id(Gamma)
% HMM_FIT_ID  Fingerprint of an HMM fit: SHA-256 of its state probabilities.
%
%   id = hmm_fit_id(R.Gamma)        % cell of per-subject Gamma, or one matrix
%
% Any change in the fit (another run, another solution, reordered states)
% changes the ID. run4e stores it; every later step writes it into its outputs
% and stops if the file it reads does not carry the same ID.

if iscell(Gamma)
    Gamma = vertcat(Gamma{:});
end
bytes = typecast(double(Gamma(:)), 'uint8');
if exist('OCTAVE_VERSION', 'builtin')
    id = hash('sha256', char(bytes(:)'));
else
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(bytes(:));
    d = typecast(md.digest(), 'uint8');
    id = lower(reshape(dec2hex(d, 2)', 1, []));
end
end
