% Simulated ROI series with known slice-timing offsets, analysed with the
% pipeline's own dfc_analytic_signal / dfc_cpca_fit / dfc_cpca_subject_stats.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '..'));     % dfc/: the pipeline's own functions
TR = 2; T = 570; n = 20; M = 20; R = 82; dt = 0.1; pad = 100;
tf = (0:dt:(T * TR + 2 * pad)).';  Nf = numel(tf);
fr = (0:Nf - 1).' / (Nf * dt); fr = min(fr, 1 / dt - fr);
bp = (fr >= 0.01 & fr <= 0.1);
up = 0:n - 1;
ord_alt_p = [up(1:2:end), up(2:2:end)];            % alt+z (true)
dn = fliplr(up); ord_alt_m = [dn(1:2:end), dn(2:2:end)];   % alt-z (RABIES default)
tau_true = zeros(1, n); tau_true(ord_alt_p + 1) = (0:n - 1) * TR / n;
tau_am = zeros(1, n);   tau_am(ord_alt_m + 1) = (0:n - 1) * TR / n;
H = cell(M, 1);
for m = 1:M
    H{m} = load(sprintf('labels/H_%02d.txt', m));
    H{m} = H{m} ./ sum(H{m}, 2);
end
pos = mean(cat(3, H{:}), 3) * ((0:n - 1).' / (n - 1));
rand('seed', 3); sgn = sign(rand(R, 1) - 0.5);   % seesaw pattern (fixed)
if ~exist('scen', 'var'), scen = {'none', 'correct', 'wrongdir', 'gradient'}; end
if ~exist('seed0', 'var'), seed0 = 11; end
if ~exist('tag', 'var'), tag = ''; end
for q = 1:numel(scen)
    rand('seed', seed0); randn('seed', seed0);
    switch scen{q}
        case 'none',     off = tau_true;           lagr = zeros(R, 1);
        case 'correct',  off = zeros(1, n);        lagr = zeros(R, 1);
        case 'wrongdir', off = tau_true - tau_am;  lagr = zeros(R, 1);
        case 'gradient', off = zeros(1, n);        lagr = 1.0 * (pos - 0.5);
    end
    Zs = cell(M, 1);
    for m = 1:M
        g = real(ifft(fft(randn(Nf, 1)) .* bp));  g = g / std(g);
        e = real(ifft(fft(randn(Nf, R)) .* bp));  e = e ./ std(e);
        h2 = real(ifft(fft(randn(Nf, 1)) .* bp)); h2 = h2 / std(h2);   % seesaw source
        X = zeros(T, R);
        for s = 1:n
            tt = pad + (0:T - 1).' * TR + off(s);
            for r = 1:R
                if H{m}(r, s) > 0
                    xr = interp1(tf, g, tt - lagr(r)) + 0.9 * sgn(r) * interp1(tf, h2, tt) ...
                         + 1.5 * interp1(tf, e(:, r), tt);
                    X(:, r) = X(:, r) + H{m}(r, s) * xr;
                end
            end
        end
        X = (X - mean(X)) ./ std(X);
        if ~exist(['sig_' scen{q} tag], 'dir'), mkdir(['sig_' scen{q} tag]); end
        dlmwrite(sprintf('sig_%s%s/signal%d.txt', scen{q}, tag, m), X, ' ');
        Zs{m} = dfc_analytic_signal(X);
    end
    G = dfc_cpca_fit(Zs, 3);
    fk = zeros(M, 3);
    for m = 1:M
        st = dfc_cpca_subject_stats(Zs{m}, G.V, TR);
        fk(m, :) = st.freq_hz.';
    end
    fk = median(fk).';
    V = G.V; reg = repmat((1:R).', 3, 1); comp = kron((1:3).', ones(R, 1));
    ph = angle(V(:)); lag = ph ./ (2 * pi * fk(comp));
    fid = fopen(sprintf('loadings_%s%s.csv', scen{q}, tag), 'w');
    fprintf(fid, 'region,component,magnitude,phase_rad,lag_sec\n');
    fprintf(fid, '%d,%d,%.10g,%.10g,%.10g\n', [reg, comp, abs(V(:)), ph, lag].');
    fclose(fid);
    printf('%s: C1-3 explain %.1f %.1f %.1f%%, f = %.3f Hz\n', scen{q}, 100 * G.explained, fk(1));
end
