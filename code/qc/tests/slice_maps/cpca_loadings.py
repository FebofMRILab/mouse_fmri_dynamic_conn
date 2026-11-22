"""Group complex PCA as in dfc_cpca_fit.m (test only): pooled Hermitian
covariance of standardised analytic signals, top components, phase fixed so
the largest loading has phase 0; frequency = median over mice of the mean
rotation rate of Z V (dfc_cpca_subject_stats.m); lag = angle / (2 pi f)."""
import sys, glob, numpy as np, pandas as pd
from scipy.signal import hilbert
sig_dir, out, TR, K = sys.argv[1], sys.argv[2], 2.0, 3
Zs = []
for f in sorted(glob.glob(f"{sig_dir}/signal*.txt")):
    X = np.loadtxt(f); X = (X - X.mean(0)) / X.std(0, ddof=1); Zs.append(hilbert(X, axis=0))
C = sum(Z.conj().T @ Z for Z in Zs) / sum(len(Z) for Z in Zs); C = (C + C.conj().T) / 2
lam, V = np.linalg.eigh(C); o = np.argsort(lam)[::-1]; V = V[:, o[:K]]
V = V * np.exp(-1j * np.angle(V[np.argmax(np.abs(V), 0), range(K)]))[None, :]
fk = np.median([[np.mean(np.angle((Z @ V)[1:, k] * np.conj((Z @ V)[:-1, k]))) / (2 * np.pi * TR)
                 for k in range(K)] for Z in Zs], 0)
rows = [dict(region=r + 1, component=k + 1, magnitude=abs(V[r, k]), phase_rad=np.angle(V[r, k]),
             lag_sec=np.angle(V[r, k]) / (2 * np.pi * fk[k])) for k in range(K) for r in range(V.shape[0])]
pd.DataFrame(rows).to_csv(out, index=False)
print("explained", np.round(lam[o[:K]] / lam.sum(), 3), "f", np.round(fk, 3))
