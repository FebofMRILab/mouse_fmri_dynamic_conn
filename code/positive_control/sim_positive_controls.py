#!/usr/bin/env python3
"""
sim_positive_controls.py
-------------------------------------------------------------------
Simulated cohorts for the positive control: the same three mechanisms as the
schematic (Fig_schematic_event_mechanism), built to look like the mouse data
to the pipelines, so that run_dcc_cpca_null.m and run4c_confirm_lowrank.m can
be run on them UNCHANGED and their outputs set beside the mouse outputs.

What each simulated cohort copies from the real data (the template folder)
  - number of mice, ROIs, volumes, TR and group labels (meta_data.csv)
  - the group FC (pooled correlation of the standardised data), split into a
    signal part (the top k eigen-components, k = parallel-analysis signal
    rank) and a noise part (the rest), so static FC matches by construction
  - each mouse's own network strengths (variance of its data on each signal
    network), so between-mouse and between-group differences carry over
Signal networks are the top-k loadings after varimax rotation (localised
network patterns rather than whole-brain eigenmodes), signed so that each
network's largest loading is positive.

Mechanisms (all share the haemodynamic response, the 0.01-0.1 Hz band-pass
and, in expectation, every mouse's static FC and spectra):
  A         stationary Gaussian: each network is driven by white noise
            (what the stationary replica is: the negative control)
  B         sparse shared events: each network is driven by Poisson events
            (lognormal amplitudes) that activate all of its ROIs at once;
            stationary, non-Gaussian. One knob: events per network per minute.
  C         state switching: n_states covariance states shared by all mice,
            differing in which networks dominate and which networks couple:
            Sigma_q = I + sep * s_max * B_q with sum_q B_q = 0, so the states
            average to the static FC; dwell times uniform in --dwell seconds,
            abrupt at the neural level (smoothed by the HRF). One knob: the
            separation sep in [0, 1) (1 = the least valid state is singular).
            --c_variance_share sets how much the states differ in network
            amplitude vs coupling (0.5 default; 0 = coupling only).
  C_strong  the same mice as C with sep = --c_strong_sep (0.9): states that
            should be easy to find, to show the pipelines CAN detect switching.
Noise: Gaussian, band-passed, with the spatial covariance of the non-signal
eigen-components (thermal noise plus unmodelled structure).

Matching (two stages). Each mechanism is set to the strength at which it
reproduces the mouse's headline result, the calibrated DCC excess (median
d = z_observed - z_replica for dcc_a); everything else is then a prediction:
  --stage grid   writes A (all mice) and small grid cohorts of B (event
                 rates --b_grid) and C (separations --c_grid). Run the DFC
                 pipeline on each (submit_positive_controls.sh grid).
  --stage final  reads the grid DFC outputs and the mouse DFC output, fits
                 median d against strength (monotone), interpolates the
                 strength that gives the mouse median d, and writes B, C and
                 C_strong with all mice. Run DFC and the HMM on A, B, C and
                 C_strong (submit_positive_controls.sh final).
  --stage fixed  no matching: A, B, C, C_strong at --b_rate / --c_sep.
Every stage must use the same template, rank and seed (checked).
Mouse i of a mechanism always gets the same random numbers, so grid, final
and C / C_strong cohorts differ only in the strength knob.

Outputs (<out_dir>/):
  <cohort>/signal{n}.txt, <cohort>/meta_data.csv   pipeline inputs
  <cohort>/truth_events.csv (B) or truth_states.csv (C)
  cohorts_<stage>.txt       cohort names, read by submit_positive_controls.sh
  networks.npz              loadings, noise covariance, gains, HRF, states
  sim_manifest_<stage>.json every parameter, seed and matched value
  sim_diagnostics_<stage>.csv  per simulated and per real mouse: static FC
                            similarity to the group FC, burst_z, fc_change_z
                            (as dfc_confound_checks.py), kurtosis measures
  FigS_positive_control_design.{png,pdf}   (final / fixed stage)

Usage (HiPerGator, from the folder with hmm_signals/ and dfc_dcc_cpca/):
  python sim_positive_controls.py --stage grid
  bash submit_positive_controls.sh grid          # DFC on the grid cohorts
  python sim_positive_controls.py --stage final
  bash submit_positive_controls.sh final         # DFC + HMM on A, B, C, C_strong
  python summarize_positive_controls.py
Local test without real data:
  python sim_positive_controls.py --stage fixed --synthetic_template 40 \
      --n_subjects 16 --rank 12 --b_rate 4 --c_sep 0.6 --out_dir sim_test
"""

import argparse
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 1.2,
                     "savefig.dpi": 350, "savefig.facecolor": "white",
                     "figure.facecolor": "white"})
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
warnings.filterwarnings("ignore", message=".*findfont.*")

INK, SECONDARY, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MECH_COLORS = {"A": MUTED, "B": "#eb6834", "C": "#2a78d6", "C_strong": "#1f5fae", "Mouse": INK}
MECH_LABELS = {"A": "A Gaussian", "B": "B Events", "C": "C Switching", "C_strong": "C strong",
               "Mouse": "Mouse"}
DT_FINE = 0.5            # s, grid on which neural drive and the HRF live
PAD_VOL = 40             # volumes simulated before and after (trimmed after filtering)
N_SURR = 19              # phase surrogates per mouse for burst_z / fc_change_z


# ---------------- statistics (identical to dfc_confound_checks.py) ----------------
def standardise(X):
    Xs = X - X.mean(axis=0)
    sd = Xs.std(axis=0, ddof=1)
    sd[sd <= 0] = 1
    return Xs / sd


def group_pca(Xs_list, d):
    C = sum(X.T @ X for X in Xs_list) / (sum(X.shape[0] for X in Xs_list) - 1)
    lam, V = np.linalg.eigh((C + C.T) / 2)
    o = np.argsort(lam)[::-1]
    A = V[:, o[:d]]
    for k in range(A.shape[1]):
        A[:, k] *= np.sign(A[np.argmax(np.abs(A[:, k])), k])
    return A


def phase_randomise(X, rng):
    T = X.shape[0]
    F = np.fft.rfft(X, axis=0)
    ph = rng.uniform(0, 2 * np.pi, F.shape[0])
    ph[0] = 0
    if T % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(F * np.exp(1j * ph)[:, None], n=T, axis=0)


def excess_kurtosis(x):
    return stats.kurtosis(x, axis=0, fisher=True, bias=False)


def burst_stat(P):
    Ps = P / P.std(axis=0, ddof=1)
    return excess_kurtosis((Ps ** 2).sum(axis=1))


def fc_change(X):
    h = X.shape[0] // 2
    iu = np.triu_indices(X.shape[1], 1)
    r1 = np.corrcoef(X[:h], rowvar=False)[iu]
    r2 = np.corrcoef(X[h:], rowvar=False)[iu]
    return 1 - np.corrcoef(r1, r2)[0, 1]


def surrogate_z(X, A, rng, n_surr=N_SURR):
    """burst_z and fc_change_z exactly as in dfc_confound_checks.per_mouse."""
    Xs = standardise(X)
    b_obs, c_obs = burst_stat(Xs @ A), fc_change(Xs)
    b_s, c_s = [], []
    for _ in range(n_surr):
        S = phase_randomise(Xs, rng)
        b_s.append(burst_stat(S @ A))
        c_s.append(fc_change(S))
    return ((b_obs - np.mean(b_s)) / np.std(b_s, ddof=1),
            (c_obs - np.mean(c_s)) / np.std(c_s, ddof=1))


def diagnostics(X, A, R_group, rng, n_edges=1000):
    Xs = standardise(X)
    bz, cz = surrogate_z(X, A, rng)
    iu = np.triu_indices(X.shape[1], 1)
    R = np.corrcoef(Xs, rowvar=False)
    pick = rng.choice(iu[0].size, min(n_edges, iu[0].size), replace=False)
    ets = Xs[:, iu[0][pick]] * Xs[:, iu[1][pick]]
    ets_all = Xs[:, iu[0]] * Xs[:, iu[1]]
    rss = np.sqrt((ets_all ** 2).sum(axis=1))
    return dict(fc_similarity=float(np.corrcoef(R[iu], R_group[iu])[0, 1]),
                burst_z=float(bz), fc_change_z=float(cz),
                kurt_roi=float(np.median(excess_kurtosis(Xs))),
                kurt_pc=float(np.mean(excess_kurtosis(Xs @ A))),
                kurt_cofluct=float(np.median(excess_kurtosis(ets))),
                kurt_rss=float(excess_kurtosis(rss)))


def parallel_analysis_rank(X, rng, n_shift=30):
    Xs = standardise(X)
    T, N = Xs.shape
    lam = np.sort(np.linalg.eigvalsh(np.corrcoef(Xs, rowvar=False)))[::-1]
    S = []
    for _ in range(n_shift):
        Y = np.column_stack([np.roll(Xs[:, i], rng.integers(0, T)) for i in range(N)])
        S.append(np.sort(np.linalg.eigvalsh(np.corrcoef(Y, rowvar=False)))[::-1])
    above = lam > np.percentile(np.array(S), 95, axis=0)
    return int(np.argmin(above)) if not above.all() else N


# ---------------- network model ----------------
def varimax(Phi, n_iter=200, tol=1e-9):
    p, k = Phi.shape
    Rm = np.eye(k)
    d = 0.0
    for _ in range(n_iter):
        L = Phi @ Rm
        u, s, vh = np.linalg.svd(Phi.T @ (L ** 3 - L @ np.diag((L ** 2).sum(axis=0)) / p))
        Rm = u @ vh
        d_old, d = d, s.sum()
        if d_old and d / d_old < 1 + tol:
            break
    return Phi @ Rm


def build_networks(R, k):
    lam, U = np.linalg.eigh((R + R.T) / 2)
    o = np.argsort(lam)[::-1]
    lam, U = np.clip(lam[o], 0, None), U[:, o]
    M = varimax(U[:, :k] * np.sqrt(lam[:k]))
    M *= np.sign(M[np.argmax(np.abs(M), axis=0), np.arange(k)])
    C_noise = U[:, k:] @ np.diag(lam[k:]) @ U[:, k:].T
    return M, C_noise, lam


def subject_gains(Xs_list, M, lo=0.25, hi=4.0):
    """Each real mouse's variance on each signal network, relative to the group
    (least-squares projection onto the networks), clipped to [lo, hi]."""
    P = np.linalg.pinv(M)
    V = np.array([np.var(Xs @ P.T, axis=0, ddof=1) for Xs in Xs_list])
    V = V / V.mean(axis=0, keepdims=True)
    return np.clip(V, lo, hi)


def noise_mixers(M, C_noise, gains, floor=0.05):
    """Per subject: noise covariance with the group noise correlation, scaled so
    every ROI has unit total variance given that subject's network gains."""
    dn = np.sqrt(np.clip(np.diag(C_noise), 1e-6, None))
    Cn_corr = C_noise / np.outer(dn, dn)
    Ln = np.linalg.cholesky(Cn_corr + 1e-6 * np.eye(len(dn)))
    out = []
    for g in gains:
        v_sig = (M ** 2 * g[None, :]).sum(axis=1)
        out.append(np.sqrt(np.clip(1 - v_sig, floor, None))[:, None] * Ln)
    return out


# ---------------- time courses ----------------
def make_hrf(peak, undershoot=10.0, ratio=6.0):
    t = np.arange(0, 32, DT_FINE)
    h = stats.gamma.pdf(t, 1 + peak) - stats.gamma.pdf(t, 1 + undershoot) / ratio
    return h / h.sum()


class Timing:
    def __init__(self, T, TR, hrf_peak):
        self.T, self.TR = T, TR
        self.T_pad = T + 2 * PAD_VOL
        self.up = int(round(TR / DT_FINE))
        self.n_fine = self.T_pad * self.up
        self.hrf = make_hrf(hrf_peak)
        self.b, self.a = butter(3, [0.01 * 2 * TR, 0.1 * 2 * TR], btype="band")

    def to_bold(self, drive):
        """drive: n_fine x k -> T x k: HRF, sampling at TR, band-pass, trim, z."""
        y = np.apply_along_axis(lambda d: np.convolve(d, self.hrf)[:d.size], 0, drive)
        y = filtfilt(self.b, self.a, y[::self.up], axis=0)[PAD_VOL:PAD_VOL + self.T]
        return standardise(y)

    def noise(self, rng, n):
        e = filtfilt(self.b, self.a, rng.standard_normal((self.T_pad, n)), axis=0)
        return standardise(e[PAD_VOL:PAD_VOL + self.T])


def drive_gaussian(tm, rng, k):
    return rng.standard_normal((tm.n_fine, k)), None


def drive_events(tm, rng, k, rate_per_min, amp_sd=0.35):
    """Poisson events per network. Common random numbers: the same uniforms
    and amplitudes for every rate, so the tuning curve is monotone."""
    u = rng.random((tm.n_fine, k))
    amp = rng.lognormal(0, amp_sd, (tm.n_fine, k))
    ev = u < rate_per_min / 60 * DT_FINE
    d = np.where(ev, amp, 0.0) + 1e-3 * rng.standard_normal((tm.n_fine, k))
    return d, ev


def state_sequence(tm, rng, n_states, dwell):
    s = np.empty(tm.n_fine, int)
    i, cur = 0, rng.integers(n_states)
    while i < tm.n_fine:
        L = int(rng.uniform(*dwell) / DT_FINE)
        s[i:i + L] = cur
        cur = (cur + rng.integers(1, n_states)) % n_states
        i += L
    return s


def state_patterns(k, n_states, seed, variance_share=0.5):
    """Zero-sum directions B_q (sum over states = 0), shared by all mice:
    a diagonal part (which networks dominate: state q boosts every
    n_states-th network) plus an off-diagonal part (which networks couple).
    variance_share = share of the squared Frobenius norm in the diagonal part
    (0.5: equal; 0: states differ in coupling only, not in network amplitude).
    Sigma_q = I + sep * s_max * B_q, so every state is a valid covariance for
    sep < 1 and the states average to I (static FC unchanged in expectation)."""
    rng = np.random.default_rng(seed)
    D = np.full((n_states, k), -1.0 / (n_states - 1))
    for j in range(k):
        D[j % n_states, j] = 1.0
    G = rng.standard_normal((n_states, k, k))
    G = (G + G.transpose(0, 2, 1)) / 2
    G[:, np.arange(k), np.arange(k)] = 0
    G -= G.mean(axis=0, keepdims=True)
    B = []
    for q in range(n_states):
        Dq = np.diag(D[q])
        B.append(np.sqrt(2 * variance_share) * Dq
                 + np.sqrt(2 * (1 - variance_share)) * G[q] * np.linalg.norm(Dq)
                 / np.linalg.norm(G[q]))
    B = np.array(B)
    s_max = min(1.0 / -np.linalg.eigvalsh(b).min() for b in B)
    return B, s_max


def state_covariances(B, s_max, sep):
    return np.array([np.eye(B.shape[1]) + sep * s_max * b for b in B])


def drive_switching(tm, rng, k, sep, n_states, dwell, B, s_max):
    e = rng.standard_normal((tm.n_fine, k))
    s = state_sequence(tm, rng, n_states, dwell)
    L = np.array([np.linalg.cholesky(S + 1e-9 * np.eye(k))
                  for S in state_covariances(B, s_max, sep)])
    return np.einsum("tij,tj->ti", L[s], e), s


class Model:
    """Everything shared by the simulated mice of all mechanisms."""
    def __init__(self, tm, M, gains, Lns, n_states, dwell, B, s_max, seed):
        self.tm, self.M, self.gains, self.Lns = tm, M, gains, Lns
        self.n_states, self.dwell, self.B, self.s_max, self.seed = n_states, dwell, B, s_max, seed

    # mouse i of a mechanism always gets the same random numbers, in tuning and
    # in the final cohort (common random numbers: the tuning curve is smooth and
    # the written cohort has exactly the tuned statistic)
    def rng_sim(self, mech, i):
        return np.random.default_rng([self.seed, MECH_SEEDS[mech], i])

    def rng_stat(self, mech, i):
        return np.random.default_rng([self.seed, 99, MECH_SEEDS[mech], i])

    def simulate(self, mech, param, i):
        rng = self.rng_sim(mech, i)
        k = self.M.shape[1]
        if mech == "A":
            d, extra = drive_gaussian(self.tm, rng, k)
        elif mech == "B":
            d, extra = drive_events(self.tm, rng, k, param)
        else:
            d, extra = drive_switching(self.tm, rng, k, param, self.n_states, self.dwell,
                                       self.B, self.s_max)
        S = self.tm.to_bold(d) * np.sqrt(self.gains[i])[None, :]
        X = S @ self.M.T + self.tm.noise(rng, self.M.shape[0]) @ self.Lns[i].T
        return X, extra


MECH_SEEDS = {"A": 1, "B": 2, "C": 3, "C_strong": 3}   # C_strong = C's mice at larger separation


# ---------------- matching to the mouse DCC excess ----------------
def calibrated_d(dfc_dir, metric):
    """Per-subject calibrated DCC excess d = z_observed - z_replica (phase null),
    from a run_dcc_cpca_null.m output folder."""
    f = os.path.join(dfc_dir, "dfc_subject_metrics.csv")
    if not os.path.isfile(f):
        return None
    m = pd.read_csv(f)
    m = m[m.metric == metric]
    return (m.phase_z - m.phase_replica_z).dropna().to_numpy(float)


def match_parameter(points, target, log):
    """points: [(parameter, median d)] from the grid cohorts. Monotone
    (isotonic) fit of median d against the parameter (log scale for event
    rates), then linear interpolation to the target."""
    pts = sorted(points)
    x = np.log([p for p, _ in pts]) if log else np.array([p for p, _ in pts], float)
    y = np.array([v for _, v in pts], float)
    increasing = np.corrcoef(x, y)[0, 1] >= 0 if len(x) > 2 else y[-1] >= y[0]
    yy = y if increasing else -y
    # pool-adjacent-violators
    blocks = [[v, 1] for v in yy]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (
                blocks[i][1] + blocks[i + 1][1])
            blocks[i:i + 2] = [[v, blocks[i][1] + blocks[i + 1][1]]]
            i = max(i - 1, 0)
        else:
            i += 1
    fit = np.concatenate([[v] * n for v, n in blocks])
    fit = fit if increasing else -fit
    lo, hi = min(fit), max(fit)
    if not lo <= target <= hi:
        j = int(np.argmin(np.abs(fit - target)))
        note = (f"mouse median d {target:.2f} outside the grid's range "
                f"[{lo:.2f}, {hi:.2f}]: clamped to the nearest grid point")
        return float(np.exp(x[j]) if log else x[j]), note
    o = np.argsort(fit)
    xm = float(np.interp(target, fit[o], x[o]))
    return float(np.exp(xm) if log else xm), "interpolated"


# ---------------- template ----------------
def load_template(args):
    if args.synthetic_template:
        return synthetic_template(args)
    pat = re.compile("^" + re.escape(args.file_pattern).replace(r"\{\}", r"(\d+)") + "$")
    found = sorted(int(m.group(1)) for f in os.listdir(args.template)
                   if (m := pat.match(f)))
    if args.max_template:
        found = [s for s in found if s <= args.max_template]
    Xs_list, ids = [], []
    for s in found:
        fn = os.path.join(args.template, args.file_pattern.format(s))
        sep = "," if "," in open(fn).readline() else None
        X = np.loadtxt(fn, delimiter=sep).astype(float)
        if X.shape[0] < X.shape[1]:
            X = X.T
        if not np.all(np.isfinite(X)):
            print(f"Template subject {s}: non-finite values; skipped")
            continue
        Xs_list.append(standardise(X))
        ids.append(s)
    if len(Xs_list) < 4:
        raise SystemExit(f"Only {len(Xs_list)} template subjects found in {args.template}.")
    meta = os.path.join(args.template, "meta_data.csv")
    groups = {i: "All" for i in ids}
    if os.path.isfile(meta):
        m = pd.read_csv(meta, encoding="utf-8-sig")
        m.columns = [c.strip().lower() for c in m.columns]
        groups.update(dict(zip(m.subject_id.astype(int), m.group.astype(str).str.strip())))
    T = min(X.shape[0] for X in Xs_list)
    return [X[:T] for X in Xs_list], ids, [groups[i] for i in ids], True


def synthetic_template(args):
    """Stand-in for the mouse data when testing: Gaussian, 12 signal dimensions
    with a spectrum like the mouse's (lambda1 ~ 8%, ~40% variance above noise)."""
    rng = np.random.default_rng(7)
    N, T = args.synthetic_template, args.T or 570
    k = 12
    lam_sig = 0.42 * N * np.geomspace(1, 0.25, k) / np.geomspace(1, 0.25, k).sum()
    lam = np.r_[lam_sig, np.full(N - k, (N - lam_sig.sum()) / (N - k))]
    U, _ = np.linalg.qr(rng.standard_normal((N, N)))
    C = U @ np.diag(lam) @ U.T
    d = np.sqrt(np.diag(C))
    L = np.linalg.cholesky(C / np.outer(d, d))
    tm = Timing(T, args.TR, args.hrf_peak)
    n = args.n_subjects or 16
    Xs_list = [standardise(tm.noise(rng, N) @ L.T) for _ in range(n)]
    groups = ["G1" if i % 2 == 0 else "G2" for i in range(n)]
    return Xs_list, list(range(1, n + 1)), groups, False


# ---------------- figure ----------------
def design_figure(D, out_dir):
    order = [m for m in ["Mouse", "A", "B", "C", "C_strong"] if m in set(D.cohort)]
    panels = [("fc_similarity", "Static FC vs group FC (r)", "Static FC"),
              ("burst_z", "Shared bursts (z vs surrogates)", "Shared bursts"),
              ("fc_change_z", "FC change, halves (z)", "FC change"),
              ("kurt_rss", "RSS kurtosis", "Co-fluctuation peaks")]
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.9), gridspec_kw=dict(wspace=0.62))
    rng = np.random.default_rng(3)
    for ax, (col, ylab, title) in zip(axes, panels):
        for i, m in enumerate(order):
            v = D.loc[D.cohort == m, col].dropna().to_numpy(float)
            if not v.size:
                continue
            c = MECH_COLORS[m]
            ax.boxplot(v, positions=[i], widths=0.56, showfliers=False, patch_artist=True,
                       medianprops=dict(color=INK, lw=1.1),
                       boxprops=dict(facecolor="white", edgecolor=c, lw=1.1),
                       whiskerprops=dict(color=c, lw=1.0), capprops=dict(color=c, lw=1.0))
            ax.scatter(i + rng.uniform(-0.14, 0.14, v.size), v, s=4.5 ** 2 / 2, color=c,
                       alpha=0.6, lw=0, zorder=3)
        if "Mouse" in order:
            mv = D.loc[D.cohort == "Mouse", col].median()
            ax.axhline(mv, color=INK, lw=0.7, ls=":", zorder=0)
        if col in ("burst_z", "fc_change_z"):
            ax.axhline(0, color=BASELINE, lw=0.7, zorder=0)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([MECH_LABELS[m] for m in order], fontsize=7, rotation=45, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(title, fontsize=8, loc="left")
        sns.despine(ax=ax)
    for ax, letter in zip(axes, "ABCD"):
        ax.text(-0.42, 1.1, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.text(0.01, -0.2, "Dotted line: median of the mouse data. Each simulated mouse copies one "
             "real mouse's network strengths and group label.", fontsize=6.5, color=SECONDARY)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"FigS_positive_control_design.{ext}"),
                    bbox_inches="tight", dpi=450 if ext == "png" else None)
    plt.close(fig)




# ---------------- main ----------------
def write_cohort(name, letter, param, n, model, args, T, sim_groups, A, R_group):
    mdir = os.path.join(args.out_dir, name)
    os.makedirs(mdir, exist_ok=True)
    truth, rows = [], []
    for i in range(n):
        X, extra = model.simulate(letter, param, i)
        np.savetxt(os.path.join(mdir, args.file_pattern.format(i + 1)), X, fmt="%.6g",
                   delimiter=",")
        if letter == "B":
            tt, jj = np.nonzero(extra)
            t_s = tt * DT_FINE - PAD_VOL * args.TR
            keep = (t_s >= 0) & (t_s < T * args.TR)
            truth += [dict(subject_id=i + 1, network=int(j) + 1, time_s=float(t))
                      for t, j in zip(t_s[keep], jj[keep])]
        elif letter == "C":
            st = extra[::model.tm.up][PAD_VOL:PAD_VOL + T]
            truth += [dict(subject_id=i + 1, volume=v + 1, state=int(q) + 1)
                      for v, q in enumerate(st)]
        rows.append(dict(cohort=name, mechanism=letter, parameter=param, subject_id=i + 1,
                         group=sim_groups[i],
                         **diagnostics(X, A, R_group, model.rng_stat(letter, i))))
    pd.DataFrame(dict(subject_id=np.arange(1, n + 1), group=sim_groups[:n])).to_csv(
        os.path.join(mdir, "meta_data.csv"), index=False)
    if truth:
        fn = "truth_events.csv" if letter == "B" else "truth_states.csv"
        pd.DataFrame(truth).to_csv(os.path.join(mdir, fn), index=False)
    print(f"  {name}: {n} mice -> {mdir}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["grid", "final", "fixed"], required=True)
    ap.add_argument("--template", default="hmm_signals")
    ap.add_argument("--synthetic_template", type=int, default=0, metavar="N_ROI",
                    help="test mode: no real data, synthetic Gaussian template")
    ap.add_argument("--file_pattern", default="signal{}.txt")
    ap.add_argument("--max_template", type=int, default=None)
    ap.add_argument("--out_dir", default="sim_positive_control")
    ap.add_argument("--n_subjects", type=int, default=None, help="default: as the template")
    ap.add_argument("--n_grid", type=int, default=32, help="mice per grid cohort")
    ap.add_argument("--b_grid", type=float, nargs="+", default=[1.0, 4.0, 16.0, 64.0],
                    help="events per network per minute")
    ap.add_argument("--c_grid", type=float, nargs="+", default=[0.3, 0.6, 0.9],
                    help="state separations")
    ap.add_argument("--c_strong_sep", type=float, default=0.9)
    ap.add_argument("--b_rate", type=float, default=None, help="--stage fixed")
    ap.add_argument("--c_sep", type=float, default=None, help="--stage fixed")
    ap.add_argument("--mouse_dfc", default="dfc_dcc_cpca",
                    help="the mouse run_dcc_cpca_null output (target of the matching)")
    ap.add_argument("--dfc_prefix", default="dfc_sim_",
                    help="grid DFC outputs are <prefix><cohort>")
    ap.add_argument("--match_metric", default="dcc_a")
    ap.add_argument("--TR", type=float, default=2.0)
    ap.add_argument("--T", type=int, default=None, help="default: as the template")
    ap.add_argument("--rank", type=int, default=None,
                    help="signal networks; default: median parallel-analysis rank")
    ap.add_argument("--pca_dim", type=int, default=25, help="for burst_z, as the DCC space")
    ap.add_argument("--hrf_peak", type=float, default=3.0, help="s (mouse)")
    ap.add_argument("--n_states", type=int, default=3)
    ap.add_argument("--dwell", type=float, nargs=2, default=[30.0, 90.0], metavar=("MIN", "MAX"))
    ap.add_argument("--c_variance_share", type=float, default=0.5,
                    help="how much states differ in network amplitude vs coupling "
                         "(0 = coupling only)")
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- template and network model (deterministic: identical in every stage)
    Xs_list, ids, groups, real = load_template(args)
    T = args.T or Xs_list[0].shape[0]
    N = Xs_list[0].shape[1]
    n_sub = args.n_subjects or len(Xs_list)
    print(f"Template: {len(Xs_list)} subjects, {N} ROIs, {T} volumes"
          f"{'' if real else ' (SYNTHETIC)'}.")
    R_group = sum(X.T @ X for X in Xs_list) / (sum(X.shape[0] for X in Xs_list) - 1)
    dg = np.sqrt(np.diag(R_group))
    R_group = R_group / np.outer(dg, dg)
    A = group_pca(Xs_list, min(args.pca_dim, N - 1))
    if args.rank:
        k = args.rank
    else:
        k = int(np.clip(np.median([parallel_analysis_rank(X, rng) for X in Xs_list]), 2, N - 2))
        print(f"Signal rank (parallel analysis, median over subjects): {k}")
    M, C_noise, lam = build_networks(R_group, k)
    tmpl_idx = [i % len(Xs_list) for i in range(n_sub)]     # simulated mouse i <- template i
    gains = subject_gains(Xs_list, M)[tmpl_idx]
    Lns = noise_mixers(M, C_noise, gains)
    sim_groups = [groups[i] for i in tmpl_idx]
    tm = Timing(T, args.TR, args.hrf_peak)
    B_pat, s_max = state_patterns(k, args.n_states, args.seed + 11, args.c_variance_share)
    model = Model(tm, M, gains, Lns, args.n_states, args.dwell, B_pat, s_max, args.seed)
    print(f"Signal networks: {k} (varimax), {100 * lam[:k].sum() / lam.sum():.1f}% of group-FC "
          f"variance.")
    net_f = os.path.join(args.out_dir, "networks.npz")
    if args.stage == "final" and os.path.isfile(net_f):
        old = np.load(net_f)
        if old["M"].shape != M.shape or not np.allclose(old["M"], M, atol=1e-8):
            raise SystemExit("The network model differs from the grid stage (template, rank or "
                             "seed changed?). Use the same arguments as for --stage grid.")

    # ---- the real mice, described the same way
    rows = []
    if real:
        for j, X in enumerate(Xs_list):
            rows.append(dict(cohort="Mouse", mechanism="Mouse", parameter=np.nan,
                             subject_id=ids[j], group=groups[j],
                             **diagnostics(X, A, R_group, np.random.default_rng([args.seed, 98, j]))))

    # ---- cohorts
    info = {}
    if args.stage == "grid":
        cohorts = [("A", "A", None, n_sub)]
        cohorts += [(f"B_r{r:g}", "B", r, min(args.n_grid, n_sub)) for r in args.b_grid]
        cohorts += [(f"C_s{c:g}", "C", c, min(args.n_grid, n_sub)) for c in args.c_grid]
    elif args.stage == "fixed":
        if args.b_rate is None or args.c_sep is None:
            raise SystemExit("--stage fixed needs --b_rate and --c_sep.")
        cohorts = [("A", "A", None, n_sub), ("B", "B", args.b_rate, n_sub),
                   ("C", "C", args.c_sep, n_sub), ("C_strong", "C", args.c_strong_sep, n_sub)]
    else:
        d_mouse = calibrated_d(args.mouse_dfc, args.match_metric)
        if d_mouse is None:
            raise SystemExit(f"No dfc_subject_metrics.csv in {args.mouse_dfc} (mouse DFC run).")
        target = float(np.median(d_mouse))
        grid_manifest = json.load(open(os.path.join(args.out_dir, "sim_manifest_grid.json")))
        fitted = {}
        for letter, log in [("B", True), ("C", False)]:
            pts = []
            for name, par in grid_manifest["grid"][letter].items():
                d = calibrated_d(args.dfc_prefix + name, args.match_metric)
                if d is None:
                    raise SystemExit(f"Grid cohort {name}: no DFC output at "
                                     f"{args.dfc_prefix + name}. Run the grid stage first.")
                pts.append((par, float(np.median(d))))
                print(f"  grid {name}: median d ({args.match_metric}) = {np.median(d):.2f}")
            fitted[letter] = match_parameter(pts, target, log)
            info[letter] = dict(grid=pts, target_median_d=target, parameter=fitted[letter][0],
                                note=fitted[letter][1])
            print(f"Matched {letter}: parameter {fitted[letter][0]:.4g} "
                  f"(mouse median d = {target:.2f}; {fitted[letter][1]})")
        cohorts = [("B", "B", fitted["B"][0], n_sub), ("C", "C", fitted["C"][0], n_sub),
                   ("C_strong", "C", args.c_strong_sep, n_sub)]

    print(f"Writing {len(cohorts)} cohorts ...")
    for name, letter, par, n in cohorts:
        rows += write_cohort(name, letter, par, n, model, args, T, sim_groups, A, R_group)

    # ---- records
    stage = args.stage
    np.savez(net_f, M=M, C_noise=C_noise, R_group=R_group, gains=gains, eig=lam, hrf=tm.hrf,
             pca_A=A, state_patterns=B_pat, s_max=s_max)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(args.out_dir, f"sim_diagnostics_{stage}.csv"), index=False)
    with open(os.path.join(args.out_dir, f"cohorts_{stage}.txt"), "w") as f:
        f.write("\n".join(c[0] for c in cohorts) + "\n")
    manifest = dict(
        stage=stage, created=time.strftime("%Y-%m-%d %H:%M"),
        template=os.path.abspath(args.template) if real else None, synthetic_template=not real,
        n_template=len(Xs_list), n_roi=N, T=T, TR=args.TR, signal_rank=k,
        signal_share_of_fc_variance=float(lam[:k].sum() / lam.sum()), hrf_peak_s=args.hrf_peak,
        band_hz=[0.01, 0.1], n_states=args.n_states, dwell_s=args.dwell, seed=args.seed,
        c_variance_share=args.c_variance_share,
        cohorts={c[0]: dict(mechanism=c[1], parameter=c[2], n=c[3]) for c in cohorts},
        grid=dict(B={c[0]: c[2] for c in cohorts if c[1] == "B"},
                  C={c[0]: c[2] for c in cohorts if c[1] == "C"}) if stage == "grid" else None,
        matching=info or None, match_metric=args.match_metric,
        runtime_min=round((time.time() - t_start) / 60, 1))
    with open(os.path.join(args.out_dir, f"sim_manifest_{stage}.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # figure: mouse + the cohorts that will go through every pipeline
    Dfig = D
    if stage == "final":
        g = os.path.join(args.out_dir, "sim_diagnostics_grid.csv")
        if os.path.isfile(g):
            G = pd.read_csv(g)
            Dfig = pd.concat([G[G.cohort == "A"], D], ignore_index=True)
    if stage != "grid":
        design_figure(Dfig, args.out_dir)
    cols = ["fc_similarity", "burst_z", "fc_change_z", "kurt_pc", "kurt_cofluct", "kurt_rss"]
    pd.set_option("display.width", 200)
    print("\nMedians per cohort:")
    print(Dfig.groupby("cohort", sort=False)[cols].median().round(2).to_string())
    print(f"\nDone in {manifest['runtime_min']} min. Cohort list: "
          f"{args.out_dir}/cohorts_{stage}.txt")


if __name__ == "__main__":
    main()
