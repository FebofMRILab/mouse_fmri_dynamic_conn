#!/usr/bin/env python3
"""
mp_signal_rank.py
-------------------------------------------------------------------
How many dimensions of each subject's FC stand above noise?  An HMM-free
test of the spin-echo argument: if the SE mouse signal sits in a small
subspace buried in thermal noise, only ~10-20 eigenvalues of each mouse's
correlation matrix should exceed the noise level, and a rank-15 PCA-HMM is
justified a priori.

Two noise thresholds per subject (ROI time series standardised):
  Marchenko-Pastur  upper edge (1 + sqrt(N / T_eff))^2 of the eigenvalues of a
                    pure-noise correlation matrix. T_eff is the effective
                    number of independent samples, from each ROI's
                    autocorrelation (Bartlett): T / (1 + 2 sum_tau rho(tau)^2),
                    since band-passed BOLD has far fewer than T.
                    Approximate: treat parallel analysis as primary.
  Parallel analysis Horn's method with circularly shifted surrogates: every
                    ROI shifted by its own random offset keeps its
                    autocorrelation and distribution but loses all
                    covariance; the i-th observed eigenvalue must exceed the
                    95th percentile of the i-th surrogate eigenvalue
                    (leading eigenvalues counted until the first failure).
Reported: signal rank by each method, and the share of variance carried by
the signal eigenvalues (comparable across different numbers of ROIs).

Caveat: "signal" = any structure shared across ROIs, which includes shared
physiological or global noise. Spatial smoothing also adds shared variance.

Usage (one or more datasets; label, folder, number of subjects, TR in s):
    python mp_signal_rank.py --dataset mouse_SE11T hmm_signals 95 2 \
        --dataset human_GE3T ../human/hmm_signals 54 3
Outputs in --out_dir (default signal_rank): signal_rank_per_subject.csv,
FigS_signal_rank.{pdf,png}
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns

plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 1.2,
                     "savefig.dpi": 350, "savefig.facecolor": "white",
                     "figure.facecolor": "white"})
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
warnings.filterwarnings("ignore", message=".*findfont.*")

INK, SECONDARY, MUTED, BASELINE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
DATASET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]      # fixed order, validated slots
SEED = 20261003


def standardise(X):
    X = X - X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd <= 0] = 1
    return X / sd


def effective_samples(Xs, max_lag=None):
    """Bartlett effective sample size for correlations between ROIs:
    T / (1 + 2 sum_{tau=1..L} rho_bar(tau)^2), rho_bar = mean ROI autocorrelation."""
    T = Xs.shape[0]
    L = max_lag or min(T // 4, 150)
    F = np.fft.rfft(Xs, n=2 * T, axis=0)
    ac = np.fft.irfft(np.abs(F) ** 2, axis=0)[:L + 1]
    ac = ac / ac[0]
    rho = ac.mean(axis=1)
    # squared autocorrelation summed over all lags up to L: band-passed BOLD
    # oscillates, and the negative lobes reduce T_eff as much as positive ones
    return T / (1 + 2 * np.sum(rho[1:L + 1] ** 2))


def circular_shift(Xs, rng):
    T, N = Xs.shape
    out = np.empty_like(Xs)
    for i in range(N):
        out[:, i] = np.roll(Xs[:, i], rng.integers(0, T))
    return out


def eigvals_corr(Xs):
    C = np.corrcoef(Xs, rowvar=False)
    return np.sort(np.linalg.eigvalsh((C + C.T) / 2))[::-1]


def subject_rank(X, rng, n_shift):
    Xs = standardise(X)
    T, N = Xs.shape
    lam = eigvals_corr(Xs)
    teff = effective_samples(Xs)
    edge = (1 + np.sqrt(N / teff)) ** 2
    rank_mp = int((lam > edge).sum())
    S = np.array([eigvals_corr(circular_shift(Xs, rng)) for _ in range(n_shift)])
    thr = np.percentile(S, 95, axis=0)
    above = lam > thr
    rank_pa = int(np.argmin(above)) if not above.all() else N
    return dict(n_roi=N, T=T, T_eff=teff, mp_edge=edge, rank_mp=rank_mp, rank_pa=rank_pa,
                var_signal_mp=float(lam[:rank_mp].sum() / lam.sum()),
                var_signal_pa=float(lam[:rank_pa].sum() / lam.sum()),
                lambda1_share=float(lam[0] / lam.sum())), lam, thr


def spectrum_axis(ax, scale="log", ymin=None):
    """Y-axis of the eigenvalue-spectrum panel. 'log': log scale labelled with
    plain numbers (0.2, 0.5, 1, 2, 5, 10) instead of powers of ten.
    'linear': ordinary axis from 0."""
    if scale == "linear":
        ax.set_ylim(0, None)
        return
    ax.set_yscale("log")
    if ymin is not None:
        ax.set_ylim(ymin, None)
    lo, hi = ax.get_ylim()
    ticks = [v for v in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if lo <= v <= hi]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs=4, action="append", required=True,
                    metavar=("LABEL", "DATA_DIR", "N_SUBJECTS", "TR"))
    ap.add_argument("--file_pattern", default="signal{}.txt")
    ap.add_argument("--n_shift", type=int, default=50)
    ap.add_argument("--out_dir", default="signal_rank")
    ap.add_argument("--spectrum_scale", choices=["log", "linear"], default="log",
                    help="y-axis of the eigenvalue-spectrum panel (log: plain-number ticks)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows, spectra = [], {}
    for label, ddir, n_sub, tr in args.dataset:
        lams, thrs = [], []
        for s in range(1, int(n_sub) + 1):
            fn = os.path.join(ddir, args.file_pattern.format(s))
            if not os.path.isfile(fn):
                continue
            sep = "," if "," in open(fn).readline() else None
            X = np.loadtxt(fn, delimiter=sep).astype(float)
            if X.shape[0] < X.shape[1]:
                print(f"{label} subject {s}: {X.shape} looks like ROI x time; transposed")
                X = X.T
            r, lam, thr = subject_rank(X, rng, args.n_shift)
            rows.append(dict(dataset=label, subject_id=s, TR=float(tr), **r))
            lams.append(lam)
            thrs.append(thr)
        if lams:
            L = np.array(lams)
            spectra[label] = (np.median(L, axis=0), np.median(thrs, axis=0),
                              float(np.median([r["mp_edge"] for r in rows if r["dataset"] == label])),
                              np.percentile(L, 25, axis=0), np.percentile(L, 75, axis=0))
        print(f"{label}: {len(lams)} subjects loaded from {ddir}")
    R = pd.DataFrame(rows)
    if R.empty:
        raise SystemExit("No subjects loaded.")
    R.to_csv(os.path.join(args.out_dir, "signal_rank_per_subject.csv"), index=False)
    # i-th largest eigenvalue across subjects (median and IQR), as plotted in panel A
    pd.DataFrame([dict(dataset=lab, rank=i + 1, eig_median=sp[0][i], eig_q25=sp[3][i],
                       eig_q75=sp[4][i], pa_threshold_median=sp[1][i], mp_edge_median=sp[2])
                  for lab, sp in spectra.items() for i in range(sp[0].size)]).to_csv(
        os.path.join(args.out_dir, "signal_rank_spectrum_quantiles.csv"), index=False)

    # ---------------- figure ----------------
    labels = list(spectra)
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9), gridspec_kw=dict(wspace=0.5))
    ax = axes[0]
    for i, lab in enumerate(labels):
        lam, thr, edge, q1, q3 = spectra[lab]
        col = DATASET_COLORS[i % 3]
        k = np.arange(1, lam.size + 1)
        ax.fill_between(k, q1, q3, color=col, alpha=0.18, lw=0)
        ax.plot(k, lam, "-", color=col, lw=1.3, label=f"{lab} (median)")
        ax.plot(k, thr, "--", color=col, lw=0.9)
        ax.axhline(edge, color=col, lw=0.7, ls=":")
    ax.set_xlim(0, 40)
    lo = min(float(np.min(spectra[lab][3][:40])) for lab in labels)
    spectrum_axis(ax, args.spectrum_scale, lo * 0.7)
    ax.set_xlabel("Eigenvalue rank", fontsize=8.5)
    ax.set_ylabel("Eigenvalue (correlation matrix)", fontsize=8.5)
    ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=DATASET_COLORS[i % 3], lw=1.3, label=lab)
               for i, lab in enumerate(labels)]
    from matplotlib.patches import Patch
    handles += [Patch(facecolor=SECONDARY, alpha=0.18, lw=0, label="IQR across subjects"),
                Line2D([], [], color=SECONDARY, lw=0.9, ls="--", label="Parallel analysis, 95th pct"),
                Line2D([], [], color=SECONDARY, lw=0.7, ls=":", label="Marchenko–Pastur edge")]
    ax.legend(handles=handles, frameon=False, fontsize=6,
              loc="lower left" if args.spectrum_scale == "log" else "upper right")
    ax.set_title("Median eigenvalue spectrum", fontsize=8, loc="left")
    sns.despine(ax=ax)

    for ax, col, ylab, title in [
            (axes[1], ["rank_pa", "rank_mp"], "Signal dimensions", "Signal rank per subject"),
            (axes[2], ["var_signal_pa", "var_signal_mp"], "Variance above noise (share)",
             "Signal variance share")]:
        pos = 0
        ticks, ticklabels = [], []
        for i, lab in enumerate(labels):
            d = R[R.dataset == lab]
            c = DATASET_COLORS[i % 3]
            for j, (cc, meth) in enumerate(zip(col, ["PA", "MP"])):
                v = d[cc].to_numpy(float)
                ax.boxplot(v, positions=[pos], widths=0.6, showfliers=False, patch_artist=True,
                           medianprops=dict(color=INK, lw=1.1),
                           boxprops=dict(facecolor="white", edgecolor=c, lw=1.0),
                           whiskerprops=dict(color=c, lw=0.9), capprops=dict(color=c, lw=0.9))
                ax.scatter(pos + rng.uniform(-0.18, 0.18, v.size), v, s=6, color=c,
                           alpha=0.55 if meth == "PA" else 0.3, lw=0, zorder=3)
                ticks.append(pos)
                ticklabels.append(meth)
                pos += 1
            ax.text(pos - 1.5, -0.2, lab, transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=7, color=SECONDARY)
            pos += 0.6
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=7)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(title, fontsize=8, loc="left")
        sns.despine(ax=ax)
    for ax, letter in zip(axes, "ABC"):
        ax.text(-0.3, 1.08, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"FigS_signal_rank.{ext}"), bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
    plt.close(fig)

    # ---------------- console ----------------
    summ = R.groupby("dataset").agg(
        n=("subject_id", "size"), n_roi=("n_roi", "first"), T=("T", "median"),
        T_eff=("T_eff", "median"), rank_PA=("rank_pa", "median"),
        rank_PA_q1=("rank_pa", lambda x: x.quantile(0.25)),
        rank_PA_q3=("rank_pa", lambda x: x.quantile(0.75)),
        rank_MP=("rank_mp", "median"), var_signal_PA=("var_signal_pa", "median"),
        lambda1_share=("lambda1_share", "median"))
    pd.set_option("display.width", 200)
    print("\n" + summ.round(3).to_string())
    print("\nPA = parallel analysis (circular shifts, 95th pct); MP = Marchenko-Pastur with "
          "Bartlett T_eff.")
    print(f"Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()