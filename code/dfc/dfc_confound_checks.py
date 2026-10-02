#!/usr/bin/env python3
"""
dfc_confound_checks.py
-------------------------------------------------------------------
Is the positive calibrated non-stationarity test (DCC / GARCH / cPCA
envelope) explained by stationary non-Gaussian transients, head motion, or
slow drift over the scan, rather than by time-varying coupling?

The calibrated test compares each mouse with a stationary GAUSSIAN replica,
so it rejects anything the data have that such a process lacks. Three
alternatives are checked here, per mouse, with no MATLAB rerun:

1. Non-Gaussian transients: filtered spikes or brief global events (no
   censoring, no GSR) are stationary but bursty, and shared across ROIs.
     kurt_roi   median excess kurtosis of the ROI time series
     kurt_gs    excess kurtosis of the global signal (mean over ROIs)
     kurt_pc    mean excess kurtosis of the group-PCA components DCC uses
     burst_z    kurtosis of instantaneous total power over the components
                (co-occurring bursts), as z against the mouse's own phase
                surrogates (which are Gaussian and stationary)
2. Head motion: mean_FD, max_FD, prop_FD_over_0.05 (RABIES FD summary).
3. Drift across the scan (e.g. sedation depth):
     coupling_drift   mean off-diagonal r, 2nd half minus 1st half
     amplitude_drift  log(median ROI SD, 2nd half / 1st half)
     fc_change_z      1 - corr(FC 1st half, FC 2nd half), as z against
                      phase surrogates (more change than stationarity allows)

4. Time scale of the coupling changes (descriptive): half-life of a DCC
   correlation shock, 0.5 = (a + b)^n, in seconds (decimated TR), for mice
   whose DCC a exceeds their surrogates (z > 1.645), next to the same
   quantity for their surrogates. On simulated FC-switching mice the
   observed half-life was about twice the surrogates' (~28 vs ~16 s).

Outcome per mouse and measure: d = z_observed - z_replica (the quantity the
calibrated test is run on).

Signatures on simulated band-passed data (8 mice each, median):
                         burst_z  fc_change_z  kurt_gs  coupling_drift
  stationary Gaussian       0.7        -0.2      0.0        0.00
  shared filtered spikes   29.7         0.6     20.7       -0.01
  covariance switching      0.1         6.0     -0.1        0.00
  slow drift (global)      -0.2         0.4      0.8        0.28
So: spikes -> bursts/kurtosis with little FC change; switching -> FC change
without bursts; drift -> consistent coupling_drift across mice.
Caveat: switching between states that also differ in overall variance does
raise kurtosis and burst_z (seen on the pipeline's own test mice), so a
kurtosis-d association alone is not proof of artifact. Motion associations
and an effect confined to high-burst, low-fc_change mice would be.

Analyses
  A. Spearman rho between every outcome and every covariate (BH-FDR).
  B. Calibrated test repeated within the low and high halves of each
     covariate (median split), and after excluding motion outliers.
     If the effect is present in the low-kurtosis / low-motion halves, those
     explanations are weakened; if it is confined to the high halves, they
     are supported.
  C. Drift: consistent direction across mice (Wilcoxon on the signed log
     ratios), and change beyond stationarity (Wilcoxon on fc_change_z > 0);
     coupling and amplitude by scan quarter.

Usage
  python dfc_confound_checks.py --in_dir dfc_dcc_cpca --data_dir hmm_signals \
      --motion mouse_motion.csv --pca_dim 25
Outputs go to <in_dir>/confound_checks/.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ---------------- settings ----------------
PRIMARY_TESTS = ["dcc_a", "dcc_LR", "corr_fluct", "garch_alpha", "garch_LR",
                 "vol_cv", "cpca_roi_c1_env_cv"]
SHORT = {"dcc_a": "DCC a", "dcc_LR": "DCC LR", "corr_fluct": "Corr. fluct.",
         "garch_alpha": "GARCH α", "garch_LR": "GARCH LR", "vol_cv": "Volatility CV",
         "cpca_roi_c1_env_cv": "Envelope CV, C1"}
COVARIATES = {
    "mean_FD": "Mean FD", "max_FD": "Max FD", "prop_FD_over_0.05": "FD > 0.05 (prop.)",
    "kurt_roi": "Kurtosis, ROIs", "kurt_gs": "Kurtosis, global signal",
    "kurt_pc": "Kurtosis, PCs", "burst_z": "Shared bursts (z)",
    "coupling_drift": "Coupling drift", "amplitude_drift": "Amplitude drift",
    "fc_change_z": "FC change, halves (z)",
}
SPLIT_COVARIATES = ["mean_FD", "kurt_gs", "burst_z", "fc_change_z"]
N_SURR = 19
N_BOOT = 10000
SEED = 20260929

INK, SECONDARY, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LOW_HIGH = {"low": "#2a78d6", "high": "#eb6834"}
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#1f5fae", "#2a78d6", "#f0efec", "#e34948", "#b3262f"])

# Same style calls as the static FC figures / plot_dfc_results.py
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 1.2,
                     "savefig.dpi": 350, "savefig.facecolor": "white",
                     "figure.facecolor": "white"})
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
warnings.filterwarnings("ignore", message=".*findfont.*")


# ---------------- signal helpers (match run_dcc_cpca_null.m) ----------------
def standardise(X):
    Xs = X - X.mean(axis=0)
    sd = Xs.std(axis=0, ddof=1)
    sd[sd <= 0] = 1
    return Xs / sd


def group_pca(Xs_list, d):
    """As local_group_pca: eig of the pooled covariance of standardised data."""
    C = sum(X.T @ X for X in Xs_list) / (sum(X.shape[0] for X in Xs_list) - 1)
    lam, V = np.linalg.eigh((C + C.T) / 2)
    o = np.argsort(lam)[::-1]
    A = V[:, o[:d]]
    for k in range(A.shape[1]):
        A[:, k] *= np.sign(A[np.argmax(np.abs(A[:, k])), k])
    return A, lam[o[:d]] / lam.sum()


def phase_randomise(X, rng):
    """Multivariate phase randomisation: the same random phases for every
    column, so all auto- and cross-spectra are preserved."""
    T = X.shape[0]
    F = np.fft.rfft(X, axis=0)
    nb = F.shape[0]
    ph = rng.uniform(0, 2 * np.pi, nb)
    ph[0] = 0
    if T % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(F * np.exp(1j * ph)[:, None], n=T, axis=0)


def excess_kurtosis(x):
    return stats.kurtosis(x, axis=0, fisher=True, bias=False)


def mean_offdiag_r(X):
    R = np.corrcoef(X, rowvar=False)
    return R[np.triu_indices_from(R, 1)].mean()


def burst_stat(P):
    """Kurtosis of instantaneous total power across standardised components."""
    Ps = P / P.std(axis=0, ddof=1)
    return excess_kurtosis((Ps ** 2).sum(axis=1))


def fc_change(X):
    h = X.shape[0] // 2
    iu = np.triu_indices(X.shape[1], 1)
    r1 = np.corrcoef(X[:h], rowvar=False)[iu]
    r2 = np.corrcoef(X[h:], rowvar=False)[iu]
    return 1 - np.corrcoef(r1, r2)[0, 1]


def per_mouse(X, A, rng, n_surr):
    Xs = standardise(X)
    P = Xs @ A
    T = X.shape[0]
    h = T // 2
    sd1 = X[:h].std(axis=0, ddof=1)
    sd2 = X[h:].std(axis=0, ddof=1)
    out = dict(
        kurt_roi=float(np.median(excess_kurtosis(Xs))),
        kurt_gs=float(excess_kurtosis(Xs.mean(axis=1))),
        kurt_pc=float(np.mean(excess_kurtosis(P))),
        coupling_drift=float(mean_offdiag_r(Xs[h:]) - mean_offdiag_r(Xs[:h])),
        amplitude_drift=float(np.median(np.log(sd2 / sd1))),
    )
    b_obs, c_obs = burst_stat(P), fc_change(Xs)
    b_s, c_s = [], []
    for _ in range(n_surr):
        S = phase_randomise(Xs, rng)
        b_s.append(burst_stat(S @ A))
        c_s.append(fc_change(S))
    out["burst_z"] = float((b_obs - np.mean(b_s)) / np.std(b_s, ddof=1))
    out["fc_change_z"] = float((c_obs - np.mean(c_s)) / np.std(c_s, ddof=1))
    q = np.array_split(np.arange(T), 4)
    for i, idx in enumerate(q, 1):
        out[f"coupling_q{i}"] = float(mean_offdiag_r(Xs[idx]))
        out[f"amplitude_q{i}"] = float(np.median(X[idx].std(axis=0, ddof=1)))
    return out


# ---------------- statistics ----------------
def bh(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    m = q.size
    if m:
        o = np.argsort(q)
        r = q[o] * m / np.arange(1, m + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        adj = np.empty(m)
        adj[o] = np.minimum(r, 1)
        out[ok] = adj
    return out


def wilcoxon_greater(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size < 6:
        return np.nan
    return float(stats.wilcoxon(d, alternative="greater").pvalue)


def median_ci(x, rng, n_boot=N_BOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan, np.nan, np.nan
    m = np.median(x[rng.integers(0, x.size, (n_boot, x.size))], axis=1)
    return float(np.median(x)), *np.percentile(m, [2.5, 97.5]).tolist()


def fmt_p(p):
    if not np.isfinite(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3g}"


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="dfc_dcc_cpca")
    ap.add_argument("--data_dir", default="hmm_signals")
    ap.add_argument("--file_pattern", default="signal{}.txt")
    ap.add_argument("--motion", default="mouse_motion.csv")
    ap.add_argument("--pca_dim", type=int, default=25)
    ap.add_argument("--n_surr", type=int, default=N_SURR)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out = args.out_dir or os.path.join(args.in_dir, "confound_checks")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- outcomes: d = z_observed - z_replica
    m = pd.read_csv(os.path.join(args.in_dir, "dfc_subject_metrics.csv"))
    m = m[m.metric.isin(PRIMARY_TESTS)]
    if "phase_replica_z" not in m or m.phase_replica_z.isna().all():
        raise SystemExit("No stationary-replica z in dfc_subject_metrics.csv (n_replica = 0).")
    m = m.assign(d=m.phase_z - m.phase_replica_z)
    D = m.pivot(index="subject_id", columns="metric", values="d")[
        [t for t in PRIMARY_TESTS if t in set(m.metric)]]
    groups = m.drop_duplicates("subject_id").set_index("subject_id").group
    tests = list(D.columns)

    # ---- time scale of DCC coupling changes (from the diagnostics rows)
    cfg = {}
    cfg_path = os.path.join(args.in_dir, "dfc_config.txt")
    if os.path.isfile(cfg_path):
        for line in open(cfg_path):
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    try:
        tr_dcc = float(cfg.get("TR", 2)) * float(cfg.get("dcc_decimate", 1))
    except ValueError:
        tr_dcc = float(cfg.get("TR", 2))
    full = pd.read_csv(os.path.join(args.in_dir, "dfc_subject_metrics.csv"))
    a_z = full[full.metric == "dcc_a"].set_index("subject_id").phase_z
    pers = full[full.metric == "dcc_persist"].set_index("subject_id")
    hl = lambda p: tr_dcc * np.log(0.5) / np.log(np.clip(p, 1e-6, 0.999))
    H = pd.DataFrame({"dcc_a_z": a_z, "persist_obs": pers.observed,
                      "persist_surr_mean": pers.phase_mean})
    H["half_life_obs_s"] = hl(H.persist_obs)
    H["half_life_surr_s"] = hl(H.persist_surr_mean)
    H["dynamic"] = H.dcc_a_z > 1.645


    # ---- motion
    mot = pd.read_csv(args.motion)
    mot = mot.rename(columns={"index": "subject_id"}).set_index("subject_id")
    common = D.index.intersection(mot.index)
    if "genotype" in mot:
        bad = [s for s in common if str(mot.loc[s, "genotype"]).strip() != str(groups.loc[s]).strip()]
        if bad:
            print(f"WARNING: group labels differ between motion file and metrics for "
                  f"subjects {bad[:10]}{' ...' if len(bad) > 10 else ''}: check the id mapping.")
    print(f"Mice with metrics: {len(D)}; with motion: {len(common)}")

    # ---- signals: group PCA then per-mouse measures
    Xraw = {}
    for s in D.index:
        fn = os.path.join(args.data_dir, args.file_pattern.format(int(s)))
        if os.path.isfile(fn):
            sep = "," if "," in open(fn).readline() else None
            Xraw[int(s)] = np.loadtxt(fn, delimiter=sep).astype(float)
    if not Xraw:
        raise SystemExit(f"No signal files found in {args.data_dir}")
    A, expl = group_pca([standardise(X) for X in Xraw.values()],
                        min(args.pca_dim, next(iter(Xraw.values())).shape[1]))
    print(f"Loaded {len(Xraw)} mice; group PCA: PC1 {100 * expl[0]:.1f}% of variance, "
          f"{int((A[:, 0] > 0).sum())}/{A.shape[0]} ROIs load positively on PC1; "
          f"{A.shape[1]} PCs = {100 * expl.sum():.1f}%")
    rows = []
    for s, X in Xraw.items():
        rows.append(dict(subject_id=s, **per_mouse(X, A, rng, args.n_surr)))
    C = pd.DataFrame(rows).set_index("subject_id")
    for c in ["mean_FD", "max_FD", "prop_FD_over_0.05"]:
        if c in mot:
            C[c] = mot[c]
    C["group"] = groups
    covs = [c for c in COVARIATES if c in C]
    per = C.join(D).join(H)
    per.to_csv(os.path.join(out, "confound_measures_per_mouse.csv"))

    # ---- A. correlations
    corr = []
    for t in tests:
        for c in covs:
            x, y = per[c], per[t]
            ok = x.notna() & y.notna()
            rho, p = stats.spearmanr(x[ok], y[ok]) if ok.sum() > 5 else (np.nan, np.nan)
            corr.append(dict(outcome=t, covariate=c, n=int(ok.sum()), rho=rho, p=p))
    corr = pd.DataFrame(corr)
    corr["q_BH"] = bh(corr.p)
    corr.to_csv(os.path.join(out, "confound_correlations.csv"), index=False)

    # ---- B. calibrated test within covariate halves and after exclusions
    strat = []
    for t in tests:
        med, lo, hi = median_ci(per[t], rng)
        strat.append(dict(outcome=t, subset="all mice", n=int(per[t].notna().sum()),
                          median_d=med, ci_low=lo, ci_high=hi, p=wilcoxon_greater(per[t])))
    for c in [c for c in SPLIT_COVARIATES if c in per]:
        cut = per[c].median()
        for half, mask in [("low", per[c] <= cut), ("high", per[c] > cut)]:
            for t in tests:
                x = per.loc[mask, t]
                med, lo, hi = median_ci(x, rng)
                strat.append(dict(outcome=t, subset=f"{c} {half}", n=int(x.notna().sum()),
                                  median_d=med, ci_low=lo, ci_high=hi, p=wilcoxon_greater(x)))
    if "mean_FD" in per:
        q1, q3 = per.mean_FD.quantile([0.25, 0.75])
        outl = per.index[per.mean_FD > q3 + 3 * (q3 - q1)].tolist()
        print(f"Motion outliers (mean FD > Q3 + 3 IQR): {outl}")
        for t in tests:
            x = per.loc[~per.index.isin(outl), t]
            med, lo, hi = median_ci(x, rng)
            strat.append(dict(outcome=t, subset="motion outliers excluded", n=int(x.notna().sum()),
                              median_d=med, ci_low=lo, ci_high=hi, p=wilcoxon_greater(x)))
    strat = pd.DataFrame(strat)
    strat.to_csv(os.path.join(out, "calibrated_test_by_subset.csv"), index=False)

    # ---- C. drift
    drift = []
    for c, lab in [("coupling_drift", "coupling r, 2nd minus 1st half (signed)"),
                   ("amplitude_drift", "amplitude, 2nd vs 1st half (signed)")]:
        x = per[c].dropna()
        drift.append(dict(measure=c, test=lab, n=len(x), median=x.median(),
                          frac_positive=float((x > 0).mean()),
                          p_two_sided=float(stats.wilcoxon(x).pvalue)))
    x = per.fc_change_z.dropna()
    drift.append(dict(measure="fc_change_z", test="FC change between halves vs stationarity (> 0)",
                      n=len(x), median=x.median(), frac_positive=float((x > 0).mean()),
                      p_two_sided=np.nan, p_greater=wilcoxon_greater(x)))
    x = per.burst_z.dropna()
    drift.append(dict(measure="burst_z", test="shared bursts vs Gaussian stationarity (> 0)",
                      n=len(x), median=x.median(), frac_positive=float((x > 0).mean()),
                      p_two_sided=np.nan, p_greater=wilcoxon_greater(x)))
    hd = per.loc[per.dynamic == True, "half_life_obs_s"].dropna()  # noqa: E712
    hs = per.loc[per.dynamic == True, "half_life_surr_s"]  # noqa: E712
    ok = hd.index.intersection(hs.dropna().index)
    drift.append(dict(measure="dcc_half_life_s",
                      test="half-life of coupling shocks (s), mice with DCC a z > 1.645; "
                           "frac_positive = share longer than own surrogates",
                      n=len(hd), median=hd.median() if len(hd) else np.nan,
                      frac_positive=float((hd[ok] > hs[ok]).mean()) if len(ok) else np.nan,
                      p_two_sided=np.nan,
                      p_greater=wilcoxon_greater(hd[ok] - hs[ok]) if len(ok) else np.nan))
    drift.append(dict(measure="dcc_half_life_surr_s", test="same, surrogate mean persistence",
                      n=int(hs.notna().sum()), median=hs.median(), frac_positive=np.nan,
                      p_two_sided=np.nan, p_greater=np.nan))
    drift = pd.DataFrame(drift)
    drift.to_csv(os.path.join(out, "drift_and_burst_tests.csv"), index=False)

    # ---- figure
    fig = plt.figure(figsize=(8.0, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.15, 1.0],
                          hspace=0.55, wspace=0.55)

    # A: correlation heatmap
    ax = fig.add_subplot(gs[0, 0])
    R = corr.pivot(index="outcome", columns="covariate", values="rho").reindex(index=tests, columns=covs)
    Q = corr.pivot(index="outcome", columns="covariate", values="q_BH").reindex(index=tests, columns=covs)
    vmax = max(0.3, float(np.nanmax(np.abs(R.values))))
    im = ax.imshow(R.values, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            v, q = R.iat[i, j], Q.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}" + ("*" if q < 0.05 else ""), ha="center", va="center",
                        fontsize=5.5, color="white" if abs(v) > 0.6 * vmax else INK)
    ax.set_xticks(range(len(covs)))
    ax.set_xticklabels([COVARIATES[c] for c in covs], rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=6.5)
    ax.set_yticks(range(len(tests)))
    ax.set_yticklabels([SHORT.get(t, t) for t in tests], fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman ρ", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    ax.set_title("z$_{obs}$ − z$_{replica}$ vs covariates (* q < 0.05)", fontsize=8, loc="left")
    ax.text(-0.42, 1.04, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")

    # B: time scale of coupling changes
    ax = fig.add_subplot(gs[0, 1])
    dyn = per[per.dynamic == True]  # noqa: E712
    vals = dyn.half_life_obs_s.dropna().to_numpy()
    if vals.size:
        yj = rng.uniform(-0.12, 0.12, vals.size)
        ax.scatter(vals, yj, s=12, color=INK, alpha=0.6, lw=0)
        ax.plot([np.median(vals)] * 2, [-0.25, 0.25], color=INK, lw=1.5)
        sv = dyn.half_life_surr_s.dropna().to_numpy()
        if sv.size:
            ax.scatter(sv, yj[:sv.size] - 0.6, s=12, color=MUTED, alpha=0.6, lw=0)
            ax.plot([np.median(sv)] * 2, [-0.85, -0.35], color=MUTED, lw=1.5)
        allv = np.r_[vals, dyn.half_life_surr_s.dropna().to_numpy()]
        if allv.max() / max(allv.min(), 1e-9) > 20:      # log only for wide ranges
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([0, -0.6])
        ax.set_yticklabels(["Observed", "Surrogate\nmean"], fontsize=7)
        ax.set_ylim(-1.0, 0.6)
        ax.set_xlabel("Half-life of a coupling shock (s)", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"DCC time scale, mice with DCC a z > 1.645 (n = {vals.size})",
                     fontsize=8, loc="left")
    else:
        ax.text(0.5, 0.5, "No mouse with DCC a z > 1.645", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
    sns.despine(ax=ax)
    ax.text(-0.3, 1.04, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")

    # C: calibrated effect in low vs high halves (DCC and GARCH measures)
    ax = fig.add_subplot(gs[1, 0])
    splits = [c for c in SPLIT_COVARIATES if c in per]
    show = [t for t in ["dcc_LR", "corr_fluct", "garch_LR", "cpca_roi_c1_env_cv"] if t in tests]
    ypos, labels = [], []
    y0 = 0
    for c in splits:
        for t in show:
            for k, half in enumerate(["low", "high"]):
                r = strat[(strat.outcome == t) & (strat.subset == f"{c} {half}")].iloc[0]
                yy = y0 + (0.18 if half == "low" else -0.18)
                ax.plot([r.ci_low, r.ci_high], [yy, yy], color=LOW_HIGH[half], lw=1.1)
                ax.plot(r.median_d, yy, "o", ms=3.5, color=LOW_HIGH[half])
            ypos.append(y0)
            labels.append(f"{SHORT.get(t, t)} | {COVARIATES[c]}")
            y0 -= 1
        y0 -= 0.5
    ax.axvline(0, color=BASELINE, lw=0.8, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=5.8)
    ax.set_xlabel("Median z$_{obs}$ − z$_{replica}$ (95% CI)", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.plot([], [], "o-", color=LOW_HIGH["low"], ms=3.5, lw=1.1, label="Below median")
    ax.plot([], [], "o-", color=LOW_HIGH["high"], ms=3.5, lw=1.1, label="Above median")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=2, borderaxespad=0.1)
    sns.despine(ax=ax)
    ax.set_title("Calibrated effect within covariate halves", fontsize=8, loc="left", pad=14)
    ax.text(-0.75, 1.04, "C", transform=ax.transAxes, fontsize=11, fontweight="bold")

    # D: drift across quarters, as change from each mouse's own mean
    sub = gs[1, 1].subgridspec(2, 1, hspace=0.25)
    for i, (key, lab, col, tf) in enumerate([
            ("coupling", "Δ mean coupling (r)", "#2a78d6", lambda M: M - M.mean(axis=1, keepdims=True)),
            ("amplitude", "Δ ROI SD (%)", "#eb6834",
             lambda M: 100 * (M / M.mean(axis=1, keepdims=True) - 1))]):
        ax = fig.add_subplot(sub[i, 0])
        M = tf(per[[f"{key}_q{j}" for j in range(1, 5)]].to_numpy(float))
        mu = np.nanmedian(M, axis=0)          # median: robust to a few spiky mice
        boot = np.array([np.nanmedian(M[rng.integers(0, len(M), len(M))], axis=0)
                         for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        xq = np.arange(1, 5)
        ax.fill_between(xq, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(xq, mu, "o-", color=col, ms=3.5, lw=1.2)
        ax.axhline(0, color=BASELINE, lw=0.8, zorder=0)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylabel(lab, fontsize=7.5)
        ax.tick_params(labelsize=7)
        sns.despine(ax=ax)
        if i == 0:
            ax.set_xticklabels([])
            ax.set_title("Drift across the scan (median, 95% CI)", fontsize=8, loc="left")
            ax.text(-0.3, 1.08, "D", transform=ax.transAxes, fontsize=11, fontweight="bold")
        else:
            ax.set_xlabel("Scan quarter", fontsize=8.5)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"FigS_dfc_confounds.{ext}"), bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
    plt.close(fig)

    # ---- console summary
    pd.set_option("display.width", 200)
    print("\nA. Strongest associations (|rho|, top 12):")
    print(corr.reindex(corr.rho.abs().sort_values(ascending=False).index).head(12)
          .round(3).to_string(index=False))
    print(f"\n   q < 0.05: {int((corr.q_BH < 0.05).sum())} of {len(corr)} outcome x covariate pairs")
    print("\nB. Calibrated test by subset (median d, one-sided Wilcoxon p):")
    tab = strat.assign(cell=[f"{a:.2f} (p={fmt_p(b)})" for a, b in zip(strat.median_d, strat.p)]) \
        .pivot(index="subset", columns="outcome", values="cell")[tests]
    print(tab.to_string())
    print("\nC. Drift and bursts:")
    print(drift.round(4).to_string(index=False))
    print("\nHow to read it: spikes -> high burst_z / kurtosis, d concentrated in the "
          "high-burst half, little FC change; switching -> fc_change_z > 0 and d present in "
          "low-burst mice; drift -> consistent sign of coupling_drift or amplitude_drift; "
          "motion -> d correlates with FD and shrinks in the low-FD half.")
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
