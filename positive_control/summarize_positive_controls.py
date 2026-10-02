#!/usr/bin/env python3
"""
summarize_positive_controls.py
-------------------------------------------------------------------
Sets the outputs of the unchanged DFC and PCA-HMM pipelines on the simulated
cohorts (sim_positive_controls.py) beside the same outputs for the mouse data.

Per cohort (Mouse, A Gaussian, B events, C switching, C_strong):
  DCC / GARCH / cPCA  calibrated excess d = z_observed - z_replica (phase null)
                      for the 7 pre-specified test measures: median, 95% CI,
                      one-sided Wilcoxon (as dfc_group_tests_calibrated.csv)
  Half-life           of a DCC coupling shock, observed vs the mouse's own
                      surrogates, for mice with DCC a z > 1.645 (as
                      dfc_confound_checks.py): the time-scale discriminator
  Bursts / FC change  burst_z and fc_change_z (as dfc_confound_checks.py) and
                      the kurtosis of the co-fluctuation magnitude (RSS)
  PCA-HMM             the pre-registered decision from analyze_confirm_lowrank.py
                      (run automatically if its output is missing): cells with
                      state structure, best cell, calibrated median
  State recovery      C and C_strong only: how well the lowrank-15 full-fit
                      state time courses match the simulated states (best
                      one-to-one matching; chance level from circularly
                      shifted truths)
Mouse vs each cohort: two-sided Mann-Whitney on every per-mouse measure.

Outputs in --out_dir (default positive_control_summary):
  positive_control_table.csv      long table, every number
  positive_control_schematic.csv  the rows of the schematic's table, filled in
  positive_control_vs_mouse.csv   per-mouse measures, mouse vs each cohort
  FigS_positive_controls.{png,pdf}
  FigS_positive_control_matching.{png,pdf}   grid stage: strength vs DCC excess

Usage (from the folder with the pipeline outputs):
  python summarize_positive_controls.py
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 1.2,
                     "savefig.dpi": 350, "savefig.facecolor": "white",
                     "figure.facecolor": "white"})
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
warnings.filterwarnings("ignore", message=".*findfont.*")

INK, SECONDARY, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"Mouse": INK, "A": MUTED, "B": "#eb6834", "C": "#2a78d6", "C_strong": "#1f5fae"}
LABELS = {"Mouse": "Mouse", "A": "A Gaussian", "B": "B Events", "C": "C Switching",
          "C_strong": "C strong"}
PRIMARY = ["dcc_a", "dcc_LR", "corr_fluct", "garch_alpha", "garch_LR", "vol_cv",
           "cpca_roi_c1_env_cv"]
SHORT = {"dcc_a": "DCC a", "dcc_LR": "DCC LR", "corr_fluct": "Corr. fluct.",
         "garch_alpha": "GARCH α", "garch_LR": "GARCH LR", "vol_cv": "Volatility CV",
         "cpca_roi_c1_env_cv": "Envelope CV, C1"}
SEED = 20261005
N_BOOT = 10000


# ---------------- helpers ----------------
def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3g}"


def wilcoxon_greater(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    return float(stats.wilcoxon(d, alternative="greater").pvalue) if d.size >= 6 else np.nan


def wilcoxon_two(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    return float(stats.wilcoxon(d).pvalue) if d.size >= 6 else np.nan


def median_ci(x, rng):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return (float(np.median(x)) if x.size else np.nan), np.nan, np.nan
    m = np.median(x[rng.integers(0, x.size, (N_BOOT, x.size))], axis=1)
    return float(np.median(x)), *np.percentile(m, [2.5, 97.5]).tolist()


def read_dfc(dfc_dir):
    f = os.path.join(dfc_dir, "dfc_subject_metrics.csv")
    if not os.path.isfile(f):
        return None
    m = pd.read_csv(f)
    cfg = {}
    cp = os.path.join(dfc_dir, "dfc_config.txt")
    if os.path.isfile(cp):
        for line in open(cp):
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    try:
        tr_dcc = float(cfg.get("TR", 2)) * float(cfg.get("dcc_decimate", 1))
    except ValueError:
        tr_dcc = float(cfg.get("TR", 2))
    d = (m[m.metric.isin(PRIMARY)].assign(d=lambda x: x.phase_z - x.phase_replica_z)
         .pivot(index="subject_id", columns="metric", values="d"))
    hl = lambda p: tr_dcc * np.log(0.5) / np.log(np.clip(p, 1e-6, 0.999))
    pers = m[m.metric == "dcc_persist"].set_index("subject_id")
    H = pd.DataFrame({"a_z": m[m.metric == "dcc_a"].set_index("subject_id").phase_z,
                      "hl_obs": hl(pers.observed), "hl_surr": hl(pers.phase_mean)})
    H["ratio"] = H.hl_obs / H.hl_surr
    H["dynamic"] = H.a_z > 1.645
    return dict(d=d, H=H, tr_dcc=tr_dcc)


def read_hmm(root, analyze_script):
    f = os.path.join(root, "analysis", "confirm_cells.csv")
    if not os.path.isfile(f) and os.path.isdir(os.path.join(root, "observed")):
        if analyze_script and os.path.isfile(analyze_script):
            print(f"  running {os.path.basename(analyze_script)} on {root} ...")
            subprocess.run([sys.executable, analyze_script, "--root", root], check=False,
                           stdout=subprocess.DEVNULL)
        else:
            print(f"  {root}: no analysis/confirm_cells.csv and analyze_confirm_lowrank.py "
                  f"not found (--analyze_script)")
    return pd.read_csv(f) if os.path.isfile(f) else None


def load_gamma(f):
    try:
        import h5py
        with h5py.File(f, "r") as h:
            G = np.array(h["Gamma"]).T
            T_sub = np.array(h["T_sub"]).ravel().astype(int)
            ids = np.array(h["subject_ids"]).ravel().astype(int)
        return G, T_sub, ids
    except ImportError:
        raise SystemExit("h5py is needed to read the -v7.3 gamma files (pip install h5py).")
    except OSError:
        from scipy.io import loadmat
        d = loadmat(f)
        return d["Gamma"], d["T_sub"].ravel().astype(int), d["subject_ids"].ravel().astype(int)


def state_recovery(gamma_file, truth, rng, n_null=200):
    """Mean correlation of best-matched (state time course, true state indicator)
    pairs, pooled over mice; chance = 95th percentile with each mouse's truth
    circularly shifted (keeps dwell times, breaks alignment)."""
    G, T_sub, ids = load_gamma(gamma_file)
    Q = int(truth.state.max())
    tv = []
    for s, T in zip(ids, T_sub):
        t = truth[truth.subject_id == s].sort_values("volume").state.to_numpy()[:T]
        if t.size != T:
            return None
        tv.append(t)

    def score(tl):
        I = np.eye(Q)[np.concatenate(tl) - 1]
        C = np.corrcoef(G.T, I.T)[:G.shape[1], G.shape[1]:]
        C = np.nan_to_num(C)
        r, c = linear_sum_assignment(-C)
        return float(C[r, c].mean())
    obs = score(tv)
    null = [score([np.roll(t, rng.integers(20, t.size - 20)) for t in tv]) for _ in range(n_null)]
    return dict(K=G.shape[1], recovery=obs, chance95=float(np.percentile(null, 95)),
                chance_median=float(np.median(null)))


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_dir", default="sim_positive_control")
    ap.add_argument("--mouse_dfc", default="dfc_dcc_cpca")
    ap.add_argument("--mouse_hmm", default="hmm_confirm_lowrank")
    ap.add_argument("--dfc_prefix", default="dfc_sim_")
    ap.add_argument("--hmm_prefix", default="hmm_confirm_sim_")
    ap.add_argument("--analyze_script", default="analyze_confirm_lowrank.py")
    ap.add_argument("--out_dir", default="positive_control_summary")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cohorts = ["Mouse", "A", "B", "C", "C_strong"]
    dfc_dirs = {"Mouse": args.mouse_dfc, **{c: args.dfc_prefix + c for c in cohorts[1:]}}
    hmm_dirs = {"Mouse": args.mouse_hmm, **{c: args.hmm_prefix + c for c in cohorts[1:]}}

    # ---- per-mouse measures
    diag = []
    for f in sorted(glob.glob(os.path.join(args.sim_dir, "sim_diagnostics_*.csv"))):
        diag.append(pd.read_csv(f))
    diag = (pd.concat(diag, ignore_index=True).drop_duplicates(["cohort", "subject_id"], keep="last")
            if diag else pd.DataFrame(columns=["cohort", "subject_id"]))
    dfc = {c: read_dfc(dfc_dirs[c]) for c in cohorts}
    for c in cohorts:
        print(f"{c:9s} DFC: {'found' if dfc[c] else 'missing'} ({dfc_dirs[c]})")
    per = []                                       # long: cohort, subject, measure, value
    for c in cohorts:
        if dfc[c] is not None:
            for met in dfc[c]["d"].columns:
                for s, v in dfc[c]["d"][met].items():
                    per.append((c, s, f"d_{met}", v))
            H = dfc[c]["H"]
            for s, v in H.loc[H.dynamic, "ratio"].items():
                per.append((c, s, "half_life_ratio", v))
            for s, v in H.hl_obs.items():
                per.append((c, s, "half_life_obs_s", v))
        dd = diag[diag.cohort == c]
        for col in ["burst_z", "fc_change_z", "kurt_rss", "fc_similarity"]:
            if col in dd:
                per += [(c, s, col, v) for s, v in zip(dd.subject_id, dd[col])]
    P = pd.DataFrame(per, columns=["cohort", "subject_id", "measure", "value"])

    # ---- cohort summaries
    rows = []
    for c in cohorts:
        if dfc[c] is not None:
            for met in PRIMARY:
                if met in dfc[c]["d"]:
                    x = dfc[c]["d"][met].to_numpy(float)
                    med, lo, hi = median_ci(x, rng)
                    rows.append(dict(cohort=c, measure=f"calibrated d, {met}", value=med,
                                     ci_low=lo, ci_high=hi, p=wilcoxon_greater(x),
                                     n=int(np.isfinite(x).sum()),
                                     note="one-sided Wilcoxon d > 0"))
            H = dfc[c]["H"]
            Hd = H[H.dynamic].dropna(subset=["ratio"])
            for lab, sub in [("dynamic mice", Hd), ("all mice", H.dropna(subset=["ratio"]))]:
                rows.append(dict(cohort=c, measure=f"half-life ratio obs/surr ({lab})",
                                 value=float(sub.ratio.median()) if len(sub) else np.nan,
                                 ci_low=float(sub.hl_obs.median()) if len(sub) else np.nan,
                                 ci_high=float(sub.hl_surr.median()) if len(sub) else np.nan,
                                 p=wilcoxon_two(np.log(sub.ratio)), n=len(sub),
                                 note="ci_low/ci_high hold median half-life obs/surr (s); "
                                      "two-sided Wilcoxon on log ratio"))
        dd = diag[diag.cohort == c]
        for col in ["fc_similarity", "burst_z", "fc_change_z", "kurt_rss", "kurt_cofluct"]:
            if col in dd and dd[col].notna().any():
                med, lo, hi = median_ci(dd[col], rng)
                rows.append(dict(cohort=c, measure=col, value=med, ci_low=lo, ci_high=hi,
                                 p=wilcoxon_greater(dd[col]) if col in ("burst_z", "fc_change_z")
                                 else np.nan, n=len(dd), note=""))

    hmm = {c: read_hmm(hmm_dirs[c], args.analyze_script) for c in cohorts}
    for c in cohorts:
        T = hmm[c]
        if T is None:
            continue
        best = T.sort_values("p_calibrated_holm").iloc[0]
        rows.append(dict(cohort=c, measure="HMM cells with state structure",
                         value=int(T.state_structure.astype(bool).sum()), n=len(T),
                         ci_low=np.nan, ci_high=np.nan, p=float(best.p_calibrated_holm),
                         note=f"best cell {best.model} K={int(best.K)}, calibrated median "
                              f"{best.calibrated_median:.3f}; p = its Holm p"))

    truth_rec = []
    for c in ["C", "C_strong"]:
        tf = os.path.join(args.sim_dir, c, "truth_states.csv")
        if not os.path.isfile(tf):
            continue
        truth = pd.read_csv(tf)
        for gf in sorted(glob.glob(os.path.join(hmm_dirs[c], "observed", "gamma_lowrank_15_K*.mat"))):
            r = state_recovery(gf, truth, rng)
            if r:
                truth_rec.append(dict(cohort=c, **r))
                rows.append(dict(cohort=c, measure=f"state recovery, lowrank 15, K={r['K']}",
                                 value=r["recovery"], ci_low=r["chance_median"],
                                 ci_high=r["chance95"], p=np.nan, n=np.nan,
                                 note="mean matched correlation; ci_low/ci_high = chance "
                                      "median / 95th percentile"))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(args.out_dir, "positive_control_table.csv"), index=False)
    TR = pd.DataFrame(truth_rec)

    # ---- mouse vs each cohort, per-mouse measures
    cmp_rows = []
    for meas in sorted(P.measure.unique()):
        m = P[(P.cohort == "Mouse") & (P.measure == meas)].value.dropna()
        for c in cohorts[1:]:
            x = P[(P.cohort == c) & (P.measure == meas)].value.dropna()
            if len(m) >= 3 and len(x) >= 3:
                cmp_rows.append(dict(measure=meas, cohort=c, median_mouse=m.median(),
                                     median_cohort=x.median(),
                                     difference=m.median() - x.median(),
                                     p_mann_whitney=float(stats.mannwhitneyu(m, x).pvalue),
                                     n_mouse=len(m), n_cohort=len(x)))
    CMP = pd.DataFrame(cmp_rows)
    CMP.to_csv(os.path.join(args.out_dir, "positive_control_vs_mouse.csv"), index=False)

    # ---- the schematic's table, filled in
    def get(c, meas):
        r = R[(R.cohort == c) & (R.measure == meas)]
        return None if r.empty else r.iloc[0]

    sch = {}
    for c in cohorts:
        col = {}
        r = get(c, "fc_similarity")
        col["Static FC (similarity to group FC, r)"] = f"{r.value:.2f}" if r is not None else ""
        r = get(c, "kurt_rss")
        col["Co-fluctuation peaks (RSS kurtosis)"] = f"{r.value:.1f}" if r is not None else ""
        r = get(c, "calibrated d, dcc_a")
        col["Excess over Gaussian replica (DCC)"] = (
            f"{'yes' if r.p < 0.05 else 'no'}, d = {r.value:.2f} (p {fmt_p(r.p)})"
            if r is not None else "")
        r = get(c, "calibrated d, garch_alpha")
        col["Amplitude excess (GARCH α)"] = (f"d = {r.value:.2f} (p {fmt_p(r.p)})"
                                             if r is not None else "")
        r = get(c, "calibrated d, cpca_roi_c1_env_cv")
        col["Propagation-envelope excess (cPCA C1)"] = (f"d = {r.value:.2f} (p {fmt_p(r.p)})"
                                                        if r is not None else "")
        r = get(c, "half-life ratio obs/surr (dynamic mice)")
        col["Coupling half-life vs surrogates"] = (
            f"{r.value:.2f}x ({r.ci_low:.1f} vs {r.ci_high:.1f} s, n = {int(r.n)})"
            if r is not None and r.n > 0 else ("no dynamic mice" if r is not None else ""))
        r = get(c, "HMM cells with state structure")
        if r is not None:
            col["Reproducible HMM states"] = (f"{'yes' if r.value > 0 else 'no'} "
                                              f"({int(r.value)}/{int(r.n)} cells; best Holm p "
                                              f"{fmt_p(r.p)})")
        else:
            col["Reproducible HMM states"] = ""
        if not TR.empty and c in set(TR.cohort):
            t = TR[(TR.cohort == c)].set_index("K")
            col["HMM recovers the true states"] = "; ".join(
                f"K={k}: {t.loc[k, 'recovery']:.2f} (chance {t.loc[k, 'chance95']:.2f})"
                for k in t.index)
        r = get(c, "burst_z")
        col["Shared bursts (burst_z)"] = f"{r.value:.2f}" if r is not None else ""
        r = get(c, "fc_change_z")
        col["FC change between halves (z)"] = f"{r.value:.2f}" if r is not None else ""
        sch[LABELS[c]] = col
    S = pd.DataFrame(sch).fillna("")
    S.index.name = "row"
    S.to_csv(os.path.join(args.out_dir, "positive_control_schematic.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + S.to_string())

    figure(P, R, hmm, TR, cohorts, args.out_dir)
    matching_figure(args, rng)
    print(f"\nOutputs in {args.out_dir}/")


# ---------------- figures ----------------
def box_panel(ax, P, meas, cohorts, ylab, title, ref=None, log=False):
    rng = np.random.default_rng(3)
    shown = []
    for i, c in enumerate(cohorts):
        v = P[(P.cohort == c) & (P.measure == meas)].value.dropna().to_numpy(float)
        if log:
            v = v[v > 0]
        if not v.size:
            continue
        shown.append(i)
        col = COLORS[c]
        ax.boxplot(v, positions=[i], widths=0.56, showfliers=False, patch_artist=True,
                   medianprops=dict(color=INK, lw=1.1),
                   boxprops=dict(facecolor="white", edgecolor=col, lw=1.1),
                   whiskerprops=dict(color=col, lw=1.0), capprops=dict(color=col, lw=1.0))
        ax.scatter(i + rng.uniform(-0.14, 0.14, v.size), v, s=4.5 ** 2 / 2, color=col,
                   alpha=0.6, lw=0, zorder=3)
    if ref is not None:
        ax.axhline(ref, color=BASELINE, lw=0.8, zorder=0)
    if log:
        ax.set_yscale("log")
        fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.3g}")
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(fmt)
        ax.tick_params(axis="y", which="minor", labelsize=6)
    mv = P[(P.cohort == "Mouse") & (P.measure == meas)].value.median()
    if np.isfinite(mv):
        ax.axhline(mv, color=INK, lw=0.7, ls=":", zorder=0)
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels([LABELS[c] for c in cohorts], fontsize=7, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(cohorts) - 0.4)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, fontsize=8, loc="left")
    if not shown:
        ax.text(0.5, 0.5, "not available yet", transform=ax.transAxes, ha="center",
                fontsize=7, color=MUTED)
    sns.despine(ax=ax)


def figure(P, R, hmm, TR, cohorts, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 5.4),
                             gridspec_kw=dict(wspace=0.78, hspace=0.95))
    box_panel(axes[0, 0], P, "d_dcc_a", cohorts, "z obs − z replica", "Coupling (DCC)", ref=0)
    box_panel(axes[0, 1], P, "d_garch_alpha", cohorts, "z obs − z replica",
              "Amplitude (GARCH)", ref=0)
    box_panel(axes[0, 2], P, "d_cpca_roi_c1_env_cv", cohorts, "z obs − z replica",
              "Envelope (cPCA)", ref=0)
    box_panel(axes[0, 3], P, "half_life_ratio", cohorts, "Half-life ratio (obs / surr)",
              "Half-life", ref=1, log=True)
    box_panel(axes[1, 0], P, "burst_z", cohorts, "z vs phase surrogates", "Shared bursts", ref=0)
    box_panel(axes[1, 1], P, "fc_change_z", cohorts, "z vs phase surrogates",
              "FC change", ref=0)

    # HMM: calibrated median (95% CI) per cell and cohort
    ax = axes[1, 2]
    cells = []
    for c in cohorts:
        if hmm.get(c) is not None:
            cells += [(m, int(k)) for m, k in zip(hmm[c].model, hmm[c].K)]
    cells = sorted(set(cells), key=lambda x: (int("".join(ch for ch in x[0] if ch.isdigit()) or 0),
                                              x[1]))
    have = [c for c in cohorts if hmm.get(c) is not None]
    if cells and have:
        w = 0.72 / max(len(have), 1)
        for j, c in enumerate(have):
            T = hmm[c].set_index(["model", "K"])
            for i, cell in enumerate(cells):
                if cell not in T.index:
                    continue
                r = T.loc[cell]
                x = i - 0.36 + w * (j + 0.5)
                ax.plot([x, x], [r.ci_low, r.ci_high], color=COLORS[c], lw=1.0)
                ax.plot(x, r.calibrated_median, "o", ms=3.8, mec=COLORS[c],
                        mfc=COLORS[c] if bool(r.state_structure) else "white", mew=1.0)
        ax.axhline(0, color=BASELINE, lw=0.8, zorder=0)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"r{''.join(ch for ch in m if ch.isdigit())}\nK{k}" for m, k in cells],
                           fontsize=5.8)
    else:
        ax.text(0.5, 0.5, "not available yet", transform=ax.transAxes, ha="center", fontsize=7,
                color=MUTED)
    ax.set_ylabel("Calibrated similarity gain", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("PCA-HMM decision", fontsize=8, loc="left")
    sns.despine(ax=ax)

    # state recovery
    ax = axes[1, 3]
    if not TR.empty:
        for j, c in enumerate(["C", "C_strong"]):
            t = TR[TR.cohort == c].sort_values("K")
            if t.empty:
                continue
            x = t.K.to_numpy() + (j - 0.5) * 0.22
            ax.bar(x, t.recovery, width=0.2, color=COLORS[c], label=LABELS[c])
            for xi, ch in zip(x, t.chance95):
                ax.plot([xi - 0.1, xi + 0.1], [ch, ch], color=INK, lw=1.0)
        ax.set_xticks(sorted(TR.K.unique()))
        ax.set_xlabel("K (lowrank 15)", fontsize=8.5)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.legend(frameon=False, fontsize=6, loc="upper left", ncol=2, handlelength=1,
                  columnspacing=0.8)
    else:
        ax.text(0.5, 0.5, "not available yet", transform=ax.transAxes, ha="center", fontsize=7,
                color=MUTED)
    ax.set_ylabel("State recovery (r)", fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.set_title("HMM vs true states", fontsize=8, loc="left")
    sns.despine(ax=ax)

    for ax, letter in zip(axes.ravel(), "ABCDEFGH"):
        ax.text(-0.5, 1.12, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.text(0.02, -0.06, "Dotted line: mouse median. A stationary Gaussian, B sparse shared "
             "events, C state switching (B and C matched to the mouse DCC excess); C strong: the "
             "same mice as C, larger state separation.\nA-C: calibrated excess over stationary "
             "replicas. D: mice with DCC a z > 1.645. G: calibrated median (95% CI) per model "
             "and K; filled = state structure by the pre-registered rule.\nH: bars = "
             "best-matched correlation with the simulated states; black lines = chance "
             "(95th percentile).", fontsize=6.3, color=SECONDARY, linespacing=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"FigS_positive_controls.{ext}"), bbox_inches="tight",
                    dpi=450 if ext == "png" else None)
    plt.close(fig)


def matching_figure(args, rng):
    gm = os.path.join(args.sim_dir, "sim_manifest_grid.json")
    if not os.path.isfile(gm):
        return
    grid = json.load(open(gm)).get("grid") or {}
    fm = os.path.join(args.sim_dir, "sim_manifest_final.json")
    matched = (json.load(open(fm)).get("matching") or {}) if os.path.isfile(fm) else {}
    mouse = read_dfc(args.mouse_dfc)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5), gridspec_kw=dict(wspace=0.5))
    for ax, letter, xlabel, log in [(axes[0], "B", "Events per network per minute", True),
                                    (axes[1], "C", "State separation", False)]:
        pts = []
        for name, par in (grid.get(letter) or {}).items():
            r = read_dfc(args.dfc_prefix + name)
            if r is not None:
                pts.append((par, *median_ci(r["d"]["dcc_a"], rng)))
        if pts:
            pts.sort()
            x = np.array([p[0] for p in pts])
            ax.plot(x, [p[1] for p in pts], "-", color=COLORS[letter], lw=1.2)
            for p in pts:
                ax.plot([p[0]] * 2, [p[2], p[3]], color=COLORS[letter], lw=1.0)
                ax.plot(p[0], p[1], "o", ms=4, color=COLORS[letter])
        if mouse is not None:
            med, lo, hi = median_ci(mouse["d"]["dcc_a"], rng)
            ax.axhspan(lo, hi, color=GRID, lw=0, zorder=0)
            ax.axhline(med, color=INK, lw=0.8, ls=":")
        if letter in matched and matched[letter].get("parameter") is not None:
            ax.axvline(matched[letter]["parameter"], color=COLORS[letter], lw=0.8, ls="--")
        ax.axhline(0, color=BASELINE, lw=0.7, zorder=0)
        if log:
            ax.set_xscale("log")
            xs = sorted(float(v) for v in (grid.get(letter) or {}).values())
            if xs:
                ax.set_xlim(xs[0] / 1.6, xs[-1] * 1.6)
                ax.set_xticks(xs)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel("DCC excess d (median)", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{letter}  {'Events' if letter == 'B' else 'Switching'}", fontsize=8,
                     loc="left")
        sns.despine(ax=ax)
    fig.text(0.02, -0.1, "Dotted line and band: mouse median d and 95% CI; dashed: matched "
             "strength used for the final cohort.", fontsize=6.3, color=SECONDARY)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"FigS_positive_control_matching.{ext}"),
                    bbox_inches="tight", dpi=450 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
