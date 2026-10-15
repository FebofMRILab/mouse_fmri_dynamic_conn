#!/usr/bin/env python3
"""
schematic_event_mechanism.py
Three generative mechanisms with the SAME haemodynamic response, band-pass
(0.01-0.1 Hz) and static correlation between two regions (r ~ 0.45):
  A  stationary Gaussian drive          (what the stationary replica is)
  B  sparse shared neural events        (stationary, non-Gaussian)
  C  switching between coupled/uncoupled epochs (non-stationary)
Rows: neural drive -> BOLD (z) -> co-fluctuation z_A * z_B.
Bottom: positive-control results (95 simulated mice per mechanism, same
pipelines as the mice) next to the 95 mice; values in the TABLE block below.
"""
import textwrap

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import gamma, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "axes.labelsize": 12, "axes.linewidth": 1.2,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})

# ============================ TABLE: RESULTS (EDIT HERE) ============================
# Values are medians over 95 mice (real) or 95 simulated mice per mechanism, from:
#   positive_control_summary/positive_control_schematic.csv and
#   positive_control_vs_mouse.csv                 (summarize_positive_controls.py)
#   hmm_confirm_*/analysis/confirm_cells.csv      (analyze_confirm_lowrank.py; mice
#                                                  and simulated A, B, C)
#   dfc_dcc_cpca/figures/table_calibrated_test.csv (mouse DCC excess and its CI)
# Each cell is (text, same_as_mice): same_as_mice = Mann-Whitney p > 0.05 against
# the mice in positive_control_vs_mouse.csv; those cells are printed in bold.
TABLE_COLUMNS = ["A  Gaussian", "B  Events", "C  Switching", "Mice (n = 95)"]
TABLE_ROWS = [
    ("Coupling excess over replica (DCC, d)",
     [("0.24", False), ("1.25", True), ("1.16", True), ("1.33", None)]),
    ("Shared co-fluctuation bursts (burst z)",
     [("−0.27", False), ("2.23", False), ("0.06", True), ("0.20", None)]),
    ("Co-fluctuation peaks (RSS kurtosis)",
     [("0.6", False), ("2.9", False), ("0.9", False), ("1.2", None)]),
    ("Amplitude excess (GARCH α, d)",
     [("0.32", False), ("1.93", False), ("0.89", True), ("0.58", None)]),
    ("HMM split-half replication: cells of 6",
     [("0  (≤ 0.01)", None), ("6  (0.10–0.40)", None), ("5  (0.03–0.43)", None),
      ("2  (0.01–0.07)", None)]),
    ("FC change between scan halves (z)",
     [("0.24", False), ("0.59", False), ("0.12", False), ("1.75", None)]),
]
VERDICT_ROW = ("Fits the mouse data", ["no", "no", "partly", ""])
TABLE_CAPTION = ("Positive control: 95 simulated mice per mechanism, each built from one "
                 "mouse's FC and scan, run through the same pipelines as the 95 mice.")
FOOTNOTE = [   # one entry per printed line
    "d: median z(observed) − z(own stationary replicas). B and C were set to the mice's DCC excess; "
    "A's 0.24 (p = 0.02) is the residual bias.",
    "HMM: pre-specified test (HMM-PCA rank 15 and 25 × K = 2–4, 30 split-halves, calibrated against "
    "replicas, Holm): cells meeting the rule",
    "(range of calibrated medians). Bold: not distinguishable from the mice (Mann–Whitney p > 0.05).",
    "The mice match none of the three; what sets them apart is FC change between scan halves.",
    "Not discriminating: coupling-change half-life vs surrogates, 1.00–1.02× in every cohort "
    "(mice 12.7 vs 12.6 s).",
    "Stronger switching (C, separation 0.9): HMM 5 of 6 cells (0.51–0.79); true states recovered "
    "0.77 at K = 3 (C: 0.49; chance 0.03).",
    "Dotted lines in B: shared events shifted +5 s to the haemodynamic peak. Traces are "
    "illustrative simulated windows.",
]
# ===================================================================================

INK, SECONDARY, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL_A, COL_B = "#2a78d6", "#eb6834"
dt, TR = 0.5, 2.0
t_h = np.arange(0, 32, dt)
hrf = gamma.pdf(t_h, 6) - gamma.pdf(t_h, 16) / 6
hrf /= hrf.sum()
b, a = butter(3, [0.01 * 2 * dt, 0.1 * 2 * dt], btype="band")


def bold(drive):
    y = filtfilt(b, a, np.convolve(drive, hrf)[:drive.size])
    return (y - y.mean()) / y.std()


def gaussian(n, rng, c=0.45):
    s = rng.standard_normal(n)
    return [np.sqrt(c) * s + np.sqrt(1 - c) * rng.standard_normal(n) for _ in range(2)], None


def events(n, rng, rate_shared=1 / 60, rate_private=1 / 60, noise=0.05):
    def train(rate):
        x = np.zeros(n)
        k = rng.random(n) < rate * dt
        x[k] = rng.lognormal(0, 0.35, k.sum())
        return x
    sh = train(rate_shared)
    priv = [train(rate_private), train(rate_private)]
    return [sh + p + noise * rng.standard_normal(n) for p in priv], (sh, priv)


def switching(n, rng, c_on=0.9, epoch=(40, 90)):
    on = np.zeros(n, bool)
    i, state = 0, bool(rng.random() < 0.5)
    while i < n:
        L = int(rng.uniform(*epoch) / dt)
        on[i:i + L] = state
        state = not state
        i += L
    s = rng.standard_normal(n)
    out = []
    for _ in range(2):
        e = rng.standard_normal(n)
        out.append(np.where(on, np.sqrt(c_on) * s + np.sqrt(1 - c_on) * e, e))
    return out, on


MODELS = [("A", "Stationary Gaussian", "what the stationary replica is", gaussian),
          ("B", "Sparse shared events", "stationary, non-Gaussian", events),
          ("C", "State switching", "non-stationary", switching)]

# long simulations: static r and co-fluctuation kurtosis at TR sampling
rng = np.random.default_rng(1)
stats_long = {}
for key, _, _, f in MODELS:
    rs, ks = [], []
    for _ in range(5):
        (dA, dB), _ = f(int(20000 / dt), rng)
        zA, zB = bold(dA)[::int(TR / dt)], bold(dB)[::int(TR / dt)]
        rs.append(np.corrcoef(zA, zB)[0, 1])
        ks.append(kurtosis(zA * zB))
    stats_long[key] = (np.median(rs), np.median(ks))

# display window
T_total, t0, t1 = 700, 100, 400
n = int(T_total / dt)
t = np.arange(n) * dt
win = (t >= t0) & (t < t1)
tt = t[win] - t0
seeds = {"A": 11, "B": 21, "C": 70}   # illustrative windows
from matplotlib.lines import Line2D


def shade(ax, on):
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    for s0, s1 in zip(edges[::2], edges[1::2]):
        ax.axvspan(tt[s0], tt[min(s1, len(tt) - 1)], color=GRID, lw=0, zorder=0)


fig = plt.figure(figsize=(8.0, 8.0))
gs = fig.add_gridspec(4, 3, height_ratios=[0.75, 1.0, 1.0, 1.55], hspace=0.42, wspace=0.16)
for j, (key, title, sub, f) in enumerate(MODELS):
    r = np.random.default_rng(seeds[key])
    (dA, dB), extra = f(n, r)
    zA, zB = bold(dA), bold(dB)
    e = zA[win] * zB[win]
    # --- row 0: neural drive
    ax = fig.add_subplot(gs[0, j])
    if key == "B":
        sh, (pA, pB) = extra
        for x, col in [(pA, COL_A), (pB, COL_B), (sh, INK)]:
            idx = np.where(x[win] > 0)[0]
            ax.vlines(tt[idx], 0, x[win][idx], color=col, lw=1.4 if col == INK else 1.0)
        ax.set_ylim(0, 3.2)
        ax.legend(handles=[Line2D([], [], color=INK, lw=1.4, label="shared"),
                           Line2D([], [], color=COL_A, lw=1.0, label="A only"),
                           Line2D([], [], color=COL_B, lw=1.0, label="B only")],
                  frameon=False, fontsize=6, ncol=3, loc="upper right", handlelength=0.9,
                  columnspacing=0.8, borderaxespad=0.0)
    else:
        ax.plot(tt, dA[win], color=COL_A, lw=0.35, alpha=0.8)
        ax.plot(tt, dB[win], color=COL_B, lw=0.35, alpha=0.6)
        ax.set_ylim(-4.5, 6.5)
        if key == "C":
            shade(ax, extra[win])
            ax.text(0.99, 0.98, "grey = coupled epochs", fontsize=6, color=SECONDARY,
                    ha="right", va="top", transform=ax.transAxes)
    ax.set_xlim(0, t1 - t0)
    ax.set_yticks([])
    ax.set_xticks([])
    sns.despine(ax=ax, left=True)
    ax.set_title(f"{key}   {title}", fontsize=8.5, loc="left", pad=13)
    ax.text(0.0, 1.03, sub, fontsize=6.8, color=SECONDARY, transform=ax.transAxes,
            va="bottom", ha="left")
    if j == 0:
        ax.set_ylabel("Neural\ndrive", fontsize=8)
    # --- row 1: BOLD
    ax = fig.add_subplot(gs[1, j])
    if key == "C":
        shade(ax, extra[win])
    ax.plot(tt, zA[win], color=COL_A, lw=1.1, label="Region A")
    ax.plot(tt, zB[win], color=COL_B, lw=1.1, label="Region B")
    ax.axhline(0, color=BASELINE, lw=0.6, zorder=0)
    ax.set_xlim(0, t1 - t0)
    ax.set_ylim(-3.6, 5.6)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=6.5)
    if j == 0:
        ax.set_ylabel("BOLD (z)", fontsize=8)
        ax.legend(frameon=False, fontsize=6.3, loc="upper left", ncol=2, handlelength=1.2)
    else:
        ax.set_yticklabels([])
    sns.despine(ax=ax)
    # --- row 2: co-fluctuation
    ax = fig.add_subplot(gs[2, j])
    if key == "C":
        shade(ax, extra[win])
    if key == "B":
        for x in tt[np.where(extra[0][win] > 0)[0]]:
            ax.axvline(x + 5, color=MUTED, lw=0.6, ls=":", zorder=0)
    ax.fill_between(tt, 0, e, where=e > 0, color=MUTED, alpha=0.4, lw=0)
    ax.plot(tt, e, color=INK, lw=0.9)
    ax.axhline(0, color=BASELINE, lw=0.6, zorder=0)
    ax.set_xlim(0, t1 - t0)
    ax.set_ylim(-3.5, 15)
    ax.set_xticks([0, 100, 200, 300])
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("Time (s)", fontsize=8)
    if j == 0:
        ax.set_ylabel("Co-fluctuation\nz$_A$ × z$_B$", fontsize=8)
    else:
        ax.set_yticklabels([])
    sns.despine(ax=ax)

# ---- bottom: positive-control results next to the mice
axt = fig.add_subplot(gs[3, :])
axt.axis("off")
x_label, x_cols = 0.0, [0.47, 0.62, 0.77, 0.92]
y0, dy = 0.855, 0.087
r_lo = min(v[0] for v in stats_long.values())
r_hi = max(v[0] for v in stats_long.values())
axt.text(x_label, y0 + 0.13,
         textwrap.fill(f"Illustration above: the three share the HRF, 0.01–0.1 Hz band-pass "
                       f"and static FC (r = {r_lo:.2f}–{r_hi:.2f}). " + TABLE_CAPTION, 125),
         fontsize=6.6, color=SECONDARY, transform=axt.transAxes, va="center", linespacing=1.4)
for x, k in zip(x_cols, TABLE_COLUMNS):
    axt.text(x, y0, k, fontsize=7.5, fontweight="bold", ha="center", va="center",
             transform=axt.transAxes)
axt.plot([0, 0.99], [y0 - 0.045] * 2, color=BASELINE, lw=0.6, transform=axt.transAxes)
rows = TABLE_ROWS + [VERDICT_ROW]
for i, (lab, vals) in enumerate(rows):
    y = y0 - (i + 1) * dy
    last = i == len(rows) - 1
    if last:
        axt.plot([0, 0.99], [y + dy / 2] * 2, color=BASELINE, lw=0.6, transform=axt.transAxes)
    axt.text(x_label, y, lab, fontsize=7, color=INK, va="center", transform=axt.transAxes,
             fontweight="bold" if last else "normal")
    for j, (x, v) in enumerate(zip(x_cols, vals)):
        text, same = (v, None) if last else v
        mice = j == len(x_cols) - 1
        axt.text(x, y, text, fontsize=7, va="center", ha="center", transform=axt.transAxes,
                 color=INK if (mice or same or (last and text != "no")) else SECONDARY,
                 fontweight="bold" if (same or (last and text == "partly")) else "normal")
yf = y0 - (len(rows) + 0.75) * dy
axt.text(x_label, yf, "\n".join(FOOTNOTE),
         fontsize=6.0, color=SECONDARY, va="top", transform=axt.transAxes, linespacing=1.5)
for ext in ("png", "pdf"):
    fig.savefig(f"Fig_schematic_event_mechanism.{ext}", bbox_inches="tight",
                dpi=450 if ext == "png" else None)
print({k: (round(v[0], 2), round(v[1], 1)) for k, v in stats_long.items()})
