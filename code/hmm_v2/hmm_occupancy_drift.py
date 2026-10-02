#!/usr/bin/env python3
"""
hmm_occupancy_drift.py
-------------------------------------------------------------------
Pre-registered drift test for the REPORTED HMM fit (the one run4e saved).

Same test as the confirmatory analysis (analyze_confirm_lowrank.py, drift
section): for each state, each mouse's change in fractional occupancy between
scan halves, (Q3 + Q4)/2 - (Q1 + Q2)/2, tested across mice with a two-sided
Wilcoxon signed-rank test (scipy.stats.wilcoxon), Holm-corrected across states
within K (statsmodels multipletests, method='holm'). A state whose occupancy
changes consistently over the scan is a drift signature rather than a
recurring state.

Input   <model_dir>/occupancy_by_quarter.csv   written by run4e_hmm_mar.m
Output  <model_dir>/occupancy_drift.csv        one row per state, with fit_id

Usage (from the folder you run the MATLAB steps in)
    python hmm_occupancy_drift.py      # /blue/febo/share/Abeta_Age/rabies/signals/hmmmar_simple_k4_hmmpca15
    python hmm_occupancy_drift.py --model_dir <folder>
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def occupancy_drift(occ, label):
    """Drift table for one model; the rows of analyze_confirm_lowrank.py."""
    drift = []
    for K, ok in occ.groupby("K"):
        rows = []
        for k, sk in ok.groupby("state"):
            w = sk.pivot_table(index="subject_id", columns="quarter", values="fo_quarter")
            d = (w[3] + w[4]) / 2 - (w[1] + w[2]) / 2
            p = float(stats.wilcoxon(d).pvalue) if (d != 0).sum() > 5 else np.nan
            rows.append(dict(model=label, K=int(K), state=int(k), n=len(d),
                             fo_total=sk.drop_duplicates("subject_id").fo_total.mean(),
                             median_change=d.median(), frac_increase=float((d > 0).mean()),
                             p_two_sided=p))
        rows = pd.DataFrame(rows)
        rows["p_holm_within_K"] = holm(rows.p_two_sided)
        drift.append(rows)
    return pd.concat(drift, ignore_index=True)


def holm(p):
    """Holm adjustment (statsmodels); NaN p-values are left out and stay NaN."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    # same defaults and folder names as hmm_config.m
    k = int(os.environ.get("HMM_K", "4"))
    r = int(os.environ.get("HMM_LOWRANK", "15"))
    c = int(os.environ.get("HMM_PCA", "0"))
    base = os.environ.get("HMM_OUT_BASE", "/blue/febo/share/Abeta_Age/rabies/signals")
    folder = f"hmmmar_simple_k{k}_hmmpca{r}" if r > 0 else f"hmmmar_simple_k{k}_pca{c}"
    default_dir = os.environ.get("HMM_MODEL_DIR") or os.path.join(base, folder)
    ap.add_argument("--model_dir", default=default_dir)
    args = ap.parse_args()

    occ = pd.read_csv(os.path.join(args.model_dir, "occupancy_by_quarter.csv"))
    ids = occ["fit_id"].unique()
    if len(ids) != 1:
        raise SystemExit(f"occupancy_by_quarter.csv mixes {len(ids)} fits.")
    fid_file = os.path.join(args.model_dir, "fit_id.txt")
    if os.path.isfile(fid_file) and open(fid_file).read().strip() != ids[0]:
        raise SystemExit("occupancy_by_quarter.csv does not come from the fit in fit_id.txt.")

    label = os.path.basename(os.path.normpath(args.model_dir))
    D = occupancy_drift(occ, label)
    D["fit_id"] = ids[0]
    out = os.path.join(args.model_dir, "occupancy_drift.csv")
    D.to_csv(out, index=False)
    with pd.option_context("display.width", 160):
        print(D.drop(columns="fit_id").round(4).to_string(index=False))
    print(f"\nWritten: {out}  (fit {ids[0][:12]})")


if __name__ == "__main__":
    main()
