#!/usr/bin/env python3
"""
select_screen_candidate.py -- items 1-5 of MODEL_SELECTION_RULE.md, applied mechanically

Reads step 5's table_hmm_splithalf_null.csv (plot_hmm_splithalf_null.py) and the
run's metadata_aligned.csv, and writes which model x K cell is the candidate and
how it is confirmed. No choice is made by hand.

  lead       metric = demeaned, null = phase, one-sided Mann-Whitney p < 0.05/n_K
             (n_K = number of K for that model: 9 for K 2-10)
  candidate  leading cell with the largest delta_median; ties at three decimals
             -> smaller K -> smaller dimension (PCA dim / HMM-PCA rank; all ROIs = n ROIs)
  confirm    HMM-PCA rank 15 or 25 at K 2-4: step 6 (already run)
             otherwise: new calibrated test, K 2-4 if the candidate's K is 2-4,
             else K-1, K, K+1 (MODEL_SELECTION_RULE.md item 4)

Status: NEW (selection helper; reads files only).

Usage (in signals/)
    python select_screen_candidate.py --table hmm_raw_timeseries_null/figures/table_hmm_splithalf_null.csv \
        --metadata hmm_raw_timeseries_null/metadata_aligned.csv --rule MODEL_SELECTION_RULE.md
    -> screen_candidate.txt next to the table
"""
import argparse
import hashlib
import os
import re
from datetime import datetime

import pandas as pd


def dimension(model, n_roi):
    m = re.search(r"(\d+)$", model)
    return int(m.group(1)) if m else n_roi            # gaussian_full -> all ROIs


def family(model):
    if model.startswith("lowrank"):
        return "HMM-PCA (options.lowrank)"
    if model == "gaussian_full":
        return "Gaussian, all ROIs (no reduction)"
    return "Gaussian on PCA-reduced data (options.pca)"


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--metadata", required=True, help="metadata_aligned.csv of the same step 5 run")
    ap.add_argument("--expected_n", type=int, default=95)
    ap.add_argument("--n_roi", type=int, default=82)
    ap.add_argument("--rule", default=None, help="MODEL_SELECTION_RULE.md (its SHA-256 is recorded)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    n = len(pd.read_csv(a.metadata))
    if n != a.expected_n:
        raise SystemExit(f"{a.metadata} lists {n} mice, not {a.expected_n}: the screen is not valid (E49).")
    t = pd.read_csv(a.table)
    d = t[(t.metric == "demeaned") & (t.null == "phase")].copy()
    if d.empty:
        raise SystemExit("no demeaned / phase rows in the table")
    if (d.n_null < 10).any() or (d.n_obs < 10).any():
        raise SystemExit("cells with fewer than 10 split-halves or null fits: not the pre-specified screen")
    nK = d.groupby("model").K.transform("nunique")
    d["alpha"] = 0.05 / nK
    d["lead"] = d.p_mw < d.alpha
    d["dimension"] = [dimension(m, a.n_roi) for m in d.model]
    leads = d[d.lead].copy()

    lines = [f"Screen candidate (MODEL_SELECTION_RULE.md items 1-5), {datetime.now().isoformat(timespec='seconds')}",
             f"table     {os.path.abspath(a.table)}  sha256 {sha256(a.table)}",
             f"metadata  {os.path.abspath(a.metadata)}  ({n} mice)"]
    if a.rule:
        lines.append(f"rule      {os.path.abspath(a.rule)}  sha256 {sha256(a.rule)}")
    lines.append(f"cells: {len(d)} (demeaned, phase null); leading (p < 0.05/n_K): {len(leads)}")
    if leads.empty:
        lines += ["", "NO LEAD. No new confirmation test (item 5); step 6 is reported under item 7."]
    else:
        leads["delta3"] = leads.delta_median.round(3)
        leads = leads.sort_values(["delta3", "K", "dimension"], ascending=[False, True, True])
        c = leads.iloc[0]
        K = int(c.K)
        covered = c.model in ("lowrank_15", "lowrank_25") and K in (2, 3, 4)
        cells = [2, 3, 4] if K in (2, 3, 4) else [K - 1, K, K + 1]
        lines += ["", "Leading cells (ranked):",
                  leads[["model", "K", "delta_median", "p_mw", "alpha"]].to_string(index=False), "",
                  f"CANDIDATE  {c.model}, K = {K}  ({family(c.model)}, dimension {int(c.dimension)}); "
                  f"delta {c.delta_median:.4f}, p {c.p_mw:.2g}"]
        if covered:
            lines.append("CONFIRMATION  step 6 (already run: HMM-PCA rank 15/25, K 2-4); no new test (item 4).")
        else:
            lines.append(f"CONFIRMATION  new calibrated test of {c.model} at K {cells} with step 6's design, "
                         "seed 20261026, replica seed 20261027 (item 4).")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.table)), "screen_candidate.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
