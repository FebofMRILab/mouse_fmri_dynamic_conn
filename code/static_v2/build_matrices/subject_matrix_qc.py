"""
Subject-level sanity check of merged.csv before any group analysis.

For every subject's FC matrix it reports:
  * non-finite values, and ROIs whose row is all zero or constant
    (a dead or unmasked ROI in the time series)
  * mean FC and FC spread
  * similarity to the leave-one-out group mean (Spearman over edges):
    a subject whose connectivity pattern looks unlike everyone else's
    usually has a registration, masking or signal problem

Outliers are flagged with a robust z-score (median / MAD) so that one bad
subject cannot hide itself by inflating the SD. Set SPOTLIGHT to a subject
index to print that subject's values next to the cohort's.

Also enforces that the number of blocks in merged.csv equals the number of
rows in the mapping file, the check that caught the 94 / 95 discrepancy.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = os.environ.get("DATA_DIR", ".")
MERGED_CSV = os.environ.get("MERGED_CSV", "merged.csv")
MAPPING_CSV = os.environ.get("MAPPING_CSV", "subject_group_order_mappings.csv")
N_ROI = int(os.environ.get("N_ROI", 82))
SPOTLIGHT = os.environ.get("SPOTLIGHT")            # e.g. "73"
ROBUST_Z_FLAG = 3.5


def read_merged(path, n_roi):
    """
    Read merged.csv without depending on how the top-left cell is written
    (empty, 'NaN', or 0). Reads with no header, drops the row-label column by
    position, and removes every block's header row by content.
    """
    values = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if values.shape[1] == n_roi + 1:
        values = values[:, 1:]                      # row-label column
    elif values.shape[1] != n_roi:
        raise ValueError(f"merged.csv has {values.shape[1]} columns; expected {n_roi} or {n_roi + 1}")
    header = np.isclose(values, np.arange(1, n_roi + 1)[None, :], atol=1e-8).all(1)
    body = values[~header]
    if body.shape[0] % n_roi:
        raise ValueError(f"{body.shape[0]} matrix rows is not a multiple of {n_roi}")
    return body


def robust_z(x):
    x = np.asarray(x, float)
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median)) * 1.4826
    return (x - median) / mad if mad > 0 else np.zeros_like(x)


def main():
    body = read_merged(os.path.join(DATA_DIR, MERGED_CSV), N_ROI)
    matrices = body.reshape(-1, N_ROI, N_ROI)

    meta = pd.read_csv(os.path.join(DATA_DIR, MAPPING_CSV))
    if matrices.shape[0] != len(meta):
        raise ValueError(f"merged.csv has {matrices.shape[0]} matrices but the mapping file "
                         f"has {len(meta)} rows. Resolve before any analysis.")
    print(f"{matrices.shape[0]} matrices, {len(meta)} mapping rows: counts agree")

    matrices = matrices[meta["index"].to_numpy() - 1]
    iu = np.triu_indices(N_ROI, 1)
    edges = matrices[:, iu[0], iu[1]]

    rows = []
    for s, matrix in enumerate(matrices):
        off = matrix.copy()
        np.fill_diagonal(off, np.nan)
        row_sd = np.nanstd(off, axis=1)
        rows.append({
            "index": int(meta["index"].iloc[s]),
            "group": meta.iloc[s].get("genotype", meta.iloc[s].get("group", "")),
            "n_nonfinite": int((~np.isfinite(edges[s])).sum()),
            "n_dead_rois": int(((np.nanmax(np.abs(off), axis=1) < 1e-8) | (row_sd < 1e-8)).sum()),
            "mean_fc": float(np.nanmean(edges[s])),
            "sd_fc": float(np.nanstd(edges[s])),
        })

    finite = np.where(np.isfinite(edges), edges, np.nan)
    total = np.nansum(finite, axis=0)
    counts = np.sum(np.isfinite(finite), axis=0)
    for s, row in enumerate(rows):
        loo = (total - np.nan_to_num(finite[s])) / (counts - np.isfinite(finite[s]))
        ok = np.isfinite(finite[s]) & np.isfinite(loo)
        row["similarity_to_group"] = float(stats.spearmanr(finite[s][ok], loo[ok])[0])

    report = pd.DataFrame(rows)
    for column in ("mean_fc", "sd_fc", "similarity_to_group"):
        report[f"z_{column}"] = robust_z(report[column])

    report["flag"] = (
        (report["n_nonfinite"] > 0) | (report["n_dead_rois"] > 0)
        | (report[["z_mean_fc", "z_sd_fc", "z_similarity_to_group"]].abs()
           > ROBUST_Z_FLAG).any(axis=1)
    )
    report.to_csv(os.path.join(DATA_DIR, "subject_matrix_qc.csv"), index=False)

    print("\nCohort:")
    print(report[["mean_fc", "sd_fc", "similarity_to_group"]].describe().round(3).to_string())

    flagged = report[report["flag"]]
    print(f"\nFlagged subjects ({len(flagged)}):")
    print("  none" if flagged.empty else flagged.round(3).to_string(index=False))

    if SPOTLIGHT:
        row = report[report["index"] == int(SPOTLIGHT)]
        print(f"\nSubject {SPOTLIGHT}:")
        print(row.round(3).to_string(index=False) if len(row) else "  not found")

    print("\nWritten to subject_matrix_qc.csv")


if __name__ == "__main__":
    main()