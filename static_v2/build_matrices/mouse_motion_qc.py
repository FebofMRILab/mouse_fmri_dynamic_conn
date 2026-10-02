"""
Head-motion QC for the mouse cohort.

1. Collects per-subject motion from the RABIES FD files and writes a copy of
   the mapping file with the motion columns appended.
2. Tests whether motion differs between groups (Welch ANOVA with permutation,
   plus the 8 planned contrasts) -- if it does, any group difference in
   connectivity or dynamics needs that caveat.
3. Correlates motion with the HMM dynamic metrics, which are the measures most
   vulnerable to motion.
4. QC-FC: correlates each edge's connectivity with motion across subjects, the
   standard check that denoising removed the motion signal.

RABIES FD files have one row per volume with MeanFD and MaxFD columns
(displacement averaged / maximised over brain voxels).
"""

import os
import glob
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ../welch_tests_scipy.py import welch_anova_permutation, welch_t_test

# ---------------- configuration ----------------
FD_ROOT = os.environ.get(
    "FD_ROOT", "../preprocess_outputs/motion_datasink/FD_csv")
MAPPING_CSV = os.environ.get("MAPPING_CSV", "subject_group_order_mappings.csv")
MERGED_CSV = os.environ.get("MERGED_CSV", "merged.csv")          # set to "" to skip QC-FC
HMM_METRICS_CSV = os.environ.get("HMM_METRICS_CSV", "")          # optional, long format
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "subject_group_order_mappings_with_motion.csv")
N_ROI = int(os.environ.get("N_ROI", 82))
FD_THRESHOLD = float(os.environ.get("FD_THRESHOLD", 0.05))       # mm, rodent scale
N_PERMUTATIONS = int(os.environ.get("N_PERM", 10000))
RANDOM_SEED = 20260920

# The analysed time series drop 30 s at each end of the 600-volume scan
# (TR = 2 s -> 15 volumes each end, leaving 570). Motion must be summarised
# over the same 570 volumes that went into the FC and HMM analyses.
TR_SEC = float(os.environ.get("TR_SEC", 2.0))
TRIM_SEC = float(os.environ.get("TRIM_SEC", 30.0))
EXPECTED_ANALYSED_VOLUMES = int(os.environ.get("EXPECTED_VOLUMES", 570))

GROUP_ORDER = ["NTG_Y", "NTG_M", "NTG_A", "CRND8_Y", "CRND8_M", "5XFAD_Y", "5XFAD_M"]
PAIRWISE_COMPARISONS = [
    ("NTG_Y", "NTG_M"), ("NTG_Y", "NTG_A"), ("NTG_Y", "CRND8_Y"),
    ("NTG_M", "CRND8_M"), ("NTG_Y", "5XFAD_Y"), ("NTG_M", "5XFAD_M"),
    ("CRND8_Y", "CRND8_M"), ("5XFAD_Y", "5XFAD_M"),
]


def find_fd_file(subject_number):
    """RABIES nests the FD file under a directory naming the subject id."""
    subject = f"{int(subject_number):03d}"
    patterns = [
        os.path.join(FD_ROOT, f"*subject_id{subject}.*", "*", f"*sub-{subject}*FD_file.csv"),
        os.path.join(FD_ROOT, f"*subject_id{subject}.*", "**", "*FD_file.csv"),
        os.path.join(FD_ROOT, "**", f"sub-{subject}_*FD_file.csv"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return None


def trim(values):
    """Drop the volumes removed from the analysed time series."""
    n_trim = int(round(TRIM_SEC / TR_SEC))
    kept = values[n_trim:len(values) - n_trim] if n_trim else values
    if kept.size != EXPECTED_ANALYSED_VOLUMES:
        raise ValueError(f"{values.size} volumes minus {n_trim} at each end = {kept.size}, "
                         f"expected {EXPECTED_ANALYSED_VOLUMES}")
    return kept


def summarise(path):
    table = pd.read_csv(path)
    if "MeanFD" not in table.columns:
        raise ValueError(f"No MeanFD column in {path}")
    fd = trim(pd.to_numeric(table["MeanFD"], errors="coerce").to_numpy())
    fd = fd[np.isfinite(fd)]
    peak = trim(pd.to_numeric(table.get("MaxFD", pd.Series(np.nan, index=table.index)),
                              errors="coerce").to_numpy())
    peak = peak[np.isfinite(peak)]
    return {"n_volumes": fd.size,
            "mean_FD": fd.mean(), "median_FD": np.median(fd), "max_FD": fd.max(),
            "max_MaxFD": peak.max() if peak.size else np.nan,
            f"prop_FD_over_{FD_THRESHOLD}": (fd > FD_THRESHOLD).mean()}


def main():
    meta = pd.read_csv(MAPPING_CSV)
    if "index" not in meta.columns:
        raise ValueError("Mapping file needs an `index` column (BIDS subject number).")

    rows, missing = [], []
    for subject_number in meta["index"]:
        path = find_fd_file(subject_number)
        if path is None:
            missing.append(int(subject_number))
            rows.append({})
        else:
            rows.append(summarise(path))

    motion = pd.DataFrame(rows, index=meta.index)
    merged_meta = pd.concat([meta, motion], axis=1)
    merged_meta.to_csv(OUTPUT_CSV, index=False)

    found = merged_meta["mean_FD"].notna()
    print(f"Motion summarised for {int(found.sum())}/{len(merged_meta)} mice")
    if missing:
        print(f"  no FD file for subjects: {missing}")
    print(merged_meta.loc[found, ["mean_FD", "median_FD", "max_FD",
                                  f"prop_FD_over_{FD_THRESHOLD}", "n_volumes"]]
          .describe().round(4).to_string())

    group_column = "genotype" if "genotype" in merged_meta.columns else "group"
    analysed = merged_meta.loc[found].copy()

    print("\nMotion by group:")
    print(analysed.groupby(group_column)["mean_FD"]
          .agg(["count", "mean", "std", "max"]).reindex(GROUP_ORDER).round(4).to_string())

    samples = [analysed.loc[analysed[group_column] == g, "mean_FD"].to_numpy()
               for g in GROUP_ORDER]
    samples = [x[np.isfinite(x)] for x in samples]
    if all(len(x) >= 2 and np.var(x, ddof=1) > 0 for x in samples):
        F, p = welch_anova_permutation(samples, N_PERMUTATIONS, RANDOM_SEED)
        print(f"\nmean_FD across groups: Welch F = {F:.3f}, permutation p = {p:.4f}")

    print("\nPlanned contrasts on mean_FD:")
    for group_a, group_b in PAIRWISE_COMPARISONS:
        a = analysed.loc[analysed[group_column] == group_a, "mean_FD"].to_numpy()
        b = analysed.loc[analysed[group_column] == group_b, "mean_FD"].to_numpy()
        t, p = welch_t_test(a, b)
        flag = "  <-- differs" if np.isfinite(p) and p < 0.05 else ""
        print(f"  {group_a:9s} vs {group_b:9s} t = {t:6.2f}  p = {p:.4f}{flag}")

    # ---------- motion vs HMM metrics ----------
    if HMM_METRICS_CSV and os.path.exists(HMM_METRICS_CSV):
        metrics = pd.read_csv(HMM_METRICS_CSV)
        metrics["index"] = pd.Series(metrics["subject_id"]).astype(str) \
            .str.extract(r"(\d+)")[0].astype(float)
        joined = metrics.merge(analysed[["index", "mean_FD"]], on="index", how="inner")
        print("\nHMM metrics vs mean_FD (Spearman):")
        for column in [c for c in ("switching_rate", "max_FO", "fractional_occupancy",
                                   "mean_dwell_seconds") if c in joined.columns]:
            sub = joined[["mean_FD", column]].dropna()
            rho, p = stats.spearmanr(sub["mean_FD"], sub[column])
            print(f"  {column:22s} rho = {rho:6.3f}  p = {p:.4f}")

    # ---------- QC-FC ----------
    if MERGED_CSV and os.path.exists(MERGED_CSV):
        table = pd.read_csv(MERGED_CSV)
        table = table.loc[:, ~table.columns.astype(str).str.startswith("Unnamed")]
        values = table.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").to_numpy(float)
        header = np.isclose(values, np.arange(1, N_ROI + 1)[None, :], atol=1e-8).all(1)
        matrices = values[~header].reshape(-1, N_ROI, N_ROI)
        matrices = matrices[merged_meta["index"].to_numpy() - 1]

        iu = np.triu_indices(N_ROI, 1)
        Z = np.arctanh(np.clip(matrices, -0.999999, 0.999999))[:, iu[0], iu[1]]
        fd = merged_meta["mean_FD"].to_numpy(float)
        keep = np.isfinite(fd)

        qc_fc = np.array([stats.spearmanr(fd[keep], Z[keep, e])[0] for e in range(Z.shape[1])])
        qc_p = np.array([stats.spearmanr(fd[keep], Z[keep, e])[1] for e in range(Z.shape[1])])
        print("\nQC-FC (edge connectivity vs mean_FD):")
        print(f"  median rho = {np.nanmedian(qc_fc):.3f}; "
              f"|rho| > 0.3 in {np.mean(np.abs(qc_fc) > 0.3) * 100:.1f}% of edges; "
              f"p < 0.05 in {np.mean(qc_p < 0.05) * 100:.1f}% (chance 5%)")
        pd.DataFrame({"edge": np.arange(Z.shape[1]), "qc_fc_rho": qc_fc, "qc_fc_p": qc_p}) \
            .to_csv("qc_fc_edgewise.csv", index=False)

    print(f"\nWritten to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()