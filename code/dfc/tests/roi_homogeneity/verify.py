"""Checks of roi_signal_homogeneity.py outputs against known answers and independent code."""
import sys
import numpy as np, pandas as pd, nibabel as nib
out, res = sys.argv[1], sys.argv[2]
mr = pd.read_csv(f"{res}/roi_homogeneity_mouse_roi.csv", dtype={"subject_id": str, "bids_id": str})
by = pd.read_csv(f"{res}/roi_homogeneity_by_roi.csv").set_index("label")
mouse = pd.read_csv(f"{res}/roi_homogeneity_by_mouse.csv", dtype={"subject_id": str})
A = nib.load(f"{out}/slice_maps/atlas_commonspace.nii.gz").get_fdata().astype(int)
fails = 0
def check(name, ok, val):
    global fails
    fails += (not ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {val}")

# 1. independent implementation (np.corrcoef, eigvalsh) on 3 mice x 4 ROIs
dmax = 0
for bid in ["001", "014", "028"]:
    D = nib.load(f"{out}/confound_correction_outputs/confound_correction_datasink/cleaned_timeseries/"
                 f"_split_name_sub-{bid}_task-rest_bold/sub-{bid}_task-rest_bold_cleaned.nii.gz").get_fdata()
    for v in (1, 2, 3, 4):
        X = D[A == v].T
        m = X.mean(1)
        Xu = X[:, X.std(0) > 0]
        C = np.corrcoef(Xu.T)
        n = C.shape[0]
        ref = dict(r2_mean=np.mean([np.corrcoef(Xu[:, j], m)[0, 1] ** 2 for j in range(n)]),
                   r2_pc1=np.linalg.eigvalsh(C)[-1] / n, r2_pc2=np.linalg.eigvalsh(C)[-2] / n,
                   mean_r=C[np.triu_indices(n, 1)].mean(), n_vox_used=n)
        row = mr[(mr.bids_id.astype(str).str.zfill(3) == bid) & (mr.region == v)].iloc[0]
        for k, x in ref.items():
            dmax = max(dmax, abs(row[k] - x))
check("measures equal np.corrcoef / eigvalsh reference (max |diff|)", dmax < 1e-10, f"{dmax:.1e}")

# 2. known answers (medians over mice; group-effect ROI from groups without the effect)
med = lambda lab, k: by.loc[lab, f"{k}_median"]
check("HOMOG r2_mean ~ 0.50", abs(med("HOMOG", "r2_mean") - 0.50) < 0.02, f"{med('HOMOG', 'r2_mean'):.3f}")
check("HOMOG mean_r ~ 0.50", abs(med("HOMOG", "mean_r") - 0.50) < 0.02, f"{med('HOMOG', 'mean_r'):.3f}")
check("HOMOG mean/ceiling ~ 1", by.loc["HOMOG", "mean_over_pc1_median"] > 0.97, f"{by.loc['HOMOG', 'mean_over_pc1_median']:.3f}")
check("SPLIT r2_mean ~ 0.35", abs(med("SPLIT", "r2_mean") - 0.35) < 0.02, f"{med('SPLIT', 'r2_mean'):.3f}")
check("SPLIT r2_pc1 ~ (1 + 95 x 0.5)/128 = 0.379", abs(med("SPLIT", "r2_pc1") - 0.379) < 0.02, f"{med('SPLIT', 'r2_pc1'):.3f}")
check("SPLIT r2_pc2 ~ (1 + 31 x 0.5)/128 = 0.129", abs(med("SPLIT", "r2_pc2") - 0.129) < 0.02, f"{med('SPLIT', 'r2_pc2'):.3f}")
check("HOMOG r2_pc2 small (noise only)", med("HOMOG", "r2_pc2") < 0.03, f"{med('HOMOG', 'r2_pc2'):.3f}")
check("SMALL_EDGE: 32 atlas voxels, 24 used", (mr[mr.region == 3].n_vox.eq(32) & mr[mr.region == 3].n_vox_used.eq(24)).all(), "32 / 24")
g4 = mr[mr.region == 4]
check("GROUPFX r2_mean ~ 0.80 in 5XFAD_M, ~0.50 elsewhere",
      abs(g4[g4.group == "5XFAD_M"].r2_mean.median() - 0.80) < 0.02 and abs(g4[g4.group != "5XFAD_M"].r2_mean.median() - 0.50) < 0.02,
      f"{g4[g4.group == '5XFAD_M'].r2_mean.median():.3f} / {g4[g4.group != '5XFAD_M'].r2_mean.median():.3f}")
check("invariant r2_mean <= r2_pc1", (mr.r2_mean <= mr.r2_pc1 + 1e-12).all(), "all pairs")
# 3. group tests: planted effect found, others not
check("GROUPFX group effect q < 0.05", by.loc["GROUPFX", "welch_q_r2_mean"] < 0.05, f"q = {by.loc['GROUPFX', 'welch_q_r2_mean']:.4f}")
others = by.drop("GROUPFX").welch_q_r2_mean
check("no group effect in the other ROIs (q >= 0.05)", (others >= 0.05).all(), others.round(3).tolist())
# 4. file checks: FSL-spec eigenvariate files and ROI-mean files reproduced; negative controls flagged
m = mouse.set_index("subject_id")
good = m.drop(["3", "5"])
check("eigenvariate = FSL-spec files (lowest r, good mice)", good.eig_check_min_r.min() > 0.99999, f"{good.eig_check_min_r.min():.7f}")
check("ROI means = signal files (lowest r, good mice)", good.signals_check_min_r.min() > 0.99999, f"{good.signals_check_min_r.min():.7f}")
check("negative control: swapped columns flagged (mouse 3)", m.loc["3", "signals_check_min_r"] < 0.9, f"{m.loc['3', 'signals_check_min_r']:.3f}")
check("negative control: other mouse's eigenvariates flagged (mouse 5)", m.loc["5", "eig_check_min_r"] < 0.9, f"{m.loc['5', 'eig_check_min_r']:.3f}")
print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(fails > 0)
