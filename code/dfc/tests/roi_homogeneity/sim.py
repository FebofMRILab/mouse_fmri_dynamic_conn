"""Synthetic RABIES-like tree with known answers for roi_signal_homogeneity.py.

Grid 16 x 16 x 8, T = 300, white noise (sd 1) in every voxel, 28 mice (4 per group).
  region 1  128 voxels, one signal, amplitude 1      -> r2_mean ~ r2_pc1 ~ mean_r ~ 0.50
  region 2  128 voxels, 96 with signal s1, 32 with s2 -> r2_mean ~ 0.35, r2_pc1 ~ 0.379, r2_pc2 ~ 0.129
  region 3  32 voxels, one signal, 8 voxels constant 0 (outside the brain mask) -> n_vox_used 24
  region 4  128 voxels, one signal, amplitude 1, but 2 in group 5XFAD_M -> r2 0.50 vs 0.80 (group effect)
FSL-style eigenvariates (time x time covariance, eigh, as fslmeants.cc) and ROI means are
written as text with 6 significant digits, like fslmeants output. Negative controls:
mouse 3's ROI-mean file has columns 1 and 2 swapped; mouse 5's eigenvariate file is from mouse 6."""
import os, sys
import numpy as np, pandas as pd, nibabel as nib

out = sys.argv[1]
rng = np.random.default_rng(7)
shape, T = (16, 16, 8), 300
A = np.zeros(shape, int)
A[0:8, 0:8, 0:2] = 1
A[8:16, 0:8, 0:2] = 2
A[0:4, 8:12, 2:4] = 3
A[0:8, 0:8, 4:6] = 4
groups = ["NTG_Y", "NTG_M", "NTG_A", "CRND8_Y", "CRND8_M", "5XFAD_Y", "5XFAD_M"]
os.makedirs(f"{out}/slice_maps", exist_ok=True)
aff = np.diag([0.2, 0.2, 0.5, 1])
nib.save(nib.Nifti1Image(A.astype(np.int16), aff), f"{out}/slice_maps/atlas_commonspace.nii.gz")
pd.DataFrame({"region": [1, 2, 3, 4], "atlas_value": [1, 2, 3, 4],
              "label": ["HOMOG", "SPLIT", "SMALL_EDGE", "GROUPFX"]}).to_csv(f"{out}/roi_labels.csv", index=False)

def fsl_eig_spec(X):
    dat = X - X.mean(0)                      # remmean(dat, 1)
    C = dat @ dat.T / dat.shape[1]           # Corr << dat * dat.t() / dat.Ncols()
    w, V = np.linalg.eigh(C)                 # ascending; take the largest
    e = V[:, -1] * np.sqrt(dat.shape[0])
    if dat.mean(1) @ e < 0:
        e = -e
    return e

meta, eigs, means = [], {}, {}
for i in range(28):
    sid, bid, g = str(i + 1), f"{i + 1:03d}", groups[i // 4]
    D = rng.standard_normal(shape + (T,)).astype(np.float32)
    s = rng.standard_normal((5, T))
    D[A == 1] += s[0]
    idx2 = np.argwhere(A == 2)
    for k, (x, y, z) in enumerate(idx2):
        D[x, y, z] += s[1] if k < 96 else s[2]
    D[A == 3] += s[3]
    edge = np.argwhere(A == 3)[:8]
    for x, y, z in edge:
        D[x, y, z] = 0.0
    D[A == 4] += (2.0 if g == "5XFAD_M" else 1.0) * s[4]
    d = f"{out}/confound_correction_outputs/confound_correction_datasink/cleaned_timeseries/_split_name_sub-{bid}_task-rest_bold"
    os.makedirs(d, exist_ok=True)
    nib.save(nib.Nifti1Image(D, aff), f"{d}/sub-{bid}_task-rest_bold_cleaned.nii.gz")
    means[sid] = np.column_stack([D[A == v].mean(0) for v in (1, 2, 3, 4)])
    eigs[sid] = np.column_stack([fsl_eig_spec(D[A == v].T.astype(np.float64)) for v in (1, 2, 3, 4)])
    meta.append(dict(subject_id=sid, group=g, bids_id=bid))
pd.DataFrame(meta).to_csv(f"{out}/meta_data_with_bids.csv", index=False)
os.makedirs(f"{out}/hmm_signals", exist_ok=True)
os.makedirs(f"{out}/hmm_signals_eig", exist_ok=True)
for sid in means:
    M = means[sid].copy()
    if sid == "3":
        M[:, [0, 1]] = M[:, [1, 0]]
    E = eigs["6"] if sid == "5" else eigs[sid]
    np.savetxt(f"{out}/hmm_signals/signal{sid}.txt", M, fmt="%.6g")
    np.savetxt(f"{out}/hmm_signals_eig/signal{sid}.txt", E, fmt="%.6g")
print("simulated", len(meta), "mice")
