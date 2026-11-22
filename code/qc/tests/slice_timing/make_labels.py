# Simulated native-space label maps: 82 compact ROIs in an ellipsoid brain,
# 20 coronal slices (thick axis = voxel axis 1), 10 mice with jitter.
import numpy as np, nibabel as nib, os
rng = np.random.default_rng(7)
nx, ny, nz = 24, 20, 14
X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
mask = ((X - 11.5) / 11.5) ** 2 + ((Y - 9.5) / 10.5) ** 2 + ((Z - 6.5) / 6.5) ** 2 <= 1
# physical coordinates (mm) so ROIs are compact in space, not in voxels
P = np.stack([X * 0.2, Y * 0.6, Z * 0.2], -1)[mask]
seeds = P[rng.choice(len(P), 82, replace=False)]
lab = np.zeros(mask.shape, int)
lab[mask] = 1 + np.argmin(((P[:, None, :] - seeds[None]) ** 2).sum(-1), axis=1)
aff = np.diag([0.2, 0.6, 0.2, 1.0])
os.makedirs("labels", exist_ok=True)
for m in range(20):
    L = np.roll(lab, rng.integers(-2, 3), axis=1)
    L = np.roll(L, rng.integers(-1, 2), axis=0)
    nib.save(nib.Nifti1Image(L.astype(np.int16), aff), f"labels/sub-{m+1:02d}_native_labels.nii.gz")
    # voxel counts per ROI x slice for the signal simulation
    H = np.stack([np.bincount(L[:, s, :].ravel(), minlength=83)[1:] for s in range(ny)], 1)
    np.savetxt(f"labels/H_{m+1:02d}.txt", H, fmt="%d")
print("ROI sizes (voxels):", np.bincount(lab.ravel())[1:].min(), "-", np.bincount(lab.ravel())[1:].max())
sl = [(np.bincount(np.nonzero(lab == r)[1], minlength=ny) > 0).sum() for r in range(1, 83)]
print("slices per ROI: median", np.median(sl), "range", min(sl), max(sl))
