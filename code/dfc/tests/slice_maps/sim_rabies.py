"""
Synthetic RABIES 0.6.1 output tree for testing make_slice_maps.py and
slice_timing_lag_check.py --fractions. Native EPI: 14 slices along voxel axis
2 (physical y, front-back), acquired interleaved 0,2,...,12,1,...,13 over TR 2 s.
Per mouse: rigid bold->anat (different slice placement: y shift up to
+-1.5 slices, small rotation), affine + smooth warp anat->unbiased, group
unbiased->atlas affine + warp. commonspace_bold = native BOLD pushed with the
chain in RABIES's order; ROI time series = atlas means in commonspace.
Scenario 'none': each native slice samples the signal at its acquisition time
(uncorrected). Scenario 'correct': all slices at the volume time.
"""
import os, sys, json, numpy as np, nibabel as nib, ants
rng = np.random.default_rng(11)
root = sys.argv[1]; M = int(sys.argv[2]) if len(sys.argv) > 2 else 12
TR, T, n = 2.0, 570, 14
order = list(range(0, n, 2)) + list(range(1, n, 2))
tau = np.empty(n); tau[order] = np.arange(n) * TR / n
# commonspace grid (atlas), 0.3 mm, origin at centre
cs_shape, cs_sp = (36, 44, 30), 0.3
cs_aff = np.diag([cs_sp] * 3 + [1.0]); cs_aff[:3, 3] = -0.5 * cs_sp * (np.array(cs_shape) - 1)
I, J, K = np.indices(cs_shape)
P = np.stack([I, J, K, np.ones_like(I)], -1) @ cs_aff.T
x, y, z = P[..., 0], P[..., 1], P[..., 2]
brain = (x / 4.6) ** 2 + (y / 5.8) ** 2 + (z / 3.8) ** 2 <= 1
pts = np.column_stack([x[brain], y[brain], z[brain]])
seeds = pts[rng.choice(len(pts), 82, replace=False)]
lab = np.zeros(cs_shape, np.int16)
lab[brain] = 1 + np.argmin(((pts[:, None] - seeds[None]) ** 2).sum(-1), 1)
os.makedirs(f"{root}/atlas", exist_ok=True)
nib.save(nib.Nifti1Image(lab, cs_aff), f"{root}/atlas/atlas.nii.gz")
# native grid: voxel axes (i -> x, j -> z, k -> y), 0.5 x 0.5 x 0.9 mm, 14 slices along k (= y)
nat_shape = (24, 20, n)
nat_aff = np.array([[0.5, 0, 0, 0], [0, 0, 0.9, 0], [0, 0.5, 0, 0], [0, 0, 0, 1.0]])
nat_aff[:3, 3] = -nat_aff[:3, :3] @ ((np.array(nat_shape) - 1) / 2)
fixed_cs = ants.from_nibabel_nifti(nib.Nifti1Image(np.zeros(cs_shape, np.float32), cs_aff))
def affine_tx(path, rot_deg=(0, 0, 0), shift=(0, 0, 0), scale=1.0):
    a, b, c = np.deg2rad(rot_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3,
                                    matrix=(scale * Rx @ Ry @ Rz), translation=list(shift))
    ants.write_transform(tx, path)
def warp_field(path, amp):
    g = np.stack([np.sin(2 * np.pi * (y / 9 + ph)) * np.cos(2 * np.pi * x / 11) for ph in rng.uniform(0, 1, 3)], -1)
    f = ants.from_numpy((amp * g).astype(np.float32), origin=fixed_cs.origin, spacing=fixed_cs.spacing,
                        direction=fixed_cs.direction, has_components=True)
    ants.image_write(f, path)
Td = f"{root}/preprocess_outputs/transforms_datasink"
for d in ["unbiased_to_atlas_affine", "unbiased_to_atlas_warp"]:
    os.makedirs(f"{Td}/{d}", exist_ok=True)
affine_tx(f"{Td}/unbiased_to_atlas_affine/unbiased_template_0GenericAffine.mat", (1, 0, 0), (0.1, 0.0, -0.1), 1.01)
warp_field(f"{Td}/unbiased_to_atlas_warp/unbiased_template_1Warp.nii.gz", 0.10)
# signals: global source with true lag 0 + seesaw source + ROI noise, band-limited
f = np.fft.rfftfreq(4 * T, TR / 4); band = (f >= 0.01) & (f <= 0.1)
def bl(k):
    Z = (rng.standard_normal((k, len(f))) + 1j * rng.standard_normal((k, len(f)))) * band
    return Z / np.sqrt(band.sum())
def sample(Z, times):          # value at arbitrary times (s)
    return np.real(Z @ np.exp(2j * np.pi * np.outer(f, times))) * 2
sgn = np.sign(rng.uniform(-1, 1, 82))
meta = []
for m in range(1, M + 1):
    bid = f"{m:03d}"
    sd = f"_scan_info_subject_id{bid}.sessionNone_split_name_sub-{bid}_T2w"
    paths = {}
    for d in ["bold_to_anat_affine", "bold_to_anat_warp"]:
        os.makedirs(f"{Td}/{d}/{sd}/_run_None", exist_ok=True)
    for d in ["anat_to_unbiased_affine", "anat_to_unbiased_warp"]:
        os.makedirs(f"{Td}/{d}", exist_ok=True)
    paths["b2a_a"] = f"{Td}/bold_to_anat_affine/{sd}/_run_None/sub-{bid}_task-rest_bold_inho_cor_output_0GenericAffine.mat"
    paths["b2a_w"] = f"{Td}/bold_to_anat_warp/{sd}/_run_None/sub-{bid}_task-rest_bold_inho_cor_output_1Warp.nii.gz"
    paths["a2u_a"] = f"{Td}/anat_to_unbiased_affine/sub-{bid}_T2w_RAS_inho_cor_0GenericAffine.mat"
    paths["a2u_w"] = f"{Td}/anat_to_unbiased_warp/sub-{bid}_T2w_RAS_inho_cor_1Warp.nii.gz"
    affine_tx(paths["b2a_a"], rng.uniform(-3, 3, 3), (rng.uniform(-.3, .3), rng.uniform(-1.35, 1.35), rng.uniform(-.3, .3)))
    warp_field(paths["b2a_w"], 0.12)
    affine_tx(paths["a2u_a"], rng.uniform(-2, 2, 3), rng.uniform(-.2, .2, 3), rng.uniform(0.97, 1.03))
    warp_field(paths["a2u_w"], 0.12)
    chain = [f"{Td}/unbiased_to_atlas_warp/unbiased_template_1Warp.nii.gz",
             f"{Td}/unbiased_to_atlas_affine/unbiased_template_0GenericAffine.mat",
             paths["a2u_w"], paths["a2u_a"], paths["b2a_w"], paths["b2a_a"]]
    # forward map: native physical point of every commonspace voxel (push coordinate images)
    nat_vox = np.indices(nat_shape).reshape(3, -1).T
    nat_phys = (np.column_stack([nat_vox, np.ones(len(nat_vox))]) @ nat_aff.T)[:, :3]
    coords = []
    for c in range(3):
        img = ants.from_nibabel_nifti(nib.Nifti1Image(nat_phys[:, c].reshape(nat_shape).astype(np.float32), nat_aff))
        coords.append(ants.apply_transforms(fixed_cs, img, chain, interpolator="linear").numpy())
    # native voxel each brain commonspace voxel lands in -> ROI of each native voxel (mode)
    pc = np.stack(coords, -1)[brain]
    vox = np.rint(np.column_stack([pc, np.ones(len(pc))]) @ np.linalg.inv(nat_aff).T)[:, :3].astype(int)
    ok = np.all((vox >= 0) & (vox < nat_shape), 1)
    nat_roi = np.zeros(nat_shape, int)
    lin = np.ravel_multi_index(vox[ok].T, nat_shape)
    for v in np.unique(lin):
        nat_roi.flat[v] = np.bincount(lab[brain][ok][lin == v]).argmax()
    # native BOLD: ROI signal sampled at slice acquisition time (scenario) + voxel noise
    G, S2, E = bl(1), bl(1), bl(82)
    for scen, off in (("none", tau), ("correct", np.zeros(n))):
        D = np.zeros(nat_shape + (T,), np.float32)
        tt = np.arange(T) * TR
        for k in range(n):
            sl = nat_roi[:, :, k]
            rois = np.unique(sl[sl > 0])
            if not len(rois):
                continue
            t = tt + off[k]
            g, s2 = sample(G, t)[0], sample(S2, t)[0]
            e = sample(E[rois - 1], t)
            for q, r in enumerate(rois):
                D[:, :, k][sl == r] = 1000 + 20 * (g + 0.9 * sgn[r - 1] * s2 + 1.5 * e[q])
        D[nat_roi == 0] = 50
        D += rng.standard_normal(D.shape).astype(np.float32) * 2
        base = f"{root}/{scen}/preprocess_outputs"
        if True:  # both scenarios
            os.makedirs(f"{base}/bold_datasink/input_bold", exist_ok=True)
            nib.save(nib.Nifti1Image(D, nat_aff), f"{base}/bold_datasink/input_bold/sub-{bid}_task-rest_bold.nii.gz")
            img = ants.from_nibabel_nifti(nib.Nifti1Image(D, nat_aff))
            W = ants.apply_transforms(fixed_cs, img, chain, interpolator="linear", imagetype=3).numpy()
            od = f"{base}/bold_datasink/commonspace_bold/{sd}/_run_None"
            os.makedirs(od, exist_ok=True)
            nib.save(nib.Nifti1Image(W.astype(np.float32), cs_aff), f"{od}/sub-{bid}_task-rest_bold_resampled.nii.gz")
            X = np.column_stack([W[lab == r].mean(0) for r in range(1, 83)])
            os.makedirs(f"{root}/{scen}/hmm_signals", exist_ok=True)
            np.savetxt(f"{root}/{scen}/hmm_signals/signal{m}.txt", X, fmt="%.6f")
    meta.append(dict(subject_id=m, group="A" if m % 2 else "B", bids_id=f"sub-{bid}"))
    print("mouse", bid, flush=True)
import pandas as pd
for scen in ("none", "correct"):
    pd.DataFrame(meta).to_csv(f"{root}/{scen}/meta_data_with_bids.csv", index=False)
    os.system(f"ln -sfn ../../preprocess_outputs/transforms_datasink {root}/{scen}/preprocess_outputs/transforms_datasink")
