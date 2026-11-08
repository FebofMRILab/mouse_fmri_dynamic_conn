# Synthetic EPI: 14 slices along axis 2 acquired interleaved, TR 2 s, 600 vols.
# raw_<order>: no STC. stcY: RABIES-like 3dTshift (Fourier, alt-z) along axis 1.
import numpy as np, nibabel as nib, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from stc_axis_check import pattern_times
rng = np.random.default_rng(5)
nx, ny, nz, T, TR = 32, 24, 14, 600, 2.0
f = np.fft.rfftfreq(T, TR); band = (f >= 0.01) & (f <= 0.1)
def bl(shape):                       # band-limited spectra (complex), last axis = freq
    Z = (rng.standard_normal(shape + (len(f),)) + 1j * rng.standard_normal(shape + (len(f),))) * band
    return Z
def delay(Z, d):                     # value at t + d  (d broadcast over leading dims)
    return np.fft.irfft(Z * np.exp(2j * np.pi * f * np.asarray(d)[..., None]), n=T, axis=-1)
X, Y, Zc = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
brain = ((X - 15.5) / 13) ** 2 + ((Y - 11.5) / 9) ** 2 + ((Zc - 6.5) / 7.5) ** 2 <= 1
aff = np.diag([0.25, 0.3125, 0.9, 1])
os.makedirs("data", exist_ok=True)
for case, order, grad in (("raw_altp", "alt+z", 0.0), ("raw_altm", "alt-z", 0.0),
                          ("raw_altp_grad", "alt+z", 0.5)):
    for m in range(8):
        G = bl(()); G /= np.sqrt(np.sum(np.abs(G) ** 2) / len(f))
        tau = pattern_times(order, nz, TR)[Zc]                    # acquisition time per voxel
        real = grad * (Zc / (nz - 1) - 0.5)                       # smooth real lag along axis 2
        sig = delay(np.broadcast_to(G, brain.shape + (len(f),)), tau - real)
        noise = np.fft.irfft(bl(brain.shape), n=T, axis=-1) / np.sqrt(len(f)) * 1.0
        D = np.where(brain[..., None], 1000 + 20 * (sig + noise), 50 + rng.standard_normal((nx, ny, nz, T)))
        img = nib.Nifti1Image(D.astype(np.float32), aff); img.header.set_zooms((0.25, 0.3125, 0.9, 1.5))
        nib.save(img, f"data/{case}_m{m}.nii.gz")
        if case == "raw_altp":       # STC applied along axis 1 with alt-z (as RABIES did)
            ta = pattern_times("alt-z", ny, TR)[Y]                # assumed time per row
            F = np.fft.rfft(D - D.mean(-1, keepdims=True), axis=-1)
            out = np.fft.irfft(F * np.exp(-2j * np.pi * f * ta[..., None]), n=T, axis=-1) + D.mean(-1, keepdims=True)
            nib.save(nib.Nifti1Image(out.astype(np.float32), aff), f"data/stcY_m{m}.nii.gz")
print("ok")
