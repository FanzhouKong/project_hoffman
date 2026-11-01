# data for the "neighbourhood noise" explainer page: one real file's noise surface as peak3d computes it (16 log-m/z
# bands x 1-minute RT blocks; per cell the median intensity of the centroids that form no trace, sidebands excluded;
# cells with < noise_cell_min_points fall back to the file's floor), one cell in detail (its centroids, which of them
# are noise, their median) and two candidates in that cell: a peak that clears 3x the cell noise and one that does not.
# usage: noise_cells_demo.py RUN OUT_JSON   (first file of the run; default settings, scan gain applied)
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS  # noqa: E402
from peak3d import kernels as K  # noqa: E402
from peak3d.estimate import prepare  # noqa: E402
from peak3d.features import build_features  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from peak3d.pick import SIDEBAND_PPM, SIDEBAND_RATIO, TRACE_MIN_SCANS, pick_cloud  # noqa: E402

run, out = sys.argv[1], Path(sys.argv[2])
rng = np.random.default_rng(0)
cl = load_cloud(RUNS[run]["files"][0])
P = prepare(cl)
b = pick_cloud(cl, P)
feats, rej = build_features(cl, b, keep_rejected=True)
off, mz, inten, rt = cl.off, cl.mz, cl.inten, cl.rt
B2 = int(b.lab.max()) + 1
smin, smax, _ = K.basin_extent(b.lab, off, B2)
isolated = K.short_basin_mask(b.lab, smin, smax, 2)
trace = ~K.short_basin_mask(b.lab, smin, smax, TRACE_MIN_SCANS - 1)
side = K.sideband_mask(off, mz, inten, isolated, trace, SIDEBAND_PPM, SIDEBAND_RATIO)
noise_pt = isolated & ~side
mz_lo, mz_hi = float(mz.min()), float(mz.max()) * (1 + 1e-9)
# the picker's surface comes from the labels after its first merge pass; b.lab holds the second pass, so the noise
# centroids here (for the picture) are a close re-derivation, while the cell values shown are the picker's own
rec, counts = K.cell_medians(off, mz, inten, noise_pt, rt, mz_lo, mz_hi, P.n_mz_bands, P.rt_block_min,
                             P.noise_cell_min_points, P.floor)
med = b.noise_surface
ok = (rec > P.floor) & (med > P.floor)
print(f"re-derived vs picker cell noise: median |ratio - 1| {np.median(np.abs(rec[ok] / med[ok] - 1)):.3f}, "
      f"max {np.max(np.abs(rec[ok] / med[ok] - 1)):.3f}, cells on the floor {(med <= P.floor).sum()} vs {(rec <= P.floor).sum()}")
n_bands, n_blocks = med.shape
edges_mz = np.exp(np.linspace(np.log(mz_lo), np.log(mz_hi), n_bands + 1))
edges_rt = rt[0] + P.rt_block_min * np.arange(n_blocks + 1)
scan = cl.scan_index()
band_of = np.clip(((np.log(mz) - np.log(mz_lo)) / ((np.log(mz_hi) - np.log(mz_lo)) / n_bands + 1e-12)).astype(int), 0, n_bands - 1)
blk_of = np.clip(((rt[scan] - rt[0]) / P.rt_block_min).astype(int), 0, n_blocks - 1)


def cell(m, r):
    i = int(np.clip(np.searchsorted(edges_mz, m, side="right") - 1, 0, n_bands - 1))
    j = int(np.clip(int((r - rt[0]) / P.rt_block_min), 0, n_blocks - 1))
    return i, j


# demo cell: a mid-m/z cell with its own noise estimate, a clear peak and a candidate rejected for low S/N
weak = rej[rej["reason"].astype(str).isin(["gate", "eic:lowsnr"]) & (rej["snr"] < P.min_snr) & (rej["snr"] > 1.2)
           & (rej["n_scans"] >= 5)]
best = None
for _, w in weak.iterrows():
    i, j = cell(w["mz"], w["rt"])
    if counts[i, j] < P.noise_cell_min_points or not (3 <= i <= n_bands - 4):
        continue
    f = feats[[cell(m, r) == (i, j) for m, r in zip(feats["mz"], feats["rt"])]] if best is None else None
    if f is None or f.empty:
        continue
    f = f[(f["snr"] > 8) & (f["snr"] < 80) & (f["n_scans"] >= 6)]
    if f.empty:
        continue
    score = counts[i, j]
    if best is None or score > best[0]:
        best = (score, i, j, w, f.iloc[int(np.argmin(np.abs(np.log(f["snr"].values / 20))))])
    break
assert best is not None, "no demo cell found"
_, ci, cj, w, g = best
in_cell = np.flatnonzero((band_of == ci) & (blk_of == cj))
pick = in_cell if len(in_cell) <= 4000 else np.sort(rng.choice(in_cell, 4000, replace=False))


def eic(m, r, half=0.45):
    s0, s1 = np.searchsorted(rt, [r - half, r + half])
    e = cl.eic(m, max(3 * float(P.sig_ppm(1e5)) + 1, 5.0), s0, s1)
    return dict(rt=[round(float(v), 4) for v in rt[s0:s1]], y=[round(float(v), 1) for v in e])


cand = lambda x, ok: dict(mz=round(float(x["mz"]), 4), rt=round(float(x["rt"]), 3), height=round(float(x["height"])),
                          snr=round(float(x["snr"]), 2), lo=round(float(x["rt_min"]), 4), hi=round(float(x["rt_max"]), 4),
                          passed=ok, eic=eic(float(x["mz"]), float(x["rt"])))
data = dict(run=run, file=Path(RUNS[run]["files"][0]).name, n_scans=int(cl.n_scans), n_centroids=int(cl.n_points),
            n_noise=int(noise_pt.sum()), floor=float(P.floor), min_points=int(P.noise_cell_min_points), k=float(P.min_snr),
            scan_s=float(np.median(np.diff(rt)) * 60), edges_mz=[round(float(v), 3) for v in edges_mz],
            edges_rt=[round(float(v), 4) for v in edges_rt], med=[[round(float(v), 1) for v in row] for row in med],
            counts=counts.tolist(),
            cell=dict(band=ci, block=cj, noise=float(med[ci, cj]), n_noise=int(counts[ci, cj]), n_all=int(len(in_cell)),
                      pts=[[round(float(rt[scan[p]]), 4), round(float(mz[p]), 4), round(float(inten[p]), 1), int(noise_pt[p]),
                            int(trace[p])] for p in pick]),
            passing=cand(g, True), failing=cand(w, False))
out.write_text(json.dumps(data, separators=(",", ":")))
print(f"{run}: {n_bands} x {n_blocks} cells, {int(noise_pt.sum()):,} noise centroids of {cl.n_points:,}; demo cell band {ci} "
      f"({edges_mz[ci]:.1f}-{edges_mz[ci + 1]:.1f}) block {cj} ({edges_rt[cj]:.2f}-{edges_rt[cj + 1]:.2f} min): noise "
      f"{med[ci, cj]:,.0f} from {counts[ci, cj]} centroids of {len(in_cell)}; passing m/z {g['mz']:.4f} S/N {g['snr']:.1f}; "
      f"failing m/z {w['mz']:.4f} S/N {w['snr']:.2f} ({w['reason']}); {out.stat().st_size / 1e3:.0f} kB")
