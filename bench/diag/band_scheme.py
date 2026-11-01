# diagnostic: would equal-Da m/z bands give a different neighbourhood noise than the picker's equal-log bands?
# Same number of bands (P.n_mz_bands) and the same 1-minute blocks; the noise centroids are the picker's (centroids
# whose final basin spans <= 2 scans, sidebands of strong traces excluded). Per scheme: how uniform the noise is
# inside a cell (median of the upper half of the cell's m/z range / median of the lower half, |log2| of it), how many
# cells fall back to the floor (< noise_cell_min_points), and for the candidates (kept features and those whose only
# failing gate is S/N): how many would change side of the S/N >= min_snr gate if their cell noise came from the other scheme.
# usage: band_scheme.py RUN ...   (first file of each run; default settings; run on a compute node)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS  # noqa: E402
from peak3d import kernels as K  # noqa: E402
from peak3d.estimate import prepare  # noqa: E402
from dataclasses import replace  # noqa: E402

from peak3d.features import build_features, gate  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from peak3d.pick import SIDEBAND_PPM, SIDEBAND_RATIO, TRACE_MIN_SCANS, pick_cloud  # noqa: E402


def grid(mz, rt_pt, edges, rt0, block, n_blocks, inten, min_points, floor):
    """per cell: median intensity, count, and the |log2| ratio of the upper-half to lower-half medians"""
    nb = len(edges) - 1
    band = np.clip(np.searchsorted(edges, mz, side="right") - 1, 0, nb - 1)
    blk = np.clip(((rt_pt - rt0) / block).astype(int), 0, n_blocks - 1)
    med = np.full((nb, n_blocks), floor)
    cnt = np.zeros((nb, n_blocks), int)
    het = np.full((nb, n_blocks), np.nan)
    key = band * n_blocks + blk
    order = np.argsort(key, kind="stable")
    ks, starts = np.unique(key[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    for k, a, b in zip(ks, starts, ends):
        i, j = divmod(int(k), n_blocks)
        idx = order[a:b]
        cnt[i, j] = len(idx)
        if len(idx) >= min_points:
            med[i, j] = np.median(inten[idx])
            mid = 0.5 * (edges[i] + edges[i + 1]) if SCHEME == "da" else np.sqrt(edges[i] * edges[i + 1])
            lo, hi = inten[idx][mz[idx] < mid], inten[idx][mz[idx] >= mid]
            if len(lo) >= min_points // 2 and len(hi) >= min_points // 2:
                het[i, j] = abs(np.log2(np.median(hi) / np.median(lo)))
    return med, cnt, het, band, blk


rows = []
for run in sys.argv[1:]:
    cl = load_cloud(RUNS[run]["files"][0])
    P = prepare(cl)
    b = pick_cloud(cl, P)
    feats, rej = build_features(cl, b, keep_rejected=True)
    off, mz, inten, rt = cl.off, cl.mz, cl.inten, cl.rt
    smin, smax, _ = K.basin_extent(b.lab, off, int(b.lab.max()) + 1)
    isolated = K.short_basin_mask(b.lab, smin, smax, 2)
    trace = ~K.short_basin_mask(b.lab, smin, smax, TRACE_MIN_SCANS - 1)
    noise_pt = isolated & ~K.sideband_mask(off, mz, inten, isolated, trace, SIDEBAND_PPM, SIDEBAND_RATIO)
    lo_mz, hi_mz = float(mz.min()), float(mz.max()) * (1 + 1e-9)
    n_blocks = max(1, int(np.ceil((rt[-1] - rt[0]) / P.rt_block_min)))
    rt_pt = rt[cl.scan_index()]
    out = {}
    for SCHEME, edges in [("log", np.exp(np.linspace(np.log(lo_mz), np.log(hi_mz), P.n_mz_bands + 1))),
                          ("da", np.linspace(lo_mz, hi_mz, P.n_mz_bands + 1))]:
        med, cnt, het, _, _ = grid(mz[noise_pt], rt_pt[noise_pt], edges, rt[0], P.rt_block_min, n_blocks, inten[noise_pt],
                                   P.noise_cell_min_points, P.floor)
        out[SCHEME] = (edges, med)
        h = het[np.isfinite(het)]
        rows.append(dict(run=run, scheme=SCHEME, cells=med.size, cells_on_floor=int((cnt < P.noise_cell_min_points).sum()),
                         median_half_ratio=float(2 ** np.median(h)) if len(h) else np.nan,
                         p90_half_ratio=float(2 ** np.quantile(h, 0.9)) if len(h) else np.nan,
                         band_lowest=f"{edges[0]:.0f}-{edges[1]:.0f}", band_highest=f"{edges[-2]:.0f}-{edges[-1]:.0f}"))
    # candidates: kept features and those rejected only by the S/N gate; their cell noise under each scheme
    g = rej[rej["reason"].astype(str) == "gate"]
    sn_only = g[gate(g, replace(P, min_snr=0.0)).values]          # every other gate passes: rejected for S/N alone
    cand = pd.concat([feats.assign(kept=True), sn_only.assign(kept=False)], ignore_index=True)
    cand = cand[np.isfinite(cand["snr"])]
    base = cand["height"].values - cand["snr"].values * cand["noise"].values
    blk = np.clip(((cand["rt"].values - rt[0]) / P.rt_block_min).astype(int), 0, n_blocks - 1)
    nz = {}
    for s, (edges, med) in out.items():
        band = np.clip(np.searchsorted(edges, cand["mz"].values, side="right") - 1, 0, len(edges) - 2)
        nz[s] = med[band, blk]
    snr_log = (cand["height"].values - base) / nz["log"]
    snr_da = (cand["height"].values - base) / nz["da"]
    ratio = nz["da"] / nz["log"]
    k = P.min_snr
    rows[-1].update(candidates=len(cand), noise_da_over_log_median=float(np.median(ratio)),
                    noise_da_over_log_p10=float(np.quantile(ratio, 0.1)), noise_da_over_log_p90=float(np.quantile(ratio, 0.9)),
                    kept_that_would_fail=int(((snr_log >= k) & (snr_da < k) & cand["kept"].values).sum()),
                    sn_rejects_that_would_pass=int(((snr_log < k) & (snr_da >= k) & ~cand["kept"].values).sum()),
                    kept=int(cand["kept"].sum()), sn_only_rejects=int((~cand["kept"]).sum()))
    print(f"{run}: done", flush=True)
T = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(T.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
T.to_csv(ROOT / "results/diag_double/band_scheme.tsv", sep="\t", index=False)
