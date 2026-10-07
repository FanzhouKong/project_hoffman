# diagnostic: what are the "noise points" behind peak3d's noise surface (Oct 2 picker settings)?
# The surface is the median intensity per (m/z band, 1-min block) of centroids whose INITIAL basin
# (steepest-ascent, before the persistence merge) spans <= 2 scans. Each noise point is classed by
# where it ends up after the merge: still a <= 2-scan basin (isolated), a 3-4 scan blip (short), or
# part of a basin >= 5 scans (a ripple maximum on a real ion's trace). Writes the region's point cloud,
# the surface cells, the noise-point intensities and the missed certified peaks for the 3D page.
# usage: noise_surface_region.py [FILE_INDEX=1] [RT_LO=1.0] [RT_HI=2.0] [MZ_LO=550] [MZ_HI=1000]
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS  # noqa: E402
from peak3d import kernels as K  # noqa: E402
from peak3d.estimate import estimate  # noqa: E402
from peak3d.features import build_features  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from peak3d.pick import pick_cloud  # noqa: E402

OLD = dict(far_frac=0.0, noise_band_cap=False, snr_local=False, snr_smooth_apex=False, min_snr=3.0)
WEAK = [124, 136, 385, 362, 163, 344, 155, 368, 379, 382, 102]


def main(fi=1, rt_lo=1.0, rt_hi=2.0, mz_lo_r=550.0, mz_hi_r=1000.0):
    path = RUNS["HZV029_cert"]["files"][fi]
    stem = Path(path).stem
    cloud = load_cloud(path)
    P = replace(estimate(cloud), **OLD)
    off, mz, inten, rt = cloud.off, cloud.mz, cloud.inten, cloud.rt
    scan_of = cloud.scan_index()
    # the picker's own steps up to the noise surface
    sig = K.per_point_sigma(inten, P.sig_a, P.sig_b)
    nbr = K.find_neighbors(off, mz, inten, sig, P.K, P.tol_max_ppm, P.same_scan_k, P.same_scan_floor_ppm)
    eff = K.effective_intensity(nbr, inten)
    lab, roots = K.dense_labels(K.resolve_roots(K.link_steepest(nbr, eff, inten)))
    smin, smax, _ = K.basin_extent(lab, off, len(roots))
    mz_lo, mz_hi = float(mz.min()), float(mz.max()) * (1 + 1e-9)
    med, counts = K.noise_cell_medians(off, mz, inten, lab, smin, smax, rt, mz_lo, mz_hi, P.n_mz_bands,
                                       P.rt_block_min, P.noise_cell_min_points, P.floor)
    # after the merge
    b = pick_cloud(cloud, P)
    feats, rej = build_features(cloud, b, None, True)
    lab2 = b.lab
    s2min, s2max, _ = K.basin_extent(lab2, off, int(lab2.max()) + 1)
    span0 = (smax - smin + 1)[lab]
    span2 = (s2max - s2min + 1)[lab2]
    noise_pt = span0 <= 2
    cls = np.full(len(mz), -1, np.int8)              # -1 not a noise point
    cls[noise_pt & (span2 <= 2)] = 0                 # isolated: still <= 2 scans after the merge
    cls[noise_pt & (span2 >= 3) & (span2 <= 4)] = 1  # short blip
    cls[noise_pt & (span2 >= 5)] = 2                 # ripple on a real ion's trace
    # isolated points within 100 ppm of a >= 10x stronger trace point in the same scan: FT sidebands
    for p in np.flatnonzero(cls == 0):
        s = scan_of[p]
        a, z = off[s], off[s + 1]
        lo = a + np.searchsorted(mz[a:z], mz[p] * (1 - 100e-6))
        hi = a + np.searchsorted(mz[a:z], mz[p] * (1 + 100e-6))
        q = np.arange(lo, hi)
        q = q[(q != p) & (span2[q] >= 5)]
        if len(q) and inten[q].max() >= 10 * inten[p]:
            cls[p] = 3
    lmz_lo = np.log(mz_lo)
    w = (np.log(mz_hi) - lmz_lo) / P.n_mz_bands + 1e-12
    band = np.clip(((np.log(mz) - lmz_lo) / w).astype(np.int64), 0, P.n_mz_bands - 1)
    n_blocks = med.shape[1]
    blk = np.clip(((rt[scan_of] - rt[0]) / P.rt_block_min).astype(np.int64), 0, n_blocks - 1)
    cell = band * n_blocks + blk

    def cell_median(sel):
        out = np.full(med.shape, np.nan)
        c = cell[sel]
        v = inten[sel]
        o = np.argsort(c, kind="stable")
        c, v = c[o], v[o]
        bounds = np.flatnonzero(np.diff(np.r_[-1, c, -2]))
        for a, z in zip(bounds[:-1], bounds[1:]):
            if z - a >= P.noise_cell_min_points:
                out.flat[c[a]] = float(np.median(v[a:z]))
        return out

    med_iso = cell_median((cls == 0) | (cls == 3))
    med_iso_short = cell_median((cls == 0) | (cls == 1) | (cls == 3))
    med_clean = cell_median(cls == 0)
    frac_ripple = np.full(med.shape, np.nan)
    for c in np.unique(cell[noise_pt]):
        s = noise_pt & (cell == c)
        frac_ripple.flat[c] = float((cls[s] == 2).mean())
    band_edges = np.exp(lmz_lo + w * np.arange(P.n_mz_bands + 1))
    blk_edges = rt[0] + P.rt_block_min * np.arange(n_blocks + 1)
    # whole-file summary
    print(f"{stem}: {len(mz)} centroids; noise points {noise_pt.sum()} ({noise_pt.mean():.1%}); of these "
          f"isolated {np.mean(np.isin(cls[noise_pt], (0, 3))):.1%} (sidebands {np.mean(cls[noise_pt] == 3):.1%}), short {np.mean(cls[noise_pt] == 1):.1%}, "
          f"ripple on a >= 5-scan trace {np.mean(cls[noise_pt] == 2):.1%}; file floor {P.floor:.0f}")
    print("median intensity of noise points: isolated %.0f, short %.0f, ripple %.0f" %
          tuple(np.median(inten[np.isin(cls, kk)]) for kk in ((0, 3), (1,), (2,))))
    # region
    reg = (rt[scan_of] >= rt_lo) & (rt[scan_of] <= rt_hi) & (mz >= mz_lo_r) & (mz <= mz_hi_r)
    print(f"region RT {rt_lo}-{rt_hi} min, m/z {mz_lo_r}-{mz_hi_r}: {reg.sum()} centroids, noise points "
          f"{(reg & noise_pt).sum()}: isolated {np.mean(cls[reg & noise_pt] == 0):.1%}, sidebands {np.mean(cls[reg & noise_pt] == 3):.1%}, short "
          f"{np.mean(cls[reg & noise_pt] == 1):.1%}, ripple {np.mean(cls[reg & noise_pt] == 2):.1%}")
    rows = []
    for bi in range(P.n_mz_bands):
        if band_edges[bi + 1] < mz_lo_r or band_edges[bi] > mz_hi_r:
            continue
        for bj in range(n_blocks):
            if blk_edges[bj + 1] < rt_lo or blk_edges[bj] > rt_hi:
                continue
            rows.append(dict(band=bi, mz_from=round(band_edges[bi], 1), mz_to=round(band_edges[bi + 1], 1),
                             block=bj, rt_from=round(blk_edges[bj], 3), rt_to=round(blk_edges[bj + 1], 3),
                             n_noise=int(counts[bi, bj]), frac_ripple=round(frac_ripple[bi, bj], 3),
                             surface_now=round(med[bi, bj]), surface_isolated=round(med_iso[bi, bj]),
                             surface_iso_short=round(med_iso_short[bi, bj]), surface_clean=round(med_clean[bi, bj]),
                             frac_sideband=round(float(np.mean(cls[noise_pt & (band == bi) & (blk == bj)] == 3)), 3)))
    C = pd.DataFrame(rows)
    print(C.to_string(index=False))
    # whole-file surfaces
    ratio = med / med_iso
    print(f"\nwhole file: surface now / isolated-only, median over cells {np.nanmedian(ratio):.2f}, "
          f"max {np.nanmax(ratio):.2f}; cells where ripples are > 50 % of the noise points: "
          f"{int(np.nansum(frac_ripple > 0.5))} of {int(np.isfinite(frac_ripple).sum())}")
    # missed certified peaks: candidate S/N now and against the isolated-only surface
    J = json.loads((ROOT / "results/diag_excess/hzv029_missed_2026-10-02.json").read_text())
    items = {it["truth_id"]: it for it in J["items"]}
    allc = pd.concat([feats.assign(stage="kept"), rej.assign(stage=rej["reason"])], ignore_index=True)
    peaks = []
    for i in WEAK:
        it = items[i]
        rtn = [f for f in it["files"] if f["stem"] == stem][0]["rt_native_of_truth"]
        x = allc[(np.abs(allc["mz"] - it["mz"]) / it["mz"] * 1e6 <= 10) & (np.abs(allc["rt"] - rtn) <= 0.1)]
        x = x.sort_values("height", ascending=False).head(1)
        bb = int(np.clip((np.log(it["mz"]) - lmz_lo) / w, 0, P.n_mz_bands - 1))
        kk = int(np.clip((rtn - rt[0]) / P.rt_block_min, 0, n_blocks - 1))
        s0, s1 = int(np.searchsorted(rt, rtn - 0.3)), int(np.searchsorted(rt, rtn + 0.3))
        e = cloud.eic(it["mz"], 5.0, s0, s1)
        d = dict(id=i, mz=it["mz"], rt=rtn, cert_rt=it["rt"], band=bb, block=kk,
                 surface_now=float(med[bb, kk]), surface_isolated=float(med_iso[bb, kk]), surface_clean=float(med_clean[bb, kk]),
                 eic_rt=[round(float(v), 5) for v in rt[s0:s1]], eic=[float(v) for v in e],
                 zero_frac=float(np.mean(e == 0)))
        if len(x):
            r = x.iloc[0]
            d.update(height=float(r["height"]), baseline=float(r["baseline"]), stage=str(r["stage"]),
                     snr_now=float(r["snr"]),
                     snr_isolated=float((r["height"] - r["baseline"]) / med_iso[bb, kk]),
                     snr_isolated_nobase=float(r["height"] / med_iso[bb, kk]),
                     snr_clean_nobase=float(r["height"] / med_clean[bb, kk]),
                     rt_min=float(r["rt_min"]), rt_max=float(r["rt_max"]))
        peaks.append(d)
    Pk = pd.DataFrame(peaks)
    print(Pk[[c for c in ("id", "mz", "rt", "height", "baseline", "stage", "surface_now", "surface_isolated",
                          "snr_now", "snr_isolated", "snr_isolated_nobase", "zero_frac") if c in Pk]].round(2).to_string(index=False))
    # page data: region points (trace points thinned to keep the 3D view light)
    idx = np.flatnonzero(reg)
    rng = np.random.default_rng(0)
    keep = (cls[idx] >= 0) | (rng.random(len(idx)) < 0.5)
    idx = idx[keep]
    pts = dict(rt=np.round(rt[scan_of[idx]], 4).tolist(), mz=np.round(mz[idx], 4).tolist(),
               li=np.round(np.log10(inten[idx]), 3).tolist(), c=cls[idx].astype(int).tolist(),
               span=np.minimum(span2[idx], 99).astype(int).tolist())
    hist = {k: np.round(np.log10(inten[reg & (cls == code)]), 3).tolist() for k, code in
            (("isolated", 0), ("short", 1), ("ripple", 2), ("sideband", 3))}
    surf = dict(band_edges=band_edges.tolist(), blk_edges=blk_edges.tolist(), now=np.nan_to_num(med).tolist(),
                isolated=np.nan_to_num(med_iso).tolist(), clean=np.nan_to_num(med_clean).tolist(), frac_ripple=np.nan_to_num(frac_ripple, nan=-1).tolist())
    out = dict(stem=stem, region=dict(rt=[rt_lo, rt_hi], mz=[mz_lo_r, mz_hi_r]), floor=P.floor,
               n_region=int(reg.sum()), n_shown=int(len(idx)), trace_thin=0.5, points=pts, hist=hist, surf=surf,
               cells=C.to_dict(orient="records"), peaks=peaks,
               summary=dict(n=int(len(mz)), noise=int(noise_pt.sum()),
                            frac=[float(np.mean(cls[noise_pt] == k)) for k in (0, 1, 2, 3)],
                            med=[float(np.median(inten[cls == k])) for k in (0, 1, 2, 3)],
                            region_frac=[float(np.mean(cls[reg & noise_pt] == k)) for k in (0, 1, 2, 3)]))
    (ROOT / "results/diag_excess/noise_region.json").write_text(json.dumps(out, separators=(",", ":")))
    print("points for the page:", len(idx))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 1, *(float(v) for v in a[1:5]))
