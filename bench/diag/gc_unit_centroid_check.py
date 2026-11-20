"""Duplicates and cut peaks in a Mosa GC-mode run of .cdf files, counted against the raw data.

For every file of a `peak3d process` output: its accepted features against the chromatogram of the whole nominal
mass (every centroid with n - 0.3 <= m/z < n + 0.7 summed per scan, the unit-mass rule of peak3d/cdf.py):
  twin     another accepted feature of the same nominal mass whose apex is within half the file's median peak width
           (twin_10mDa: and within 10 mDa);
  cut      the feature ends in the middle of a flank: one scan beyond its first or last scan the nominal-mass
           chromatogram is still above half the feature's height, and within the next 1.5 median peak widths it
           falls below half of that (a feature riding on a steady background of its mass is not counted: the
           chromatogram stays up);
  hole     the feature ends at a hole of the raw file: beyond its first or last scan the nominal mass has no
           centroid at all for up to 3 median peak widths, and then is back at half the feature's height or more
           (a chromatographic signal cannot do that: the file lacks those points);
  peak     the feature is a peak of the nominal-mass chromatogram: its apex stands above the lowest point within
           1.5 median peak widths on each side by at least 5 x the chromatogram's point-to-point noise there.
Usage: python bench/diag/gc_unit_centroid_check.py OUT_DIR CDF_DIR [--files N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from peak3d import cdf  # noqa: E402


def nominal_matrix(path):
    rt, off, mz, it = cdf.read_cdf(path)
    masses, X = cdf.unit_matrix(rt, off, mz, it)
    return rt, masses, X


def check_file(out, cdf_dir, stem):
    f = pd.read_csv(out / "features" / f"{stem}.tsv", sep="\t")
    pj = json.loads((out / "features" / f"{stem}.params.json").read_text())
    scale = float(pj.get("mass_scale") or 1.0)
    rt, masses, X = nominal_matrix(cdf_dir / f"{stem}.cdf")
    f = f.sort_values(["mz", "rt_apex"]).reset_index(drop=True)
    nom = np.floor(f.mz.to_numpy() / scale + cdf.BIN_SHIFT).astype(int)
    fw = float(np.nanmedian(f.fwhm))
    apex, mz, h = f.rt_apex.to_numpy(), f.mz.to_numpy(), f.height.to_numpy()
    twin = np.zeros(len(f), bool)
    twin10 = np.zeros(len(f), bool)
    for n in np.unique(nom):
        i = np.flatnonzero(nom == n)
        if len(i) < 2:
            continue
        d = np.abs(apex[i][:, None] - apex[i][None, :]) <= 0.5 * fw
        np.fill_diagonal(d, False)
        twin[i] = d.any(1)
        twin10[i] = (d & (np.abs(mz[i][:, None] - mz[i][None, :]) <= 0.010 * scale)).any(1)
    col = nom - masses[0]
    s0 = np.clip(np.searchsorted(rt, f.rt_min.to_numpy() * 60.0 - 1e-6) - 1, 0, len(rt) - 1)
    s1 = np.clip(np.searchsorted(rt, f.rt_max.to_numpy() * 60.0 + 1e-6), 0, len(rt) - 1)
    ok = (col >= 0) & (col < X.shape[1])
    before = np.where(ok, X[s0, np.clip(col, 0, X.shape[1] - 1)], 0.0)
    after = np.where(ok, X[s1, np.clip(col, 0, X.shape[1] - 1)], 0.0)
    w = max(3, int(round(1.5 * fw * 60.0 / float(np.median(np.diff(rt))))))
    cc = np.clip(col, 0, X.shape[1] - 1)
    sa = np.clip(np.searchsorted(rt, apex * 60.0), 0, len(rt) - 1)
    cut = np.zeros(len(f), bool)
    peak = np.zeros(len(f), bool)
    hole = np.zeros(len(f), bool)
    first = np.searchsorted(rt, f.rt_min.to_numpy() * 60.0 - 1e-6)
    last = np.searchsorted(rt, f.rt_max.to_numpy() * 60.0 + 1e-6) - 1
    for i in np.flatnonzero(ok):
        y = X[:, cc[i]]
        lo = y[max(0, s0[i] - w):s0[i] + 1]
        hi = y[s1[i]:s1[i] + w + 1]
        cut[i] = (before[i] >= 0.5 * h[i] and lo.min() <= 0.5 * before[i]) or (after[i] >= 0.5 * h[i] and hi.min() <= 0.5 * after[i])
        a, b = max(0, sa[i] - w), min(len(y), sa[i] + w + 1)
        seg = y[a:b]
        top = seg[max(0, sa[i] - a - 1):sa[i] - a + 2].max()
        left, right = seg[:sa[i] - a + 1].min(), seg[sa[i] - a:].min()
        d2 = np.diff(seg, 2)
        noise = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0) if len(d2) > 4 else 0.0
        peak[i] = top - max(left, right) >= 5.0 * noise and top > max(left, right)
        for edge, step in ((first[i], -1), (last[i], 1)):
            k, n0 = edge + step, 0
            while 0 <= k < len(y) and y[k] == 0 and n0 <= 2 * w:
                k += step
                n0 += 1
            if 0 < n0 <= 2 * w and 0 <= k < len(y) and y[k] >= 0.5 * h[i]:
                hole[i] = True
    strong = h >= 10 * np.nanmedian(h)
    return dict(file=stem, mass_scale=scale, features=len(f), twin=int(twin.sum()), twin_10mDa=int(twin10.sum()), cut=int(cut.sum()),
                twin_share=round(float(twin.mean()), 3), cut_share=round(float(cut.mean()), 3),
                cut_share_strong=round(float(cut[strong].mean()), 3) if strong.any() else np.nan,
                peak=int(peak.sum()), peak_share=round(float(peak.mean()), 3), fwhm_feat_s=round(60 * fw, 2),
                hole=int(hole.sum()), hole_share=round(float(hole.mean()), 3), cut_or_hole=int((cut | hole).sum()),
                tol_max_ppm=round(pj.get("tol_max_ppm", np.nan), 2), sig_a=round(pj.get("sig_a", np.nan), 2),
                fwhm_s=round(60 * pj.get("fwhm_med", np.nan), 2), mz_scan_n=pj.get("mz_scan_n", 0)), f.assign(twin=twin, twin10=twin10, cut=cut, peak=peak, hole=hole, nom=nom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("cdf_dir"); ap.add_argument("--files", type=int, default=0)
    ap.add_argument("--examples", type=int, default=0)
    a = ap.parse_args()
    out, cdf_dir = Path(a.out), Path(a.cdf_dir)
    stems = sorted(p.name[:-len(".params.json")] for p in (out / "features").glob("*.params.json"))
    if a.files:
        stems = stems[:: max(1, len(stems) // a.files)][:a.files]
    rows = []
    for s in stems:
        r, f = check_file(out, cdf_dir, s)
        rows.append(r)
        print(r, flush=True)
        if a.examples:
            ex = f[f.cut & (f.height > 1e6)].sort_values("height", ascending=False).head(a.examples)
            print(ex[["mz", "rt_apex", "height", "rt_min", "rt_max", "n_scans", "twin", "flags"]].to_string())
    t = pd.DataFrame(rows)
    t.to_csv(out / "unit_centroid_check.tsv", sep="\t", index=False)
    print("ALL files %d  features %d (%.0f per file)  twin %d (%.3f)  twin_10mDa %d (%.3f)  cut %d (%.3f)  cut(strong) %.3f  peak %d (%.3f)" % (
        len(t), t.features.sum(), t.features.mean(), t.twin.sum(), t.twin.sum() / t.features.sum(), t.twin_10mDa.sum(),
        t.twin_10mDa.sum() / t.features.sum(), t.cut.sum(), t.cut.sum() / t.features.sum(), np.nanmean(t.cut_share_strong),
        t.peak.sum(), t.peak.sum() / t.features.sum()))
    print("    end at a hole of the raw file %d (%.3f)  cut or hole %d (%.3f)  median feature fwhm %.2f s" % (
        t.hole.sum(), t.hole.sum() / t.features.sum(), t.cut_or_hole.sum(), t.cut_or_hole.sum() / t.features.sum(), t.fwhm_feat_s.median()))
    a_ = out / "aligned_feature_table.tsv"
    if a_.exists():
        g = pd.read_csv(a_, sep="\t", usecols=["mz", "rt", "n_detected"])
        scale = float(t.mass_scale.iloc[0])
        g = g.sort_values("mz").reset_index(drop=True)
        mzv, rtv = g.mz.to_numpy() / scale, g.rt.to_numpy() * 60.0
        tol_rt = 0.5 * float(t.fwhm_feat_s.median())
        near10 = np.zeros(len(g), bool)
        near_nom = np.zeros(len(g), bool)
        j0 = 0
        for i in range(len(g)):             # rows with another row at nearly the same mass and retention time
            while mzv[i] - mzv[j0] > 0.5:
                j0 += 1
            j = j0
            while j < len(g) and mzv[j] - mzv[i] <= 0.5:
                if j != i and abs(rtv[j] - rtv[i]) <= tol_rt:
                    d = abs(mzv[j] - mzv[i])
                    near_nom[i] |= np.floor(mzv[j] + cdf.BIN_SHIFT) == np.floor(mzv[i] + cdf.BIN_SHIFT)
                    near10[i] |= d <= 0.010
                j += 1
        print("aligned table: rows %d; with another row within %.2f s at the same nominal mass %d (%.3f), within 10 mDa %d (%.3f)" % (
            len(g), tol_rt, near_nom.sum(), near_nom.mean(), near10.sum(), near10.mean()))
        print("aligned rows", len(g), " gc compounds", sum(1 for _ in open(out / "gc" / "compounds.tsv")) - 1 if (out / "gc" / "compounds.tsv").exists() else "-")


if __name__ == "__main__":
    main()
