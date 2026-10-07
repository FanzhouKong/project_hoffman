# diagnostic: holistic audit of the 20,000 IDSL.IPA labels (IDSL003) against the raw data, independent of every
# picker (no peak3d, no tool output). Per label, in the labelled file (Zenodo 003.mzML):
#   peak evidence inside the scoring window (10 ppm, +/-0.1 min): apex height, peak-to-background ratio,
#   points, Gaussian r2, prominence, FWHM, "flank" (window max rises on into a peak outside the window);
#   m/z placement (centroid offset at 10 ppm, dominant ion within 30 ppm), isotopes (M+1 support, label is an M+1),
#   label-label neighbours within the scoring tolerance;
# and recurrence in N neighbouring injections of the same study (MTBLS1684 cord blood, our msconvert mzML) after a
# per-file RT drift fit on strong label positions, plus two RT-shifted decoy windows per label (chance rate).
# usage: idsl_label_audit.py OUT_DIR [N_REPLICATES=20]   (reads 20+ mzML -> run in srun)
#        REP_FILES=004,123,... picks the replicate injections explicitly
import sys, json, time, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.msdata import load_run, C13

LAB = ROOT / "data/raw/IDSL_IPA/idslipa_benchmarking_dataset.csv"
F003 = ROOT / "data/raw/IDSL_IPA/003.mzML"
REPDIR = ROOT / "data/mzml/MTBLS1684"
PPM, RT_TOL = 10.0, 0.1        # scoring tolerance (bench/score.py default)
BG_HALF, BG_EXCL = 1.5, 15     # background: +/-1.5 min around the label, minus +/-15 scans around the apex
PROM_W, GAUSS_W = 13, 8        # scans (~3 median FWHM; Gaussian fit half-window)
DECOY_SHIFT = 0.6              # min, RT-shifted decoy windows at the same m/z
SIGMAS = np.array([0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 7.0])
MUS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


def eic_win(run, mz, ppm, s0, s1):
    """max intensity per scan in [s0, s1) within +/- ppm of mz"""
    tol = mz * ppm * 1e-6
    a, b = np.searchsorted(run.mz, [mz - tol, mz + tol])
    out = np.zeros(s1 - s0)
    if b > a:
        sc = run.scan[a:b]
        m = (sc >= s0) & (sc < s1)
        np.maximum.at(out, sc[m] - s0, run.inten[a:b][m])
    return out


def ions_win(run, mz, ppm, s0, s1):
    """centroids in scans [s0, s1) within +/- ppm: (scan - s0, mz, intensity)"""
    tol = mz * ppm * 1e-6
    a, b = np.searchsorted(run.mz, [mz - tol, mz + tol])
    sc = run.scan[a:b]
    m = (sc >= s0) & (sc < s1)
    return sc[m] - s0, run.mz[a:b][m], run.inten[a:b][m]


def smooth(e):
    return np.convolve(np.pad(e, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")


def gauss_r2(y, kc):
    """best r2 of y ~ a * gaussian + b over a small (centre, sigma) grid, a > 0"""
    if len(y) < 5 or y.std() == 0:
        return 0.0, np.nan
    x = np.arange(len(y), dtype=float)[:, None, None]
    g = np.exp(-0.5 * ((x - (kc + MUS)[None, :, None]) / SIGMAS[None, None, :]) ** 2)
    gc = g - g.mean(0)
    yc = (y - y.mean())[:, None, None]
    num = (gc * yc).sum(0)
    den = np.sqrt((gc ** 2).sum(0) * (yc ** 2).sum())
    r = np.where(den > 0, num / np.where(den > 0, den, 1), 0)
    r2 = np.where(r > 0, r ** 2, 0)
    i = np.unravel_index(np.argmax(r2), r2.shape)
    return float(r2[i]), float(SIGMAS[i[1]])


def peak_at(e, w0, w1, floor, full=True):
    """evidence for a chromatographic peak with its apex inside e[w0:w1] (raw EIC segment)"""
    w0, w1 = max(0, w0), min(len(e), w1)
    if w1 <= w0 or e[w0:w1].max() <= 0:
        return dict(h=0.0, k=-1, flank=False, pbr=0.0, n_pts=0, r2=0.0, prom=0.0, fwhm=0, bg_nz=np.nan,
                    bg_q90=np.nan, climb_ratio=np.nan, k_climb=-1, sig=np.nan)
    es = smooth(e)
    k = w0 + int(np.argmax(es[w0:w1]))
    kc = k
    while kc + 1 < len(es) and es[kc + 1] > es[kc]:
        kc += 1
    while kc - 1 >= 0 and es[kc - 1] > es[kc]:
        kc -= 1
    climb_ratio = es[kc] / es[k] if es[k] > 0 else np.nan
    flank = bool(kc != k and climb_ratio > 1.2)
    h = float(e[max(0, k - 1):k + 2].max())
    hs = es[k]
    # contiguous signal, single empty scans tolerated (TOF dropouts)
    lo = k
    while lo - 1 >= 0 and (e[lo - 1] > 0 or (lo - 2 >= 0 and e[lo - 2] > 0)):
        lo -= 1
    hi = k
    while hi + 1 < len(e) and (e[hi + 1] > 0 or (hi + 2 < len(e) and e[hi + 2] > 0)):
        hi += 1
    n_pts = int((e[lo:hi + 1] > 0).sum())
    bgm = np.ones(len(e), bool)
    bgm[max(0, k - BG_EXCL):k + BG_EXCL + 1] = False
    bg = e[bgm]
    bg_q90 = float(np.quantile(bg, 0.9)) if len(bg) >= 20 else np.nan
    bg_nz = float((bg > 0).mean()) if len(bg) >= 20 else np.nan
    pbr = h / max(bg_q90 if np.isfinite(bg_q90) else 0.0, floor)
    lmin = es[max(0, k - PROM_W):k].min() if k > 0 else 0.0
    rmin = es[k + 1:k + PROM_W + 1].min() if k + 1 < len(es) else 0.0
    prom = float((hs - max(lmin, rmin)) / hs) if hs > 0 else 0.0
    a, b = max(0, k - GAUSS_W), min(len(e), k + GAUSS_W + 1)
    r2, sig = gauss_r2(e[a:b], k - a)
    out = dict(h=h, k=k, flank=flank, pbr=float(pbr), n_pts=n_pts, r2=r2, prom=prom, bg_nz=bg_nz, bg_q90=bg_q90,
               climb_ratio=float(climb_ratio), k_climb=kc, sig=sig)
    if full:
        l, r = k, k
        while l - 1 >= 0 and es[l - 1] >= hs / 2:
            l -= 1
        while r + 1 < len(es) and es[r + 1] >= hs / 2:
            r += 1
        out["fwhm"] = r - l + 1
    return out


def run_floor(run):
    return float(np.quantile(run.inten, 0.05))


# ----------------------------------------------------------------------------------------------- labelled file
def audit_003(run, mz, rt):
    floor = run_floor(run)
    rows = []
    for i, (m0, t0) in enumerate(zip(mz, rt)):
        s0, s1 = run.scan_window(t0, BG_HALF)
        s0, s1 = max(0, s0), min(run.n_scans, s1)
        w0, w1 = run.scan_window(t0, RT_TOL)
        w0, w1 = w0 - s0, w1 - s0
        e = eic_win(run, m0, PPM, s0, s1)
        p = peak_at(e, w0, w1, floor)
        d = dict(h=p["h"], pbr=p["pbr"], n_pts=p["n_pts"], r2=p["r2"], prom=p["prom"], fwhm=p.get("fwhm", 0),
                 sig=p["sig"], bg_nz=p["bg_nz"], flank=p["flank"], climb_ratio=p["climb_ratio"],
                 drt_s=np.nan, drt_climb_s=np.nan, d_ppm=np.nan, mz_sd_ppm=np.nan, d_ppm_dom=np.nan,
                 dom_ratio=np.nan, corr_p1=np.nan, r_p1=np.nan, corr_m1=np.nan, r_m1=np.nan, apex_rt=np.nan)
        k = p["k"]
        if k >= 0:
            d["apex_rt"] = float(run.rt[s0 + k])
            d["drt_s"] = (run.rt[s0 + k] - t0) * 60
            d["drt_climb_s"] = (run.rt[s0 + p["k_climb"]] - t0) * 60
            # m/z placement: centroids at 10 ppm over apex +/- 2 scans; dominant ion within 30 ppm in the window
            sc, mm, ii = ions_win(run, m0, PPM, s0 + k - 2, s0 + k + 3)
            if len(ii):
                d["d_ppm"] = float(((mm - m0) / m0 * 1e6 * ii).sum() / ii.sum())
            sc, mm, ii = ions_win(run, m0, PPM, s0 + max(k - 3, 0), s0 + k + 4)
            if len(ii) >= 3:
                d["mz_sd_ppm"] = float(np.std((mm - m0) / m0 * 1e6))
            sc, mm, ii = ions_win(run, m0, 30.0, s0 + w0, s0 + w1)
            if len(ii):
                j = int(np.argmax(ii))
                d["d_ppm_dom"] = float((mm[j] - m0) / m0 * 1e6)
                d["dom_ratio"] = float(ii[j] / max(p["h"], 1.0))
            # isotopes over apex +/- 6 scans
            a, b = max(0, k - 6), min(len(e), k + 7)
            y = e[a:b]
            for tag, mm2 in (("p1", m0 + C13), ("m1", m0 - C13)):
                z = eic_win(run, mm2, PPM, s0 + a, s0 + b)
                if y.std() > 0 and z.std() > 0:
                    d[f"corr_{tag}"] = float(np.corrcoef(y, z)[0, 1])
                zk = z[max(0, k - a - 2):k - a + 3].max() if len(z) else 0.0
                d[f"r_{tag}"] = float(zk / p["h"]) if p["h"] > 0 else np.nan
        rows.append(d)
    return pd.DataFrame(rows), floor


def neighbours(mz, rt, is_tp, ppm, rt_tol):
    """for each label: number of other TP / TN labels within ppm and rt_tol"""
    o = np.argsort(mz)
    ms, ts, tps = mz[o], rt[o], is_tp[o]
    n_tp = np.zeros(len(mz), int)
    n_tn = np.zeros(len(mz), int)
    for j in range(len(ms)):
        lo, hi = np.searchsorted(ms, [ms[j] * (1 - ppm * 1e-6), ms[j] * (1 + ppm * 1e-6)])
        idx = np.arange(lo, hi)
        idx = idx[(idx != j) & (np.abs(ts[idx] - ts[j]) <= rt_tol)]
        n_tp[o[j]] = int(tps[idx].sum())
        n_tn[o[j]] = int((~tps[idx]).sum())
    return n_tp, n_tn


# ----------------------------------------------------------------------------------------------- replicates
G = {}


def _init(mz, rt, anchors):
    G.update(mz=mz, rt=rt, anchors=anchors)


def drift_fit(run, anchors, floor):
    """per-file RT shift (min) vs 003 RT from strong, isolated label positions"""
    xs, ys = [], []
    for m0, t_apex in anchors:
        s0, s1 = run.scan_window(t_apex, 0.6)
        s0, s1 = max(0, s0), min(run.n_scans, s1)
        e = eic_win(run, m0, PPM, s0, s1)
        w0, w1 = run.scan_window(t_apex, 0.4)
        p = peak_at(e, w0 - s0, w1 - s0, floor, full=False)
        if p["k"] < 0 or p["flank"] or p["pbr"] < 10 or p["r2"] < 0.8:
            continue
        k = p["k"]
        es = smooth(e)
        dk = 0.0
        if 0 < k < len(es) - 1:  # parabolic apex
            den = es[k - 1] - 2 * es[k] + es[k + 1]
            dk = 0.5 * (es[k - 1] - es[k + 1]) / den if den < 0 else 0.0
        t = np.interp(s0 + k + dk, np.arange(run.n_scans), run.rt)
        xs.append(t_apex); ys.append(t - t_apex)
    xs, ys = np.array(xs), np.array(ys)
    edges = np.arange(0, 14.01, 0.5)
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (xs >= a) & (xs < b)
        if m.sum() >= 5:
            cx.append(xs[m].mean()); cy.append(np.median(ys[m]))
    if len(cx) < 2:
        cx, cy = [0.0, 14.0], [float(np.median(ys)) if len(ys) else 0.0] * 2
    c0 = np.array(cy, float)
    cy = c0.copy()
    for i in range(len(c0)) if len(c0) >= 3 else ():  # a lone bin > 2 s off two agreeing neighbours = mismatched anchors
        nb = [c0[j] for j in (i - 1, i + 1) if 0 <= j < len(c0)]
        if len(nb) == 1:  # end knot: its two inner neighbours
            nb = [c0[1], c0[2]] if i == 0 else [c0[-2], c0[-3]]
        if all(abs(c0[i] - v) > 2.0 / 60 for v in nb) and abs(nb[0] - nb[1]) <= 2.0 / 60:
            cy[i] = float(np.median(nb))
    resid = ys - np.interp(xs, cx, cy) if len(xs) else np.array([np.nan])
    return np.array(cx), np.array(cy), dict(n_anchor=len(xs), shift_med_s=float(np.median(ys) * 60) if len(ys) else np.nan,
                                            shift_iqr_s=float(np.subtract(*np.quantile(ys, [0.75, 0.25])) * 60) if len(ys) else np.nan,
                                            resid_mad_s=float(np.median(np.abs(resid)) * 60))


def replicate_worker(path):
    t0 = time.time()
    run = load_run(path)
    floor = run_floor(run)
    cx, cy, info = drift_fit(run, G["anchors"], floor)
    mz, rt = G["mz"], G["rt"]
    n = len(mz)
    keys = ("h", "pbr", "n_pts", "r2", "prom", "flank", "drt_s")
    out = {f"{w}_{k}": np.zeros(n, float) for w in ("tgt", "dlo", "dhi") for k in keys}
    span = (run.rt.min(), run.rt.max())
    for i in range(n):
        tc = rt[i] + float(np.interp(rt[i], cx, cy))
        for w, t in (("tgt", tc), ("dlo", tc - DECOY_SHIFT), ("dhi", tc + DECOY_SHIFT)):
            if t - RT_TOL < span[0] or t + RT_TOL > span[1]:
                for k in keys:
                    out[f"{w}_{k}"][i] = np.nan
                continue
            s0, s1 = run.scan_window(t, BG_HALF)
            s0, s1 = max(0, s0), min(run.n_scans, s1)
            w0, w1 = run.scan_window(t, RT_TOL)
            e = eic_win(run, mz[i], PPM, s0, s1)
            p = peak_at(e, w0 - s0, w1 - s0, floor, full=False)
            for k in keys[:-1]:
                out[f"{w}_{k}"][i] = p[k] if k in p else np.nan
            out[f"{w}_drt_s"][i] = (run.rt[s0 + p["k"]] - t) * 60 if p["k"] >= 0 else np.nan
    info.update(file=Path(path).stem, floor=floor, n_scans=run.n_scans, secs=round(time.time() - t0, 1),
                drift_knots_min=[round(float(a), 3) for a in cx], drift_knots_s=[round(float(b) * 60, 2) for b in cy])
    return info, {k: v.astype(np.float32) for k, v in out.items()}


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    n_rep = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    t0 = time.time()
    lab = pd.read_csv(LAB, encoding="utf-8-sig").rename(columns={"Unnamed: 0": "id"})
    if os.environ.get("AUDIT_N"):  # quick test on a subset
        lab = lab.sample(int(os.environ["AUDIT_N"]), random_state=0).reset_index(drop=True)
    mz, rt = lab["m/z"].values.astype(float), lab["RT(min)"].values.astype(float)
    is_tp = (lab["Manual Curation"] == "TP").values
    run = load_run(F003)
    print(f"003 (Zenodo): {run.n_scans} MS1 scans, RT {run.rt.min():.3f}-{run.rt.max():.3f} min, "
          f"dt {np.median(np.diff(run.rt)) * 60:.3f} s, {len(run.mz)} centroids, "
          f"intensity q01/q05/q50 {np.quantile(run.inten, [0.01, 0.05, 0.5]).round(1).tolist()}  ({time.time() - t0:.0f} s)", flush=True)
    A, floor = audit_003(run, mz, rt)
    print(f"audit 003 done, floor {floor:.1f} ({time.time() - t0:.0f} s)", flush=True)
    for (pp, tt) in ((10, 0.1), (20, 0.2)):
        a, b = neighbours(mz, rt, is_tp, pp, tt)
        A[f"nb_tp_{pp}"], A[f"nb_tn_{pp}"] = a, b
    # anchors chosen from raw evidence only (never from the TP/TN column)
    strong = ((A.h >= 2e4) & (A.pbr >= 20) & (A.r2 >= 0.9) & (A.prom >= 0.8) & ~A.flank).values
    cand = np.where(strong)[0]
    keep = []
    for j in cand:
        close = cand[(np.abs(mz[cand] - mz[j]) / mz[j] * 1e6 <= 20) & (np.abs(rt[cand] - rt[j]) <= 0.3)]
        if len(close) == 1:
            keep.append(j)
    anchors = [(float(mz[j]), float(A.apex_rt.values[j])) for j in keep]
    print(f"drift anchors: {len(anchors)}", flush=True)
    files = sorted(p for p in REPDIR.glob("*.mzML") if p.stem != "003")[:n_rep]
    if os.environ.get("REP_FILES"):  # explicit comma-separated stems, e.g. a random draw
        files = [REPDIR / f"{s}.mzML" for s in os.environ["REP_FILES"].split(",")]
    files = [REPDIR / "003.mzML"] + files   # our own conversion of 003 = conversion-consistency check
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    with Pool(min(ncpu, len(files)), initializer=_init, initargs=(mz, rt, anchors)) as pool:
        res = pool.map(replicate_worker, [str(f) for f in files], chunksize=1)
    infos = [r[0] for r in res]
    for inf in infos:
        print(json.dumps(inf), flush=True)
    names = [inf["file"] for inf in infos]
    np.savez_compressed(out / "replicates.npz", files=np.array(names),
                        **{k: np.stack([r[1][k] for r in res]) for k in res[0][1]})
    json.dump(infos, open(out / "replicates.json", "w"), indent=1)
    A = pd.concat([lab, A], axis=1)
    A.to_csv(out / "labels_003.tsv", sep="\t", index=False)
    print(f"done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
