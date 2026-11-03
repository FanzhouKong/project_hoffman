"""Record every intermediate of peak3d's per-file parameter estimation on real files, for the slide figures.

usage (compute node): envs/peak3d/bin/python presentation/param_estimation/extract.py
writes presentation/param_estimation/estimation_demo.npz, demo_params.json, all_file_params.tsv
Nothing in the benchmark is touched: the estimator is rerun read-only on the first HZV029_cert injection (and on the
file with the largest scan-gain jitter among the replicate runs, for the gain panel)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS  # noqa: E402
from peak3d import estimate as E  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402

OUT = Path(__file__).resolve().parent
RUN_KEYS = ["HZV029_cert", "YEAST_NEG", "SZ22_12C", "YEAST_12C", "IDSL003", "LI2018", "BM21_HILIC", "BM21_RP",
            "MC_Human_Serum_QTOF_RP_positive", "MC_NIST_Plasma_Orbitrap_RP_negative",
            "MC_Mouse_Feces_QTOF_HILIC_positive", "MC_NIST_Feces_Orbitrap_HILIC_negative",
            "MC_Fruit_Fly_QTOF_HILIC_positive", "MC_Human_Urine_QTOF_HILIC_negative",
            "MC_Mouse_Plasma_Orbitrap_HILIC_positive", "MC_NIST_Plasma_Orbitrap_RP_positive"]
KEEP = ["dt", "span", "floor", "fwhm_med", "fwhm_scans", "sig_a", "sig_b", "n_seeds", "n_fwhm_seeds", "n_sigma_points",
        "tol_max_ppm", "K", "min_scans", "n_smooth", "gain_sd", "n_gain_scans", "noise_c", "noise_r", "noise_s0",
        "n_noise_pts", "n_noise_bins", "n_scans", "n_centroids", "polarity", "n_features"]

# ---- 1. the measured parameters of every file of the current default run (sidecars written by the pipeline)
rows = []
for run in RUN_KEYS:
    d = ROOT / "results/peak3d_GL2" / run / "peak3d_out/features"
    if not d.is_dir():
        print("no sidecars for", run); continue
    for pj in sorted(d.glob("*.params.json")):
        p = json.loads(pj.read_text())
        rows.append(dict(run=run, stem=pj.name[:-len(".params.json")], **{k: p.get(k) for k in KEEP}))
with open(OUT / "all_file_params.tsv", "w") as f:
    cols = ["run", "stem"] + KEEP
    f.write("\t".join(cols) + "\n")
    for r in rows:
        f.write("\t".join(str(r[c]) for c in cols) + "\n")
print(len(rows), "files with sidecars")


def seed_intermediates(cl, n_seeds=2000):
    """everything estimate() computes, kept rather than reduced"""
    D = {}
    D["rt"] = cl.rt
    P = E.Params()
    P.dt = float(np.median(np.diff(cl.rt)))
    P.span = float(cl.rt[-1] - cl.rt[0])
    P.floor = float(np.percentile(cl.inten, 1))
    seeds = E.strongest_distinct(cl, n_seeds)
    scan_of = np.searchsorted(cl.off, seeds, side="right") - 1
    D["seed_mz"], D["seed_rt"], D["seed_inten"] = cl.mz[seeds], cl.rt[scan_of], cl.inten[seeds].astype(np.float64)
    widths, wscan = [], []
    reach = np.full(len(seeds), 8.0)
    fw = np.full(len(seeds), np.nan)
    for i, (p, s) in enumerate(zip(seeds, scan_of)):
        r = E.fwhm_at(cl, cl.mz[p], int(s))
        if r is not None:
            widths.append(r[0]); wscan.append(r[1]); fw[i] = r[0]
            reach[i] = max(8.0, 3.0 * r[0] / P.dt)
    D["seed_fwhm"] = fw
    P.n_seeds, P.n_fwhm_seeds = len(seeds), len(widths)
    P.fwhm_med = float(np.median(widths)); P.fwhm_scans = P.fwhm_med / P.dt
    reach = np.where(reach == 8.0, max(8.0, 3.0 * P.fwhm_scans), reach)
    # example seed for the FWHM panel: width near the median, strong, clean half-max crossings
    order = np.argsort(np.abs(fw - P.fwhm_med))
    ex = next(i for i in order if np.isfinite(fw[i]) and D["seed_inten"][i] > np.percentile(D["seed_inten"], 50))
    s = int(scan_of[ex]); S = cl.n_scans; win = max(60, S // 20)
    s0, s1 = max(0, s - win), min(S, s + win + 1)
    raw = cl.eic(cl.mz[seeds[ex]], E.SEED_PPM, s0, s1).astype(np.float64)
    D["ex_rt"], D["ex_raw"] = cl.rt[s0:s1], raw
    D["ex_smooth"] = np.convolve(raw, np.ones(3) / 3.0, mode="same")
    D["ex_mz"], D["ex_fwhm"] = cl.mz[seeds[ex]], fw[ex]
    # m/z error model
    I_all, dev_all = E.trace_deviations(cl, seeds, scan_of, reach)
    P.sig_a, P.sig_b, P.n_sigma_points = E.fit_sigma(cl, seeds, scan_of, reach, traces=(I_all, dev_all))
    I = np.concatenate(I_all); dev = np.concatenate(dev_all)
    D["mz_I"], D["mz_dev"] = I, dev
    lg = np.log10(I)
    edges = np.arange(np.floor(lg.min() * 2) / 2, lg.max() + 0.5, 0.5)
    b = np.clip(np.searchsorted(edges, lg, side="right") - 1, 0, len(edges) - 2)
    bx, bsd, bn = [], [], []
    for k in range(len(edges) - 1):
        sel = b == k
        if sel.sum() < 20:
            continue
        d = dev[sel]
        bx.append(float(np.median(I[sel]))); bsd.append(1.4826 * float(np.median(np.abs(d - np.median(d))))); bn.append(int(sel.sum()))
    D["mz_bin_I"], D["mz_bin_sd"], D["mz_bin_n"] = np.array(bx), np.array(bsd), np.array(bn)
    # one trace of the m/z model: a strong seed's contiguous run (its m/z per scan vs its intensity)
    for cand in np.argsort(-D["seed_inten"])[:50]:
        p, sc = seeds[cand], int(scan_of[cand]); rc = int(reach[cand])
        a0, a1 = max(0, sc - rc), min(S, sc + rc + 1)
        idx = cl.eic_argmax(cl.mz[p], E.SEED_PPM, a0, a1)
        k = sc - a0; lo = k; hi = k
        while lo - 1 >= 0 and idx[lo - 1] >= 0: lo -= 1
        while hi + 1 < len(idx) and idx[hi + 1] >= 0: hi += 1
        idx = idx[lo:hi + 1]
        if len(idx) >= 12:
            mzs, ins = cl.mz[idx], cl.inten[idx].astype(np.float64)
            mw = np.average(mzs, weights=ins)
            D["tr_rt"], D["tr_mz_ppm"], D["tr_I"], D["tr_mz0"] = cl.rt[a0 + lo:a0 + hi + 1], (mzs - mw) / mw * 1e6, ins, mw
            break
    # noise law: the same loop as fit_intensity_noise, keeping the residuals and one example trace
    sg = np.array([-3.0, 12.0, 17.0, 12.0, -3.0]) / 35.0
    corr = 1.0 / np.sqrt(1.0 - 17.0 / 35.0)
    lev_all, res_all = [], []
    ex_noise = None
    for p, sc, rc in zip(seeds, scan_of, reach):
        a0, a1 = max(0, int(sc) - int(rc)), min(S, int(sc) + int(rc) + 1)
        e = cl.eic(cl.mz[p], E.SEED_PPM, a0, a1).astype(np.float64)
        if len(e) < 7:
            continue
        k = int(np.argmax(e)); thr = max(0.1 * e[k], 1e-9)
        lo = k; hi = k
        while lo - 1 >= 0 and e[lo - 1] >= thr: lo -= 1
        while hi + 1 < len(e) and e[hi + 1] >= thr: hi += 1
        n = hi - lo + 1
        if n < 7:
            continue
        y = e[lo:hi + 1]
        fit = np.convolve(y, sg[::-1], mode="valid")
        j = np.arange(2, n - 2)
        keep = (j != k - lo) & (fit > 0)
        if not keep.any():
            continue
        res_all.append((y[j][keep] - fit[keep]) * corr); lev_all.append(fit[keep])
        if ex_noise is None and 14 <= n <= 30 and np.percentile(D["seed_inten"], 60) < e[k] < np.percentile(D["seed_inten"], 80):
            ex_noise = dict(nz_rt=cl.rt[a0 + lo:a0 + hi + 1], nz_y=y, nz_fit_idx=j, nz_fit=fit, nz_apex=k - lo, nz_mz=cl.mz[p])
    lev, res = np.concatenate(lev_all), np.concatenate(res_all)
    s0_, c_, r_, n_pts, n_bins = E.fit_noise_law(lev, res)
    P.noise_s0, P.noise_c, P.noise_r, P.n_noise_pts, P.n_noise_bins = s0_, c_, r_, n_pts, n_bins
    D["nl_lev"], D["nl_res"] = lev, res
    lg = np.log10(lev); edges = np.arange(np.floor(lg.min() * 2) / 2, lg.max() + 0.5, 0.5)
    b = np.clip(np.searchsorted(edges, lg, side="right") - 1, 0, len(edges) - 2)
    bx, bsd, bn = [], [], []
    for kb in range(len(edges) - 1):
        sel = b == kb
        if sel.sum() < 30:
            continue
        d = res[sel]; sd = 1.4826 * np.median(np.abs(d - np.median(d)))
        if sd > 0:
            bx.append(float(np.median(lev[sel]))); bsd.append(float(sd)); bn.append(int(sel.sum()))
    D["nl_bin_I"], D["nl_bin_sd"], D["nl_bin_n"] = np.array(bx), np.array(bsd), np.array(bn)
    D.update(ex_noise or {})
    # derived settings exactly as estimate() derives them
    P.tol_max_ppm = float(np.clip(4 * P.sig_ppm(P.floor), 5.0, 30.0))
    P.K = int(np.clip(round(0.5 * P.fwhm_scans), 1, 3))
    P.min_scans = int(np.clip(round(0.5 * P.fwhm_scans), 3, 5))
    sm_sigma = P.smooth_fwhm * P.fwhm_scans / 2.3548
    P.n_smooth = int(np.clip(round(2.0 * sm_sigma ** 2), 1, 50))
    # cross-check against the library's own estimate()
    Q = E.estimate(cl)
    for k in ["dt", "floor", "fwhm_med", "sig_a", "sig_b", "noise_c", "noise_r", "tol_max_ppm", "K", "min_scans", "n_smooth"]:
        assert np.isclose(getattr(P, k), getattr(Q, k), rtol=1e-9, atol=1e-12), (k, getattr(P, k), getattr(Q, k))
    return P, D, seeds, scan_of


def gain_intermediates(cl, P, seeds, scan_of):
    g, g_sd, g_n = E.scan_gains(cl, P)
    D = dict(g_rt=cl.rt, gains=g, gain_sd=g_sd, n_gain_scans=g_n)
    # a strong trace before and after the correction
    for cand in np.argsort(-cl.inten[seeds])[:200]:
        p, sc = seeds[cand], int(scan_of[cand]); rc = int(max(8, 3 * P.fwhm_scans))
        a0, a1 = max(0, sc - rc), min(cl.n_scans, sc + rc + 1)
        e = cl.eic(cl.mz[p], E.SEED_PPM, a0, a1).astype(np.float64)
        if np.count_nonzero(e > 0.1 * e.max()) >= 10:
            D.update(g_ex_rt=cl.rt[a0:a1], g_ex_raw=e, g_ex_corr=e / g[a0:a1], g_ex_mz=cl.mz[p])
            break
    return D


main_path = RUNS["HZV029_cert"]["files"][0]
cl = load_cloud(main_path)
P, D, seeds, scan_of = seed_intermediates(cl)
D.update(gain_intermediates(cl, P, seeds, scan_of))
meta = dict(main_file=str(main_path), main_params=P.to_dict(), n_scans=cl.n_scans, n_centroids=cl.n_points)
print("main file:", main_path, "gain_sd", D["gain_sd"], "n_gain_scans", D["n_gain_scans"])

# the gain panel wants a file with visible gain jitter: pick the largest gain_sd among the four replicate runs
best = max((r for r in rows if r["run"] in ["HZV029_cert", "YEAST_NEG", "SZ22_12C", "YEAST_12C"]), key=lambda r: float(r["gain_sd"] or 0))
print("largest gain_sd:", best["run"], best["stem"], best["gain_sd"])
if float(best["gain_sd"] or 0) > 0 and best["stem"] != Path(main_path).stem:
    gpath = next(f for f in RUNS[best["run"]]["files"] if Path(f).stem == best["stem"])
    cl2 = load_cloud(gpath)
    P2 = E.estimate(cl2)
    seeds2, scan2 = E.seed_ions(cl2)
    G2 = gain_intermediates(cl2, P2, seeds2, scan2)
    D.update({"alt_" + k: v for k, v in G2.items()})
    meta.update(gain_file=str(gpath), gain_run=best["run"], gain_params=P2.to_dict())
    print("gain file:", gpath, "gain_sd", G2["gain_sd"])

np.savez_compressed(OUT / "estimation_demo.npz", **{k: np.asarray(v) for k, v in D.items()})
(OUT / "demo_params.json").write_text(json.dumps(meta, indent=1, default=float))
print("params:", {k: meta["main_params"][k] for k in ["dt", "fwhm_med", "fwhm_scans", "sig_a", "sig_b", "tol_max_ppm", "K", "min_scans", "n_smooth", "noise_c", "noise_r", "floor"]})
