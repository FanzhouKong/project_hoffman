# diagnostic: classify every kept peak3d feature of a run by an independent 1D check of its raw EIC
# (smoothed at half the file's median peak width), then report how many features and aligned groups
# each class holds, how reproducible / tool-supported they are, and which truth features they hold.
#   edge      side window truncated by the run start or end (the ridge gate sees an empty side) AND
#             the available side holds the ion at > half the apex (90th percentile, as the gate)
#   trunc     the smoothed maximum sits on the first / last scan of the run: a peak cut by the run edge
#   noapex    the smoothed EIC (kernel FWHM = half the median FWHM) has no local maximum inside the
#             feature's bounds: a slice of a tail, ramp, step or of a neighbouring peak (split fragment)
#   dup       the smoothed maximum it sits on is claimed by a stronger feature of the same ion
#   lowsnr    smoothed prominence < 3x the trace's own residual noise (zig-zag) or the cell noise
#             (spike from an empty baseline)
#   ok        everything else
# usage: junk_classes.py RUN [MAX_FILES=6]
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402

K_SNR = 3.0
HALF_WIN = 10.0     # EIC window, median FWHMs each side
SMOOTH = 0.5        # smoothing kernel FWHM / file median FWHM: kills 1-3 scan spikes, keeps doublets >= 1 FWHM apart


def classify_file(cloud, F, P):
    S = cloud.n_scans
    fs = P["fwhm_scans"]
    sig_k = SMOOTH * fs / 2.355
    n = len(F)
    out = {k: np.zeros(n) for k in ("edge", "noapex", "trunc", "snr_s", "prom_s_rel", "noise_res", "k_abs", "top_s")}
    wf = np.maximum(F["fwhm"].values, P["fwhm_med"]) / P["dt"]
    far = np.maximum(np.round(2.0 * wf) + 2, np.round(6.0 * wf)).astype(int)
    near = np.maximum(1, np.round(2.0 * wf)).astype(int)
    for i in range(n):
        f = F.iloc[i]
        sa = int(f["scan_apex"])
        W = max(int(np.ceil(HALF_WIN * fs)), far[i] + 1)
        s0, s1 = max(0, sa - W), min(S, sa + W + 1)
        ppm = max(5.0, 3.0 * np.sqrt(P["sig_a"] ** 2 + P["sig_b"] ** 2 / max(f["height"], 1.0)))
        e = cloud.eic(float(f["mz"]), ppm, s0, s1).astype(np.float64)
        es = gaussian_filter1d(e, sig_k, mode="nearest")
        # one-sided ridge level where the run start / end truncates a side window
        trunc_l, trunc_r = sa - far[i] < 0, sa + far[i] > S - 1
        if trunc_l or trunc_r:
            lv = []
            for lo_, hi_, tr_ in ((sa - far[i], sa - near[i] + 1, trunc_l), (sa + near[i], sa + far[i] + 1, trunc_r)):
                seg = e[max(0, lo_ - s0):max(0, min(s1, hi_) - s0)]
                if not tr_ and len(seg):
                    lv.append(np.sort(seg)[min(len(seg) - 1, int(0.9 * (len(seg) - 1) + 0.5))])
            out["edge"][i] = bool(lv) and max(lv) / max(f["height"], 1e-9) > 0.5
        # the smoothed EIC must have a local maximum inside the feature's own bounds
        b0 = max(0, int(f["scan_min"]) - s0)
        b1 = min(len(es) - 1, int(f["scan_max"]) - s0)
        k = b0 + int(np.argmax(es[b0:b1 + 1]))
        left_ok = k == 0 or es[k] >= es[k - 1]
        right_ok = k == len(es) - 1 or es[k] >= es[k + 1]
        out["noapex"][i] = not (left_ok and right_ok)
        out["trunc"][i] = (s0 + k <= 1) or (s0 + k >= S - 2)
        out["k_abs"][i] = s0 + k
        # topographic prominence of es[k] inside the window
        top = es[k]
        j = k
        lmin = top
        while j > 0 and es[j - 1] <= top:
            j -= 1
            lmin = min(lmin, es[j])
        j = k
        rmin = top
        while j < len(es) - 1 and es[j + 1] <= top:
            j += 1
            rmin = min(rmin, es[j])
        prom = top - max(lmin, rmin)
        out["top_s"][i] = top
        r = e - es
        mask = np.ones(len(e), bool)
        mask[max(0, int(f["scan_min"]) - s0):max(0, int(f["scan_max"]) - s0 + 1)] = False
        rr = r[mask] if mask.sum() >= 10 else r
        noise_res = 1.4826 * np.median(np.abs(rr - np.median(rr)))
        out["noise_res"][i] = noise_res
        out["snr_s"][i] = prom / max(noise_res, f["noise"])
        out["prom_s_rel"][i] = prom / top if top > 0 else 0.0
    C = pd.DataFrame(out)
    # dup: another, stronger feature of the same ion on the same smoothed maximum
    mz, ht, ka = F["mz"].values, F["height"].values, C["k_abs"].values
    o = np.argsort(mz)
    smz = mz[o]
    lo5 = np.searchsorted(smz, mz * (1 - 5e-6))
    hi5 = np.searchsorted(smz, mz * (1 + 5e-6), side="right")
    dup = np.zeros(n, bool)
    for i in range(n):
        kk = o[lo5[i]:hi5[i]]
        dup[i] = np.any((kk != i) & (np.abs(ka[kk] - ka[i]) <= max(1.0, 0.25 * fs)) & (ht[kk] > ht[i]))
    C["dup"] = dup
    cls = np.full(n, "ok", dtype=object)
    cls[C["snr_s"].values < K_SNR] = "lowsnr"
    cls[dup] = "dup"
    cls[C["noapex"].values > 0] = "noapex"
    cls[(C["trunc"].values > 0) & (C["noapex"].values > 0)] = "trunc"
    cls[C["edge"].values > 0] = "edge"
    C["cls"] = cls
    return C


def rep_check(F, P, others):
    """fraction of the other files whose raw EIC holds a smoothed peak (local maximum within one peak
    width of the feature's corrected RT, prominence >= K_SNR x noise, top >= 5 % of this feature's
    smoothed top) at the feature's m/z: is the feature reproducible in the replicates?"""
    n = len(F)
    hits = np.zeros(n)
    if not others:
        return np.full(n, np.nan)
    for cloudB, warpB, PB in others:
        S = cloudB.n_scans
        fs = PB["fwhm_scans"]
        sig_k = SMOOTH * fs / 2.355
        rtB = np.interp(F["rt_corr"].values, warpB["rt_corr"].values, warpB["rt_native"].values)
        sB = np.clip(np.searchsorted(cloudB.rt, rtB), 0, S - 1)
        hw = np.maximum(1, np.round(np.maximum(F["fwhm"].values, P["fwhm_med"]) / PB["dt"])).astype(int)
        W = int(np.ceil(HALF_WIN * fs))
        for i in range(n):
            f_mz, h = float(F["mz"].values[i]), max(float(F["height"].values[i]), 1.0)
            s0, s1 = max(0, sB[i] - W - hw[i]), min(S, sB[i] + W + hw[i] + 1)
            ppm = max(5.0, 3.0 * np.sqrt(PB["sig_a"] ** 2 + PB["sig_b"] ** 2 / h))
            e = cloudB.eic(f_mz, ppm, s0, s1).astype(np.float64)
            if e.max() <= 0:
                continue
            es = gaussian_filter1d(e, sig_k, mode="nearest")
            c = sB[i] - s0
            lo, hi = max(1, c - hw[i]), min(len(es) - 2, c + hw[i])
            if hi < lo:
                continue
            seg = es[lo:hi + 1]
            ismax = (seg >= es[lo - 1:hi]) & (seg >= es[lo + 1:hi + 2])
            if not ismax.any():
                continue
            k = lo + int(np.flatnonzero(ismax)[np.argmax(seg[ismax])])
            top = es[k]
            j, lmin = k, top
            while j > 0 and es[j - 1] <= top:
                j -= 1
                lmin = min(lmin, es[j])
            j, rmin = k, top
            while j < len(es) - 1 and es[j + 1] <= top:
                j += 1
                rmin = min(rmin, es[j])
            prom = top - max(lmin, rmin)
            r = e - es
            noise_res = 1.4826 * np.median(np.abs(r - np.median(r)))
            ok = prom / max(noise_res, F["noise"].values[i]) >= K_SNR and top >= 0.05 * F["_top"].values[i]
            hits[i] += ok
    return hits / len(others)


def truth_table(run):
    if run in ("HZV029_cert", "HZV029_full"):
        p = next((ROOT / "data/raw/ASARI_DATA/x").glob("*/data/hzv029_manual_certified.txt"))
        tr = pd.read_csv(p, sep="\t")
        return pd.DataFrame({"mz": tr["moverz"], "rt": tr["RT_minutes"]})
    if run == "YEAST_NEG":
        tr = pd.read_csv(ROOT / "data/raw/NETID_YEAST/paper_SI/manual_curation_yeast_neg.tsv", sep="\t")
        tr = tr[tr["Confidence"].astype(str).str.upper() == "TRUE"]
        return pd.DataFrame({"mz": tr["medMz"].values, "rt": tr["medRt"].values})
    if run == "LI2018":
        t = pd.read_csv(ROOT / "data/raw/LI2018_QE/m_MTBLS733_mass_spectrometry_v2_maf.tsv", sep="\t")
        return pd.DataFrame({"mz": t["mass_to_charge"].astype(float), "rt": t["retention_time"].astype(float)})
    if run == "IDSL003":
        lab = pd.read_csv(ROOT / "data/raw/IDSL_IPA/idslipa_benchmarking_dataset.csv", encoding="utf-8-sig")
        return pd.DataFrame({"mz": lab["m/z"], "rt": lab["RT(min)"], "tp": lab["Manual Curation"] == "TP"})
    return None


def match_groups(T, g_mz, g_rt, ppm=10.0, rt_tol=0.1):
    """for each truth row: indices of aligned groups within ppm / rt_tol"""
    o = np.argsort(g_mz)
    smz = g_mz[o]
    res = []
    for m, r in zip(T["mz"].values, T["rt"].values):
        lo = np.searchsorted(smz, m * (1 - ppm * 1e-6))
        hi = np.searchsorted(smz, m * (1 + ppm * 1e-6), side="right")
        k = o[lo:hi]
        res.append(k[np.abs(g_rt[k] - r) <= rt_tol])
    return res


REP_MAX_FILES = 10   # replicate check only for runs this small (every file vs every other)


def _one(job):
    run, path, stem, all_files = job
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    F = pd.read_csv(d / "features" / f"{stem}.tsv", sep="\t")
    P = json.load(open(d / "features" / f"{stem}.params.json"))
    cloud = load_cloud(path)
    C = classify_file(cloud, F, P)
    others = []
    if len(all_files) <= REP_MAX_FILES:
        for pth in all_files:
            st = Path(pth).stem
            if st == stem:
                continue
            others.append((load_cloud(pth), pd.read_csv(d / "rt_correction" / f"{st}.tsv", sep="\t"),
                           json.load(open(d / "features" / f"{st}.params.json"))))
    C["rep_frac"] = rep_check(F.assign(_top=C["top_s"].values), P, others)
    C["stem"], C["group_id"], C["row"] = stem, F["group_id"].values, np.arange(len(F))
    for c in ("snr", "height", "n_scans", "fwhm", "score", "iso_offset"):
        C[c] = F[c].values
    C["fwhm_rel"] = F["fwhm"].values / P["fwhm_med"]
    C["n_scans_rel"] = F["n_scans"].values / P["fwhm_scans"]
    return C


def truth_hits(T, G, keep):
    m = match_groups(T, G["mz"].values, G["rt"].values)
    return np.array([len(k) > 0 and np.any(keep[k]) for k in m])


def main(run, max_files=6):
    files = RUNS[run]["files"]
    stems = [Path(f).stem for f in files]
    nf = len(files)
    G = pd.read_csv(ROOT / f"results/diag_excess/{run}_groups.tsv", sep="\t")
    use = list(range(nf)) if nf <= max_files else list(np.linspace(0, nf - 1, max_files).astype(int))
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    with Pool(workers) as pool:
        rows = pool.map(_one, [(run, files[fi], stems[fi], files) for fi in use], chunksize=1)
    if len(rows) <= 12:
        for C in rows:
            print(f"{C['stem'].iloc[0]}: {len(C)} features: " + ", ".join(f"{k} {v}" for k, v in C["cls"].value_counts().items()))
    A = pd.concat(rows, ignore_index=True)
    A = A.merge(G[["group_id", "n_detected", "supported"]], on="group_id", how="left")
    out = ROOT / "results/diag_excess"
    A.to_csv(out / f"{run}_classes.tsv", sep="\t", index=False, float_format="%.5g")
    print(f"\n== {run}: per-file features by class ({len(use)} of {nf} files)")
    g = A.groupby("cls")
    tab = pd.DataFrame({"n": g.size(), "share": (g.size() / len(A)).round(3),
                        "supported": g["supported"].mean().round(3),
                        "det_frac": (g["n_detected"].mean() / nf).round(3),
                        "in_1_file": g["n_detected"].apply(lambda x: (x == 1).mean()).round(3),
                        "med_snr": g["snr"].median().round(1), "med_snr_s": g["snr_s"].median().round(1),
                        "med_fwhm_rel": g["fwhm_rel"].median().round(2), "med_scans": g["n_scans"].median(),
                        "rep_peak": g["rep_frac"].mean().round(3), "rep_none": g["rep_frac"].apply(lambda x: (x == 0).mean()).round(3)})
    print(tab.to_string())
    if A["rep_frac"].notna().any():
        print("\nreplicate peak (share of the other files with a peak there) by n_detected of the group, ok features only:")
        ok = A[A["cls"] == "ok"]
        print(ok.groupby("n_detected")["rep_frac"].agg(["size", "mean", lambda x: (x == 0).mean()]).round(3).to_string())
    # group level: junk if the majority of its classified members are not ok
    gm = A.assign(bad=A["cls"] != "ok").groupby("group_id")["bad"].mean()
    G2 = G.merge(gm.rename("bad_frac"), left_on="group_id", right_index=True, how="left")
    seen = G2["bad_frac"].notna().values
    junk = seen & (G2["bad_frac"].values > 0.5)
    G2["junk"] = junk
    G2.to_csv(out / f"{run}_groups_cls.tsv", sep="\t", index=False, float_format="%.6g")
    print(f"\ngroups with a classified member: {seen.sum()} of {len(G2)}; junk-majority groups: {junk.sum()} "
          f"({junk.sum() / max(seen.sum(), 1):.1%})")
    print(f"  junk-majority groups supported by another tool: {G2.loc[junk, 'supported'].mean():.3f}; "
          f"ok groups supported: {G2.loc[seen & ~junk, 'supported'].mean():.3f}")
    # scenarios: presence filter, EIC re-check, both
    ndet = G2["n_detected"].values
    need = max(2, int(np.ceil(0.1 * nf))) if nf >= 3 else 1
    scen = {"current": np.ones(len(G2), bool),
            f"presence >= {need} files": ndet >= need,
            "EIC re-check": ~junk,
            "both": (ndet >= need) & ~junk}
    T = truth_table(run)
    print(f"\nscenario                     groups   kept   " + ("truth" if T is not None else ""))
    base = None
    for name, keep in scen.items():
        line = f"  {name:26s} {keep.sum():7d}  {keep.mean():5.1%}"
        if T is not None:
            h = truth_hits(T, G2, keep)
            if "tp" in T:
                tp = T["tp"].values
                line += f"   TP {int((h & tp).sum())}  FP {int((h & ~tp).sum())}"
            else:
                line += f"   recall {h.mean():.3f} ({int(h.sum())}/{len(T)})"
            if base is None:
                base = h
        print(line)
    if T is not None and "tp" not in T:
        lost = np.flatnonzero(base & ~truth_hits(T, G2, ~junk))
        if len(lost):
            m = match_groups(T, G2["mz"].values, G2["rt"].values)
            ids = np.concatenate([m[i] for i in lost])
            L = A[A["group_id"].isin(G2["group_id"].values[ids])]
            print("  truth lost by the EIC re-check, member classes: " + ", ".join(f"{k} {v}" for k, v in L["cls"].value_counts().items()))
            L.to_csv(out / f"{run}_lost.tsv", sep="\t", index=False, float_format="%.5g")
        lostp = np.flatnonzero(base & ~truth_hits(T, G2, ndet >= need))
        print(f"  truth lost by the presence filter: {len(lostp)}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
