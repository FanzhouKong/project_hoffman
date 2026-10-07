# diagnostic: why does peak3d report more features than asari / MassCube / the other tools? Breaks the aligned
# table down by cross-file presence, cross-tool support and the members' own descriptors, and
# counts within-file same-ion neighbours (split candidates) and co-eluting m/z satellites.
# usage: feature_excess.py RUN [RUN ...]
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import LOCAL_TOOLS  # noqa: E402
from bench.score import load  # noqa: E402

OTHERS = ["asari", "masscube", *LOCAL_TOOLS]
PPM, RT_TOL = 10.0, 0.1


def support(q_mz, q_rt, feats, ppm=PPM, rt_tol=RT_TOL):
    o = np.argsort(feats["mz"].values)
    mz, rt = feats["mz"].values[o], feats["rt"].values[o]
    lo = np.searchsorted(mz, q_mz * (1 - ppm * 1e-6))
    hi = np.searchsorted(mz, q_mz * (1 + ppm * 1e-6), side="right")
    out = np.zeros(len(q_mz), bool)
    for i in range(len(q_mz)):
        out[i] = np.any(np.abs(rt[lo[i]:hi[i]] - q_rt[i]) <= rt_tol)
    return out


def neighbours(mz, rt, h, ppm, rt_win, stronger=True, min_ppm=0.0):
    """per feature: is there another (stronger) feature within [min_ppm, ppm] and rt_win?"""
    o = np.argsort(mz)
    smz = mz[o]
    lo = np.searchsorted(smz, mz * (1 - ppm * 1e-6))
    hi = np.searchsorted(smz, mz * (1 + ppm * 1e-6), side="right")
    out = np.zeros(len(mz), bool)
    for i in range(len(mz)):
        k = o[lo[i]:hi[i]]
        k = k[k != i]
        if min_ppm > 0:
            k = k[np.abs(mz[k] - mz[i]) / mz[i] * 1e6 >= min_ppm]
        k = k[np.abs(rt[k] - rt[i]) <= rt_win]
        if stronger:
            k = k[h[k] > h[i]]
        out[i] = len(k) > 0
    return out


def run_report(run):
    d = ROOT / "results/peak3d" / run / "peak3d_out"
    t = pd.read_csv(d / "aligned_feature_table.tsv", sep="\t")
    stems = [c for c in t.columns if c not in ("group_id", "mz", "rt", "rt_min", "rt_max", "n_detected",
                                               "n_filled", "mz_ppm_spread", "rt_sd", "iso_offset", "charge")]
    nf = len(stems)
    P = [json.load(open(d / "features" / f"{s}.params.json")) for s in stems]
    fwhm = float(np.median([p["fwhm_med"] for p in P]))
    files = [f for f in glob.glob(str(d / "features/*.tsv")) if not f.endswith(("manifest.tsv", "rejected.tsv"))]
    F = pd.concat([pd.read_csv(f, sep="\t").assign(stem=Path(f).stem) for f in files], ignore_index=True)
    per_file = F.groupby("stem").size()
    print(f"\n=========== {run}: {nf} files, FWHM_med {60 * fwhm:.1f} s")
    print(f"peak3d per-file features: median {per_file.median():.0f} (min {per_file.min()}, max {per_file.max()}); "
          f"aligned groups {len(t)} ({len(t) / per_file.median():.2f}x a file)")
    others = {}
    for tool in OTHERS:
        x = load(tool, run)
        if x is not None:
            others[tool] = x[0]
            print(f"  {tool:9s} aligned features {len(x[0]):7d}")
    # cross-file presence
    frac = t["n_detected"] / nf
    bins = [0, 1.5 / nf, 0.1, 0.25, 0.5, 0.75, 1.0001] if nf > 10 else [0] + [(k + 0.5) / nf for k in range(nf)] + [1.0001]
    lab = pd.cut(frac, bins, right=False)
    # cross-tool support
    sup = {k: support(t["mz"].values, t["rt"].values, v) for k, v in others.items()}
    any_sup = np.any(np.column_stack(list(sup.values())), axis=1) if sup else np.zeros(len(t), bool)
    t["supported"] = any_sup
    # member descriptors (median over the detected members of a group)
    agg = F.groupby("group_id").agg(snr=("snr", "median"), height=("height", "median"), r2=("gauss_r2", "median"),
                                    n_scans=("n_scans", "median"), fwhm=("fwhm", "median"),
                                    prom=("prominence_rel", "median"), pers=("persistence_rel", "median"),
                                    score=("score", "median"), sd=("mz_sd_ppm", "median"),
                                    gaps=("n_gaps", "median"), iso=("iso_offset", "max"),
                                    ridge=("ridge_ratio", "median") if "ridge_ratio" in F else ("snr", "size"))
    t = t.merge(agg, left_on="group_id", right_index=True, how="left")
    print("\npresence (fraction of files detected) -> groups, share supported by >=1 other tool:")
    g = t.groupby(lab, observed=True)
    print(pd.DataFrame({"groups": g.size(), "share": (g.size() / len(t)).round(3),
                        "supported": g["supported"].mean().round(3),
                        "med_snr": g["snr"].median().round(1), "med_score": g["score"].median().round(2)}).to_string())
    print("\nsnr bin (median of members) -> groups, supported:")
    sb = pd.cut(t["snr"], [0, 5, 10, 20, 50, 100, 1e12], right=False)
    g = t.groupby(sb, observed=True)
    print(pd.DataFrame({"groups": g.size(), "share": (g.size() / len(t)).round(3),
                        "supported": g["supported"].mean().round(3)}).to_string())
    # within-file classification of every detected feature
    rows = []
    for stem, f in F.groupby("stem"):
        mz, rt, h = f["mz"].values, f["rt"].values, f["height"].values
        fw = json.load(open(d / "features" / f"{stem}.params.json"))["fwhm_med"]
        split = neighbours(mz, rt, h, 5.0, 3.0 * fw)                        # same ion, within 3 FWHM, stronger
        sat = neighbours(mz, rt, h * 0 + 1, 60.0, 0.25 * fw, stronger=False, min_ppm=8.0)   # co-eluting m/z neighbour
        strong_sat = np.zeros(len(f), bool)
        # satellite = co-eluting (same apex within 1/4 FWHM) neighbour 8-60 ppm away AND >= 20x stronger
        o = np.argsort(mz)
        smz = mz[o]
        lo = np.searchsorted(smz, mz * (1 - 60e-6))
        hi = np.searchsorted(smz, mz * (1 + 60e-6), side="right")
        for i in np.flatnonzero(sat):
            k = o[lo[i]:hi[i]]
            k = k[(k != i) & (np.abs(mz[k] - mz[i]) / mz[i] * 1e6 >= 8) & (np.abs(rt[k] - rt[i]) <= 0.25 * fw)]
            strong_sat[i] = np.any(h[k] >= 20 * h[i])
        rows.append(pd.DataFrame({"stem": stem, "group_id": f["group_id"].values, "split": split,
                                  "satellite": strong_sat}))
    W = pd.concat(rows, ignore_index=True)
    gw = W.groupby("group_id").agg(split=("split", "mean"), satellite=("satellite", "mean"))
    t = t.merge(gw, left_on="group_id", right_index=True, how="left")
    print(f"\nwithin-file: features with a STRONGER same-ion feature within 5 ppm / 3 FWHM: {W['split'].mean():.1%}; "
          f"with a >=20x stronger co-eluting feature 8-60 ppm away (satellite): {W['satellite'].mean():.1%}")
    uns = t[~t["supported"]]
    s = t[t["supported"]]
    print(f"\nunsupported groups: {len(uns)} ({len(uns) / len(t):.1%}); comparison supported vs unsupported (medians / shares):")
    cmp = pd.DataFrame({
        "n_detected/n_files": [s["n_detected"].mean() / nf, uns["n_detected"].mean() / nf],
        "snr": [s["snr"].median(), uns["snr"].median()],
        "height": [s["height"].median(), uns["height"].median()],
        "gauss_r2": [s["r2"].median(), uns["r2"].median()],
        "n_scans": [s["n_scans"].median(), uns["n_scans"].median()],
        "fwhm/fwhm_med": [s["fwhm"].median() / fwhm, uns["fwhm"].median() / fwhm],
        "prominence_rel": [s["prom"].median(), uns["prom"].median()],
        "persistence_rel": [s["pers"].median(), uns["pers"].median()],
        "score": [s["score"].median(), uns["score"].median()],
        "mz_sd_ppm": [s["sd"].median(), uns["sd"].median()],
        "isotope share": [(s["iso"] > 0).mean(), (uns["iso"] > 0).mean()],
        "split-candidate share": [(s["split"] > 0.5).mean(), (uns["split"] > 0.5).mean()],
        "satellite share": [(s["satellite"] > 0.5).mean(), (uns["satellite"] > 0.5).mean()],
        "detected in 1 file": [(s["n_detected"] == 1).mean(), (uns["n_detected"] == 1).mean()],
    }, index=["supported", "unsupported"]).T
    print(cmp.round(3).to_string())
    out = ROOT / "results/diag_excess"
    out.mkdir(exist_ok=True)
    t.to_csv(out / f"{run}_groups.tsv", sep="\t", index=False, float_format="%.6g")
    return t


if __name__ == "__main__":
    for r in sys.argv[1:]:
        run_report(r)
