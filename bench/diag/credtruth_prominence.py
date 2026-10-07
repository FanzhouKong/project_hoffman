# diagnostic / truth QC: the credentialed truth (bench/credtruth.py) seeds every local maximum of the smoothed EIC
# (find_peaks(prominence=0)), so a bump on the tail or shoulder of a bigger peak of the same ion becomes a truth entry
# whenever its U-13C partner shows the same bump. This script measures, in the three 12C injections, the raw prominence
# of every truth entry (apex above the deeper of the two side minima within 3 FWHM) and whether a stronger entry of the
# same ion lies within 0.3 min, writes data/truth/<DS>/credentialed_prominence.tsv, and scores each tool's benchmark
# output on the full list and on the entries that pass (median relative prominence >= 0.2, our own minimum).
# usage: credtruth_prominence.py DATASET [DATASET ...]   (YEAST | SZ22; reads mzML -> srun)
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import CRED_PAIRS, TOOLS  # noqa: E402
from bench.score import hits, load  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402

MIN_PROM = 0.2
WIN = 0.1          # +/- min around the truth RT for the apex
PPM = 5.0


def prominence(c, mz, rt, side_min):
    s0, s1 = int(max(0, np.searchsorted(c.rt, rt - WIN - side_min))), int(min(c.n_scans, np.searchsorted(c.rt, rt + WIN + side_min)))
    e = c.eic(mz, PPM, s0, s1).astype(np.float64)
    r = c.rt[s0:s1]
    if len(e) < 3:
        return np.nan, 0.0
    win = np.abs(r - rt) <= WIN
    if not win.any() or e[win].max() <= 0:
        return np.nan, 0.0
    i = int(np.flatnonzero(win)[np.argmax(e[win])])
    h = e[i]
    left = e[(r < r[i]) & (r >= r[i] - side_min)]
    right = e[(r > r[i]) & (r <= r[i] + side_min)]
    lmin = left.min() if len(left) else 0.0
    rmin = right.min() if len(right) else 0.0
    return float((h - max(lmin, rmin)) / h), float(h)


def main(ds):
    T = pd.read_csv(ROOT / f"data/truth/{ds}/credentialed.tsv", sep="\t")
    prm = json.loads((ROOT / f"data/truth/{ds}/params.json").read_text())
    files = prm["files_12C"]
    side_min = 3.0 * prm["FWHM_s"] / 60.0
    for k, f in enumerate(files):
        c = load_cloud(f)
        pr, hh = zip(*[prominence(c, float(m), float(r), side_min) for m, r in zip(T["mz"], T["rt"])])
        T[f"prom_rel_{k}"] = pr
        T[f"apex_{k}"] = hh
        print(ds, Path(f).stem, "done", flush=True)
    pcols = [f"prom_rel_{k}" for k in range(len(files))]
    T["prom_rel_med"] = T[pcols].median(axis=1)
    # a stronger entry of the same ion within 0.3 min
    mz, rt, h = T["mz"].values, T["rt"].values, T["height12"].values
    sib = np.zeros(len(T), bool)
    for i in range(len(T)):
        m = (np.abs(mz - mz[i]) / mz[i] * 1e6 <= 10) & (np.abs(rt - rt[i]) <= 0.3) & (h > h[i])
        sib[i] = m.any()
    T["weaker_sibling"] = sib
    T["keep"] = T["prom_rel_med"] >= MIN_PROM
    out = ROOT / f"data/truth/{ds}/credentialed_prominence.tsv"
    T.to_csv(out, sep="\t", index=False)
    print(f"\n== {ds}: {len(T)} truth entries; relative prominence (median of {len(files)} injections) quantiles",
          np.nanpercentile(T["prom_rel_med"], [5, 10, 25, 50, 75]).round(2).tolist())
    print(f"   pass (prom_rel_med >= {MIN_PROM}): {int(T['keep'].sum())}; fail: {int((~T['keep']).sum())}; of the failing, weaker sibling of a found-able entry: {int((~T['keep'] & T['weaker_sibling']).sum())}")
    print("   by tier:", T.groupby("tier")["keep"].agg(["size", "mean"]).round(3).to_dict())
    print("   by class:", T.groupby("feature_class")["keep"].agg(["size", "mean"]).round(3).to_dict())
    r12, _ = CRED_PAIRS[ds]
    rows = []
    for tool in TOOLS:
        x = load(tool, r12)
        if x is None:
            continue
        hit = hits(x[0], T["mz"].values, T["rt"].values, 10.0, 0.1)
        rows.append(dict(tool=tool, recall_all=round(hit.mean(), 3), recall_pass=round(hit[T["keep"].values].mean(), 3),
                         recall_fail=round(hit[~T["keep"].values].mean(), 3), n_features=len(x[0])))
    print(pd.DataFrame(rows).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        main(ds)
