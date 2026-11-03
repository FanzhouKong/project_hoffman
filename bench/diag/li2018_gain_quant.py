# diagnostic: why the scan-gain correction moves LI2018's median |log2 fold-change error| (score_spike, 10 ppm / 0.1 min).
# Per spiked standard: observed SB/SA log2 under two peak3d variants (results/peak3d_<variant>/LI2018), the change, and
# whether it is a few standards or a shift of every ratio; per file: the gains the correction applied.
# usage: li2018_gain_quant.py BASE_VARIANT GAIN_VARIANT
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.score import load  # noqa: E402

A, B = sys.argv[1:3]
t = pd.read_csv(ROOT / "data/raw/LI2018_QE/m_MTBLS733_mass_spectrometry_v2_maf.tsv", sep="\t")
ab = t["Compound concentration ratio"].str.split("\\\\", expand=True).astype(float)
t["exp"] = np.log2(ab[0] / ab[1])


def observed(v):
    feats, S = load(f"peak3d_{v}", "LI2018")
    sa = [c for c in S.columns if c.startswith("SA")]
    sb = [c for c in S.columns if c.startswith("SB")]
    o = np.argsort(feats["mz"].values)
    mz, rt = feats["mz"].values[o], feats["rt"].values[o]
    obs, a_, b_ = np.full(len(t), np.nan), np.full(len(t), np.nan), np.full(len(t), np.nan)
    for i, (m, r) in enumerate(zip(t["mass_to_charge"], t["retention_time"])):
        lo, hi = np.searchsorted(mz, [m - m * 1e-5, m + m * 1e-5])
        k = lo + np.flatnonzero(np.abs(rt[lo:hi] - r) <= 0.1)
        if len(k):
            j = o[k[np.argmin(np.abs(rt[k] - r))]]
            a, b = S.loc[j, sa].mean(), S.loc[j, sb].mean()
            a_[i], b_[i] = a, b
            if a > 0 and b > 0:
                obs[i] = np.log2(b / a)
    return obs, a_, b_


oa, aa, ba = observed(A)
ob, ab_, bb = observed(B)
diff = t["exp"] != 0
both = ~np.isnan(oa) & ~np.isnan(ob)
ea, eb = np.abs(oa - t["exp"]), np.abs(ob - t["exp"])
print(f"differential standards with a ratio in both: {int((diff & both).sum())}")
print(f"median |err| {A} {np.median(ea[diff & both]):.3f}  {B} {np.median(eb[diff & both]):.3f}")
print(f"median signed error (obs - exp), differential: {A} {np.median((oa - t['exp'])[diff & both]):+.3f}  "
      f"{B} {np.median((ob - t['exp'])[diff & both]):+.3f}")
print(f"median obs log2, constant standards (expected 0): {A} {np.nanmedian(oa[~diff]):+.3f}  {B} {np.nanmedian(ob[~diff]):+.3f}")
d = (eb - ea)[diff & both]
print(f"per standard change in |err| ({B} - {A}): worse {int((d > 0.01).sum())}, better {int((d < -0.01).sum())}, "
      f"same {int((np.abs(d) <= 0.01).sum())}; quartiles {np.round(np.percentile(d, [10, 25, 50, 75, 90]), 3)}")
h = np.log2(bb / ba)[both & (ba > 0)]
print(f"height change {B}/{A} (log2), SB mean heights: median {np.median(h):+.3f}, IQR {np.round(np.percentile(h, [25, 75]), 3)}")
h = np.log2(ab_ / aa)[both & (aa > 0)]
print(f"height change {B}/{A} (log2), SA mean heights: median {np.median(h):+.3f}, IQR {np.round(np.percentile(h, [25, 75]), 3)}")
w = np.argsort(-np.abs(np.nan_to_num((eb - ea).where(diff & both, 0) if hasattr(eb, 'where') else 0)))
x = pd.DataFrame({"compound": t.get("database_identifier", pd.Series(range(len(t)))), "exp": t["exp"], A: oa, B: ob,
                  "d_err": eb - ea})[diff & both].sort_values("d_err")
print("\nlargest changes:")
print(pd.concat([x.head(5), x.tail(8)]).round(3).to_string(index=False))
print("\nper-file gains applied:")
for f in sorted((ROOT / f"results/peak3d_{B}/LI2018/peak3d_out/features").glob("*.params.json")):
    p = json.loads(f.read_text())
    print(f"  {f.name.split('.')[0]}: gain_sd {p['gain_sd']:.4f}  noise_r {p['noise_r']:.4f}  scans with an equation {p['n_gain_scans']}/{p['n_scans']}")
