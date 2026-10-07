# diagnostic: is the HZV029 certified RT on the same axis as the three cert files? Per certified feature:
# raw EIC apex (5 ppm, highest point within +/-0.25 min) in each file on its native axis, and the
# nearest feature of each tool within 10 ppm / 0.25 min, as offsets (s) from the certified RT.
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import LOCAL_TOOLS, RUNS  # noqa: E402
from bench.score import load  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402

p = next((ROOT / "data/raw/ASARI_DATA/x").glob("*/data/hzv029_manual_certified.txt"))
tr = pd.read_csv(p, sep="\t")
out = pd.DataFrame({"truth_id": np.arange(len(tr)), "mz": tr["moverz"], "rt": tr["RT_minutes"]})
for f in RUNS["HZV029_cert"]["files"]:
    c = load_cloud(f)
    tag = Path(f).stem[-1]
    d = []
    for mz, rt in zip(tr["moverz"], tr["RT_minutes"]):
        s0, s1 = np.searchsorted(c.rt, rt - 0.25), np.searchsorted(c.rt, rt + 0.25)
        e = c.eic(mz, 5.0, s0, s1)
        d.append(60 * (c.rt[s0 + int(np.argmax(e))] - rt) if e.max() > 0 else np.nan)
    out[f"raw_{tag}"] = d
for tool in ("peak3d", "asari", "masscube", *LOCAL_TOOLS):
    x = load(tool, "HZV029_cert")[0]
    d = []
    for mz, rt in zip(tr["moverz"], tr["RT_minutes"]):
        k = (np.abs(x["mz"].values - mz) / mz * 1e6 <= 10) & (np.abs(x["rt"].values - rt) <= 0.25)
        d.append(60 * (x["rt"].values[k] - rt)[np.argmin(np.abs(x["rt"].values[k] - rt))] if k.any() else np.nan)
    out[tool] = d
out.to_csv(ROOT / "results/diag_excess/hzv029_rt_offset.tsv", sep="\t", index=False, float_format="%.4f")
b = pd.cut(out["rt"], [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5])
cols = ["raw_C", "raw_G", "raw_K", "peak3d", "asari", "masscube", *LOCAL_TOOLS]
print("median offset (s) from the certified RT, by certified RT bin (min); n per bin")
print(out.groupby(b, observed=True)[cols].median().round(1).assign(n=out.groupby(b, observed=True).size()).to_string())
print("\nshare with |offset| > 6 s (= 0.1 min):")
print(out.groupby(b, observed=True)[cols].apply(lambda g: (g.abs() > 6).mean()).round(2).to_string())
