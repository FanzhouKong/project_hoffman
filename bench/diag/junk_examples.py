# data for the junk-check explainer page: for one replicate run, the features a peak3d variant reports in its first
# injection, each judged exactly as version_junk.py does (trace check against one reference noise per file, and a
# smoothed peak at the corrected RT in each other injection), with example features of every outcome and their raw EICs
# in all injections (the expected RT window in the others, and whether a peak was found there).
# usage: junk_examples.py RUN VARIANT OUT_JSON   (VARIANT: results/peak3d_<VARIANT>, or "default" for results/peak3d)
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS  # noqa: E402
from junk_classes import HALF_WIN, JUNK_RT_WINDOW_S, classify_file, rep_check  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from version_junk import reference_noise  # noqa: E402

run, variant, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
d = ROOT / ("results/peak3d" if variant == "default" else f"results/peak3d_{variant}") / run / "peak3d_out"
files = RUNS[run]["files"]
stems = [Path(f).stem for f in files]
clouds = [load_cloud(f) for f in files]
Ps = [json.load(open(d / "features" / f"{s}.params.json")) for s in stems]
warps = [pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t") for s in stems]
F = pd.read_csv(d / "features" / f"{stems[0]}.tsv", sep="\t")
if "group_kept" in F.columns:
    F = F[F["group_kept"]].reset_index(drop=True)
look = reference_noise(clouds[0])
F["noise"] = look(F["mz"].values, F["rt"].values)
C = classify_file(clouds[0], F, Ps[0])
F["_top"] = C["top_s"].values
per = np.stack([rep_check(F, Ps[0], [(clouds[k], warps[k], Ps[k])], run=run) for k in range(1, len(files))], axis=1)
rep = per.mean(axis=1)
cls = C["cls"].values
norep = rep == 0
junk = norep | (cls != "ok")
good = (rep >= 0.5) & (cls == "ok")
summary = dict(features=int(len(F)), junk=int(junk.sum()), good=int(good.sum()), not_reproduced=int(norep.sum()),
               fails_trace=int((cls != "ok").sum()), only_not_reproduced=int((norep & (cls == "ok")).sum()),
               only_trace=int((~norep & (cls != "ok")).sum()), both=int((norep & (cls != "ok")).sum()),
               partly_reproduced=int(((rep > 0) & (rep < 0.5) & (cls == "ok")).sum()),
               classes={k: int((cls == k).sum()) for k in np.unique(cls)})


def example(i, cat):
    f = F.iloc[i]
    fw = max(float(f["fwhm"]), Ps[0]["fwhm_med"], JUNK_RT_WINDOW_S[run] / 60.0)    # the replicate search window
    half = max(0.25, 4 * fw)
    traces = []
    for k, cl in enumerate(clouds):
        center = float(f["rt"])                                                   # raw RT in every injection
        s0, s1 = np.searchsorted(cl.rt, [center - half, center + half])
        e = cl.eic(float(f["mz"]), 5.0, s0, s1)
        traces.append(dict(injection=k + 1, rt=[round(float(v), 4) for v in cl.rt[s0:s1]], y=[round(float(v), 1) for v in e],
                           expected=round(center, 4), window=round(fw, 4) if k else None,
                           found=None if k == 0 else bool(per[i, k - 1])))
    return dict(cat=cat, mz=round(float(f["mz"]), 4), rt=round(float(f["rt"]), 3), height=float(f["height"]),
                lo=round(float(f["rt_min"]), 4), hi=round(float(f["rt_max"]), 4), cls=str(cls[i]), rep=float(rep[i]),
                snr_s=round(float(C["snr_s"].values[i]), 2), noise=round(float(F["noise"].values[i]), 1), traces=traces)


rng = np.random.default_rng(1)
h = F["height"].values
mid = (h > np.quantile(h, 0.3))                       # examples visible enough to read on a chart
picks = []
for cat, mask, n in [("good", good & mid, 4), ("not reproduced", norep & (cls == "ok") & mid, 4),
                     ("partly reproduced", (rep > 0) & (rep < 0.5) & (cls == "ok") & mid, 2),
                     ("noapex", (cls == "noapex") & mid, 3), ("lowsnr", (cls == "lowsnr"), 3), ("dup", (cls == "dup") & mid, 2),
                     ("edge", np.isin(cls, ["edge", "trunc"]), 2)]:
    idx = np.flatnonzero(mask)
    for i in (rng.choice(idx, min(n, len(idx)), replace=False) if len(idx) else []):
        picks.append(example(int(i), cat))
out.write_text(json.dumps(dict(run=run, variant=variant, injections=len(files), half_win=HALF_WIN, summary=summary,
                               examples=picks), separators=(",", ":")))
print(json.dumps(summary, indent=1))
print(f"{len(picks)} examples -> {out} ({out.stat().st_size / 1e3:.0f} kB)")
