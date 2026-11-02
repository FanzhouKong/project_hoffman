# data for the synthetic double-peak review page: every inserted double (MassCube synthetic file) that a peak3d
# variant counts wrong under the paper's rule (bench/score.py score_mc_synth: features within 0.01 Da over the
# whole run, correct = exactly 2), with the recorded EIC over the signal's 50 scans, the variant's reported features
# (apex, edges), the candidates it rejected and why (in-process re-pick with the variant's settings, checked against
# its own table), and the counts MassCube 1.0 (published) and asari report for the same signal. The noise-free
# curve each signal was made from follows from its resolution and height ratio (the authors' notebook:
# 50 points on x = 0..4, Gaussians of width 4 sigma = 1, means 2 -/+ res / 2, first peak ratio x the second, scaled
# to 1e6), so the page draws it from those two numbers. Also writes accuracy grids by noise x resolution x ratio.
# usage: synth_double_failures.py VARIANT OUT_JSON   (VARIANT: results/peak3d_<VARIANT>, e.g. G3)
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bench.score as S  # noqa: E402
from bench.runs import RUNS  # noqa: E402
from peak3d.estimate import parse_overrides, prepare  # noqa: E402
from peak3d.features import build_features  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from peak3d.pick import pick_cloud  # noqa: E402

variant, out = sys.argv[1], Path(sys.argv[2])
SETS = {r["variant"]: r["sets"] for r in csv.DictReader((ROOT / "manifest/peak3d_variants.tsv").open(), delimiter="\t")}
SPAN, PAD, TOL, EIC_DA = 50, 8, 0.01, 0.005

t = pd.read_csv(ROOT / "data/truth/MASSCUBE/synth_peaks.tsv", sep="\t")
feats_scored = S.load(f"peak3d_{variant}", "MC_SYNTH")[0]
t["count"] = S.window_counts(feats_scored, t.mz.values, TOL)
asari = S.load("asari", "MC_SYNTH")
t["count_asari"] = S.window_counts(asari[0], t.mz.values, TOL) if asari is not None else -1
t["count_mc10"] = t["pub_MassCube 1.0"]

# accuracy grids (all signals, not only the failing ones)
grid = []
for (n_peaks, noise, res, ratio), g in t.groupby(["n_peaks", "noise", "resolution", "intensity_ratio"]):
    grid.append(dict(n_peaks=int(n_peaks), noise=float(noise), res=float(res), ratio=int(ratio), n=len(g),
                     peak3d=int((g["count"] == n_peaks).sum()), mc10=int((g.count_mc10 == n_peaks).sum()),
                     asari=int((g.count_asari == n_peaks).sum()), merged=int(((g["count"] > 0) & (g["count"] < n_peaks)).sum()),
                     missed=int((g["count"] == 0).sum()), extra=int((g["count"] > n_peaks).sum())))

bad = t[(t.n_peaks == 2) & (t["count"] != 2)].copy()
print(f"{variant}: {len(bad)} of {(t.n_peaks == 2).sum()} doubles wrong", flush=True)

path = RUNS["MC_SYNTH"]["files"][0]
cloud = load_cloud(path)
rt = cloud.rt.copy()
raw = {r.Index: cloud.eic(r.mz, EIC_DA / r.mz * 1e6, max(0, r.start_scan - PAD), min(cloud.n_scans, r.start_scan + SPAN + PAD))
       for r in bad.itertuples()}
inten_raw = cloud.inten.copy()
P = prepare(cloud, overrides=parse_overrides(SETS[variant].split()))
gain = np.ones(cloud.n_scans)
if P.scan_gain:   # recover the per-scan gains from the corrected cloud (prepare divides in place)
    idx = np.repeat(np.arange(cloud.n_scans), np.diff(cloud.off))
    ok = cloud.inten > 0
    gsum = np.bincount(idx[ok], weights=np.log(inten_raw[ok] / cloud.inten[ok]), minlength=cloud.n_scans)
    gcnt = np.bincount(idx[ok], minlength=cloud.n_scans)
    gain = np.exp(np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), 0.0))
b = pick_cloud(cloud, P)
kept, rej = build_features(cloud, b, keep_rejected=True)
per = pd.read_csv(ROOT / f"results/peak3d_{variant}/MC_SYNTH/peak3d_out/features/{Path(path).stem}.tsv", sep="\t",
                  usecols=["mz", "rt_apex", "rt_min", "rt_max", "height", "fwhm", "n_scans", "score"])
print(f"re-pick {len(kept)} features vs {len(per)} in the variant's table; scored table {len(feats_scored)}", flush=True)


def near(T, m):
    return T[np.abs(T["mz"].values - m) < TOL]


cases = []
for r in bad.itertuples():
    s0 = max(0, r.start_scan - PAD)
    s1 = min(cloud.n_scans, r.start_scan + SPAN + PAD)
    n = int(r.count)
    feats = [dict(rt=round(float(x.rt_apex), 4), lo=round(float(x.rt_min), 4), hi=round(float(x.rt_max), 4),
                  height=float(x.height), fwhm_s=round(float(x.fwhm) * 60, 2), n_scans=int(x.n_scans),
                  score=round(float(x.score), 3), dmda=round((x.mz - r.mz) * 1e3, 2)) for x in near(per, r.mz).itertuples()]
    rejected = [dict(rt=round(float(x["rt"]), 4), lo=round(float(x["rt_min"]), 4), hi=round(float(x["rt_max"]), 4),
                     height=float(x["height"]), fwhm_s=round(float(x["fwhm"]) * 60, 2), n_scans=int(x["n_scans"]),
                     reason=str(x["reason"]) + (" (fragment of a neighbour)" if x.get("fragment", False) else ""))
                for _, x in near(rej, r.mz).iterrows()]
    cases.append(dict(mz=round(float(r.mz), 1), noise=float(r.noise), res=float(r.resolution), ratio=int(r.intensity_ratio),
                      start=int(r.start_scan), s0=s0, count=n, count_mc10=int(r.count_mc10), count_asari=int(r.count_asari),
                      cat="missed" if n == 0 else ("merged" if n == 1 else "extra"),
                      raw=[round(float(v)) for v in raw[r.Index]], feats=feats, rejected=rejected))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dict(variant=variant, sets=SETS[variant], scan_gain=bool(P.scan_gain), span=SPAN, pad=PAD,
                               rt=[round(float(v), 4) for v in rt], gain=[round(float(v), 4) for v in gain],
                               grid=grid, cases=cases), separators=(",", ":")))
c = pd.DataFrame(cases)
print(f"{len(cases)} failing doubles -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
print(c.groupby(["noise", "cat"]).size().unstack(fill_value=0).to_string())
print(c.groupby(["res", "ratio"]).size().unstack(fill_value=0).to_string())
print("MassCube 1.0 right on", int((c.count_mc10 == 2).sum()), "| asari right on", int((c.count_asari == 2).sum()))
