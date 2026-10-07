# diagnostic: what does an IDSL.IPA label denote? Tests (1) whether each label m/z-RT is an actual centroid of
# 003.mzML (a sampled data point) rather than a peak summary, (2) where TP / TN points sit on their own ion trace
# (apex vs flank vs tail), (3) which matching rule reproduces the authors' published tool columns: apex within
# +/-0.1 min (bench/score.py) or "point inside the tool's peak RT bounds", using our own runs of IDSL.IPA / peak3d /
# asari, which report peak bounds.
# usage: idsl_label_semantics.py AUDIT_DIR   (reads 003.mzML -> srun)
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.msdata import load_run  # noqa: E402

D = Path(sys.argv[1])
A = pd.read_csv(D / "verdict.tsv", sep="\t")
tp = (A["Manual Curation"] == "TP").values
mz, rt = A["m/z"].values, A["RT(min)"].values
run = load_run(ROOT / "data/raw/IDSL_IPA/003.mzML")
pd.set_option("display.width", 220, "display.max_columns", 30)


# ---- (1) is the label an actual centroid? nearest centroid within +/-3 scans of the label RT
def nearest_centroid(m0, t0, dm=0.0):
    m0 = m0 + dm
    k0 = int(np.argmin(np.abs(run.rt - t0)))
    a, b = np.searchsorted(run.mz, [m0 - 0.002, m0 + 0.002])
    sc, mm, ii = run.scan[a:b], run.mz[a:b], run.inten[a:b]
    w = np.abs(sc - k0) <= 3
    if not w.any():
        return np.inf, np.nan, np.nan
    j = np.argmin(np.abs(mm[w] - m0))
    return abs(mm[w][j] - m0), sc[w][j] - k0, ii[w][j]


res = np.array([nearest_centroid(m, t) for m, t in zip(mz, rt)])
ctl = np.array([nearest_centroid(m, t, 0.00137) for m, t in zip(mz, rt)])  # control: m/z shifted 1.37 mDa
dec = A["m/z"].astype(str).str.split(".").str[1].str.len().values
print("== (1) nearest centroid to each label within +/-3 scans")
for nm, r in (("labels", res), ("control (m/z + 1.37 mDa)", ctl)):
    d = r[:, 0]
    print(f"  {nm:26s} |dm| <= 5e-6 Da (exact at 5 decimals): {np.mean(d <= 5.01e-6):.3f}   <= 5e-5: {np.mean(d <= 5e-5):.3f}"
          f"   median |dm| {np.median(d) * 1e3:.4f} mDa")
ex = res[:, 0] <= 5.01e-6
print(f"  exact-match labels: scan offset of the matched centroid {pd.Series(res[ex, 1]).value_counts().sort_index().to_dict()}")
print(f"  exact-match rate TP {ex[tp].mean():.3f}  TN {ex[~tp].mean():.3f}; labels printed with 5 decimals {np.mean(dec == 5):.3f}")

# ---- (2) where does the point sit on its own trace? climb from the label scan to the local apex (10 ppm EIC, smoothed)
def smooth(e):
    return np.convolve(np.pad(e, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")


pos = []
for m0, t0 in zip(mz, rt):
    k0 = int(np.argmin(np.abs(run.rt - t0)))
    e = run.eic(m0, 10.0)
    es = smooth(e)
    k = k0
    for kk in (k0 - 1, k0 + 1):  # start from the best of the label scan +/-1 (RT is rounded to 0.01 min)
        if 0 <= kk < len(es) and es[kk] > es[k]:
            k = kk
    kc = k
    while kc + 1 < len(es) and es[kc + 1] > es[kc]:
        kc += 1
    while kc - 1 >= 0 and es[kc - 1] > es[kc]:
        kc -= 1
    pos.append(((run.rt[kc] - t0) * 60, es[k] / es[kc] if es[kc] > 0 else np.nan, e[k0]))
pos = np.array(pos)
A["apex_off_s"], A["rel_to_apex"], A["e_at_label"] = pos[:, 0], pos[:, 1], pos[:, 2]
print("\n== (2) label point vs the apex it climbs to (10 ppm, smoothed)")
for nm, m in (("TP", tp), ("TN", ~tp)):
    print(f"  {nm}: |apex offset| s quantiles {np.nanquantile(np.abs(pos[m, 0]), [.25, .5, .75, .9]).round(1).tolist()}, "
          f"point/apex height quantiles {np.nanquantile(pos[m, 1], [.1, .25, .5, .75]).round(2).tolist()}, "
          f"zero signal at label scan {np.mean(pos[m, 2] == 0):.3f}")
for nm, m in (("TP confirmed", A.verdict.str.startswith("TP confirmed").values),
              ("TP misplaced", A.verdict.str.startswith("TP misplaced").values)):
    print(f"  {nm}: |apex offset| > 6 s: {np.mean(np.abs(pos[m, 0]) > 6):.3f}, point/apex median {np.nanmedian(pos[m, 1]):.2f}")
print("  TP labels whose own apex is > 6 s away (unmatchable at 0.1 min by an apex reporter): "
      f"{int((tp & (np.abs(pos[:, 0]) > 6)).sum())}")

# ---- (3) matching rules vs the authors' tool columns, using our own tool runs
ipa = pd.read_csv(next((ROOT / "results/idslipa/IDSL003/idslipa_out/peaklists").glob("peaklist_*.csv")))
ipa_b = pd.DataFrame(dict(mz=ipa["m/z 12C"], rt=ipa["retentionTimeApex"],
                          lo=run.rt[np.clip(ipa.ScanNumberStart.values - 1, 0, run.n_scans - 1)],
                          hi=run.rt[np.clip(ipa.ScanNumberEnd.values - 1, 0, run.n_scans - 1)]))
p3 = pd.read_csv(ROOT / "results/peak3d/IDSL003/peak3d_out/features/003.tsv", sep="\t")
p3_b = pd.DataFrame(dict(mz=p3["mz"], rt=p3["rt"], lo=p3["rt_min"], hi=p3["rt_max"]))
asr = pd.read_csv(next((ROOT / "results/asari/IDSL003").glob("asari_out*/export/full_Feature_table.tsv")), sep="\t")
asr_b = pd.DataFrame(dict(mz=asr["mz"], rt=asr["rtime"] / 60, lo=asr["rtime_left_base"] / 60, hi=asr["rtime_right_base"] / 60))


def detect(F, rule, ppm=10.0, da=None, rt_tol=0.1, pad=0.0):
    o = np.argsort(F.mz.values)
    fm, fr, flo, fhi = (F[c].values[o] for c in ("mz", "rt", "lo", "hi"))
    out = np.zeros(len(mz), bool)
    for i, (m0, t0) in enumerate(zip(mz, rt)):
        tol = da if da is not None else m0 * ppm * 1e-6
        a, b = np.searchsorted(fm, [m0 - tol, m0 + tol])
        if b <= a:
            continue
        if rule == "apex":
            out[i] = np.any(np.abs(fr[a:b] - t0) <= rt_tol)
        else:
            out[i] = np.any((flo[a:b] - pad <= t0) & (t0 <= fhi[a:b] + pad))
    return out


pub_ipa = A["IDSL.IPA"].isin(["TP", "FP"]).values
print("\n== (3) which rule reproduces the published IDSL.IPA column (our IDSL.IPA run, default parameters)?")
print(f"  published IDSL.IPA reports {pub_ipa.sum()} of the 20,000 pairs; our run has {len(ipa_b)} peaks")
rows = []
for nm, kw in (("apex +/-0.1 min, 10 ppm", dict(rule="apex")), ("apex +/-0.1 min, 0.01 Da", dict(rule="apex", da=0.01)),
               ("apex +/-0.2 min, 0.01 Da", dict(rule="apex", da=0.01, rt_tol=0.2)),
               ("inside RT bounds, 10 ppm", dict(rule="bounds")), ("inside RT bounds, 0.01 Da", dict(rule="bounds", da=0.01))):
    d = detect(ipa_b, **kw)
    agree = (d == pub_ipa).mean()
    n11, n10, n01 = (d & pub_ipa).sum(), (d & ~pub_ipa).sum(), (~d & pub_ipa).sum()
    pe = d.mean() * pub_ipa.mean() + (1 - d.mean()) * (1 - pub_ipa.mean())
    kappa = (agree - pe) / (1 - pe)
    rows.append(dict(rule=nm, n_detected=int(d.sum()), both=int(n11), ours_only=int(n10), published_only=int(n01),
                     agreement=round(agree, 4), kappa=round(kappa, 3)))
print(pd.DataFrame(rows).to_string(index=False))

# ---- score every tool with bounds under both rules
print("\n== IDSL003 scores under the two matching rules (10 ppm; apex +/-0.1 min vs label point inside the peak's RT bounds)")
srows = []
for nm, F in (("idslipa", ipa_b), ("peak3d (per-file features)", p3_b), ("asari", asr_b)):
    for rule, kw in (("apex 0.1 min", dict(rule="apex")), ("inside bounds", dict(rule="bounds"))):
        d = detect(F, **kw)
        TP, FP, FN = (d & tp).sum(), (d & ~tp).sum(), (~d & tp).sum()
        srows.append(dict(tool=nm, rule=rule, n_feat=len(F), TP=int(TP), FP=int(FP), FN=int(FN),
                          prec=round(TP / max(TP + FP, 1), 3), rec=round(TP / (TP + FN), 3), F1=round(2 * TP / (2 * TP + FP + FN), 3)))
print(pd.DataFrame(srows).to_string(index=False))
A.to_csv(D / "verdict.tsv", sep="\t", index=False)
