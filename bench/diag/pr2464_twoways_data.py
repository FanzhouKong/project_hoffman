"""Data for the "Two ways to find a peak" page (claude.ai/artifact/CyZs7uKTqZjLWPQHKnqvx6): one yeast-12C injection
(ST004501 vial 23, batch 1) segmented by the credentialing seeds (bench/credtruth.py) and by peak3d.
usage: pr2464_twoways_data.py OUT.json   (run on a compute node; ~2 min)"""
import json, sys, time
import numpy as np, pandas as pd
ROOT = "/quobyte/metabolomicsgrp/fanzhou/hoffmann"
sys.path.insert(0, ROOT)
from peak3d.io import load_cloud
from peak3d.estimate import estimate
from peak3d.pick import pick_cloud
from peak3d.features import build_features
from peak3d import kernels as K
from bench.msdata import Run
from bench import credtruth as ct
from bench import spiketruth as st

study = "ST004501"
f = st.study_files(study)
row = f[f.vial == 23].iloc[0]
cloud = load_cloud(row.path, "pos")
N, S = cloud.n_points, cloud.n_scans
scan = np.repeat(np.arange(S), np.diff(cloud.off)).astype(np.int32)
mz, inten, rt = cloud.mz, cloud.inten.astype(float), cloud.rt
print("file", row.file, "centroids", N, "scans", S, flush=True)

# ---- method A: global m/z sort, cut at 3 ppm gaps (credtruth.seeds_from) ----
t0 = time.time()
o = np.argsort(mz, kind="stable")
gaps = np.diff(mz[o]) / mz[o][1:] * 1e6
cid_sorted = np.concatenate([[0], np.cumsum(gaps > ct.CLUSTER_PPM)])
cluster = np.empty(N, np.int64); cluster[o] = cid_sorted
run = Run(row.file, rt, mz[o], inten[o], scan[o])
seeds = pd.DataFrame(ct.seeds_from(run), columns=["mz", "rt", "h"])
floor = float(np.percentile(inten, 1))
tA = time.time() - t0
cred = pd.read_csv(f"{ROOT}/data/truth/PR002464/{study}/yeast_cred/credentialed.tsv", sep="\t")
print("clusters", cid_sorted[-1] + 1, "seeds", len(seeds), "seeds>=10x floor", int((seeds.h >= 10 * floor).sum()), "credentialed", len(cred), f"{tA:.1f}s", flush=True)

# ---- method B: peak3d ----
t0 = time.time()
P = estimate(cloud)
b = pick_cloud(cloud, P)
feats, rej = build_features(cloud, b, None, keep_rejected=True)
tB = time.time() - t0
sig = K.per_point_sigma(cloud.inten, P.sig_a, P.sig_b)
nbr = K.find_neighbors(cloud.off, mz, cloud.inten, sig, P.K, P.tol_max_ppm, P.same_scan_k, P.same_scan_floor_ppm, P.link_da)
eff = K.effective_intensity(nbr, cloud.inten)
parent = K.link_steepest(nbr, eff, cloud.inten)
lab0, roots0 = K.dense_labels(K.resolve_roots(parent))
lab = b.lab
print("peak3d: initial basins", b.n_basins_initial, "after merge", b.n_after_merge, "candidates", len(b.basin_id),
      "features", len(feats), "rejected rows", 0 if rej is None else len(rej), f"{tB:.1f}s", flush=True)

def basin_of(fr):
    """merged basin of a feature table row: the centroid at its apex scan closest in m/z"""
    out = []
    for sa, m in zip(fr["scan_apex"].astype(int), fr["mz"]):
        a, z = cloud.off[sa], cloud.off[sa + 1]
        k = a + int(np.argmin(np.abs(mz[a:z] - m))) if z > a else -1
        out.append(int(lab[k]) if k >= 0 else -1)
    return out
feats = feats.copy(); feats["basin"] = basin_of(feats); feats["kept"] = True
rj = rej.copy() if rej is not None and len(rej) else feats.iloc[:0].copy()
if len(rj):
    rj["basin"] = basin_of(rj); rj["kept"] = False
allf = pd.concat([feats, rj], ignore_index=True)
reason_col = next((c for c in ("reject_reason", "reason", "rejected_by", "flags") if c in rj.columns), None)

# match credentialed truth to peak3d features (5 ppm, 0.05 min)
fm, fr_ = feats["mz"].values, feats["rt_apex"].values
hit = [bool(np.any((np.abs(fm - m) <= m * 5e-6) & (np.abs(fr_ - r) <= 0.05))) for m, r in zip(cred.mz, cred.rt)]
print("credentialed matched by a peak3d feature:", int(np.sum(hit)), "of", len(cred), flush=True)

# ---- choose example windows ----
seeds_big = seeds[seeds.h >= 10 * floor].reset_index(drop=True)
seed_cluster = []
srt = np.sort(mz[o])
for m in seeds_big.mz:
    seed_cluster.append(int(cid_sorted[min(np.searchsorted(srt, m), len(srt) - 1)]))
seeds_big["cluster"] = seed_cluster
fk = feats[(feats.height > 3e4)].copy()
fk["cluster"] = [int(cluster[cloud.off[sa] + int(np.argmin(np.abs(mz[cloud.off[sa]:cloud.off[sa + 1]] - m)))])
                 for sa, m in zip(fk.scan_apex.astype(int), fk.mz)]
windows = []
# (1) agree: credentialed tier A, matched, isolated
c1 = cred[(cred.tier == "A") & np.array(hit)].sort_values("height12", ascending=False)
known = {106.0499: "serine [M+H]+", 148.0604: "glutamate [M+H]+", 147.0764: "glutamine [M+H]+"}
pick1 = c1[np.isclose(c1.mz, 148.0604, atol=0.002)].head(1)
if not len(pick1):
    pick1 = c1.head(1)
q = pick1.iloc[0]
windows.append(dict(key="agree", title="Both methods agree", mz=float(q.mz), rt=float(q.rt), dmz_ppm=25, drt=0.45))
# (2) close ions: a gap cluster holding >= 2 strong peak3d features > 4 ppm apart and overlapping in RT,
# neither of them a sideband (within 0.05 Da of a >= 20x stronger feature at the same RT)
allk = feats[["mz", "rt_apex", "height"]].values
def sideband(q):
    near = (np.abs(allk[:, 0] - q.mz) <= 0.05) & (np.abs(allk[:, 1] - q.rt_apex) <= 0.05) & (allk[:, 2] >= 20 * q.height)
    return bool(near.any())
side = None
for c, g in fk.groupby("cluster"):
    if len(g) >= 2 and side is None:
        g2 = g.sort_values("height", ascending=False)
        for i in range(len(g2)):
            for j in range(i + 1, len(g2)):
                a_, b_ = g2.iloc[i], g2.iloc[j]
                dppm = abs(a_.mz - b_.mz) / a_.mz * 1e6
                if dppm > 4 and abs(a_.rt_apex - b_.rt_apex) < 0.15 and min(a_.height, b_.height) > 5e4 and sideband(a_) and sideband(b_):
                    side = (a_, b_, dppm)
                    break
            if side is not None:
                break
best = None
for c, g in fk.groupby("cluster"):
    if len(g) < 2:
        continue
    g = g.sort_values("height", ascending=False)
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a_, b_ = g.iloc[i], g.iloc[j]
            dppm = abs(a_.mz - b_.mz) / a_.mz * 1e6
            if dppm > 4 and abs(a_.rt_apex - b_.rt_apex) < 0.15 and min(a_.height, b_.height) > 2e4 \
                    and not sideband(a_) and not sideband(b_):
                sc = min(a_.height, b_.height)
                if best is None or sc > best[0]:
                    best = (sc, a_, b_, dppm)
if best:
    _, a_, b_, dppm = best
    windows.append(dict(key="close", title=f"Two ions {dppm:.1f} ppm apart", mz=float((a_.mz + b_.mz) / 2),
                        rt=float((a_.rt_apex + b_.rt_apex) / 2), dmz_ppm=max(18, dppm * 2.2), drt=0.45))
if side is not None:
    a_, b_, dppm = side
    windows.append(dict(key="side", title="Sidebands of an intense ion", mz=float((a_.mz + b_.mz) / 2),
                        rt=float((a_.rt_apex + b_.rt_apex) / 2), dmz_ppm=max(25, dppm * 2.5), drt=0.45))
# (3) split differently: a strong peak3d feature containing >= 2 seeds, or a seed span with >= 2 features
best = None
for _, q in fk.sort_values("height", ascending=False).head(3000).iterrows():
    s_in = seeds_big[(np.abs(seeds_big.mz - q.mz) <= q.mz * 5e-6) & (seeds_big.rt >= q.rt_min) & (seeds_big.rt <= q.rt_max)]
    if len(s_in) >= 2 and q.height > 5e4:
        sc = q.height
        if best is None or sc > best[0]:
            best = (sc, q, len(s_in))
if best:
    _, q, ns = best
    windows.append(dict(key="split", title=f"One peak3d peak, {ns} seeds", mz=float(q.mz), rt=float(q.rt_apex),
                        dmz_ppm=18, drt=max(0.4, (q.rt_max - q.rt_min) * 1.2)))
print("windows:", [(w["key"], round(w["mz"], 4), round(w["rt"], 2)) for w in windows], flush=True)

# ---- export ----
def export(w):
    lo, hi = w["mz"] * (1 - w["dmz_ppm"] * 1e-6), w["mz"] * (1 + w["dmz_ppm"] * 1e-6)
    t_lo, t_hi = w["rt"] - w["drt"], w["rt"] + w["drt"]
    s_lo, s_hi = int(np.searchsorted(rt, t_lo)), int(np.searchsorted(rt, t_hi))
    idx = np.flatnonzero((scan >= s_lo) & (scan < s_hi) & (mz >= lo) & (mz <= hi))
    pos = {int(k): i for i, k in enumerate(idx)}
    cl = {int(c): i for i, c in enumerate(dict.fromkeys(cluster[idx].tolist()))}
    b0 = {int(c): i for i, c in enumerate(dict.fromkeys(lab0[idx].tolist()))}
    b1 = {int(c): i for i, c in enumerate(dict.fromkeys(lab[idx].tolist()))}
    pts = [[round(float(rt[scan[k]]), 4), round(float(mz[k]), 5), round(float(inten[k]), 0), cl[int(cluster[k])],
            b0[int(lab0[k])], b1[int(lab[k])], pos.get(int(parent[k]), -1) if parent[k] != k else -1] for k in idx]
    # full-run EIC of each gap cluster in the window, cut to the window RT range
    eics = []
    for c_glob, c_loc in cl.items():
        sel = cluster == c_glob
        e = np.zeros(S); np.maximum.at(e, scan[sel], inten[sel])
        span = mz[sel]
        eics.append(dict(c=c_loc, mz_lo=round(float(span.min()), 5), mz_hi=round(float(span.max()), 5),
                         n=int(sel.sum()), rt=[round(float(x), 4) for x in rt[s_lo:s_hi]],
                         y=[round(float(x), 0) for x in e[s_lo:s_hi]]))
    sd = seeds[(seeds.mz >= lo) & (seeds.mz <= hi) & (seeds.rt >= t_lo) & (seeds.rt <= t_hi)]
    sd_rows = []
    for _, s in sd.iterrows():
        cm = cred[(np.abs(cred.mz - s.mz) <= s.mz * 5e-6) & (np.abs(cred.rt - s.rt) <= 0.05)]
        sd_rows.append(dict(mz=round(s.mz, 5), rt=round(s.rt, 4), h=round(s.h, 0), big=bool(s.h >= 10 * floor),
                            cluster=cl.get(int(cluster[o][min(np.searchsorted(mz[o], s.mz), N - 1)]), -1),
                            cred=(cm.iloc[0].tier if len(cm) else ""), n=(int(cm.iloc[0].n) if len(cm) else 0)))
    fw = allf[(allf.mz >= lo) & (allf.mz <= hi) & (allf.rt_apex >= t_lo) & (allf.rt_apex <= t_hi)]
    f_rows = [dict(mz=round(r.mz, 5), rt=round(r.rt_apex, 4), rt_min=round(r.rt_min, 4), rt_max=round(r.rt_max, 4),
                   h=round(r.height, 0), kept=bool(r.kept), basin=b1.get(int(r.basin), -1), score=round(float(r.score), 2) if pd.notna(r.score) else None,
                   snr=round(float(r.snr), 1) if pd.notna(r.snr) else None,
                   why=(str(getattr(r, reason_col)) if reason_col and not r.kept and pd.notna(getattr(r, reason_col)) else ""))
              for r in fw.itertuples()]
    return dict(**w, mz_lo=lo, mz_hi=hi, rt_lo=t_lo, rt_hi=t_hi, points=pts, eics=eics, seeds=sd_rows, features=f_rows,
                n_clusters=len(cl), n_basins0=len(b0), n_basins=len(b1))

out = dict(file=row.file, study=study, vial=23, n_centroids=int(N), n_scans=int(S), floor=floor,
           A=dict(n_clusters=int(cid_sorted[-1] + 1), n_seeds=int(len(seeds)), n_seeds_big=int((seeds.h >= 10 * floor).sum()),
                  n_cred=int(len(cred)), n_cred_A=int((cred.tier == "A").sum()), seconds=round(tA, 1), cluster_ppm=ct.CLUSTER_PPM),
           B=dict(n_initial=int(b.n_basins_initial), n_merged=int(b.n_after_merge), n_candidates=int(len(b.basin_id)),
                  n_features=int(len(feats)), seconds=round(tB, 1), tol_max_ppm=float(P.tol_max_ppm),
                  fwhm_s=round(float(P.fwhm_med) * 60, 2)),
           cred_matched=int(np.sum(hit)), reason_col=reason_col,
           windows=[export(w) for w in windows])
json.dump(out, open(sys.argv[1], "w"))
print("wrote", sys.argv[1], "windows", [(w["key"], len(w["points"])) for w in out["windows"]], flush=True)
