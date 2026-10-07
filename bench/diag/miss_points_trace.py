# diagnostic: for every missed truth feature of yeast_truth_misses.py / idsl_tp_misses.py, where did the centroids go?
# In one file, the centroids within +/- 40 ppm x +/- 0.15 min of the truth position are grouped into m/z strands (gap
# > 6 ppm), the strand nearest the truth m/z is the truth ion; its points are followed to their final basins (how many,
# which stage they ended in, apex offsets, m/z of the basin, points per scan > 1 = the basin holds a second strand).
# Also the raw prominence of the truth ion's EIC (apex above the deeper side minimum within 2 FWHM) and, for YEAST,
# whether a near-miss is a second truth entry of a peak the tool already found.
# usage: miss_points_trace.py YEAST_12C | IDSL003
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS  # noqa: E402
from bench.score import hits  # noqa: E402
from miss_common import repick, to_native  # noqa: E402

RUN = sys.argv[1]
WIN_PPM, WIN_RT, GAP_PPM = 40.0, 0.15, 6.0
if RUN == "YEAST_12C":
    M = pd.read_csv(ROOT / "results/diag_yeast/yeast_misses.tsv", sep="\t")
    T = pd.read_csv(ROOT / "data/truth/YEAST/credentialed.tsv", sep="\t")
    OUT = ROOT / "results/diag_yeast"
    ID = "truth_id"
else:
    M = pd.read_csv(ROOT / "results/diag_idsl_tp/idsl_tp_misses.tsv", sep="\t")
    T = None
    OUT = ROOT / "results/diag_idsl_tp"
    ID = "label_id"
D = ROOT / "results/peak3d" / RUN / "peak3d_out"
f = RUNS[RUN]["files"][0]
stem = Path(f).stem
c, P, b, fe, rj = repick(f)
scan_of = c.scan_index()
lab = b.lab
cand = set(int(x) for x in b.basin_id)
rej_by_basin = {int(r.basin_id): r for r in rj.itertuples()}
# basin stats of every candidate (for apex / m/z of the basin a point ended in)
bs = pd.DataFrame({"basin_id": b.basin_id})
st = b.stats
from peak3d import kernels as K  # noqa: E402
for col in ("mz", "rt", "height", "scan_apex", "n_points", "n_scans", "mz_sd_ppm", "mz_min", "mz_max"):
    bs[col] = st[:, K.C[col]] if col != "n_scans" else st[:, K.C["scan_hi"]] - st[:, K.C["scan_lo"]] + 1 - st[:, K.C["n_gaps"]]
bs = bs.set_index("basin_id")
W = pd.read_csv(D / "rt_correction" / f"{stem}.tsv", sep="\t") if (D / "rt_correction" / f"{stem}.tsv").exists() else None
fwhm_scans = P.fwhm_scans
print(RUN, stem, "candidates", len(cand), "rejected", len(rj), "kept", len(fe), flush=True)

if T is not None:
    A = pd.read_csv(D / "aligned_feature_table.tsv", sep="\t")
    T["hit"] = hits(A[["mz", "rt"]], T["mz"].values, T["rt"].values, 10.0, 0.1)
    hit_mz, hit_rt = T.loc[T["hit"], "mz"].values, T.loc[T["hit"], "rt"].values

rows = []
for r in M.itertuples():
    mz, rt = float(r.mz), float(r.rt)
    rt_nat = to_native(W, rt) if W is not None else rt
    s0 = int(max(0, np.searchsorted(c.rt, rt_nat - WIN_RT)))
    s1 = int(min(c.n_scans, np.searchsorted(c.rt, rt_nat + WIN_RT)))
    p0, p1 = int(c.off[s0]), int(c.off[s1])
    seg_mz = c.mz[p0:p1]
    tol = mz * WIN_PPM * 1e-6
    sel = np.flatnonzero(np.abs(seg_mz - mz) <= tol) + p0
    row = {ID: getattr(r, ID), "cls": r.cls, "sub": r.sub, "n_pts_40ppm": int(len(sel))}
    if len(sel) == 0:
        rows.append(row)
        continue
    ppm = (c.mz[sel] - mz) / mz * 1e6
    sc = scan_of[sel]
    I = c.inten[sel].astype(np.float64)
    # strands: sort by ppm, split at gaps > GAP_PPM
    o = np.argsort(ppm)
    cut = np.flatnonzero(np.diff(ppm[o]) > GAP_PPM)
    strand = np.zeros(len(sel), int)
    sid = np.zeros(len(sel), int)
    bounds = np.concatenate([[0], cut + 1, [len(o)]])
    for k in range(len(bounds) - 1):
        sid[o[bounds[k]:bounds[k + 1]]] = k
    n_strands = len(bounds) - 1
    # the strand whose intensity-weighted ppm is nearest 0 (within +/- 10 ppm preferred)
    cen = np.array([np.average(ppm[sid == k], weights=I[sid == k]) for k in range(n_strands)])
    k0 = int(np.argmin(np.abs(cen)))
    m = sid == k0
    row.update(n_strands=n_strands, strand_ppm=float(cen[k0]), strand_ppm_sd=float(np.std(ppm[m])),
               strand_n_scans=int(len(np.unique(sc[m]))), strand_apex=float(I[m].max()),
               strand_apex_drt_s=float(60 * (c.rt[sc[m][np.argmax(I[m])]] - rt_nat)),
               other_strand_ppm=float(cen[np.argsort(np.abs(cen))[1]]) if n_strands > 1 else np.nan,
               other_strand_apex=float(I[sid == int(np.argsort(np.abs(cen))[1])].max()) if n_strands > 1 else np.nan,
               max_pts_per_scan=int(np.bincount(sc[m] - sc[m].min()).max()))
    # raw prominence of the truth strand: per-scan max within +/- WIN_RT; apex within +/- 0.1 min of rt_nat
    e = np.zeros(s1 - s0)
    np.maximum.at(e, sc[m] - s0, I[m])
    rr = c.rt[s0:s1]
    win = np.abs(rr - rt_nat) <= 0.1
    if win.any() and e[win].max() > 0:
        i = int(np.flatnonzero(win)[np.argmax(e[win])])
        h = e[i]
        half = int(max(2, round(2 * fwhm_scans)))
        left = e[max(0, i - half):i]
        right = e[i + 1:i + 1 + half]
        lmin = left.min() if len(left) else 0.0
        rmin = right.min() if len(right) else 0.0
        prom = h - max(lmin, rmin)
        row.update(raw_h=float(h), raw_prom=float(prom), raw_prom_rel=float(prom / h),
                   raw_prom_over_cell=float(prom / max(float(bs["height"].median() * 0 + 1), 1.0)))
    # final basins of the truth strand's points
    bl = lab[sel[m]]
    u, cnt = np.unique(bl, return_counts=True)
    order = np.argsort(-cnt)
    u, cnt = u[order], cnt[order]
    dom = int(u[0])
    row.update(strand_n_basins=int(len(u)), dom_share=float(cnt[0] / cnt.sum()), dom_is_candidate=dom in cand)
    if dom in cand:
        x = bs.loc[dom]
        stage = rej_by_basin[dom].stage if dom in rej_by_basin else "kept"
        row.update(dom_stage=stage, dom_mz_ppm=float((x["mz"] - mz) / mz * 1e6), dom_mz_sd_ppm=float(x["mz_sd_ppm"]),
                   dom_mz_span_ppm=float((x["mz_max"] - x["mz_min"]) / mz * 1e6), dom_pts_per_scan=float(x["n_points"] / max(x["n_scans"], 1)),
                   dom_height=float(x["height"]), dom_apex_drt_s=float(60 * (c.rt[int(x["scan_apex"])] - rt_nat)),
                   dom_n_scans=int(x["n_scans"]))
        # does the dominant basin also hold points of another strand?
        allb = lab[sel]
        row["dom_holds_other_strand"] = bool(np.any((allb == dom) & ~m))
    else:
        row.update(dom_stage="not a candidate")
    # second basin, if the strand is split
    if len(u) > 1 and int(u[1]) in cand:
        x = bs.loc[int(u[1])]
        row.update(b2_stage=rej_by_basin[int(u[1])].stage if int(u[1]) in rej_by_basin else "kept",
                   b2_mz_ppm=float((x["mz"] - mz) / mz * 1e6), b2_apex_drt_s=float(60 * (c.rt[int(x["scan_apex"])] - rt_nat)),
                   b2_share=float(cnt[1] / cnt.sum()))
    if T is not None and r.cls == "near":
        dup = np.any((np.abs(hit_mz - mz) / mz * 1e6 <= 10.0) & (np.abs(hit_rt - rt) <= 0.3))
        row["dup_of_hit_truth"] = bool(dup)
    rows.append(row)
R = pd.DataFrame(rows)
R.to_csv(OUT / "points_trace.tsv", sep="\t", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
with open(OUT / "points_trace.log", "w") as fh:
    def w(*a):
        s = " ".join(str(x) for x in a)
        fh.write(s + "\n")
        print(s, flush=True)
    w(f"== {RUN} ({stem}): {len(R)} missed truth positions; fwhm_scans {fwhm_scans:.1f}, sig_a {P.sig_a:.2f} sig_b {P.sig_b:.0f} tol_max {P.tol_max_ppm:.1f} ppm")
    g = R.groupby(["cls", "sub"])
    w(g.agg(n=(ID, "size"), strands=("n_strands", "median"), two_strands=("n_strands", lambda v: (v >= 2).mean()),
            strand_sd_ppm=("strand_ppm_sd", "median"), strand_scans=("strand_n_scans", "median"), n_basins=("strand_n_basins", "median"),
            split=("strand_n_basins", lambda v: (v >= 2).mean()), dom_share=("dom_share", "median"),
            dom_cand=("dom_is_candidate", "mean"), dom_other=("dom_holds_other_strand", "mean"),
            dom_mz_ppm=("dom_mz_ppm", lambda v: np.nanmedian(np.abs(v))), dom_sd=("dom_mz_sd_ppm", "median"),
            dom_span=("dom_mz_span_ppm", "median"), dom_pps=("dom_pts_per_scan", "median"),
            dom_drt=("dom_apex_drt_s", lambda v: np.nanmedian(np.abs(v))), prom_rel=("raw_prom_rel", "median")).round(2).to_string())
    w("\n== dominant basin stage per class ==")
    w(pd.crosstab([R["cls"], R["sub"]], R["dom_stage"].fillna("none")).to_string())
    if "dup_of_hit_truth" in R:
        w("\n== near misses that are a second truth entry (10 ppm / 0.3 min) of a truth feature already found ==")
        w(R[R["cls"] == "near"].groupby("sub")["dup_of_hit_truth"].agg(["size", "mean"]).round(2).to_string())
    w("\n== strands: cases where a second ion sits within 40 ppm, by class ==")
    x = R[R["n_strands"] >= 2]
    w(x.groupby("cls").agg(n=(ID, "size"), other_ppm=("other_strand_ppm", lambda v: np.nanmedian(np.abs(v))),
                           other_over_truth=("other_strand_apex", "median"), truth_apex=("strand_apex", "median")).round(1).to_string())
    w("\n== dominant basin m/z sd vs the gate (3 sig + 1) for the mz_sd-rejected ==")
    y = R[R["sub"] == "gate:mz_sd"]
    if len(y):
        w(y[["dom_mz_sd_ppm", "dom_mz_span_ppm", "dom_pts_per_scan", "dom_holds_other_strand", "n_strands", "strand_ppm_sd", "dom_height"]]
          .describe(percentiles=[0.25, 0.5, 0.75]).round(2).to_string())
    w("\n== examples ==")
    for cls_sub, grp in R.groupby(["cls", "sub"]):
        w(f"-- {cls_sub}")
        for q in grp.head(6).itertuples():
            w(f"   {getattr(q, ID)}: strands {q.n_strands} (truth strand {q.strand_ppm:+.1f} ppm sd {q.strand_ppm_sd:.1f}, {q.strand_n_scans} scans, apex {q.strand_apex:.2g}"
              f" at {q.strand_apex_drt_s:+.0f}s; other {q.other_strand_ppm:+.0f} ppm apex {q.other_strand_apex:.2g}) -> {q.strand_n_basins} basins, dominant {q.dom_share:.2f}"
              f" {q.dom_stage} mz {q.dom_mz_ppm:+.1f} ppm sd {q.dom_mz_sd_ppm:.1f} span {q.dom_mz_span_ppm:.0f} pts/scan {q.dom_pts_per_scan:.2f} apex {q.dom_apex_drt_s:+.0f}s"
              f" other-strand {q.dom_holds_other_strand}; raw prom_rel {q.raw_prom_rel:.2f}")
print("wrote", OUT / "points_trace.tsv")
