# data for the EIC failure review page: every labeled EIC of the MassCube benchmark that a peak3d variant gets wrong
# under the paper's rule (bench/score.py score_mc_eic), with the raw and gain-corrected EIC, the variant's reported
# features (apex, left and right edge), the candidates it rejected and why, and MassCube 1.0's features (the labels'
# source). Rejected candidates come from an in-process re-pick with the variant's own settings
# (manifest/peak3d_variants.tsv), checked against the variant's own table. MassCube 1.0's features carry their extent
# (first to last RT, from our MassCube 1.0.22 rerun, which reproduces the published features: masscube_version_runs.py
# Ae). Each label also carries its count in a reference variant, so the page can tell failures the variant fixed, kept
# and introduced. Basins at the label that the picker drops before measuring them (fewer centroids or scans than its
# pre-filter, pick.py) are listed with the rejected candidates, so a missed label never shows nothing.
# usage: eic_failure_data.py VARIANT OUT_JSON [REF]   (VARIANT, REF: results/peak3d_<name>; REF default G3)
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import RUNS  # noqa: E402
from peak3d import features as F  # noqa: E402
from peak3d import kernels as K  # noqa: E402
from peak3d.estimate import apply_scan_gains, estimate, parse_overrides, refit_after_gains, scan_gains  # noqa: E402
from peak3d.features import build_features  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from peak3d.pick import pick_cloud  # noqa: E402

variant, out = sys.argv[1], Path(sys.argv[2])
ref = sys.argv[3] if len(sys.argv) > 3 else "G3"
SETS = {r["variant"]: r["sets"] for r in csv.DictReader((ROOT / "manifest/peak3d_variants.tsv").open(), delimiter="\t")}
overrides = parse_overrides(SETS[variant].split())
HALF = 0.6          # minutes of EIC shown on each side of the label
EIC_PPM = 10.0      # EIC extraction window for display
NEAR_PPM = 20.0     # features this close in m/z are listed (counted or not), and always those inside 1.5 x the scoring tolerance
SHORT = {"NIST_Plasma_Orbitrap_RP_positive": "NIST plasma · RP+ · Orbitrap",
         "NIST_Plasma_Orbitrap_RP_negative": "NIST plasma · RP− · Orbitrap",
         "Mouse_Plasma_Orbitrap_HILIC_positive": "Mouse plasma · HILIC+ · Orbitrap",
         "NIST_Feces_Orbitrap_HILIC_negative": "NIST feces · HILIC− · Orbitrap",
         "Human_Serum_QTOF_RP_positive": "Human serum · RP+ · QTOF",
         "Mouse_Feces_QTOF_HILIC_positive": "Mouse feces · HILIC+ · QTOF",
         "Fruit_Fly_QTOF_HILIC_positive": "Fruit fly · HILIC+ · QTOF",
         "Human_Urine_QTOF_HILIC_negative": "Human urine · HILIC− · QTOF"}

lab = pd.read_csv(ROOT / "data/truth/MASSCUBE/eic_labels.tsv", sep="\t")
pub = pd.read_csv(ROOT / "data/truth/MASSCUBE/published_features.tsv", sep="\t")
pub = pub[pub.tool == "masscube"]
lab["tol"] = np.where(lab.n_peaks == 1, 0.002, 0.005)


def aligned(v, f):
    return pd.read_csv(ROOT / f"results/peak3d_{v}/MC_{f}/peak3d_out/aligned_feature_table.tsv", sep="\t",
                       usecols=["group_id", "mz", "rt", "rt_min", "rt_max"])


def counts(A, sub):
    return np.array([int(((np.abs(A.mz.values - r.mz) < r.tol) & (np.abs(A.rt.values - r.rt) < 0.2)).sum())
                     for r in sub.itertuples()])


def gate_reason(r, P):
    fails = []
    need = int(r["min_scans_local"]) if (P.width_rule == "time" or P.min_scans_rule == "local") else P.min_scans
    if r["n_scans"] < need: fails.append(f"too few scans ({int(r['n_scans'])} < {need})")
    if r["snr"] < P.min_snr: fails.append("S/N")
    if r["mz_sd_ppm"] > 3 * r["sig_apex"] + 1: fails.append("m/z scatter")
    slack = float(r.get("dt_local", 0.0)) if P.width_rule == "time" else 0.0
    nthr = float(r.get("narrow_thr", np.nan))
    ref = nthr if np.isfinite(nthr) else 0.3 * P.fwhm_med
    if P.width_rule in ("fwhm", "time") and P.narrow_rule != "off" and not (r["fwhm"] + slack >= ref): fails.append("too narrow")
    if r["prominence_rel"] < P.min_prominence_rel: fails.append("prominence")
    if r.get("ridge_ratio", 0) > F.RIDGE_MAX: fails.append("ridge (ion present far out)")
    if r.get("background_ratio", 0) > F.BACKGROUND_MAX: fails.append("background")
    if P.far_level_max < 1 and r.get("far_level", 0) > P.far_level_max: fails.append("far level")
    if r.get("fragment", False): fails.append("fragment of a neighbour")
    if "too narrow" in fails:
        what = (f"0.3 x local median = {ref * 60:.1f} s" if np.isfinite(nthr) else f"0.3 x file median {P.fwhm_med * 60:.1f} s")
        fails[fails.index("too narrow")] = f"too narrow (FWHM {r['fwhm'] * 60:.1f} s{' + one scan' if slack else ''} < {what})"
    return ", ".join(fails) or "edge of a wider trace"


def reason(r, P):
    x = str(r["reason"])
    if x == "gate":
        return gate_reason(r, P)
    if x.startswith("eic:"):
        return {"eic:lowsnr": "chromatogram check: too weak vs noise", "eic:noapex": "chromatogram check: no apex of its own",
                "eic:dup": "chromatogram check: duplicate of a stronger peak"}.get(x, x)
    return "quality score below threshold" if x == "score" else x


cases = []
tally = {"fixed": 0, "newly_wrong": 0}
for f in lab.file.unique():
    sub = lab[lab.file == f].copy()
    A = aligned(variant, f)
    B = aligned(ref, f) if (ROOT / f"results/peak3d_{ref}/MC_{f}").exists() else None
    sub["count"] = counts(A, sub)
    sub["count_ref"] = counts(B, sub) if B is not None else -1
    for r in sub.itertuples():
        if r.count_ref == r.n_peaks and r.count != r.n_peaks:
            tally["newly_wrong"] += 1
        elif r.count_ref != r.n_peaks and r.count == r.n_peaks:
            tally["fixed"] += 1
    bad = sub[sub["count"] != sub.n_peaks]
    if bad.empty:
        continue
    cloud = load_cloud(RUNS[f"MC_{f}"]["files"][0])
    mce = pd.read_csv(ROOT / f"results/diag_double/masscube_versions/Ae/{f}.tsv", sep="\t")
    raw_rt = cloud.rt.copy()
    raw_eic = {r.Index: (lambda s0, s1: (s0, s1))(*np.searchsorted(raw_rt, [r.rt - HALF, r.rt + HALF])) for r in bad.itertuples()}
    raw_tr = {i: cloud.eic(lab.loc[i, "mz"], EIC_PPM, s0, s1).astype(float) for i, (s0, s1) in raw_eic.items()}
    P = estimate(cloud, overrides=overrides)
    g, gsd = np.ones(cloud.n_scans), 0.0
    if P.scan_gain:
        g, gsd, _ = scan_gains(cloud, P)
        apply_scan_gains(cloud, g)
        P = refit_after_gains(cloud, P)
    b = pick_cloud(cloud, P)
    kept, rej = build_features(cloud, b, keep_rejected=True)
    scan_of = cloud.scan_index()
    smin2, smax2, npts2 = K.basin_extent(b.lab, cloud.off, b.n_after_merge)
    measured = np.zeros(b.n_after_merge, bool)
    measured[b.basin_id] = True
    ms_pick = P.min_scans_pick if P.min_scans_rule == "local" else P.min_scans
    print(f"{f}: re-pick {len(kept)} features vs {len(A)} in results/peak3d_{variant} (gain sd {gsd:.3f})", flush=True)
    per = pd.read_csv(ROOT / f"results/peak3d_{variant}/MC_{f}/peak3d_out/features/{f}.tsv", sep="\t",
                      usecols=["group_id", "rt_apex", "height", "score", "flags", "snr", "fwhm"])
    A2 = A.merge(per, on="group_id", how="left")
    for r in bad.itertuples():
        s0, s1 = raw_eic[r.Index]
        lim = max(NEAR_PPM * r.mz * 1e-6, 1.5 * r.tol)
        near = lambda T: T[(np.abs(T.mz - r.mz) <= lim) & (T.rt > r.rt - HALF) & (T.rt < r.rt + HALF)]
        feats = []
        for x in near(A2).itertuples():
            in_mz, in_rt = abs(x.mz - r.mz) < r.tol, abs(x.rt - r.rt) < 0.2
            feats.append(dict(mz=round(x.mz, 5), dppm=round((x.mz - r.mz) / r.mz * 1e6, 2), rt=round(x.rt, 4),
                              lo=round(x.rt_min, 4), hi=round(x.rt_max, 4), height=float(x.height) if pd.notna(x.height) else None,
                              score=round(float(x.score), 3) if pd.notna(x.score) else None,
                              fwhm_s=round(float(x.fwhm) * 60, 2) if pd.notna(x.fwhm) else None,
                              flags=x.flags if isinstance(x.flags, str) else "", counted=bool(in_mz and in_rt),
                              why="" if in_mz and in_rt else ("m/z outside tolerance" if not in_mz else "outside ±0.2 min")))
        rejected = []
        for _, x in near(rej).iterrows():
            rejected.append(dict(mz=round(float(x["mz"]), 5), dppm=round((x["mz"] - r.mz) / r.mz * 1e6, 2),
                                 rt=round(float(x["rt"]), 4), lo=round(float(x["rt_min"]), 4), hi=round(float(x["rt_max"]), 4),
                                 height=float(x["height"]), fwhm_s=round(float(x["fwhm"]) * 60, 2), n_scans=int(x["n_scans"]),
                                 reason=reason(x, P)))
        # basins dropped before measuring: centroids inside the scoring window whose basin (2+ centroids) never became a candidate
        pts = np.flatnonzero((np.abs(cloud.mz - r.mz) < r.tol) & (np.abs(cloud.rt[scan_of] - r.rt) < 0.2))
        for L in np.unique(b.lab[pts]):
            if measured[L] or npts2[L] < 2:
                continue
            q = np.flatnonzero(b.lab == L)
            a = q[np.argmax(cloud.inten[q])]
            span = int(smax2[L] - smin2[L] + 1)
            rejected.append(dict(mz=round(float(cloud.mz[a]), 5), dppm=round((cloud.mz[a] - r.mz) / r.mz * 1e6, 2),
                                 rt=round(float(cloud.rt[scan_of[a]]), 4), lo=round(float(cloud.rt[smin2[L]]), 4),
                                 hi=round(float(cloud.rt[smax2[L]]), 4), height=float(cloud.inten[a]), fwhm_s=None,
                                 n_scans=span, dropped=True,
                                 reason=f"dropped before measuring: {int(npts2[L])} centroids over {span} scans "
                                        f"(the picker measures a basin only with >= {ms_pick} of each)"))
        mc = pub[(pub.file == f) & (np.abs(pub.mz - r.mz) < r.tol) & (np.abs(pub.rt - r.rt) < HALF)]
        mc_rows = []
        for m, t in zip(mc.mz, mc.rt):
            e = mce[(np.abs(mce.mz - m) < 2e-4) & (np.abs(mce.rt - t) < 2e-3)]
            mc_rows.append(dict(mz=round(float(m), 5), rt=round(float(t), 4),
                                lo=round(float(e.rt_lo.iloc[0]), 4) if len(e) else None,
                                hi=round(float(e.rt_hi.iloc[0]), 4) if len(e) else None,
                                n_scans=int(e.length.iloc[0]) if len(e) else None))
        n = int(r.n_peaks)
        cat = "missed" if r.count == 0 else ("merged" if r.count < n else "extra")
        # extra cases: the matched peak(s) = the nearest counted feature to the label (single) or the two highest (double);
        # each other counted feature is "touching" when its edges meet a matched peak's within 1.5 scans
        relation = ""
        cnt = [x for x in feats if x["counted"]]
        if cat == "extra" and len(cnt) > n:
            key = (lambda x: abs(x["rt"] - r.rt)) if n == 1 else (lambda x: -(x["height"] or 0))
            matched = sorted(cnt, key=key)[:n]
            scan = float(np.median(np.diff(raw_rt[s0:s1]))) if s1 - s0 > 1 else 0.0
            kinds = []
            for x in cnt:
                if any(x is m for m in matched):
                    x["role"] = "matched"
                    continue
                gap = min(max(x["lo"] - m["hi"], m["lo"] - x["hi"]) for m in matched)
                x["role"] = "extra, touching" if gap <= 1.5 * scan else "extra, separated"
                x["gap_scans"] = round(gap / scan, 1) if scan else None
                kinds.append(x["role"])
            relation = "touching" if all(k == "extra, touching" for k in kinds) else (
                "separated" if all(k == "extra, separated" for k in kinds) else "mixed")
        cases.append(dict(id=f"{f}:{r.mz}:{r.rt}", file=f, short=SHORT[f], n_peaks=n, mz=r.mz, rt=r.rt, tol=r.tol,
                          count=int(r.count), count_ref=int(r.count_ref), ref_ok=bool(r.count_ref == n), cat=cat,
                          relation=relation,
                          fwhm_med_s=round(P.fwhm_med * 60, 2),
                          eic=dict(rt=[round(v, 4) for v in raw_rt[s0:s1]], raw=[round(v, 1) for v in raw_tr[r.Index]],
                                   cor=[round(v, 1) for v in raw_tr[r.Index] / g[s0:s1]]),
                          feats=feats, rejected=rejected,
                          mc10=mc_rows))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dict(variant=variant, sets=SETS[variant], ref=ref, ref_sets=SETS.get(ref, ""), scan_gain=bool(P.scan_gain),
                               width_rule=P.width_rule, link_da=P.link_da, eic_ppm=EIC_PPM, half=HALF, cases=cases, **tally),
                          separators=(",", ":")))
print(f"vs {ref}: {tally['fixed']} labels fixed, {tally['newly_wrong']} newly wrong")
print(f"{len(cases)} failing labels -> {out} ({out.stat().st_size / 1e3:.0f} kB)")
print(pd.Series([c['cat'] + ' (' + ('single' if c['n_peaks'] == 1 else 'double') + ')' for c in cases]).value_counts().to_string())
print(pd.Series([c['relation'] + ' / ' + ('single' if c['n_peaks'] == 1 else 'double') for c in cases if c['cat'] == 'extra']).value_counts().to_string())
nar = [(c, x) for c in cases for x in c["rejected"] if "too narrow" in x["reason"]]
print(f"rejected 'too narrow' candidates near failing labels: {len(nar)} in {len(set(c['id'] for c, _ in nar))} cases;"
      f" at the label (|dRT| < 0.1 min, |dm/z| < tol) in missed cases: "
      f"{sum(1 for c, x in nar if c['cat'] == 'missed' and abs(x['rt'] - c['rt']) < 0.1 and abs(x['mz'] - c['mz']) < c['tol'])}")
for c, x in nar:
    print(f"  {c['cat']:6s} {c['short'][:28]:28s} label {c['mz']:.4f} @ {c['rt']:.3f} | cand @ {x['rt']:.3f} fwhm {x['fwhm_s']} s"
          f" (file median {c['fwhm_med_s']} s) n_scans {x['n_scans']} height {x['height']:.3g} | {x['reason']}")
