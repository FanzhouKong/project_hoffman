# diagnostic: HZV029 certified features that peak3d misses (no aligned group within 10 ppm / 0.1 min)
# and those a presence filter (detected in >= 2 of 3 files) would drop. For each: raw EIC in the three
# files, peak3d's kept features and rejected candidates nearby (re-picked in memory with the same code,
# nothing written to results/peak3d), the gate that rejected each candidate, nearby aligned groups and
# other tools' features. Writes results/diag_excess/hzv029_missed.json for the review page.
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runs import LOCAL_TOOLS, RUNS  # noqa: E402
from bench.score import load  # noqa: E402
from peak3d import kernels as K  # noqa: E402
from peak3d.features import BACKGROUND_MAX, RIDGE_MAX, build_features  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from peak3d.pick import pick_cloud  # noqa: E402

RUN = "HZV029_cert"
PPM, RT_TOL = 10.0, 0.1            # benchmark matching
NEAR_PPM, NEAR_RT = 20.0, 0.5      # what to show around each truth feature
EIC_PPM, EIC_HALF = 5.0, 0.6
D = ROOT / "results/peak3d" / RUN / "peak3d_out"


def gate_reasons(r, P):
    out = []
    if r["n_scans"] < P.min_scans:
        out.append(f"n_scans {int(r['n_scans'])} < {P.min_scans}")
    if r["snr"] < P.min_snr:
        out.append(f"snr {r['snr']:.1f} < {P.min_snr:g}")
    if r["mz_sd_ppm"] > 3.0 * r["sig_apex"] + 1.0:
        out.append(f"m/z sd {r['mz_sd_ppm']:.1f} ppm > {3 * r['sig_apex'] + 1:.1f}")
    if r["fwhm"] < 0.3 * P.fwhm_med:
        out.append(f"fwhm {60 * r['fwhm']:.2f} s < 0.3 median")
    if r["prominence_rel"] < P.min_prominence_rel:
        out.append(f"prominence {r['prominence_rel']:.2f} < {P.min_prominence_rel:g}")
    if r.get("fragment", False):
        out.append("fragment (higher centroid beyond its bound)")
    kf = int(r["kflags"])
    if (kf & K.KF_CAP_L) and (kf & K.KF_CAP_R) and r["prominence_rel"] < 0.5:
        out.append("capped both sides, prominence < 0.5")
    if r.get("ridge_ratio", 0) > RIDGE_MAX:
        out.append(f"ridge {r['ridge_ratio']:.2f} > {RIDGE_MAX:g}")
    if r.get("background_ratio", 0) > BACKGROUND_MAX:
        out.append(f"background {r['background_ratio']:.2f} > {BACKGROUND_MAX:.2f}")
    if not out and r["reason"] == "score":
        out.append(f"score {r['score_rule']:.2f} < {P.min_score:g}")
    return out


def near(df, mz, rt, ppm=NEAR_PPM, rt_win=NEAR_RT, rt_col="rt"):
    x = df[(np.abs(df["mz"] - mz) / mz * 1e6 <= ppm) & (np.abs(df[rt_col] - rt) <= rt_win)].copy()
    x["dppm"] = (x["mz"] - mz) / mz * 1e6
    x["drt_s"] = 60 * (x[rt_col] - rt)
    return x


def main():
    p = next((ROOT / "data/raw/ASARI_DATA/x").glob("*/data/hzv029_manual_certified.txt"))
    tr = pd.read_csv(p, sep="\t").reset_index().rename(columns={"index": "truth_id"})
    chk = pd.read_csv(ROOT / "results/hzv029_cert_check/hzv029_cert_check.tsv", sep="\t")
    t = pd.read_csv(D / "aligned_feature_table.tsv", sep="\t")
    files = RUNS[RUN]["files"]
    stems = [Path(f).stem for f in files]
    m = pd.read_csv(D / "filled_mask.tsv", sep="\t").set_index("group_id").loc[t["group_id"]]
    det = (m.values == 0) & (t[stems].values > 0)
    t["n_det"] = det.sum(1)
    # benchmark match and presence-filter match
    sets = {}
    for i, (mz, rt) in enumerate(zip(tr["moverz"], tr["RT_minutes"])):
        k = np.flatnonzero((np.abs(t["mz"].values - mz) / mz * 1e6 <= PPM) & (np.abs(t["rt"].values - rt) <= RT_TOL))
        if len(k) == 0:
            sets[i] = "missed"
        elif not (t["n_det"].values[k] >= 2).any():
            sets[i] = "presence"
    print({s: sum(v == s for v in sets.values()) for s in ("missed", "presence")})
    # re-pick in memory with keep_rejected
    clouds, kept, rej, params, warps = {}, {}, {}, {}, {}
    for f, s in zip(files, stems):
        c = load_cloud(f)
        b = pick_cloud(c)
        fe, rj = build_features(c, b, None, True)
        clouds[s], params[s] = c, b.params
        kept[s] = pd.read_csv(D / "features" / f"{s}.tsv", sep="\t")   # the benchmark's own kept table
        rj = rj.copy()
        rj["sig_apex"] = b.params.sig_ppm(rj["height"].values)
        rej[s] = rj
        warps[s] = pd.read_csv(D / "rt_correction" / f"{s}.tsv", sep="\t")
        print(s, "re-pick kept", len(fe), "vs benchmark", len(kept[s]), "rejected", len(rj), flush=True)
    others = {k: load(k, RUN) for k in ("asari", "masscube", *LOCAL_TOOLS)}
    items = []
    for i, cat in sets.items():
        r = tr.iloc[i]
        mz, rt = float(r["moverz"]), float(r["RT_minutes"])
        it = dict(truth_id=int(i), category=cat, mz=mz, rt=rt, intensity=float(r["Intensity"]), files=[])
        cr = chk.iloc[i]
        g = near(t, mz, rt)
        it["groups"] = [dict(group_id=int(x.group_id), mz=float(x.mz), rt=float(x.rt), rt_min=float(x.rt_min),
                             rt_max=float(x.rt_max), dppm=float(x.dppm), drt_s=float(x.drt_s), n_det=int(x.n_det),
                             heights=[float(x[s]) for s in stems],
                             detected=[bool(det[x.name, j]) for j in range(len(stems))])
                        for _, x in g.sort_values("drt_s", key=np.abs).iterrows()]
        it["others"] = {}
        for k, v in others.items():
            if v is None:
                continue
            x = near(v[0], mz, rt, ppm=NEAR_PPM, rt_win=NEAR_RT)
            it["others"][k] = [dict(mz=float(a), rt=float(b_), dppm=float(c_), drt_s=float(d_),
                                    hit=bool(abs(c_) <= PPM and abs(d_) <= 60 * RT_TOL))
                               for a, b_, c_, d_ in zip(x["mz"], x["rt"], x["dppm"], x["drt_s"])]
        for j, s in enumerate(stems):
            c = clouds[s]
            tag = s.split("_")[-1][-1]
            # truth RT is on the consensus axis; map to this file's native axis for the EIC
            w = warps[s]
            rt_nat = float(np.interp(rt, w["rt_corr"].values, w["rt_native"].values))
            s0, s1 = int(np.searchsorted(c.rt, rt_nat - EIC_HALF)), int(np.searchsorted(c.rt, rt_nat + EIC_HALF))
            e = c.eic(mz, EIC_PPM, s0, s1)
            e10 = c.eic(mz, 15.0, s0, s1)
            kf = near(kept[s], mz, rt, rt_col="rt_corr")
            rf = near(rej[s], mz, rt_nat, rt_col="rt")
            fd = dict(stem=s, tag=tag, rt_native_of_truth=rt_nat,
                      raw_check=dict(present=bool(cr[f"003{tag}_present"]), apex=float(cr[f"003{tag}_apex"]),
                                     snr=float(cr[f"003{tag}_snr"]), nscan=int(cr[f"003{tag}_nscan"]),
                                     drt=None if pd.isna(cr[f"003{tag}_dRT"]) else float(cr[f"003{tag}_dRT"])),
                      rt=[round(float(v), 5) for v in c.rt[s0:s1]], eic=[round(float(v), 1) for v in e],
                      eic15=[round(float(v), 1) for v in e10],
                      noise=float(params[s].floor),
                      kept=[dict(mz=float(x.mz), rt=float(x.rt), rt_corr=float(x.rt_corr), rt_min=float(x.rt_min),
                                 rt_max=float(x.rt_max), height=float(x.height), n_scans=int(x.n_scans),
                                 fwhm_s=float(60 * x.fwhm), snr=float(x.snr), score=float(x.score),
                                 flags=x.flags if isinstance(x.flags, str) else "", group_id=int(x.group_id),
                                 dppm=float(x.dppm), drt_s=float(x.drt_s))
                            for x in kf.itertuples()],
                      rejected=[dict(mz=float(x["mz"]), rt=float(x["rt"]), rt_min=float(x["rt_min"]),
                                     rt_max=float(x["rt_max"]), height=float(x["height"]), n_scans=int(x["n_scans"]),
                                     fwhm_s=float(60 * x["fwhm"]), snr=float(x["snr"]), dppm=float(x["dppm"]),
                                     drt_s=float(x["drt_s"]), stage=x["reason"], why=gate_reasons(x, params[s]))
                                for _, x in rf.iterrows()])
            it["files"].append(fd)
        items.append(it)
    out = ROOT / "results/diag_excess/hzv029_missed.json"
    out.write_text(json.dumps(dict(run=RUN, ppm=PPM, rt_tol=RT_TOL, stems=stems, items=items)))
    print(out, len(items))
    # compact text summary
    for it in items:
        print(f"\n[{it['category']}] #{it['truth_id']} m/z {it['mz']:.5f} @ {it['rt']:.3f} min  I {it['intensity']:.3g}")
        for g in it["groups"][:4]:
            print(f"   group {g['group_id']} {g['mz']:.5f} @ {g['rt']:.3f} dppm {g['dppm']:+.1f} drt {g['drt_s']:+.1f}s n_det {g['n_det']} det {g['detected']}")
        for fd in it["files"]:
            rc = fd["raw_check"]
            print(f"   {fd['tag']}: raw present {rc['present']} apex {rc['apex']:.3g} snr {rc['snr']:.1f} dRT {rc['drt']}; "
                  f"kept {[(round(k['dppm'], 1), round(k['drt_s'], 1), k['group_id'], k['flags']) for k in fd['kept']]}; "
                  f"rejected {[(round(k['dppm'], 1), round(k['drt_s'], 1), k['stage'], k['why']) for k in fd['rejected']]}")
        for k, v in it["others"].items():
            print(f"   {k}: {[(round(a['dppm'], 1), round(a['drt_s'], 1), a['hit']) for a in v]}")


if __name__ == "__main__":
    main()
