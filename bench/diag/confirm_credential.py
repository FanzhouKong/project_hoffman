# diagnostic: decide the cross-file confirmation rule with evidence that does not come from our own thresholds. For the
# groups the presence rule dropped (single detection, no confirmed peak elsewhere), every candidate rule from
# confirm_rules.py is evaluated on the same per-cell measurements; the groups a rule would re-admit are then
# credentialed against the U-13C run (a 13C feature at m/z + n x 1.00335, same RT, 5 ppm / 0.1 min; chance level from
# four decoy spacings), once against the 13C run's kept groups and once against every 13C per-file detection (so a weak
# 12C singleton is not penalised for its partner being a singleton too). The kept 12C groups and all dropped groups are
# credentialed the same two ways for comparison.
# usage: confirm_credential.py YEAST_12C | SZ22_12C   (reads mzML -> srun)
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/diag"))
import bench.score as S  # noqa: E402
from bench.runs import RUNS  # noqa: E402
from fix_eval import near_any  # noqa: E402
from peak3d.estimate import estimate  # noqa: E402
from peak3d.io import load_cloud  # noqa: E402
from peak3d.warp import TableWarp  # noqa: E402

pd.set_option("display.width", 230)
run = sys.argv[1]
ds = "YEAST" if run.startswith("YEAST") else "SZ22"
run13 = run.replace("12C", "13C")
files = RUNS[run]["files"]
stems = [Path(f).stem for f in files]
N = len(files)
d = ROOT / "results/peak3d" / run / "peak3d_out"
d13 = ROOT / "results/peak3d" / run13 / "peak3d_out"
T = pd.read_csv(ROOT / f"data/truth/{ds}/credentialed.tsv", sep="\t")
PM = pd.read_csv(d / "presence_mask.tsv", sep="\t")
A = pd.read_csv(d / "aligned_feature_table.tsv", sep="\t")
dropped = PM[(PM["n_detected"] + PM["n_confirmed"]) < min(2, N)].copy()
dropped["is_truth"] = near_any(dropped["mz"].values, dropped["rt"].values, T["mz"].values, T["rt"].values, 10, 0.1)
prm = json.load(open(d / "params.json"))
rt_tol, ppm_tol = float(prm["rt_tol_min"]), float(prm["ppm_tol"])
clouds = [load_cloud(f) for f in files]
P = [estimate(c) for c in clouds]
warps = [TableWarp(pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t")["rt_native"].values,
                   pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t")["rt_corr"].values) for s in stems]
feats = [pd.read_csv(d / "features" / f"{s}.tsv", sep="\t") for s in stems]
home = {}
for ci in range(N):
    F = feats[ci]
    sub = F[F["group_id"].isin(dropped["group_id"].values)]
    for g, h, nz, w, rtc in zip(sub["group_id"].values, sub["height"].values, sub["noise"].values,
                                (sub["rt_max_corr"] - sub["rt_min_corr"]).values, sub["rt_corr"].values):
        if g not in home or h > home[g][0]:
            home[g] = (h, nz, w, rtc, ci)
rows = []
for _, g in dropped.iterrows():
    gid = int(g["group_id"])
    h_ref, n_ref, width, rtc, hc = home[gid]
    for ci in range(N):
        if ci == hc:
            continue
        c, w, fs = clouds[ci], warps[ci], P[ci].fwhm_scans
        sig_k = 0.5 * fs / 2.355
        for wname, half in (("half", max(rt_tol, 0.5 * width)), ("half1fwhm", max(rt_tol, 0.5 * width, P[ci].fwhm_med))):
            s0 = int(np.searchsorted(c.rt, w.inverse(np.array([rtc - half]))[0]))
            s1 = int(np.searchsorted(c.rt, w.inverse(np.array([rtc + half]))[0], side="right"))
            ctx = int(max(3, round(10 * fs)))
            c0, c1 = max(0, s0 - ctx), min(c.n_scans, s1 + ctx)
            e = c.eic(float(g["mz"]), ppm_tol, c0, c1).astype(float)
            es = gaussian_filter1d(e, sig_k, mode="nearest")
            b0, b1 = s0 - c0, s1 - c0 - 1
            rec = dict(group_id=gid, is_truth=bool(g["is_truth"]), file=ci, win=wname, h_ref=h_ref, n_ref=n_ref, floor=P[ci].floor)
            win = e[b0:b1 + 1]
            rec["raw_max"] = win.max() if len(win) else 0.0
            contig = 0
            if len(win) and win.max() > 0:
                k = b0 + int(np.argmax(win))
                j = k
                contig = 1
                while j - 1 >= 0 and e[j - 1] > 0:
                    j -= 1
                    contig += 1
                j = k
                while j + 1 < len(e) and e[j + 1] > 0:
                    j += 1
                    contig += 1
            rec["contig"] = contig
            lm = [j for j in range(max(b0, 1), min(b1, len(es) - 2) + 1) if es[j] >= es[j - 1] and es[j] >= es[j + 1]]
            if lm:
                k = max(lm, key=lambda j: es[j])
                top = es[k]
                j = k
                lmin = top
                while j > 0 and es[j - 1] <= top:
                    j -= 1
                    lmin = min(lmin, es[j])
                j = k
                rmin = top
                while j < len(es) - 1 and es[j + 1] <= top:
                    j += 1
                    rmin = min(rmin, es[j])
                prom = top - max(lmin, rmin)
                outside = np.ones(len(e), bool)
                outside[b0:b1 + 1] = False
                r = (e - es)[outside] if outside.sum() >= 10 else e - es
                nres = 1.4826 * np.median(np.abs(r - np.median(r)))
                rec.update(top=top, prom=prom, nres=nres)
            else:
                rec.update(top=0.0, prom=0.0, nres=np.nan)
            rows.append(rec)
B = pd.DataFrame(rows)
out_dir = ROOT / "results/diag_confirm"
out_dir.mkdir(exist_ok=True)
B.to_csv(out_dir / f"cells_{run}.tsv", sep="\t", index=False, float_format="%.5g")

# ---- 13C references: kept groups, and every per-file detection (corrected RT, recalibrated m/z)
A13 = pd.read_csv(d13 / "aligned_feature_table.tsv", sep="\t")
ref_kept = pd.DataFrame({"mz": A13["mz"], "rt": A13["rt"]})
parts = []
for s in [Path(f).stem for f in RUNS[run13]["files"]]:
    F = pd.read_csv(d13 / "features" / f"{s}.tsv", sep="\t")
    parts.append(pd.DataFrame({"mz": F["mz_corr"] if "mz_corr" in F else F["mz"], "rt": F["rt_corr"]}))
ref_all = pd.concat(parts, ignore_index=True)


def cred_frac(f12, ref):
    """decoy-corrected credentialed fraction of f12 (mz, rt) against the 13C reference"""
    if len(f12) == 0:
        return np.nan, 0, 0.0
    tgt = int(S.credentialed(f12, ref, S.C13).sum())
    dec = float(np.mean([S.credentialed(f12, ref, S.C13 + x).sum() for x in (-0.035, -0.02, 0.02, 0.035)]))
    return (tgt - dec) / len(f12), tgt, dec


kept12 = pd.DataFrame({"mz": A["mz"], "rt": A["rt"]})
drop12 = pd.DataFrame({"mz": dropped["mz"], "rt": dropped["rt"]})
n_truth, n_other = int(dropped["is_truth"].sum()), int((~dropped["is_truth"]).sum())
lines = [f"== {run}: kept groups {len(A)}; dropped groups {len(dropped)} (truth {n_truth}, other {n_other}); "
         f"13C references: kept groups {len(ref_kept)}, per-file detections {len(ref_all)}"]
for name, f12 in (("kept 12C groups", kept12), ("all dropped groups", drop12)):
    fk, tk, dk = cred_frac(f12, ref_kept)
    fa, ta, da = cred_frac(f12, ref_all)
    lines.append(f"  {name:48s} n {len(f12):6d}  credentialed net fraction vs 13C kept {fk:.3f} ({tk} - {dk:.0f})  vs 13C all detections {fa:.3f} ({ta} - {da:.0f})")
top_ok = B["top"] >= 0.05 * B["h_ref"]
sd_ref = np.maximum(B["nres"].fillna(0), B["n_ref"])
sd_floor = np.maximum(B["nres"].fillna(0), B["floor"])
RULES = [("current: prom >= 3 x max(residual, home cell noise), top >= 5 %", top_ok & (B["prom"] >= 3 * sd_ref)),
         ("k = 2.5 (cell noise)", top_ok & (B["prom"] >= 2.5 * sd_ref)),
         ("k = 2 (cell noise)", top_ok & (B["prom"] >= 2 * sd_ref)),
         ("k = 2 (cell noise) and >= 4 contiguous scans", top_ok & (B["prom"] >= 2 * sd_ref) & (B["contig"] >= 4)),
         ("k = 2.5 (cell noise) and >= 4 contiguous", top_ok & (B["prom"] >= 2.5 * sd_ref) & (B["contig"] >= 4)),
         ("k = 3 with sd = max(residual, file floor) and >= 4 contiguous", top_ok & (B["prom"] >= 3 * sd_floor) & (B["contig"] >= 4)),
         ("k = 3 (floor), >= 4 contiguous, raw max >= 3 x home cell noise", top_ok & (B["prom"] >= 3 * sd_floor) & (B["contig"] >= 4) & (B["raw_max"] >= 3 * B["n_ref"])),
         ("k = 4 with sd = max(residual, file floor) and >= 4 contiguous", top_ok & (B["prom"] >= 4 * sd_floor) & (B["contig"] >= 4)),
         ("raw max >= 3 x home cell noise and >= 4 contiguous (no prominence)", top_ok & (B["contig"] >= 4) & (B["raw_max"] >= 3 * B["n_ref"]))]
res = []
for wn in ("half", "half1fwhm"):
    W = B["win"] == wn
    for name, m in RULES:
        ok = B[W & m].groupby("group_id").size()
        back = dropped["group_id"].isin(ok.index)
        tr, ot = int((back & dropped["is_truth"]).sum()), int((back & ~dropped["is_truth"]).sum())
        f12 = pd.DataFrame({"mz": dropped.loc[back, "mz"], "rt": dropped.loc[back, "rt"]})
        fk, _, _ = cred_frac(f12, ref_kept)
        fa, _, _ = cred_frac(f12, ref_all)
        res.append(dict(window=wn, rule=name, re_admitted=int(back.sum()), truth_rescued=tr, other=ot,
                        share_of_dropped=round(back.sum() / len(dropped), 3), cred_vs_kept13=round(fk, 3), cred_vs_all13=round(fa, 3)))
R = pd.DataFrame(res)
R.to_csv(out_dir / f"rules_{run}.tsv", sep="\t", index=False)
with open(out_dir / f"confirm_credential_{run}.log", "w") as fh:
    txt = "\n".join(lines) + "\n" + R.to_string(index=False) + "\n"
    fh.write(txt)
    print(txt)
