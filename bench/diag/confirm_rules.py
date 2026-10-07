# diagnostic: alternative cross-file confirmation rules scored on the groups the presence rule dropped: credentialed
# truth groups vs every other dropped group. Per other-file cell: smoothed local max in the window (+/- max(half, 1 FWHM)),
# prominence / sd for several sd definitions, contiguous nonzero run around the raw max, raw max / cell noise.
# usage: confirm_rules.py RUN
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS
from fix_eval import near_any
from peak3d.estimate import estimate
from peak3d.io import load_cloud
from peak3d.warp import TableWarp
pd.set_option("display.width", 230)
run = sys.argv[1]; ds = "YEAST" if run.startswith("YEAST") else "SZ22"
files = RUNS[run]["files"]; stems = [Path(f).stem for f in files]; N = len(files)
d = ROOT / "results/peak3d" / run / "peak3d_out"
T = pd.read_csv(ROOT / f"data/truth/{ds}/credentialed.tsv", sep="\t")
PM = pd.read_csv(d / "presence_mask.tsv", sep="\t")
dropped = PM[(PM["n_detected"] + PM["n_confirmed"]) < min(2, N)].copy()
dropped["is_truth"] = near_any(dropped["mz"].values, dropped["rt"].values, T["mz"].values, T["rt"].values, 10, 0.1)
prm = json.load(open(d / "params.json")); rt_tol = float(prm["rt_tol_min"]); ppm_tol = float(prm["ppm_tol"])
clouds = [load_cloud(f) for f in files]; P = [estimate(c) for c in clouds]
warps = [TableWarp(pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t")["rt_native"].values,
                   pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t")["rt_corr"].values) for s in stems]
feats = [pd.read_csv(d / "features" / f"{s}.tsv", sep="\t") for s in stems]
# the home feature of every dropped group
home = {}
for ci in range(N):
    F = feats[ci]; sub = F[F["group_id"].isin(dropped["group_id"].values)]
    for g, h, nz, w, rtc in zip(sub["group_id"].values, sub["height"].values, sub["noise"].values, (sub["rt_max_corr"] - sub["rt_min_corr"]).values, sub["rt_corr"].values):
        if g not in home or h > home[g][0]:
            home[g] = (h, nz, w, rtc, ci)
rows = []
for _, g in dropped.iterrows():
    gid = int(g["group_id"]); h_ref, n_ref, width, rtc, hc = home[gid]
    for ci in range(N):
        if ci == hc: continue
        c = clouds[ci]; w = warps[ci]; fs = P[ci].fwhm_scans; sig_k = 0.5 * fs / 2.355
        for wname, half in (("half", max(rt_tol, 0.5 * width)), ("half1fwhm", max(rt_tol, 0.5 * width, P[ci].fwhm_med))):
            s0 = int(np.searchsorted(c.rt, w.inverse(np.array([rtc - half]))[0])); s1 = int(np.searchsorted(c.rt, w.inverse(np.array([rtc + half]))[0], side="right"))
            ctx = int(max(3, round(10 * fs))); c0, c1 = max(0, s0 - ctx), min(c.n_scans, s1 + ctx)
            e = c.eic(float(g["mz"]), ppm_tol, c0, c1).astype(float); es = gaussian_filter1d(e, sig_k, mode="nearest")
            b0, b1 = s0 - c0, s1 - c0 - 1
            rec = dict(group_id=gid, is_truth=bool(g["is_truth"]), file=ci, win=wname, h_ref=h_ref, n_ref=n_ref, floor=P[ci].floor)
            win = e[b0:b1 + 1]
            rec["raw_max"] = win.max() if len(win) else 0.0
            contig = 0
            if len(win) and win.max() > 0:
                k = b0 + int(np.argmax(win)); j = k; contig = 1
                while j - 1 >= 0 and e[j - 1] > 0: j -= 1; contig += 1
                j = k
                while j + 1 < len(e) and e[j + 1] > 0: j += 1; contig += 1
            rec["contig"] = contig
            lm = [j for j in range(max(b0, 1), min(b1, len(es) - 2) + 1) if es[j] >= es[j - 1] and es[j] >= es[j + 1]]
            if lm:
                k = max(lm, key=lambda j: es[j]); top = es[k]
                j = k; lmin = top
                while j > 0 and es[j - 1] <= top: j -= 1; lmin = min(lmin, es[j])
                j = k; rmin = top
                while j < len(es) - 1 and es[j + 1] <= top: j += 1; rmin = min(rmin, es[j])
                prom = top - max(lmin, rmin)
                outside = np.ones(len(e), bool); outside[b0:b1 + 1] = False
                r = (e - es)[outside] if outside.sum() >= 10 else e - es
                nres = 1.4826 * np.median(np.abs(r - np.median(r)))
                rec.update(top=top, prom=prom, nres=nres)
            else:
                rec.update(top=0.0, prom=0.0, nres=np.nan)
            rows.append(rec)
B = pd.DataFrame(rows)
B.to_csv(ROOT / f"results/diag_idsl/confirm_rules_{run}.tsv", sep="\t", index=False, float_format="%.5g")
n_truth = int(dropped["is_truth"].sum()); n_other = int((~dropped["is_truth"]).sum())
print(f"== {run}: dropped groups {len(dropped)}: truth {n_truth}, other {n_other}")
def rule_eval(name, mask_cell):
    ok = B[mask_cell].groupby("group_id").size()
    back = dropped["group_id"].isin(ok.index)
    tr = int((back & dropped["is_truth"]).sum()); ot = int((back & ~dropped["is_truth"]).sum())
    print(f"  {name:70s} truth rescued {tr:4d}/{n_truth}  other re-admitted {ot:5d}/{n_other} ({ot / max(n_other, 1):.0%})  truth share of re-admitted {tr / max(tr + ot, 1):.2f}")
top_ok = B["top"] >= 0.05 * B["h_ref"]
sd_ref = np.maximum(B["nres"].fillna(0), B["n_ref"]); sd_floor = np.maximum(B["nres"].fillna(0), B["floor"])
for wn in ("half", "half1fwhm"):
    W = B["win"] == wn
    print(f"-- window {wn}")
    rule_eval("current: prom >= 3 x max(residual, home cell noise), top >= 5 %", W & top_ok & (B["prom"] >= 3 * sd_ref))
    rule_eval("k = 2.5", W & top_ok & (B["prom"] >= 2.5 * sd_ref))
    rule_eval("k = 2", W & top_ok & (B["prom"] >= 2 * sd_ref))
    rule_eval("k = 2 and >= 4 contiguous nonzero scans", W & top_ok & (B["prom"] >= 2 * sd_ref) & (B["contig"] >= 4))
    rule_eval("k = 2 and >= 5 contiguous nonzero scans", W & top_ok & (B["prom"] >= 2 * sd_ref) & (B["contig"] >= 5))
    rule_eval("k = 2.5 and >= 4 contiguous", W & top_ok & (B["prom"] >= 2.5 * sd_ref) & (B["contig"] >= 4))
    rule_eval("k = 3 with sd = max(residual, file floor) and >= 4 contiguous", W & top_ok & (B["prom"] >= 3 * sd_floor) & (B["contig"] >= 4))
    rule_eval("k = 3 with sd = max(residual, floor), >= 4 contiguous, raw max >= 3 x home cell noise", W & top_ok & (B["prom"] >= 3 * sd_floor) & (B["contig"] >= 4) & (B["raw_max"] >= 3 * B["n_ref"]))
    rule_eval("raw max >= 3 x home cell noise and >= 4 contiguous (no prominence test)", W & top_ok & (B["contig"] >= 4) & (B["raw_max"] >= 3 * B["n_ref"]))
    rule_eval("k = 2 and >= 4 contiguous and raw max >= 0.3 x h_ref", W & top_ok & (B["prom"] >= 2 * sd_ref) & (B["contig"] >= 4) & (B["raw_max"] >= 0.3 * B["h_ref"]))
