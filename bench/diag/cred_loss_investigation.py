# diagnostic: why the current picker finds fewer credentialed truth compounds than the Oct 5 version.
# Two loss channels: (A) per-file chromatogram check (lowsnr / noapex) on the Oct 5 feature, (B) presence rule
# (single detection, no confirmed peak in the other files). For each lost compound: the test values, alternative
# noise definitions, the truth set's own presence criterion in the other files, and a 3-file EIC gallery.
# usage: cred_loss_investigation.py RUN   (RUN = YEAST_12C | SZ22_12C; reads mzML -> srun)
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench/diag"))
from bench.runs import RUNS
from fix_eval import near_any
from peak3d import kernels as K
from peak3d.estimate import estimate
from peak3d.features import basins_to_frame, eic_check, gate, mark_fragments, ridge_ratio
from peak3d.io import load_cloud
from peak3d.pick import pick_cloud
from peak3d.warp import TableWarp
pd.set_option("display.width", 230)
run = sys.argv[1]; ds = "YEAST" if run.startswith("YEAST") else "SZ22"
OUT = ROOT / "results/diag_idsl"
files = RUNS[run]["files"]; stems = [Path(f).stem for f in files]; N = len(files)
d = ROOT / "results/peak3d" / run / "peak3d_out"
T = pd.read_csv(ROOT / f"data/truth/{ds}/credentialed.tsv", sep="\t")
old = pd.read_csv(ROOT / f"results/peak3d_2026-10-05/{run}/peak3d_out/aligned_feature_table.tsv", sep="\t", usecols=["mz", "rt"])
PM = pd.read_csv(d / "presence_mask.tsv", sep="\t")
kept = PM[(PM["n_detected"] + PM["n_confirmed"]) >= min(2, N)]
f_old = near_any(T["mz"].values, T["rt"].values, old["mz"].values, old["rt"].values, 10, 0.1)
f_all = near_any(T["mz"].values, T["rt"].values, PM["mz"].values, PM["rt"].values, 10, 0.1)
f_now = near_any(T["mz"].values, T["rt"].values, kept["mz"].values, kept["rt"].values, 10, 0.1)
lostA = f_old & ~f_all; lostB = f_all & ~f_now
print(f"== {run}: truth {len(T)}; found Oct 5 {f_old.sum()}, now {f_now.sum()}; lost per-file {lostA.sum()}, lost by presence {lostB.sum()}, "
      f"gained {int((~f_old & f_now).sum())}")
for nm, m in (("lost per-file", lostA), ("lost by presence", lostB), ("found now", f_now)):
    s = T[m]
    print(f"  {nm:17s} n {len(s):5d}  tier B {np.mean(s['tier'] == 'B'):.2f}  primary {np.mean(s['feature_class'] == 'primary'):.2f}  "
          f"height12 median {s['height12'].median():.3g}  reps12 == 3: {np.mean(s['reps12'] == 3):.2f}  n_iso median {s['n_iso'].median():.0f}")

clouds = [load_cloud(f) for f in files]
P = [estimate(c) for c in clouds]
warps = [TableWarp(pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t")["rt_native"].values,
                   pd.read_csv(d / "rt_correction" / f"{s}.tsv", sep="\t")["rt_corr"].values) for s in stems]
prm = json.load(open(d / "params.json")); rt_tol = float(prm["rt_tol_min"]); ppm_tol = float(prm["ppm_tol"])
rows_gal = []

def eic_profile(ci, mz, s_apex, s_lo, s_hi, half_win, kern_sig):
    c = clouds[ci]; S = c.n_scans
    a0, a1 = max(0, s_apex - half_win), min(S, s_apex + half_win + 1)
    ppm = max(5.0, 3 * np.sqrt(P[ci].sig_a ** 2 + P[ci].sig_b ** 2 / 1.0) + 1)   # tolerance for a weak ion
    e = c.eic(float(mz), ppm, a0, a1).astype(float)
    es = gaussian_filter1d(e, kern_sig, mode="nearest")
    return a0, e, es

# ---------------- A. per-file losses: the Oct 5 features and the chromatogram check variants
A_rows = []
for ci, s in enumerate(stems):
    F5 = pd.read_csv(ROOT / f"results/peak3d_2026-10-05/{run}/peak3d_out/features/{s}.tsv", sep="\t")
    hit = near_any(F5["mz"].values, F5["rt_corr"].values, T.loc[lostA, "mz"].values, T.loc[lostA, "rt"].values, 10, 0.1)
    sub = F5[hit].copy()
    if not len(sub): continue
    sub["scan_lo"] = sub["scan_min"]; sub["scan_hi"] = sub["scan_max"]; sub["sig_apex"] = P[ci].sig_ppm(sub["height"].values)
    E = eic_check(sub, clouds[ci], P[ci])
    fs = P[ci].fwhm_scans; sig_k = 0.5 * fs / 2.355; hw = int(max(3, round(10 * fs)))
    for (i, f), (_, e_) in zip(sub.iterrows(), E.iterrows()):
        a0, e, es = eic_profile(ci, f["mz"], int(f["scan_apex"]), int(f["scan_lo"]), int(f["scan_hi"]), hw, sig_k)
        b0, b1 = max(0, int(f["scan_lo"]) - a0), min(len(e) - 1, int(f["scan_hi"]) - a0)
        outside = np.ones(len(e), bool); outside[b0:b1 + 1] = False
        r = e - es
        top = es[b0:b1 + 1].max()
        base = outside & (es < 0.2 * top)                       # baseline scans only
        mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) >= 5 else np.nan
        n_cur = max(e_["es_noise"], f["noise"])
        n_base = mad(r[base]); n_left = mad(r[:b0]); n_right = mad(r[b1 + 1:])
        n_side = np.nanmin([n_left, n_right])
        A_rows.append(dict(stem=s, mz=f["mz"], rt=f["rt"], height=f["height"], snr=f["snr"], cls=e_["eic_class"], prom=e_["es_prom"],
                           noise_res=e_["es_noise"], noise_cell=f["noise"], snr_cur=e_["eic_snr"],
                           snr_base=e_["es_prom"] / max(np.nan_to_num(n_base, nan=0), f["noise"]),
                           snr_side=e_["es_prom"] / max(np.nan_to_num(n_side, nan=0), f["noise"]),
                           interference=e[outside].max() / max(f["height"], 1e-9), n_scans=f["n_scans"], fwhm_rel=f["fwhm"] / P[ci].fwhm_med,
                           scan_apex=int(f["scan_apex"]), scan_lo=int(f["scan_lo"]), scan_hi=int(f["scan_hi"]), file=ci, channel="A"))
A = pd.DataFrame(A_rows)
if len(A):
    print(f"\n-- A. per-file losses: {len(A)} Oct 5 features of {lostA.sum()} compounds; classes: " + ", ".join(f"{k} {v}" for k, v in A["cls"].value_counts().items()))
    print("   medians: height %.3g  snr(3D) %.1f  prominence %.3g  residual noise %.3g  cell noise %.3g  eic_snr %.2f  n_scans %.0f  fwhm/med %.2f  max signal outside bounds / apex %.2f" % tuple(
        A[c].median() for c in ["height", "snr", "prom", "noise_res", "noise_cell", "snr_cur", "n_scans", "fwhm_rel", "interference"]))
    print("   residual noise > cell noise in %.0f%% of them; residual noise / cell noise median %.2f" % (100 * np.mean(A["noise_res"] > A["noise_cell"]), (A["noise_res"] / A["noise_cell"]).median()))
    for nm, c in (("current (all scans outside bounds)", "snr_cur"), ("baseline scans only (smoothed < 20 % apex)", "snr_base"), ("cleaner side only", "snr_side")):
        print(f"   eic_snr under noise = {nm:44s}: pass at 3: {np.mean(A[c] >= 3):.2f}, median {A[c].median():.2f}")
    # cost side on file 0: candidates the chromatogram check rejected as lowsnr, under the same alternatives
    ci = 0; c = clouds[ci]; b = pick_cloud(c, P[ci]); df = basins_to_frame(b); df["fragment"] = mark_fragments(df, c, P[ci])
    for nm_, v in ridge_ratio(df, c, P[ci]).items(): df[nm_] = v
    cand = df[gate(df, P[ci]).values].reset_index(drop=True); E = eic_check(cand, c, P[ci])
    rej = cand[E["eic_class"].values == "lowsnr"].copy(); Er = E[E["eic_class"].values == "lowsnr"]
    fs = P[ci].fwhm_scans; sig_k = 0.5 * fs / 2.355; hw = int(max(3, round(10 * fs)))
    rs = rej.sample(min(2000, len(rej)), random_state=0); Es = Er.loc[rs.index]
    pb, ps = [], []
    for (i, f), (_, e_) in zip(rs.iterrows(), Es.iterrows()):
        a0, e, es = eic_profile(ci, f["mz"], int(f["scan_apex"]), int(f["scan_lo"]), int(f["scan_hi"]), hw, sig_k)
        b0, b1 = max(0, int(f["scan_lo"]) - a0), min(len(e) - 1, int(f["scan_hi"]) - a0)
        outside = np.ones(len(e), bool); outside[b0:b1 + 1] = False
        r = e - es; top = es[b0:b1 + 1].max(); base = outside & (es < 0.2 * top)
        mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) >= 5 else np.nan
        pb.append(e_["es_prom"] / max(np.nan_to_num(mad(r[base]), nan=0), f["noise"]) >= 3)
        ps.append(e_["es_prom"] / max(np.nan_to_num(np.nanmin([mad(r[:b0]), mad(r[b1 + 1:])]), nan=0), f["noise"]) >= 3)
    print(f"   cost on {stems[ci]}: {len(rej)} candidates rejected as lowsnr; of a sample of {len(rs)}, would pass with baseline-only noise {np.mean(pb):.2f}, with cleaner-side noise {np.mean(ps):.2f}")
    print(f"   -> re-admitted features per file ~ {len(rej) * np.mean(pb):.0f} (baseline) / {len(rej) * np.mean(ps):.0f} (side) vs truth features rescued per file ~ {np.sum(A['snr_base'] >= 3) / N:.0f} / {np.sum(A['snr_side'] >= 3) / N:.0f}")
    rows_gal += A.to_dict("records")

# ---------------- B. presence losses: why the other files did not confirm
dropped = PM[(PM["n_detected"] + PM["n_confirmed"]) < min(2, N)]
hitT = near_any(dropped["mz"].values, dropped["rt"].values, T.loc[lostB, "mz"].values, T.loc[lostB, "rt"].values, 10, 0.1)
Dt = dropped[hitT]
feats = [pd.read_csv(d / "features" / f"{s}.tsv", sep="\t") for s in stems]
B_rows = []
def confirm_diag(gid, home, mz, rt_c, h_ref, n_ref, width):
    res = []
    for ci in range(N):
        if ci == home: continue
        c = clouds[ci]; w = warps[ci]; fs = P[ci].fwhm_scans; sig_k = 0.5 * fs / 2.355
        half = max(rt_tol, 0.5 * width)
        s0 = int(np.searchsorted(c.rt, w.inverse(np.array([rt_c - half]))[0])); s1 = int(np.searchsorted(c.rt, w.inverse(np.array([rt_c + half]))[0], side="right"))
        ctx = int(max(3, round(10 * fs))); c0, c1 = max(0, s0 - ctx), min(c.n_scans, s1 + ctx)
        e = c.eic(float(mz), ppm_tol, c0, c1).astype(float); es = gaussian_filter1d(e, sig_k, mode="nearest")
        b0, b1 = s0 - c0, s1 - c0 - 1
        win = e[b0:b1 + 1]
        rec = dict(file=ci, h_fill=win.max() if len(win) else 0.0, h_ref=h_ref)
        # truth-style presence: >= 4 contiguous nonzero scans around the window max
        if len(win) and win.max() > 0:
            k = b0 + int(np.argmax(win)); j = k; run_len = 1
            while j - 1 >= 0 and e[j - 1] > 0: j -= 1; run_len += 1
            j = k
            while j + 1 < len(e) and e[j + 1] > 0: j += 1; run_len += 1
            rec["contig"] = run_len
        else:
            rec["contig"] = 0
        lm = [j for j in range(max(b0, 1), min(b1, len(es) - 2) + 1) if es[j] >= es[j - 1] and es[j] >= es[j + 1]]
        if not lm:
            # nearest smoothed local max anywhere in the context, in scans from the window
            allm = [j for j in range(1, len(es) - 1) if es[j] >= es[j - 1] and es[j] >= es[j + 1] and es[j] >= 0.05 * h_ref]
            rec["reason"] = "no signal" if rec["h_fill"] <= 0 else ("below 5 % of ref" if rec["h_fill"] < 0.05 * h_ref else "no local max in window")
            rec["nearest_max_scans"] = min([min(abs(j - b0), abs(j - b1)) for j in allm]) if allm else np.nan
            res.append(rec); continue
        k = max(lm, key=lambda j: es[j]); top = es[k]
        j = k; lmin = top
        while j > 0 and es[j - 1] <= top: j -= 1; lmin = min(lmin, es[j])
        j = k; rmin = top
        while j < len(es) - 1 and es[j + 1] <= top: j += 1; rmin = min(rmin, es[j])
        prom = top - max(lmin, rmin)
        outside = np.ones(len(e), bool); outside[b0:b1 + 1] = False
        r = (e - es)[outside] if outside.sum() >= 10 else e - es
        nres = 1.4826 * np.median(np.abs(r - np.median(r)))
        rec.update(top=top, prom=prom, noise_res=nres, noise_ref=n_ref, ratio=prom / max(nres, n_ref))
        if top < 0.05 * h_ref: rec["reason"] = "below 5 % of ref"
        elif prom < 3 * max(nres, n_ref): rec["reason"] = "prominence < 3 sd (residual)" if nres >= n_ref else "prominence < 3 sd (ref noise)"
        else: rec["reason"] = "confirmed"
        res.append(rec)
    return res
for _, g in Dt.iterrows():
    gid = int(g["group_id"]); home = None
    for ci in range(N):
        F = feats[ci]; m = F["group_id"].values == gid
        if m.any():
            f = F[m].iloc[0]; home = ci
            width = f["rt_max_corr"] - f["rt_min_corr"]
            res = confirm_diag(gid, home, g["mz"], f["rt_corr"], f["height"], f["noise"], width)
            for r_ in res:
                r_.update(group_id=gid, mz=g["mz"], rt=g["rt"], home=home, height=f["height"], snr=f["snr"], eic_snr=f["eic_snr"], score=f["score"])
                B_rows.append(r_)
            rows_gal.append(dict(stem=stems[home], mz=f["mz"], rt=f["rt"], height=f["height"], snr=f["snr"], cls="presence", scan_apex=int(f["scan_apex"]),
                                 scan_lo=int(f["scan_min"]), scan_hi=int(f["scan_max"]), file=home, channel="B", snr_cur=f["eic_snr"],
                                 reasons="; ".join(f"f{r_['file']}: {r_['reason']}" for r_ in res)))
            break
B = pd.DataFrame(B_rows)
if len(B):
    print(f"\n-- B. presence losses: {Dt.shape[0]} dropped truth groups (single detection); other-file cells {len(B)}")
    print("   failure reason per other-file cell: " + ", ".join(f"{k} {v}" for k, v in B["reason"].value_counts().items()))
    print(f"   truth-style presence (>= 4 contiguous nonzero scans at the position) holds in {np.mean(B['contig'] >= 4):.2f} of these cells; "
          f"median contiguous run {B['contig'].median():.0f}; median h_fill / h_ref {(B['h_fill'] / B['h_ref']).median():.2f}")
    pr = B[B["reason"].str.startswith("prominence")]
    if len(pr):
        print(f"   where a local max exists but prominence is too low: median prominence / sd {pr['ratio'].median():.2f}, residual noise >= ref noise in {np.mean(pr['noise_res'] >= pr['noise_ref']):.2f}; "
              f"would pass at 2 sd: {np.mean(pr['ratio'] >= 2):.2f}")
    nl = B[B["reason"] == "no local max in window"]
    if len(nl):
        print(f"   where no smoothed maximum lies in the window: nearest maximum (>= 5 % ref) is {nl['nearest_max_scans'].median():.0f} scans away (median); within 1 FWHM: {np.mean(nl['nearest_max_scans'] <= P[0].fwhm_scans):.2f}")
    print("   home-file feature of the dropped truth groups: height median %.3g, snr %.1f, eic_snr %.1f, score %.2f" % (
        B.drop_duplicates("group_id")["height"].median(), B.drop_duplicates("group_id")["snr"].median(), B.drop_duplicates("group_id")["eic_snr"].median(), B.drop_duplicates("group_id")["score"].median()))
    # the same diagnostics for non-truth dropped groups (sample), to see whether anything separates them
    Dn = dropped[~hitT].sample(min(300, int((~hitT).sum())), random_state=0)
    C_rows = []
    for _, g in Dn.iterrows():
        gid = int(g["group_id"])
        for ci in range(N):
            F = feats[ci]; m = F["group_id"].values == gid
            if m.any():
                f = F[m].iloc[0]
                for r_ in confirm_diag(gid, ci, g["mz"], f["rt_corr"], f["height"], f["noise"], f["rt_max_corr"] - f["rt_min_corr"]):
                    r_.update(group_id=gid); C_rows.append(r_)
                break
    Cn = pd.DataFrame(C_rows)
    print("   non-truth dropped groups (sample): reasons " + ", ".join(f"{k} {v}" for k, v in Cn["reason"].value_counts().items()) +
          f"; truth-style presence holds in {np.mean(Cn['contig'] >= 4):.2f}; median h_fill / h_ref {(Cn['h_fill'] / Cn['h_ref']).median():.2f}; "
          f"prominence / sd median {Cn.loc[Cn['reason'].str.startswith('prominence'), 'ratio'].median():.2f}")
    B.to_csv(OUT / f"cred_loss_B_{run}.tsv", sep="\t", index=False, float_format="%.5g")
if len(A): A.to_csv(OUT / f"cred_loss_A_{run}.tsv", sep="\t", index=False, float_format="%.5g")

# ---------------- gallery: 3-file overlay for 30 lost compounds (A and B mixed)
G = pd.DataFrame(rows_gal)
if len(G):
    pick = G.sample(min(30, len(G)), random_state=1)
    fig, axes = plt.subplots(6, 5, figsize=(17, 15), dpi=110, squeeze=False)
    cols = ["#0969da", "#bf8700", "#8250df"]
    for ax, (_, f) in zip(axes.ravel(), pick.iterrows()):
        home = int(f["file"]); fs = P[home].fwhm_scans
        W = int(max(10 * fs, 30)); sa = int(f["scan_apex"]); c_home = clouds[home]
        t0, t1 = warps[home].forward(np.array([c_home.rt[max(0, sa - W)], c_home.rt[min(c_home.n_scans - 1, sa + W)]]))
        for ci in range(N):
            c = clouds[ci]; w = warps[ci]
            s0 = int(np.searchsorted(c.rt, w.inverse(np.array([t0]))[0])); s1 = int(np.searchsorted(c.rt, w.inverse(np.array([t1]))[0]))
            x = w.forward(c.rt[s0:s1]) * 60
            e = c.eic(float(f["mz"]), 8.0, s0, s1)
            ax.plot(x, e, color=cols[ci], lw=1.2 if ci == home else 0.8, alpha=1.0 if ci == home else 0.75, marker="." if ci == home else None, ms=2.5)
        lo = warps[home].forward(np.array([c_home.rt[int(f["scan_lo"])]]))[0] * 60; hi = warps[home].forward(np.array([c_home.rt[int(f["scan_hi"])]]))[0] * 60
        ax.axvspan(lo, hi, color="#bf8700", alpha=0.12)
        ttl = f"{f['channel']} {f['stem'][-10:]} m/z {f['mz']:.4f} rt {f['rt']:.2f} h {f['height']:.3g} snr3d {f['snr']:.0f} eic_snr {f['snr_cur']:.1f} {f['cls']}"
        if f["channel"] == "B": ttl += "\n" + str(f.get("reasons", ""))[:95]
        else: ttl += f"\nres noise {f.get('noise_res', np.nan):.3g} cell {f.get('noise_cell', np.nan):.3g} interf {f.get('interference', np.nan):.2f}"
        ax.set_title(ttl, fontsize=6.2, loc="left"); ax.tick_params(labelsize=5)
    for ax in axes.ravel()[len(pick):]: ax.axis("off")
    fig.suptitle(f"{run}: credentialed truth lost vs Oct 5 (A = chromatogram check in the home file, B = presence rule); the 3 files overlaid on the corrected RT axis, home file with markers", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / f"cred_loss_{run}.png"); plt.close(fig)
    print("gallery written")
