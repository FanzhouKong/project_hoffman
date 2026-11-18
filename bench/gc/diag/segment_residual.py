#!/usr/bin/env python3
"""Where the unexplained signal of one DeepCPR segment sits in Mosa's model: by mass (real ion of the compound or
not, picked or not, in the component or not) and by cause (spectrum incomplete, profile, noise). The reference
spectrum is DeepCPR's component at the top level of the set.

    python bench/gc/diag/segment_residual.py SET STEM T [--mosa mosa_dsk] [--top STEM]
SET: aa or fa; T: seconds (the target's time in STEM)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bench.gc.cdf_to_mzml import unit_mass                                              # noqa: E402
from bench.gc.deepcpr_score import (cosine, deepcpr_mass_offset, load_deepcpr, load_mosa, observed,   # noqa: E402
                                    r2, root_spectra)

M = Path("/quobyte/metabolomicsgrp/fanzhou/hoffmann")
R = M / "results/gc_deepcpr"
RAW = {"aa": M / "data/raw/DEEPCPR_GC/data/amino acids", "fa": M / "data/raw/DEEPCPR_GC/data/fatty acids"}
TOP = {"aa": "LMJ-50", "fa": "zhi10-5vs1"}
np.set_printoptions(linewidth=260, suppress=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set"); ap.add_argument("stem"); ap.add_argument("t", type=float)
    ap.add_argument("--mosa", default="mosa_dsk"); ap.add_argument("--top", default=None)
    g = ap.parse_args()
    top = g.top or TOP[g.set]
    rt, masses, Xb, _ = observed(RAW[g.set] / f"{g.stem}.CDF")
    rt_t, masses_t, Xb_t, _ = observed(RAW[g.set] / f"{top}.CDF")
    off = deepcpr_mass_offset(R / "deepcpr" / g.set, top, masses_t, rt_t, Xb_t)
    D = load_deepcpr(R / "deepcpr" / g.set, g.stem, masses, rt, mass_offset=off)
    Dt = load_deepcpr(R / "deepcpr" / g.set, top, masses, rt_t, mass_offset=off)
    A = load_mosa(R / g.mosa / g.set, g.stem, masses, rt)
    G = pd.read_csv(R / "deepcpr" / g.set / "seg" / f"{g.stem}.csv")
    st, ed = G["RT_st"].values * 60.0, G["RT_ed"].values * 60.0
    seg = int(np.argmin(np.where((st - 0.5 <= g.t) & (g.t <= ed + 0.5), 0.0, 1.0) + 1e-9 * np.abs(0.5 * (st + ed) - g.t)))
    s0 = int(np.searchsorted(rt, st[seg] - 1e-6)); s1 = max(int(min(np.searchsorted(rt, ed[seg] - 1e-6), len(rt))), s0 + 3)
    obs = Xb[s0:s1]
    W = s1 - s0
    # reference: the DeepCPR component at this time in this file -> its counterpart at the top level (best cosine)
    jd = int(np.argmin(np.abs(D["rt"] - g.t)))
    near_t = np.flatnonzero(np.abs(Dt["rt"] - D["rt"][jd]) <= 3.0)
    rd = root_spectra(D["spec"][jd:jd + 1])[0]
    jt = near_t[int(np.argmax([cosine(root_spectra(Dt["spec"][k:k + 1])[0], rd) for k in near_t]))]
    ref = Dt["spec"][jt] / Dt["spec"][jt].max()
    real = ref >= 0.01
    print(f"== {g.set} {g.stem} target {g.t:.1f} s: segment {st[seg]:.1f}-{ed[seg]:.1f} s ({W} scans); DeepCPR here rt {D['rt'][jd]:.2f} "
          f"({int((D['spec'][jd] >= 0.01 * D['spec'][jd].max()).sum())} masses >= 1 %), top-level rt {Dt['rt'][jt]:.2f} "
          f"({int(real.sum())} masses >= 1 %); DeepCPR components in segment: {int(((D['rt'] >= st[seg]) & (D['rt'] <= ed[seg])).sum())}")
    # Mosa's own model in the segment
    own = np.zeros_like(obs)
    inseg = []
    for c in range(len(A["rt"])):
        lo, p = A["profs"][c]
        a, b = max(lo, s0), min(lo + len(p), s1)
        if b > a:
            own[a - s0:b - s0] += np.outer(p[a - lo:b - lo], A["spec"][c])
            inseg.append(c)
    apex = np.searchsorted(rt, A["rt"])
    k_in = [c for c in inseg if s0 <= apex[c] < s1]
    rr = root_spectra(ref[None])[0]
    print("   Mosa components reaching the segment:")
    for c in inseg:
        lo, p = A["profs"][c]
        print(f"     #{c} rt {A['rt'][c]:.2f} scans {lo - s0}..{lo + len(p) - 1 - s0} of 0..{W - 1}, {int((A['spec'][c] > 0).sum())} ions, "
              f"sum {A['spec'][c].sum():.0f}, cosine to reference {cosine(root_spectra(A['spec'][c:c + 1])[0], rr):.3f}"
              f"{'' if c in k_in else '  (apex outside)'}")
    v = np.var(obs, ddof=1)
    print(f"   R2 own model {r2(obs, own):.3f}")

    def free(Sp):                                      # spectra fixed, one free non-negative amount per scan
        fit = np.zeros_like(obs)
        if len(Sp):
            for s in range(W):
                coef, _ = nnls(Sp.T, obs[s]); fit[s] = Sp.T @ coef
        return fit
    Sm = A["spec"][k_in]
    print(f"   R2 Mosa spectra, profiles refitted   {r2(obs, free(Sm)):.3f}")
    kd = np.flatnonzero((np.searchsorted(rt, D["rt"]) >= s0) & (np.searchsorted(rt, D["rt"]) < s1))
    print(f"   R2 DeepCPR spectra, profiles refitted {r2(obs, free(D['spec'][kd])):.3f}   (its own report {D['own_r2'][jd]:.3f})")
    Dc = D["spec"][kd].copy(); Dc[Dc < 0.01 * Dc.max(axis=1, keepdims=True)] = 0
    print(f"   R2 DeepCPR spectra without masses < 1 % of base, profiles refitted {r2(obs, free(Dc)):.3f}")
    print(f"   R2 reference (top-level) spectrum, one profile refitted {r2(obs, free(ref[None] * real)):.3f}")
    # profiles fixed (Mosa's own), every mass given its best non-negative amount
    P = np.zeros((W, len(inseg)))
    for q, c in enumerate(inseg):
        lo, p = A["profs"][c]
        a, b = max(lo, s0), min(lo + len(p), s1)
        P[a - s0:b - s0, q] = p[a - lo:b - lo]
    allm = np.zeros_like(obs); sig = np.zeros(obs.shape[1], bool); tval = np.zeros(obs.shape[1])
    for j in range(obs.shape[1]):
        coef, _ = nnls(P, obs[:, j]); allm[:, j] = P @ coef
        res = obs[:, j] - allm[:, j]
        sd = np.sqrt((res ** 2).sum() / max(W - P.shape[1], 1))
        best = int(np.argmax(coef * P.sum(axis=0))) if coef.max() > 0 else 0
        se = sd / max(np.linalg.norm(P[:, best]), 1e-12)
        tval[j] = coef[best] / se if se > 0 else 0.0
        sig[j] = tval[j] >= 4.0
    print(f"   R2 Mosa's profiles, every mass fitted {r2(obs, allm):.3f}; only masses whose amount is 4 sd above zero "
          f"({int(sig.sum())} masses) {r2(obs, allm * sig):.3f}; only the reference's masses {r2(obs, allm * real):.3f}")
    # the found component's profile against the per-scan amount the data ask for
    if k_in:
        cb = k_in[int(np.argmax([cosine(root_spectra(A['spec'][c:c + 1])[0], rr) for c in k_in]))]
        sp = A["spec"][cb]
        if sp.sum() > 0:
            want = obs @ sp / (sp @ sp)
            lo, p = A["profs"][cb]; have = np.zeros(W)
            a, b = max(lo, s0), min(lo + len(p), s1); have[a - s0:b - s0] = p[a - lo:b - lo]
            print(f"   component #{cb}: its profile       {(100 * have).round(0).astype(int)}")
            print(f"                  data ask (x100)    {(100 * want).round(0).astype(int)}")
    # residual by mass
    res = obs - own
    mu = obs.mean()
    ss_o = ((obs - mu) ** 2).sum(axis=0); ss_r = (res ** 2).sum(axis=0)
    F = pd.read_csv(R / g.mosa / g.set / "features" / f"{g.stem}.tsv", sep="\t", low_memory=False)
    fm = unit_mass(F["mz"].values.astype(float)); fa = F["scan_apex"].values.astype(int)
    ins = (fa >= s0) & (fa < s1)
    picked = np.isin(masses, fm[ins])
    in_comp = (A["spec"][inseg] > 0).any(axis=0) if inseg else np.zeros(len(masses), bool)
    cat = np.where(real & in_comp, "real, in a component", np.where(real & picked, "real, picked, not in a component",
                   np.where(real, "real, not picked", np.where(in_comp, "not in reference, in a component", "not in reference, not modelled"))))
    T = pd.DataFrame(dict(cat=cat, ss_obs=ss_o, ss_res=ss_r))
    S = T.groupby("cat").agg(masses=("ss_obs", "size"), obs_share=("ss_obs", "sum"), unexplained_share=("ss_res", "sum"))
    S["obs_share"] /= ss_o.sum(); S["unexplained_share"] /= ss_o.sum()
    print("   shares of the segment's variance (unexplained sums to 1 - R2):")
    print("   " + S.round(3).to_string().replace("\n", "\n   "))
    o = np.argsort(-ss_r)[:14]
    h = obs.max(axis=0)
    snr = {}
    for m_, s_ in zip(fm[ins], F["snr"].values[ins]):
        snr[m_] = max(snr.get(m_, 0), s_)
    print("   masses with the most unexplained variance:")
    print("   " + pd.DataFrame(dict(mass=masses[o], ref_pct=(100 * ref[o]).round(1), max_obs=h[o].round(0), unexplained=(ss_r[o] / ss_o.sum()).round(4),
                                    of_its_own=(ss_r[o] / np.maximum(ss_o[o], 1e-12)).round(2), mosa_ion=in_comp[o], picked=picked[o],
                                    pick_snr=[round(snr.get(m_, 0), 1) for m_ in masses[o]], t_on_profile=tval[o].round(1))).to_string(index=False).replace("\n", "\n   "))


if __name__ == "__main__":
    main()
