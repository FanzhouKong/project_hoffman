# diagnostic: how many aligned features have a stronger neighbour of the same ion nearby (duplicate upper bound)
import sys, re, glob
import numpy as np, pandas as pd
ROOT = "/quobyte/metabolomicsgrp/fanzhou/hoffmann"
run = sys.argv[1] if len(sys.argv) > 1 else "BM21_RP"

def dup_stats(name, mz, rt, inten, rt_tol_s=3.0, ppm=5.0):
    o = np.argsort(mz); smz = mz[o]
    dup = np.zeros(len(mz), bool)
    for i in range(len(mz)):
        lo = np.searchsorted(smz, mz[i] * (1 - ppm * 1e-6)); hi = np.searchsorted(smz, mz[i] * (1 + ppm * 1e-6), side="right")
        k = o[lo:hi]; k = k[(k != i) & (np.abs(rt[k] - rt[i]) * 60 <= rt_tol_s) & (inten[k] > inten[i])]
        dup[i] = len(k) > 0
    print(f"{name:9s} {run}: {len(mz):6d} features; with a stronger same-ion neighbour within {ppm:.0f} ppm / {rt_tol_s:.0f} s: {dup.sum():6d} ({dup.mean():.1%})")
    return dup

t = pd.read_csv(f"{ROOT}/results/peak3d/{run}/peak3d_out/aligned_feature_table.tsv", sep="\t")
cols = [c for c in t.columns if c not in ("group_id", "mz", "rt", "rt_min", "rt_max", "n_detected", "n_filled", "mz_ppm_spread", "rt_sd")]
d_p = dup_stats("peak3d", t["mz"].values, t["rt"].values, t[cols].max(axis=1).values)
a = pd.read_csv(glob.glob(f"{ROOT}/results/asari/{run}/asari_out*/export/full_Feature_table.tsv")[0], sep="\t")
ac = [c for c in a.columns if c.endswith(".mzML") or re.match(r"^(MT|Blank|S[AB]|batch|posi|1[23]C)", str(c))]
dup_stats("asari", a["mz"].values, a["rtime"].values / 60, a[ac].astype(float).max(axis=1).values if ac else a["peak_area"].values)
m = pd.read_csv(f"{ROOT}/results/masscube/{run}/aligned_feature_table.txt", sep="\t", low_memory=False)
mc = [c for c in m.columns if str(c).endswith(".mzML")]
dup_stats("masscube", m["m/z"].values, m["RT"].values, m[mc].astype(float).max(axis=1).values)
# isotopes in our table: groups whose per-file members are mostly annotated iso_offset > 0
f = pd.concat([pd.read_csv(p, sep="\t", usecols=["group_id", "iso_offset"]) for p in glob.glob(f"{ROOT}/results/peak3d/{run}/peak3d_out/features/*.tsv") if not p.endswith(("manifest.tsv", "rejected.tsv"))])
iso_frac = f.groupby("group_id")["iso_offset"].apply(lambda x: (x > 0).mean())
print(f"peak3d {run}: groups that are isotopologues (M+n) by majority of members: {(iso_frac > 0.5).sum()} ({(iso_frac > 0.5).mean():.1%})")
print(f"masscube {run}: rows with isotope_state != 0: {(m['isotope_state'].fillna(0) != 0).sum()} of {len(m)}" if "isotope_state" in m else "")
