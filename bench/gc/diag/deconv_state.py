"""state of the deconvolution in a time window: features (sub-scan apex, tolerance, cut, component) and components
    python state.py SET STEM T0 T1 [explain=0 ...]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from peak3d.gc import deconv as DC
from peak3d.io import Cloud
from bench.gc.cdf_to_mzml import unit_mass

R = Path("/quobyte/metabolomicsgrp/fanzhou/hoffmann/results/gc_deepcpr/mosa_dsk")
st, stem, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
kw = {}
for a in sys.argv[5:]:
    k, v = a.split("=")
    kw[k] = type(getattr(DC.DeconvParams(), k))(float(v)) if not isinstance(getattr(DC.DeconvParams(), k), bool) else bool(int(v))
out = Path(st) if Path(st).is_dir() else R / st          # SET may be a peak3d output folder
P = DC.params_from_dict(json.loads((out / "features" / f"{stem}.params.json").read_text()))
cloud = Cloud.load_npz(out / ".cache" / f"{stem}.npz")
F = pd.read_csv(out / "features" / f"{stem}.tsv", sep="\t", low_memory=False)
D = DC.DeconvParams(**kw)
d = DC.deconvolve(cloud, F, P, D)
np.set_printoptions(linewidth=250, suppress=True, precision=2)
print("params", kw, "| calibration:", {k: d.info.get(k) for k in ("shape_classes", "snr_edges", "tau_classes_s", "n_iso_pairs")},
      {k: d.info.get(k) for k in ("n_components", "n_composite", "n_explained", "n_rejoined")})
f_mz = F["mz"].values.astype(float); f_h = F["height"].values.astype(float)
f_tol = DC.ion_tol(f_mz, f_h, P)
half = int(np.ceil(2.0 * P.fwhm_med / P.dt)) + 2
f_t, _ = DC._apex_times(cloud.off, cloud.mz, cloud.inten, cloud.rt, f_mz, f_tol,
                     np.clip(F["scan_apex"].values.astype(np.int64), 0, cloud.n_scans - 1), half, float(P.dt),
                     np.zeros(len(F), np.int64), np.full(len(F), cloud.n_scans - 1, np.int64))
tau, rcut, _ = DC.calibrate(F, f_t, cloud, f_tol, P, D)
print('shoulder apexes kept:', d.info.get('n_shoulder_apexes'))
C = d.components
sel = np.flatnonzero((f_t * 60 >= t0) & (f_t * 60 <= t1))
T = pd.DataFrame(dict(fid=sel, m=unit_mass(f_mz[sel]), t=(f_t[sel] * 60).round(3), h=f_h[sel].round(0), snr=F["snr"].values[sel].round(1),
                      lo=(cloud.rt[F["scan_min"].values[sel]] * 60).round(1), hi=(cloud.rt[F["scan_max"].values[sel]] * 60).round(1),
                      tau=(tau[sel] * 60).round(3), rcut=rcut[sel].round(3), comp=d.feature_component[sel],
                      r=np.asarray(d.feature_r)[sel].round(2), shared=np.asarray(d.feature_shared)[sel].astype(int),
                      loose=np.asarray(d.feature_loose)[sel].astype(int), iso=F["iso_parent"].values[sel]))
print(T.sort_values("t").to_string(index=False))
cs = C[(C.rt * 60 >= t0 - 2) & (C.rt * 60 <= t1 + 2)]
print(cs.assign(rt_s=(cs.rt * 60).round(2), lo=(cs.rt_min * 60).round(1), hi=(cs.rt_max * 60).round(1), base=unit_mass(cs.base_mz.values))[
    ["component_id", "rt_s", "lo", "hi", "n_features", "n_loose", "n_delayed", "n_ions", "n_shared_ions", "base", "tic", "model_r", "apex_tol_s"]].to_string(index=False))
a, b = np.searchsorted(cloud.rt * 60, [t0, t1])
for c in cs.component_id:
    r = C.iloc[int(c)]; p = d.profile(int(c)); v = np.zeros(b - a)
    lo, hi = int(r.scan_min), int(r.scan_max)
    x0, x1 = max(lo, a), min(hi + 1, b)
    if x1 > x0:
        v[x0 - a:x1 - a] = p[x0 - lo:x1 - lo]
    print(f"model #{int(c):3d}", (100 * v).round(0).astype(int))

# second maxima in the window (before the coincidence filter)
snr = np.maximum(F["snr"].values.astype(float), 1e-9)
flo = np.clip(F["scan_min"].values.astype(np.int64), 0, cloud.n_scans - 1); fhi = np.clip(F["scan_max"].values.astype(np.int64), 0, cloud.n_scans - 1)
fap = np.clip(F["scan_apex"].values.astype(np.int64), 0, cloud.n_scans - 1)
par, sa, slo, shi, sh, ss = DC._sub_apexes(cloud.off, cloud.mz, cloud.inten, f_mz, f_tol, flo, fhi, fap, float(P.k_valley) * np.sqrt(2.0) * f_h / snr)
k = np.flatnonzero((cloud.rt[sa] * 60 >= t0) & (cloud.rt[sa] * 60 <= t1))
print("second maxima (all candidates):")
print(pd.DataFrame(dict(m=unit_mass(f_mz[par[k]]), t=(cloud.rt[sa[k]] * 60).round(1), h=sh[k].round(0), parent_h=f_h[par[k]].round(0),
                        thr=(float(P.k_valley) * np.sqrt(2.0) * f_h / snr)[par[k]].round(0), lo=(cloud.rt[slo[k]] * 60).round(1),
                        hi=(cloud.rt[shi[k]] * 60).round(1), side=ss[k])).sort_values("t").to_string(index=False))
