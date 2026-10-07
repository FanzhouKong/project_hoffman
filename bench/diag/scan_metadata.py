# diagnostic: consecutive MS1 scan metadata of one file (interleaved scan types?)
import sys
import numpy as np
from pyteomics import mzml
path = sys.argv[1]
rows = []
with mzml.read(path, use_index=False) as r:
    for k, s in enumerate(r):
        if k >= 60: break
        sc = s["scanList"]["scan"][0]
        rows.append(dict(i=k, ms=s.get("ms level"), rt=float(sc["scan start time"]),
                         pol="neg" if "negative scan" in s else ("pos" if "positive scan" in s else "?"),
                         filt=str(sc.get("filter string", ""))[:60], inj=sc.get("ion injection time"),
                         win=(s.get("scanList", {}).get("scan", [{}])[0].get("scanWindowList", {}).get("scanWindow", [{}])[0].get("scan window lower limit"),
                              s.get("scanList", {}).get("scan", [{}])[0].get("scanWindowList", {}).get("scanWindow", [{}])[0].get("scan window upper limit")),
                         n=len(s["m/z array"]), tic=float(s.get("total ion current", np.nan)), bp=float(s.get("base peak intensity", np.nan)),
                         mzmin=float(s["m/z array"].min()) if len(s["m/z array"]) else np.nan))
for r_ in rows[:40]:
    print(r_["i"], r_["ms"], "%.4f" % r_["rt"], r_["pol"], r_["filt"], "inj", r_["inj"], "win", r_["win"], "n", r_["n"], "tic %.3g" % r_["tic"], "bp %.3g" % r_["bp"], "mzmin %.1f" % r_["mzmin"])
tic = np.array([r_["tic"] for r_ in rows]); n = np.array([r_["n"] for r_ in rows])
print("\nTIC odd/even scan ratio (median):", np.median(tic[1::2][:len(tic[::2])] / tic[::2][:len(tic[1::2])]).round(3), " n centroids odd/even:", np.median(n[1::2][:len(n[::2])] / n[::2][:len(n[1::2])]).round(3))
