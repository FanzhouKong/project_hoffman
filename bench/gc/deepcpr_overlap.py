#!/usr/bin/env python3
"""Component by component, two tools on the same files of the DeepCPR benchmark: for every component of the
reference tool (B), the components of the other tool (A) within --rt-tol seconds, split into those with the same
spectrum (cosine of square-root intensities, ions above 1 % of the base peak, >= --min-cos) and the others.

  one      B's compound is one component in A
  split    two or more A components with B's spectrum at that time (A divided the compound)
  missed   no A component with B's spectrum
and for A: components with no B component of the same spectrum nearby ("extra"), with their share of A's total area.

    python bench/gc/deepcpr_overlap.py --raw DIR --stems a,b,c --a mosa:PATH --b deepcpr:PATH [--rt-tol 1.5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.gc.cdf_to_mzml import read_cdf, unit_matrix                     # noqa: E402
from bench.gc.deepcpr_score import LOADERS, cosine, deepcpr_mass_offset, observed, root_spectra   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--stems", required=True)
    ap.add_argument("--a", required=True, metavar="KIND:PATH")
    ap.add_argument("--b", required=True, metavar="KIND:PATH")
    ap.add_argument("--rt-tol", type=float, default=1.5)
    ap.add_argument("--min-cos", type=float, default=0.8)
    ap.add_argument("--min-rel-area", type=float, default=0.0, help="B components below this share of B's largest are left out")
    g = ap.parse_args()
    stems = g.stems.split(",")
    lo = hi = None
    axes = {}
    for s in stems:
        rt, off, mz, it = read_cdf(g.raw / f"{s}.CDF")
        masses, _ = unit_matrix(rt, off, mz, it)
        axes[s] = rt
        lo = masses[0] if lo is None else min(lo, masses[0])
        hi = masses[-1] if hi is None else max(hi, masses[-1])
    masses = np.arange(lo, hi + 1)
    tot = dict(one=0, split=0, missed=0, b=0, a=0, extra=0, extra_area=0.0, a_area=0.0)
    kws = {}
    for side in ("a", "b"):
        kind, path = getattr(g, side).split(":", 1)
        if kind == "deepcpr":
            rt, m0, Xb, _ = observed(g.raw / f"{stems[-1]}.CDF")
            full = np.zeros((Xb.shape[0], len(masses)))
            full[:, m0[0] - lo:m0[-1] - lo + 1] = Xb
            kws[side] = dict(mass_offset=deepcpr_mass_offset(path, stems[-1], masses, rt, full))
        else:
            kws[side] = {}
    print("file\tB\tone\tsplit\tmissed\tA\textra\textra_area_share")
    for s in stems:
        rt = axes[s]
        A = LOADERS[g.a.split(":", 1)[0]](g.a.split(":", 1)[1], s, masses, rt, **kws["a"])
        B = LOADERS[g.b.split(":", 1)[0]](g.b.split(":", 1)[1], s, masses, rt, **kws["b"])
        sa, sb = root_spectra(A["spec"]), root_spectra(B["spec"])
        keep = np.flatnonzero(B["area"] >= g.min_rel_area * B["area"].max()) if len(B["area"]) else np.zeros(0, int)
        partner = np.zeros(len(A["rt"]), bool)
        n = dict(one=0, split=0, missed=0)
        for j in keep:
            near = np.flatnonzero(np.abs(A["rt"] - B["rt"][j]) <= g.rt_tol)
            same = [i for i in near if cosine(sa[i], sb[j]) >= g.min_cos]
            partner[same] = True
            n["one" if len(same) == 1 else "split" if len(same) > 1 else "missed"] += 1
        for j in range(len(B["rt"])):                       # any B component, whatever its size, explains an A one
            near = np.flatnonzero(np.abs(A["rt"] - B["rt"][j]) <= g.rt_tol)
            for i in near:
                if cosine(sa[i], sb[j]) >= g.min_cos:
                    partner[i] = True
        extra = ~partner
        share = float(A["area"][extra].sum() / A["area"].sum()) if A["area"].sum() > 0 else 0.0
        print(f"{s}\t{len(keep)}\t{n['one']}\t{n['split']}\t{n['missed']}\t{len(partner)}\t{int(extra.sum())}\t{share:.3f}")
        for k in n:
            tot[k] += n[k]
        tot["b"] += len(keep); tot["a"] += len(partner); tot["extra"] += int(extra.sum())
        tot["extra_area"] += float(A["area"][extra].sum()); tot["a_area"] += float(A["area"].sum())
    print(f"all\t{tot['b']}\t{tot['one']}\t{tot['split']}\t{tot['missed']}\t{tot['a']}\t{tot['extra']}\t"
          f"{tot['extra_area'] / max(tot['a_area'], 1e-12):.3f}")


if __name__ == "__main__":
    main()
