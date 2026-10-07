#!/usr/bin/env python3
"""Why do credentialed-feature counts differ so much between tools? (yeast + SZ22)

1. Chance matches, estimated two ways:
   - mass decoy (as in score.py): wrong spacing per carbon
   - RT decoy: right 13C spacing, but partner searched 1-3 min away in RT; keeps the
     real mass-defect structure, so it tests whether the mass decoy undercounts chance
2. Redundancy: share of credentialed features that are an isotope (+1/+2 x 13C) or
   Na/K/NH4 adduct of another credentialed feature at the same RT
3. Overlap: of IDSL.IPA / MassCube credentialed features, how many asari also reports,
   and the intensities of the ones asari misses (asari default min_peak_height 1e5)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.runs import CRED_PAIRS, TOOLS  # noqa: E402
from bench.score import C13, credentialed, hits, load  # noqa: E402


def rt_decoy(f12, f13, shift):
    g = f12.copy()
    g["rt"] = g["rt"] + shift
    return credentialed(g, f13, C13)


# mass differences to a co-eluting "parent": 13C isotopes, Na/K/NH4 vs H adducts
RELATIONS = [C13, 2 * C13, 21.98194, 37.95588, 17.02655]


def redundant(f, ppm=5, rt_tol=0.02):
    """credentialed features explainable as isotope/adduct of another credentialed one"""
    out = np.zeros(len(f), bool)
    for d in RELATIONS:
        out |= hits(f, f["mz"].values - d, f["rt"].values, ppm, rt_tol)
    return out


for name, (r12, r13) in CRED_PAIRS.items():
    print(f"\n######## {name}")
    cred = {}
    for tool in TOOLS:
        a, b = load(tool, r12), load(tool, r13)
        if a is None or b is None:
            continue
        f12, S12 = a
        c = credentialed(f12, b[0], C13)
        mass_dec = np.mean([credentialed(f12, b[0], C13 + d).sum() for d in (-0.035, -0.02, 0.02, 0.035)])
        rt_dec = np.mean([rt_decoy(f12, b[0], s).sum() for s in (-3, -2, -1.5, 1.5, 2, 3)])
        height = S12.max(axis=1).values
        cred[tool] = (f12[c].reset_index(drop=True), height[c])
        print(f"{tool:9s} features={len(f12):6d} credentialed={c.sum():6d} "
              f"mass-decoy={mass_dec:8.1f} RT-decoy={rt_dec:8.1f} "
              f"net(mass)={c.sum() - mass_dec:8.1f} net(RT)={c.sum() - rt_dec:8.1f} "
              f"| credentialed that are isotope/adduct of another: {redundant(f12[c].reset_index(drop=True)).mean():.1%}")
    if "asari" in cred:
        fa = cred["asari"][0]
        asari_all = load("asari", r12)[0]
        for tool in ("masscube", "idslipa"):
            if tool not in cred:
                continue
            f, h = cred[tool]
            in_asari = hits(asari_all, f["mz"].values, f["rt"].values, 5, 0.1)
            miss = ~in_asari
            print(f"  {tool} credentialed also in asari output: {in_asari.mean():.1%}; "
                  f"of those asari misses, max height <1e5: {(h[miss] < 1e5).mean():.1%} "
                  f"(median {np.median(h[miss]):.0f}); found-by-asari median {np.median(h[in_asari]):.0f}")
        for tool in ("masscube", "idslipa"):
            if tool in cred:
                back = hits(cred[tool][0], fa["mz"].values, fa["rt"].values, 5, 0.1)
                print(f"  asari credentialed also credentialed-by-{tool}: {back.mean():.1%}")
