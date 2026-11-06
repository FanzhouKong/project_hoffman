"""The demo data of `peak3d demo` and the app's "Create example data": three synthetic centroid LC-MS/MS runs and a
small spectral library to search them with.

MS1: 60 compounds with an M+1 isotope, noise, and RT drift between runs. The twelve strongest are metabolites
([M+H]+ of the formulas in COMPOUNDS, M+1 from their carbon and nitrogen counts); they also carry a sodium adduct and
a water-loss fragment, six of them a fragment with no table relation (shown by its m/z difference), and three a
broad co-eluting ion two 13C steps up (a possible isotope: the m/z fits, the peak shape does not), so the compound
groups have something to show.
MS2: data-dependent spectra, two per compound and run around its apex, for every compound of at least
MS2_MIN_HEIGHT: the fragments of its template (the metabolites: sub-formula ions of [M+H]+; the other compounds:
fixed random m/z), their m/z a few ppm off and their intensities scattered, the unfragmented precursor, and random
noise ions for the denoising to remove.
Library (LIBRARY): the metabolites' templates, plus L-isoleucine, an isomer of the L-leucine in the runs with another
fragment pattern, so a search lists two candidates for one feature. The spectra are made for the demo; they are not
measured reference spectra."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

LIBRARY = "demo_library.msp"     # written next to the mzML folder
MS2_MIN_HEIGHT = 8e4             # weaker compounds are not picked for fragmentation
MS2_SCALE = 0.2                  # base fragment intensity / precursor MS1 intensity at the spectrum's time


@dataclass(frozen=True)
class Compound:
    name: str
    formula: str                 # neutral
    smiles: str
    fragments: tuple             # (ion formula of a fragment of [M+H]+, relative intensity)
    precursor: float             # relative intensity of the unfragmented [M+H]+

    def ion(self, formula: str | None = None) -> float:
        """m/z of [M+H]+ (formula None) or of a fragment ion formula"""
        from specdenoise import formula as fm
        f = fm.parse(formula or self.formula)
        if formula is None:
            f[fm.H] += 1
        return fm.ion_mz(f, 1)

    @property
    def mz(self) -> float:
        return self.ion()

    def template(self) -> np.ndarray:
        """(m/z, relative intensity) rows, ascending m/z, the precursor included"""
        t = [(self.ion(f), float(r)) for f, r in self.fragments] + [(self.mz, float(self.precursor))]
        return np.array(sorted(t))


COMPOUNDS = (       # strongest first
    Compound("LysoPC(16:0)", "C24H50NO7P", "CCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COP(=O)([O-])OCC[N+](C)(C)C",
             (("C5H15NO4P", 100), ("C5H14NO", 35), ("C24H49NO6P", 10), ("C19H37O3", 6), ("C5H12N", 8)), 20),
    Compound("Reserpine", "C33H40N2O9", "COC(=O)C1C(OC)C(OC(=O)c2cc(OC)c(OC)c(OC)c2)CC2CN3CCc4c([nH]c5cc(OC)ccc45)C3CC21",
             (("C10H11O4", 100), ("C23H29N2O3", 55), ("C11H12NO", 25), ("C22H25N2O2", 15)), 30),
    Compound("Glutathione", "C10H17N3O6S", "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O",
             (("C6H11N2O3S", 100), ("C8H13N2O4S", 40), ("C6H8NO3S", 35), ("C5H8NO3", 30), ("C4H6NO", 20),
              ("C2H6NS", 12)), 25),
    Compound("Sphingosine", "C18H37NO2", "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@@H](N)CO",
             (("C18H36NO", 100), ("C18H34N", 60), ("C17H34N", 20), ("C2H6NO", 15)), 10),
    Compound("Adenosine", "C10H13N5O4", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
             (("C5H6N5", 100), ("C5H3N4", 6), ("C4H4N3", 4)), 8),
    Compound("Guanosine", "C10H13N5O5", "Nc1nc2n(cnc2c(=O)[nH]1)[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
             (("C5H6N5O", 100), ("C5H3N4O", 10), ("C4H4N3O", 6)), 12),
    Compound("L-Tryptophan", "C11H12N2O2", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
             (("C11H10NO2", 100), ("C9H8NO", 55), ("C11H8NO", 18), ("C8H8N", 20), ("C10H10N", 12)), 15),
    Compound("Caffeine", "C8H10N4O2", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
             (("C6H8N3O", 100), ("C5H8N3", 18), ("C4H7N2", 8), ("C2H4N", 10)), 60),
    Compound("L-Tyrosine", "C9H11NO3", "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
             (("C8H10NO", 100), ("C9H9O3", 35), ("C8H7O", 30), ("C7H7", 15), ("C7H7O2", 12)), 10),
    Compound("L-Phenylalanine", "C9H11NO2", "N[C@@H](Cc1ccccc1)C(=O)O",
             (("C8H10N", 100), ("C8H7", 20), ("C7H7", 12), ("C9H9O2", 6), ("C9H7O", 8)), 10),
    Compound("L-Carnitine", "C7H15NO3", "C[N+](C)(C)C[C@@H](O)CC(=O)[O-]",
             (("C4H5O2", 100), ("C4H7O3", 55), ("C3H10N", 40), ("C7H14NO2", 10)), 35),
    Compound("L-Leucine", "C6H13NO2", "CC(C)C[C@H](N)C(=O)O",
             (("C5H12N", 100), ("C2H6N", 20), ("C3H7", 12), ("CH4N", 6)), 15),
)
ISOMER = Compound("L-Isoleucine", "C6H13NO2", "CC[C@H](C)[C@H](N)C(=O)O",       # library only
                  (("C5H12N", 100), ("C5H9", 40), ("C3H5", 15), ("C2H6N", 12)), 15)


def _m1(c: Compound) -> float:
    """M+1 / M of the compound: 13C and 15N"""
    from specdenoise import formula as fm
    f = fm.parse(c.formula)
    return round(0.0107 * f[fm.C] + 0.0037 * f[fm.N], 3)


def _spectrum(rng, template: np.ndarray, precursor_mz: float, scale: float):
    """one measured spectrum of a template: m/z 3 ppm off, intensities scattered 20 %, plus 6-15 noise ions (median
    3 % of the base fragment, some above the precursor)"""
    mz = template[:, 0] * (1.0 + 3e-6 * rng.standard_normal(len(template)))
    it = scale * template[:, 1] / 100.0 * np.exp(0.2 * rng.standard_normal(len(template)))
    n = int(rng.integers(6, 16))
    mz = np.concatenate([mz, rng.uniform(40.0, precursor_mz + 20.0, n)])
    it = np.concatenate([it, scale * 0.03 * np.exp(0.7 * rng.standard_normal(n))])
    o = np.argsort(mz)
    return mz[o], it[o]


def write_library(path: Path) -> Path:
    """the demo library (.msp) of COMPOUNDS and ISOMER"""
    blocks = []
    for c in (*COMPOUNDS, ISOMER):
        t = c.template()
        rows = "\n".join(f"{m:.5f}\t{i * 999.0 / t[:, 1].max():.0f}" for m, i in t)
        blocks.append(f"NAME: {c.name}\nPRECURSORMZ: {c.mz:.5f}\nPRECURSORTYPE: [M+H]+\nFORMULA: {c.formula}\n"
                      f"SMILES: {c.smiles}\nIONMODE: Positive\n"
                      "COMMENT: synthetic spectrum of the Mosa demo data, not a measured reference\n"
                      f"Num Peaks: {len(t)}\n{rows}\n")
    path.write_text("\n".join(blocks))
    return path


def write_demo(out: Path) -> Path:
    """OUT/demo_mzml/demo{1,2,3}.mzML and OUT/demo_library.msp (see the module text); returns the mzML folder"""
    from .synth import Peak, make_cloud, profile, write_mzml
    out = Path(out)
    data = out / "demo_mzml"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    base = list(zip(rng.uniform(150, 900, 60), rng.uniform(0.6, 3.4, 60), 10 ** rng.uniform(4.5, 6.5, 60)))
    iso = [(0.15,)] * 60
    strongest = sorted(range(60), key=lambda i: -base[i][2])[:12]
    for c, i in zip(COMPOUNDS, strongest):
        base[i] = (c.mz, base[i][1], base[i][2])
        iso[i] = (_m1(c),)
    partners = []                                  # (compound, m/z offset, relative height, FWHM in minutes; 0 = the compound's)
    for j, i in enumerate(strongest):
        partners += [(i, 22.9892207 - 1.00727646, 0.35, 0.0), (i, -18.0105647, 0.25, 0.0)]      # [M+Na]+, [M+H-H2O]+
        if j < 6:
            partners.append((i, -57.0214637, 0.18, 0.0))                                       # a fragment, unnamed
        if j < 3:
            partners.append((i, 2 * 1.0033548, 0.12, 0.25))                                    # M+2 m/z, broad peak
    rng2 = np.random.default_rng(1)                # MS2 draws: the MS1 data stay those of the MS1-only demo
    templates = {c_i: c.template() for c, c_i in zip(COMPOUNDS, strongest)}
    for i, (m, r, hgt) in enumerate(base):
        if i not in templates and hgt >= MS2_MIN_HEIGHT:
            nf = int(rng2.integers(3, 7))
            t = np.column_stack([rng2.uniform(50.0, m - 10.0, nf), np.r_[100.0, rng2.uniform(5, 80, nf - 1)]])
            templates[i] = np.array(sorted([*map(tuple, t), (m, float(rng2.uniform(5, 40)))]))
    for k, shift in enumerate((0.0, 0.05, -0.04)):
        peaks = [Peak(float(m), float(r + shift + 0.01 * r * k), float(h), 0.05, iso=iso[i])
                 for i, (m, r, h) in enumerate(base)]
        for i, dm, rel, w in partners:
            m, r, hgt = base[i]
            peaks.append(Peak(float(m + dm), float(r + shift + 0.01 * r * k), float(hgt * rel), w or 0.05,
                              iso=iso[i] if not w else ()))
        cloud, _ = make_cloud(peaks, n_scans=800, dt=0.005, thresh=200.0, floor=100.0, n_noise=20000, seed=k,
                              name=f"demo{k + 1}")
        ms2 = []
        for i, t in templates.items():
            pk = peaks[i]
            for _ in range(2):
                when = pk.rt + float(rng2.uniform(-0.012, 0.012))
                scale = MS2_SCALE * pk.height * float(profile(np.array([when]), pk)[0])
                prec = pk.mz * (1.0 + 2e-6 * rng2.standard_normal())
                ms2.append((when, prec, *_spectrum(rng2, t, pk.mz, scale)))
        write_mzml(cloud, data / f"demo{k + 1}.mzML", ms2=ms2)
    write_library(out / LIBRARY)
    return data
