"""End-to-end check of bench/spiketruth.py on a synthetic PR002464-like study.

Five standards with known spike A/B amounts; the vials of the real design (matrix-free C and E
series, blanks, a plasma-only vial); four injections per vial with a per-batch RT shift. Planted
traps: a background ion present in every file, an A-only contaminant at a standard's adduct
mass, an ion seen in only two injections, and an isomer at a second RT. Co-eluting M+1 isotopes
and an in-source fragment must be found by the sweep.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import spiketruth as st  # noqa: E402
from peak3d.synth import Peak, make_cloud, write_mzml  # noqa: E402

# vial -> (spike A uL, spike B uL, plasma uL, sample_type)
VIALS = {9: (500, 500, 0, "sample"), 10: (250, 250, 0, "sample"), 11: (100, 100, 0, "sample"),
         12: (50, 50, 0, "sample"), 18: (1000, 0, 0, "sample"), 19: (0, 1000, 0, "sample"),
         20: (250, 750, 0, "sample"), 21: (750, 250, 0, "sample"), 22: (500, 500, 0, "sample"),
         13: (0, 0, 1000, "sample"), 39: (0, 0, 0, "process.blank"), 42: (0, 0, 0, "solvent.blank")}
BATCH_SHIFT = [0.0, 0.01, -0.01, 0.02]     # min
STDS = pd.DataFrame([
    # std, name, formula, a, b, rt, response (height at 1000 uL of a 1.0 amount)
    (1, "Aspartame", "C14H18N2O5", 1.0, 0.125, 1.0, 4e6),
    (2, "Propylparaben", "C10H12O3", 0.25, 1.0, 1.6, 2e6),
    (3, "Artemetin", "C20H20O8", 1.0, 0.5, 2.0, 3e6),
    (4, "Saccharin", "C7H5NO3S", 0.0625, 1.0, 2.4, 1e6),
    (5, "Absent", "C11H22O", 1.0, 0.25, 1.2, 0.0),
], columns=["std", "name", "formula", "spikeA_vol", "spikeB_vol", "rt", "resp"])
STDS["log2FC"] = np.log2(STDS["spikeA_vol"] / STDS["spikeB_vol"])
STDS["mass"] = [st.formula_mass(f) for f in STDS["formula"]]
STDS["monoisotopic_mass"] = STDS["mass"]
STDS["xlogp"] = 0.0
H = st.PROTON
NA = 22.9892207
FRAG_MZ = 120.0808                         # in-source fragment of standard 1
BACKGROUND = (STDS.loc[1, "mass"] + 18.0338257, 1.6)   # std 2 [M+NH4]+, co-eluting, in every file incl. blanks
CONTAM = (STDS.loc[2, "mass"] + 38.9631579, 2.0)   # std 3 [M+K]+ mass, only in the A-only vial
TWO_REPS = (STDS.loc[3, "mass"] + NA, 2.7)          # std 4 [M+Na]+ at another RT, two injections only
ISOMER_RT = 0.7                            # std 2 [M+H]+ also here (an isomeric impurity)


def peaks_for(vial, batch):
    A, B, plasma, _ = VIALS[vial]
    sh = BATCH_SHIFT[batch]
    out = []
    for _, s in STDS.iterrows():
        amt = (s["spikeA_vol"] * A + s["spikeB_vol"] * B) / 1000
        h = s["resp"] * amt
        if h <= 0:
            continue
        m = s["mass"]
        out.append(Peak(m + H, s["rt"] + sh, h, 0.04, iso=(0.15,)))
        out.append(Peak(m + NA, s["rt"] + sh, 0.3 * h, 0.04))
        if s["std"] == 1:
            out.append(Peak(FRAG_MZ, s["rt"] + sh, 0.2 * h, 0.04))
        if s["std"] == 2:
            out.append(Peak(m + H, ISOMER_RT + sh, 0.5 * h, 0.04))
    out.append(Peak(BACKGROUND[0], BACKGROUND[1] + sh, 5e5, 0.05))
    if plasma:
        out.append(Peak(450.3000, 1.8 + sh, 1e6, 0.04))
    if vial == 18:
        out.append(Peak(CONTAM[0], CONTAM[1] + sh, 8e5, 0.04))
    if vial == 19 and batch < 2:
        out.append(Peak(TWO_REPS[0], TWO_REPS[1] + sh, 5e5, 0.04))
    return out


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    d = tmp_path_factory.mktemp("pr2464")
    rows = []
    order = 0
    for b in range(4):
        for v, (A, B, plasma, stype) in VIALS.items():
            order += 1
            cloud, _ = make_cloud(peaks_for(v, b), n_scans=700, dt=0.005, thresh=2e3, floor=3e3,
                                  n_noise=4000, seed=100 * b + v, mz_range=(60, 900))
            name = f"syn_{b + 1}_{order:03d}_{v:02d}"
            write_mzml(cloud, d / f"{name}.mzML")
            rows.append({"sample_name": name + ".raw", "batch": b + 1, "injection_order": order, "vial": v,
                         "ms_level": 1, "sample_type": stype, "SRM 1950": plasma / 2, "SRM 8231": plasma / 2,
                         "Yeast 12C": 0.0, "Yeast 13C": 0.0, "[spikeA]": float(A), "[spikeB]": float(B),
                         "path": str(d / f"{name}.mzML"), "file": name})
    files = st.classify_files(pd.DataFrame(rows))
    truth = st.build("SYN", files, STDS, "pos", "orbitrap", d / "out", cpus=2)
    return truth, d / "out"


def near(truth, mz, rt, ppm=5, tol=0.03):
    return truth[(np.abs(truth["mz_obs"] - mz) <= mz * ppm * 1e-6) & (np.abs(truth["rt_apex"] - rt) <= tol)]


def test_formula_ions_found(study):
    truth, _ = study
    for _, s in STDS[STDS["resp"] > 0].iterrows():
        for ion, d in (("[M+H]+", H), ("[M+Na]+", NA)):
            hit = near(truth, s["mass"] + d, s["rt"])
            assert len(hit) == 1, (s["name"], ion)
            assert hit["std"].iloc[0] == s["std"] and hit["annotation"].iloc[0] == ion
            assert hit["rt_rank"].iloc[0] == 1


def test_traps_rejected(study):
    truth, out = study
    assert near(truth, *BACKGROUND).empty
    assert near(truth, *CONTAM).empty
    assert near(truth, *TWO_REPS).empty
    assert near(truth, 450.3000, 1.8).empty
    summ = pd.read_csv(out / "standards_summary.tsv", sep="\t")
    assert summ.set_index("std").loc[5, "status"] == "not_found"


def test_isomer_second_rt(study):
    truth, _ = study
    hit = near(truth, STDS.loc[1, "mass"] + H, ISOMER_RT)
    assert len(hit) == 1 and hit["std"].iloc[0] == 2 and hit["rt_rank"].iloc[0] == 2


def test_coeluting_ions(study):
    truth, _ = study
    frag = near(truth, FRAG_MZ, 1.0)
    assert len(frag) == 1 and frag["std"].iloc[0] == 1 and frag["annotation_class"].iloc[0] == "fragment"
    for _, s in STDS[STDS["resp"] > 0].iterrows():
        iso = near(truth, s["mass"] + H + st.C13, s["rt"])
        assert len(iso) == 1 and iso["annotation_class"].iloc[0] == "isotope", s["name"]
    # nothing else: 4 standards x (M+H, M+Na, M+1) + fragment + isomer
    assert len(truth) == 4 * 3 + 2


def test_signatures_and_decoys(study):
    truth, out = study
    assert (truth["tier"] == "A").all()
    f = truth[(truth["source"] == "formula") & ~truth["weak_at_noise"]]
    assert len(f) and (np.abs(f["log2_18_19"] - f["log2FC"]) <= st.SIG_TOL).all()
    qc = pd.read_csv(out / "decoy_qc.tsv", sep="\t").set_index("search")
    dec = [i for i in qc.index if i != "target"]
    assert qc.loc[dec, "tier_A"].sum() == 0 and qc.loc[dec, "tier_B"].sum() == 0
    pf = pd.read_csv(out / "per_file.tsv.gz", sep="\t")
    assert len(pf) == len(truth) * 4 * len(VIALS)
