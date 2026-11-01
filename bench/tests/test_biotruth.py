"""End-to-end check of the plasma truth in bench/biotruth.py on a synthetic PR002464-like study.

Plasma vials of the real design (dilution 1-4, SRM 1950 : 8231 mixing 13-17), blanks, a yeast-only and
a spike-only vial; 4 injections each. Real metabolites scale with plasma; traps: a background ion in
every file, an internal-standard-like ion (process blank and every sample, constant), an ion that is the
same in every sample vial whatever the plasma amount, and an ion seen in only two injections.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import biotruth as bt  # noqa: E402
from bench import spiketruth as st  # noqa: E402
from peak3d.synth import Peak, make_cloud, write_mzml  # noqa: E402

# vial -> (SRM 1950 uL, SRM 8231 uL, yeast uL, spike uL, sample_type)
VIALS = {1: (500, 500, 0, 0, "sample"), 2: (250, 250, 0, 0, "sample"), 3: (100, 100, 0, 0, "sample"),
         4: (50, 50, 0, 0, "sample"), 13: (1000, 0, 0, 0, "sample"), 14: (0, 1000, 0, 0, "sample"),
         15: (250, 750, 0, 0, "sample"), 16: (750, 250, 0, 0, "sample"), 17: (500, 500, 0, 0, "sample"),
         23: (0, 0, 1000, 0, "sample"), 9: (0, 0, 0, 1000, "sample"),
         39: (0, 0, 0, 0, "process.blank"), 42: (0, 0, 0, 0, "solvent.blank")}
BATCH_SHIFT = [0.0, 0.01, -0.01, 0.02]
# metabolites: (m/z, rt, height per uL in SRM 1950, in SRM 8231)
MET = {"equal": (180.0634, 1.0, 2000.0, 2000.0), "only1950": (204.1230, 1.6, 3000.0, 0.0),
       "strong": (132.1019, 2.2, 8000.0, 6000.0)}
BACKGROUND = (250.1500, 1.3)        # every file, blanks included
IS_LIKE = (210.1100, 2.6)           # process blank and every non-solvent-blank vial, constant
CONSTANT = (320.2000, 1.9)          # every sample vial, same height regardless of plasma
TWO_REPS = (275.0500, 2.4)          # vial 17, batches 1-2 only


def peaks_for(vial, b):
    s1950, s8231, yeast, spike, stype = VIALS[vial]
    sh = BATCH_SHIFT[b]
    out = []
    for name, (mz, rt, k1, k2) in MET.items():
        h = k1 * s1950 + k2 * s8231
        if h > 0:
            out.append(Peak(mz, rt + sh, h, 0.04, iso=(0.08,) if name == "strong" else ()))
    out.append(Peak(BACKGROUND[0], BACKGROUND[1] + sh, 4e5, 0.05))
    if stype != "solvent.blank":
        out.append(Peak(IS_LIKE[0], IS_LIKE[1] + sh, 6e5, 0.04))
    if stype == "sample":
        out.append(Peak(CONSTANT[0], CONSTANT[1] + sh, 5e5, 0.04))
    if vial == 17 and b < 2:
        out.append(Peak(TWO_REPS[0], TWO_REPS[1] + sh, 5e5, 0.04))
    if yeast:
        out.append(Peak(MET["equal"][0], MET["equal"][1] + sh, 1e6, 0.04))   # shared with yeast
    return out


@pytest.fixture(scope="module")
def plasma(tmp_path_factory):
    d = tmp_path_factory.mktemp("plasma")
    rows, order = [], 0
    for b in range(4):
        for v, (s1950, s8231, yeast, spike, stype) in VIALS.items():
            order += 1
            cloud, _ = make_cloud(peaks_for(v, b), n_scans=700, dt=0.005, thresh=2e3, floor=3e3,
                                  n_noise=4000, seed=100 * b + v, mz_range=(60, 600))
            name = f"syn_{b + 1}_{order:03d}_{v:02d}"
            write_mzml(cloud, d / f"{name}.mzML")
            rows.append({"sample_name": name + ".raw", "batch": b + 1, "injection_order": order, "vial": v,
                         "ms_level": 1, "sample_type": stype, "SRM 1950": float(s1950), "SRM 8231": float(s8231),
                         "Yeast 12C": float(yeast), "Yeast 13C": 0.0, "[spikeA]": spike / 2, "[spikeB]": spike / 2,
                         "path": str(d / f"{name}.mzML"), "file": name})
    files = st.classify_files(pd.DataFrame(rows))
    files["off"] = [BATCH_SHIFT[b - 1] for b in files["batch"]]
    spike = pd.DataFrame(columns=["truth_id", "mz_obs", "rt_apex"])
    out = d / "out"
    out.mkdir()
    truth = bt.plasma_core("SYN", files, out, "pos", 5.0, 0.05, spike, cpus=2)
    return truth, out


def near(t, mz, rt, ppm=5, tol=0.03):
    return t[(np.abs(t["mz_obs"] - mz) <= mz * ppm * 1e-6) & (np.abs(t["rt_apex"] - rt) <= tol)]


def test_metabolites_found(plasma):
    truth, _ = plasma
    for name, (mz, rt, _, _) in MET.items():
        hit = near(truth, mz, rt)
        assert len(hit) == 1 and hit["tier"].iloc[0] == "A", name
        assert abs(hit["dil_slope"].iloc[0] - 1) < 0.15, name
    assert near(truth, MET["only1950"][0], MET["only1950"][1])["log2_1950_8231"].iloc[0] > 3
    assert bool(near(truth, *MET["equal"][:2])["also_in_yeast"].iloc[0])
    iso = near(truth, MET["strong"][0] + st.C13, MET["strong"][1])
    assert len(iso) == 1 and iso["annotation"].iloc[0].startswith("isotope M+1")


def test_traps_rejected(plasma):
    truth, _ = plasma
    for trap in (BACKGROUND, IS_LIKE, CONSTANT, TWO_REPS):
        assert near(truth, *trap).empty, trap
    assert len(truth) == len(MET) + 1      # the three metabolites + one isotope


def test_null_and_outputs(plasma):
    truth, out = plasma
    qc = pd.read_csv(out / "plasma_null_qc.tsv", sep="\t").set_index("order")
    assert qc.loc[[i for i in qc.index if i != "true"], "tier_AB"].sum() == 0
    assert (out / "plasma_truth.tsv").exists() and (out / "plasma_params.json").exists()
