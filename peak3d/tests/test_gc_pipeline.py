import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peak3d.gc import library as L
from peak3d.gc.synth import Compound, alkane, make_run, random_spectrum
from peak3d.synth import write_mzml

ROOT = Path(__file__).resolve().parents[2]
DT = 1.0 / 300.0
SCALE = {"gc_a": 1.0, "gc_b": 0.5, "gc_c": 2.0}
SHIFT_S = {"gc_a": 0.0, "gc_b": 0.6, "gc_c": -0.4}


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def _run(*argv):
    """the command line in a process of its own (it sets NUMBA_NUM_THREADS, which must not leak into the test run)"""
    return subprocess.run([sys.executable, "-m", "peak3d", *argv], cwd=ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """three runs of the same 30 compounds and a C10-C16 alkane ladder, shifted in time and scaled; two of the
    compounds co-elute 1.5 s apart and share an ion; a unit-mass library holds 12 of them with their indices"""
    root = tmp_path_factory.mktemp("gc")
    rng = np.random.default_rng(11)
    ladder_rt = {n: 0.6 + 0.6 * (n - 10) for n in range(10, 17)}
    comps = [alkane(n, t, 2e5) for n, t in ladder_rt.items()]
    known = []
    for k in range(30):
        m, r = random_spectrum(rng, 20)
        rt = 0.75 + 0.115 * k
        known.append(Compound(f"compound {k:02d}", rt, 10 ** rng.uniform(4.3, 5.3), m, r))
    known[11].rt = known[10].rt + 1.5 / 60                          # a co-eluting pair ...
    known[11].mz[4] = known[10].mz[7]                               # ... with a shared ion
    o = np.argsort(known[11].mz)
    known[11].mz, known[11].rel = known[11].mz[o], known[11].rel[o]
    comps += known
    data = root / "mzml"
    data.mkdir()
    for stem in SCALE:
        run = [Compound(c.name, c.rt + SHIFT_S[stem] / 60, c.height * SCALE[stem], c.mz, c.rel, c.fwhm) for c in comps]
        cloud, _ = make_run(run, n_scans=1500, dt=DT, seed=17 + list(SCALE).index(stem), name=stem)
        write_mzml(cloud, data / f"{stem}.mzML")
    # library: unit masses, index from the ladder (linear in this synthetic run: 100 index units per 0.6 min)
    ri_of = lambda t: 1000.0 + 100.0 * (t - 0.6) / 0.6            # noqa: E731
    lib = root / "ei.msp"
    with open(lib, "w") as fh:
        for c in known[:12]:
            nm = L.nominal_mass(c.mz)
            fh.write(f"Name: {c.name}\nRetention_index: {ri_of(c.rt):.1f}\nNum Peaks: {len(nm)}\n")
            fh.write("".join(f"{a} {999 * b:.1f}\n" for a, b in zip(nm, c.rel)) + "\n")
        c = known[0]                                                # the same spectrum under a far index
        fh.write(f"Name: decoy of 00\nRetention_index: 2500\nNum Peaks: {len(c.mz)}\n")
        fh.write("".join(f"{a} {999 * b:.1f}\n" for a, b in zip(L.nominal_mass(c.mz), c.rel)) + "\n")
    out = root / "out"
    r = _run("process", "--input", str(data), "--output", str(out), "--no-report")
    assert r.returncode == 0, r.stderr[-2000:]
    before = {f: _md5(out / f) for f in ("aligned_feature_table.tsv", "features/gc_a.tsv", "params.json")}
    r = _run("gc", str(out), "--ri-ladder", "alkane", "--library", str(lib))
    assert r.returncode == 0, r.stderr[-2000:]
    return dict(out=out, known=known, ri_of=ri_of, before=before, lib=lib, ladder_rt=ladder_rt)


def test_lc_outputs_are_untouched(study):
    for f, h in study["before"].items():
        assert _md5(study["out"] / f) == h


def test_compounds_found_in_all_files_with_ladder_and_index(study):
    gc = study["out"] / "gc"
    comp = pd.read_csv(gc / "compounds.tsv", sep="\t")
    lad = pd.read_csv(gc / "ladder.tsv", sep="\t")
    assert sorted(lad["carbon"].unique()) == list(range(10, 17)) and lad["file"].nunique() == 3
    assert lad["molecular_ion"].all()
    known = study["known"]
    n_one = 0
    for c in known:
        near = comp[np.abs(comp["rt"] - c.rt) * 60 < 1.0]
        hit = near[near["n_files"] == 3]
        n_one += len(hit) == 1
        if len(hit) == 1:
            assert abs(hit["ri"].iloc[0] - study["ri_of"](c.rt)) < 3.0          # 3 index units = 1 s here
    assert n_one >= len(known) - 2
    assert len(comp) <= len(known) + 7 + 3                                       # 30 compounds + 7 alkanes


def test_library_annotation_and_index_window(study):
    comp = pd.read_csv(study["out"] / "gc" / "compounds.tsv", sep="\t")
    named = comp[comp["name"].notna() & (comp["name"] != "")]
    assert set(named["name"]) >= {f"compound {k:02d}" for k in range(12)} - {"compound 10", "compound 11"}
    assert "decoy of 00" not in set(named["name"])
    for _, row in named.iterrows():
        c = next(x for x in study["known"] if x.name == row["name"])
        assert abs(row["rt"] - c.rt) * 60 < 1.0 and row["match"] >= 0.7 and abs(row["delta_ri"]) < 3.0
    hits = pd.read_csv(study["out"] / "gc" / "identifications.tsv", sep="\t")
    assert "decoy of 00" not in set(hits["name"])                                # outside the index window
    assert {"compound 10", "compound 11"} <= set(named["name"])                  # the co-eluting pair, both named


def test_quantifier_follows_the_amounts(study):
    gc = study["out"] / "gc"
    tab = pd.read_csv(gc / "compound_table.tsv", sep="\t")
    full = tab.dropna(subset=list(SCALE))
    assert len(full) >= 30
    assert np.median(np.abs(full["gc_b"] / full["gc_a"] / 0.5 - 1.0)) < 0.03
    assert np.median(np.abs(full["gc_c"] / full["gc_a"] / 2.0 - 1.0)) < 0.03
    assert (np.abs(full["gc_c"] / full["gc_a"] / 2.0 - 1.0) < 0.15).mean() > 0.9


def test_outputs_and_params(study):
    gc = study["out"] / "gc"
    p = json.loads((gc / "params.json").read_text())
    assert p["n_files"] == 3 and p["ladder"] == "alkane" and p["n_ladder"] == 7 and p["ri_tol"] == 20.0
    assert all(f["calibrated"] for f in p["files"])
    spectra = L.read_msp(gc / "spectra.msp")
    comp = pd.read_csv(gc / "compounds.tsv", sep="\t")
    assert len(spectra) == len(comp) and not spectra.nominal
    per_file = pd.read_csv(gc / "components" / "gc_a.tsv", sep="\t")
    assert {"component_id", "rt", "rt_corr", "ri", "compound_id", "n_ions", "quant_mz"} <= set(per_file.columns)
    assigned = pd.read_csv(gc / "components" / "gc_a.features.tsv", sep="\t")
    feats = pd.read_csv(study["out"] / "features" / "gc_a.tsv", sep="\t")
    assert len(assigned) == len(feats) and (assigned["component_id"] >= 0).mean() > 0.95


def test_gc_needs_an_output_folder(tmp_path):
    r = _run("gc", str(tmp_path))
    assert r.returncode != 0 and "not a peak3d output folder" in r.stderr
    r = _run("gc", str(tmp_path), "--library", str(tmp_path / "missing.msp"))
    assert r.returncode != 0
