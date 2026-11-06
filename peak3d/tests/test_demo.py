"""The demo data: MS2 in the runs and a library whose spectra the denoising and the search can use."""
import numpy as np

from peak3d import ms2 as ms2mod
from peak3d.demo import COMPOUNDS, ISOMER, LIBRARY, write_demo
from peak3d.io import load_cloud
from peak3d.library import read_library
from specdenoise import formula as fm
from specdenoise.denoise import denoise
from specdenoise.smiles import smiles_formula


def test_compounds_are_consistent():
    """formula = the SMILES' formula, every fragment a sub-formula of [M+H]+ that the T1 rule keeps"""
    for c in (*COMPOUNDS, ISOMER):
        got = smiles_formula(c.smiles)
        assert got is not None and fm.format(got[0]) == c.formula, c.name
        p = fm.parse(c.formula)
        p[fm.H] += 1
        for f, _ in c.fragments:
            assert (fm.parse(f) <= p).all(), (c.name, f)
        t = c.template()
        r = denoise(t[:, 0], t[:, 1], c.mz, formula=c.formula, adduct="[M+H]+")
        assert r.applied and r.keep.all(), (c.name, t[~r.keep, 0])


def test_write_demo(tmp_path):
    d = write_demo(tmp_path)
    assert sorted(p.name for p in d.iterdir()) == ["demo1.mzML", "demo2.mzML", "demo3.mzML"]
    lib = read_library(tmp_path / LIBRARY)
    assert sorted(lib.names[:]) == sorted(c.name for c in (*COMPOUNDS, ISOMER))
    assert np.allclose(sorted(lib.precursor_mz), sorted(c.mz for c in (*COMPOUNDS, ISOMER)))
    col = ms2mod.Collector()
    load_cloud(d / "demo1.mzML", ms2=col)
    prec = np.array([s[2] for s in col])
    assert len(col) >= 2 * 40 and col.n_skipped == 0                       # two spectra per fragmented compound
    for c in COMPOUNDS:                                                    # the metabolites are all fragmented
        assert (np.abs(prec - c.mz) / c.mz * 1e6 < 10).sum() == 2, c.name
    assert (lib.mode == 1).all() and all(lib.formula[:])
