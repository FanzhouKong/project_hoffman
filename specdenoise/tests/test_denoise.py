import numpy as np

from specdenoise import Params, denoise
from specdenoise import formula as fm
from specdenoise.denoise import electronic_mask


def spectrum(formula, adduct, losses, noise_losses, iso=None):
    p, z = fm.precursor_formula(fm.parse(formula), adduct)
    pmz = fm.ion_mz(p, z)
    mz = [fm.ion_mz(p - fm.parse(l), z) for l in losses]
    inten = [100.0 - 5 * i for i in range(len(mz))]
    for d in noise_losses:
        mz.append(pmz - d)
        inten.append(20.0)
    true = [True] * len(losses) + [False] * len(noise_losses)
    if iso is not None:
        mz.append(mz[iso] + fm.C13)
        inten.append(inten[iso] * 0.1)
        true.append(True)
    order = np.argsort(mz)
    return np.array(mz)[order], np.array(inten)[order], np.array(true)[order], pmz


LOSSES = ["H2O", "NH3", "CO2", "C2H4O2", "CH3", "H2O2", "CH5N3"]        # the last is a radical-free guanidine loss
NOISE = [0.5, 3.37, 7.77, 11.11, 25.3]                                  # forbidden neutral losses


def test_t1_keeps_true_removes_forbidden():
    mz, inten, true, pmz = spectrum("C4H9N3O2", "[M+H]+", LOSSES, NOISE, iso=0)
    r = denoise(mz, inten, pmz, formula="C4H9N3O2", adduct="[M+H]+")
    assert r.applied and r.tier == 1
    assert (r.keep == true).all(), list(zip(mz, r.reasons, true))
    assert "isotope" in r.reasons


def test_t1_precursor_and_above():
    mz, inten, true, pmz = spectrum("C4H9N3O2", "[M+H]+", ["H2O"], [])
    mz = np.append(mz, [pmz, pmz + 0.4]); inten = np.append(inten, [50, 10])
    r = denoise(mz, inten, pmz, formula="C4H9N3O2", adduct="[M+H]+")
    assert r.reasons[-2] == "precursor" and r.reasons[-1] == "above_precursor" and r.keep[-2] and not r.keep[-1]


def test_t1_adduct_loss_without_element_in_m():
    # [M+NH4]+ of glucose loses NH3 although glucose has no nitrogen
    mz, inten, true, pmz = spectrum("C6H12O6", "[M+NH4]+", ["NH3", "NH3H2O"], NOISE)
    r = denoise(mz, inten, pmz, formula="C6H12O6", adduct="[M+NH4]+")
    assert (r.keep == true).all()


def test_t0_infers_formula_and_denoises():
    mz, inten, true, pmz = spectrum("C4H9N3O2", "[M+H]+", LOSSES, NOISE, iso=0)
    r = denoise(mz, inten, pmz, params=Params(top_k=3))
    assert r.applied and r.tier == 0
    names = [fm.format(c) for c in r.candidates]
    assert "C4H10N3O2" in names[:3], names[:5]
    assert (r.keep == true).all(), list(zip(mz, r.reasons, true, r.rank))


def test_t0_falls_back_when_formula_disagrees():
    mz, inten, true, pmz = spectrum("C4H9N3O2", "[M+H]+", LOSSES, NOISE)
    r = denoise(mz, inten, pmz, formula="C6H12O6", adduct="[M+H]+")       # wrong formula for this precursor m/z
    assert r.tier == 0 and (r.keep == true).all()


def test_multiply_charged_untouched():
    r = denoise([100.0, 200.0], [1.0, 1.0], 300.0, formula="C20H40N2O4", adduct="[M+2H]2+")
    assert not r.applied and r.keep.all()


def test_electronic_mask():
    inten = np.array([10, 10.005, 9.999, 10.0, 50, 51, 52, 53, 1, 2, 3])
    m = electronic_mask(inten, 4)
    assert m[:4].all() and not m[4:].any()


def test_t1_halide_exchange():
    # 7-bromo-2-chloroquinazoline [M+H]+: the HCl -> H2O exchange product (loss 17.966 Da) is real
    p, z = fm.precursor_formula(fm.parse("C8H4BrClN2"), "[M+H]+")
    pmz = fm.ion_mz(p, z)
    mz = np.array([pmz - 17.9662, pmz - 35.9767]); inten = np.array([100.0, 50.0])
    r = denoise(mz, inten, pmz, formula="C8H4BrClN2", adduct="[M+H]+")
    assert r.keep.all()
    r = denoise(mz, inten, pmz, formula="C8H4BrClN2", adduct="[M+H]+", params=Params(exchange=False))
    assert not r.keep[0] and r.keep[1]


def test_precursor_isotopologue_kept():
    mz, inten, true, pmz = spectrum("C4H9N3O2", "[M+H]+", ["H2O"], [])
    mz = np.append(mz, [pmz, pmz + fm.C13]); inten = np.append(inten, [50, 3])
    r = denoise(mz, inten, pmz, formula="C4H9N3O2", adduct="[M+H]+")
    assert r.keep.all() and r.reasons[-1] == "isotope"


def test_unsorted_input():
    mz, inten, true, pmz = spectrum("C4H9N3O2", "[M+H]+", LOSSES, NOISE, iso=0)
    perm = np.random.default_rng(0).permutation(len(mz))
    r = denoise(mz[perm], inten[perm], pmz, formula="C4H9N3O2", adduct="[M+H]+")
    assert (r.keep == true[perm]).all()


def test_free_and_margin_modes():
    mz, inten, true, pmz = spectrum("C4H9N3O2", "[M+H]+", LOSSES, NOISE)
    free = denoise(mz, inten, pmz, params=Params(tier="free"))
    assert free.tier == 0 and (free.keep == true).all()
    marg = denoise(mz, inten, pmz, params=Params(margin=0.05, top_k=10))
    assert (marg.keep == true).all()
    forced = denoise(mz, inten, pmz, formula="C4H9N3O2", adduct="[M+H]+", params=Params(tier="t0"))
    assert forced.tier == 0 and (forced.keep == true).all()


def test_t1_recalibrates_on_observed_precursor():
    # the whole spectrum is 4 mDa high (calibration offset); losses measured from the observed precursor cancel it
    mz, inten, true, pmz = spectrum("C30H46O4", "[M-H]-", ["CO2", "H2O", "C2H4O2"], [])
    r = denoise(mz + 0.004, inten, pmz + 0.004, formula="C30H46O4", adduct="[M-H]-", params=Params(tol_ppm=5, tol_abs=0.002))
    assert r.keep.all()
    r = denoise(mz + 0.004, inten, pmz + 0.004, formula="C30H46O4", adduct="[M-H]-", params=Params(tol_ppm=5, tol_abs=0.002, recalibrate=False))
    assert not r.keep.all()


def test_t1_multisodium_fragments_kept():
    """[M-2H+3Na]+ fragments keep two or three Na although the Golden Rules cap Na at 1 for neutral molecules"""
    from specdenoise import formula as fm
    f = fm.parse("C6H8O7")                                     # citric acid
    p, z = fm.precursor_formula(f, "[M-2H+3Na]+")
    pmz = fm.ion_mz(p, z)
    frag = p.copy(); frag[fm.IDX["H"]] -= 2; frag[fm.IDX["O"]] -= 1   # loss of H2O: fragment still carries 3 Na
    frag2 = p.copy(); frag2[fm.IDX["C"]] -= 1; frag2[fm.IDX["O"]] -= 2  # loss of CO2: 3 Na
    frag3 = p.copy(); frag3[fm.IDX["C"]] -= 2; frag3[fm.IDX["H"]] -= 1; frag3[fm.IDX["O"]] -= 2; frag3[fm.IDX["Na"]] -= 1  # loss of NaOAc-like C2H1O2Na... keep 2 Na
    mz = np.array([fm.ion_mz(frag, z), fm.ion_mz(frag2, z), fm.ion_mz(frag3, z), pmz - 7.777])
    it = np.array([100.0, 80.0, 60.0, 40.0])
    r = denoise(mz, it, pmz, formula="C6H8O7", adduct="[M-2H+3Na]+")
    assert r.tier == 1
    assert r.keep.tolist() == [True, True, True, False]


def test_multi_sodium_fragments_kept():
    """[M-2H+3Na]+ of inositol triphosphate: fragments carrying 2-3 Na are permitted (Na cap raised above 1)"""
    from specdenoise import formula as fm
    p = fm.parse("C6H13O15P3Na3")
    pmz = fm.ion_mz(p, 1)
    frag = fm.parse("C6H12O12P2Na3")          # loss HPO3 keeps all three Na
    frag2 = fm.parse("C6H12O14P3Na2")         # loss NaOH... 
    mz = np.array([fm.ion_mz(frag, 1), fm.ion_mz(frag2, 1), pmz - 40.0])
    r = denoise(mz, np.array([100.0, 50.0, 10.0]), pmz, formula="C6H15O15P3", adduct="[M-2H+3Na]+")
    assert r.tier == 1 and r.keep[0] and r.keep[1]


def test_precursor_peak_recalibration():
    """spectrum 12 mDa off its stated precursor m/z: the observed precursor peak sets the calibration"""
    from specdenoise import formula as fm
    p = fm.parse("C9H9O4")                    # caffeic acid [M+H]+
    pmz = fm.ion_mz(p, 1)
    off = 0.012
    mz = np.array([pmz - 18.010565 - 43.989830, pmz - 18.010565, pmz]) + off
    r = denoise(mz, np.array([50.0, 100.0, 80.0]), pmz, formula="C9H8O4", adduct="[M+H]+", params=Params(precursor_peak=True))
    assert abs(r.offset - off) < 1e-6 and r.keep.all()
    r2 = denoise(mz, np.array([50.0, 100.0, 80.0]), pmz, formula="C9H8O4", adduct="[M+H]+")   # default: off
    assert r2.offset == 0.0 and not r2.keep[2] and not r2.keep[1]


def test_t1_inorganic_sodium_fragments_kept():
    """sugar-phosphate [M-H+2Na]+: the Na2PO3+ fragment (three element kinds) and the C6H12O6 loss are permitted"""
    from specdenoise import formula as fm
    f = fm.parse("C6H13O9P")                                    # mannose 6-phosphate
    p, z = fm.precursor_formula(f, "[M-H+2Na]+")
    pmz = fm.ion_mz(p, z)
    na2po3 = fm.parse("Na2PO3")
    mz = np.array([fm.ion_mz(na2po3, 1), pmz - 18.010565, pmz - 7.777])
    it = np.array([999.0, 300.0, 50.0])
    r = denoise(mz, it, pmz, formula="C6H13O9P", adduct="[M-H+2Na]+")
    assert r.tier == 1 and r.keep.tolist() == [True, True, False]


def test_precursor_peak_picks_closest_and_falls_back():
    """two peaks near the precursor: the closer one is the precursor; and a shift that explains less is dropped"""
    from specdenoise import formula as fm
    f = fm.parse("C9H8O4"); p, z = fm.precursor_formula(f, "[M+H]+"); pmz = fm.ion_mz(p, z)
    frags = np.array([pmz - 18.010565, pmz - 44.026215, pmz - 46.005479])        # H2O, C2H4O, CH2O2 losses, exact
    mz = np.concatenate([frags, [pmz - 0.0074, pmz - 0.0008]])                 # a stray ion 7.4 mDa low, precursor 0.8 mDa low
    it = np.array([999.0, 500.0, 300.0, 20.0, 8.0])
    r = denoise(mz, it, pmz, formula="C9H8O4", adduct="[M+H]+")
    assert r.keep.tolist() == [True, True, True, False, True]       # fragments and the precursor kept, the stray ion not
    # only the stray ion near the precursor: shifting by -7.4 mDa would lose every fragment, so the shift is dropped
    mz2 = np.concatenate([frags, [pmz - 0.0074]]); it2 = np.array([999.0, 500.0, 300.0, 20.0])
    r2 = denoise(mz2, it2, pmz, formula="C9H8O4", adduct="[M+H]+")
    assert r2.offset == 0.0 and r2.keep[:3].all()
