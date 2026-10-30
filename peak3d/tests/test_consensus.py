"""Consensus MS2 spectrum of a feature: precursor-weighted vote, noise kept out, the top spectrum's protection."""
import numpy as np
import pytest

from peak3d.consensus import build
from peak3d.io import ISOTOPE_DA, precursor_in_ms1

PREC = 300.0
TRUE = np.array([80.0, 120.0, 160.0, 200.0])


def _spec(shares, scale, extra=(), floor=None):
    """the true fragments at shares x scale, extra (m/z, share) peaks, and optionally a lowest recorded intensity"""
    m = list(TRUE) + [e[0] for e in extra]
    i = [s * scale for s in shares] + [e[1] * scale for e in extra]
    if floor is not None:                       # a tiny peak just below the cleaning cut sets the lowest recording
        m.append(299.0)
        i.append(floor)
    o = np.argsort(m)
    return np.array(m)[o], np.array(i)[o]


def _mz_set(c):
    return set(np.round(c.mz, 2))


SHARES = [0.4, 0.3, 0.2, 0.1]


def test_one_spectrum_is_kept_as_it_is():
    c = build([_spec(SHARES, 1e4)], [5e5], [np.nan], PREC)
    assert _mz_set(c) == set(TRUE) and c.used.tolist() == [0] and c.inten.sum() == pytest.approx(1.0)


def test_noise_of_any_single_spectrum_is_voted_out():
    sp = [_spec(SHARES, 1e4, [(95.31, 0.15)]), _spec(SHARES, 8e3, [(133.77, 0.15)]), _spec(SHARES, 6e3, [(171.05, 0.2)])]
    c = build(sp, [1e6, 6e5, 4e5], [np.nan] * 3, PREC)
    assert _mz_set(c) == set(TRUE) and c.n_dropped == 3 and c.status == ["used"] * 3


def test_protect_keeps_the_most_abundant_spectrum_whole():
    sp = [_spec(SHARES, 1e4, [(133.77, 0.15)]), _spec(SHARES, 8e3, [(95.31, 0.15)]), _spec(SHARES, 6e3, [(171.05, 0.2)])]
    c = build(sp, [4e5, 1e6, 6e5], [np.nan] * 3, PREC, protect=True)    # spectrum 1 has the top precursor
    assert c.top == 1 and _mz_set(c) == set(TRUE) | {95.31} and c.n_protected == 1 and c.n_dropped == 2
    lone = c.inten[np.isclose(c.mz, 95.31)][0]
    assert lone < 0.15 / 1.15 * 0.75             # protected, but the other spectra that could see it pull it down


def test_weak_spectra_abstain_strong_ones_vote_against():
    faint = [(140.0, 0.03)]
    a = _spec(SHARES, 1e5, faint, floor=10.0)                # both strong spectra show the 3 % fragment
    b = _spec(SHARES, 9e4, faint, floor=10.0)
    weak = _spec(SHARES, 1e3, floor=60.0)                    # lowest recording 6 % of its total: cannot show 3 %
    c = build([a, b, weak], [1e6, 9e5, 2e5], [np.nan] * 3, PREC)
    assert 140.0 in _mz_set(c)
    only_a = _spec(SHARES, 1e5, [(140.0, 0.05)], floor=10.0)     # in one strong spectrum, missing in another
    plain_b = _spec(SHARES, 9e4, floor=10.0)
    c = build([only_a, plain_b, weak], [1e6, 9e5, 2e5], [np.nan] * 3, PREC)
    assert 140.0 not in _mz_set(c)


def test_intensities_follow_the_precursor_weights():
    a = _spec([0.6, 0.2, 0.1, 0.1], 1e4)
    b = _spec([0.2, 0.6, 0.1, 0.1], 1e4)
    c = build([a, b], [3e5, 1e5], [np.nan] * 2, PREC, min_similarity=0.0)
    r = dict(zip(np.round(c.mz, 2), c.inten))
    assert r[80.0] / r[120.0] == pytest.approx((3 * 0.6 + 0.2) / (3 * 0.2 + 0.6), rel=1e-6)


def test_gates_purity_precursor_and_agreement():
    good = [_spec(SHARES, 1e4), _spec(SHARES, 9e3), _spec(SHARES, 8e3)]
    other = (np.array([60.0, 90.0, 110.0, 230.0]), np.array([1e4, 5e3, 3e3, 2e3]))      # another compound
    c = build(good + [other, _spec(SHARES, 1e4), _spec(SHARES, 1e4), _spec(SHARES, 1e4)],
              [1e6, 9e5, 8e5, 9.5e5, 7e5, 0.0, 5e3], [0.9, np.nan, 0.8, 0.9, 0.3, 0.9, 0.9], PREC,
              min_purity=0.5, min_similarity=0.6)
    assert c.status == ["used", "used", "used", "dissimilar", "purity", "no_precursor", "weak"]
    assert sorted(c.used.tolist()) == [0, 1, 2] and _mz_set(c) == set(TRUE)


def test_two_that_disagree_keep_the_more_abundant():
    other = (np.array([60.0, 90.0, 110.0, 230.0]), np.array([1e4, 5e3, 3e3, 2e3]))
    c = build([_spec(SHARES, 1e4), other], [2e5, 9e5], [np.nan] * 2, PREC, min_similarity=0.6)
    assert c.used.tolist() == [1] and c.status == ["dissimilar", "used"] and _mz_set(c) == {60.0, 90.0, 110.0, 230.0}


def test_without_precursor_intensities_tic_weights_and_no_gates():
    sp = [_spec(SHARES, 1e4, [(95.31, 0.15)]), _spec(SHARES, 8e3)]
    c = build(sp, [np.nan, np.nan], [np.nan, np.nan], PREC)
    assert c.status == ["used", "used"] and _mz_set(c) == set(TRUE) and c.top == 0


def test_precursor_in_ms1():
    mz = np.array([299.5, PREC, PREC + ISOTOPE_DA, 300.4, 302.5])
    it = np.array([2e4, 1e5, 2e4, 3e4, 5e4])
    p_int, purity = precursor_in_ms1(mz, it, PREC + 0.001, PREC - 0.5, PREC + 1.5)
    assert p_int == 1e5 and purity == pytest.approx((1e5 + 2e4) / (2e4 + 1e5 + 2e4 + 3e4))
    assert np.isnan(precursor_in_ms1(mz, it, PREC, np.nan, np.nan)[1])         # window unknown
    assert precursor_in_ms1(mz, it, 301.2, 300.7, 301.7)[0] == 0.0               # not in the MS1 scan
    assert np.isnan(precursor_in_ms1(None, None, PREC, 299.5, 300.5)[0])         # no MS1 scan before


def test_defaults_merge_all_and_record_purity_only():
    good = [_spec(SHARES, 1e4), _spec(SHARES, 9e3)]
    c = build(good, [1e6, 9e5], [0.2, 0.9], PREC)          # low purity: kept by default, the gate is off
    assert c.status == ["used", "used"] and len(c.used) == 2


def test_two_unrelated_spectra_without_gate_fall_back_to_the_top_one():
    other = (np.array([60.0, 90.0, 110.0, 230.0]), np.array([1e4, 5e3, 3e3, 2e3]))
    c = build([_spec(SHARES, 1e4), other], [2e5, 9e5], [np.nan] * 2, PREC)
    assert c.used.tolist() == [1] and _mz_set(c) == {60.0, 90.0, 110.0, 230.0}     # nothing confirmed: the top one
