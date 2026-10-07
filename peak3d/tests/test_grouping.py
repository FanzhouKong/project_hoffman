import numpy as np

from peak3d.grouping import consensus, group_features, merge_complementary


def test_isomer_doublet_in_every_file_gives_two_groups():
    n_files = 5
    mz, rt, h, f = [], [], [], []
    for k in range(n_files):
        for r in (2.0, 2.3):
            mz.append(400.0 * (1 + 1e-7 * k)); rt.append(r + 0.001 * k); h.append(1e5); f.append(k)
    gid, cons = group_features(np.array(mz), np.array(rt), np.array(rt) - 0.02, np.array(rt) + 0.02,
                               np.array(h), np.array(f), n_files, ppm_tol=5.0, rt_tol=0.05)
    assert len(cons["mz"]) == 2
    assert sorted(cons["n_detected"].tolist()) == [5, 5]
    assert sorted(np.round(cons["rt"], 2).tolist()) == [2.0, 2.3]


def test_same_file_duplicate_resolved_to_closest_in_rt():
    # file 0 has the seed; file 1 has two features within rt_tol, the closer one must be chosen
    mz = np.array([500.0, 500.0, 500.0])
    rt = np.array([3.0, 3.03, 3.01])
    h = np.array([1e6, 5e5, 1e5])
    f = np.array([0, 1, 1])
    gid, cons = group_features(mz, rt, rt, rt, h, f, 2, ppm_tol=5.0, rt_tol=0.05)
    assert gid[0] == gid[2] and gid[1] != gid[0]
    assert len(cons["mz"]) == 2


def test_consensus_mz_is_height_weighted_median_and_dense_ids():
    mz = np.array([400.0, 400.0004, 400.0008])
    rt = np.array([1.0, 1.0, 1.0])
    h = np.array([1.0, 10.0, 1.0])
    f = np.array([0, 1, 2])
    gid, cons = group_features(mz, rt, rt, rt, h, f, 3, ppm_tol=5.0, rt_tol=0.1)
    assert len(cons["mz"]) == 1 and cons["mz"][0] == 400.0004
    assert cons["n_detected"][0] == 3 and set(gid.tolist()) == {0}
    assert cons["mz_ppm_spread"][0] > 1.9


def test_empty_input():
    gid, cons = group_features(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, int),
                               1, 5.0, 0.1)
    assert len(gid) == 0 and len(cons["mz"]) == 0


def test_sigma_widens_tolerance_for_weak_features_only():
    # the same weak ion in three files scatters by 8 ppm; strong ions must still be held to 5 ppm
    mz = np.array([400.0, 400.0032, 399.9968, 600.0, 600.0048])
    rt = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    h = np.array([800.0, 500.0, 500.0, 1e6, 1e6])   # the central, strongest one seeds the group
    f = np.array([0, 1, 2, 0, 1])
    sig = np.array([3.0, 3.0, 3.0, 0.5, 0.5])        # ppm sigma per feature
    gid, cons = group_features(mz, rt, rt, rt, h, f, 3, ppm_tol=5.0, rt_tol=0.1, sig=sig)
    assert gid[0] == gid[1] == gid[2]                # weak trio grouped (tol 3*sqrt(18) = 12.7 ppm)
    assert gid[3] != gid[4]                          # strong pair 8 ppm apart stays separate
    assert len(cons["mz"]) == 3


def test_broad_peaks_group_by_overlap_narrow_ones_do_not():
    # same m/z; apexes 3 s apart. Broad peaks (base width 12 s) overlap -> one group; narrow ones (2 s) -> two
    mz = np.array([400.0, 400.0, 500.0, 500.0])
    rt = np.array([2.0, 2.05, 3.0, 3.05])
    f = np.array([0, 1, 0, 1])
    h = np.array([1e5, 1e5, 1e5, 1e5])
    w_b, w_n = 12.0 / 60, 2.0 / 60
    rt_min = rt - np.array([w_b, w_b, w_n, w_n]) / 2
    rt_max = rt + np.array([w_b, w_b, w_n, w_n]) / 2
    gid, cons = group_features(mz, rt, rt_min, rt_max, h, f, 2, ppm_tol=5.0, rt_tol=1.0 / 60)
    assert gid[0] == gid[1] and gid[2] != gid[3]


def test_complementary_fragments_merge_but_isomers_do_not():
    # ion A: files 0,1 at rt 2.00 and files 2,3 at rt 2.06 (apex wandered; disjoint files) -> one group
    # ion B: isomers at 3.00 and 3.06 both present in files 0..3 -> two groups stay
    mz = np.array([400.0] * 4 + [500.0] * 8)
    rt = np.array([2.0, 2.0, 2.06, 2.06] + [3.0, 3.0, 3.0, 3.0, 3.06, 3.06, 3.06, 3.06])
    f = np.array([0, 1, 2, 3] + [0, 1, 2, 3, 0, 1, 2, 3])
    h = np.ones(12) * 1e5
    w = 0.1  # base width 6 s -> overlap window 0.5*(w+w) = 6 s > 3.6 s separation
    gid, cons = group_features(mz, rt, rt - w / 2, rt + w / 2, h, f, 4, ppm_tol=5.0, rt_tol=0.5 / 60)
    assert len(cons["mz"]) == 4                      # first pass keeps all four apart (tight rt_tol)
    g2 = merge_complementary(gid, cons, f, 4, 5.0, 0.5 / 60, np.full(len(cons["mz"]), 0.5))
    c2 = consensus(g2, mz, rt, rt - w / 2, rt + w / 2, h)
    assert len(c2["mz"]) == 3
    assert g2[0] == g2[1] == g2[2] == g2[3]
    assert g2[4] != g2[8]
