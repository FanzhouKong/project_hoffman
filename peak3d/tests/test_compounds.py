import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from peak3d import compounds as C
from peak3d.synth import Peak, make_cloud, write_mzml

ROOT = Path(__file__).resolve().parents[2]
DT = 0.005


def test_pearson_masks_double_zeros_and_needs_three_points():
    a = np.array([0, 0, 1, 2, 3, 0], np.float32)
    b = np.array([0, 0, 2, 4, 6, 0], np.float32)
    assert np.isclose(C._pearson(a, b), 1.0)
    assert np.isnan(C._pearson(np.array([0, 1], np.float32), np.array([0, 1], np.float32)))
    assert np.isnan(C._pearson(np.array([1, 1, 1], np.float32), np.array([1, 2, 3], np.float32)))


def test_candidate_pairs_need_coelution_overlap_and_a_shared_file():
    rt = np.array([1.00, 1.01, 1.30, 1.02])
    rt_min = rt - 0.05
    rt_max = rt + 0.05
    det = np.array([[1, 1], [1, 0], [1, 1], [0, 1]], bool)       # rows 1 and 3 share no file
    packed = np.zeros((4, 1), np.uint64)
    for f in range(2):
        packed[:, 0] |= det[:, f].astype(np.uint64) << np.uint64(f)
    I, J, trunc = C._candidate_kernel(rt, rt_min, rt_max, np.argsort(rt), 0.05, packed, 100)
    pairs = set(zip(I.tolist(), J.tolist()))
    assert not trunc
    assert pairs == {(0, 1), (0, 3)}            # 2 is 0.3 min away; (1, 3) share no detected file


def test_pick_files_takes_strongest_shared_files():
    H = np.array([[10, 1, 5, 7], [10, 9, 5, 0]], np.float64)
    det = np.array([[1, 1, 1, 1], [1, 1, 1, 0]], bool)
    tp, tf = C._pick_files_kernel(np.array([0]), np.array([1]), H, det, 2)
    assert sorted(tf.tolist()) == [0, 2] and tp.tolist() == [0, 0]


def test_star_clusters_do_not_chain():
    # 0-1 and 1-2 are edges, 0-2 is not: the strongest (0) claims 1; 2 seeds its own cluster
    src = np.array([0, 1, 1, 2]); dst = np.array([1, 0, 2, 1])
    indptr = np.array([0, 1, 3, 4])
    order = np.array([0, 1, 2])
    cid, n = C._star_kernel(order, indptr, dst, np.ones(3, bool))
    assert cid.tolist() == [0, 0, 1] and n == 2


def test_roles_name_adducts_isotopes_and_leave_deltas():
    # cluster 0: [M+H]+ 300.1, [M+Na]+ 322.082, in-source fragment 256.07 (no relation in the table), M+1 of the base
    M = 299.0927
    mz = np.array([M + C.PROTON, M + 22.9892207, 256.0700, M + C.PROTON + C.C13, 500.0])
    cid = np.array([0, 0, 0, 0, 1])
    h = np.array([1e6, 2e5, 1e5, 1.5e5, 1e4])
    g_iso = np.array([0, 0, 0, 1, 0]); g_z = np.ones(5, np.int64); par = np.array([-1, -1, -1, 0, -1])
    roles, neutral, is_base = C._assign_roles(cid, 2, mz, h, g_iso, g_z, par, "pos", 5.0)
    assert roles[0] == "[M+H]+" and roles[1] == "[M+Na]+" and roles[3] == "[M+H]+ M+1"
    assert roles[2] == f"{mz[2] - mz[0]:+.4f}"
    assert np.isclose(neutral[0], M, atol=1e-4) and np.isnan(neutral[2])
    assert is_base.tolist() == [True, False, False, False, True] and roles[4] == ""


def test_roles_without_any_relation_mark_base_and_deltas():
    mz = np.array([400.0, 377.123, 350.5])
    roles, neutral, is_base = C._assign_roles(np.zeros(3, int), 1, mz, np.array([1.0, 3.0, 2.0]),
                                              np.zeros(3, int), np.ones(3, int), -np.ones(3, int), "neg", 5.0)
    assert roles[1] == "base" and roles[0] == "+22.8770" and roles[2] == "-26.6230"
    assert np.all(np.isnan(neutral))


def test_group_iso_parent_majority():
    # two files; group 1 is the M+1 of group 0 in both; group 2 claims group 0 in one file only but is monoisotopic
    A = pd.DataFrame({"file": [0, 0, 1, 1, 1], "row": [0, 1, 0, 1, 2], "iso_parent": [-1, 0, -1, 0, 0]})
    gid = np.array([0, 1, 0, 1, 2])
    out = C.group_iso_parent(A, gid, 3, np.array([0, 1, 0]))
    assert out.tolist() == [-1, 0, -1]


def test_pair_relations_hold_na_h_and_c13():
    rel = C.pair_relations("pos")
    assert np.any(np.isclose(rel, 22.9892207 - C.PROTON, atol=1e-6))
    assert np.any(np.isclose(rel, C.C13, atol=1e-6))


def _study(tmp_path, n_files=4):
    """4 runs: compound A = [M+H]+, [M+Na]+, a fragment and M+1 sharing one profile; compound B co-eluting
    0.6 FWHM later with its own [M+H]+ / M+1; compound D far away. Heights vary per file, A and B independently."""
    rng = np.random.default_rng(1)
    d = tmp_path / "data"
    d.mkdir()
    MA, MB, MD = 299.0927, 451.2210, 700.3456
    for k in range(n_files):
        a = 10 ** rng.uniform(5.0, 6.0); b = 10 ** rng.uniform(5.0, 6.0)
        sh = 0.01 * k
        peaks = [Peak(MA + C.PROTON, 1.0 + sh, a, 0.05, iso=(0.17,)),
                 Peak(MA + 22.9892207, 1.0 + sh, 0.3 * a, 0.05, iso=(0.17,)),
                 Peak(256.0700, 1.0 + sh, 0.2 * a, 0.05),
                 Peak(MB + C.PROTON, 1.0 + sh + 0.03, b, 0.05, tau=0.03, iso=(0.25,)),
                 Peak(MD + C.PROTON, 2.0 + sh, 5e5, 0.05, iso=(0.4,))]
        # background compounds so the picker's estimators have something to work with
        # background compounds, kept away from A/B's RT: a clean synthetic Gaussian at the same apex is
        # indistinguishable by shape, which is the method's known limit, not what this test checks
        bg_rt = rng.uniform(0.5, 3.5, 40)
        bg_rt = np.where(np.abs(bg_rt - 1.0) < 0.15, bg_rt + 0.3, bg_rt)
        peaks += [Peak(float(m), float(r + sh), float(h), 0.05, iso=(0.15,)) for m, r, h in
                  zip(rng.uniform(150, 900, 40), bg_rt, 10 ** rng.uniform(4.5, 6.0, 40))]
        cloud, _ = make_cloud(peaks, n_scans=800, dt=DT, thresh=200.0, floor=100.0, n_noise=20000, seed=k,
                              name=f"S{k}")
        write_mzml(cloud, d / f"S{k}.mzML")
    return d, (MA, MB, MD)


def test_end_to_end_compound_columns(tmp_path):
    d, (MA, MB, MD) = _study(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "peak3d", "process", "--input", str(d), "--output", str(out),
                        "--cores", "2"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    for c in ("compound_id", "role", "neutral_mass", "is_base", "n_members"):
        assert c in t.columns
    assert (out / "compounds.tsv").exists() and (out / "compound_pairs.tsv").exists()

    def row(m):
        k = np.flatnonzero(np.abs(t["mz"].values - m) / m * 1e6 < 5)
        assert len(k) >= 1, f"no feature at {m}"
        return t.iloc[k[np.argmax(t["n_detected"].values[k])]]
    a_h, a_na, a_frag, a_m1 = (row(MA + C.PROTON), row(MA + 22.9892207), row(256.0700), row(MA + C.PROTON + C.C13))
    assert a_h["compound_id"] == a_na["compound_id"] == a_frag["compound_id"] == a_m1["compound_id"]
    assert a_h["role"] == "[M+H]+" and a_na["role"] == "[M+Na]+" and a_m1["role"] == "[M+H]+ M+1"
    assert a_frag["role"].startswith("-44.0")
    assert a_h["is_base"] == 1 and np.isclose(a_h["neutral_mass"], MA, atol=2e-3)
    assert row(MD + C.PROTON)["compound_id"] != a_h["compound_id"]
    # the per-file tables carry the same ids
    f0 = pd.read_csv(out / "features" / "S0.tsv", sep="\t")
    assert "compound_id" in f0.columns and "role" in f0.columns
    comp = pd.read_csv(out / "compounds.tsv", sep="\t")
    me = comp[comp["compound_id"] == a_h["compound_id"]].iloc[0]
    assert me["n_members"] >= 4 and "[M+Na]+" in me["roles"]
    # B co-elutes 0.6 FWHM later with a different profile: it must not be swallowed
    assert row(MB + C.PROTON)["compound_id"] != a_h["compound_id"]
    # --no-compounds leaves the columns out
    out2 = tmp_path / "out2"
    r = subprocess.run([sys.executable, "-m", "peak3d", "process", "--input", str(d), "--output", str(out2),
                        "--cores", "1", "--no-compounds"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "compound_id" not in pd.read_csv(out2 / "aligned_feature_table.tsv", sep="\t").columns
