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



def test_group_iso_parent_parent_row_missing_from_alignment():
    """an isotope whose parent row was dropped before alignment (e.g. by the m/z offset check) gets no parent
    instead of an out-of-range lookup"""
    A = pd.DataFrame({"file": [0, 0, 1, 1], "row": [0, 1, 0, 1], "iso_parent": [-1, 0, -1, 7]})   # file 1: row 7 absent
    gid = np.array([0, 1, 0, 1])
    out = C.group_iso_parent(A, gid, 2, np.array([0, 1]))
    assert out.tolist() == [-1, 0]


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


def test_pooled_sums_match_numpy_on_scaled_concatenated_traces():
    """the pooled shape score equals Pearson r of the two traces scaled per file and joined end to end"""
    rng = np.random.default_rng(3)
    acc = np.zeros((1, 6))
    xs, ys = [], []
    for f in range(4):
        n = 12 + f
        x = np.abs(rng.normal(size=n)).astype(np.float32) * (10 ** f); y = (x * 0.3 + np.abs(rng.normal(size=n)) * 0.2).astype(np.float32)
        E = np.concatenate([x, y]); eoff = np.array([0, n, 2 * n]); ws0 = np.array([0, 0])
        lo = np.array([1, 2]); hi = np.array([n - 2, n - 1])
        sums = C._pair_sums_kernel(E, eoff, ws0, lo, hi, np.array([0]), np.array([1]))
        acc += sums
        s0, s1 = min(lo), max(hi) + 1
        xs.append(x[s0:s1] / x[s0:s1].max()); ys.append(y[s0:s1] / y[s0:s1].max())
    r = C._pooled_r(acc)[0]
    assert np.isclose(r, np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1], atol=1e-5)
    # flat or short traces give nothing
    E = np.array([1, 1, 1, 1, 2, 3, 4, 5], np.float32); eoff = np.array([0, 4, 8]); ws0 = np.array([0, 0])
    assert C._pair_sums_kernel(E, eoff, ws0, np.array([0, 0]), np.array([3, 3]), np.array([0]), np.array([1]))[0, 0] == 4
    assert np.isnan(C._pooled_r(np.zeros((1, 6)))[0])


def test_tasks_for_file():
    det = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], bool)
    I = np.array([0, 0, 1]); J = np.array([1, 2, 2])
    assert C._tasks_for_file(I, J, det, 0).tolist() == [0]
    assert C._tasks_for_file(I, J, det, 1).tolist() == [2]
    assert C._tasks_for_file(I, J, det, 2).tolist() == [1]


def test_int_ratio_kernel():
    """fraction of shared files whose height ratio sits within a factor of 2 of the pair's median ratio"""
    H = np.array([[1e5, 1e5, 1e5, 1e5, 1e5], [2e4, 2e4, 2e4, 2e4, 2e4], [2e4, 2e5, 2e3, 2e4, 2e4], [1e4, 1e4, 0, 0, 0]])
    L = np.where(H > 0, np.log10(np.maximum(H, 1e-30)), 0.0)
    present = H > 0
    I = np.array([0, 0, 0]); J = np.array([1, 2, 3])
    out = C._int_ratio_kernel(L, present, I, J, 3, C.RATIO_TOL)
    assert out[0] == 1.0 and abs(out[1] - 0.6) < 1e-9 and np.isnan(out[2])      # 2 shared files < 3


def test_consistent_clusters_moves_and_splits():
    """a member that agrees better with another group moves there; one that disagrees with its own group leaves;
    agreement is height-weighted, so a tiny group with one strong pair cannot pull a member away"""
    # stars: group 0 = {0, 1, 2}, group 1 = {3, 4}; member 2 pairs weakly with 0 and 1 but strongly with 3 and 4;
    # member 1 pairs weakly with everything in its group
    I = np.array([0, 0, 1, 2, 2, 3, 0]); J = np.array([1, 2, 2, 3, 4, 4, 3])
    r = np.array([0.5, 0.6, 0.4, 0.95, 0.9, 0.97, 0.2])
    cid = np.array([0, 0, 0, 1, 1]); h = np.array([1e6, 1e5, 1e5, 5e5, 2e5], float)
    out, C_, moved, left, it = C._consistent_clusters(cid, 2, I, J, r, 0.8, np.ones(5, bool), h)
    assert out[2] == out[3] == out[4] and out[0] != out[2]
    assert out[1] != out[0]                                   # left its group
    assert moved == 1 and left >= 1 and C_ == len(set(out.tolist()))
    # an unassigned member (isotope child) stays unassigned
    cid2 = np.array([0, 0, -1, 1, 1]); elig2 = np.array([1, 1, 0, 1, 1], bool)
    assert C._consistent_clusters(cid2, 2, I, J, r, 0.8, elig2, h)[0][2] == -1
    # a true group {base, adduct, fragment}: the two weak members pair noisily with each other but strongly with the
    # heavy base, and a tiny unrelated group offers one strong pair: nobody moves, nobody leaves
    I = np.array([0, 0, 1, 1]); J = np.array([1, 2, 2, 3]); r = np.array([0.95, 0.93, 0.5, 0.97])
    cid = np.array([0, 0, 0, 1]); h = np.array([1e7, 2e5, 1e5, 3e4], float)
    out, _, moved, left, _ = C._consistent_clusters(cid, 2, I, J, r, 0.85, np.ones(4, bool), h)
    assert moved == 0 and left == 0 and out[0] == out[1] == out[2] != out[3]
    # moves only: the first case's member 2 still moves, member 1 is not evicted
    I = np.array([0, 0, 1, 2, 2, 3, 0]); J = np.array([1, 2, 2, 3, 4, 4, 3])
    r = np.array([0.5, 0.6, 0.4, 0.95, 0.9, 0.97, 0.2])
    out, _, moved, left, _ = C._consistent_clusters(np.array([0, 0, 0, 1, 1]), 2, I, J, r, 0.8, np.ones(5, bool),
                                                    np.array([1e6, 1e5, 1e5, 5e5, 2e5], float), allow_leave=False)
    assert moved == 1 and left == 0 and out[2] == out[3] and out[1] == out[0]


def test_end_to_end_pooled_consistent(tmp_path):
    """the alternative modes run through the CLI and group the synthetic compound the same way"""
    d, (MA, MB, MD) = _study(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "peak3d", "process", "--input", str(d), "--output", str(out), "--cores", "2",
                        "--compound-shape", "pooled", "--compound-shape-files", "0", "--compound-cluster", "consistent",
                        "--compound-veto", "ratio"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")

    def row(m):
        k = np.flatnonzero(np.abs(t["mz"].values - m) / m * 1e6 < 5)
        assert len(k) >= 1, f"no feature at {m}"
        return t.iloc[k[np.argmax(t["n_detected"].values[k])]]
    a_h, a_na, a_frag, a_m1 = (row(MA + C.PROTON), row(MA + 22.9892207), row(256.0700), row(MA + C.PROTON + C.C13))
    assert a_h["compound_id"] == a_na["compound_id"] == a_frag["compound_id"] == a_m1["compound_id"]
    assert a_h["role"] == "[M+H]+" and a_na["role"] == "[M+Na]+"
    import json
    info = json.loads((out / "params.json").read_text())["stage_times_s"]["compounds"]
    assert info["shape_mode"] == "pooled" and info["cluster_mode"] == "consistent" and info["int_mode"] == "ratio"
