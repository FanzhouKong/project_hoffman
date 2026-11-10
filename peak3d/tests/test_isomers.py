import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from peak3d.gapfill import fill_file
from peak3d.isomers import clip_windows, proven_pairs, territories
from peak3d.synth import Peak, make_cloud, write_mzml
from peak3d.warp import TableWarp

ROOT = Path(__file__).resolve().parents[2]


def _peaks(groups):
    """groups: list of (files, rt or per-file rts) -> gid, file_idx, rt arrays"""
    gid, fi, rt = [], [], []
    for g, (files, r) in enumerate(groups):
        r = np.broadcast_to(np.asarray(r, float), (len(files),))
        gid += [g] * len(files)
        fi += list(files)
        rt += list(r)
    return np.array(gid), np.array(fi), np.array(rt)


def _pairs(groups, mz=(300.0, 300.0), half=0.2, rt_tol=0.02, **kw):
    gid, fi, rt = _peaks(groups)
    g_rt = np.array([np.median(rt[gid == g]) for g in range(len(groups))])
    return proven_pairs(gid, fi, rt, np.array(mz[:len(groups)]), g_rt, np.full(len(groups), half), 5.0, rt_tol, **kw)


def test_pair_picked_apart_often_is_proven():
    p = _pairs([(range(20), 1.00), (range(10), 1.10)])
    assert len(p) == 1 and p.iloc[0]["n_codetected"] == 10 and p.iloc[0]["share"] == 1.0
    assert abs(p.iloc[0]["gap_s"] - 6.0) < 1e-6


def test_pair_picked_apart_in_few_injections_is_not_proven():
    # one compound whose peak the picker split in 4 of 40 injections: the second group is mostly elsewhere
    p = _pairs([(range(40), 1.00), (list(range(4)) + list(range(40, 56)), 1.10)])
    assert len(p) == 0
    assert len(_pairs([(range(40), 1.00), (list(range(4)) + list(range(40, 56)), 1.10)], min_share=0.1)) == 1
    assert len(_pairs([(range(40), 1.00), (range(2), 1.10)], min_share=0.0)) == 0      # min_codetected = 3


def test_inconsistent_order_small_gap_or_other_mz_is_not_a_pair():
    alternating = np.where(np.arange(10) % 2 == 0, 1.10, 0.90)
    assert len(_pairs([(range(10), 1.00), (range(10), alternating)])) == 0
    assert len(_pairs([(range(10), 1.00), (range(10), 1.01)])) == 0                      # gap 0.6 s < RT tolerance
    assert len(_pairs([(range(10), 1.00), (range(10), 1.10)], mz=(300.0, 300.01))) == 0  # 33 ppm apart


def test_territories_midpoints_and_a_chain_of_three():
    groups = [(range(10), 1.00), (range(10), 1.10), (range(10), 1.30)]
    gid, fi, rt = _peaks(groups)
    p = proven_pairs(gid, fi, rt, np.full(3, 300.0), np.array([1.0, 1.1, 1.3]), np.full(3, 0.2), 5.0, 0.02)
    assert len(p) == 3                       # 1.0-1.3 reach each other too
    lo, hi, pos = territories(p, gid, fi, rt, 3)
    assert np.allclose(pos, [1.0, 1.1, 1.3])
    assert lo[0] == -np.inf and np.isclose(hi[0], 1.05)
    assert np.isclose(lo[1], 1.05) and np.isclose(hi[1], 1.2)
    assert np.isclose(lo[2], 1.2) and hi[2] == np.inf


def test_groups_without_partner_keep_their_window_bit_for_bit():
    rng = np.random.default_rng(0)
    g_rt, g_half = rng.uniform(0.5, 10, 1000), rng.uniform(0.01, 0.2, 1000)
    lo, hi = np.full(1000, -np.inf), np.full(1000, np.inf)
    lo[5], hi[7] = g_rt[5] + 0.001, g_rt[7] - 0.001
    c, h, cut = clip_windows(g_rt, g_half, lo, hi)
    keep = ~cut
    assert cut.sum() == 2 and np.array_equal(c[keep], g_rt[keep]) and np.array_equal(h[keep], g_half[keep])
    assert c[5] > g_rt[5] and h[5] < g_half[5] and c[7] < g_rt[7]
    lo[9] = g_rt[9] + g_half[9] + 1.0                  # boundary past the whole window: nothing to read
    c, h, _ = clip_windows(g_rt, g_half, lo, hi)
    assert h[9] == 0.0


def test_missing_isomer_reads_only_its_own_side():
    # isomer A at 1.00 min is in the injection, isomer B (expected at 1.06) is not; B's window (1.06 +- 0.07)
    # holds A's apex, so the plain fill returns A's height; cut at the midpoint (1.03) it reads only A's flank
    cloud, _ = make_cloud([Peak(300.0, 1.00, 1e5, 0.04)], n_scans=600, dt=0.005, thresh=0.0)
    warp = TableWarp.identity(cloud.rt)
    h0, _, _ = fill_file(cloud, warp, np.array([300.0]), np.array([1.06]), np.array([0.07]), 5.0)
    c, half, _ = clip_windows(np.array([1.06]), np.array([0.07]), np.array([1.03]), np.array([np.inf]))
    h1, _, _ = fill_file(cloud, warp, np.array([300.0]), c, half, 5.0)
    assert h0[0] > 0.95e5 and h1[0] < 0.35e5


def test_end_to_end_isomer_pair_fill_and_unchanged_rest(tmp_path):
    # 8 injections: 40 compounds plus an isomer pair at m/z 412.1717 (A at 1.80 min, B 5 s later and tailing, so its
    # fill window reaches A's apex); B is missing from injections 5-7. Only B's gap-filled cells may differ from a run without the rule, and they must fall.
    rng = np.random.default_rng(3)
    base = [(float(m), float(r), float(h)) for m, r, h in
            zip(rng.uniform(150, 900, 40), rng.uniform(0.6, 3.4, 40), 10 ** rng.uniform(4.5, 6.0, 40))]
    base = [b for b in base if abs(b[0] - 412.1717) > 0.05]
    d = tmp_path / "data"
    d.mkdir()
    for k in range(8):
        sh = 0.01 * (k % 3 - 1)
        peaks = [Peak(m, r + sh, h, 0.05, iso=(0.15,)) for m, r, h in base]
        peaks.append(Peak(412.1717, 1.80 + sh, 2e5, 0.04))
        if k < 5:
            peaks.append(Peak(412.1717, 1.8833 + sh, 1.2e5, 0.04, tau=0.05))    # tailing: a wide fill window
        cloud, _ = make_cloud(peaks, n_scans=800, dt=0.005, thresh=200.0, floor=100.0, n_noise=15000, seed=k, name=f"S{k}")
        write_mzml(cloud, d / f"S{k}.mzML")

    def run(out, *extra):
        r = subprocess.run([sys.executable, "-m", "peak3d", "process", "--input", str(d), "--output", str(out),
                            "--no-report", "--cores", "2", *extra], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr + r.stdout
        return pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    new = run(tmp_path / "new")
    old = run(tmp_path / "old", "--no-isomer-fill")
    pairs = pd.read_csv(tmp_path / "new" / "isomer_pairs.tsv", sep="\t")
    assert not (tmp_path / "old" / "isomer_pairs.tsv").exists()
    stems = [f"S{k}" for k in range(8)]
    sel = (new["mz"] - 412.1717).abs() < 0.002
    a = new[sel & ((new["rt"] - 1.80).abs() < 0.02)]
    b = new[sel & ((new["rt"] - 1.883).abs() < 0.02)]
    assert len(a) == 1 and len(b) == 1
    ga, gb = int(a["group_id"].iloc[0]), int(b["group_id"].iloc[0])
    assert {ga, gb} == {int(pairs["group_a"].iloc[0]), int(pairs["group_b"].iloc[0])} and len(pairs) == 1
    assert list(new["group_id"]) == list(old["group_id"])
    N, O = new.set_index("group_id")[stems], old.set_index("group_id")[stems]
    other = N.index != gb
    assert np.array_equal(N[other].values, O[other].values)            # every other cell unchanged
    missing = [f"S{k}" for k in (5, 6, 7)]
    assert np.array_equal(N.loc[gb, stems[:5]].values, O.loc[gb, stems[:5]].values)
    assert (N.loc[gb, missing].values < 0.5 * O.loc[gb, missing].values).all()
