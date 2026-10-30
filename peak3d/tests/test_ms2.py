"""MS2 spectra: collected in the MS1 pass, mapped to aligned features, end to end through process + identify."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peak3d import ms2
from peak3d.io import load_cloud
from peak3d.library import read_msp, write_msp_record
from peak3d.synth import Peak, make_cloud, write_mzml

ROOT = Path(__file__).resolve().parents[2]
DT = 0.005


def _frag(seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(50, 140, 6)).round(4), rng.uniform(5, 100, 6)


# ------------------------------------------------------------------ reading ----
def test_ms2_collected_in_the_ms1_pass(tmp_path):
    cloud, _ = make_cloud([Peak(300.0, 1.0, 1e5, 0.05)], n_scans=200, dt=DT, floor=50.0, n_noise=500, name="m")
    m, i = _frag(1)
    spectra = [(0.5, 300.0012, m, i), (0.7501, 412.2, m + 1, i), (0.9, 500.0, np.zeros(0), np.zeros(0))]
    p = tmp_path / "m.mzML"
    write_mzml(cloud, p, ms2=spectra)
    col = ms2.Collector()
    back = load_cloud(p, None, col)
    plain = load_cloud(p)
    assert back.n_scans == cloud.n_scans and np.allclose(back.mz, plain.mz) and np.allclose(back.rt, plain.rt)
    assert len(col) == 2 and col.n_skipped == 1                          # the empty spectrum is skipped
    sid, rt, prec, pol, p_int, purity, mz, it = col[0]
    assert sid.startswith("scan=") and rt == pytest.approx(0.5, abs=1e-6) and prec == pytest.approx(300.0012)
    assert p_int >= 0 and np.isnan(purity)                              # the test writer gives no window offsets
    assert pol == "pos" and np.allclose(mz, m) and np.allclose(it, i) and it.dtype == np.float32
    s = ms2.from_collected(col, back.polarity)
    assert len(s) == 2 and s.polarity == "pos" and np.diff(s.off).tolist() == [6, 6]
    assert s.tic[0] == pytest.approx(i.sum(), rel=1e-5)
    ms2.save(s, tmp_path / "m.npz")
    r = ms2.load(tmp_path / "m.npz")
    assert r.scan.tolist() == s.scan.tolist() and np.allclose(r.mz, s.mz) and r.polarity == "pos"
    assert np.array_equal(r.prec_int, s.prec_int) and np.isnan(r.purity).all()


@pytest.mark.parametrize("unit", ["second", None])
def test_ms2_time_follows_the_ms1_unit_rule(tmp_path, unit):
    cloud, _ = make_cloud([Peak(300.0, 4.0, 1e5, 0.05)], n_scans=500, dt=0.02, floor=50.0, n_noise=500, name="u")
    m, i = _frag(2)
    p = tmp_path / "u.mzML"
    write_mzml(cloud, p, ms2=[(4.0, 300.0, m, i)], rt_unit=unit)        # 10 min run: 600 s, seconds guard kicks in
    col = ms2.Collector()
    back = load_cloud(p, None, col)
    assert back.rt.max() == pytest.approx(cloud.rt.max(), abs=1e-3)
    assert col[0][1] == pytest.approx(4.0, abs=1e-3)


# ------------------------------------------------------------------ mapping ----
def _fake_output(tmp_path):
    out = tmp_path / "out"
    (out / "ms2").mkdir(parents=True)
    pd.DataFrame({"group_id": [11, 12, 13], "mz": [300.0, 300.001, 400.0], "rt": [1.0, 1.02, 2.0],
                  "rt_min": [0.95, 0.99, 1.97], "rt_max": [1.05, 1.06, 2.03],
                  "A": [1e5, 5e5, 1e5], "B": [1e5, 0.0, 1e5]}).to_csv(out / "aligned_feature_table.tsv", sep="\t", index=False)
    m, i = _frag(3)
    big = i * 10
    nan = float("nan")
    a = [("scan=1", 1.01, 300.0005, "pos", nan, nan, m, i), ("scan=2", 2.0, 400.0, "pos", nan, nan, m, big),
         ("scan=3", 2.01, 400.0, "pos", nan, nan, m, i), ("scan=4", 1.0, 500.0, "pos", nan, nan, m, i)]
    b = [("controllerType=0 controllerNumber=1 scan=5", 0.91, 300.0005, "pos", nan, nan, m, i),   # 1.01 corrected
         ("scan=6", 1.9, 400.0, "pos", nan, nan, m, i * 100)]                     # 1.9 native = 2.0 corrected
    ms2.save(ms2.from_collected(a, "pos"), out / "ms2" / "A.npz")
    ms2.save(ms2.from_collected(b, "pos"), out / "ms2" / "B.npz")
    (out / "rt_correction").mkdir()
    pd.DataFrame({"rt_native": [0.0, 3.0], "rt_corr": [0.1, 3.1]}).to_csv(out / "rt_correction" / "B.tsv", sep="\t",
                                                                          index=False)
    return out, m, i


def test_map_to_groups(tmp_path):
    out, m, i = _fake_output(tmp_path)
    t = ms2.map_to_groups(out)
    key = {(r.file, r.scan): r.group_id for r in t.itertuples()}
    assert key[("A", "scan=1")] == 12                     # both 300 groups fit: the one strongest in A wins
    assert key[("B", "controllerType=0 controllerNumber=1 scan=5")] == 11       # in B the 300.001 group is absent
    assert key[("A", "scan=2")] == key[("A", "scan=3")] == 13 and ("A", "scan=4") not in key
    assert key[("B", "scan=6")] == 13                     # found through B's RT correction
    rep = t[t["representative"] == 1].set_index("group_id")
    assert rep.loc[13, "scan"] == "scan=6" and rep.loc[13, "file"] == "B"     # highest TIC of the three
    assert list(pd.read_csv(out / "ms2_map.tsv", sep="\t").columns) == ms2.MAP_COLUMNS
    r = ms2.read_representatives(out)
    assert r["group_id"].tolist() == [11, 12, 13] and r["n_ms2"].tolist() == [1, 1, 3]
    assert r["file"] == ["B", "A", "B"] and r["scan"][0] == "controllerType=0 controllerNumber=1 scan=5"
    assert r["prec_mz"].tolist() == pytest.approx([300.0, 300.001, 400.0])
    assert r["rt_s"][2] == pytest.approx(114.0) and r["ion_mode"] == ["pos"] * 3
    assert np.allclose(r["mz"][r["off"][0]:r["off"][1]], m)
    lib = read_msp(out / "ms2_spectra.msp")
    assert lib.names == ["feature_11", "feature_12", "feature_13"]
    # the consensus copy: all three spectra of feature 13 (same peaks, different scales) merge, from both files
    c = ms2.read_representatives(out, "consensus")
    assert c["group_id"].tolist() == [11, 12, 13] and c["n_merged"].tolist() == [1, 1, 3]
    assert c["n_files"].tolist() == [1, 1, 2]
    assert np.allclose(c["mz"][c["off"][2]:c["off"][3]], m, atol=1e-6)
    assert t.loc[t["group_id"] == 13, "in_consensus"].tolist() == [1, 1, 1]
    assert (t["rt_aligned_s"] - t["rt_s"]).abs().max() == pytest.approx(6.0)        # B's correction: +0.1 min


def test_map_without_ms2_removes_stale_files(tmp_path):
    out, _, _ = _fake_output(tmp_path)
    ms2.map_to_groups(out)
    for f in (out / "ms2").iterdir():
        f.unlink()
    assert len(ms2.map_to_groups(out)) == 0
    assert not (out / "ms2_map.tsv").exists() and ms2.read_representatives(out) is None
    assert not (out / "ms2_consensus.msp").exists()


# ------------------------------------------------------------------ end to end ----
def test_process_maps_ms2_and_identifies(tmp_path):
    rng = np.random.default_rng(0)
    base = [(float(m), float(r), float(h)) for m, r, h in
            zip(rng.uniform(150, 900, 25), rng.uniform(0.6, 2.4, 25), 10 ** rng.uniform(5, 6.5, 25))]
    frags = [_frag(10 + k) for k in range(len(base))]
    d = tmp_path / "data"
    d.mkdir()
    for k, sh in enumerate((0.0, 0.03)):
        peaks = [Peak(m, r + sh, h, 0.05, iso=(0.15,)) for m, r, h in base]
        cloud, _ = make_cloud(peaks, n_scans=600, dt=DT, thresh=200.0, floor=100.0, n_noise=15000, seed=k, name=f"S{k}")
        spectra = [(r + sh, m + 2e-4, f[0], f[1] * (1 + 0.05 * k)) for (m, r, _), f in zip(base[:12], frags)]
        write_mzml(cloud, d / f"S{k}.mzML", ms2=spectra)
    lib = tmp_path / "lib.msp"
    with open(lib, "w") as fh:
        for j, ((m, _, _), f) in enumerate(zip(base, frags)):
            write_msp_record(fh, [("NAME", f"compound {j}"), ("PRECURSORMZ", m), ("IONMODE", "Positive")], *f)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "peak3d", "process", "--input", str(d), "--output", str(out),
                        "--no-report", "--library", str(lib), "--similarity", "entropy"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "[peak3d] ms2:" in r.stdout and "[peak3d] identify:" in r.stdout
    assert (out / "ms2" / "S0.npz").exists() and pd.read_csv(out / "features" / "manifest.tsv", sep="\t")["n_ms2"].tolist() == [12, 12]
    t = pd.read_csv(out / "aligned_feature_table.tsv", sep="\t")
    hits = pd.read_csv(out / "identifications.tsv", sep="\t")
    assert set(hits["spectrum"]) == {"consensus", "single"}
    top = hits[(hits["rank"] == 1) & (hits["spectrum"] == "consensus")].merge(t[["group_id", "mz"]], on="group_id")
    names = {j: f"compound {j}" for j in range(len(base))}
    right = sum(any(abs(row.mz - base[j][0]) < 0.005 and row.name == names[j] for j in range(12)) for row in top.itertuples())
    assert len(top) >= 10 and right == len(top)
    ann = pd.read_csv(out / "feature_annotations.tsv", sep="\t").set_index("group_id")
    assert (ann.loc[top["group_id"], "annotation"] == top["name"].to_numpy()).all()
    assert set(ann.loc[top["group_id"], "source"]) == {"library"}
    info = json.loads((out / "params.json").read_text())
    assert "ms2_s" in info["stage_times_s"] and info["options"]["similarity"] == "entropy"
    # re-search with another score through the identify command
    r = subprocess.run([sys.executable, "-m", "peak3d", "identify", str(out), "--library", str(lib), "--similarity", "dot"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    assert json.loads((out / "identifications.json").read_text())["method"] == "dot"
    r = subprocess.run([sys.executable, "-m", "peak3d", "identify", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "--library" in r.stderr
