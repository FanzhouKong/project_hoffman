"""bench/gc/mrh_score.py on a synthetic stand-in for one MRH injection: four compounds, a truth file in the MRH
format with good rows and the three kinds of bad rows the real files hold."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.gc import mrh_score as M
from peak3d.gc.synth import Compound, make_run, random_spectrum
from peak3d.synth import write_mzml

ROOT = Path(__file__).resolve().parents[2]


def test_consensus_rt_is_the_densest_group():
    assert abs(M.consensus_rt([5.00, 5.001, 5.002, 5.2, 4.7], 1.5 / 60, 3) - 5.001) < 1e-9
    assert np.isnan(M.consensus_rt([5.0, 5.3, 5.6], 1.5 / 60, 3))          # no three ions together
    assert np.isnan(M.consensus_rt([5.0, 5.0], 1.5 / 60, 3))


def test_match_features_found_inside_and_absent():
    f_mz = np.array([100.0, 100.0, 200.0])
    f_rt = np.array([1.00, 2.00, 1.00])
    idx, d, found, inside = M.match_features(np.array([100.0003, 100.0003, 200.0, 300.0]), np.array([1.01, 1.50, 3.0, 1.0]),
                                             f_mz, f_rt, f_rt - 0.6, f_rt + 0.6, ppm=10.0, rt_tol_s=2.0)
    assert found.tolist() == [True, False, False, False]
    assert inside.tolist() == [False, True, False, False]                  # 1.50 min lies in both peaks' bounds
    assert idx[0] == 0 and abs(d[0] - 0.6) < 1e-6 and idx[3] == -1


def test_dose_response_slope_is_the_compound_share():
    rows = []
    for lv in M.LEVELS:
        c = M.level_value(lv)
        rows.append(dict(level=lv, entry="A", mz_expected=100.0, intensity=300.0 * c))            # the compound alone
        rows.append(dict(level=lv, entry="A", mz_expected=207.0, intensity=2e6))                  # column bleed
        rows.append(dict(level=lv, entry="A", mz_expected=117.0, intensity=1e4 + 100.0 * c))      # on a background
    rows += [dict(level=lv, entry="B", mz_expected=150.0, intensity=5.0 * M.level_value(lv)) for lv in ("500", "1000")]
    t = pd.DataFrame(rows).assign(matched=True, mass_ok=True, rt_ok=True)
    t["slope"] = M.dose_response(t)
    assert np.allclose(t.loc[t["mz_expected"] == 100.0, "slope"], 1.0)
    assert np.allclose(t.loc[t["mz_expected"] == 207.0, "slope"], 0.0)
    bg = t[t["mz_expected"] == 117.0].set_index("level")["slope"]
    assert bg["0_1"] < 0.05 and bg["1000"] > 0.8 and (np.diff(bg[M.LEVELS].values) > 0).all()
    assert bg["100"] == pytest.approx(0.5, abs=0.1)              # background = compound at 100 ng/mL
    assert t.loc[t["entry"] == "B", "slope"].isna().all()        # two levels: not estimable


def test_raw_apex_rejects_a_point_on_the_flank():
    c = Compound("X", 1.0, 2e5, np.array([150.0, 250.0]), np.array([1.0, 0.5]))
    cloud, _ = make_run([c], n_scans=600, dt=1 / 300, seed=1)
    t = pd.DataFrame(dict(matched=[True, True, True, False], mz=[150.0, 150.0, 250.0, 150.0],
                          rt=[1.0, 1.0 - 0.04, 1.0 + 0.004, 1.0]))            # apex, 2.4 s before it, one scan off
    peak, apex = M.raw_peak(t, cloud, 10.0)
    assert peak.tolist() == [True, True, True, False]
    assert apex.tolist() == [True, False, True, False]


def test_end_to_end_on_a_synthetic_injection(tmp_path):
    rng = np.random.default_rng(4)
    comps = [Compound(f"std{k}", 1.0 + 0.6 * k, 2e5, *random_spectrum(rng, 12, 100, 600)) for k in range(4)]
    comps += [Compound(f"bg{k}", 0.5 + 0.2 * k, 5e4, *random_spectrum(rng, 12, 100, 600)) for k in range(14)]
    cloud, _ = make_run(comps, n_scans=1500, dt=1 / 300, seed=6, name="NEWSTD_100")
    data = tmp_path / "mzml"
    data.mkdir()
    write_mzml(cloud, data / "NEWSTD_100.mzML")
    rows = []
    for c in comps[:4]:
        for m, r in zip(c.mz[:8], c.rel[:8]):
            rows.append([c.name, "Sample", round(c.rt + 0.08, 2), round(m, 4), True, m * (1 + 2e-6), c.rt + 0.0005, r * c.height])
    c = comps[0]
    rows.append([c.name, "Sample", round(c.rt + 0.08, 2), 300.1, True, 300.1 * (1 + 150e-6), c.rt, 900.0])       # wrong ion
    rows.append([c.name, "Sample", round(c.rt + 0.08, 2), round(c.mz[8], 4), True, c.mz[8], c.rt + 10 / 60, 700.0])   # 10 s off
    rows.append([c.name, "Sample", round(c.rt + 0.08, 2), 555.5, False, "", "", ""])                              # no data
    truth = tmp_path / "truth"
    truth.mkdir()
    pd.DataFrame(rows, columns=["CompoundName", "Category", "Retention Time", "m/z", "matched", "mz_matched", "rt_matched",
                                "intensity_matched"]).to_csv(truth / "groundtruth_corrected_100.csv", index=False)
    out = tmp_path / "out"
    for argv in (["process", "--input", str(data), "--output", str(out), "--no-report", "--keep-rejected"],
                 ["gc", str(out)]):
        r = subprocess.run([sys.executable, "-m", "peak3d", *argv], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    M.main(["--output", str(out), "--truth", str(truth), "--out", str(tmp_path / "score")])
    t = pd.read_csv(tmp_path / "score" / "truth_rows.tsv", sep="\t")
    assert len(t) == 35 and t["matched"].sum() == 34
    good = t.iloc[:32]
    assert good["clean"].all() and good["mass_ok"].all() and good["rt_ok"].all() and good["raw_peak"].all()
    assert good["raw_apex"].all() and good["responds"].all() and t["response_slope"].isna().all()      # one level
    assert good["found"].mean() > 0.9
    bad = t.iloc[32:]
    assert bad["clean"].tolist() == [False, False, False]
    assert not bad["mass_ok"].iloc[0] and bad["mass_ok"].iloc[1] and not bad["rt_ok"].iloc[1]
    assert bad["outcome"].iloc[2] == "no data in truth"
    lv = pd.read_csv(tmp_path / "score" / "by_level.tsv", sep="\t").iloc[0]
    assert lv["rows"] == 35 and lv["recorded"] == 34 and lv["clean"] == 32 and lv["recall_clean"] > 0.9
    comp = pd.read_csv(tmp_path / "score" / "compounds.tsv", sep="\t")
    assert len(comp) == 4 and (comp["best"] >= 0.85).all() and (comp["d_rt_s"].abs() < 0.3).all()
    assert (comp["n_components"] == 1).all()
