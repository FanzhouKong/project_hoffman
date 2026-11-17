"""--mode gc: a fragment peak on a continuous background of the same ion is kept (background_rule "net"); lc is the
default and rejects it; a background ion with nothing on it gives no feature in either mode."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from peak3d import __main__ as CLI
from peak3d.estimate import PROFILES, Params, estimate, mode_overrides, variant_label, VARIANT
from peak3d.features import build_features
from peak3d.gc.synth import Compound, make_run, random_spectrum
from peak3d.pick import pick_cloud
from peak3d.synth import Peak, write_mzml

ROOT = Path(__file__).resolve().parents[2]
DT, FWHM = 1.0 / 300.0, 0.05
SHARED, BLEED = 152.0621, 207.0324            # an ion of T that the background also holds; a pure bleed ion
RT_T = 2.52


def _run(seed=3):
    rng = np.random.default_rng(seed)
    comps = [Compound(f"X{k}", 0.4 + 0.25 * k, 10 ** rng.uniform(4.3, 5.3), *random_spectrum(rng, 15, 230, 500), FWHM)
             for k in range(16)]
    mz, rel = random_spectrum(rng, 12, 230, 500)
    comps.append(Compound("T", RT_T, 1e5, np.append(mz, SHARED), np.append(rel, 0.5), FWHM))      # 5e4 on a 5e4 floor
    cloud, _ = make_run(comps, n_scans=1500, dt=DT, iso=0.0, seed=seed,
                        extra_peaks=[Peak(SHARED, 2.5, 5e4, 60.0), Peak(BLEED, 2.5, 2e5, 60.0)])
    return cloud


def _features(mode, seed=3):
    cloud = _run(seed)
    b = pick_cloud(cloud, estimate(cloud, overrides=mode_overrides(mode)))
    feats, rej = build_features(cloud, b, keep_rejected=True)
    return feats, rej, b.params


def _at(df, mz, rt=None, tol_s=1.0):
    m = np.abs(df["mz"].values - mz) < 0.003
    if rt is not None:
        m &= np.abs(df["rt"].values - rt) * 60 < tol_s
    return df[m]


def test_profiles_and_overrides():
    assert CLI.MODE_CHOICES == list(PROFILES)
    assert mode_overrides(None) == {} and mode_overrides("lc") == {}
    assert Params(**mode_overrides("gc")).background_rule == "net" and Params().background_rule == "level"
    assert mode_overrides("gc", ["background_rule=level", "net_ripple_k=8"]) == {"background_rule": "level", "net_ripple_k": 8.0}
    assert variant_label(mode_overrides("lc")) == VARIANT
    assert variant_label(mode_overrides("gc")) == VARIANT + " + background_rule=net"
    with pytest.raises(ValueError):
        mode_overrides("ce")
    with pytest.raises(ValueError):
        mode_overrides("gc", ["background_rule=sideways"])


def test_fragment_on_a_background_of_its_ion_is_kept_in_gc_mode_only():
    lc, lc_rej, _ = _features("lc")
    gc, gc_rej, P = _features("gc")
    assert P.background_rule == "net"
    assert len(_at(lc, SHARED, RT_T)) == 0
    assert len(_at(lc_rej[lc_rej["reason"] == "gate"], SHARED, RT_T, 3.0)) >= 1      # the level gates saw it
    f = _at(gc, SHARED, RT_T)
    assert len(f) == 1
    f = f.iloc[0]
    assert "on_background" in f["flags"]
    assert abs(f["rt"] - RT_T) * 60 < 0.3
    assert 0.8e5 < f["height"] < 1.25e5                                              # compound + floor
    # nothing else of the two background ions becomes a feature, in either mode
    assert len(_at(gc, SHARED)) == 1 and len(_at(lc, SHARED)) == 0
    assert len(_at(gc, BLEED)) == 0 and len(_at(lc, BLEED)) == 0
    # the peaks that stand on nothing are the same features in both modes
    plain = gc[~gc["flags"].astype(str).str.contains("on_background")]
    assert len(plain) == len(lc)
    assert np.allclose(np.sort(plain["mz"].values), np.sort(lc["mz"].values))
    assert "on_background" not in " ".join(lc["flags"].astype(str))
    assert "net_fwhm" in gc_rej and "net_fwhm" not in lc_rej                         # lc tables keep their columns


def test_net_width_is_the_width_above_the_floor():
    gc, gc_rej, P = _features("gc")
    from peak3d.features import net_width
    cloud = _run()
    row = _at(gc, SHARED, RT_T).assign(floor_rel=0.5, sig_apex=0.5)
    w = net_width(row, cloud, P)[0]
    assert abs(w - FWHM) / FWHM < 0.2                                                # the compound's 3 s, not the hump's
    assert np.isnan(net_width(row.assign(floor_rel=-0.9), cloud, P)[0])              # a level the trace never falls to


def test_cli_records_the_mode(tmp_path):
    data = tmp_path / "mzml"
    data.mkdir()
    write_mzml(_run(), data / "run1.mzML")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "peak3d", "process", "--input", str(data), "--output", str(out), "--no-report",
                        "--mode", "gc"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    p = json.loads((out / "params.json").read_text())
    assert p["mode"] == "gc" and p["variant"].endswith("background_rule=net")
    fp = json.loads((out / "features" / "run1.params.json").read_text())
    assert fp["background_rule"] == "net"
