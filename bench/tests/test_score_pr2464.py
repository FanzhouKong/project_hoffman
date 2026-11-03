"""PR002464 scorers in bench/score.py on a hand-built study: a few truth peaks per truth set, one tool output whose
intensities are chosen so every recall, ratio and slope score has a known value.

Spike: P1 (tier A, A/B 4:1) found with the designed ratio and dilution slope 1; P2 (tier A, A/B 1:2) found with
A/B 1:8 (log2 error 2) and a flat dilution series (slope 0); P3 (tier B) missing. A feature 15 ppm above P1 at
nearly the same RT must not be taken for P1 when P1 itself is there, and a feature at P1's m/z 0.3 min away never
matches. Yeast: Y1 with its 13C partner and dilution slope 1; Y2 12C only. Plasma: L1 proportional to plasma with
SRM 1950 / 8231 = 2, as in the raw data, and a feature 0.05 Da above it that one of its decoys meets.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import score  # noqa: E402

RUN, STUDY = "PR2464_STX", "STX"
# vial -> (A, B, SRM 1950, SRM 8231, Yeast 12C, Yeast 13C)
VIALS = {18: (1000, 0, 0, 0, 0, 0), 19: (0, 1000, 0, 0, 0, 0), 9: (500, 500, 0, 0, 0, 0), 10: (250, 250, 0, 0, 0, 0),
         11: (100, 100, 0, 0, 0, 0), 12: (50, 50, 0, 0, 0, 0), 5: (0, 0, 0, 0, 500, 500), 6: (0, 0, 0, 0, 250, 250),
         7: (0, 0, 0, 0, 100, 100), 8: (0, 0, 0, 0, 50, 50), 1: (0, 0, 500, 500, 0, 0), 2: (0, 0, 250, 250, 0, 0),
         3: (0, 0, 100, 100, 0, 0), 4: (0, 0, 50, 50, 0, 0), 13: (0, 0, 1000, 0, 0, 0), 14: (0, 0, 0, 1000, 0, 0)}
FILES = [(f"f_{v}_{b}", v) for v in VIALS for b in (1, 2)]


def table(rows):
    """feature table: mz, rt and one column per file; rows = (mz, rt, {vial: intensity})"""
    out = []
    for mz, rt, by_vial in rows:
        out.append({"mz": mz, "rt": rt} | {f: by_vial.get(v, 0.0) for f, v in FILES})
    return pd.DataFrame(out)


@pytest.fixture()
def study(tmp_path, monkeypatch):
    truth = tmp_path / "truth" / STUDY
    truth.mkdir(parents=True)
    pd.DataFrame([dict(file=f, vial=v, A=VIALS[v][0], B=VIALS[v][1], **{"SRM 1950": VIALS[v][2]},
                       **{"SRM 8231": VIALS[v][3], "Yeast 12C": VIALS[v][4], "Yeast 13C": VIALS[v][5]})
                  for f, v in FILES]).to_csv(truth / "files.tsv", sep="\t", index=False)
    pd.DataFrame([
        dict(truth_id="P1", tier="A", mz=200.1, mz_obs=200.1, rt_apex=2.0, flags="", annotation_class="formula_ion",
             log2FC=2.0, weak_at_noise=False, c_dilution_slope=1.0),
        dict(truth_id="P2", tier="A", mz=300.2, mz_obs=300.2, rt_apex=3.0, flags="ft_artifact?",
             annotation_class="isotope", log2FC=-1.0, weak_at_noise=False, c_dilution_slope=1.0),
        dict(truth_id="P3", tier="B", mz=400.3, mz_obs=None, rt_apex=4.0, flags="", annotation_class="unannotated",
             log2FC=1.0, weak_at_noise=False, c_dilution_slope=0.9),
    ]).to_csv(truth / "spiked_truth.tsv", sep="\t", index=False)
    pd.DataFrame([
        dict(truth_id="Y1", mz=150.05, mz_13C=150.05 + 5 * 1.0033548, rt=1.5, n=5, tier="A", feature_class="primary",
             confirmed=True, dil_slope_12C=1.0),
        dict(truth_id="Y2", mz=250.1, mz_13C=250.1 + 10 * 1.0033548, rt=2.5, n=10, tier="B",
             feature_class="isotope_M+1", confirmed=False, dil_slope_12C=np.nan),
    ]).to_csv(truth / "yeast_truth.tsv", sep="\t", index=False)
    pd.DataFrame([dict(truth_id="L1", tier="A", mz_obs=120.06, rt_apex=1.2, flags="", dil_slope=1.0,
                       log2_1950_8231=1.0)]).to_csv(truth / "plasma_truth.tsv", sep="\t", index=False)

    spike_dil = {9: 1000.0, 10: 500.0, 11: 200.0, 12: 100.0}
    feats = table([
        (200.1, 2.0, {18: 4000.0, 19: 1000.0} | spike_dil),                         # P1: exact
        (200.1 * (1 + 15e-6), 2.05, {18: 1.0, 19: 1000.0}),                            # 15 ppm neighbour of P1
        (200.1, 2.3, {18: 1.0, 19: 1.0}),                                              # P1's m/z, 0.3 min away
        (300.2, 3.0, {18: 1000.0, 19: 8000.0, 9: 50.0, 10: 50.0, 11: 50.0, 12: 50.0}),  # P2: 1:8, flat
        (150.05, 1.5, {5: 500.0, 6: 250.0, 7: 100.0, 8: 50.0}),                       # Y1 12C
        (150.05 + 5 * 1.0033548, 1.5, {5: 500.0}),                                     # Y1 13C partner
        (250.1, 2.5, {5: 10.0}),                                                       # Y2 12C
        (120.06, 1.2, {1: 10000.0, 2: 5000.0, 3: 2000.0, 4: 1000.0, 13: 2000.0, 14: 1000.0}),  # L1
        (120.11, 1.2, {13: 10.0}),                                                     # 0.05 Da above L1
    ])
    out = tmp_path / "results/peak3d_GLR" / RUN / "peak3d_out"
    out.mkdir(parents=True)
    feats.to_csv(out / "aligned_feature_table.tsv", sep="\t", index=False)
    (out.parent / "DONE").write_text("x\n")

    monkeypatch.setattr(score, "RES", tmp_path / "results")
    monkeypatch.setattr(score, "PR2464_TRUTH", tmp_path / "truth")
    monkeypatch.setattr(score, "PR2464_RUNS", {RUN: STUDY})
    monkeypatch.setattr(score, "TOOLS", [])
    monkeypatch.setitem(score.RUNS, RUN, dict(mode="pos", truth="pr2464", files=[f"/x/{f}.mzML" for f, _ in FILES]))


def at(df, ppm=10, rt=0.1):
    r = df[(df["ppm"] == ppm) & (df["rt_tol_min"] == rt)]
    assert len(r) == 1 and r["tool"].iloc[0] == "peak3d_GLR"
    return r.iloc[0]


def test_spike(study):
    df = score.score_pr2464_spike()
    assert len(df) == len(score.GRID_PPM) * len(score.GRID_RT)
    r = at(df)
    assert r["recall"] == pytest.approx(2 / 3) and r["recall_A"] == 1.0 and r["recall_B"] == 0.0
    assert r["decoy_match"] == 0.0
    assert r["recall_A_unflagged"] == 1.0 and r["recall_formula_ion"] == 1.0
    assert r["ratio_eligible"] == 2 and r["ratio_quantified"] == 2
    assert r["ratio_median_abs_log2_err"] == pytest.approx(1.0)        # errors 0 and 2
    assert r["ratio_within_2fold"] == 0.5 and r["ratio_right_direction"] == 1.0
    assert r["dil_quantified"] == 2 and r["dil_slope_median"] == pytest.approx(0.5)
    assert r["dil_median_abs_err_vs_raw"] == pytest.approx(0.5) and r["dil_within_025_of_raw"] == 0.5
    # at 20 ppm the 15 ppm neighbour is in the window, but P1 itself is closer in RT and keeps the exact ratio
    assert at(df, 20)["ratio_median_abs_log2_err"] == pytest.approx(1.0)


def test_yeast(study):
    r = at(score.score_pr2464_yeast())
    assert r["recall"] == 1.0 and r["recall_A"] == 1.0 and r["recall_B"] == 1.0 and r["recall_confirmed"] == 1.0
    assert r["recall_primary"] == 1.0
    assert r["recall_13C_partner"] == 0.5 and r["recall_both"] == 0.5
    # Y2 has no raw slope (and only one level): only Y1 is compared
    assert r["dil_quantified"] == 1 and r["dil_slope_median"] == pytest.approx(1.0)
    assert r["dil_median_abs_err_vs_raw"] == pytest.approx(0.0, abs=1e-9)


def test_plasma(study):
    r = at(score.score_pr2464_plasma())
    assert r["recall"] == 1.0 and r["recall_A"] == 1.0 and np.isnan(r["recall_B"])
    assert r["decoy_match"] == pytest.approx(0.5)      # the feature 0.05 Da above L1 meets one of its two decoys
    assert r["dil_quantified"] == 1 and r["dil_slope_median"] == pytest.approx(1.0)
    assert r["mix_quantified"] == 1 and r["mix_median_abs_log2_err_vs_raw"] == pytest.approx(0.0, abs=1e-9)


def test_runs_listed():
    """the three Orbitrap studies are runs; ST004504 (contents do not match its metadata) is not"""
    assert {"PR2464_ST004500", "PR2464_ST004501", "PR2464_ST004502"} <= set(score.RUNS)
    assert "PR2464_ST004504" not in score.RUNS
    files = score.RUNS["PR2464_ST004501"]["files"]
    assert len(files) == 224 and all(Path(f).exists() for f in files)
