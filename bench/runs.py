"""Benchmark runs: which mzML files each tool processes together, and the truth used.

Each run is processed as one batch by every tool (both tools align across the
files of a run). Credentialing runs are split into a 12C and a 13C run so the
two label states are never aligned with each other. PR002464 is the exception: its
design mixes 12C and 13C yeast in the same vials, so each study is one run.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.mz_registry import mz_test_files  # noqa: E402
MZML = ROOT / "data/mzml"
RAW = ROOT / "data/raw"
YEAST_LAB = RAW / "NETID_YEAST/ccms_peak/2-Yeast-labeling/2A-U13C glucose"
SZ22 = RAW / "ASARI_DATA/SZ22/SZ22_Dataset"
MCSI = RAW / "MASSCUBE_SI/extracted"  # Zenodo 14159704 (MassCube paper, Yu et al. 2025)
# the eight files with manually labeled EICs; every file is its own run (different samples,
# instruments and ion modes, so they are never aligned together)
MC_EXP = ["NIST_Plasma_Orbitrap_RP_positive", "NIST_Plasma_Orbitrap_RP_negative",
          "Mouse_Plasma_Orbitrap_HILIC_positive", "NIST_Feces_Orbitrap_HILIC_negative",
          "Human_Serum_QTOF_RP_positive", "Mouse_Feces_QTOF_HILIC_positive",
          "Fruit_Fly_QTOF_HILIC_positive", "Human_Urine_QTOF_HILIC_negative"]


def _glob(d, pat):
    return sorted(str(p) for p in Path(d).glob(pat))


def _bm21(analysis):
    hdr = (RAW / f"BM21/ST002454_{analysis}_Results.txt").open().readline()
    return [str(MZML / f"BM21/{s}.mzML") for s in hdr.rstrip("\n").split("\t")[1:]]


# PR002464 (Broeckling/Du/Patti) Orbitrap studies with truth (bench/spiketruth.py, credtruth.py, biotruth.py);
# ST004504 is not used: its file contents do not match the sample metadata
PR2464 = RAW / "PR002464"
PR2464_STUDIES = {"ST004500": "neg", "ST004501": "pos", "ST004502": "neg"}   # RP neg, HILIC pos, HILIC neg


def _pr2464(study, mz_test=False):
    """every MS1 injection of the study in metadata order, as bench/spiketruth.study_files lists them (224:
    samples, QCs, blanks; MeOH conditioning injections left out), less the injections whose m/z calibration is
    off (bench/mz_registry.py) unless mz_test. A file not on disk keeps a placeholder path, so 30_make_jobs.py
    reports it."""
    d = PR2464 / study
    meta = d / "2-sample_metadata/sample_metadata.csv"
    if not meta.exists():
        return []
    skip = set() if mz_test else mz_test_files(study)
    with meta.open() as fh:
        names = [Path(r["sample_name"]).stem for r in csv.DictReader(fh)
                 if r["ms_level"] == "1" and r["sample_type"] != "conditioning.meoh"
                 and Path(r["sample_name"]).stem not in skip]
    paths = {p.stem: str(p) for sub in d.glob("3-*centroid") for p in sub.rglob("*.mzML")}
    return [paths.get(n, str(d / f"{n}.mzML")) for n in names]


RUNS = {
    # 20,000 manually labeled TP/TN m/z-RT pairs (Agilent Q-TOF, pos)
    "IDSL003": dict(mode="pos", truth="idsl",
                    files=[str(RAW / "IDSL_IPA/003.mzML")]),
    # 401 manually certified features on these three samples (Orbitrap ID-X, HILIC pos)
    "HZV029_cert": dict(mode="pos", truth="hzv029",
                        files=[str(MZML / f"HZV029/batch4_MT_20210729_003{x}.mzML") for x in "CGK"]),
    # 12C / 13C E. coli credentialing (Orbitrap ID-X, RP pos)
    "SZ22_12C": dict(mode="pos", truth="cred", files=_glob(SZ22, "12C_*.mzML")),
    "SZ22_13C": dict(mode="pos", truth="cred", files=_glob(SZ22, "13C_*.mzML")),
    # 12C / U-13C glucose yeast credentialing (Q Exactive Plus, pos)
    "YEAST_12C": dict(mode="pos", truth="cred", files=_glob(YEAST_LAB, "posi-Yeast-12C14N-*.mzML")),
    "YEAST_13C": dict(mode="pos", truth="cred", files=_glob(YEAST_LAB, "posi-Yeast-13C14N-*.mzML")),
    # full studies: runtime / memory / feature counts at scale
    "HZV029_full": dict(mode="pos", truth=None, files=_glob(MZML / "HZV029", "*.mzML")),
    # plasma : vegetable-juice mixing series (13 ratios x 3) + pooled/QC, as listed by the
    # authors per analysis; HILIC and RP injections are interleaved and must not be mixed
    "BM21_HILIC": dict(mode="pos", truth="ratio", files=_bm21("AN004003")),
    "BM21_RP": dict(mode="pos", truth="ratio", files=_bm21("AN004004")),
    # ~1100 standards: 836 confirmed true features (760 equal in SA/SB, 76 at known
    # SB:SA ratios 1/16..16) in black-pepper matrix (Q Exactive HF, RP pos, 35 min)
    "LI2018": dict(mode="pos", truth="spike", files=_glob(MZML / "LI2018_QE", "S[AB][1-5].mzML")),
    # "Yeast2021" of the asari paper: NetID yeast MS1, negative mode (Q Exactive Plus).
    # Truth = 314 peaks with Confidence TRUE in NetID Supplementary Dataset 2 ("manual curation")
    "YEAST_NEG": dict(mode="neg", truth="netid314",
                      files=_glob(RAW / "NETID_YEAST/ccms_peak/1-Yeast-MS1", "neg-12C14N-[123]-0ev.mzML")),
    # MassCube benchmark: 13,500 single + 13,500 double Gaussian peaks inserted at m/z 1550-4250
    # into a urine QTOF neg file (Bruker impact II); truth = the inserted list. The published file's
    # byte index is stale; this copy has it recomputed (bench/masscube_truth.py reindex_mzml)
    "MC_SYNTH": dict(mode="neg", truth="mc_synth",
                     files=[str(MCSI.parent / "reindexed/simulated_file.mzML")]),
}
# MassCube benchmark: 722 EICs (676 single, 46 double peaks) labeled by the authors on these files
for _s in MC_EXP:
    RUNS[f"MC_{_s}"] = dict(mode="pos" if _s.endswith("positive") else "neg", truth="mc_eic",
                            files=[str(MCSI / f"8_experimental_data/raw_data/{_s}.mzML")])
# PR002464: one run per study, the 224 MS1 injections (Orbitrap ID-X) less those whose m/z calibration is off
# (bench/mz_registry.py: 1 in ST004500, 24 in ST004502); truth = spiked standards, credentialed yeast and plasma
# dilution/mixing peaks (data/truth/PR002464/<study>/, built without them too)
for _s, _m in PR2464_STUDIES.items():
    RUNS[f"PR2464_{_s}"] = dict(mode=_m, truth="pr2464", files=_pr2464(_s))
# the same studies with every injection, for testing m/z calibration only (never part of the standard benchmark,
# so kept out of RUNS)
MZTEST_RUNS = {f"PR2464_{_s}_mztest": dict(mode=_m, truth="pr2464", files=_pr2464(_s, mz_test=True))
               for _s, _m in PR2464_STUDIES.items() if mz_test_files(_s)}

# credentialing pairs: (12C run, 13C run)
CRED_PAIRS = {"SZ22": ("SZ22_12C", "SZ22_13C"), "YEAST": ("YEAST_12C", "YEAST_13C")}

TOOLS = ["asari", "masscube", "idslipa", "peak3d"]

# tools benchmarked locally but not published (bench/local_tools.py, git-ignored)
try:
    from bench import local_tools
    LOCAL_TOOLS, LOCAL_NAMES = list(local_tools.TOOLS), dict(local_tools.NAMES)
except ImportError:
    local_tools, LOCAL_TOOLS, LOCAL_NAMES = None, [], {}
TOOLS += LOCAL_TOOLS
