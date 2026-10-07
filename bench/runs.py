"""Benchmark runs: which mzML files each tool processes together, and the truth used.

Each run is processed as one batch by every tool (both tools align across the
files of a run). Credentialing runs are split into a 12C and a 13C run so the
two label states are never aligned with each other.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MZML = ROOT / "data/mzml"
RAW = ROOT / "data/raw"
YEAST_LAB = RAW / "NETID_YEAST/ccms_peak/2-Yeast-labeling/2A-U13C glucose"
SZ22 = RAW / "ASARI_DATA/SZ22/SZ22_Dataset"


def _glob(d, pat):
    return sorted(str(p) for p in Path(d).glob(pat))


def _bm21(analysis):
    hdr = (RAW / f"BM21/ST002454_{analysis}_Results.txt").open().readline()
    return [str(MZML / f"BM21/{s}.mzML") for s in hdr.rstrip("\n").split("\t")[1:]]


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
}

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
