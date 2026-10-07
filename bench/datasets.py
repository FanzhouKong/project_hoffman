"""Registry of public datasets used to build the peak-picking benchmark.

Each entry says where the data lives, how to list its files, and what kind of
ground truth it provides. `priority` orders downloads (1 first); MassCube's own
benchmark data is kept but deliberately ranked last.

truth types
  labeled_tp_tn   manually labeled true and false signals (FDR anchor)
  manual_features manually verified true features (coverage anchor)
  credentialing   12C/13C labeled pairs: real biology vs artifacts
  mixing_ratio    graded mixtures: real features follow the ratio
  spike_ratio     known standards at defined concentration ratios
  replicates      many repeat injections (noise model, reproducibility)
  blanks          solvent/process blanks (background-only signal)
  injected        synthetic peaks inserted into real background
"""

DATASETS = [
    # --- asari series (Li lab, JAX) ---------------------------------------
    dict(
        id="BM21", priority=1, source="mwb", accession="ST002454",
        instrument="Orbitrap (JAX), HILIC+ / RP",
        truth=["mixing_ratio"],
        note="Human plasma x vegetable juice serial mixtures (asari paper).",
    ),
    dict(
        id="HZV029", priority=1, source="mwb", accession="ST002233",
        instrument="Orbitrap ID-X, HILIC+",
        truth=["replicates", "manual_features"],
        note="Repeat injections of pooled plasma; 402 manually certified "
             "features in shuzhao-li-lab/data (hzv029_manual_certified.txt).",
    ),
    dict(
        id="ASARI_DATA", priority=1, source="github", accession="shuzhao-li-lab/data",
        instrument="various (MT02, SZ22, NIST SRM 1950)",
        truth=["manual_features"],
        note="MT02/SZ22 datasets, NIST SRM1950 verified features, HZV029 "
             "certified list, and the asari comparison notebooks.",
    ),
    dict(
        id="NETID_YEAST", priority=1, source="gnps", accession="MSV000087434",
        instrument="Q Exactive Plus",
        truth=["credentialing", "blanks", "manual_features"],
        note="Yeast2021 in the asari paper (NetID raw data, Rabinowitz lab). "
             "12C/13C/15N yeast, solvent blanks, liver, mouse labeling. "
             "Only ccms_peak/ is taken; raw/ holds mzXML duplicates.",
        include_prefix=["ccms_peak/", "ccms_parameters/"],
    ),
    dict(
        id="SLAW_ECOLI", priority=2, source="gnps", accession="MSV000086486",
        instrument="Q Exactive HF",
        truth=["replicates"],
        note="~2000-sample E. coli KO dataset (SLAW); scale test, asari paper.",
    ),
    # --- labeled / spiked benchmarks ---------------------------------------
    dict(
        id="IDSL_IPA", priority=1, source="zenodo", accession="6302236",
        instrument="LC/HRMS (MTBLS1684 file 003)",
        truth=["labeled_tp_tn"],
        note="20,000 manually labeled TP/TN m/z-RT pairs on one mzML.",
    ),
    dict(
        id="MTBLS1684", priority=2, source="metabolights", accession="MTBLS1684",
        instrument="Agilent Q-TOF (.d.zip)",
        truth=[],
        note="Full study behind the IDSL.IPA labels; needs vendor conversion.",
    ),
    dict(
        id="LI2018_QE", priority=1, source="metabolights", accession="MTBLS733",
        instrument="Thermo Q Exactive HF",
        truth=["spike_ratio"],
        note="~1100 standards at defined ratios (Li et al. 2018).",
    ),
    dict(
        id="LI2018_TTOF", priority=1, source="metabolights", accession="MTBLS736",
        instrument="SCIEX TripleTOF 6600",
        truth=["spike_ratio"],
        note="Same design as MTBLS733. Listing shows only small .wiff files; "
             "check whether .wiff.scan files are present before relying on it.",
    ),
    # --- lower priority -----------------------------------------------------
    dict(
        id="MASSCUBE_SI", priority=3, source="zenodo", accession="14159704",
        instrument="Orbitrap + Bruker QTOF",
        truth=["injected", "manual_features"],
        note="MassCube supplementary data; kept as a secondary check only. "
             "The 3.2 GB mouse-brain output archive is skipped.",
        exclude_names=["Mouse_brain_MassCube_output.zip"],
    ),
]
