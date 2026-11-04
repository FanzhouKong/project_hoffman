import pytest

from specdenoise import formula as fm
from specdenoise.search import ion_formula
from specdenoise.smiles import smiles_formula

KNOWN = [   # SMILES, formula (Hill order as formula.format writes it), net charge
    ("c1ccccc1", "C6H6", 0), ("C1=CC=CC=C1", "C6H6", 0), ("n1ccccc1", "C5H5N", 0), ("c1cc[nH]c1", "C4H5N", 0),
    ("c1ccsc1", "C4H4S", 0), ("S1C=CC=C1", "C4H4S", 0), ("c1ccoc1", "C4H4O", 0), ("c1nn[nH]n1", "CH2N4", 0),
    ("c1cscn1", "C3H3NS", 0), ("c1ccc2ccccc2c1", "C10H8", 0), ("c1ccc2[nH]ccc2c1", "C8H7N", 0),
    ("[O-][n+]1ccccc1", "C5H5NO", 0), ("O=c1cc[nH]cc1", "C5H5NO", 0), ("O=C1NC(=O)c2ccccc12", "C8H5NO2", 0),
    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "C8H10N4O2", 0),                                     # caffeine
    ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "C6H12O6", 0),                         # glucose
    ("CC(C)C[C@H](N)C(O)=O", "C6H13NO2", 0), ("N[C@@H](Cc1c[nH]c2ccccc12)C(O)=O", "C11H12N2O2", 0),
    ("OC(=O)CC(O)(CC(O)=O)C(O)=O", "C6H8O7", 0),
    ("Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(O)(=O)OP(O)(=O)OP(O)(O)=O)[C@@H](O)[C@H]1O", "C10H16N5O13P3", 0),  # ATP
    ("C[N+](C)(C)CCO", "C5H14NO", 1), ("[NH3+]CC([O-])=O", "C2H5NO2", 0), ("[NH4+]", "H4N", 1),
    ("F[P-](F)(F)(F)(F)F", "F6P", -1), ("CS(C)=O", "C2H6OS", 0), ("CS(=O)(=O)O", "CH4O3S", 0),
    ("OS(O)(=O)=O", "H2O4S", 0), ("OP(=O)(O)O", "H3O4P", 0), ("CP(C)C", "C3H9P", 0), ("CP(=O)(C)C", "C3H9OP", 0),
    ("C[N+](=O)[O-]", "CH3NO2", 0), ("CN(=O)=O", "CH3NO2", 0), ("C#N", "CHN", 0), ("N#N", "N2", 0),
    ("C1=CC1", "C3H4", 0), ("C=1CC1", "C3H4", 0), ("C%10CC%10", "C3H6", 0), ("C/C=C/C", "C4H8", 0),
    ("C\\C=C/C", "C4H8", 0), ("[H]OC", "CH4O", 0), ("[CH3:1]C", "C2H6", 0), ("F[C@TH1](Cl)(Br)I", "CBrClFI", 0),
    ("Clc1ccc(Br)cc1I", "C6H3BrClI", 0), ("CC(=O)[O-].[Na+]", "C2H3NaO2", 0), ("[K+].[Cl-]", "ClK", 0),
    ("C[Si](C)(C)OC", "C4H12OSi", 0), ("C(C)1CC1", "C4H8", 0),
]


@pytest.mark.parametrize("smi,formula,charge", KNOWN)
def test_known_formulas(smi, formula, charge):
    f, z = smiles_formula(smi)
    assert fm.format(f) == formula and z == charge, (smi, fm.format(f), z)


@pytest.mark.parametrize("smi", ["", "C*", "[13CH4]", "[2H]OC", "C(C", "C1CC", "CC)", "[Se]", "B(O)(O)c1ccccc1",
                                 "[Fe+2]", "C!C", "C C", "C=1CC-1", "=CC", "C==C", "C1C1", "InChI=1S/CH4/h1H4", "N/A",
                                 "C(=)C", "[C@XY]"])
def test_refuses_what_it_cannot_read(smi):
    assert smiles_formula(smi) is None


def test_largest_component():
    f, z = smiles_formula("CC(=O)[O-].[Na+]", "largest")
    assert fm.format(f) == "C2H3O2" and z == -1
    assert smiles_formula("[Na+].[Cl-]", "largest") is None                 # a tie: no single organic ion
    f, z = smiles_formula("CCO", "largest")
    assert fm.format(f) == "C2H6O"


def test_ion_formula_from_smiles():
    leu = "CC(C)C[C@H](N)C(O)=O"
    p, z = ion_formula("", "[M+H]+", 1, 132.1019, smiles=leu)
    assert fm.format(p) == "C6H14NO2" and z == 1
    p, z = ion_formula(None, "", 1, 154.0838, smiles=leu)                    # adduct inferred
    assert fm.format(p) == "C6H13NNaO2"
    assert ion_formula("", "[M+H]+", 1, float("nan"), smiles=leu) is None     # a structure needs the m/z to confirm
    assert ion_formula("", "[M+H]+", 1, 133.1019, smiles=leu) is None         # 1 Da off: nothing fits
    p, z = ion_formula("", "[M-H]-", 2, 59.0139, smiles="CC(=O)[O-].[Na+]")   # the salt's acid, deprotonated
    assert fm.format(p) == "C2H3O2" and z == -1
    p, z = ion_formula("", "[M]+", 1, 104.1070, smiles="C[N+](C)(C)CCO")      # a permanent cation
    assert fm.format(p) == "C5H14NO" and z == 1
    p, z = ion_formula("C6H13NO2", "[M+H]+", 1, 132.1019, smiles="c1ccccc1")  # the formula wins over the SMILES
    assert fm.format(p) == "C6H14NO2"
    p, z = ion_formula("C11H13N3O5�", "[M+H]+", 1, 132.1019, smiles=leu)  # a damaged formula: the SMILES
    assert fm.format(p) == "C6H14NO2"


def test_cxsmiles_extension_and_tied_components():
    assert fm.format(smiles_formula("CC(=O)O[C@@H](CO)CO |&1:4,r|")[0]) == "C5H10O4"
    assert smiles_formula("CC(=O)O some name") is None                     # only a |...| extension is dropped
    f, z = smiles_formula("OC1=NC(Cl)=C(Cl)C=C1Cl.O=C1NC(Cl)=C(Cl)C=C1Cl", "largest")   # two tautomers
    assert fm.format(f) == "C5H2Cl3NO" and z == 0


def test_charged_structures_and_radical_guess():
    p, z = ion_formula("", "[2M+H]+", 1, 235.1652, smiles="C[N+](C)(C)CC(=O)O")      # betaine written protonated
    assert fm.format(p) == "C10H23N2O4" and z == 1
    p, z = ion_formula("", "[M-H]-", 2, 263.0972, smiles="Cc1ncc(C[n+]2csc(CCO)c2C)c(=N)[nH]1")   # thiamine
    assert fm.format(p) == "C12H15N4OS" and z == -1
    # a neutral structure is never guessed as a radical [M]-: this wrong SMILES (C#C for entacapone's C#N) fit
    # 304.0939 as [M]- within 0.02 Da
    assert ion_formula("", "[M-H]-", 2, 304.093894,
                       smiles="CCN(CC)C(=O)/C(=C/C1=CC(O)=C(O)C(=C1)[N+]([O-])=O)/C#C") is None
    assert ion_formula("", "", 1, 104.1070, smiles="C[N+](C)(C)CCO")[0].tolist() == \
        fm.parse("C5H14NO").tolist()                                       # a cation may be [M]+


def test_formula_that_fits_nothing_gives_way_to_smiles():
    smi = "CC(C(=O)O)C(O)(CC(=O)O)C(=O)O"                                     # MoNA: formula typed C7H710O7
    p, z = ion_formula("C7H710O7", "[M-H]-", 2, 205.03535, smiles=smi)
    assert fm.format(p) == "C7H9O7" and z == -1
    assert ion_formula("C7H710O7", "[M-H]-", 2, 205.03535) is None             # no SMILES: nothing
    assert ion_formula("C7H710O7", "[M-H]-", 2, float("nan"), smiles=smi)[0][fm.H] == 709   # no m/z: as written
