# specdenoise: neutral-loss plausibility denoising of MS/MS spectra — plan (2026-10-29)

## 1. Idea

A collision-induced fragment ion `f` of a precursor ion `p` (charge 1) is `p` minus a neutral piece.
Its position on the m/z axis is therefore not free: the neutral loss

    Δ = m/z(p) − m/z(f)

must equal the mass of a chemically possible neutral (or radical) formula `L` that can be carved out of
the precursor formula `P`, leaving a chemically possible ion `P − L`. Everything else on the axis is a
*forbidden region*: an ion found there cannot be a fragment of `p` and is removed.

The method evaluates every ion by its neutral loss, in three tiers depending on what is known about the
precursor:

| tier | known about the precursor | permitted losses |
|---|---|---|
| T0 | only m/z (unknown compound, raw DDA spectrum) | `L` plausible by the Golden Rules, with at least one plausible precursor formula `P` decomposed from m/z(p) such that `L ⊆ P` and `P − L` is plausible |
| T1 | elemental formula + adduct (library hit, annotation, CSI:FingerID) | `L ⊆ P` (sub-formula), `L` and `P − L` plausible |
| T2 | structure (SMILES) | T1; bond-cut plausibility is noted as a later extension, not part of v1 |

T1 is the formula denoising of Kong et al. (Nat Methods 22:1008, 2025) put on a precise footing (explicit rules
for `L` and `P − L`, a tolerance model, isotopologue handling). T0 is the new part: the precursor
*mass* alone already forbids large parts of the loss axis, and the joint constraint "`L` is a sub-formula of a
precursor formula that fits m/z(p)" forbids far more than "`L` is some plausible formula" (which is the
constraint chemical-noise ions are generated to satisfy). How much T0 forbids as a function of loss mass and
tolerance is the first analysis to run (§3.0).

## 2. Rules (from Kind & Fiehn, BMC Bioinformatics 2007, 8:105, "Seven Golden Rules")

Applied to the loss `L` and to the remaining ion `P − L`. The rules were derived for intact molecules; the
deviations for losses are stated explicitly.

1. **Element count limits (rule 1)** per mass window, 99.7 % coverage of known compounds:
   < 500 Da: C ≤ 39, H ≤ 72, N ≤ 20, O ≤ 20, P ≤ 9, S ≤ 10, F ≤ 16, Cl ≤ 10, Br ≤ 4, Si ≤ 8;
   < 1000 Da: C ≤ 78, H ≤ 126, N ≤ 20, O ≤ 27, P ≤ 9, S ≤ 14, F ≤ 34, Cl ≤ 12, Br ≤ 8, Si ≤ 14;
   < 2000 Da: C ≤ 156, H ≤ 236, N ≤ 20, O ≤ 63, P ≤ 9, S ≤ 14, F ≤ 48, Cl ≤ 12, Br ≤ 10, Si ≤ 15.
   Used as the enumeration bounds for T0 precursor candidates and for every `L`.
2. **LEWIS / SENIOR (rule 2)**: *not* applied to `L` (radical losses such as CH3•, OH•, Cl• are real) and *not* to
   `P − L` (odd-electron fragment ions are real). Applied only to the neutral `M` of T0 precursor candidates
   (even-electron `[M+H]+`): the valence sum of `M` must be even and ≥ 2·(atoms − 1).
3. **Isotope pattern (rule 3)**: not applicable to a single centroided MS/MS ion; replaced by the
   isotopologue rule in §2.1.
4. **H/C (rule 4)**: extended range 0.1 ≤ H/C ≤ 6 (99.99 % coverage) for `P − L`; for `L` the range is relaxed
   to 0 ≤ H/C ≤ 6 (losses such as CO, CO2, C2O2, CS2 have no hydrogen) and C-free losses are allowed
   only from a whitelist: H, H2, O, O2, H2O, NH3, N2, NO, NO2, HNO2, HNO3, HF, HCl, HBr, HI, H2S, SO, SO2, SO3,
   H2SO4, HPO3, H3PO4, H4P2O7, Cl2, Br2, I2, plus any multiple of H added to these. C-free `P − L` is rejected
   (an organic fragment ion carries carbon) except the adduct cation itself (Na+, K+, NH4+).
5. **Heteroatom ratios (rule 5)**, extended ranges: N/C ≤ 4, O/C ≤ 3, P/C ≤ 2, S/C ≤ 3, F/C ≤ 6, Cl/C ≤ 2,
   Br/C ≤ 2, Si/C ≤ 1; for `P − L`. For `L` the same bounds apply once C ≥ 1.
6. **Element-probability combinations (rule 6)**: NOPS all present → N < 10, O < 20, P < 4, S < 3;
   NOP > 1 each → N < 11, O < 22, P < 6; OPS > 1 → O < 14, P < 3, S < 3; PSN > 1 → P < 3, S < 3, N < 4;
   NOS > 6 → N < 19, O < 14, S < 8. Applied to `P − L`; trivially satisfied by small `L`.
7. **TMS (rule 7)**: GC-only, not used.

Extra constraints on `L` beyond the Golden Rules: 0 ≤ Δ; `H(L) ≤ H(P)` etc. is the sub-formula condition;
hydrogen of `L` may range over `0..H(P)` without parity constraint (H transfers and radicals).

### 2.1 Mass tolerance, calibration, isotopologues, precursor region, adducts, charge

- **Tolerance**: `tol(Δ) = sqrt(tol_p² + tol_f²)`, `tol_f = max(ppm·m/z(f), floor)`, defaults 10 ppm / 3 mDa
  (Orbitrap, QTOF), to be *calibrated* on NIST23 (§3.1) per instrument type; the user-facing parameter is one
  ppm value plus a floor.
- **Recalibration**: if the precursor ion (or its 13C isotopologue) is present in the spectrum, its deviation
  from m/z(p) is used as a constant offset for the whole spectrum (the published method inflates the
  tolerance instead; offsetting is tighter).
- **Isotopologues**: an ion that fails on its own is kept, labelled `iso`, if an ion 1.00336 (13C), 1.9958
  (34S), 1.9970 (37Cl) or 1.9980 (81Br) below it passes and is more intense (34S/37Cl/81Br only when the
  passing ion's explanation contains that element).
- **Precursor region**: ions within 1.6 Da below m/z(p) are only accepted as H, H2, or isotopologue; ions above
  m/z(p) + tol are removed unless they are precursor isotopologues (z = 1 ⇒ nothing above the precursor).
- **Adducts** (T1 only; T0 needs none, the ion mass is decomposed directly): the loss mass needs no adduct, but
  the sub-formula test needs the ion's atom budget `P` = `M` + carrier ([M+NH4]+ loses NH3 even when `M` has
  no N; [M+HCOO]− loses HCOOH). The carrier is taken from the adduct string when given and checked against
  m/z(p) − mass(M); the carrier-derived losses are on the whitelist so an unknown carrier costs nothing.
- **Charge**: v1 handles |z| = 1; multiply charged precursors are passed through unchanged with a flag.
- **In-source adduct reactions** (`[M+N2+H2O]` for substituted aromatics, used in the published code): off by
  default; evaluated on NIST23 as an option.

#### 2.1b Findings on NIST23 that changed the rules (2026-10-29)

- Losses are sums of several neutrals (NH3 + CH4 = CH7N): the H/C <= 6 bound is replaced by a saturated-hydride
  bound H <= 2(C+O+S) + 3(N+P) + monovalent + 2, for losses and fragment ions alike.
- Halide -> OH exchange products ([M+H − HCl + H2O]+ of aryl halides) are real and frequent in NIST flow
  injection spectra: losses of `P + H2O` carrying a halogen are permitted (option `exchange`).
- Precursor isotopologues sit above the precursor m/z: the isotopologue rescue also applies to them.
- Part of NIST23 is on a 2- or 3-decimal m/z grid: half a grid step is added to the tolerance for those spectra.

## 2.2 Electronic noise

Kept as the first step, as in the published method: groups of ≥ 4 ions sharing one intensity within ±0.1 %
are removed (threshold = 99.5th percentile of such groups in NIST23; re-derived on our NIST23 copy).

### 2.3 Output and score

Per ion: `keep` (bool), `reason` (`loss`, `precursor`, `iso`, `forbidden`, `above_precursor`, `electronic`),
best loss formula, mass error, and the number of candidate explanations. v1 decides binary; a soft score
(mass-error Gaussian × 1/#explanations) is written too and compared against binary in §3.

### 2.4 Algorithm and speed (hard requirement: fast)

- T1: enumerate the sub-formula masses of `P` once (product of element ranges; ≤ ~10^5 for metabolites,
  pruned by the rules), sort, one `searchsorted` for all fragments. Cached per (formula, adduct).
- T0: decompose m/z(p) into candidate `P` (bounded CHNOPS, halogens optional) within tolerance once per
  spectrum; decompose each Δ into candidate `L` (small masses, fast); keep `f` if some pair (`P`, `L`) has
  `L ⊆ P` with `P − L` plausible. Both decomposers in numba with precomputed element-mass tables.
- Target: NIST23 [M+H]+ positive (567,631 spectra, 21.3 M ions) in < 5 min on one zen4 node, T1.

## 3. Validation

All real-data computation runs through `srun -A metabolomicsgrp -p high` (login node only edits). Each
dataset is a separate script under `bench/denoise/` writing to `results/denoise/<dataset>/` and a short
markdown table; the user decides when each runs.

### 3.0 Loss-axis coverage (no data needed)

For precursor masses 100–1000 Da and tolerances 3/5/10/20 mDa: the fraction of the loss axis that T0 permits,
and the fraction that "any plausible formula" permits, as a function of loss mass. This quantifies what T0 can
and cannot remove, and where the chemical-noise model of §3.3 is unremovable by construction.

### 3.1 NIST23 (local: `/quobyte/metabolomicsgrp/fanzhou/parsed_msp/nist23/*.npz`, 567k `[M+H]+` pos, plus neg and other adducts; raw `.msp` 2.8 GB)

Library spectra are consensus, curated spectra: nearly every ion is a real fragment, so this measures
**false removal**. Metrics per spectrum: ions and intensity retained, T1 (NIST formula + adduct) and T0 (m/z
only); stratified by instrument type (HCD, QTOF, IT), NCE, precursor mass bin, adduct, polarity.
Expected: ≥ 98–99 % intensity retained under T1. Removed ions are categorised (isotopologue, precursor region,
formula/adduct error in the record, N2/H2O in-source adduct, true noise) on a random sample with a gallery.
Tolerance calibration: the distribution of loss mass errors of explained ions, per instrument type → ppm/floor
set at its 99.5th percentile. Baseline: the published `spectral_denoising` package (new conda env with rdkit)
on a 20k random subset; agreement per ion and the ions on which the two disagree.

### 3.2 Serial dilution series (Kong et al. 2025; Zenodo 14920689)

Files: `dilution_data_mapped.csv` (16.6 MB; spectra from 200 → 0.02 pmol, each paired with its 500 pmol
reference, with raw, published-denoised and reference peaks, SMILES, formula, adduct, NCE) and
`dilution_data_all.csv` (21 MB). 240 standards, 13 amounts, Q Exactive HF, HILIC, NCE 25/35/65.
Metrics (as the paper): entropy similarity to the reference per amount for raw, 1 % threshold, published
denoising (column in the file), ours T1, ours T0; annotation rate at similarity > 0.75; lowest amount with a
confident match per compound (fold change); explained intensity. T0 is the real test: it uses nothing but the
precursor m/z, which is what an unknown in a real sample offers.

### 3.3 Synthetic noise

Protocol of the paper, re-implemented in `specdenoise/synth.py`: clean spectra = the 6,091 mapped dilution
spectra and a 20k NIST23 sample; chemical noise m/z from a formula database (all Golden-Rules-plausible neutral
formulas 50 – m/z(p), built by us; the paper's 3.5 M-formula `formulaDB_sorted.csv` is on Zenodo for
cross-checking) with Poisson(λ = 50)/100 intensities; electronic noise uniform in [50, m/z(p)] with
Poisson(λ = 5)/100; noise counts 1/10, 1/5, 1/2 of the true ions, 3 × 3 levels, ≥ 1 ion. Metrics: noise ions
removed (recall), true ions kept (specificity), entropy similarity to clean before/after; by tier and tolerance;
baseline published package. Two harder variants: (a) chemical noise drawn only from losses that are plausible
sub-formulas of the precursor (the ceiling of any loss-based method); (b) precursor m/z error of 2–10 ppm
to test the tolerance model. The Zenodo `synthetic_noise_data.zip` (446 MB) is used to check that our
generator reproduces the paper's numbers.

## 4. Code layout

```
specdenoise/            new package (pure numpy + numba; no rdkit)
  formula.py            parse/format formulas, monoisotopic masses, adduct table, Golden-Rules checks
  losses.py             sub-formula enumeration (T1), mass decomposers (T0), permitted-loss tables
  denoise.py            electronic step, loss step, isotopologue/precursor rules, API: denoise(mz, inten, precursor_mz, formula=None, adduct=None, ...)
  synth.py              noise generators (chemical / electronic / sub-formula-constrained)
  __main__.py           CLI: .msp in → .msp out with per-ion labels
  tests/                unit tests (formula algebra, rules on literature examples, synthetic spectra, tolerance, isotopes, adducts)
bench/denoise/          nist23.py, dilution.py, synthetic.py, coverage.py  → results/denoise/
data/denoise/zenodo_14920689/   downloaded files (job, ~90 MB without the zips; +0.5 GB with synthetic_noise_data.zip)
```

## 5. Order of work and stop points

1. Package + unit tests + §3.0 coverage analysis (no real data).  → stop, report.
2. Download Zenodo files; conda env for the published package (comparison).
3. §3.1 NIST23 → report → 4. §3.2 dilution → report → 5. §3.3 synthetic → report.
Each real-data step waits for a go.

## 6. Decisions needed

- Element set for T0 candidate precursors: CHNOPS only (default) or + F/Cl/Br/I/Si (slower, permits more).
- `[M+N2+H2O]` in-source option: evaluate on NIST23 and keep only if it rescues real ions.
- Package name/location `specdenoise/` in this repo, or a separate repo.
