# g16dump

FCIDUMP files for SHCI (Dice) and DMRG (Block2), built from a Gaussian 16
matrix-element file without ever transforming integrals over the full MO space.

Gaussian transforms the two-electron integrals over the active window only. The
frozen core is folded analytically into an effective one-electron Hamiltonian
`h'` and a scalar `E_core`. This takes seconds where a full-space dump takes
days.

**Status: the core package is implemented and validated against PySCF (M2
gates pass); the Gaussian side is not yet verified.** Every downstream stage --
bundle, frozen-core algebra, FCIDUMP writer, rotations, CLI -- has been checked
against an independent code on real molecules. What has *not* happened yet is a
real Gaussian `.mat` going through `g16dump extract`: the route section in
`gaussian/` and the Fock-matrix labels remain unconfirmed until the two M0 probe
jobs have run. See [Project status](#project-status).

## Why this exists

An earlier set of scripts (kept verbatim in `legacy/`) implemented this idea for
closed shells. The algebra in them is wrong for open shells, because they assume
the MO Fock matrix is diagonal with orbital energies on the diagonal:

```python
me.MOE  = me.matlist["ALPHA ORBITAL ENERGIES"].expand().copy()
me.MOEd = np.diag(me.MOE)
h1e_act = me.MOEd[ncore:iact, ncore:iact]     # <- f assumed diagonal
```

That holds for canonical RHF and for nothing else. It is false for ROHF, and
false for Kohn–Sham orbitals (the KS matrix is not the HF Fock matrix). The
symptom is visible in the shipped dumps: `legacy/FePorph_3_Window.dat` puts the
ROHF triplet reference 0.67 Ha *below* the RHF singlet in
`legacy/FePorph_1_Window.dat`, which is impossible. The two files also differ in
`E_core` by 0.48 Ha, and 186 of the 210 one-electron off-diagonals in each are
exactly zero — the fingerprint of a `h'` whose only off-diagonal content comes
from the J/K correction, because the Fock off-diagonals were never there.

`legacy/FCIDUMP_Write_MOe_3.py` is separately inconsistent with itself: it builds
`E_core` from both the α and β one-electron blocks, then writes only the α block.

**The fix:** build `h'` from the full MO Fock matrices `f^α` and `f^β`,
off-diagonals included, with one code path for RHF and ROHF. Not from orbital
energies, and not with `ak`/`bk`/`ck` coupling coefficients — that approach has
already been tried and failed.

## The math

Notation: spatial orbitals shared by α and β (RHF/ROHF); `C` the frozen-core
orbitals (doubly occupied); `A` the active orbitals; `O_σ ⊂ A` the active
orbitals occupied in the reference determinant for spin σ — the first
`nalpha − ncore` and `nbeta − ncore` active orbitals. Chemist's notation
`(pq|rs)`.

Three cheap inputs, none of them a full-space transform:

```
h_pq   = C h_ao Cᵀ                 over all MOs, O(N³)
f^σ_pq = C F^σ_ao Cᵀ               over all MOs, O(N³)
(tu|vw)                            for t,u,v,w ∈ A only, from Gaussian's window
```

### Effective one-electron Hamiltonian

For each spin σ, over the full active block — a matrix, not a diagonal:

```
h'_tu = f^σ_tu − Σ_{k∈O_σ} [ (tu|kk) − (tk|ku) ] − Σ_{k∈O_σ̄} (tu|kk)
```

This is exact when `f^σ` is the UHF-type Fock operator built from the reference
determinant's densities, `F^σ = H + J[P^α + P^β] − K[P^σ]`. The result is
spin-independent, so the α and β expressions must agree — which gives a free
internal check:

```
max |h'(α) − h'(β)| < 1e-8
```

If that fails for ROHF, the stored Fock is Gaussian's Roothaan *effective*
operator rather than `f^α`/`f^β`. **Do not average the two.** Disagreement is a
bug signal; switch to the rebuilt-Fock path below.

### Reference, active and core energies

```
E_ref  = E_nuc + ½ Σ_σ Σ_{i∈C∪O_σ} ( h_ii + f^σ_ii )

E_act  = Σ_σ Σ_{t∈O_σ} h'_tt
       + ½ Σ_σ Σ_{t,u∈O_σ} [ (tt|uu) − (tu|ut) ]
       + Σ_{t∈O_α, u∈O_β} (tt|uu)

E_core = E_ref − E_act
```

### Reduction to the closed-shell case

For RHF, `O_α = O_β = O` and `f^α = f^β = f`, so the two sums over `O_σ` and
`O_σ̄` collapse:

```
h'_tu = f_tu − Σ_{k∈O} [ (tu|kk) − (tk|ku) ] − Σ_{k∈O} (tu|kk)
      = f_tu − 2 Σ_{k∈O} (tu|kk) + Σ_{k∈O} (tk|ku)
```

which is exactly the legacy closed-shell expression

```python
h1e_act -= 2*np.einsum('acbb->ac', h2e[:nact, :nact, :nact_2e, :nact_2e])
h1e_act += np.einsum('abbc->ac', h2e[:nact, :nact_2e, :nact_2e, :nact])
```

with the single difference that `f_tu` is the full Fock matrix here and
`diag(ε)` there. The two agree only when the orbitals are canonical RHF. This
equivalence is asserted numerically by the full-space and legacy-regression
tests, not just claimed here.

### Rebuilt-Fock fallback

Required when the `.mat` carries no usable HF-type Fock matrix, and **always**
required for KS orbitals:

1. Load the molecule from the matching `.fch` with MOKIT's `load_mol_from_fch`.
2. Build `P^α`, `P^β` from the Gaussian orbitals and occupations.
3. One PySCF JK build: `F^σ = H + J[P^α + P^β] − K[P^σ]`. One Fock build, far
   cheaper than an `ao2mo`.
4. Feed the result into the same functions as above.

`pyscf` and `mokit` become runtime dependencies for this mode only. They are
imported lazily, and their absence is reported as a clear error rather than a
traceback.

## KS warning

**For Kohn–Sham orbitals the stored KS matrix contains exchange–correlation and
must never be used as `f^σ`.** The active-space Hamiltonian is always the *HF*
Hamiltonian evaluated in the KS orbitals. A KS bundle fed to the stored-Fock
path raises; use the rebuilt-Fock path. This warning is repeated in the CLI.

## Not in v1

- **UHF references** (different α and β orbitals). A spin-restricted FCIDUMP
  cannot represent them. The workaround is to generate UHF natural orbitals
  (UNOs) in Gaussian and send those through the normal path. A UHF-FCIDUMP
  writer can come later if a solver we use needs one.
- **ORCA input.** Possible later through the same bundle format.
- **Orbital optimization (SHCISCF) across the core/active boundary.** That needs
  integrals from outside the window; use PySCF for it.

## Design constraints

- Correctness is verified against an independent code, never assumed.
- Plain modules and plain functions. No class hierarchies, no plugin systems.
- Core runtime dependency is `numpy` only. `pyscf` and `mokit` are
  validation/fallback dependencies. `gauopen` is needed only by the reader.
- gauopen is never vendored — it is depended on, and its installation
  documented.
- No hardcoded paths. Everything comes from CLI arguments.

## Layout

```
g16dump/      matfile.py   .mat -> .npz bundle (the only module importing QCMatEl)
              bundle.py    load/validate the .npz bundle, shape assertions
              hamiltonian.py   h', E_core, E_ref from Fock matrices
              rotate.py    unitary rotations inside the active space
              write.py     vectorized FCIDUMP writer + Dice nocc line
              solvers.py   Dice / Block2 inputs, output parsers, FCI validation
              manifest.py  benchmark manifests (JSON) and Gaussian job generation
              sweep.py     active-space sweeps and convergence analysis
              cli.py       extract / dump / validate / rotate / info
gaussian/     route templates + how to run them
legacy/       the original scripts, untouched, for reference and regression
scripts/      inspect_mat.py (the .mat probe), gate_report.py (the M2 gates)
benchmarks/   systems.json, the benchmark manifest
tests/        fixtures as committed .npz, no Gaussian and no gauopen needed
```

The split exists because gauopen may not build everywhere and needs a compiled
component. `matfile.py` runs on the cluster and emits a plain `.npz`; everything
downstream reads only the `.npz`, and the tests run off committed `.npz`
fixtures.

## Usage

```bash
# On the cluster, where gauopen lives. --ref-type is required: a .mat does not
# record whether its orbitals are Hartree-Fock or Kohn-Sham.
g16dump extract JOB.mat --ref-type ROHF --fch JOB.fch --out JOB.npz

# Anywhere, with NumPy alone.
g16dump validate JOB.npz --check-hamiltonian
g16dump dump     JOB.npz --out FCIDUMP --dice-input input.dat
g16dump rotate   JOB.npz --rotation U.npy --out rotated.npz
g16dump dump     rotated.npz --out FCIDUMP.rotated
g16dump info     JOB.npz          # summary plus full provenance
```

For Kohn-Sham orbitals, `extract` needs `--rebuild-fock --fch JOB.fch` and
refuses to run without it.

Every output records its provenance: source file, active window, reference
type, basis and method where known, the g16dump commit, the schema version, and
whether the Fock matrices came from Gaussian or were rebuilt with PySCF.

### Why `rotate` produces a Hamiltonian file, not a bundle

A general rotation of the active orbitals mixes occupied with virtual, which
changes the reference determinant -- and therefore the density that built the
stored Fock matrices. A rotated *bundle* would carry Fock matrices describing a
determinant that no longer exists, and `h'` rebuilt from them would be wrong.
The reduced Hamiltonian (`h'`, ERIs, `E_core`) has no such problem: the
transform is exact for any orthogonal `U`, and `E_core` is unchanged because the
frozen core is not involved. So `rotate` takes a bundle and emits a reduced
Hamiltonian, which `dump` accepts directly.

### Solvers, manifests and sweeps

`g16dump.solvers` writes Dice and Block2 inputs whose reference determinant,
electron count and spin are taken from the Hamiltonian rather than typed by
hand. `g16dump.manifest` validates benchmark definitions -- electrons counted
from the geometry and checked against the multiplicity and the active window --
and generates the Gaussian jobs. `g16dump.sweep` builds active-window and
spin-state sweeps and reports, per window, the energy change to the next larger
space, the change in spin-state splitting, and natural-orbital diagnostics.

## Validation

`python3 scripts/gate_report.py` runs the M2 gates against PySCF and prints the
residuals. Current output:

```
system                       |dE_ref|      spin   max|dh|  |dEcore| max|dERI|  roundtrip   |dE_FCI|
---------------------------------------------------------------------------------------------------
H2O   RHF  STO-3G            7.11e-14  0.00e+00  3.55e-15  1.42e-14  0.00e+00   8.33e-16   1.78e-14
H2O   RHF  6-31G             8.53e-14  0.00e+00  8.88e-15  4.26e-14  0.00e+00   1.08e-15   4.90e-13
CH2   ROHF 6-31G             2.13e-14  3.96e-15  1.09e-14  2.84e-14  0.00e+00   2.36e-15   9.72e-13
O2    ROHF 6-31G             1.42e-13  2.00e-15  1.07e-14  5.68e-14  0.00e+00   8.05e-16   1.78e-13
NH    ROHF 6-31G             2.13e-14  8.88e-16  4.00e-15  7.11e-15  0.00e+00   5.55e-16   1.43e-12
N2    RHF  cc-pVDZ 2.0A      9.95e-14  0.00e+00  1.78e-15  0.00e+00  0.00e+00   2.78e-16   3.17e-12
CH2   ROHF rotated           2.13e-14  1.76e-15  5.77e-15  2.84e-14  0.00e+00   4.00e-15   2.79e-13
H2O   RKS  6-31G                n/a    0.00e+00  6.66e-15  2.13e-14  0.00e+00   1.01e-15   2.63e-13

worst residual across all gates: 3.165e-12  (tolerance 1e-08)
```

Columns: `|E_ref - E_SCF|` (including the determinant energy rebuilt from the
written file), `max|h'(a) - h'(b)|`, then `h'`, `E_core` and ERIs against
PySCF's `CASCI.get_h1eff` + `ao2mo`, the FCIDUMP round trip through an
independent parser, and the FCI energy change under a random active-space
rotation. `E_ref` is n/a for KS because the KS total energy includes
exchange-correlation and is not a determinant energy.

"CH2 ROHF rotated" mixes orbitals inside each occupation block, which leaves the
determinant unchanged but makes the MO Fock matrix non-diagonal by 0.35 Ha --
the regime where `diag(orbital energies)` fails. The legacy form is off by
0.097 Ha for canonical ROHF already, and 0.126 Ha after rotation; for canonical
RHF it agrees with ours to 1e-7, as it should.

Two findings from building the tests, both now pinned by tests:

- **A C-order/Fortran-order reshape of the ERI tensor cannot be detected by
  symmetry.** It is exactly `transpose(3,2,1,0)`, one of the eight operations the
  8-fold symmetry already guarantees, so the tensor comes out numerically
  identical. Index order has to be checked against an independent code.
- **PySCF's default FCI tolerance is too loose to compare orbital bases.** On
  stretched N2 in a randomly rotated basis it converges ~4.7e-4 Ha high. The
  invariance tests use `conv_tol=1e-12`, which brings the two to ~1e-11.

### Against MOKIT, and the reader end to end

With MOKIT installed (`scripts/install_mokit_without_conda.sh` where there is
no conda; `conda install mokit -c mokit -c conda-forge` where there is),
`tests/test_extract.py` runs `matfile.extract` end to end: a stand-in for
gauopen serves a `.mat` built from a PySCF calculation with every AO quantity
in **Gaussian's** AO order, next to a real `.fch` written by MOKIT, for bases
with d functions (6-31G*, def2-SVP, cc-pVDZ) and RHF, ROHF and RKS references.

That test found a real bug, now fixed. Gaussian and PySCF order spherical
d/f/g functions differently (`m = 0,+1,-1,+2,-2,...` vs `m = -l..+l`), and
the reader mixed `.mat` quantities (Gaussian order) with integrals from the
`.fch` molecule (PySCF order). s/p-only bases hide it. With d functions the
orbitals were not even orthonormal against the `.fch` overlap
(`max|CSC'-I| = 1.58` for H2O/6-31G*), and `--rebuild-fock` built densities in
the wrong AO basis: **for a Kohn-Sham reference that produced an `h'` wrong by
0.39 Ha with no error raised**, because KS has no SCF energy to check against.
`g16dump/aoorder.py` now maps between the two orders; the permutation matches
MOKIT's own reordering exactly for d, f and g functions (6-31G*, cc-pVTZ,
def2-TZVP, cc-pVQZ, Ni/def2-SVP). Before using a `.fch`, `extract` checks
that the `.mat` core Hamiltonian equals PySCF's after reordering, which catches
a wrong `.fch` or a Cartesian basis as an O(1) mismatch. Reintroducing the bug
makes all eight end-to-end cases fail. Cartesian 6D/10F functions are refused;
the Gaussian templates now request `5D 7F`.

A precision floor, measured rather than assumed: a `.fch` stores basis
exponents and coefficients to 9 significant figures, so J and K rebuilt from
it shift `E_ref` by ~1e-9 Ha for first-row molecules and **1.2-1.7e-7 Ha for
Ni complexes** (NiH, NiCl4(2-), def2-SVP). That is irrelevant at SHCI/DMRG
accuracy but above the 1e-8 `E_ref` gate, so for bundles whose Fock was rebuilt
from a `.fch` (recorded in provenance) the default gate is 1e-5 Ha. Genuine
errors, like the one above, are 1e-3 Ha and up. Every other path keeps 1e-8,
and `--tol-scf` overrides either.

MOKIT's `gen_fcidump` (Route A of the original plan) agrees with our dumps to
1e-8-8e-8 on H2O, CH2 and O2, including d-function bases. It is *not* an
independent check of the algebra -- internally it is PySCF's
`CASCI.get_h1eff` again -- and the residual is the same `.fch` precision limit,
so the original 1e-9 A-vs-B gate cannot be met through a `.fch`.

The test suite (358 tests) runs with NumPy alone; the PySCF oracle tests are
skipped when PySCF is absent. CI runs Python 3.9-3.13 NumPy-only, plus PySCF on
3.10 and 3.12, and fails if a core module ever imports an optional dependency.

## Installation

```bash
pip install -e .                  # core: numpy only
pip install -e ".[validate]"      # + pyscf, mokit for the oracles and fallback
pip install -e ".[test]"          # + pytest
```

`gauopen` is **not** installable from PyPI and is **not** vendored here. Get it
from Gaussian, Inc. (it ships with the Gaussian distribution as the `gauopen`
directory), build its compiled component, and put it on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/gauopen:$PYTHONPATH
python3 -c "import QCMatEl; print(QCMatEl.__file__)"
```

Only `g16dump extract` (i.e. `matfile.py`) needs it.

MOKIT is needed only for `--rebuild-fock`, the overlap fallback, and the
validation tests. On the cluster: `conda install mokit -c mokit -c conda-forge`.
Without conda: `scripts/install_mokit_without_conda.sh`, which unpacks MOKIT's
prebuilt package and puts its executables (`load_mol_from_fch` needs
`bas_fch2py`) on `PATH`.

## Project status

| Milestone | What it delivers | State |
|---|---|---|
| M0 | legacy inventory, `.mat` probe, route templates | probe and templates written; **waiting on the two cluster probe jobs** |
| M1 | test systems + independent oracle | done with PySCF-generated systems, plus MOKIT `.fch` round trips; Gaussian-generated fixtures still pending M0 |
| M2 | `bundle.py`, `hamiltonian.py`, `write.py` + gates | **gates pass** (table above) against PySCF; not yet on a real `.mat` |
| M3 | writer and solver interface | writer, Dice/Block2 inputs done; solver output parsers unverified against real output |
| M4 | active-space rotations | done; FCI invariance to ~1e-11 |
| M5 | benchmarks | manifest (`benchmarks/systems.json`) and sweep tooling done; no timings run |
| M6 | packaging, CI, DOI | CI done; `LICENSE`, `CITATION.cff`, DOI pending the author's name |

What is explicitly **not yet verified**, and why:

- **The Gaussian route section.** Proposed from documentation; it has not run.
- **The Fock-matrix labels in a `.mat`.** No legacy script reads one, so
  `matfile.py` searches rather than assumes, and falls back to
  `--rebuild-fock` when it finds nothing.
- **`matfile.extract` on a real `.mat`.** It now runs end to end against a
  synthetic `.mat` and a real MOKIT-written `.fch` (see above), but the gauopen
  API details -- labels, packing, what `expand()` returns -- are still
  assumptions until the probe output comes back.
- **MOKIT in CI.** The MOKIT-dependent tests skip in CI; they have run only in
  this development container.
- **Dice and Block2 output parsers.** Written from documented formats; no real
  output was available to check them against.
- **Porphyrin and legacy-Fe manifest entries.** Charge, spin, basis and window
  are specified, but the geometries are placeholders.

Open questions blocking M0's gate are listed in
[`gaussian/README.md`](gaussian/README.md) and in the probe script's output
section.

## Credits and licensing

The original closed-shell derivation and the scripts in `legacy/` are by an
author to be named here before release; `CITATION.cff` and `LICENSE` are
deliberately not yet written, so that neither invents an attribution. Both land
in M6.
