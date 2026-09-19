# g16dump

[![CI](https://github.com/lgutsev/fcidump_exporter/actions/workflows/ci.yml/badge.svg)](https://github.com/lgutsev/fcidump_exporter/actions/workflows/ci.yml)

FCIDUMP files for SHCI (Dice) and DMRG (Block2), built from a Gaussian 16
matrix-element file without ever transforming integrals over the full MO space.

Gaussian transforms the two-electron integrals over the active window only. The
frozen core is folded analytically into an effective one-electron Hamiltonian
`h'` and a scalar `E_core`. This takes seconds where a full-space dump takes
days.

**Status: M1-M4. The core package, the FCIDUMP writer and the active-space
rotations are implemented and tested.** `bundle.py`, `matfile.py`,
`hamiltonian.py`, `write.py` and `rotate.py` are in place; the frozen-core
algebra is checked against an independent full-transform route, the written
FCIDUMP against an independent parser, and the rotations against the exact
many-body spectrum. The Gaussian route section in `gaussian/` is still
unverified and the matrix-element labels are still unconfirmed, so `extract` has
not yet read a real `.mat`. See [Project status](#project-status).

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

The gate behaves as designed on a Roothaan operator. Handed one for CH2/6-31G
it rejects the input by 0.652 Ha, while the same job's genuine `f^α`/`f^β` agree
to 7e-15. Which of the two a Gaussian `.mat` actually carries is still the M0
question, and the answer changes nothing in the code: a Roothaan operator is
rejected either way, and the rebuilt-Fock path is there either way.

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

Step 1 is the only part that needs MOKIT, and it is separable:
`hamiltonian.with_rebuilt_fock(bundle, mol)` takes any PySCF `Mole`, so the path
can be exercised, and a molecule supplied, without MOKIT present.

`pyscf` and `mokit` become runtime dependencies for this mode only. They are
imported lazily, and their absence is reported as a clear error rather than a
traceback. MOKIT is not installable from PyPI.

## The bundle

`matfile.py` runs where gauopen lives and emits one `.npz`; everything
downstream reads only that. The schema is version `2`:

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | `2` |
| `source_program`, `source_file` | str | where this came from |
| `reference_type` | str | `RHF`, `ROHF`, `RKS` or `ROKS` |
| `charge`, `multiplicity` | int | |
| `nelec`, `nalpha`, `nbeta` | int | |
| `nao`, `nmo` | int | AO and MO counts |
| `ncore`, `nact` | int | frozen core and active orbital counts |
| `active_first`, `active_last` | int | the active window, **1-based and inclusive** |
| `enuc` | float | nuclear repulsion |
| `C` | (nao, nmo) | MO coefficients |
| `S`, `Hcore_ao` | (nao, nao) | AO overlap and core Hamiltonian |
| `eri_active` | (nact,)×4 | chemist's `(tu|vw)`, MO basis, active window only |
| `fock_source` | str | `gaussian`, `pyscf_rebuilt` or `none` |
| `provenance` | str | JSON |

Optional, and absent rather than null when unknown: `F_alpha_ao`, `F_beta_ao`,
`orbital_energies`, `orbital_energies_beta`, `escf`, `atom_charges`.

### Accessors

Do not do index arithmetic on the window. `bundle.active` is the 0-based slice,
`bundle.core` the frozen-core slice, and `bundle.active_start` /
`bundle.active_stop` the 0-based half-open pair. Also there:
`nocc_active_alpha`, `nocc_active_beta`, `nelec_active` and `ms2` for the
FCIDUMP header, `nfrozen_virtual`, `is_ks` and `has_fock`. Every conversion
between the stored 1-based window and 0-based array indices happens inside
those and nowhere else.

### Conventions

`C` is stored **column-wise**, `C[ao, mo]`, so that `CᵀSC = I`. That is PySCF's
convention, and every oracle here is PySCF. The algebra above is written in the
row convention, where the same transform reads `C h_ao Cᵀ`; `matfile.py` decides
which one Gaussian handed it by testing both numerically, rather than trusting a
storage convention, and normalises to the column one.

`active_first` and `active_last` are the **1-based, inclusive** `NFIRST` and
`NLAST` of the Gaussian route, stored unchanged so that a bundle records the job
that was actually asked for. `ncore` and `nact` are stored redundantly, and
validation requires all four to agree, so a window that does not mean what the
route meant cannot pass quietly.

Fock matrices are stored in the **AO** basis, so that a rotation of the active
space never has to touch them. `hamiltonian.py` does the `CᵀFC` transform.

`orbital_energies` is a **diagnostic** and is never a substitute for a Fock
matrix. That substitution is the defect this package replaces.

A **KS bundle never carries a stored Fock matrix at all.** `matfile.py` refuses
to read the KS matrix into `F_alpha_ao`, so a KS bundle arrives with
`fock_source = "none"` and the rebuilt-Fock path is the only way to use it.
Validation rejects a KS reference together with `fock_source = "gaussian"` as a
second line of defence.

`provenance` carries the Gaussian filename, the route line, basis and method
where known, the 1-based window as Gaussian saw it, the Git commit, the schema
version, and whether the Fock matrix came from Gaussian or was rebuilt through
PySCF. Every rotation appends a record of itself, so a bundle says how it was
re-expressed and never silently forgets.

## The FCIDUMP

`write.py` writes standard FCIDUMP: the `&FCI` namelist with `NORB`, `NELEC`,
`MS2`, `ORBSYM` and `ISYM`, then the symmetry-unique two-electron integrals
(`t ≥ u`, `v ≥ w`, `(tu) ≥ (vw)` — one entry in eight), the lower triangle of
`h'`, and `E_core` last with all four indices zero. Integrals carry seventeen
significant digits, which round-trips a double exactly; `--threshold` omits
two-electron integrals below a magnitude and never touches the one-electron
block, which is too small to be worth compressing and too consequential to
truncate.

`ORBSYM` defaults to all-`1`, i.e. C1. The windowed route does not carry
Gaussian's irrep labels through, and a rotated active space has no symmetry left
to label; a wrong `ORBSYM` makes a solver search the wrong symmetry sector, so
the default is the one that claims nothing.

FCIDUMP has no comment syntax its readers agree on — PySCF's parser, for one,
joins the lines above `&END` and splits them on `=`, so a comment line there
makes the file unreadable. Provenance therefore goes in `FCIDUMP.provenance.json`
beside the file, and the FCIDUMP itself stays strictly standard.

## Rotating the active space

`h' → Uᵀh'U` with the matching four-index transform of the ERIs, for any real
orthogonal `U`. `E_core`, the electron counts and the many-body spectrum are
invariant, which is the sharpest correctness test in the package: the FCI
eigenvalues are computed before and after a random rotation and compared, with
numpy alone, in `tests/test_rotate.py`. A non-orthogonal matrix is refused
unless `--diagnostic` says to allow it.

Two things can be rotated and the difference matters. `rotate_hamiltonian`
rotates `h'` and the ERIs and is exact for any `U`. `rotate_active_space`
rotates the orbitals themselves, and there is a precondition: the stored Fock
matrices were built from the reference determinant's own density, so a `U` that
mixes an active occupied orbital with an active virtual one leaves them
describing a determinant that no longer exists. Those bundles come back with
their Fock matrices **dropped**, a warning, and the reason in their provenance,
so `active_hamiltonian` refuses them and says to rebuild through PySCF rather
than quietly returning a wrong `h'`.

## Rotating the active space

`h' -> Uᵀh'U` with the matching four-index transform of the ERIs, for any real
orthogonal `U`. `E_core`, the electron counts and the many-body spectrum are
invariant, which is the sharpest correctness test in the package: the FCI
eigenvalues are computed before and after a random rotation and compared, with
numpy alone, in `tests/test_rotate.py`. A non-orthogonal matrix is refused unless
`--diagnostic` says to allow it.

Two things can be rotated and the difference matters. `rotate_hamiltonian`
rotates `h'` and the ERIs and is exact for any `U`. `rotate_active_space` rotates
the orbitals themselves, and there is a precondition: the stored Fock matrices
were built from the reference determinant's own density, so a `U` that mixes an
active occupied orbital with an active virtual one leaves them describing a
determinant that no longer exists. Those bundles come back with their Fock
matrices **dropped**, a warning, and the reason in their provenance, so
`active_hamiltonian` refuses them and says to rebuild through PySCF rather than
quietly returning a wrong `h'`. Orbital energies are dropped by every rotation:
inside a rotated window they are the diagonal of nothing.

## Using it

```bash
g16dump inspect JOB.mat                    # can extract read this file?
g16dump extract JOB.mat --fch JOB.fch --reference ROHF --window 6 40 --out JOB.npz
g16dump validate JOB.npz --hamiltonian     # builds h', checks E_ref against escf
g16dump dump JOB.npz --out FCIDUMP          # + FCIDUMP.provenance.json
g16dump rotate JOB.npz --rotation U.npy --out rotated.npz
```

`inspect` reports which of the blocks g16dump needs are present, under which
labels, and whether their dimensions agree with the header. It is the M0 gate in
miniature. For an exhaustive dump of everything a `.mat` contains, assuming
nothing about labels at all, use `scripts/inspect_mat.py`.

`--reference` is required and is never inferred. `--window` may be omitted, in
which case the partition the `.mat` reports is used; either way it is
cross-checked against the dimension of the two-electron block that is actually
present, and a disagreement stops the run.

Errors say what is scientifically wrong — that the α- and β-derived effective
Hamiltonians disagree, that the MO coefficients are not orthonormal in the
supplied AO overlap, that a multiplicity is incompatible with the electron
count — rather than surfacing a raw numpy exception.

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
              rotate.py    rotations inside the active space
              write.py     FCIDUMP writer
              cli.py       inspect / extract / validate / dump / rotate
gaussian/     route templates + how to run them
legacy/       the original scripts, untouched, for reference and regression
scripts/      inspect_mat.py, the matrix-element probe
tests/        fixtures as committed .npz, no Gaussian and no gauopen needed
              make_fixtures.py regenerates them (needs pyscf)
```

The split exists because gauopen may not build everywhere and needs a compiled
component. `matfile.py` runs on the cluster and emits a plain `.npz`; everything
downstream reads only the `.npz`, and the tests run off committed `.npz`
fixtures.

## Installation

```bash
pip install -e .                  # core: numpy only
pip install -e ".[validate]"      # + pyscf, for the oracles and the fallback
pip install -e ".[test]"          # + pytest
pip install -e ".[dev]"           # + pyscf, pytest, ruff
```

Neither `gauopen` nor `mokit` is on PyPI, so neither can appear in an extra.

`gauopen` is **not** vendored here either. Get it from Gaussian, Inc. (it ships
with the Gaussian distribution as the `gauopen` directory), build its compiled
component, and put it on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/gauopen:$PYTHONPATH
python3 -c "import QCMatEl; print(QCMatEl.__file__)"
```

Only `g16dump extract` (i.e. `matfile.py`) needs it.

`mokit` comes from conda-forge or a source build, and only the rebuilt-Fock
comparison path uses it. Nothing in CI does: CI runs the Gaussian-independent
suite on Python 3.9 through 3.13 with numpy alone, and the PySCF oracles
separately. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to mark a test that
needs any of these.

## Project status

| Milestone | What it delivers | State |
|---|---|---|
| M0 | legacy inventory, `.mat` probe, route templates | probe and templates written; **waiting on cluster output** |
| M1 | test systems + two independent oracles | done: H2O RHF, CH2/NH/O2 triplet ROHF, a rotated ROHF set and B3LYP orbitals, each checked against a full AO→MO transform written with explicit loops and importing nothing from `g16dump` |
| M2 | `bundle.py`, `hamiltonian.py`, `write.py` + gates | done, with the schema, electron-count, window, Hermiticity, orthonormality, ERI-symmetry and α/β gates |
| M3 | writer and solver interface | writer done, round-tripped through an independent parser and through PySCF's; the Dice/Block2 examples are outstanding |
| M4 | active-space rotations | done, with the many-body spectrum checked before and after in a numpy-only FCI |
| M5 | benchmarks | not started |
| M6 | packaging, CI, DOI | **CI and packaging done** (3.9–3.13, numpy-only core enforced); LICENSE, CITATION.cff and DOI still open |

What M0 still gates is `extract` alone. Everything downstream of the bundle is
exercised by the committed fixtures, so the labels in
`g16dump.matfile.LABELS` are the only thing waiting on a real `.mat`.

Open questions blocking M0's gate are listed in
[`gaussian/README.md`](gaussian/README.md) and in the probe script's output
section.

## Credits and licensing

The original closed-shell derivation and the scripts in `legacy/` are by an
author to be named here before release; `CITATION.cff` and `LICENSE` are
deliberately not yet written, so that neither invents an attribution. Both land
in M6.
