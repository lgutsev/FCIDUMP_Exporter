# g16dump

[![CI](https://github.com/lgutsev/fcidump_exporter/actions/workflows/ci.yml/badge.svg)](https://github.com/lgutsev/fcidump_exporter/actions/workflows/ci.yml)

FCIDUMP files for SHCI (Dice) and DMRG (Block2), built from a Gaussian 16
matrix-element file without ever transforming integrals over the full MO space.

Gaussian transforms the two-electron integrals over the active window only. The
frozen core is folded analytically into an effective one-electron Hamiltonian
`h'` and a scalar `E_core`. This takes seconds where a full-space dump takes
days.

**Status: M0 (inventory). Nothing downstream of the reader is implemented yet.**
The Gaussian route section in `gaussian/` is still unverified, and the
matrix-element labels for the Fock matrices are unconfirmed. See
[Project status](#project-status).

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
              cli.py       g16dump extract / g16dump dump
gaussian/     route templates + how to run them
legacy/       the original scripts, untouched, for reference and regression
scripts/      inspect_mat.py, the matrix-element probe
tests/        fixtures as committed .npz, no Gaussian and no gauopen needed
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
| M1 | test systems + two independent oracles | not started |
| M2 | `bundle.py`, `hamiltonian.py`, `write.py` + gates | not started |
| M3 | writer and solver interface | not started |
| M4 | active-space rotations | not started |
| M5 | benchmarks | not started |
| M6 | packaging, CI, DOI | not started |

Open questions blocking M0's gate are listed in
[`gaussian/README.md`](gaussian/README.md) and in the probe script's output
section.

## Credits and licensing

The original closed-shell derivation and the scripts in `legacy/` are by an
author to be named here before release; `CITATION.cff` and `LICENSE` are
deliberately not yet written, so that neither invents an attribution. Both land
in M6.
