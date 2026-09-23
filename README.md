# g16dump

[![CI](https://github.com/lgutsev/FCIDUMP_Exporter/actions/workflows/ci.yml/badge.svg)](https://github.com/lgutsev/FCIDUMP_Exporter/actions/workflows/ci.yml)

Solver-agnostic FCIDUMP files for large-active-space solvers such as DMRG
(Block2) and SHCI (Dice), built from a Gaussian 16 matrix-element file without
ever transforming integrals over the full MO space.

Gaussian transforms the two-electron integrals over the active window only. The
frozen core is folded analytically into an effective one-electron Hamiltonian
`h'` and a scalar `E_core`. This takes seconds where a full-space dump takes
days.

**Status: the pipeline is complete from the `.npz` bundle onward and tested;
`extract` has not yet read a real `.mat`.** The bundle, the open-shell
frozen-core algebra, the FCIDUMP writer and the active-space rotations are all
implemented, with the reference energy reproduced to 1e-9 Ha through an
independent read-back and through PySCF. What remains unverified is the front
door: the Gaussian route section in `gaussian/` and the matrix-element labels
are still unconfirmed against a real Gaussian run. See [Project status](#project-status).

## Current goal and scope

This repository is **not** a Dice workflow and it is **not** intended to grow
into a general multireference electronic-structure package. Its job is narrower:

> **Gaussian 16 windowed integrals -> validated active-space Hamiltonian ->
> solver-ready FCIDUMP.**

The FCIDUMP is the product. The downstream solver is deliberately replaceable.
Block2/DMRG and Dice/SHCI are current consumers and independent cross-checks,
not architectural dependencies.

### What we are doing right now

The code is effectively **feature-frozen for v1 until M0 is closed**. The next
work is empirical validation, not more infrastructure:

1. run the real Gaussian smoke-test suite in `gaussian/run_probes.sh`;
2. establish the actual `.mat` labels, window dimensions and RHF/ROHF/RKS
   Fock-matrix behaviour;
3. pass those files through `extract -> validate -> dump`;
4. compare the resulting Hamiltonians against the independent PySCF oracle;
5. read the same real Gaussian-derived FCIDUMP with Block2 and Dice where
   practical, and use allowed active-space rotations as a solver-level
   invariance check;
6. only change the implementation if one of those real tests exposes a defect.

Do **not** add new readers, abstractions, result databases, solver wrappers,
or additional polishing merely because they are possible. Those are outside
the present scope unless real validation reveals a concrete need.

### Scientific target after validation

The immediate scientific use is the **Fe analogue of the intramolecular
porphyrin "record-player" spin switch** that motivated earlier work on this
workflow. The legacy Gaussian-to-Dice route existed before there was a
production-quality large-active-space calculation behind it; this repository is
the missing correctness layer needed to revisit that problem properly.

The intended progression is:

- use Fe/Ni porphine-scale jobs as regression and stress tests of the exporter;
- recover the real Fe record-player geometries and build a **physically
  motivated** active-space ladder rather than inheriting the old CAS(22,21)
  window as a production choice;
- treat the relevant Fe spin manifold explicitly, including the quintet where
  chemically appropriate;
- use **Block2/DMRG as the leading production candidate**, with Dice/SHCI as an
  independent check on selected cases when useful;
- add a defensible dynamic-correlation treatment downstream if quantitative
  spin-state energetics require it.

Active-space selection, DMRG/SHCI convergence strategy, orbital optimization and
dynamic correlation are **scientific workflow responsibilities downstream of
this package**. They must not be silently folded into the exporter itself.

The stopping condition for this repository is therefore simple: once real
Gaussian extraction is validated end-to-end, independent consumers agree, and
the release metadata is complete, v1 is done. At that point effort should move
to the Fe record-player calculations rather than continued exporter expansion.

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
triplet reference 0.67 Ha *below* the singlet in `legacy/FePorph_1_Window.dat`,
which is impossible. The two files also differ in `E_core` by 0.48 Ha, and 186
of the 210 one-electron off-diagonals in each are exactly zero — the fingerprint
of a `h'` whose only off-diagonal content comes from the J/K correction, because
the Fock off-diagonals were never there. All three numbers are measured from the
committed files by `tests/test_legacy_regression.py`.

Those two jobs were almost certainly **UHF**, not RHF or ROHF: the legacy writer
reads `AA`, `BA` and `BB MO 2E INTEGRALS`, and three blocks is the unrestricted
form — a restricted job writes only `AA`. `legacy/Template.py` calls `scf.UHF`
as well. That makes the original approach stranger still, since a spin-restricted
FCIDUMP assumes one set of spatial orbitals and UHF has two.

`legacy/FCIDUMP_Write_MOe_3.py` is separately inconsistent with itself: it builds
$`E_{\text{core}}`$ from both the $`\alpha`$ and $`\beta`$ one-electron blocks, then
writes only the $`\alpha`$ block.

**The fix:** build $`h'`$ from the full MO Fock matrices $`f^{\alpha}`$ and $`f^{\beta}`$,
off-diagonals included, with one code path for RHF and ROHF. Not from orbital
energies, and not with `ak`/`bk`/`ck` coupling coefficients — that approach has
already been tried and failed.

## The math

Spatial orbitals are shared by $`\alpha`$ and $`\beta`$ (RHF/ROHF), and integrals
are in chemist's notation $`(pq|rs)`$. Three orbital sets matter:

| Symbol | Meaning |
|---|---|
| $`\mathcal{C}`$ | the frozen-core orbitals, doubly occupied |
| $`\mathcal{A}`$ | the active orbitals, i.e. Gaussian's window |
| $`\mathcal{O}_\sigma \subset \mathcal{A}`$ | the active orbitals occupied in the reference determinant for spin $`\sigma`$ — by index order, the first $`n_\sigma - n_{\text{core}}`$ of them |

$`\mathbf{C}`$, upright and bold, is the matrix of MO coefficients; $`\mathcal{C}`$,
script, is the frozen-core set. They are different objects and the distinction
matters in every expression below.

Three cheap inputs, none of them a full-space transform:

```math
\begin{aligned}
h_{pq} &= \mathbf{C}\, h^{\text{AO}}\, \mathbf{C}^{\mathsf{T}}
  &&\text{over all MOs, } O(N^3) \\
f^{\sigma}_{pq} &= \mathbf{C}\, F^{\sigma,\text{AO}}\, \mathbf{C}^{\mathsf{T}}
  &&\text{over all MOs, } O(N^3) \\
(tu|vw) && \text{for } t,u,v,w \in \mathcal{A} \text{ only, from the window}
\end{aligned}
```

### Effective one-electron Hamiltonian

For each spin $`\sigma`$, over the full active block — a matrix, not a diagonal:

```math
h'_{tu} = f^{\sigma}_{tu}
  - \sum_{k \in \mathcal{O}_{\sigma}} \Big[ (tu|kk) - (tk|ku) \Big]
  - \sum_{k \in \mathcal{O}_{\bar{\sigma}}} (tu|kk)
```

This is exact when $`f^{\sigma}`$ is the UHF-type Fock operator built from the
reference determinant's densities,

```math
F^{\sigma} = H + J\big[P^{\alpha} + P^{\beta}\big] - K\big[P^{\sigma}\big]
```

The result is spin-independent, so the $`\alpha`$ and $`\beta`$ expressions must
agree — which gives a free internal check:

```math
\max_{tu} \left| h'^{(\alpha)}_{tu} - h'^{(\beta)}_{tu} \right| < 10^{-8}
```

If that fails for ROHF, the stored Fock is Gaussian's Roothaan *effective*
operator rather than $`f^{\alpha}`$/$`f^{\beta}`$. **Do not average the two.**
Disagreement is a bug signal; switch to the rebuilt-Fock path below.

The gate behaves as designed on a Roothaan operator. Handed one for CH2/6-31G
it rejects the input by 0.652 Ha, while the same job's genuine
$`f^{\alpha}`$/$`f^{\beta}`$ agree to $`7 \times 10^{-15}`$. Which of the two a
Gaussian `.mat` actually carries is still the M0 question, and the answer
changes nothing in the code: a Roothaan operator is rejected either way, and
the rebuilt-Fock path is there either way.

### Reference, active and core energies

```math
E_{\text{ref}} = E_{\text{nuc}}
  + \frac{1}{2} \sum_{\sigma} \sum_{i \in \mathcal{C} \cup \mathcal{O}_{\sigma}}
    \Big( h_{ii} + f^{\sigma}_{ii} \Big)
```

```math
E_{\text{act}} = \sum_{\sigma} \sum_{t \in \mathcal{O}_{\sigma}} h'_{tt}
  + \frac{1}{2} \sum_{\sigma} \sum_{t,u \in \mathcal{O}_{\sigma}}
    \Big[ (tt|uu) - (tu|ut) \Big]
  + \sum_{t \in \mathcal{O}_{\alpha}} \sum_{u \in \mathcal{O}_{\beta}} (tt|uu)
```

```math
E_{\text{core}} = E_{\text{ref}} - E_{\text{act}}
```

$`E_{\text{core}}`$ is a *residual*, not an independent quantity, and that has a
consequence worth knowing: an error in the active two-electron integrals moves
$`E_{\text{act}}`$ and $`E_{\text{core}}`$ by equal and opposite amounts, so
$`E_{\text{ref}}`$ recovered from a FCIDUMP cannot see it. The test suite checks
the two separately for exactly this reason.

### Reduction to the closed-shell case

For RHF, $`\mathcal{O}_{\alpha} = \mathcal{O}_{\beta} = \mathcal{O}`$ and
$`f^{\alpha} = f^{\beta} = f`$, so the two sums over $`\mathcal{O}_{\sigma}`$ and
$`\mathcal{O}_{\bar{\sigma}}`$ collapse:

```math
\begin{aligned}
h'_{tu} &= f_{tu}
  - \sum_{k \in \mathcal{O}} \Big[ (tu|kk) - (tk|ku) \Big]
  - \sum_{k \in \mathcal{O}} (tu|kk) \\
&= f_{tu} - 2 \sum_{k \in \mathcal{O}} (tu|kk) + \sum_{k \in \mathcal{O}} (tk|ku)
\end{aligned}
```

which is exactly the legacy closed-shell expression

```python
h1e_act -= 2*np.einsum('acbb->ac', h2e[:nact, :nact, :nact_2e, :nact_2e])
h1e_act += np.einsum('abbc->ac', h2e[:nact, :nact_2e, :nact_2e, :nact])
```

with the single difference that $`f_{tu}`$ is the full Fock matrix here and
$`\mathrm{diag}(\varepsilon)`$ there. The two agree only when the orbitals
are canonical RHF. This equivalence is asserted numerically by the full-space
and legacy-regression tests, not just claimed here.

### Rebuilt-Fock fallback

Required when the `.mat` carries no usable HF-type Fock matrix, and **always**
required for KS orbitals:

1. Load the molecule from the matching `.fch` with MOKIT's `load_mol_from_fch`.
2. Build $`P^{\alpha}`$, $`P^{\beta}`$ from the Gaussian orbitals and occupations.
3. One PySCF JK build:
   $`F^{\sigma} = H + J[P^{\alpha} + P^{\beta}] - K[P^{\sigma}]`$.
   One Fock build, far cheaper than an `ao2mo`.
4. Feed the result into the same functions as above.

Step 1 is the only part that needs MOKIT, and it is separable:
`hamiltonian.with_rebuilt_fock(bundle, mol)` takes any PySCF `Mole`, so the path
can be exercised, and a molecule supplied, without MOKIT present.

`pyscf` and `mokit` become runtime dependencies for this mode only. They are
imported lazily, and their absence is reported as a clear error rather than a
traceback. MOKIT is not installable from PyPI.

## The bundle

`matfile.py` runs where gauopen lives and emits one `.npz`; everything
downstream reads only that. The schema is version `2`, and its keys are the
ones the project plan names:

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | `2` |
| `source_program`, `source_file` | str | where the bundle came from |
| `reference_type` | str | `RHF`, `ROHF`, `RKS` or `ROKS` |
| `charge`, `multiplicity` | int | |
| `nelec`, `nalpha`, `nbeta` | int | |
| `nao`, `nmo` | int | AO and MO counts |
| `ncore`, `nact` | int | frozen core and active orbital counts |
| `active_first`, `active_last` | int | the active window, 1-based and inclusive |
| `enuc` | float | nuclear repulsion |
| `C` | (nao, nmo) | MO coefficients |
| `S`, `Hcore_ao` | (nao, nao) | AO overlap and core Hamiltonian |
| `eri_active` | (nact,)×4 | chemist's `(tu|vw)`, MO basis, active window only |
| `fock_source` | str | `gaussian`, `pyscf_rebuilt` or `none` |
| `provenance` | str | JSON |

Optional, and absent rather than null when unknown: `F_alpha_ao`, `F_beta_ao`,
`orbital_energies`, `orbital_energies_beta`, `escf`, `atom_charges`.

Do not do index arithmetic on the window. `bundle.active` is the 0-based slice
of active orbitals and `bundle.core` the frozen-core slice; `active_start` and
`active_stop` are the 0-based half-open pair. Also there: `nocc_active_alpha`,
`nocc_active_beta`, `nelec_active` and `ms2` for the FCIDUMP header, plus
`nfrozen_virtual`, `is_ks` and `has_fock`. Every conversion between the stored
1-based window and 0-based array indices happens inside those and nowhere else,
because `C[:, bundle.active_first:bundle.active_last]` silently drops the first
active orbital and takes a virtual in its place, with every shape still correct.

Four conventions are worth knowing before writing against it.

`C` is stored **column-wise**, `C[ao, mo]`, so that $`\mathbf{C}^{\mathsf{T}} \mathbf{S} \mathbf{C} = \mathbf{I}`$.
That is PySCF's convention, and every oracle here is PySCF. The algebra above is
written in the row convention, where the same transform reads $`\mathbf{C}\, h^{\text{AO}}\, \mathbf{C}^{\mathsf{T}}`$;
`matfile.py` decides which one Gaussian handed it by testing both numerically,
rather than trusting a storage convention, and normalises to the column one.

The window is stored **exactly as the Gaussian route writes it**:
`active_first = NFIRST` and `active_last = NLAST`, 1-based and inclusive, so a
bundle can be read against the route that produced it without arithmetic.
`ncore` and `nact` are stored redundantly, and validation requires all three to
agree, so a window that does not mean what the route meant cannot pass quietly.

Fock matrices are stored in the **AO** basis, so that a rotation of the active
space never has to touch them. `hamiltonian.py` does the
$`\mathbf{C}^{\mathsf{T}} F \mathbf{C}`$ transform.

A **KS bundle never carries a stored Fock matrix at all.** `matfile.py` refuses
to read the KS matrix into `fock_ao_*`, so a KS bundle arrives with
`fock_source = "none"` and the rebuilt-Fock path is the only way to use it.
Validation rejects a KS reference together with `fock_source = "gaussian"` as a
second line of defence.

`provenance` carries the Gaussian filename, the route line, basis and method
where known, the 1-based window as Gaussian saw it, the Git commit, the schema
version, and whether the Fock matrix came from Gaussian or was rebuilt through
PySCF.

## Using it

```bash
g16dump extract JOB.mat --fch JOB.fch --reference ROHF --window 6 40 --out JOB.npz
g16dump validate JOB.npz --hamiltonian     # builds h', checks E_ref against E_scf
g16dump dump JOB.npz --out FCIDUMP --dice-nocc
g16dump rotate JOB.npz --random 1 --out rotated.npz
```

`--reference` is required and is never inferred. `--window` may be omitted, in
which case the partition the `.mat` reports is used; either way it is
cross-checked against the dimension of the two-electron block that is actually
present, and a disagreement stops the run.

Always run `validate --hamiltonian` before dumping. It is the step that catches
a Roothaan operator or a Kohn-Sham matrix, and it costs a second.

The FCIDUMP carries only the format. Provenance is written beside it as
`FCIDUMP.provenance.json`, because FCIDUMP has no comment syntax and every way
of smuggling one in breaks some reader: above the namelist, a `/` in a path or
a date can truncate the header; below the last record, a parser that reads every
remaining line as an integral fails on it.

`--threshold` omits small two-electron integrals. It defaults to `0.0` and
should usually stay there: the damage is neither proportional to the threshold
nor smooth. For the CH2 fixture, `1e-2` leaves `E_ref` untouched to 1e-14 while
`1e-1` costs 0.2 Ha, so a value that looks harmless on one system says nothing
about the next.

`rotate` mixes the active orbitals among themselves, which cannot change any
energy — making it the cheapest real test of a result. Dump both, run the
solver twice, and the correlated energies must agree. A rotation that mixes
occupied with virtual orbitals is refused, because that replaces the reference
determinant rather than re-expressing it; see `g16dump/rotate.py`.

[`examples/`](examples/) has a complete Dice `input.dat` and a runnable Block2
DMRG script.

## KS warning

**For Kohn–Sham orbitals the stored KS matrix contains exchange–correlation and
must never be used as $`f^{\sigma}`$.** The active-space Hamiltonian is always the *HF*
Hamiltonian evaluated in the KS orbitals. A KS bundle fed to the stored-Fock
path raises; use the rebuilt-Fock path. This warning is repeated in the CLI.

## Not in v1

- **UHF references** (different $`\alpha`$ and $`\beta`$ orbitals). A spin-restricted FCIDUMP
  cannot represent them. The workaround is to generate UHF natural orbitals
  (UNOs) in Gaussian and send those through the normal path.
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
              write.py     FCIDUMP writer, symmetry-unique and vectorized
              cli.py       extract / dump / validate / rotate
gaussian/     route templates + how to run them
legacy/       the original scripts, untouched, for reference and regression
scripts/      inspect_mat.py, the matrix-element probe
examples/     Dice input.dat and a Block2 DMRG script
tests/        fixtures as committed .npz, no Gaussian and no gauopen needed
              make_fixtures.py regenerates them (needs pyscf; CI can do it)
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
| M1 | test systems + two independent oracles | six `.npz` fixtures committed (RHF, two ROHF, a rotated ROHF, a Roothaan one to be rejected, and a real B3LYP set); `h'` and `E_core` checked against a full-transform route |
| M2 | `bundle.py`, `hamiltonian.py` + gates | done, with the schema, electron-count, window, Hermiticity, orthonormality, ERI-symmetry and α/β gates |
| M3 | writer and solver interface | done; read back by an independent parser and by PySCF, reference energy reproduced to 1e-9 Ha |
| M4 | active-space rotations | done; every energy and the FCI ground state verified invariant |
| M5 | benchmarks | preserved on `claude/benchmarks-and-sweeps`, deliberately not a release blocker |
| M6 | packaging, CI, DOI | CI, packaging, LICENSE and usage examples done; `CITATION.cff` awaits attribution, DOI still open |

What M0 still gates is `extract` alone. Everything downstream of the bundle is
exercised by the committed fixtures, so the labels in `g16dump.matfile.LABELS`
are the only thing waiting on a real `.mat`. The two probe jobs and the
procedure for reading their result are in
[`gaussian/README.md`](gaussian/README.md); both of the likely outcomes -- real
`f^α`/`f^β`, or the Roothaan operator -- are already handled, which is why the
writer and the rotations did not wait on it.

Open questions blocking M0's gate are listed in
[`gaussian/README.md`](gaussian/README.md) and in the probe script's output
section.

**Development rule while M0 is open:** do not expand scope. Treat the current
implementation as the v1 candidate and let the real Gaussian smoke tests decide
what, if anything, still needs to change.

## Credits and licensing

This project is released under the [MIT License](LICENSE).

**Attribution for the original closed-shell derivation is still open.** The
scripts preserved byte-for-byte in `legacy/` predate this repository, and
nothing in its Git history records who wrote them — the import commit is not
evidence of authorship. [`CITATION.cff`](CITATION.cff) therefore carries a
marked placeholder rather than a guess, and it must be filled in before the
work is cited or a DOI is minted. Inferring a name from the cluster paths
inside those scripts would assign scientific credit on no better basis than a
guess, which is worse than admitting the gap.

The rebuilt package is MIT-licensed as above; `legacy/` is kept unmodified as
reference material and carries whatever terms its original author intended.
