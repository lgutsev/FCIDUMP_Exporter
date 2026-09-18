# Benchmarks and active-space sweeps

Two things live here. A **manifest suite** describing every system this workflow
is meant to handle, from molecules small enough to diagonalise exactly up to the
Ni-PAP complex. And a **sweep workflow** that runs one system in a series of
growing active spaces and reports whether the answer has stopped moving.

The second exists because of what the first is for. An active-space calculation
is a claim that the active space was big enough, and the Ni-PAP work is the
reason not to take that on trust: correlation there reaches past the nominal Ni
3d orbitals into the porphyrin framework, and the reported state ordering moved
when the active space or the basis was expanded. A single window is not evidence.

```bash
python -m benchmarks validate                     # check the committed suite
python -m benchmarks sweep  SYSTEM... --out DIR   # generate the Gaussian jobs
python -m benchmarks analyze DIR --results DIR    # read the results back
```

Numpy only. PySCF is used by one optional function and imported lazily.

## The suite

One file in `systems/` is one Gaussian job, because that is the unit the
workflow actually runs: a molecule, a spin state, a basis and a window. Entries
that differ only in spin state share a `spin_family`, which is what lets the
analysis form a splitting; a test enforces that members of a family share a
geometry and a basis, so a splitting is never a comparison of two different
systems.

| Tier | Role | Entries | What it is for |
|---|---|---|---|
| 0 | `validation` | `h2o_rhf_sto3g`, `ch2_rohf_631g`, `nh_rohf_631g`, `o2_rohf_631g`, `ch2_rohf_631g_rotated`, `h2o_rks_b3lyp_631g` | Small enough for exact diagonalisation. Mathematical correctness, nothing else. |
| 1 | `ligand_field` | `nicl4_td_*`, `nicl4_sqp_*`, `fecl4_td_*` | Chemistry that the method has to get right: d8 singlet/triplet ordering flipping between tetrahedral and square-planar, and a non-nickel d6 case. |
| 2 | `macrocycle` | `ni_porphine_singlet`, `ni_porphine_triplet` | The smallest real metalloporphyrin. Where the active-space sweep starts to matter. |
| 3 | `regression` | `feporph_legacy_*` | The shipped legacy dumps, as evidence of the bug rather than as a target. |
| 4 | `stress` | `ni_pap_full` | The molecule the original work was about. Deliberately last. |

Four of the tier-0 entries are worth calling out. `o2_rohf_631g` is the first
window that closes below the last orbital, so it exercises frozen virtuals as
well as frozen core. `ch2_rohf_631g_rotated` is the same Gaussian job as
`ch2_rohf_631g` with a fixed random rotation applied inside the active space, so
the MO Fock matrix is not diagonal -- the case the legacy `diag(orbital_energies)`
expression cannot reproduce, and whose spectrum must match the unrotated one.
`h2o_rks_b3lyp_631g` is Kohn-Sham orbitals through the rebuilt-Fock path.
`nh_rohf_631g` is a second open-shell reference with a different bonding pattern
from CH2, so the open-shell algebra is not tested on one shape of molecule.

Two entries are `blocked`, and say so rather than carrying invented inputs. The
Fe-porphyrin geometry is not in this repository -- `legacy/` has the FCIDUMP
files and the Dice input but no `.gjf`, no `.log` and no coordinates. Neither is
the Ni-PAP structure, and its charge and spin state are not recorded anywhere
that can be checked: `legacy/Template.py` carries `spin = 2` and a Ni/N basis
assignment with the geometry block empty. Both entries record what *is* known,
name what is missing in `blocked_on`, and are refused by the sweep generator
until it arrives.

### Geometries

Small molecules carry their geometry inline. `ni_porphine.xyz` is built by
`geometries/build_ni_porphine.py`, which constructs the 37-atom D4h skeleton in
closed form from standard metalloporphine bond lengths and the Ca-N-Ca angle --
no optimiser, so the committed file is exactly reproducible, and the script
reports its own bond lengths and planarity:

```bash
python3 geometries/build_ni_porphine.py --check-only
```

It is an **idealised model geometry, not an optimised or experimental one**, and
every manifest that uses it says so. The same is true of the `MCl4` models: their
value is the sign of a splitting, not its magnitude.

## The manifest format

`manifest.py` is the whole definition. A minimal entry:

```json
{
  "schema_version": "1.0",
  "id": "h2o_rhf_sto3g",
  "status": "ready",
  "role": "validation",
  "tier": 0,
  "spin_family": "h2o_sto3g",
  "molecule": {
    "charge": 0,
    "multiplicity": 1,
    "geometry": {"units": "angstrom", "xyz": "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692"}
  },
  "reference": {"type": "RHF", "basis": "STO-3G", "expected_nbasis": 7, "fock_source": "gaussian", "scf": {"conver": 10}},
  "active_space": {"window": {"nfirst_1based": 2, "nlast_1based": 7}, "frozen_core": 1, "n_orbitals": 6, "n_electrons": 8, "ms2": 0},
  "solver": {"kind": "pyscf_fci", "parameters": {}, "convergence": {"conv_tol": 1e-10}},
  "provenance": {"added": "2026-09-18", "geometry_source": "gaussian/probe_h2o_rhf.gjf"}
}
```

JSON is the on-disk format. YAML loads too when PyYAML happens to be installed,
but the committed suite is JSON so the core dependency stays numpy-only.

### Three conventions

**Windows are the 1-based inclusive numbers you write in a Gaussian route**,
which is why the fields are `nfirst_1based` and `nlast_1based`. The `.npz`
interchange bundle stores a 0-based half-open window instead, and
`manifest.bundle_window()` is the only place the two meet:
`act_start = NFIRST - 1`, `act_stop = NLAST`.

**Electron counts are never taken on trust.** They are recomputed from the
nuclear charges and the molecular charge; `frozen_core`, `n_orbitals`,
`n_electrons` and `ms2` are declared so that the validator has something to
disagree with, not so that anything reads them.

**A window is either explicit or a policy.** For anything the size of a
metalloporphyrin the absolute MO indices are not known until a preliminary SCF
has run, so those entries carry a `window_policy` naming a reference orbital and
how many orbitals to take below and above it, plus the `sweep` steps to expand
into:

```json
"window_policy": {
  "reference_orbital": "homo",
  "n_below": 9, "n_above": 6,
  "sweep": [
    {"n_below": 4,  "n_above": 1,  "label": "Ni 3d only"},
    {"n_below": 9,  "n_above": 6,  "label": "3d plus the porphine frontier pi"},
    {"n_below": 13, "n_above": 10, "label": "3d plus an extended pi set"}
  ]
}
```

`reference_orbital` is `"homo"`, `"lumo"`, or a 1-based index. A policy
resolves only when the HOMO index is supplied, and refuses to guess.

### What the validator rejects

It collects every problem rather than raising on the first, because a manifest
is usually edited by hand and one error per run is a slow way to fix six.

- a `frozen_core`, `n_orbitals`, `n_electrons` or `ms2` that disagrees with what
  the window and the geometry imply;
- an electron count and a multiplicity whose parities are incompatible;
- a window running past the end of the basis, or a reference determinant that
  does not fit inside the window (more active alpha electrons than orbitals, or a
  `NFIRST` that freezes orbitals the molecule does not have electrons for);
- unknown element symbols, and nuclei closer than 0.5 A;
- a KS reference routed to the stored Fock matrix -- the stored Kohn-Sham matrix
  carries exchange-correlation and is not the HF Fock operator, so `RKS`/`ROKS`
  must use `fock_source: "pyscf_rebuilt"`;
- a functional on a Hartree-Fock reference, or a missing one on a KS reference;
- a `pyscf_fci` solver on an active space too large to diagonalise;
- `status: "blocked"` with nothing in `blocked_on`;
- **unknown keys anywhere**, because a silently ignored typo is a wrong run.

## Sweeps

```bash
python -m benchmarks sweep ni_porphine_singlet ni_porphine_triplet --homo 94 --out sweeps/ni_porphine
```

That writes, per system and window: a `.gjf`, a `.pbs` job script, and a derived
manifest under `manifests/`. Plus one `sweep.json` recording every active space
it generated, which is what the analysis reads back. Windows can also be given
outright with `--windows 85-100,81-104`.

Each derived job is itself a full benchmark manifest -- explicit window, geometry
inlined -- so it goes through exactly the same validator, and a sweep directory
can be copied to a cluster on its own. The Gaussian input comes from the
committed templates in `gaussian/`, so when the route section is finally verified
on the cluster, fixing it there fixes every sweep.

Two things are refused rather than warned about:

- **Windows that do not nest.** Comparing a window to the next larger one is only
  a convergence statement if the smaller is contained in the larger.
- **Two spin states in different windows.** A splitting between two different
  active spaces is not a splitting. One `--homo` and one set of windows is
  applied to every state in the family.

A third is refused at render time: a route template that has lost its
`Window=(NFIRST,NLAST)`. That job would run happily, transform the full MO space,
and return the wrong answer days later.

## Reading results back

```bash
python -m benchmarks analyze sweeps/ni_porphine --results results/ --csv table.csv
```

The analysis reads **result records**: one small JSON per finished job. It does
not parse solver output -- the solver adapters own that, and emit these. Keeping
the boundary there means the analysis runs on a laptop against a directory of
JSON copied off the cluster.

```json
{
  "schema_version": "1.0",
  "kind": "active_space_result",
  "job": "ni_porphine_singlet_w85_100",
  "system": "ni_porphine_singlet",
  "spin_family": "ni_porphine",
  "multiplicity": 1,
  "window": {"nfirst_1based": 85, "nlast_1based": 100},
  "solver": "block2",
  "energy_hartree": -2244.41000000,
  "energy_uncertainty_hartree": 2e-5,
  "converged": true,
  "natural_occupations": [1.998, 1.981, 1.620, 0.380, 0.019],
  "runtime_seconds": 5881.3,
  "peak_memory_gb": 4.8
}
```

`schema_version`, `job`, `solver` and `energy_hartree` are required; everything
else is optional and **absent rather than null** when the solver did not report
it, matching the `.npz` bundle's convention. `analyze.validate_result()` is the
definition.

What comes out:

```
   window  orb  elec  mult       energy / Ha        +/-  d(next) / mHa  nfrac      Nu   time / s  label
-------------------------------------------------------------------------------------------------------
  90-95     6    10     1    -2244.10000000   2.00e-05      +310.0000      4   2.366       1006  Ni 3d only
  85-100   16    20     1    -2244.41000000   2.00e-05      +130.0000      4   2.456       5881  3d plus the porphine frontier pi
  ...

   window  orb   states   splitting / kcal  shift to next  ground state
-----------------------------------------------------------------------
  90-95     6    1/3               -6.275        +15.060  multiplicity 3  <-- ordering changes at the next window
  85-100   16    1/3                8.785         +1.883  multiplicity 1
```

Per active space: orbitals and electrons as actually run, the solver energy with
its uncertainty, the deviation from the next larger window, natural-orbital
occupation diagnostics, and runtime and memory when recorded. Per window that has
two spin states: the splitting, how far it moved from the previous window, and
whether the ground state changed -- which is the Ni-PAP effect, and the reason
the splitting series matters more than any single number in it.

A job whose result never arrived is listed, never dropped. A sweep with a hole in
it is exactly the case where a quiet omission reads as a converged series.

`n_unpaired` is Head-Gordon's `sum min(n_i, 2 - n_i)` over natural occupations;
`n_unpaired_nl` is his nonlinear form, which suppresses the weakly correlated
tail. `n_fractional` is the number to watch across a sweep: if the largest
occupation in the virtual set is still well above the threshold at the widest
window, the active space has not caught everything that is correlated.

## What this expects from `g16dump`

Nothing here imports the package, on purpose -- the suite has to load on a
machine with no Gaussian and no solver. Two interfaces are assumed:

1. **The `.npz` bundle keys**, read by `analyze.check_bundle_against_manifest()`:
   `act_start`, `act_stop`, `ncore`, `nact`, `nelec`, `nalpha`, `nbeta`,
   `charge`, `multiplicity`, `reference`, `fock_source`, `eri_act`. It takes
   anything indexable by those names -- the object `numpy.load` returns, or a
   plain dict. Run it before believing a sweep: a window that came back different
   from the one requested is invisible in the energies.
2. **`g16dump rotate`** applies the `reference.orbital_transform` that
   `ch2_rohf_631g_rotated` declares (`{"kind": "random_orthogonal", "seed": N}`),
   to the bundle rather than in Gaussian, so that entry shares its `.mat` with
   the unrotated one.

The solver adapters are expected to emit the result record above. Nothing in
this directory writes one.

## Running one benchmark end to end

```bash
python -m benchmarks validate h2o_rhf_sto3g
python -m benchmarks sweep h2o_rhf_sto3g --out run/ --no-job-script

g16 run/h2o_rhf_sto3g_w2_7.gjf
formchk h2o_rhf_sto3g_w2_7.chk h2o_rhf_sto3g_w2_7.fch
g16dump extract h2o_rhf_sto3g_w2_7.mat --fch h2o_rhf_sto3g_w2_7.fch --out bundle.npz
g16dump dump bundle.npz --out FCIDUMP
```

Then, for a window small enough to diagonalise exactly, skip the solver
adapters entirely:

```python
from benchmarks.analyze import fci_energy_from_fcidump
print(fci_energy_from_fcidump("FCIDUMP"))
```

## Tests

`tests/test_bench_manifest.py`, `tests/test_bench_sweep.py` and
`tests/test_bench_analyze.py`. They run on numpy alone; the one test that needs
PySCF carries `pytest.mark.pyscf`. Nothing needs Gaussian, gauopen or a solver.
