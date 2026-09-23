# Block2 (DMRG)

[Block2](https://block2.readthedocs.io/) reads an FCIDUMP and runs DMRG on it.
This directory holds one worked example and a script that writes the input file
for you.

## Getting an FCIDUMP

```bash
g16dump dump JOB.npz --out FCIDUMP
```

That writes a strictly standard FCIDUMP, plus a `FCIDUMP.provenance.json`
sidecar recording which job it came from, whether the Fock matrices were
Gaussian's or rebuilt, and what the threshold discarded. The sidecar is separate
because FCIDUMP has no comment syntax its readers agree on; Block2 never sees
it, and deleting it costs you only the provenance.

## Running it

`dmrg.conf` in this directory is the CH2 triplet example. Put it beside the
FCIDUMP and run:

```bash
block2main dmrg.conf > dmrg.out
grep "DMRG Energy" dmrg.out
```

To generate the same file for your own dump:

```bash
python3 make_block2_input.py FCIDUMP --out dmrg.conf
```

## What each line means

```
sym c1              point group. g16dump writes ORBSYM all 1, i.e. C1
orbitals FCIDUMP    path to the integral file, relative to where Block2 runs
nelec 6             NELEC from the FCIDUMP header: active electrons
spin 2              MS2 from the FCIDUMP header, which Block2 calls 2S
irrep 1             the target irrep; the only one there is in C1
hf_occ integral     ignored by Block2, required by StackBlock
schedule default    Block2's built-in sweep schedule
maxM 500            maximum bond dimension
maxiter 30          maximum number of sweeps
sweep_tol 1E-10     energy convergence tolerance
```

`sym`, `nelec`, `spin` and `maxM` are the only lines that change between jobs,
and the first three come straight out of the FCIDUMP header.

`maxM` is a scientific choice, not a formatting detail. 500 is far more than
this twelve-orbital example needs; for a real active space, converge the energy
against it rather than trusting a number copied from here.

## Checking the result

The energy Block2 prints is the ground state **in the active space**, so it
must come out below the reference determinant energy that

```bash
g16dump validate JOB.npz --hamiltonian
```

prints as `E_ref` for the same bundle. That comparison is worth making every
time, because DMRG converging to the wrong state does not look like an error:
the sweeps converge, the discarded weight goes to `1e-14`, and the energy is
simply too high. It happened while this example was being written — a
hand-written schedule with an aggressive bond dimension ramp settled at
−38.6989055 Ha, 0.25 Ha above the true active-space ground state and 0.17 Ha
above the reference determinant itself. `schedule default` found −38.9500811 Ha
on the same input. That is why this example uses the default schedule.

## How this example was checked

Block2 3.x (installed from PyPI) was run on the FCIDUMP that
`g16dump dump tests/data/ch2_rohf.npz` produces, using the `dmrg.conf`
committed here:

| | Energy (Ha) |
|---|---|
| Block2, `dmrg.conf` as committed | −38.950081068018 |
| PySCF `fci.direct_spin1` on the same `h'` and active ERIs | −38.950081068018 |
| reference determinant, `E_ref` | −38.868485458473 |

The first two agree to 4e-13, and both lie below `E_ref`, as they must. The
closed-shell fixture was run the same way: `g16dump dump tests/data/h2o_rhf.npz`
gives six orbitals and eight electrons, on which Block2 finds −75.012500153955
against −75.012500153957 from FCI, below its own `E_ref` of −74.963023138463.

`make_block2_input.py` was then run on both dumps. Its output reproduced the
committed `dmrg.conf` for CH2, and running the generated file through Block2
gave the same energy. The header parser was also run against the two FCIDUMPs in
`legacy/`, which use a different header spacing from the one this project
writes, and against truncated, non-FCIDUMP, impossible-multiplicity and
symmetry-carrying headers to check that each is refused with a message that says
what is wrong.

This checks the example, not the exporter. The scientific gates on `h'` and
`E_core` are the oracle tests, and they are separate work.
