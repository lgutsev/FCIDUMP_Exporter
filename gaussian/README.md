# Gaussian route templates

These produce the `.mat` file that `g16dump extract` reads. **The route section is
currently UNVERIFIED** — it was proposed from the Gaussian 16 documentation for
`Output=MatrixElement` and `Window=`, not copied from a job that has run. Confirm
it on the cluster before trusting anything downstream (this is the M0 gate).

## Placeholders

| Placeholder | Meaning |
|---|---|
| `NAME` | job basename; also names the `.chk` and the `.mat` |
| `BASIS` | e.g. `STO-3G`, `6-31G`, `6-31G*`, `cc-pVDZ` |
| `CHARGE`, `MULTIPLICITY` | charge/spin line |
| `GEOMETRY_HERE` | Cartesian or Z-matrix geometry |
| `NFIRST`, `NLAST` | **1-based** index of the first and last active MO |

`NFIRST`/`NLAST` define the active window. Everything below `NFIRST` becomes
frozen core (`me.nfc`), everything above `NLAST` frozen virtual (`me.nfv`). The
partition the `.mat` reports must match the one the PySCF oracle uses in M1, so
record the numbers you pass here.

## Why each keyword is there

- `Output=MatrixElement` — writes the matrix-element file. The filename is the
  last section of the input (after a blank line following the geometry).
- `Tran=Full` — performs the integral transformation so MO 2e integrals are
  actually present. Without a transformation the `AA MO 2E INTEGRALS` block that
  the reader needs will be missing.
- `Window=(NFIRST,NLAST)` — restricts that transformation to the active space.
  This is the whole point of the method: the transform never touches the full MO
  space.
- `NoSymm` — required, because MOKIT's `.fch` orbital transfer (used by both
  validation routes in M1) assumes no symmetry reordering. It also keeps the
  Gaussian orbital order equal to the order in the `.mat`.
- `Int=NoBasisTransform` — keeps Gaussian from re-expressing the basis, so the
  AO quantities in the `.mat` and the `.fch` refer to the same AO set.
- `SCF=(Conver=10)` — tight convergence. The `E_ref` gate in M2 is at 1e-8 Ha,
  and a loosely converged SCF will fail it for reasons that are not our bug.

## Running

```bash
g16 NAME.gjf          # produces NAME.log, NAME.chk, NAME.mat
formchk NAME.chk NAME.fch
```

The `.fch` is required — MOKIT's `load_mol_from_fch` reads it for both M1
validation routes and for the rebuilt-Fock fallback path.

## If the route fails

Do not guess a replacement silently; report the error. Things worth trying, in
order, and the reason each is plausible:

1. `Window=(NFIRST,NLAST)` rejected → try it as an option to the transformation
   keyword rather than standalone, e.g. `Tran=(Full,Window=(NFIRST,NLAST))`.
2. Negative window indices count from the end in some Gaussian options; if
   `NLAST` is awkward to compute, `Window=(NFIRST,-NFV)` may be accepted.
3. No `... MO 2E INTEGRALS` blocks in the `.mat` → the transformation did not
   run or was not saved. Check whether the job needs a post-SCF method present
   for the transformation step to be invoked at all.
4. The `.mat` is written but empty/truncated → check that the filename section
   at the end of the input is present and followed by a blank line.

Whatever ends up working, paste the exact working route back into these
templates and drop the `UNVERIFIED` banner above.

## Probing what a `.mat` actually contains

```bash
python3 ../scripts/inspect_mat.py NAME.mat --json NAME.probe.json
```

Run this on the cluster (it is the only place gauopen lives) and keep the JSON.
It assumes no label names; it reports what is there.

## The two M0 probe jobs

`probe_h2o_rhf.gjf` and `probe_ch2_rohf.gjf` are concrete, runnable versions of
the templates — small enough that the probe output can be read by eye. They are
what the M0 gate needs: one RHF `.mat` and one ROHF `.mat`.

Expected sizes, so a mismatch in the probe output is obvious immediately:

| Job | basis fns | electrons | ncore | active MOs | nact |
|---|---|---|---|---|---|
| `probe_h2o_rhf` (H2O, STO-3G, singlet) | 7 | 10 | 1 | 2–7 | 6 |
| `probe_ch2_rohf` (CH2, 6-31G, triplet) | 13 | 8 | 1 | 2–13 | 12 |

So `eri_act` should expand to 6^4 = 1296 and 12^4 = 20736 elements respectively.
If `inspect_mat.py` reports a packing candidate with a different `n`, the window
did not apply the way we think it did — say so rather than working around it.

```bash
g16 probe_h2o_rhf.gjf  && formchk probe_h2o_rhf.chk  probe_h2o_rhf.fch
g16 probe_ch2_rohf.gjf && formchk probe_ch2_rohf.chk probe_ch2_rohf.fch

python3 ../scripts/inspect_mat.py probe_h2o_rhf.mat  --json probe_h2o_rhf.probe.json
python3 ../scripts/inspect_mat.py probe_ch2_rohf.mat --json probe_ch2_rohf.probe.json
```

Send back both `.probe.json` files (and the `.log` files if the jobs fail).
