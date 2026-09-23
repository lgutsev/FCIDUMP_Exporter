# Gaussian route templates

These produce the `.mat` file that `g16dump extract` reads. **The route has not
yet been run**, so it is still the M0 gate — but it was corrected against the
Gaussian 16 documentation after the first draft, and one of those corrections
was load-bearing:

> `Output=MatrixElement` **alone does not write MO two-electron integrals.**
> The separate `MO2ElectronIntegrals` option is required. Without it the job
> completes, writes a `.mat` containing the overlap, core Hamiltonian, MO
> coefficients and Fock matrix, and sets `ITran=0` with no `AA MO 2E INTEGRALS`
> block at all — which looks exactly like a label-spelling problem and is not.

The documentation says so twice: the `Output` keyword page defines
`MO2ElectronIntegrals` as "when combined with the MatrixElement option, include
two-electron integrals over MOs", and the `Window` page names the pairing
unprompted — "it is typically used with `Output=(Matrix, MO2ElectronIntegrals)`".

`Tran=Force` is the second correction. The Transformation page documents it as
"forces an AO-to-MO integral transformation to be done during a single-point SCF
calculation", which is exactly this project's case: an SCF-only job with no
post-SCF method. It costs nothing and removes the largest remaining doubt.

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

- `Output=(MatrixElement,MO2ElectronIntegrals)` — writes the matrix-element
  file **and** puts the MO two-electron integrals in it. Both halves are
  required; see the note at the top. The filename is the last section of the
  input (after a blank line following the geometry). `MatrixElement` gives the
  Fortran-unformatted form, which is what gauopen's `QCMatEl` reads — not
  `RawMatrixElement`.
- `Tran=(Full,Force)` — `Full` selects the complete `(pq|rs)` set over the
  window rather than one of the partial sets aimed at MP/CC (`IJAB`, `IABC`,
  ...), and `Force` makes the transformation happen during a single-point SCF
  that would otherwise not run one. The result shows up as `ITran=5` in the
  `.mat` header; `ITran=4` means a partial transform and `ITran=0` means none.
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

1. No `... MO 2E INTEGRALS` block and `ITran=0` → the route is missing
   `MO2ElectronIntegrals`. Check it is `Output=(MatrixElement,MO2ElectronIntegrals)`.
   This is the single most likely failure and it is not a label problem.
2. `Window=` rejected → it is a **standalone route keyword**, not an option to
   `Tran`. `Tran=(Full,Window=(N,M))` is not valid syntax and will be rejected;
   the Transformation page lists no `Window` option. Note also that `FC`, `Full`,
   `RW` and `Window` are mutually exclusive, so a bare `Full` in the route
   collides with `Window=`; keep `Full` inside the `Tran=(...)` parentheses.
3. `Window=(NFIRST,NLAST)` is 1-based and inclusive at both ends. `0` means
   "first or last orbital", so `Window=(2,0)` is the safer spelling whenever
   `NLAST` is just the top of the space — it cannot go off the end if any basis
   function is dropped for linear dependence and `NBsUse < NBasis`. A negative
   second index freezes that many virtuals from the top: `Window=(2,-3)` is the
   same window as `Window=(2,10)` on a 13-orbital space.
4. Still no MO integrals with `MO2ElectronIntegrals` present → try, in order:
   `Tran=(Full,Force)` (already in these templates); then adding a post-SCF
   method such as `ROMP2` so the transformation is invoked by the method itself;
   then `Output=(RawMatrixElement,MO2ElectronIntegrals)` in case the installed
   gauopen expects the raw form rather than the Fortran-unformatted one.
5. The `.mat` is written but empty or unreadable → check the filename section at
   the end of the input is present and followed by a blank line, and check the
   `JOB STATUS` scalar, which `inspect_mat.py` now probes: QCMatEl sets it to
   1.0 only when the job completed.

Whatever ends up working, paste the exact working route back into these
templates and say so in the commit message.

## Probing what a `.mat` actually contains

```bash
python3 ../scripts/inspect_mat.py NAME.mat --json NAME.probe.json
```

Run this on the cluster (it is the only place gauopen lives) and keep the JSON.
It assumes no label names; it reports what is there.

## The M0 probe jobs

Eight jobs in the default set, all tiny -- the largest basis is 13 functions --
so the whole set runs in seconds and the probe output can be read by eye. Two
more, on porphine at real scale, are opt-in and described further down. Run the
default set with:

```bash
cd gaussian
GAUOPEN_PATH=/path/to/gauopen ./run_probes.sh
```

That runs each job, `formchk`s it, runs `inspect_mat.py` where gauopen is
available, records the environment and tars the lot into one archive. It keeps
going after a failure, so one broken route does not cost the whole set. **Send
back the archive; nothing else is needed.**

| Job | What it alone can answer |
|---|---|
| `probe_h2_smoke` | **Run first.** Two basis functions, under a second. Is `ITran` 5? Is there an `ALPHA FOCK MATRIX` record at all? If this fails, nothing below is worth reading. |
| `probe_ch2_uhf` | **Run second.** The only job that writes BETA blocks, so it is the reference for what a beta record looks like when one exists. Also: is UHF (out of scope) detectable from the `.mat` alone? |
| `probe_h2o_rhf` | Does the route work for a closed shell, and does `Window=` produce a 6-orbital block? |
| `probe_h2o_nowindow` | The control. Identical but with no `Window=`, so a 7-orbital block here against 6 above is the only direct proof the keyword restricts anything. |
| `probe_ch2_rohf` | The Fock question, open shell. |
| `probe_nh_rohf` | A second open shell, whose singly occupied orbitals are a degenerate pi pair. |
| `probe_h2o_frozen_virtual` | The only job with frozen **virtuals** (`NFV=3`). No fixture and no other probe has ever exercised that path. |
| `probe_h2o_rks` | Does a DFT job write a Fock-like matrix, and under what label? |

### Expected dimensions

A mismatch here is the first thing to notice, so check it before anything else.
`inspect_mat.py` now prints the window it derives from the header
(`NBsUse - NFC - NFV`) and the `n**4` it implies.

| Job | basis fns | electrons | NFC | NFV | active MOs | nact | `eri_act` elements |
|---|---|---|---|---|---|---|---|
| `probe_h2_smoke` | 2 | 2 | 0 | 0 | 1-2 | 2 | 16 |
| `probe_h2o_rhf` | 7 | 10 | 1 | 0 | 2-7 | 6 | 1296 |
| `probe_h2o_nowindow` | 7 | 10 | 0 | 0 | 1-7 | 7 | 2401 |
| `probe_ch2_rohf` | 13 | 8 | 1 | 0 | 2-13 | 12 | 20736 |
| `probe_ch2_uhf` | 13 | 8 | 1 | 0 | 2-13 | 12 | 20736 x3 blocks |
| `probe_nh_rohf` | 11 | 8 | 1 | 0 | 2-11 | 10 | 10000 |
| `probe_h2o_frozen_virtual` | 13 | 10 | 1 | 3 | 2-10 | 9 | 6561 |
| `probe_h2o_rks` | 13 | 10 | 1 | 0 | 2-13 | 12 | 20736 |

Note the `.mat` stores the two-electron block packed as a triangle of triangles,
not as `n**4` loose numbers, so `inspect_mat.py` will report a packing candidate
with `n = nact(nact+1)/2` -- 21 for H2O/STO-3G, 78 for CH2. That is the expected
shape, not a failure.

### Tier 2: porphine, the system this is actually for

Everything above is a toy. Ni(II) porphine is 37 atoms and about 260 basis
functions, and it is the first job here that resembles what the method was built
for. It is opt-in because it takes minutes to hours rather than seconds:

```bash
PORPHINE=1 ./run_probes.sh
```

Run it only after the cheap set has confirmed `ITran=5`. There is no sense
spending a porphine SCF to discover the route was wrong.

| Job | Metal | Reference | Window | Active space |
|---|---|---|---|---|
| `probe_fe_porphine_singlet` | Fe, 186 e- | RHF | `(83,103)` | CAS(22,21), MS2=0 |
| `probe_fe_porphine_triplet` | Fe, 186 e- | ROHF | `(83,103)` | CAS(22,21), MS2=2 |
| `probe_ni_porphine_singlet` | Ni, 188 e- | RHF | `(84,104)` | CAS(22,21), MS2=0 |
| `probe_ni_porphine_triplet` | Ni, 188 e- | ROHF | `(84,104)` | CAS(22,21), MS2=2 |

The Fe pair is the one to run if you only run one: it is the same element as the
legacy dumps. The windows differ by one because Fe has two electrons fewer than
Ni, which moves the HOMO down by one orbital -- a good reminder that the window
is a property of the electron count, not of the file.

**Why these windows.** Ni porphine has 188 electrons, so 94 occupied orbitals
and `(84,104)` is 11 occupied plus 10 virtual. Fe porphine has 186, so 93
occupied and `(83,103)`. Both are exactly the CAS(22,21) of the legacy dumps. The indices depend only on the electron count, not on the
basis, so they stay correct if you change basis set. Expect `NFC=83` and
`NFV = NBsUse - 104`, which is around 156: this is the only job where the frozen
core and the frozen virtual space are both large, and where getting the window
wrong would be expensive rather than merely incorrect.

**What it proves.** A full transformation over ~260 orbitals is roughly 5x10^8
unique integrals. Windowed to 21 it is about 24000. If this job finishes in
minutes, the "seconds where a full-space dump takes days" claim in the top-level
README stops being an assertion and becomes a measurement. Capture the wall time.

**The comparison that matters.** Dump both spin states and compare their
reference energies. The legacy dumps put the ROHF triplet **0.673 Ha below** the
RHF singlet, which is impossible; `tests/test_legacy_regression.py` measures
that from the committed files. A correct pipeline must put the triplet above the
singlet. That is the fix, demonstrated on a real system instead of on CH2.

If the ROHF triplet is slow to converge, run the singlet first and add
`Guess=Read` -- the `.chk` is right there.

### The Fe-porphyrin regression is still blocked, and you may be able to unblock it

`legacy/FePorph_1_Window.dat` and `FePorph_3_Window.dat` are the original
dumps, but **the geometry that produced them is not in this repository** --
there is no `.gjf`, no `.log` and no coordinates anywhere in `legacy/`, and
`legacy/FCIDUMP_Launch.zip` holds only the two writer scripts.
`legacy/Template.py` has an empty geometry placeholder that a job script
substituted into.

So `fe_porphine.xyz` is an *idealised D4h model* (Fe-N 2.00 A), built from the
same verified construction as the Ni one. It is the right element, the right
size and the right active space -- but it is a model, not a reproduction, and it
will **not** return the legacy -2201.53 Ha.

If you still have the Fe-porphyrin input, output or checkpoint, that would turn
a stand-in into a direct regression: same molecule, same CAS(22,21), new
pipeline against the shipped numbers. The basis matters too -- `legacy/Template.py`
names `anoroostz`, so the original was not a Pople basis.

### Expected energies

Every probe geometry, basis and window matches a committed fixture in
`tests/data/`, so each result is directly comparable to a number PySCF produced
independently. Agreement to about 1e-6 Ha is right; the residual is SCF
convergence, not method. No basis here has d functions, so there is no
Cartesian/spherical ambiguity.

| Job | `E_nuc` | `E(SCF)` | `E_core` | fixture |
|---|---|---|---|---|
| `probe_h2o_rhf` | 9.1895337629 | -74.9630231385 | -51.4711696318 | `h2o_rhf` |
| `probe_ch2_rohf` | 5.3800248517 | -38.8684854585 | -28.6809764260 | `ch2_rohf` |
| `probe_nh_rohf` | 3.5447277287 | -54.9382261545 | -- | `nh_rohf` |
| `probe_h2o_frozen_virtual` | 9.1895337629 | -75.9839744727 | -52.1215325375 | `h2o_frozen_virtual` |
| `probe_h2o_rks` | 9.1895337629 | -76.3849509041 (DFT) | -- | `h2o_rks` |

For `probe_h2o_rks`, note that -76.3849509041 is the **DFT** total energy. The
HF energy evaluated in those same KS orbitals is -75.9813320027, a different
number by 0.40 Ha, and it is the second one the rebuilt-Fock path reproduces.
Comparing `E_ref` against the DFT energy and calling the 0.4 Ha gap a bug is the
mistake this fixture exists to prevent.

## Step 2: getting the verdict on the Fock matrices

The probe says which labels exist. It does not say whether the matrices behind
them are the ones the algebra needs, and that is the question M0 actually has to
answer: **does Gaussian store `F^α`/`F^β`, or only its Roothaan effective ROHF
operator?**

That question is decided numerically, not by reading label names, and the
package already decides it. Run the pipeline:

```bash
g16dump extract probe_h2o_rhf.mat  --fch probe_h2o_rhf.fch  \
        --reference RHF  --window 2 7  --out probe_h2o_rhf.npz
g16dump validate probe_h2o_rhf.npz  --hamiltonian

g16dump extract probe_ch2_rohf.mat --fch probe_ch2_rohf.fch \
        --reference ROHF --window 2 13 --out probe_ch2_rohf.npz
g16dump validate probe_ch2_rohf.npz --hamiltonian
```

`validate --hamiltonian` builds `h'` from both spins and compares them. There
are exactly three outcomes, and all three are useful:

| What you see | What it means | What to do |
|---|---|---|
| `h' alpha/beta agreement` around 1e-15, and `E_scf ... (agrees)` | Gaussian stored genuine `F^α`/`F^β`. The stored-Fock path works. | Nothing. Record it and update the banner at the top of this file. |
| `SpinConsistencyError`, deviation ~0.1–1 Ha, mentioning the Roothaan operator | Gaussian stored its effective ROHF operator. **Expected, and not a bug.** | Use the rebuilt-Fock path (`hamiltonian.with_rebuilt_fock`), which needs the `.fch`. |
| `open-shell reference ... but no beta Fock matrix is stored` | **The outcome the documentation predicts for ROHF.** `BETA FOCK MATRIX` is documented as written "for unrestricted SCF calculations", and ROHF is a *restricted* job, so it should write `ALPHA FOCK MATRIX` and no beta partner. The one stored matrix is then almost certainly the Roothaan effective operator. | Use the rebuilt-Fock path. This is not a missing label and not a bug. |
| `no Fock matrix was found in the .mat` | The labels are spelled differently, or Gaussian does not write them at all. | Send the `.probe.json`; `matfile.LABELS` needs the real spelling. |

Only the last outcome requires a code change. The first three are all handled,
which is why the writer and the rotations did not wait on this gate.

Two predictions worth recording before the run, so that confirming or refuting
them is itself informative:

- **A restricted job writes only `AA MO 2E INTEGRALS`.** The three-block form
  (`AA`, `BA`, `BB`) is documented as unrestricted-only. Since
  `legacy/FCIDUMP_Write_MOe_3.py` reads all three, that script can only ever
  have been run against a **UHF** job, never an ROHF one -- so its spin algebra
  is evidence about the unrestricted case and does not transfer to ROHF.
  `probe_ch2_uhf` is the job that should show three blocks.
- **An RKS `.mat` is label-identical to an RHF one.** There is no Kohn-Sham
  label in the format; the KS matrix is written under `ALPHA FOCK MATRIX`. So
  `probe_h2o_rks` will *look* like a success, and the KS refusal cannot key off
  anything in the `.mat`. It has to come from the method string, which is why
  `--reference` is required and never inferred.

### Expected numbers

Both probe geometries and both windows are identical to the committed fixtures
in `tests/data/`, so the Gaussian run can be checked directly against numbers
PySCF produced independently. Agreement to about 1e-6 Ha is the right
expectation — the residual is SCF convergence and integral thresholds, not
method. Neither basis has d functions, so there is no Cartesian/spherical
ambiguity to account for.

| Quantity | `probe_h2o_rhf` | `probe_ch2_rohf` |
|---|---|---|
| `E_nuc` | 9.1895337629 | 5.3800248517 |
| `E(SCF)` | −74.9630231385 | −38.8684854585 |
| `E_core` (from `validate`) | −51.4711696318 | −28.6809764260 |
| `E_ref` (must equal `E(SCF)`) | −74.9630231385 | −38.8684854585 |
| `nao` / `nmo` | 7 / 7 | 13 / 13 |
| `ncore` / `nact` | 1 / 6 | 1 / 12 |

A disagreement in `E_nuc` means the geometry was edited; in `E(SCF)` with
`E_nuc` correct, the basis or convergence differs; in `E_core` with both of
those right, the window did not apply as intended.

### Then the FCIDUMP

Once `validate --hamiltonian` reports agreement, the rest is already tested:

```bash
g16dump dump probe_ch2_rohf.npz --out probe_ch2_rohf.FCIDUMP --dice-nocc
```
