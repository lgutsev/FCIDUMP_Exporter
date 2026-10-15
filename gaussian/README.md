# Gaussian route templates

These produce the `.mat` file that `g16dump extract` reads. The route below
**has been run**: the ten cheap probes ran on LONI QB4 under G16 C.01 (job
1072685, 2026-10-06, all Normal termination), and their `.mat`/`.fch` pairs are
committed gzipped in `tests/data/g16_c01_probes/`. What they showed is in the
next section and is asserted by `tests/test_real_mat.py`.

## M0 outcome

Read with g16dump's own numpy reader (`g16dump/matfile.py`; gauopen is not
installed on QB4 and is no longer needed anywhere).

| Question | Answer | Evidence |
|---|---|---|
| Is `ITran` 5? | **Yes**, in all ten files. | Header record 8 holds `NFC, NFV, ITran, IDum9`. |
| Is a Fock matrix written for RHF? | **Yes**, `ALPHA FOCK MATRIX`. | Diagonal in the MOs to 1e-14; matches PySCF's AO Fock to 2e-7; `E_ref` equals Gaussian's SCF energy to 2.5e-12 (H2O) and 5e-11 (frozen virtuals); `E_core` matches the PySCF numbers below to 3e-8. |
| Does `Window=` restrict the transformation? | **Yes.** | `NFV` = 3, 11 and 7 for `Window=(2,10)` of 13, `(2,13)` of 24, `(2,12)` of 19, and the 2e block is exactly `nact^4` each time. **But `probe_h2o_nowindow` proves nothing:** without `Window=` Gaussian applies its default frozen core, so it came back identical to `probe_h2o_rhf` (NFC=1, 6 orbitals), not with 7. |
| ROHF: real `F^α`/`F^β` or the Roothaan operator? | **Real `F^α` and `F^β`**, both written. | They match PySCF's `F^σ = h + J[Pα+Pβ] − K[Pσ]` to 3e-7 of scale, differ from each other by 0.66, and satisfy the ROHF stationarity conditions (`Fα+Fβ` docc–virt, `Fβ` docc–socc, `Fα` socc–virt) to 1e-11. |
| ROHF: are the integrals over the ROHF orbitals? | **No. This blocks ROHF.** | `TRANS MO COEFFICIENTS` (the orbitals the transformation used) are per-spin **semicanonical**: each diagonalizes its own `F^σ` within the whole occupied and the whole virtual space, frozen orbitals included. `AA`, `BA`, `BB` are over three different orbital sets, none of them the ROHF one, and their active span has mixed with the frozen orbitals (overlap defect 5e-6 to 4.5e-3). No rotation recovers integrals over the ROHF orbitals, so `extract` refuses these files; the α/β gate fails them by 0.1–0.4 Ha for the same reason. |
| A restricted job writes only `AA MO 2E INTEGRALS`? | **No.** ROHF writes `AA`, `BA`, `BB` and `BETA MO COEFFICIENTS` (equal to alpha). RHF and RKS write `AA` only. | `BA` is `(ββ|αα)`, packed `(-n, n, -n, n)`. |
| Is an RKS `.mat` label-identical to an RHF one? | **Yes.** The KS matrix is `ALPHA FOCK MATRIX`. | Diagonal in the KS orbitals; `--reference RKS` remains the only way to know. |
| Is UHF detectable from the `.mat` alone? | **Yes**: `BETA MO COEFFICIENTS` differs from alpha. Not from the header: `IOpCl=1, ICGU=112` for ROHF and UHF alike. | `extract` and `inspect` refuse it. |

Format facts the reader relies on, true of all ten files: Fortran sequential
unformatted, little-endian, 4-byte record markers; 11 header records; each
object a 144-byte header (64-char label + `NI, NR, NTot, LenBuf, N1..N5, ASym`)
and `ceil(NTot/LenBuf)` data records, each padded to `LenBuf`; a negative
dimension packs that index with the next as a row-wise lower triangle;
two-electron integrals in chemists' order with 8-fold `(-n,-n,-n,n)` packing
(checked against PySCF element by element); `GAUSSIAN SCALARS` is sparse
(index, value) pairs, with slots 1 virial ratio, 32 total energy, 41 nuclear
repulsion, 43 SCF energy and 44 S² identified against the `.fch`.

**What is unblocked.** RHF windowed jobs work end to end, so the porphine
**singlets** can run. The porphine **triplets** (ROHF) will produce exactly the
semicanonical integrals above and be refused. Running them is useful only to
measure timing, until a route that keeps the ROHF orbitals through the
transformation is found -- see "The ROHF route" below.

### The ROHF route

Gaussian documents two overlay-8 switches for exactly this (Overlay 8 page,
gaussian.com/overlay8):

| IOp | Documented meaning |
|---|---|
| `IOp(8/5)` | "Whether to pseudo-canonicalize ROHF orbitals": 0 default (yes if ROHF), 1 no, -1 yes, -2 yes and save over the canonical MOs. |
| `IOp(8/40)` | "Handling of ROHF window": 0 default (documented as 2), 1 ROMP2 approach -- pseudo-canonical alpha and beta orbitals and a UHF transformation, 2 treat as RHF, transforming only the alpha orbitals. |

The M0 ROHF probes set neither (their overlay-8 line is
`8/6=5,9=40000,10=91,11=11,37=2,38=13/1,4`) and L804 reported an "Open-shell
transformation" with two passes -- the pseudo-canonical, UHF-style handling,
although 8/40's documented default is 2. So the documentation alone does not
settle what these switches do in C.01; `run_rohf_route.slurm` tests it, five
ROHF probes in seconds on one core:

| Job | Route addition | Answers |
|---|---|---|
| `probe_ch2_rohf_iop_both` | `IOp(8/5=1,8/40=2)` | Does the combination give integrals over the ROHF orbitals? |
| `probe_ch2_rohf_iop_nopc` | `IOp(8/5=1)` | Is switching off the pseudo-canonicalization enough on its own? |
| `probe_ch2_rohf_iop_rhfwin` | `IOp(8/40=2)` | Is the RHF-style window enough on its own? |
| `probe_ch2_rohf_ccpvdz_iop_both` | `IOp(8/5=1,8/40=2)` | The same with 11 frozen virtuals, where mixing hit both ends. |
| `probe_nh_rohf_iop_both` | `IOp(8/5=1,8/40=2)` | The same with degenerate pi SOMOs. |

**Reading the result needs no new code.** `g16dump extract --reference ROHF`
already compares `TRANS MO COEFFICIENTS` with the active ROHF orbitals: if it
accepts a file, the integrals are over the ROHF orbitals. Then
`g16dump validate --hamiltonian` must show `h'` alpha/beta agreement near 1e-12
and `E_ref` equal to E(SCF), and `E_core` for the CH2 6-31G jobs must match the
PySCF value in the table below (−28.6809764260), since it does not depend on how
the orbitals are rotated within their blocks. An IOp Gaussian rejects ends in
"Error termination"; that is an answer too.

If one combination works, it goes into the ROHF route templates and the porphine
triplets, and the singlet–triplet comparison can run. If none does, the fallback
is to transform the window in PySCF from the `.fch` orbitals. RKS jobs extract fine and need `rebuild-fock`, which needs MOKIT and
PySCF and has **still not been run** on these real pairs
(`tests/test_gaussian_outputs.py`).

## Why the route looks like this

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
  Fortran-unformatted form, which is what `g16dump/matfile.py` reads -- not
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

Locally, or in an interactive session:

```bash
g16 NAME.gjf          # produces NAME.log, NAME.chk, NAME.mat
formchk NAME.chk NAME.fch
```

On LONI, through Slurm (`job_template.slurm` does both, plus the probe):

```bash
cd gaussian
sbatch --export=ALL,NAME=probe_ch2_rohf job_template.slurm
```

Submit from the directory holding the `.gjf`. Slurm copies the batch script to a
spool area, so the scripts here locate everything through `$SLURM_SUBMIT_DIR`
rather than `$0`.

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
   nothing further: the Fortran-unformatted form is the one the reader knows.
5. The `.mat` is written but empty or unreadable → check the filename section at
   the end of the input is present and followed by a blank line. The reader refuses
   a truncated file (no `END` record, or a broken record marker) with a sentence
   saying so.

Whatever ends up working, paste the exact working route back into these
templates and say so in the commit message.

## Probing what a `.mat` actually contains

```bash
PYTHONPATH=.. python3 -m g16dump inspect NAME.mat --json NAME.probe.json
```

Numpy only, so it runs anywhere, the login node included. The JSON lists every
object in the file with its stored dimensions, so it assumes nothing about label
names. `bash probe_only.sh` does this for every `.mat` in the directory without
rerunning Gaussian. (`../scripts/inspect_mat.py` is the older gauopen-based
probe, kept as an optional cross-check where gauopen exists.)

## The M0 probe jobs

Eight jobs in the default set, all tiny -- the largest basis is 13 functions --
so the whole set runs in seconds and the probe output can be read by eye. Two
more, on porphine at real scale, are opt-in and described further down. Run the
default set with:

```bash
cd gaussian
sbatch run_probes.slurm                  # LONI, Slurm
bash run_probes.sh                       # or interactively, anywhere
```

`run_probes.slurm` loads `gaussian/g16-c01` and points `GAUSS_SCRDIR` at
node-local scratch. One core covers the whole cheap set, which runs in minutes;
the two-hour limit is headroom. The probe inputs ask for `%nprocshared=1` and at
most `%mem=2GB`. There is no `--mem` anywhere: QB4's sbatch filter rejects it.

That runs each job, `formchk`s it, probes the `.mat` with the numpy reader,
records the environment and tars the lot into one archive. It keeps
going after a failure, so one broken route does not cost the whole set. **Send
back the archive; nothing else is needed.**

| Job | What it alone can answer |
|---|---|
| `probe_h2_smoke` | **Run first.** Two basis functions, under a second. Is `ITran` 5? Is there an `ALPHA FOCK MATRIX` record at all? If this fails, nothing below is worth reading. |
| `probe_ch2_uhf` | **Run second.** The only job that writes BETA blocks, so it is the reference for what a beta record looks like when one exists. Also: is UHF (out of scope) detectable from the `.mat` alone? |
| `probe_h2o_rhf` | Does the route work for a closed shell, and does `Window=` produce a 6-orbital block? |
| `probe_h2o_nowindow` | Meant as the control, but Gaussian's default frozen core makes it identical to `probe_h2o_rhf`; the frozen-virtual jobs are the proof `Window=` works. |
| `probe_ch2_rohf` | The Fock question, open shell. |
| `probe_nh_rohf` | A second open shell, whose singly occupied orbitals are a degenerate pi pair. |
| `probe_h2o_frozen_virtual` | The only job with frozen **virtuals** (`NFV=3`). No fixture and no other probe has ever exercised that path. |
| `probe_h2o_rks` | Does a DFT job write a Fock-like matrix, and under what label? |
| `probe_ch2_rohf_ccpvdz` | The MOKIT gate with pure d functions (cc-pVDZ, `5D 7F`): does the `.mat`/`.fch` pair pass `g16dump rebuild-fock`? |
| `probe_h2o_rks_631gs` | The MOKIT gate with Cartesian d functions and SP shells (6-31G*, Gaussian's default `6D`), on a KS job. |

### Expected dimensions

A mismatch here is the first thing to notice, so check it before anything else.
`inspect_mat.py` now prints the window it derives from the header
(`NBsUse - NFC - NFV`) and the `n**4` it implies.

| Job | basis fns | electrons | NFC | NFV | active MOs | nact | `eri_act` elements |
|---|---|---|---|---|---|---|---|
| `probe_h2_smoke` | 2 | 2 | 0 | 0 | 1-2 | 2 | 16 |
| `probe_h2o_rhf` | 7 | 10 | 1 | 0 | 2-7 | 6 | 1296 |
| `probe_h2o_nowindow` | 7 | 10 | 1 (FC default) | 0 | 2-7 | 6 | 1296 (2401 was expected) |
| `probe_ch2_rohf` | 13 | 8 | 1 | 0 | 2-13 | 12 | 20736 |
| `probe_ch2_uhf` | 13 | 8 | 1 | 0 | 2-13 | 12 | 20736 x3 blocks |
| `probe_nh_rohf` | 11 | 8 | 1 | 0 | 2-11 | 10 | 10000 |
| `probe_h2o_frozen_virtual` | 13 | 10 | 1 | 3 | 2-10 | 9 | 6561 |
| `probe_h2o_rks` | 13 | 10 | 1 | 0 | 2-13 | 12 | 20736 |
| `probe_ch2_rohf_ccpvdz` | 24 | 8 | 1 | 0 | 2-13 | 12 | 20736 |
| `probe_h2o_rks_631gs` | 19 | 10 | 1 | 0 | 2-12 | 11 | 14641 |

Note the `.mat` stores the two-electron block packed as a triangle of triangles,
not as `n**4` loose numbers, so `inspect_mat.py` will report a packing candidate
with `n = nact(nact+1)/2` -- 21 for H2O/STO-3G, 78 for CH2. That is the expected
shape, not a failure.

### Tier 2: porphine, the system this is actually for

Everything above is a toy. Ni(II) porphine is 37 atoms and about 260 basis
functions, and it is the first job here that resembles what the method was built
for. It is opt-in because it takes minutes to hours rather than seconds:

```bash
PORPHINE=1 bash run_probes.sh
sbatch --export=ALL,PORPHINE=1 --cpus-per-task=8 --time=12:00:00 run_probes.slurm

# the two RHF singlets only, without the cheap set or the ROHF triplets:
sbatch --export=ALL,PORPHINE=singlets --cpus-per-task=8 --time=12:00:00 run_probes.slurm
```

`sbatch` flags override the `#SBATCH` lines, and `--cpus-per-task` has to match
the `%nprocshared=8` in the porphine inputs: Gaussian will not notice a mismatch,
it will simply oversubscribe the allocation.

The cheap set has confirmed `ITran=5`. **The triplets will not extract**: they
are ROHF; see "M0 outcome" above. Run the singlets; run the triplets only to
measure timing, or once an ROHF route exists.

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

**Measured: the singlets (LONI job 1079620, 2026-10-08, `run_porphine_singlets.slurm`, 8 cores).**

| Job | NBasis | NFC / NFV | active | ITran | E(RHF) | cycles | wall / CPU |
|---|---|---|---|---|---|---|---|
| `probe_fe_porphine_singlet` | 269 | 82 / 166 | 21 (MOs 83–103) | 5 | −2244.01125002 | 19 | 25.3 s / 3 min 19.5 s |
| `probe_ni_porphine_singlet` | 269 | 83 / 165 | 21 (MOs 84–104) | 5 | −2488.41241972 | 20 | 26.3 s / 3 min 28.5 s |

Per Gaussian link (Fe; Ni within 0.1 s): SCF (l502) 20.2 s wall / 160 s CPU;
the windowed transformation itself (l804, 21 orbitals, 26,796 unique integrals)
**3.4 s wall / 27 s CPU**; everything else about 1 s. That is the measured half
of the claim above. The other half -- how long a full-space transformation and
dump takes -- has **not** been measured here: no full 269-orbital job was run
(about 6.6x10^8 unique integrals, ~5 GB). Read with `g16dump extract
--reference RHF`: 15 objects per file span several data records and read
correctly; MO coefficients match the `.fch` to 4e-8 and densities to 5e-9;
`TRANS MO COEFFICIENTS` equal the active orbitals exactly; `E_ref` reproduces
E(RHF) to 9e-9 (Fe) and 2e-8 (Ni), with `E_core` −2199.45758103 and
−2444.81541265. Each `.mat` is 7.8 MB; it is not committed. The output is on D:
in `hpc_smoke_tests/22_g16dump_probes/gaussian/probe_results/probe_results_qbd405_20261008/`.

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
convergence, not method.

Gaussian and PySCF order the basis functions differently even without d
functions: PySCF splits 6-31G's SP shells and sorts each atom's shells by
angular momentum. Nothing here compares AO matrices across the two programs
directly, and `g16dump rebuild-fock` converts through MOKIT; see Step 3.

| Job | `E_nuc` | `E(SCF)` | `E_core` | fixture |
|---|---|---|---|---|
| `probe_h2o_rhf` | 9.1895337629 | -74.9630231385 | -51.4711696318 | `h2o_rhf` |
| `probe_ch2_rohf` | 5.3800248517 | -38.8684854585 | -28.6809764260 | `ch2_rohf` |
| `probe_nh_rohf` | 3.5447277287 | -54.9382261545 | -- | `nh_rohf` |
| `probe_h2o_frozen_virtual` | 9.1895337629 | -75.9839744727 | -52.1215325375 | `h2o_frozen_virtual` |
| `probe_h2o_rks` | 9.1895337629 | -76.3849509041 (DFT) | -- | `h2o_rks` |
| `probe_ch2_rohf_ccpvdz` | 5.3800248517 | -38.8863647621 | -- | generated in `tests/gaussian_jobs.py` |

For `probe_h2o_rks`, note that -76.3849509041 is the **DFT** total energy. The
HF energy evaluated in those same KS orbitals is -75.9813320027, a different
number by 0.40 Ha, and it is the second one the rebuilt-Fock path reproduces.
Comparing `E_ref` against the DFT energy and calling the 0.4 Ha gap a bug is the
mistake this fixture exists to prevent.

## Step 2: getting the verdict on the Fock matrices

> **Verdict (job 1072685):** neither of the outcomes below that this section
> expected for ROHF. Gaussian stores genuine `F^α`/`F^β`, but `extract` now
> refuses ROHF before `validate` is reached, because the ROHF integrals are over
> semicanonical orbitals. See "M0 outcome" at the top. The text below is kept as
> written before the run.

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
| `SpinConsistencyError`, deviation ~0.1–1 Ha, mentioning the Roothaan operator | Gaussian stored its effective ROHF operator. **Expected, and not a bug.** | Use the rebuilt-Fock path: `g16dump rebuild-fock JOB.npz --fch JOB.fch --out JOB_rebuilt.npz`. |
| `open-shell reference ... but no beta Fock matrix is stored` | **The outcome the documentation predicts for ROHF.** `BETA FOCK MATRIX` is documented as written "for unrestricted SCF calculations", and ROHF is a *restricted* job, so it should write `ALPHA FOCK MATRIX` and no beta partner. The one stored matrix is then almost certainly the Roothaan effective operator. | Use the rebuilt-Fock path. This is not a missing label and not a bug. |
| `no Fock matrix was found in the .mat` | The labels are spelled differently, or Gaussian does not write them at all. | Send the `.probe.json`; `matfile.LABELS` needs the real spelling. |

Only the last outcome requires a code change. The first three are all handled,
which is why the writer and the rotations did not wait on this gate.

Two predictions recorded before the run. The first was **refuted** (ROHF also
writes `AA`, `BA` and `BB`, so the legacy three-block script is not evidence of
a UHF job after all); the second was **confirmed**:

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
method. Neither `validate` nor `dump` compares AO matrices between Gaussian and
PySCF, so the AO ordering does not enter here; it enters only in `rebuild-fock`
(Step 3).

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

## Step 3: the Gaussian -> MOKIT -> PySCF gate

Every probe that writes a `.fch` can also go through the rebuilt-Fock path, and
that path is where Gaussian's AO conventions meet PySCF's. On the machine that
has MOKIT and PySCF, with the unpacked archive (or the committed probes,
gunzipped):

```bash
G16DUMP_GAUSSIAN_OUTPUTS=probe_results_<host>_<date> \
pytest -m gaussian tests/test_gaussian_outputs.py -rA -s
```

For each `.mat`/`.fch` pair it prints and checks: the overlap and core
Hamiltonian mismatch before and after MOKIT's transfer, the orbital
orthonormality in PySCF's overlap, the difference between the `.mat` orbitals
and MOKIT's transfer of the `.fch` orbitals, `max|h'α - h'β|` for the rebuilt
Fock matrices, `h'`, `E_core` and `E_ref` against a full-transform PySCF oracle,
and `E_ref` against Gaussian's SCF energy for the Hartree–Fock jobs.
`probe_ch2_rohf_ccpvdz` and `probe_h2o_rks_631gs` are the two with d functions.
Until this has been run on real output, the gate is **unexecuted**.
