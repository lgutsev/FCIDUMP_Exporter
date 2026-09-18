# Dice (SHCI)

[Dice](https://sanshar.github.io/Dice/) reads an FCIDUMP and runs
semistochastic heat-bath CI on it. This directory holds one worked example and
a script that writes the input file for you.

## Getting an FCIDUMP

```bash
g16dump dump JOB.npz --out FCIDUMP
```

`g16dump dump` is the M3 writer and is **not implemented on this branch yet**;
it is being built separately. Until it lands, the files here are exercised with
an FCIDUMP produced directly from `g16dump.hamiltonian.active_hamiltonian`, and
`input.dat` is written against the FCIDUMP format rather than against a
particular writer.

Dice looks for a file named **`FCIDUMP`** in the directory it runs in. There is
no keyword in this example that names the integral file, so `--out FCIDUMP` is
not a suggestion.

## Running it

`input.dat` in this directory is the CH2 triplet example. Put it beside the
FCIDUMP and run:

```bash
mpirun -np 4 /path/to/Dice/Dice input.dat > output.dat
```

To generate the same file for your own dump:

```bash
python3 make_dice_input.py FCIDUMP --out input.dat
```

## What each line means

```
nocc 6              NELEC from the FCIDUMP header: active electrons
0 2 4 6 1 3         the reference determinant, as occupied spin orbitals
end                 closes the determinant block

nroots 1            ground state only

schedule            epsilon1, the variational selection threshold, per iteration
0 1e-3                1e-3 from iteration 0 up to (not including) iteration 3
3 1e-4
6 1e-5
end
davidsonTol 5e-5    Davidson convergence for the final variational step
dE 1e-8             variational energy convergence
maxiter 10          maximum variational iterations

epsilon2 1e-8       threshold for the perturbative correction
sampleN 200         samples per stochastic PT iteration
targetError 1e-4    stop the stochastic PT at this statistical error, in Ha
```

Only the first three lines are determined by the dump. Everything below them is
a convergence choice: `epsilon1` in the schedule is the one that decides both
cost and accuracy, and a production run wants it extrapolated, not copied from
here.

### The determinant line is the part worth reading twice

Dice numbers **spin** orbitals, not spatial ones: spatial orbital `i` (0-based)
is alpha `2i` and beta `2i + 1`. So for this example, four alpha electrons in
the lowest four active orbitals and two beta electrons in the lowest two:

```
alpha  spatial 0 1 2 3  ->  0 2 4 6
beta   spatial 0 1      ->  1 3
```

`nocc` counts the electrons, which is the length of that list, not the number
of orbitals.

Two ways to get this wrong quietly. Writing spatial indices instead of spin
indices gives Dice a determinant with the wrong electron count or the wrong
orbitals, and it will happily optimize from it. And the aufbau filling above is
the reference determinant only if the active orbitals arrive in an order that
makes it so — true for a window taken straight out of Gaussian, not guaranteed
after an active-space rotation, where the occupied orbitals can end up anywhere
in the window and the list has to be written by hand.

## Checking the result

The variational energy Dice prints is an upper bound on the ground state **in
the active space**, so it must come out below the reference determinant energy
that

```bash
g16dump validate JOB.npz --hamiltonian
```

prints as `E_ref` for the same bundle. A number above `E_ref` means the
determinant line, not the dump, is wrong — most often spatial indices written
where spin indices belong.

## How this example was checked, and how it was not

**Dice was not run.** It is a compiled MPI/Boost program, it is not packaged for
installation here, and CI will not have it either. So this example is checked
against the format rather than against the program, three ways:

1. Every keyword used here appears in Dice's own documentation — `nroots`,
   `davidsonTol`, `dE`, `epsilon2`, `sampleN`, `targetError` in the
   [keyword list](https://sanshar.github.io/Dice/keywords.html), and `nocc`,
   the determinant block, `schedule`, `end` and `maxiter` in the worked example
   on the [Getting Started page](https://sanshar.github.io/Dice/gettingstarted.html).
   Nothing here is invented, and nothing is spelled differently from those pages.
2. The file's structure follows that same worked example: determinant block
   first, then the variational keywords, then the perturbative ones.
3. The determinant convention is checked against a real input, not just
   against prose. `legacy/input_back.dat` is an input the author wrote by hand
   for `legacy/FePorph_1_Window.dat`, a 21-orbital, 22-electron singlet dump.
   Running `make_dice_input.py` on that FCIDUMP reproduces its `nocc` block
   exactly:

   ```
   nocc 22
   0 2 4 6 8 10 12 14 16 18 20 1 3 5 7 9 11 13 15 17 19 21
   end
   ```

   Same convention, same ordering, independently derived from the header.

The header parser was also run against both FCIDUMPs in `legacy/`, which use a
different header spacing from the one this project writes, and against
truncated, non-FCIDUMP and impossible-multiplicity headers to check that each
is refused with a message that says what is wrong.

What none of that establishes is that a particular build of Dice accepts this
file and converges on it. That check needs Dice, and it is still outstanding.
