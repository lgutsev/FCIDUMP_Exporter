# Examples

Two minimal, documented examples of consuming a g16dump FCIDUMP:

- [`dice/`](dice/) — SHCI with [Dice](https://sanshar.github.io/Dice/)
- [`block2/`](block2/) — DMRG with [Block2](https://block2.readthedocs.io/)

Neither solver is a dependency of `g16dump`, neither is imported by it, and
nothing in `g16dump/` knows that either exists. The scripts here read an
FCIDUMP's namelist header and write a solver input file; they use the standard
library only and do not import `g16dump`. If you would rather write the four
lines by hand, do that — the READMEs explain what every line means.

Both examples use the same system: the CH2 triplet, ROHF/6-31G, active window
MOs 2–13, which is the `tests/data/ch2_rohf.npz` fixture. Twelve active
orbitals and six active electrons, four alpha and two beta.
