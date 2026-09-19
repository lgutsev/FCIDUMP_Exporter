# Using the FCIDUMP

Two solvers, one file. Both read the same FCIDUMP; what differs is the input
that points at it.

Everything here assumes you already have a bundle:

```bash
g16dump extract JOB.mat --fch JOB.fch --reference ROHF --window 2 13 --out JOB.npz
g16dump validate JOB.npz --hamiltonian     # confirm E_ref matches E_scf first
```

`validate --hamiltonian` before dumping, every time. It is the step that
catches a Roothaan operator or a Kohn-Sham matrix, and it costs a second.

## Dice (SHCI)

```bash
g16dump dump JOB.npz --out FCIDUMP --dice-nocc
```

`--dice-nocc` prints the occupation block for Dice's `input.dat`. Dice indexes
**spin** orbitals: spatial orbital `p` is alpha `2p` and beta `2p+1`, both
0-based, and the reference determinant fills the lowest spatial orbitals with
alpha first. For the CH2 triplet example -- 6 active electrons, MS2 = 2 -- that
is:

```
nocc 6
0 2 4 6 1 3
end
```

A complete `input.dat` for a first variational-plus-perturbative run is in
[`dice_input.dat`](dice_input.dat). Dice looks for a file literally named
`FCIDUMP` in the working directory, so either name it that or symlink it.

```bash
mpirun -np 20 /path/to/Dice/Dice > output.dat
```

## Block2 (DMRG)

Block2 reads the FCIDUMP directly. The Python interface is the least
error-prone way in, because it takes the file as given and needs nothing
restated:

```python
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

driver = DMRGDriver(scratch="./scratch", symm_type=SymmetryTypes.SU2, n_threads=8)
driver.read_fcidump(filename="FCIDUMP")
driver.initialize_system(
    n_sites=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin,
    orb_sym=driver.orb_sym,
)

mpo = driver.get_qc_mpo(h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore, iprint=1)
ket = driver.get_random_mps(tag="KET", bond_dim=250, nroots=1)
energy = driver.dmrg(
    mpo, ket,
    n_sweeps=20,
    bond_dims=[250] * 4 + [500] * 4 + [1000] * 12,
    noises=[1e-4] * 4 + [1e-5] * 8 + [0.0],
    thrds=[1e-8] * 20,
    iprint=1,
)
print(f"DMRG energy = {energy:.10f} Ha")
```

`driver.read_fcidump` fills `n_sites`, `n_elec`, `spin` and `orb_sym` from the
header, so `NORB`, `NELEC`, `MS2` and `ORBSYM` are doing real work here. The
runnable version is [`block2_dmrg.py`](block2_dmrg.py).

`SymmetryTypes.SU2` is spin-adapted and targets the spin state; `SZ` targets the
Sz sector that `MS2` names. For an open-shell reference, `SU2` is usually what
you want and `MS2` still has to be right.

## Checking the answer before trusting it

Rotating the active orbitals among themselves cannot change any energy, so the
cheapest real test of a result is to do it and see:

```bash
g16dump rotate JOB.npz --random 1 --out JOB_rotated.npz
g16dump dump   JOB_rotated.npz --out FCIDUMP_rotated
```

Run the solver on both. The correlated energies should agree to solver
precision. If they do not, the disagreement is in the solver's convergence or in
the active space, not in the integrals -- and you have learned that for the cost
of one extra run.

`E_core` and the `&FCI` header are identical between the two files by
construction, so `diff` on the first four lines is a fast sanity check that you
dumped what you think you did.
