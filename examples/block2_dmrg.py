#!/usr/bin/env python3
"""Run a Block2 DMRG calculation on a FCIDUMP produced by ``g16dump dump``.

Usage::

    python3 block2_dmrg.py FCIDUMP [--bond-dim 1000] [--scratch ./scratch]

Nothing here is specific to this project except the observation at the bottom:
because ``g16dump rotate`` can produce a second FCIDUMP that must give the same
energy, running this twice is a real test of the result rather than a repeat.

Block2 is not a dependency of g16dump and is not installed by any of its extras;
install it separately with ``pip install block2``.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fcidump", help="the FCIDUMP to read")
    parser.add_argument("--bond-dim", type=int, default=1000,
                        help="maximum bond dimension (default: 1000)")
    parser.add_argument("--scratch", default="./scratch",
                        help="Block2 scratch directory (default: ./scratch)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sz", action="store_true",
                        help="target the Sz sector named by MS2 instead of "
                             "using the spin-adapted SU(2) solver")
    args = parser.parse_args(argv)

    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError:
        print(
            "block2 is not installed. It is not a dependency of g16dump; "
            "install it with: pip install block2",
            file=sys.stderr,
        )
        return 2

    driver = DMRGDriver(
        scratch=args.scratch,
        symm_type=SymmetryTypes.SZ if args.sz else SymmetryTypes.SU2,
        n_threads=args.threads,
    )

    # Everything the system needs comes out of the FCIDUMP header: NORB, NELEC,
    # MS2 and ORBSYM. Nothing is restated here, so a header that disagrees with
    # the integrals fails loudly instead of being overridden by an argument.
    driver.read_fcidump(filename=args.fcidump)
    driver.initialize_system(
        n_sites=driver.n_sites,
        n_elec=driver.n_elec,
        spin=driver.spin,
        orb_sym=driver.orb_sym,
    )
    print(
        f"{args.fcidump}: {driver.n_sites} orbitals, {driver.n_elec} electrons, "
        f"spin {driver.spin}, E_core = {driver.ecore:.10f} Ha"
    )

    mpo = driver.get_qc_mpo(
        h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore, iprint=1
    )
    ket = driver.get_random_mps(tag="KET", bond_dim=250, nroots=1)

    top = args.bond_dim
    energy = driver.dmrg(
        mpo,
        ket,
        n_sweeps=20,
        bond_dims=[250] * 4 + [min(500, top)] * 4 + [top] * 12,
        noises=[1e-4] * 4 + [1e-5] * 8 + [0.0] * 8,
        thrds=[1e-8] * 20,
        iprint=1,
    )

    print(f"\nDMRG energy = {energy:.10f} Ha")
    print(
        "\nTo check this number rather than trust it, rotate the active space "
        "and run again:\n"
        "    g16dump rotate JOB.npz --random 1 --out JOB_rotated.npz\n"
        "    g16dump dump   JOB_rotated.npz --out FCIDUMP_rotated\n"
        "The two energies must agree to solver precision; a rotation cannot "
        "move them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
