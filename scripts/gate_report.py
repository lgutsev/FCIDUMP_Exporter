#!/usr/bin/env python3
"""Run the M2 gates and print the table.

The gates, in the order the project defines them:

1. RHF full-space oracle agreement
2. ROHF full-space oracle agreement
3. noncanonical (rotated) orbital agreement
4. rebuilt-Fock Kohn-Sham agreement
5. FCIDUMP round trip
6. reference-determinant energy reconstruction
7. active-space rotation FCI invariance

The test suite asserts all of these; this script exists to *report* the numbers
rather than just pass or fail, because the size of a residual is information
even when it is inside tolerance.

Needs PySCF for the oracle. Run from the repository root:

    python3 scripts/gate_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402

from g16dump.bundle import validate  # noqa: E402
from g16dump.hamiltonian import active_space_hamiltonian  # noqa: E402
from g16dump.rotate import random_orthogonal, rotate  # noqa: E402
from g16dump.write import write_from_hamiltonian  # noqa: E402

TOLERANCE = 1e-8


def main() -> int:
    try:
        import pyscf  # noqa: F401
    except ImportError:
        print("PySCF is required for the oracle. pip install pyscf", file=sys.stderr)
        return 2

    import tempfile

    from fcidump_reader import determinant_energy, read_fcidump
    from pyscf_systems import (
        block_diagonal_rotation,
        build_molecule,
        bundle_from_scf,
        casci_reference,
        rotate_orbitals,
        run_scf,
        tight_fci,
    )

    # name, molecule, basis, spin, ref_type, ncore, nact, rotate?
    systems = [
        ("H2O   RHF  STO-3G", "h2o", "sto-3g", 0, "RHF", 1, 6, False),
        ("H2O   RHF  6-31G", "h2o", "6-31g", 0, "RHF", 1, 8, False),
        ("CH2   ROHF 6-31G", "ch2", "6-31g", 2, "ROHF", 1, 8, False),
        ("O2    ROHF 6-31G", "o2", "6-31g", 2, "ROHF", 2, 8, False),
        ("NH    ROHF 6-31G", "nh", "6-31g", 2, "ROHF", 1, 7, False),
        ("N2    RHF  cc-pVDZ 2.0A", "n2", "cc-pvdz", 0, "RHF", 2, 8, False),
        ("CH2   ROHF rotated", "ch2", "6-31g", 2, "ROHF", 1, 8, True),
        ("H2O   RKS  6-31G", "h2o", "6-31g", 0, "RKS", 1, 8, False),
    ]

    header = (
        f"{'system':26s} {'|dE_ref|':>10s} {'spin':>9s} {'max|dh|':>9s} "
        f"{'|dEcore|':>9s} {'max|dERI|':>9s} {'roundtrip':>10s} {'|dE_FCI|':>10s}"
    )
    print(header)
    print("-" * len(header))

    worst = 0.0
    failures = []

    for label, name, basis, spin, ref, ncore, nact, do_rotate in systems:
        mol = build_molecule(name, basis, spin=spin)
        method = ref if ref in ("RKS", "ROKS") else ("ROHF" if spin else "RHF")
        mf = run_scf(mol, method)

        mo = None
        if do_rotate:
            nalpha, nbeta = mol.nelec
            nmo = mf.mo_coeff.shape[1]
            u = block_diagonal_rotation(
                nmo, [0, ncore, nbeta, nalpha, nmo], np.random.default_rng(0)
            )
            mo = rotate_orbitals(mf.mo_coeff, u)

        bundle = bundle_from_scf(mf, ncore, nact, ref, mo_coeff=mo)
        validate(bundle)

        is_ks = ref in ("RKS", "ROKS")
        ash = active_space_hamiltonian(bundle, tol_scf=None if is_ks else TOLERANCE)

        # Gate 6: E_ref reconstruction. Not applicable to KS, whose total energy
        # includes exchange-correlation and is not a determinant energy.
        d_ref = float("nan") if is_ks else abs(ash.e_ref - mf.e_tot)

        # Gates 1-4: agreement with the independent oracle.
        h1, e_core, eri = casci_reference(
            mf, ncore, nact, (bundle.nocc_a_act, bundle.nocc_b_act), mo_coeff=mo
        )
        d_h = float(np.max(np.abs(ash.h_eff - h1)))
        d_core = abs(ash.e_core - e_core)
        d_eri = float(np.max(np.abs(ash.eri_act - eri)))

        # Gate 5: FCIDUMP round trip through an independent parser.
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "FCIDUMP")
            write_from_hamiltonian(path, ash)
            parsed = read_fcidump(path)
            d_trip = max(
                float(np.max(np.abs(parsed.h1 - ash.h_eff))),
                float(np.max(np.abs(parsed.eri - ash.eri_act))),
                abs(parsed.e_core - ash.e_core),
            )
            if not is_ks:
                energy = determinant_energy(
                    parsed.h1, parsed.eri, parsed.e_core,
                    ash.nocc_a_act, ash.nocc_b_act,
                )
                d_ref = max(d_ref, abs(energy - mf.e_tot))

        # Gate 7: FCI invariance under a random active-space rotation.
        # Tight convergence is required here: comparing two orbital bases is a
        # much stricter demand than converging one calculation, and PySCF's
        # default leaves ~1e-4 Ha on a strongly multireference system in a
        # rotated basis.
        nelecas = (ash.nocc_a_act, ash.nocc_b_act)
        before = tight_fci(ash.h_eff, ash.eri_act, nact, nelecas)
        u = random_orthogonal(nact, np.random.default_rng(1))
        h_rot, eri_rot, _ = rotate(ash.h_eff, ash.eri_act, ash.e_core, u)
        after = tight_fci(h_rot, eri_rot, nact, nelecas)
        d_fci = abs(before - after)

        ref_text = "     n/a  " if is_ks else f"{d_ref:10.2e}"
        print(
            f"{label:26s} {ref_text} {ash.diagnostics['spin_error']:9.2e} "
            f"{d_h:9.2e} {d_core:9.2e} {d_eri:9.2e} {d_trip:10.2e} {d_fci:10.2e}"
        )

        for gate, value in (
            ("E_ref", 0.0 if is_ks else d_ref),
            ("spin", ash.diagnostics["spin_error"]),
            ("h'", d_h),
            ("E_core", d_core),
            ("ERI", d_eri),
            ("round trip", d_trip),
            ("FCI invariance", d_fci),
        ):
            worst = max(worst, value)
            if value > TOLERANCE:
                failures.append(f"{label}: {gate} = {value:.3e}")

    print()
    print(f"worst residual across all gates: {worst:.3e}  (tolerance {TOLERANCE:.0e})")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("all gates pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
