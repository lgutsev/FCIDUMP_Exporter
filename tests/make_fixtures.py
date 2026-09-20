#!/usr/bin/env python3
"""Generate the committed ``.npz`` test fixtures from PySCF.

The fixtures let the test suite check the schema and the frozen-core algebra
without PySCF, Gaussian or gauopen installed, which is the whole reason the
bundle format exists. Regenerating them needs PySCF::

    python3 tests/make_fixtures.py

PySCF stands in for Gaussian here, not as an oracle: it produces orbitals, an
overlap, a core Hamiltonian and a windowed two-electron block in exactly the
shape a ``.mat`` will, and the fixtures then exercise the same code path the
real files will. The independent-oracle comparison is a separate test that
imports PySCF directly.

One fixture per thing worth distinguishing:

``h2o_rhf``
    Closed shell, canonical orbitals, stored Fock. The case the legacy scripts
    got right.
``ch2_rohf``
    Open-shell triplet with genuine ``F^alpha``/``F^beta``. The case the legacy
    algebra gets wrong.
``ch2_rohf_roothaan``
    The same job carrying Gaussian's Roothaan effective ROHF operator instead of
    the two spin Fock matrices, which is what a ``.mat`` is expected to hold. It
    exists to be rejected.
``ch2_rohf_rotated``
    The same orbitals rotated within the occupied and virtual blocks, so the MO
    Fock matrix is not diagonal while the reference determinant is unchanged.
    This is where ``diag(orbital_energies)`` fails and the Fock-matrix algebra
    does not.
``nh_rohf``
    A second open-shell system, and a different one: NH is a diatomic with two
    singly occupied pi orbitals that are degenerate, where CH2's are not. A
    degeneracy is where an orbital-order convention is easiest to get wrong, so
    one system is not enough to believe the open-shell path.
``h2o_rks``
    Real B3LYP orbitals. Every other fixture is Hartree-Fock, so the KS gate was
    only ever tested on a relabelled HF bundle -- which cannot show the thing
    that actually matters, that the HF energy evaluated in KS orbitals is a
    different number from the DFT energy. This one carries both.
``h2o_frozen_virtual``
    The only fixture with frozen **virtuals**. Every other one windows to the
    top of the MO space, so ``nmo - act_stop`` is zero and any assertion about
    the frozen-virtual block compares two empty arrays and passes whatever the
    code does. Here three virtuals sit above the window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from g16dump.bundle import Bundle, make_provenance, save  # noqa: E402
from g16dump.hamiltonian import rebuild_fock  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"

H2O = "O 0.000000 0.000000 0.117300; H 0.000000 0.757200 -0.469200; " \
      "H 0.000000 -0.757200 -0.469200"
CH2 = "C 0.000000 0.000000 0.190000; H 0.000000 0.990000 -0.560000; " \
      "H 0.000000 -0.990000 -0.560000"
NH = "N 0.000000 0.000000 0.000000; H 0.000000 0.000000 1.045000"


def _windowed_eri(mol, mo, act_start, act_stop):
    """The active-window MO two-electron block, chemist's ``(tu|vw)``.

    This is the quantity Gaussian's ``Window=`` produces: a transformation over
    the active orbitals only, never over the full MO space.
    """
    from pyscf import ao2mo

    c_act = mo[:, act_start:act_stop]
    nact = act_stop - act_start
    return ao2mo.restore(
        1, ao2mo.general(mol, [c_act] * 4, compact=False), nact
    ).reshape((nact,) * 4)


def _bundle(mol, mf, mo, nalpha, nbeta, act_start, act_stop, reference,
            fock_alpha, fock_beta, fock_source, note):
    return Bundle(
        reference=reference,
        charge=int(mol.charge),
        multiplicity=int(mol.spin) + 1,
        nelec=nalpha + nbeta,
        nalpha=nalpha,
        nbeta=nbeta,
        nao=mo.shape[0],
        nmo=mo.shape[1],
        ncore=act_start,
        nact=act_stop - act_start,
        act_start=act_start,
        act_stop=act_stop,
        e_nuc=float(mol.energy_nuc()),
        mo_coeff=np.asarray(mo),
        overlap=np.asarray(mf.get_ovlp()),
        hcore_ao=np.asarray(mf.get_hcore()),
        eri_act=_windowed_eri(mol, mo, act_start, act_stop),
        fock_source=fock_source,
        fock_ao_alpha=fock_alpha,
        fock_ao_beta=fock_beta,
        mo_energy_alpha=np.asarray(mf.mo_energy, dtype=float).ravel()[: mo.shape[1]]
        if np.asarray(mf.mo_energy).ndim == 1
        else None,
        atom_charges=mol.atom_charges().astype(float),
        e_scf=float(mf.e_tot),
        provenance=make_provenance(
            basis=mol.basis if isinstance(mol.basis, str) else None,
            method=reference,
            window_1based=(act_start + 1, act_stop),
            fock_source=fock_source,
            extra={"generated_by": "tests/make_fixtures.py", "note": note},
        ),
    )


def _random_orthogonal(n, rng):
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return q * np.sign(np.diag(r))


def main() -> int:
    from pyscf import gto, scf

    DATA.mkdir(parents=True, exist_ok=True)
    written = []

    # ------------------------------------------------- H2O RHF, canonical
    mol = gto.M(atom=H2O, basis="sto-3g", verbose=0)
    # Tight, for the same reason the Gaussian route asks for SCF=(Conver=10):
    # a loosely converged SCF leaves MO Fock off-diagonals of the order of the
    # convergence threshold, and those are exactly what these fixtures measure.
    mf = scf.RHF(mol).set(conv_tol=1e-12).run()
    mo = mf.mo_coeff
    # For RHF, Hcore + J[P] - K[P]/2 over the total density is exactly
    # F^alpha = Hcore + J[Pa+Pb] - K[Pa]; it is stored as Gaussian would store it.
    written.append(save(_bundle(
        mol, mf, mo, 5, 5, 1, 7, "RHF",
        np.asarray(mf.get_fock()), None, "gaussian",
        "canonical RHF, Fock as Gaussian stores it"),
        DATA / "h2o_rhf.npz"))

    # ------------------------------------------------ CH2 ROHF triplet
    mol2 = gto.M(atom=CH2, basis="6-31g", spin=2, verbose=0)
    mf2 = scf.ROHF(mol2).set(conv_tol=1e-12).run()
    mo2 = mf2.mo_coeff
    nalpha, nbeta = (int(n) for n in mf2.nelec)
    act_start, act_stop = 1, mo2.shape[1]

    fock_a, fock_b = rebuild_fock(mol2, mo2, nalpha, nbeta, mf2.get_hcore())
    written.append(save(_bundle(
        mol2, mf2, mo2, nalpha, nbeta, act_start, act_stop, "ROHF",
        fock_a, fock_b, "pyscf_rebuilt",
        "genuine F^alpha and F^beta from the reference determinant densities"),
        DATA / "ch2_rohf.npz"))

    # The Roothaan effective operator, which is what a .mat is expected to
    # carry. Stored under both spins, as a reader with no way to tell would.
    roothaan = np.asarray(mf2.get_fock())
    written.append(save(_bundle(
        mol2, mf2, mo2, nalpha, nbeta, act_start, act_stop, "ROHF",
        roothaan, roothaan, "gaussian",
        "Roothaan effective ROHF operator, not F^alpha/F^beta: must be rejected"),
        DATA / "ch2_rohf_roothaan.npz"))

    # ------------------------------------- CH2 ROHF, non-canonical orbitals
    # Rotate within the core, the doubly occupied active, the singly occupied
    # and the virtual blocks separately. The reference determinant spans the
    # same space, so every energy is unchanged, but the MO Fock matrix picks up
    # off-diagonal elements and diag(orbital_energies) stops being h'.
    rng = np.random.default_rng(20260918)
    mo_rot = np.array(mo2, copy=True)
    for lo, hi in ((0, act_start), (act_start, nbeta), (nbeta, nalpha),
                   (nalpha, act_stop)):
        if hi - lo > 1:
            mo_rot[:, lo:hi] = mo2[:, lo:hi] @ _random_orthogonal(hi - lo, rng)

    fock_a_rot, fock_b_rot = rebuild_fock(
        mol2, mo_rot, nalpha, nbeta, mf2.get_hcore())
    rotated = _bundle(
        mol2, mf2, mo_rot, nalpha, nbeta, act_start, act_stop, "ROHF",
        fock_a_rot, fock_b_rot, "pyscf_rebuilt",
        "ROHF orbitals rotated within occupied and virtual blocks; the MO Fock "
        "matrix is not diagonal")
    # The orbital energies no longer describe these orbitals. Dropping them is
    # the honest thing: they are diagnostics, and here they diagnose nothing.
    rotated.mo_energy_alpha = None
    written.append(save(rotated, DATA / "ch2_rohf_rotated.npz"))

    # ------------------------------------------------- NH ROHF triplet
    # A second open-shell case, chosen to differ from CH2 where it counts: the
    # two singly occupied orbitals are a degenerate pi pair, so any assumption
    # about orbital ordering that happens to hold for CH2 gets a second test.
    mol3 = gto.M(atom=NH, basis="6-31g", spin=2, verbose=0)
    mf3 = scf.ROHF(mol3).set(conv_tol=1e-12).run()
    mo3 = mf3.mo_coeff
    nalpha3, nbeta3 = (int(n) for n in mf3.nelec)
    fock_a3, fock_b3 = rebuild_fock(mol3, mo3, nalpha3, nbeta3, mf3.get_hcore())
    written.append(save(_bundle(
        mol3, mf3, mo3, nalpha3, nbeta3, 1, mo3.shape[1], "ROHF",
        fock_a3, fock_b3, "pyscf_rebuilt",
        "NH triplet; the singly occupied orbitals are a degenerate pi pair"),
        DATA / "nh_rohf.npz"))

    # ------------------------------------------------ H2O RKS/B3LYP
    # A Kohn-Sham bundle carries no Fock matrix at all: the stored KS matrix
    # contains exchange-correlation and is not the HF Fock operator, so
    # matfile.py refuses to put it in fock_ao_*. The rebuilt-Fock path is the
    # only way to use these orbitals, and the energy it produces is the HF
    # energy evaluated in KS orbitals -- deliberately NOT the DFT total energy,
    # which is what e_scf holds. Both numbers are recorded so a test can assert
    # they differ rather than take it on faith.
    from pyscf import dft

    mol4 = gto.M(atom=H2O, basis="6-31g", verbose=0)
    mf4 = dft.RKS(mol4).set(xc="b3lyp", conv_tol=1e-12).run()
    mo4 = mf4.mo_coeff

    ks = _bundle(
        mol4, mf4, mo4, 5, 5, 1, mo4.shape[1], "RKS",
        None, None, "none",
        "real B3LYP orbitals; no Fock matrix, by design")
    hf_fock_a, _ = rebuild_fock(mol4, mo4, 5, 5, mf4.get_hcore())
    hcore_mo = mo4.T @ mf4.get_hcore() @ mo4
    fock_mo = mo4.T @ hf_fock_a @ mo4
    e_hf_in_ks = float(mol4.energy_nuc()) + float(
        np.sum(np.diag(hcore_mo)[:5] + np.diag(fock_mo)[:5])
    )
    ks.provenance["e_hf_in_ks_orbitals"] = e_hf_in_ks
    ks.provenance["e_dft"] = float(mf4.e_tot)
    written.append(save(ks, DATA / "h2o_rks.npz"))
    print(f"  h2o_rks: E(DFT) = {mf4.e_tot:.10f}, "
          f"E(HF in KS orbitals) = {e_hf_in_ks:.10f}")

    # ------------------------------------------- H2O RHF with frozen virtuals
    # Window MOs 2-10 of 13, so one frozen core and three frozen virtuals. This
    # is the shape `gaussian/probe_h2o_frozen_virtual.gjf` produces, and the
    # only fixture where nmo - act_stop is not zero.
    mol5 = gto.M(atom=H2O, basis="6-31g", verbose=0)
    mf5 = scf.RHF(mol5).set(conv_tol=1e-12).run()
    mo5 = mf5.mo_coeff
    written.append(save(_bundle(
        mol5, mf5, mo5, 5, 5, 1, 10, "RHF",
        np.asarray(mf5.get_fock()), None, "gaussian",
        "one frozen core and three frozen virtuals; the only fixture with nfv>0"),
        DATA / "h2o_frozen_virtual.npz"))
    print(f"  h2o_frozen_virtual: nmo={mo5.shape[1]}, window 2-10, "
          f"{mo5.shape[1] - 10} frozen virtuals, E(SCF) = {mf5.e_tot:.10f}")

    for path in written:
        print(f"  {path.relative_to(DATA.parent.parent)}  "
              f"{path.stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
