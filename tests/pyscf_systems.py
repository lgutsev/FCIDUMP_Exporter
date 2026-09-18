"""Real molecules through PySCF, as bundles, for oracle comparison.

These stand in for Gaussian ``.mat`` files while the cluster gate is open. The
integrals come from PySCF instead of Gaussian, but everything downstream --
bundle validation, the frozen-core algebra, the writer -- is exercised exactly
as it will be on real data, and the oracle (PySCF's own ``CASCI.get_h1eff``) is
independent of our implementation.

One subtlety drives the design here.

``mf.get_fock()`` for ROHF returns Gaussian's problem in PySCF form: the
**Roothaan effective operator**, a single matrix that is neither ``f^alpha`` nor
``f^beta``. The derivation needs the UHF-type operators built from the reference
determinant's own densities, so those are constructed explicitly. The Roothaan
matrix is still produced, by :func:`roothaan_effective_fock`, because feeding it
in *must* trip the alpha/beta consistency check -- that is a test, not an
oversight.

For Kohn-Sham references, ``e_scf`` is deliberately left unset. The KS total
energy contains exchange-correlation and is *not* the energy of the reference
determinant under the HF Hamiltonian, so storing it would make the E_ref gate
fail for a reason that is physics rather than a bug. The KS energy is recorded
in provenance instead.
"""

from __future__ import annotations

import numpy as np
from pyscf import ao2mo, gto, scf

from g16dump.bundle import Bundle
from g16dump.hamiltonian import density_matrix, rebuild_fock_with_pyscf

# Geometries in Angstrom. Small, standard, and cheap to converge.
GEOMETRIES = {
    "h2o": """
        O   0.0000000   0.0000000   0.1173000
        H   0.0000000   0.7572000  -0.4692000
        H   0.0000000  -0.7572000  -0.4692000
    """,
    "ch2": """
        C   0.0000000   0.0000000   0.0000000
        H   0.9921000   0.0000000   0.4216000
        H  -0.9921000   0.0000000   0.4216000
    """,
    "o2": """
        O   0.0000000   0.0000000   0.0000000
        O   0.0000000   0.0000000   1.2075000
    """,
    "nh": """
        N   0.0000000   0.0000000   0.0000000
        H   0.0000000   0.0000000   1.0362000
    """,
    "n2": """
        N   0.0000000   0.0000000   0.0000000
        N   0.0000000   0.0000000   2.0000000
    """,
}


def build_molecule(name: str, basis: str, charge: int = 0, spin: int = 0):
    """A converged-ready ``Mole``. ``spin`` is ``nalpha - nbeta``, PySCF's."""
    mol = gto.Mole()
    mol.atom = GEOMETRIES[name]
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.build()
    return mol


def run_scf(mol, method: str = "RHF", xc: str = "b3lyp"):
    """Run the requested reference. Returns the converged mean-field object."""
    if method in ("RHF", "ROHF"):
        mf = scf.ROHF(mol) if mol.spin else scf.RHF(mol)
    elif method in ("RKS", "ROKS"):
        from pyscf import dft

        mf = dft.ROKS(mol) if mol.spin else dft.RKS(mol)
        mf.xc = xc
    else:  # pragma: no cover
        raise ValueError(f"unsupported method {method!r}")
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged, f"{method} did not converge for this system"
    return mf


def roothaan_effective_fock(mf) -> np.ndarray:
    """PySCF's ROHF effective Fock -- the matrix that must *not* be used as f^s.

    Provided so the alpha/beta consistency check can be shown to reject it on a
    real molecule, rather than only on synthetic data.
    """
    return np.asarray(mf.get_fock())


def bundle_from_scf(
    mf,
    ncore: int,
    nact: int,
    ref_type: str,
    *,
    mo_coeff: np.ndarray | None = None,
    store_scf_energy: bool | None = None,
    fock_source: str = "rebuilt-pyscf",
    f_a_ao: np.ndarray | None = None,
    f_b_ao: np.ndarray | None = None,
) -> Bundle:
    """Build a bundle from a converged mean field and an active window.

    ``mo_coeff`` overrides the SCF orbitals (AO-major, PySCF's layout), which is
    how the rotated-orbital tests inject a non-canonical set.

    ``store_scf_energy`` defaults to True for HF references and False for KS,
    for the reason in the module docstring.
    """
    mol = mf.mol
    mo = np.asarray(mf.mo_coeff if mo_coeff is None else mo_coeff)
    nbasis, nmo = mo.shape

    is_ks = ref_type in ("RKS", "ROKS", "UKS")
    if store_scf_energy is None:
        store_scf_energy = not is_ks

    nalpha, nbeta = mol.nelec
    c = np.ascontiguousarray(mo.T)  # MO-major, our convention

    if f_a_ao is None or f_b_ao is None:
        # Always the HF Fock, built from these orbitals' own densities. For KS
        # this is mandatory; for ROHF it avoids the Roothaan operator.
        f_a_ao, f_b_ao = rebuild_fock_with_pyscf(mol, c, c, nalpha, nbeta)

    mo_act = mo[:, ncore : ncore + nact]
    eri_act = ao2mo.restore(1, ao2mo.kernel(mol, mo_act), nact)

    h_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    s_ao = mol.intor_symmetric("int1e_ovlp")

    provenance = {
        "source": "tests/pyscf_systems.py",
        "method": ref_type,
        "basis": str(mol.basis),
        "active_window": [ncore, ncore + nact],
    }
    if is_ks:
        provenance["ks_total_energy"] = float(mf.e_tot)
        provenance["note"] = (
            "e_scf intentionally unset: the KS total energy includes "
            "exchange-correlation and is not the HF reference determinant "
            "energy that E_ref reproduces."
        )

    return Bundle(
        ref_type=ref_type,
        fock_source=fock_source,
        charge=int(mol.charge),
        multiplicity=int(mol.spin) + 1,
        nelec=int(nalpha + nbeta),
        nalpha=int(nalpha),
        nbeta=int(nbeta),
        ncore=ncore,
        nact=nact,
        nfv=nmo - ncore - nact,
        nmo=nmo,
        nbasis=nbasis,
        e_nuc=float(mol.energy_nuc()),
        e_scf=float(mf.e_tot) if store_scf_energy else None,
        h_ao=h_ao,
        s_ao=s_ao,
        c_a=c,
        c_b=c.copy(),
        eri_act=np.ascontiguousarray(eri_act),
        f_a_ao=np.asarray(f_a_ao),
        f_b_ao=np.asarray(f_b_ao),
        mo_energies_a=np.asarray(mf.mo_energy) if mo_coeff is None else None,
        atom_charges=np.asarray(mol.atom_charges()),
        provenance=provenance,
    )


def block_diagonal_rotation(nmo, boundaries, rng) -> np.ndarray:
    """A random orthogonal matrix that mixes orbitals only within each block.

    ``boundaries`` is a sorted list of block edges, e.g.
    ``[0, ncore, nbeta, nalpha, nmo]``. Mixing strictly inside these blocks
    leaves the reference determinant -- and therefore the density, the Fock
    matrices and E_ref -- unchanged, while making the MO Fock matrix
    non-diagonal. That is exactly the regime where using ``diag(orbital
    energies)`` in place of ``f`` stops being valid.
    """
    from g16dump.rotate import random_orthogonal

    u = np.eye(nmo)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        size = hi - lo
        if size > 1:
            u[lo:hi, lo:hi] = random_orthogonal(size, rng)
    return u


def rotate_orbitals(mo_ao_major: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Apply a rotation to AO-major orbitals: new orbital p is sum_t u[t,p] old_t."""
    return np.ascontiguousarray(mo_ao_major @ u)


# ---------------------------------------------------------------- the oracle


def casci_reference(mf, ncore: int, nact: int, nelecas, mo_coeff=None):
    """PySCF's own frozen-core reduction: ``(h1eff, e_core, eri_act)``.

    ``CASCI.get_h1eff`` builds the effective one-electron Hamiltonian and the
    core energy by its own route, entirely independent of ``g16dump``. The
    active ERIs come from a separate ``ao2mo`` call.
    """
    from pyscf import mcscf

    mo = mf.mo_coeff if mo_coeff is None else mo_coeff
    mc = mcscf.CASCI(mf, nact, nelecas)
    mc.ncore = ncore
    h1eff, e_core = mc.get_h1eff(mo)
    mo_act = np.asarray(mo)[:, ncore : ncore + nact]
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, mo_act), nact)
    return np.asarray(h1eff), float(e_core), np.asarray(eri)
