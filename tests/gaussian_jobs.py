"""Gaussian-side data for the MOKIT tests, generated from PySCF.

Each :class:`GaussianJob` runs an SCF in PySCF and then presents it the way a
Gaussian job would: a bundle whose AO quantities are in Gaussian's AO order and
normalization (what ``g16dump extract`` produces from a ``.mat``), and a
formatted checkpoint in ``formchk``'s layout (what MOKIT reads). The mapping
comes from :mod:`gaussian_fch`, which knows nothing about MOKIT, so the
package's MOKIT route is compared against an independent statement of the
conventions rather than against itself.

The ``.fch`` prints basis exponents and orbital coefficients to nine
significant figures, exactly as Gaussian's does, so the molecule MOKIT rebuilds
from it differs from the one the "``.mat``" quantities were computed in by that
rounding. That is the precision floor a real ``.mat``/``.fch`` pair has too,
and every tolerance below is checked against it.

The oracle quantities are computed from the original PySCF calculation by a
full AO->MO transform (the functions in ``test_oracles.py``), never through
Gaussian order, MOKIT or ``g16dump``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gaussian_fch import gaussian_to_pyscf, write_fch

H2O = "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692"
CH2 = "C 0.0 0.0 0.19; H 0.0 0.99 -0.56; H 0.0 -0.99 -0.56"
NH = "N 0.0 0.0 0.0; H 0.0 0.0 1.0362"

#: ``stored_fock``: what the "``.mat``" carries. ``gaussian`` is a genuine
#: F^alpha/F^beta pair (RHF), ``roothaan`` Gaussian's ROHF effective operator,
#: which the alpha/beta gate rejects, and ``none`` no Fock matrix at all.
JOBS = {
    "h2o_rhf_631gs_cart": dict(
        atom=H2O, basis="6-31g*", cart=True, spin=0, reference="RHF",
        window=(2, 12), stored_fock="gaussian",
        note="RHF control; Gaussian's default 6D Cartesian d and Pople SP shells",
    ),
    "h2o_rhf_ccpvtz_cart": dict(
        atom=H2O, basis="cc-pvtz", cart=True, spin=0, reference="RHF",
        window=(2, 20), stored_fock="gaussian",
        note="Cartesian d and f (6D 10F): order and per-function normalization",
    ),
    "ch2_rohf_ccpvdz": dict(
        atom=CH2, basis="cc-pvdz", cart=False, spin=2, reference="ROHF",
        window=(2, 13), stored_fock="roothaan",
        note="ROHF with pure d; Gaussian stored the Roothaan operator",
    ),
    "ch2_rohf_631g": dict(
        atom=CH2, basis="6-31g", cart=False, spin=2, reference="ROHF",
        window=(2, 13), stored_fock="roothaan",
        note="no polarization functions at all, still reordered: SP shells",
    ),
    "nh_rohf_631gs_cart": dict(
        atom=NH, basis="6-31g*", cart=True, spin=2, reference="ROHF",
        window=(2, 11), stored_fock="none",
        note="ROHF with no usable Fock matrix; Cartesian d and SP shells",
    ),
    "h2o_rks_ccpvdz": dict(
        atom=H2O, basis="cc-pvdz", cart=False, spin=0, reference="RKS", xc="b3lyp",
        window=(2, 13), stored_fock="none",
        note="Kohn-Sham orbitals, pure d: the rebuilt path is the normal route",
    ),
}


class GaussianJob:
    def __init__(self, name: str, spec: dict, directory: Path):
        from pyscf import ao2mo, dft, gto, scf

        from test_oracles import (
            _full_eri,
            _oracle_determinant_energy,
            _oracle_frozen_core,
            _oracle_to_mo,
        )

        from g16dump.bundle import Bundle, make_provenance

        self.name, self.spec = name, spec
        mol = gto.M(
            atom=spec["atom"], basis=spec["basis"], cart=spec["cart"],
            spin=spec["spin"], verbose=0,
        )
        if spec["reference"] == "RKS":
            mf = dft.RKS(mol).set(xc=spec["xc"], conv_tol=1e-12)
        elif spec["spin"] == 0:
            mf = scf.RHF(mol).set(conv_tol=1e-12)
        else:
            mf = scf.ROHF(mol).set(conv_tol=1e-12)
        mf.run()
        assert mf.converged, name

        self.mol, self.mf = mol, mf
        mo = np.asarray(mf.mo_coeff)
        nalpha, nbeta = (int(n) for n in mol.nelec)
        first, last = spec["window"]
        start, stop = first - 1, last
        nact = stop - start

        # Gaussian's AO basis: C_pyscf = X C_gaussian, A_gaussian = X.T A X.
        x = gaussian_to_pyscf(mol)
        self.x_independent = x
        x_inv = np.linalg.inv(x)
        mo_g = x_inv @ mo
        overlap_g = x.T @ mf.get_ovlp() @ x
        hcore_g = x.T @ mf.get_hcore() @ x

        c_act = mo[:, start:stop]
        eri_active = ao2mo.restore(
            1, ao2mo.general(mol, [c_act] * 4, compact=False), nact
        ).reshape((nact,) * 4)

        fock_a = fock_b = None
        fock_source = "none"
        if spec["stored_fock"] == "gaussian":
            fock = np.asarray(mf.get_fock())
            fock_a = fock_b = x.T @ fock @ x
            fock_source = "gaussian"
        elif spec["stored_fock"] == "roothaan":
            roothaan = np.asarray(mf.get_fock())  # PySCF's ROHF effective operator
            fock_a = fock_b = x.T @ roothaan @ x
            fock_source = "gaussian"

        self.bundle = Bundle(
            reference_type=spec["reference"],
            charge=0,
            multiplicity=spec["spin"] + 1,
            nelec=nalpha + nbeta,
            nalpha=nalpha,
            nbeta=nbeta,
            nao=mol.nao,
            nmo=mo.shape[1],
            ncore=start,
            nact=nact,
            active_first=first,
            active_last=last,
            enuc=float(mol.energy_nuc()),
            C=mo_g,
            S=overlap_g,
            Hcore_ao=hcore_g,
            eri_active=eri_active,
            source_program="gaussian16",
            source_file=f"{name}.mat",
            fock_source=fock_source,
            F_alpha_ao=fock_a,
            F_beta_ao=fock_b,
            orbital_energies=np.asarray(mf.mo_energy, dtype=float),
            atom_charges=mol.atom_charges().astype(float),
            escf=float(mf.e_tot),
            provenance=make_provenance(
                source_file=f"{name}.mat",
                basis=spec["basis"],
                method=spec["reference"],
                window_1based=(first, last),
                fock_source=fock_source,
                extra={"generated_by": "tests/gaussian_jobs.py"},
            ),
        )

        directory.mkdir(parents=True, exist_ok=True)
        self.fch = write_fch(
            directory / f"{name}.fch", mol, mo_g, mf.mo_energy,
            method=("R" + spec["xc"].upper()) if spec["reference"] == "RKS"
            else spec["reference"],
            total_energy=float(mf.e_tot), nalpha=nalpha, nbeta=nbeta,
        )

        # -- the oracle: full transform in the original PySCF basis ----------
        e_nuc = float(mol.energy_nuc())
        hcore_mo = _oracle_to_mo(mf.get_hcore(), mo)
        eri_full = _full_eri(mol, mo)
        self.oracle_h_eff, self.oracle_e_core = _oracle_frozen_core(
            hcore_mo, eri_full, e_nuc, start, start, stop
        )
        self.oracle_eri_active = eri_full[start:stop, start:stop, start:stop, start:stop]
        self.oracle_e_ref = _oracle_determinant_energy(
            hcore_mo, eri_full, nalpha, nbeta, constant=e_nuc
        )
        # F^sigma in PySCF's own basis, from PySCF's own driver, for comparison.
        dm_a = mo[:, :nalpha] @ mo[:, :nalpha].T
        dm_b = mo[:, :nbeta] @ mo[:, :nbeta].T
        uhf_fock = scf.UHF(mol).get_fock(dm=np.array([dm_a, dm_b]))
        self.pyscf_fock_mo = (mo.T @ uhf_fock[0] @ mo, mo.T @ uhf_fock[1] @ mo)
