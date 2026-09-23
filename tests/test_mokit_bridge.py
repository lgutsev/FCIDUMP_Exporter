"""MOKIT's Gaussian -> PySCF conversion, tested on what it has to get right.

The rebuilt-Fock path puts ``.mat`` orbitals (Gaussian AO basis) into a PySCF
``Mole`` that MOKIT rebuilds from the ``.fch`` (PySCF AO basis). These tests
establish, per system and numerically:

1. that the two AO bases really differ -- the raw ``.mat`` orbitals are not
   orthonormal in the ``Mole``'s overlap and the AO matrices do not match --
   so using them directly is wrong, not merely untidy;
2. that the transformation g16dump reads out of MOKIT's ``fch2py`` equals an
   independent statement of Gaussian's conventions (``gaussian_fch.py``);
3. that after it, overlap and core Hamiltonian agree element by element,
   the orbitals are orthonormal, agree with MOKIT's own transfer of the
   ``.fch`` orbitals, and give the same densities;
4. that the rebuilt ``F^alpha``/``F^beta`` equal PySCF's own UHF-type Fock
   operator for those orbitals.

Every system has shells that Gaussian and PySCF order differently; most have d
or f functions, one has only Pople SP shells, because that alone is enough.
The end-to-end Hamiltonian gates are in ``test_rebuilt_fock_gates.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gaussian_jobs import JOBS

from g16dump import fch as F
from g16dump import hamiltonian as H

pytestmark = [pytest.mark.mokit, pytest.mark.pyscf]

#: The ``.fch`` precision floor, with headroom. See gaussian_jobs.py.
FCH_FLOOR = 1e-7


@pytest.fixture(scope="module", params=list(JOBS))
def job(request, gaussian_job):
    return gaussian_job(request.param)


@pytest.fixture(scope="module")
def conversion(job):
    return F.gaussian_to_pyscf(job.bundle, job.fch)


def test_mol_from_fch_rebuilds_the_molecule(job):
    mol = F.mol_from_fch(job.fch)
    assert mol.natm == job.mol.natm
    assert mol.nelectron == job.mol.nelectron
    assert mol.spin == job.mol.spin
    assert mol.nao == job.mol.nao
    assert bool(mol.cart) == bool(job.mol.cart)
    assert abs(mol.energy_nuc() - job.mol.energy_nuc()) < 1e-8
    assert np.allclose(mol.atom_coords(), job.mol.atom_coords(), atol=1e-7)


def test_mokit_transform_equals_the_independent_gaussian_convention(job, conversion):
    """Two derivations of one map: MOKIT's fch2py, and Gaussian's documentation."""
    # MOKIT numbers its atoms (O1, H2, ...); the functions themselves must match.
    def functions(mol):
        return [(atom, shell, m) for atom, _, shell, m in mol.ao_labels(fmt=False)]

    assert functions(conversion.mol) == functions(job.mol)
    assert np.max(np.abs(conversion.transform - job.x_independent)) < 1e-12
    assert conversion.checks["transfer_linearity"] < 1e-12


def test_the_raw_mat_orbitals_are_not_usable_in_the_pyscf_mole(job, conversion):
    """The mismatch is real and large, for every system here."""
    checks = conversion.checks
    assert checks["overlap_before"] > 0.1
    assert checks["hcore_before_rel"] > 1e-2
    assert checks["orthonormality_unconverted"] > 0.1
    assert checks["mo_coeff_vs_mokit_unconverted"] > 0.1


def test_the_unconverted_rebuild_is_refused(job):
    """The pre-MOKIT path: .mat orbitals straight into the MOKIT Mole."""
    mol = F.mol_from_fch(job.fch)
    with pytest.raises(H.AOBasisMismatchError, match="rebuild-fock"):
        H.with_rebuilt_fock(job.bundle, mol)


def test_the_unconverted_rebuild_would_have_been_wrong(job):
    """What the refused path computes, measured against the oracle."""
    mol = F.mol_from_fch(job.fch)
    b = job.bundle
    fock_a, _ = H.rebuild_fock(mol, b.C, b.nalpha, b.nbeta, b.Hcore_ao)
    wrong = H.to_mo(fock_a, b.C)
    right = job.pyscf_fock_mo[0]
    assert np.max(np.abs(wrong - right)) > 1e-2


def test_ao_matrices_agree_after_conversion(conversion):
    assert conversion.checks["overlap_after"] < 1e-8
    assert conversion.checks["hcore_after_rel"] < 1e-8


def test_converted_orbitals_are_orthonormal_in_pyscfs_overlap(job, conversion):
    mol = conversion.mol
    mo = conversion.mo_coeff_pyscf(job.bundle.C)
    error = np.max(np.abs(mo.T @ mol.intor("int1e_ovlp") @ mo - np.eye(job.bundle.nmo)))
    assert error == conversion.checks["orthonormality"]
    assert error < 1e-8


def test_converted_orbitals_match_mokits_transfer_of_the_fch(job, conversion):
    """Same physical orbitals, to the nine figures the .fch prints."""
    assert conversion.checks["mo_coeff_vs_mokit"] < FCH_FLOOR
    assert conversion.checks["mo_coeff_sign_flips"] == 0
    assert conversion.checks["density_alpha_vs_mokit"] < FCH_FLOOR
    assert conversion.checks["density_beta_vs_mokit"] < FCH_FLOOR


def test_converted_orbitals_are_the_original_pyscf_orbitals(job, conversion):
    """The whole round trip: PySCF -> Gaussian order -> MOKIT -> PySCF."""
    mo = conversion.mo_coeff_pyscf(job.bundle.C)
    assert np.max(np.abs(mo - job.mf.mo_coeff)) < 1e-12


def test_rebuilt_fock_equals_pyscfs_own_spin_fock(job):
    rebuilt, _ = F.with_rebuilt_fock_from_fch(job.bundle, job.fch)
    for sigma, stored in enumerate((rebuilt.F_alpha_ao, rebuilt.F_beta_ao)):
        mo_fock = H.to_mo(stored, rebuilt.C)
        assert np.max(np.abs(mo_fock - job.pyscf_fock_mo[sigma])) < FCH_FLOOR


def test_provenance_records_the_conversion(job):
    rebuilt, conversion = F.with_rebuilt_fock_from_fch(job.bundle, job.fch)
    record = rebuilt.provenance["fock_rebuild"]
    assert rebuilt.fock_source == "pyscf_rebuilt"
    assert rebuilt.provenance["fock_source"] == "pyscf_rebuilt"
    assert record["replaced_fock_source"] == job.bundle.fock_source
    ao = record["ao_conversion"]
    assert ao["method"].startswith("mokit fch2py")
    assert ao["transform"] == conversion.kind
    assert ao["mokit_version"] == F.mokit_version() != "not installed"
    assert ao["checks"]["overlap_after"] < 1e-8
    # The bundle's own provenance survives.
    assert rebuilt.provenance["source_file"] == job.bundle.provenance["source_file"]
    # And the input is untouched.
    assert job.bundle.fock_source != "pyscf_rebuilt"


def test_cartesian_jobs_need_rescaling_and_pure_ones_do_not(job, conversion):
    if job.spec["cart"]:
        assert "rescaling" in conversion.kind
    else:
        assert conversion.kind == "permutation"


# ------------------------------------------------------ wrong-file failures


def test_an_fch_from_another_basis_is_refused(gaussian_job):
    ours = gaussian_job("ch2_rohf_ccpvdz")
    other = gaussian_job("ch2_rohf_631g")
    with pytest.raises(F.FchError, match="not from the job"):
        F.gaussian_to_pyscf(ours.bundle, other.fch)


def test_an_fch_with_other_orbitals_is_refused(gaussian_job, tmp_path):
    from dataclasses import replace

    job = gaussian_job("ch2_rohf_ccpvdz")
    # Same basis and geometry, orbitals swapped inside the virtual space: the
    # AO matrices still agree, so only the orbital comparison can catch it.
    mo = job.bundle.C.copy()
    mo[:, [-1, -2]] = mo[:, [-2, -1]]
    swapped = replace(job.bundle, C=mo)
    with pytest.raises(F.FchError, match="different orbitals"):
        F.gaussian_to_pyscf(swapped, job.fch)


def test_a_mismatched_ao_matrix_is_refused(gaussian_job):
    from dataclasses import replace

    job = gaussian_job("ch2_rohf_ccpvdz")
    shifted = replace(job.bundle, S=job.bundle.S * (1 - 1e-4) + 1e-4 * np.eye(job.bundle.nao))
    with pytest.raises(F.FchError, match="disagree"):
        F.gaussian_to_pyscf(shifted, job.fch)


def test_mokit_script_names_cannot_collide(gaussian_job, monkeypatch):
    """load_mol_from_fch imports a script named gau<random>; a repeated name used
    to return the previous molecule for a different .fch, silently."""
    import mokit.lib.gaussian as mokit_gaussian

    monkeypatch.setattr(mokit_gaussian.random, "randint", lambda low, high: 4242)
    first = F.mol_from_fch(gaussian_job("h2o_rhf_ccpvtz_cart").fch)
    second = F.mol_from_fch(gaussian_job("ch2_rohf_631g").fch)
    assert first.nao != second.nao
    assert second.nao == gaussian_job("ch2_rohf_631g").mol.nao
