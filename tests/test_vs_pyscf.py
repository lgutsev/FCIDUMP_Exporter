"""Real-molecule validation against PySCF, the independent oracle.

Two routes to the same active-space Hamiltonian must agree:

1. PySCF's ``CASCI.get_h1eff`` -- a full frozen-core reduction by its own code
   path, plus ``ao2mo`` for the active integrals;
2. ours -- windowed ERIs plus Fock reconstruction.

Agreement is required for ``h'``, the active ERIs, ``E_core`` and the reference
determinant energy, on closed-shell, open-shell, deliberately rotated and
Kohn-Sham orbital sets.

Skipped wholesale when PySCF is absent, so CI never depends on it; the algebra
itself is covered without PySCF by ``test_hamiltonian.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("pyscf")

from pyscf import fci, mcscf  # noqa: E402
from pyscf.tools import fcidump as pyscf_fcidump  # noqa: E402

from fcidump_reader import determinant_energy, read_fcidump  # noqa: E402
from g16dump.bundle import validate  # noqa: E402
from g16dump.errors import ConsistencyError, ReferenceTypeError  # noqa: E402
from g16dump.hamiltonian import (  # noqa: E402
    active_space_hamiltonian,
    effective_one_electron,
    mo_transform,
)
from g16dump.write import write_from_hamiltonian  # noqa: E402
from pyscf_systems import (  # noqa: E402
    block_diagonal_rotation,
    build_molecule,
    bundle_from_scf,
    casci_reference,
    roothaan_effective_fock,
    rotate_orbitals,
    run_scf,
    tight_fci,
)

# name, basis, pyscf spin (na - nb), ref_type, ncore, nact
SYSTEMS = [
    ("h2o_rhf_sto3g", "h2o", "sto-3g", 0, "RHF", 1, 6),
    ("h2o_rhf_631g", "h2o", "6-31g", 0, "RHF", 1, 8),
    ("ch2_rohf_triplet", "ch2", "6-31g", 2, "ROHF", 1, 8),
    ("o2_rohf_triplet", "o2", "6-31g", 2, "ROHF", 2, 8),
    ("nh_rohf_triplet", "nh", "6-31g", 2, "ROHF", 1, 7),
]


@pytest.fixture(scope="module")
def scf_cache():
    return {}


def _get(scf_cache, name, mol_name, basis, spin, ref_type, ncore, nact):
    """Converge each system once per module run; SCF is the slow part."""
    if name not in scf_cache:
        mol = build_molecule(mol_name, basis, spin=spin)
        mf = run_scf(mol, "ROHF" if spin else "RHF")
        bundle = bundle_from_scf(mf, ncore, nact, ref_type)
        scf_cache[name] = (mf, bundle)
    return scf_cache[name]


@pytest.fixture(params=[pytest.param(s, id=s[0]) for s in SYSTEMS])
def system(request, scf_cache):
    name, mol_name, basis, spin, ref_type, ncore, nact = request.param
    mf, bundle = _get(scf_cache, name, mol_name, basis, spin, ref_type, ncore, nact)
    return mf, bundle, ncore, nact


# ------------------------------------------------------------ oracle agreement


def test_bundle_validates(system):
    _, bundle, _, _ = system
    validate(bundle)


def test_reference_energy_equals_the_scf_energy(system):
    """The E_ref gate. This single check would have caught the legacy triplet bug."""
    mf, bundle, _, _ = system
    ash = active_space_hamiltonian(bundle, tol_scf=1e-8)
    assert abs(ash.e_ref - mf.e_tot) < 1e-9


def test_spin_channels_agree(system):
    _, bundle, _, _ = system
    ash = active_space_hamiltonian(bundle, tol_scf=1e-8)
    assert ash.diagnostics["spin_error"] < 1e-9


def test_matches_pyscf_casci_reduction(system):
    """h', E_core and the active ERIs against PySCF's own frozen-core route."""
    mf, bundle, ncore, nact = system
    ash = active_space_hamiltonian(bundle, tol_scf=1e-8)
    h1, e_core, eri = casci_reference(
        mf, ncore, nact, (bundle.nocc_a_act, bundle.nocc_b_act)
    )

    assert np.max(np.abs(ash.h_eff - h1)) < 1e-9
    assert abs(ash.e_core - e_core) < 1e-9
    assert np.max(np.abs(ash.eri_act - eri)) < 1e-9


def test_casci_energy_from_our_fcidump_matches_pyscf(system, tmp_path):
    """The final word: a CASCI energy is invariant to orbital phase and sign.

    Solves the Hamiltonian we wrote and compares to PySCF's CASCI on the same
    orbitals and the same active space.
    """
    mf, bundle, ncore, nact = system
    ash = active_space_hamiltonian(bundle, tol_scf=1e-8)
    nelecas = (bundle.nocc_a_act, bundle.nocc_b_act)

    path = tmp_path / "FCIDUMP"
    write_from_hamiltonian(str(path), ash)
    data = pyscf_fcidump.read(str(path))

    from pyscf import ao2mo

    h1 = np.asarray(data["H1"])
    eri = ao2mo.restore(1, np.asarray(data["H2"]), nact)
    ours = fci.direct_spin1.kernel(h1, eri, nact, nelecas, verbose=0)[0] + float(
        data["ECORE"]
    )

    mc = mcscf.CASCI(mf, nact, nelecas)
    mc.ncore = ncore
    mc.verbose = 0
    theirs = mc.kernel(mf.mo_coeff)[0]

    assert abs(ours - theirs) < 1e-8, f"CASCI energies differ by {ours - theirs:.3e}"


def test_fcidump_round_trip_reproduces_the_scf_reference(system, tmp_path):
    """Read the written file back and rebuild the SCF energy from it."""
    mf, bundle, _, _ = system
    ash = active_space_hamiltonian(bundle, tol_scf=1e-8)

    path = tmp_path / "FCIDUMP"
    write_from_hamiltonian(str(path), ash)
    parsed = read_fcidump(str(path))

    energy = determinant_energy(
        parsed.h1, parsed.eri, parsed.e_core, ash.nocc_a_act, ash.nocc_b_act
    )
    assert abs(energy - mf.e_tot) < 1e-9


# ------------------------------------------------- deliberately rotated orbitals


@pytest.fixture(scope="module")
def rotated_ch2(scf_cache):
    """ROHF triplet CH2 with orbitals mixed inside each occupation block.

    Mixing strictly within the core, doubly-occupied, singly-occupied and
    virtual blocks leaves the reference determinant -- and hence the density,
    the Fock matrices and E_ref -- unchanged, while making the MO Fock matrix
    strongly non-diagonal. That is the regime the legacy approach cannot handle.
    """
    mol = build_molecule("ch2", "6-31g", spin=2)
    mf = run_scf(mol, "ROHF")
    nalpha, nbeta = mol.nelec
    nmo = mf.mo_coeff.shape[1]
    ncore, nact = 1, 8

    u = block_diagonal_rotation(
        nmo, [0, ncore, nbeta, nalpha, nmo], np.random.default_rng(0)
    )
    mo_rot = rotate_orbitals(mf.mo_coeff, u)
    bundle = bundle_from_scf(mf, ncore, nact, "ROHF", mo_coeff=mo_rot)
    return mf, bundle, mo_rot, ncore, nact


def test_rotated_orbitals_give_a_non_diagonal_fock(rotated_ch2):
    """Confirm the test is actually testing something."""
    _, bundle, _, _, _ = rotated_ch2
    f_mo = mo_transform(bundle.f_a_ao, bundle.c_a)
    off_diagonal = f_mo - np.diag(np.diag(f_mo))
    assert np.max(np.abs(off_diagonal)) > 0.1, "rotation did not break diagonality"


def test_rotated_orbitals_match_the_oracle(rotated_ch2):
    """The key scientific test: correct where diag(orbital energies) fails."""
    mf, bundle, mo_rot, ncore, nact = rotated_ch2
    validate(bundle)
    ash = active_space_hamiltonian(bundle, tol_scf=1e-8)
    h1, e_core, eri = casci_reference(
        mf, ncore, nact, (bundle.nocc_a_act, bundle.nocc_b_act), mo_coeff=mo_rot
    )

    assert ash.diagnostics["spin_error"] < 1e-9
    assert abs(ash.e_ref - mf.e_tot) < 1e-9
    assert np.max(np.abs(ash.h_eff - h1)) < 1e-9
    assert abs(ash.e_core - e_core) < 1e-9
    assert np.max(np.abs(ash.eri_act - eri)) < 1e-9


# --------------------------------------------------------- legacy regression


def _legacy_style_h_eff(bundle):
    """h' built the legacy way: diag(f) substituted for the full Fock matrix."""
    f_mo = mo_transform(bundle.f_a_ao, bundle.c_a)
    act = bundle.active
    f_diag = np.diag(np.diag(f_mo))[act, act]
    return effective_one_electron(
        f_diag, bundle.eri_act, bundle.nocc_a_act, bundle.nocc_b_act
    )


def test_legacy_expression_is_correct_for_canonical_rhf(scf_cache):
    """For canonical RHF the Fock matrix *is* diagonal, so the legacy form works.

    This is the regression half of the story: the rebuild must reproduce the old
    closed-shell result exactly where the old result was valid.
    """
    mf, bundle = _get(scf_cache, "h2o_rhf_631g", "h2o", "6-31g", 0, "RHF", 1, 8)
    ash = active_space_hamiltonian(bundle, tol_scf=1e-8)

    f_mo = mo_transform(bundle.f_a_ao, bundle.c_a)
    off_diagonal = np.max(np.abs(f_mo - np.diag(np.diag(f_mo))))
    assert off_diagonal < 1e-7, "canonical RHF Fock should be diagonal"

    assert np.max(np.abs(_legacy_style_h_eff(bundle) - ash.h_eff)) < 1e-7


def test_legacy_expression_fails_for_canonical_rohf(scf_cache):
    """Canonical ROHF orbitals diagonalize the Roothaan operator, not f^alpha.

    So diag(f) is already wrong for a plain ROHF triplet -- no rotation needed.
    """
    mf, bundle = _get(scf_cache, "ch2_rohf_triplet", "ch2", "6-31g", 2, "ROHF", 1, 8)
    ash = active_space_hamiltonian(bundle, tol_scf=1e-8)

    error = np.max(np.abs(_legacy_style_h_eff(bundle) - ash.h_eff))
    assert error > 1e-2, (
        f"expected the legacy diag(f) form to be visibly wrong for ROHF, "
        f"but it differs by only {error:.3e}"
    )


def test_legacy_expression_fails_for_rotated_orbitals(rotated_ch2):
    _, bundle, _, _, _ = rotated_ch2
    ash = active_space_hamiltonian(bundle, tol_scf=1e-8)
    error = np.max(np.abs(_legacy_style_h_eff(bundle) - ash.h_eff))
    assert error > 1e-2


# ----------------------------------------------------- Roothaan effective Fock


def test_roothaan_effective_fock_is_rejected_on_a_real_molecule(scf_cache):
    """PySCF's ROHF get_fock() is the same trap Gaussian sets. It must be caught."""
    mf, bundle = _get(scf_cache, "ch2_rohf_triplet", "ch2", "6-31g", 2, "ROHF", 1, 8)
    roothaan = roothaan_effective_fock(mf)
    bad = dataclasses.replace(bundle, f_a_ao=roothaan, f_b_ao=roothaan)

    with pytest.raises(ConsistencyError) as excinfo:
        active_space_hamiltonian(bad, tol_scf=None)
    assert "Do not average" in str(excinfo.value)


# ---------------------------------------------------------- Kohn-Sham orbitals


@pytest.fixture(scope="module")
def rks_water():
    mol = build_molecule("h2o", "6-31g")
    mf = run_scf(mol, "RKS")
    return mf, bundle_from_scf(mf, 1, 8, "RKS"), 1, 8


def test_ks_orbitals_work_through_the_rebuilt_fock_path(rks_water):
    """The HF Hamiltonian evaluated in KS orbitals must match the oracle."""
    mf, bundle, ncore, nact = rks_water
    validate(bundle)
    assert bundle.fock_source == "rebuilt-pyscf"

    ash = active_space_hamiltonian(bundle, tol_scf=None)
    h1, e_core, eri = casci_reference(
        mf, ncore, nact, (bundle.nocc_a_act, bundle.nocc_b_act)
    )
    assert ash.diagnostics["spin_error"] < 1e-9
    assert np.max(np.abs(ash.h_eff - h1)) < 1e-9
    assert abs(ash.e_core - e_core) < 1e-9
    assert np.max(np.abs(ash.eri_act - eri)) < 1e-9


def test_ks_bundle_does_not_store_the_ks_total_energy(rks_water):
    """e_scf is left unset for KS: the KS energy is not the HF reference energy."""
    mf, bundle, _, _ = rks_water
    assert bundle.e_scf is None
    assert "ks_total_energy" in bundle.provenance

    ash = active_space_hamiltonian(bundle, tol_scf=None)
    # The two genuinely differ -- by the exchange-correlation contribution.
    assert abs(ash.e_ref - mf.e_tot) > 0.1


def test_ks_bundle_on_the_stored_fock_path_is_refused(rks_water):
    _, bundle, _, _ = rks_water
    stored = dataclasses.replace(bundle, fock_source="gaussian")
    with pytest.raises(ReferenceTypeError, match="exchange-correlation"):
        active_space_hamiltonian(stored, tol_scf=None)


# ---------------------------------------------- strongly multireference case


@pytest.fixture(scope="module")
def stretched_n2():
    mol = build_molecule("n2", "cc-pvdz")
    mf = run_scf(mol, "RHF")
    return mf, bundle_from_scf(mf, 2, 8, "RHF")


def test_stretched_n2_matches_the_oracle(stretched_n2):
    """N2 at 2.0 A has genuine multireference character; the algebra must not care."""
    mf, bundle = stretched_n2
    ash = active_space_hamiltonian(bundle, tol_scf=1e-8)
    h1, e_core, eri = casci_reference(
        mf, 2, 8, (bundle.nocc_a_act, bundle.nocc_b_act)
    )
    assert abs(ash.e_ref - mf.e_tot) < 1e-9
    assert np.max(np.abs(ash.h_eff - h1)) < 1e-9
    assert abs(ash.e_core - e_core) < 1e-9


def test_stretched_n2_fci_is_invariant_under_rotation(stretched_n2):
    """The case that exposed PySCF's default Davidson tolerance.

    With default settings the rotated calculation lands ~4.7e-4 Ha *above* the
    original: the HF determinant is a poor guess in a randomly rotated basis
    for a strongly correlated system, and FCI being variational, an
    under-converged energy can only come out high. With tight convergence the
    two agree to ~1e-11.
    """
    from g16dump.rotate import random_orthogonal, rotate

    _, bundle = stretched_n2
    ash = active_space_hamiltonian(bundle, tol_scf=1e-8)
    nelecas = (ash.nocc_a_act, ash.nocc_b_act)

    before = tight_fci(ash.h_eff, ash.eri_act, 8, nelecas)
    u = random_orthogonal(8, np.random.default_rng(1))
    h_rot, eri_rot, _ = rotate(ash.h_eff, ash.eri_act, ash.e_core, u)
    after = tight_fci(h_rot, eri_rot, 8, nelecas)

    assert abs(before - after) < 1e-9
