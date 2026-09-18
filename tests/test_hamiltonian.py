"""The frozen-core algebra, checked against a brute-force full-space oracle.

The oracle in ``synthetic.py`` implements the textbook reduction directly over
the full MO integral tensor and shares no code with ``g16dump.hamiltonian``.
Agreement between the two is the M2 gate.

The orbitals here are random orthonormal vectors, never an SCF solution, so the
MO Fock matrix is dense and the orbital energies are meaningless. This is exactly
the regime where the legacy ``diag(orbital_energies)`` approach fails.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from g16dump.bundle import validate
from g16dump.errors import ConsistencyError, ReferenceTypeError, ValidationError
from g16dump.hamiltonian import (
    active_reference_energy,
    active_space_hamiltonian,
    effective_one_electron,
    mo_transform,
)
from synthetic import (
    make_system,
    oracle_determinant_energy,
    oracle_e_core,
    oracle_h_eff,
)

# (label, kwargs) covering closed shell, high spin, doublet, frozen-core
# variations and a case with frozen virtuals.
CASES = [
    ("rhf_closed_shell", dict(nalpha=4, nbeta=4, ncore=1, ref_type="RHF", seed=1)),
    ("rohf_triplet", dict(nalpha=5, nbeta=3, ncore=1, ref_type="ROHF", seed=2)),
    ("rohf_doublet", dict(nalpha=4, nbeta=3, ncore=2, ref_type="ROHF", seed=3)),
    ("no_frozen_core", dict(nalpha=4, nbeta=3, ncore=0, ref_type="ROHF", seed=4)),
    ("quintet", dict(nalpha=6, nbeta=2, ncore=1, ref_type="ROHF", seed=5)),
    ("with_frozen_virtuals", dict(nalpha=4, nbeta=3, ncore=1, nact=5, ref_type="ROHF", seed=6)),
    ("larger_basis", dict(nbasis=12, nalpha=6, nbeta=4, ncore=2, ref_type="ROHF", seed=7)),
]
IDS = [label for label, _ in CASES]
KWARGS = [kw for _, kw in CASES]


@pytest.fixture(params=KWARGS, ids=IDS)
def system(request):
    return make_system(**request.param)


def test_bundle_is_self_consistent(system):
    """The synthetic generator must itself produce a valid bundle."""
    validate(system.bundle)


def test_h_eff_matches_full_space_oracle(system):
    """h' from windowed integrals == h' from the full-space frozen-core sum."""
    bundle = system.bundle
    ash = active_space_hamiltonian(bundle, tol_scf=None)
    expected = oracle_h_eff(system.h_mo, system.eri_mo, bundle.ncore, bundle.nact)
    assert np.max(np.abs(ash.h_eff - expected)) < 1e-10


def test_e_core_matches_full_space_oracle(system):
    """E_core == E_nuc + the textbook inactive energy over the frozen core."""
    bundle = system.bundle
    ash = active_space_hamiltonian(bundle, tol_scf=None)
    expected = oracle_e_core(system.h_mo, system.eri_mo, bundle.ncore, bundle.e_nuc)
    assert abs(ash.e_core - expected) < 1e-10


def test_e_ref_matches_slater_rules(system):
    """E_ref from Fock traces == E_ref from bare integrals by Slater's rules."""
    bundle = system.bundle
    ash = active_space_hamiltonian(bundle, tol_scf=None)
    expected = oracle_determinant_energy(
        system.h_mo, system.eri_mo, bundle.nalpha, bundle.nbeta, bundle.e_nuc
    )
    assert abs(ash.e_ref - expected) < 1e-10


def test_core_plus_active_reconstructs_the_reference(system):
    """E_core + E_act == E_ref, exactly. This is what makes E_core meaningful."""
    ash = active_space_hamiltonian(system.bundle, tol_scf=None)
    assert abs((ash.e_core + ash.e_act) - ash.e_ref) < 1e-10


def test_alpha_and_beta_agree(system):
    """h' is spin-independent; the two spins must produce the same matrix."""
    ash = active_space_hamiltonian(system.bundle, tol_scf=None)
    assert ash.diagnostics["spin_error"] < 1e-10


def test_h_eff_is_symmetric(system):
    ash = active_space_hamiltonian(system.bundle, tol_scf=None)
    assert np.max(np.abs(ash.h_eff - ash.h_eff.T)) < 1e-10


# --------------------------------------------------------- full-space limit


def test_full_space_limit_returns_bare_hamiltonian():
    """With no core and everything active, h' = h_MO and E_core = E_nuc."""
    system = make_system(nbasis=8, nalpha=4, nbeta=3, ncore=0, ref_type="ROHF", seed=11)
    bundle = system.bundle
    assert bundle.nact == bundle.nmo

    ash = active_space_hamiltonian(bundle, tol_scf=None)
    assert np.max(np.abs(ash.h_eff - system.h_mo)) < 1e-10
    assert abs(ash.e_core - bundle.e_nuc) < 1e-10


# ------------------------------------------- the closed-shell legacy identity


def test_closed_shell_reduces_to_the_legacy_expression():
    """For RHF the spin sums collapse to -2*J + K, the legacy closed-shell form.

    This is the algebraic claim made in the README, checked numerically. Note it
    holds for *any* restricted orbitals, canonical or not, because it is an
    identity in f -- the legacy code's error was substituting diag(epsilon) for
    f, not the J/K structure itself.
    """
    system = make_system(nbasis=8, nalpha=4, nbeta=4, ncore=1, ref_type="RHF", seed=12)
    bundle = system.bundle
    nocc = bundle.nocc_a_act
    f_act = mo_transform(bundle.f_a_ao, bundle.c_a)[bundle.active, bundle.active]

    ours = effective_one_electron(f_act, bundle.eri_act, nocc, nocc)

    legacy_form = f_act.copy()
    legacy_form -= 2 * np.einsum(
        "acbb->ac", bundle.eri_act[:, :, :nocc, :nocc]
    )
    legacy_form += np.einsum("abbc->ac", bundle.eri_act[:, :nocc, :nocc, :])

    assert np.max(np.abs(ours - legacy_form)) < 1e-12


# ----------------------------------------------------------- failure modes


def test_ks_bundle_is_refused_on_the_stored_fock_path():
    """A KS matrix carries exchange-correlation and is never a Fock operator."""
    system = make_system(nbasis=8, nalpha=4, nbeta=4, ncore=1, ref_type="RHF", seed=13)
    ks = dataclasses.replace(system.bundle, ref_type="RKS", fock_source="gaussian")

    with pytest.raises(ReferenceTypeError, match="exchange-correlation"):
        active_space_hamiltonian(ks, tol_scf=None)


def test_ks_bundle_is_allowed_once_the_fock_is_rebuilt():
    """The same KS orbitals are fine once fock_source says the Fock was rebuilt."""
    system = make_system(nbasis=8, nalpha=4, nbeta=4, ncore=1, ref_type="RHF", seed=13)
    ks = dataclasses.replace(
        system.bundle, ref_type="RKS", fock_source="rebuilt-pyscf"
    )
    ash = active_space_hamiltonian(ks, tol_scf=None)
    expected = oracle_h_eff(system.h_mo, system.eri_mo, ks.ncore, ks.nact)
    assert np.max(np.abs(ash.h_eff - expected)) < 1e-10


def test_missing_fock_is_refused_with_an_actionable_message():
    system = make_system(nbasis=8, nalpha=4, nbeta=3, ncore=1, seed=14)
    stripped = dataclasses.replace(
        system.bundle, f_a_ao=None, f_b_ao=None, fock_source="none"
    )
    with pytest.raises(ValidationError, match="rebuild-fock"):
        active_space_hamiltonian(stripped, tol_scf=None)


def test_roothaan_style_fock_is_caught_not_averaged():
    """The ROHF failure mode: a single effective operator used for both spins.

    Gaussian's ROHF Roothaan operator is not f^alpha or f^beta. Feeding it in
    must fail the spin-agreement check loudly rather than be averaged away.
    """
    system = make_system(nbasis=8, nalpha=5, nbeta=3, ncore=1, ref_type="ROHF", seed=15)
    roothaan = 0.5 * (system.bundle.f_a_ao + system.bundle.f_b_ao)
    bad = dataclasses.replace(system.bundle, f_a_ao=roothaan, f_b_ao=roothaan)

    with pytest.raises(ConsistencyError) as excinfo:
        active_space_hamiltonian(bad, tol_scf=None)

    message = str(excinfo.value)
    assert "Do not average" in message
    assert "Roothaan" in message


def test_spin_tolerance_is_configurable():
    system = make_system(nbasis=8, nalpha=5, nbeta=3, ncore=1, ref_type="ROHF", seed=15)
    roothaan = 0.5 * (system.bundle.f_a_ao + system.bundle.f_b_ao)
    bad = dataclasses.replace(system.bundle, f_a_ao=roothaan, f_b_ao=roothaan)

    # Loose enough to pass, proving the check is what rejects it above.
    ash = active_space_hamiltonian(bad, tol_spin=1e3, tol_scf=None)
    assert ash.diagnostics["spin_error"] > 1e-8


def test_scf_energy_mismatch_is_caught():
    """The E_ref gate: the check that would have caught the legacy triplet bug."""
    system = make_system(nbasis=8, nalpha=5, nbeta=3, ncore=1, ref_type="ROHF", seed=16)
    ash = active_space_hamiltonian(system.bundle, tol_scf=None)

    wrong = dataclasses.replace(system.bundle, e_scf=ash.e_ref + 0.67)
    with pytest.raises(ConsistencyError, match="disagrees with the SCF energy"):
        active_space_hamiltonian(wrong, tol_scf=1e-8)

    right = dataclasses.replace(system.bundle, e_scf=ash.e_ref)
    ok = active_space_hamiltonian(right, tol_scf=1e-8)
    assert ok.diagnostics["scf_error"] < 1e-12


def test_active_reference_energy_handles_empty_spin_channels():
    """A fully spin-polarised active space has no beta electrons; don't crash."""
    system = make_system(nbasis=8, nalpha=5, nbeta=1, ncore=1, ref_type="ROHF", seed=17)
    bundle = system.bundle
    assert bundle.nocc_b_act == 0
    ash = active_space_hamiltonian(bundle, tol_scf=None)
    expected = oracle_h_eff(system.h_mo, system.eri_mo, bundle.ncore, bundle.nact)
    assert np.max(np.abs(ash.h_eff - expected)) < 1e-10
    assert abs(ash.e_core + ash.e_act - ash.e_ref) < 1e-10


def test_active_reference_energy_with_no_electrons_is_zero():
    h = np.eye(3)
    eri = np.zeros((3, 3, 3, 3))
    assert active_reference_energy(h, eri, 0, 0) == 0.0
