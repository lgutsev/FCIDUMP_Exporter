"""Active-space rotation gates.

The scientific claim is that an orthogonal ``U`` re-expresses the active space
without changing the physics, so the tests are invariants rather than reference
values: the many-body spectrum, ``E_core``, the electron counts and the
permutational symmetry of the ERIs all have to survive a random rotation.

The strongest of them, :func:`test_the_many_body_spectrum_is_invariant`,
diagonalises the full determinant space before and after and compares every
eigenvalue. It needs neither Gaussian nor PySCF: ``tests/fcidump_oracle.py``
builds the many-electron Hamiltonian from the Slater-Condon rules with numpy
alone, so this runs in the core CI job. The PySCF-marked tests repeat the same
claim on the real fixtures and through a written FCIDUMP.
"""

from __future__ import annotations

import numpy as np
import pytest

from g16dump.bundle import BundleError, load, save, validate
from g16dump.hamiltonian import HamiltonianError, active_hamiltonian
from g16dump.rotate import (
    RotationError,
    RotationWarning,
    check_orthogonal,
    occupation_leakage,
    preserves_occupied_space,
    random_orthogonal,
    rotate_active_space,
    rotate_hamiltonian,
    transform_eri,
    transform_one_body,
)
from g16dump.write import write_fcidump

from fcidump_oracle import exact_spectrum, read_fcidump, spin_occupations

FIXTURES = ("h2o_rhf", "ch2_rohf")


@pytest.fixture
def hamiltonians(fixture_path):
    def _build(name):
        bundle = load(fixture_path(name))
        return bundle, active_hamiltonian(bundle)

    return _build


@pytest.fixture
def bundles(fixture_path):
    def _load(name):
        return load(fixture_path(name))

    return _load


def block_diagonal_rotation(bundle, seed=0):
    """A ``U`` that mixes occupied among occupied and virtual among virtual.

    The occupied space is untouched, so the reference determinant and the stored
    Fock matrices survive. This is the shape a localisation or a natural-orbital
    transformation has.
    """
    nact = bundle.nact
    blocks, start = [], 0
    for boundary in sorted({bundle.nocc_active_beta, bundle.nocc_active_alpha, nact}):
        if boundary > start:
            blocks.append(random_orthogonal(boundary - start, seed=seed + start))
            start = boundary
    rotation = np.zeros((nact, nact))
    offset = 0
    for block in blocks:
        size = block.shape[0]
        rotation[offset:offset + size, offset:offset + size] = block
        offset += size
    return rotation


# ----------------------------------------------------------- the matrix gate


def test_a_non_orthogonal_matrix_is_refused():
    scaled = 1.5 * np.eye(4)
    with pytest.raises(RotationError, match="not orthogonal"):
        check_orthogonal(scaled, 4)


def test_diagnostic_mode_lets_a_non_orthogonal_matrix_through():
    scaled = 1.5 * np.eye(4)
    with pytest.warns(RotationWarning, match="diagnostic mode"):
        assert check_orthogonal(scaled, 4, diagnostic=True) is not None


def test_a_non_orthogonal_rotation_really_does_move_the_spectrum(hamiltonians):
    """Why the gate exists, rather than an assertion that it fires."""
    _, hamiltonian = hamiltonians("h2o_rhf")
    scaled = 1.0001 * random_orthogonal(hamiltonian.nact, seed=3)

    with pytest.warns(RotationWarning):
        rotated = rotate_hamiltonian(hamiltonian, scaled, diagnostic=True)

    before = np.linalg.eigvalsh(hamiltonian.h_eff)
    after = np.linalg.eigvalsh(rotated.h_eff)
    assert not np.allclose(before, after, atol=1e-9)


@pytest.mark.parametrize(
    "rotation, message",
    [
        (np.eye(3), "expected"),
        (np.ones(4), "expected"),
        (np.full((4, 4), np.nan), "non-finite"),
    ],
)
def test_a_misshapen_rotation_is_refused(rotation, message):
    with pytest.raises(RotationError, match=message):
        check_orthogonal(rotation, 4)


def test_random_orthogonal_is_orthogonal_and_reproducible():
    first = random_orthogonal(7, seed=11)
    assert np.allclose(first.T @ first, np.eye(7), atol=1e-13)
    assert np.array_equal(first, random_orthogonal(7, seed=11))
    assert not np.array_equal(first, random_orthogonal(7, seed=12))


# ------------------------------------------------------------ the transforms


def test_the_eri_transform_agrees_with_the_literal_contraction():
    """``tensordot`` four times, against the expression it is an optimisation of."""
    rng = np.random.default_rng(5)
    factors = rng.normal(size=(4, 4, 6))
    factors = 0.5 * (factors + factors.transpose(1, 0, 2))
    eri = np.einsum("tux,vwx->tuvw", factors, factors)
    u = random_orthogonal(4, seed=5)

    assert np.allclose(
        transform_eri(eri, u),
        np.einsum("pqrs,pt,qu,rv,sw->tuvw", eri, u, u, u, u),
        atol=1e-12,
    )


def test_the_transforms_undo_each_other(hamiltonians):
    _, hamiltonian = hamiltonians("ch2_rohf")
    u = random_orthogonal(hamiltonian.nact, seed=2)

    assert np.allclose(
        transform_one_body(transform_one_body(hamiltonian.h_eff, u), u.T),
        hamiltonian.h_eff,
        atol=1e-12,
    )
    assert np.allclose(
        transform_eri(transform_eri(hamiltonian.eri_active, u), u.T),
        hamiltonian.eri_active,
        atol=1e-12,
    )


# ------------------------------------------------- rotating the Hamiltonian


@pytest.mark.parametrize("name", FIXTURES)
def test_a_rotation_preserves_the_scalars_and_the_one_body_spectrum(
    name, hamiltonians
):
    _, hamiltonian = hamiltonians(name)
    u = random_orthogonal(hamiltonian.nact, seed=17)
    rotated = rotate_hamiltonian(hamiltonian, u)

    assert rotated.e_core == hamiltonian.e_core
    assert rotated.nact == hamiltonian.nact
    assert rotated.nelec_active == hamiltonian.nelec_active
    assert rotated.ms2 == hamiltonian.ms2
    assert np.allclose(
        np.linalg.eigvalsh(rotated.h_eff),
        np.linalg.eigvalsh(hamiltonian.h_eff),
        atol=1e-11,
    )


@pytest.mark.parametrize("name", FIXTURES)
def test_a_rotation_preserves_the_tensor_invariants(name, hamiltonians):
    """Traces and norms: cheap diagnostics that move the moment ``U`` is wrong."""
    _, hamiltonian = hamiltonians(name)
    u = random_orthogonal(hamiltonian.nact, seed=23)
    rotated = rotate_hamiltonian(hamiltonian, u)

    def invariants(h):
        eri = np.asarray(h.eri_active)
        return (
            float(np.trace(h.h_eff)),
            float(np.linalg.norm(h.h_eff)),
            float(np.einsum("ttuu->", eri)),
            float(np.einsum("tuut->", eri)),
            float(np.linalg.norm(eri)),
        )

    assert invariants(rotated) == pytest.approx(invariants(hamiltonian), abs=1e-9)


@pytest.mark.parametrize("name", FIXTURES)
def test_a_rotated_eri_keeps_its_permutational_symmetry(name, hamiltonians):
    _, hamiltonian = hamiltonians(name)
    eri = rotate_hamiltonian(
        hamiltonian, random_orthogonal(hamiltonian.nact, seed=29)
    ).eri_active

    assert np.allclose(eri, eri.transpose(1, 0, 2, 3), atol=1e-12)
    assert np.allclose(eri, eri.transpose(0, 1, 3, 2), atol=1e-12)
    assert np.allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-12)


def test_the_identity_rotation_changes_nothing(hamiltonians):
    _, hamiltonian = hamiltonians("ch2_rohf")
    rotated = rotate_hamiltonian(hamiltonian, np.eye(hamiltonian.nact))

    assert np.allclose(rotated.h_eff, hamiltonian.h_eff, atol=1e-14)
    assert np.allclose(rotated.eri_active, hamiltonian.eri_active, atol=1e-14)
    assert rotated.e_ref == pytest.approx(hamiltonian.e_ref, abs=1e-10)


def test_a_rotation_within_the_occupied_space_leaves_the_reference_energy_alone(
    bundles, hamiltonians
):
    bundle, hamiltonian = hamiltonians("ch2_rohf")
    rotated = rotate_hamiltonian(hamiltonian, block_diagonal_rotation(bundle, seed=4))

    assert rotated.e_ref == pytest.approx(hamiltonian.e_ref, abs=1e-10)
    assert rotated.e_act == pytest.approx(hamiltonian.e_act, abs=1e-10)


def test_mixing_occupied_with_virtual_moves_the_reference_determinant(hamiltonians):
    """Not a bug: it is a different determinant, and ``e_ref`` says so."""
    _, hamiltonian = hamiltonians("h2o_rhf")
    rotated = rotate_hamiltonian(
        hamiltonian, random_orthogonal(hamiltonian.nact, seed=31)
    )

    assert rotated.e_core == hamiltonian.e_core
    assert rotated.e_ref != pytest.approx(hamiltonian.e_ref, abs=1e-6)
    assert rotated.e_ref > hamiltonian.e_ref  # the SCF determinant was the lowest


# ------------------------------------------------- the many-body invariance


def test_the_many_body_spectrum_is_invariant():
    """Every FCI eigenvalue, before and after a random rotation. No Gaussian, no PySCF.

    Four active orbitals with three electrons: 24 determinants, small enough to
    diagonalise outright and open-shell enough that the alpha and beta spaces
    differ.
    """
    rng = np.random.default_rng(97)
    nact, nalpha, nbeta, e_core = 4, 2, 1, -12.25
    h_eff = rng.normal(size=(nact, nact))
    h_eff = 0.5 * (h_eff + h_eff.T)
    factors = rng.normal(size=(nact, nact, nact + 2))
    factors = 0.5 * (factors + factors.transpose(1, 0, 2))
    eri = np.einsum("tux,vwx->tuvw", factors, factors)

    u = random_orthogonal(nact, seed=97)
    before = exact_spectrum(h_eff, eri, e_core, nalpha, nbeta)
    after = exact_spectrum(
        transform_one_body(h_eff, u), transform_eri(eri, u), e_core, nalpha, nbeta
    )

    assert np.allclose(before, after, atol=1e-10)


def test_the_many_body_spectrum_moves_under_a_non_orthogonal_matrix():
    """The invariance above is a real constraint, not an artefact of the oracle."""
    rng = np.random.default_rng(98)
    nact, nalpha, nbeta = 4, 2, 1
    h_eff = rng.normal(size=(nact, nact))
    h_eff = 0.5 * (h_eff + h_eff.T)
    factors = rng.normal(size=(nact, nact, nact + 2))
    factors = 0.5 * (factors + factors.transpose(1, 0, 2))
    eri = np.einsum("tux,vwx->tuvw", factors, factors)

    stretched = random_orthogonal(nact, seed=98) @ np.diag([1.0, 1.05, 1.0, 1.0])
    before = exact_spectrum(h_eff, eri, 0.0, nalpha, nbeta)
    after = exact_spectrum(
        transform_one_body(h_eff, stretched),
        transform_eri(eri, stretched),
        0.0,
        nalpha,
        nbeta,
    )

    assert not np.allclose(before, after, atol=1e-6)


def test_a_rotated_hamiltonian_survives_the_round_trip_to_a_file(
    hamiltonians, tmp_path
):
    """Items 8 and 9 together: rotate, write, read back with the foreign parser."""
    _, hamiltonian = hamiltonians("ch2_rohf")
    rotated = rotate_hamiltonian(
        hamiltonian, random_orthogonal(hamiltonian.nact, seed=41)
    )
    parsed = read_fcidump(write_fcidump(rotated, tmp_path / "FCIDUMP"))

    assert parsed["NELEC"] == hamiltonian.nelec_active
    assert parsed["MS2"] == hamiltonian.ms2
    assert spin_occupations(parsed["NELEC"], parsed["MS2"]) == (4, 2)
    assert np.allclose(parsed["H1"], rotated.h_eff, atol=1e-12)
    assert np.allclose(parsed["H2"], rotated.eri_active, atol=1e-12)
    assert parsed["ECORE"] == pytest.approx(hamiltonian.e_core, abs=1e-12)


# ---------------------------------------------------- rotating the bundle


def test_a_rotated_bundle_still_validates(bundles):
    bundle = bundles("ch2_rohf")
    rotated = rotate_active_space(bundle, block_diagonal_rotation(bundle, seed=6))

    validate(rotated)  # raises BundleError if the orbitals stopped being orbitals
    assert rotated.nact == bundle.nact
    assert rotated.nelec == bundle.nelec
    assert np.array_equal(rotated.Hcore_ao, bundle.Hcore_ao)
    assert np.array_equal(rotated.S, bundle.S)
    assert rotated.enuc == bundle.enuc


def test_a_rotated_bundle_reproduces_the_rotated_hamiltonian(bundles):
    """The gate that ties the two rotation paths together.

    Rotating the orbitals and re-deriving ``h'`` from the Fock matrices has to
    land on the same Hamiltonian as rotating ``h'`` directly. It is the check
    that the bundle stayed self-consistent.
    """
    bundle = bundles("ch2_rohf")
    u = block_diagonal_rotation(bundle, seed=8)

    from_bundle = active_hamiltonian(rotate_active_space(bundle, u))
    from_hamiltonian = rotate_hamiltonian(active_hamiltonian(bundle), u)

    assert np.allclose(from_bundle.h_eff, from_hamiltonian.h_eff, atol=1e-10)
    assert np.allclose(from_bundle.eri_active, from_hamiltonian.eri_active, atol=1e-10)
    assert from_bundle.e_core == pytest.approx(from_hamiltonian.e_core, abs=1e-10)
    assert from_bundle.e_ref == pytest.approx(bundle.escf, abs=1e-8)


def test_a_rotated_bundle_can_be_saved_and_loaded(bundles, tmp_path):
    bundle = bundles("h2o_rhf")
    rotated = rotate_active_space(bundle, block_diagonal_rotation(bundle, seed=10))
    reloaded = load(save(rotated, tmp_path / "rotated.npz"))

    assert np.allclose(reloaded.eri_active, rotated.eri_active, atol=1e-14)
    assert np.allclose(reloaded.C, rotated.C, atol=1e-14)
    assert reloaded.provenance["active_rotations"][0]["nact"] == bundle.nact


def test_rotating_back_recovers_the_bundle(bundles):
    bundle = bundles("h2o_rhf")
    u = block_diagonal_rotation(bundle, seed=12)
    there_and_back = rotate_active_space(rotate_active_space(bundle, u), u.T)

    assert np.allclose(there_and_back.C, bundle.C, atol=1e-11)
    assert np.allclose(there_and_back.eri_active, bundle.eri_active, atol=1e-11)
    assert len(there_and_back.provenance["active_rotations"]) == 2


def test_only_the_active_orbitals_move(bundles):
    bundle = bundles("ch2_rohf")
    rotated = rotate_active_space(bundle, block_diagonal_rotation(bundle, seed=14))

    core = slice(0, bundle.ncore)
    frozen_virtual = slice(bundle.active_stop, bundle.nmo)
    assert np.array_equal(rotated.C[:, core], bundle.C[:, core])
    assert np.array_equal(
        rotated.C[:, frozen_virtual], bundle.C[:, frozen_virtual]
    )
    assert not np.allclose(
        rotated.C[:, bundle.active], bundle.C[:, bundle.active]
    )


def test_the_ao_fock_matrices_are_left_alone(bundles):
    """They are AO-basis operators; an MO rotation is not theirs to feel."""
    bundle = bundles("ch2_rohf")
    rotated = rotate_active_space(bundle, block_diagonal_rotation(bundle, seed=16))

    assert rotated.fock_source == bundle.fock_source
    assert np.array_equal(rotated.F_alpha_ao, bundle.F_alpha_ao)
    assert np.array_equal(rotated.F_beta_ao, bundle.F_beta_ao)


def test_orbital_energies_are_dropped(bundles):
    """Within a rotated window they are the diagonal of nothing; see rotate.py."""
    bundle = bundles("h2o_rhf")
    assert bundle.orbital_energies is not None

    rotated = rotate_active_space(bundle, block_diagonal_rotation(bundle, seed=18))
    assert rotated.orbital_energies is None
    assert rotated.orbital_energies_beta is None
    assert "orbital_energies_dropped" in rotated.provenance


def test_provenance_is_extended_not_replaced(bundles):
    bundle = bundles("ch2_rohf")
    assert bundle.provenance  # otherwise this test proves nothing

    rotated = rotate_active_space(bundle, block_diagonal_rotation(bundle, seed=20))
    for key, value in bundle.provenance.items():
        assert rotated.provenance[key] == value

    record = rotated.provenance["active_rotations"][-1]
    assert record["nact"] == bundle.nact
    assert abs(record["determinant"]) == pytest.approx(1.0, abs=1e-10)


# --------------------------------------- the occupied-space precondition


def test_occupation_leakage_sees_the_difference(bundles):
    bundle = bundles("ch2_rohf")
    assert preserves_occupied_space(bundle, block_diagonal_rotation(bundle, seed=22))
    assert preserves_occupied_space(bundle, np.eye(bundle.nact))
    assert occupation_leakage(bundle, random_orthogonal(bundle.nact, seed=22)) > 1e-3


def test_mixing_occupied_with_virtual_drops_the_fock_matrices(bundles):
    """The stored Fock matrices describe the old determinant; see rotate.py."""
    bundle = bundles("ch2_rohf")
    u = random_orthogonal(bundle.nact, seed=24)

    with pytest.warns(RotationWarning, match="occupied and virtual"):
        rotated = rotate_active_space(bundle, u)

    validate(rotated)
    assert rotated.fock_source == "none"
    assert rotated.F_alpha_ao is None and rotated.F_beta_ao is None
    assert "fock_dropped_reason" in rotated.provenance
    assert rotated.provenance["active_rotations"][-1]["occupation_leakage"] > 1e-3


def test_a_bundle_whose_fock_was_dropped_refuses_to_build_a_hamiltonian(bundles):
    """No silent wrong number: the next step says what is missing and why."""
    bundle = bundles("ch2_rohf")
    with pytest.warns(RotationWarning):
        rotated = rotate_active_space(bundle, random_orthogonal(bundle.nact, seed=26))

    with pytest.raises(HamiltonianError, match="no Fock matrix"):
        active_hamiltonian(rotated)


def test_an_occupation_preserving_rotation_warns_about_nothing(bundles, recwarn):
    bundle = bundles("ch2_rohf")
    rotate_active_space(bundle, block_diagonal_rotation(bundle, seed=28))
    assert [w for w in recwarn if issubclass(w.category, RotationWarning)] == []


def test_a_non_orthogonal_rotation_of_a_bundle_is_refused(bundles):
    bundle = bundles("h2o_rhf")
    with pytest.raises(RotationError, match="not orthogonal"):
        rotate_active_space(bundle, 2.0 * np.eye(bundle.nact))


def test_a_rotation_sized_for_the_wrong_space_is_refused(bundles):
    bundle = bundles("h2o_rhf")
    with pytest.raises(RotationError, match="expected"):
        rotate_active_space(bundle, np.eye(bundle.nmo))


def test_a_rotated_bundle_that_lost_its_fock_still_fails_validation_loudly(bundles):
    """A dropped Fock matrix must not be mistaken for a Gaussian-sourced one."""
    bundle = bundles("ch2_rohf")
    with pytest.warns(RotationWarning):
        rotated = rotate_active_space(bundle, random_orthogonal(bundle.nact, seed=30))

    from dataclasses import replace

    with pytest.raises(BundleError, match="no F_alpha_ao"):
        validate(replace(rotated, fock_source="gaussian"))


# ---------------------------------------------------------------- via PySCF


@pytest.mark.pyscf
@pytest.mark.parametrize("name", FIXTURES)
def test_pyscf_fci_is_invariant_under_a_random_rotation(
    name, hamiltonians, tmp_path
):
    """The same invariance on the real fixtures, through PySCF end to end."""
    from fcidump_oracle import fci_energy_from_fcidump

    _, hamiltonian = hamiltonians(name)
    rotated = rotate_hamiltonian(
        hamiltonian, random_orthogonal(hamiltonian.nact, seed=43)
    )

    before = fci_energy_from_fcidump(write_fcidump(hamiltonian, tmp_path / "plain"))
    after = fci_energy_from_fcidump(write_fcidump(rotated, tmp_path / "rotated"))

    assert after == pytest.approx(before, abs=1e-9)


@pytest.mark.pyscf
def test_pyscf_fci_is_invariant_under_a_rotation_of_the_bundle(bundles, tmp_path):
    """The bundle path, all the way from orbitals to a solver's input file."""
    from fcidump_oracle import fci_energy_from_fcidump

    bundle = bundles("ch2_rohf")
    rotated = rotate_active_space(bundle, block_diagonal_rotation(bundle, seed=45))

    before = fci_energy_from_fcidump(
        write_fcidump(active_hamiltonian(bundle), tmp_path / "plain")
    )
    after = fci_energy_from_fcidump(
        write_fcidump(active_hamiltonian(rotated), tmp_path / "rotated")
    )

    assert after == pytest.approx(before, abs=1e-9)
    assert before < bundle.escf
