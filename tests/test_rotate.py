"""Rotations inside the active space, and the invariances they buy.

A rotation of the active orbitals among themselves cannot change any physical
quantity. That makes it the cheapest strong test available here: the whole
pipeline is run twice on orbitals that differ completely, and every number that
should be invariant is required to be.

The tests are ordered by how much they would hurt to lose:

1. ``E_core``, ``E_ref`` and the electron counts survive a rotation. These come
   from the Fock matrices and the ERIs by different routes, so agreeing after a
   rotation is not a tautology.
2. The spectrum of the active Hamiltonian survives. This is the invariance that
   actually protects a CI or DMRG result, and it is checked directly on the
   FCIDUMP-ready quantities.
3. A rotated bundle is still a bundle: it validates, and it can be rotated
   again. Without that, the rotation is a dead end rather than a transform.
4. A rotation that mixes occupied with virtual active orbitals is refused with
   its cause named. Such a rotation is perfectly orthogonal and looks harmless;
   what it does is replace the reference determinant rather than re-express it,
   because the windowed algebra identifies the occupied active orbitals by index
   order. Left to itself it surfaces two steps later as an alpha/beta
   disagreement, so it is caught where it happens.
5. A non-orthogonal matrix is refused. It would change the spectrum silently,
   which is the one failure mode nothing downstream could catch.

Every invariance test therefore uses ``random_block_rotation``, which mixes
orbitals only within the doubly occupied, singly occupied and virtual groups --
the same partition ``tests/make_fixtures.py`` uses to build the rotated fixture.
A test that used a general rotation would not be testing invariance; it would be
asserting something false.

One test diagonalises the active Hamiltonian with PySCF's FCI solver and is
marked ``pyscf``; the rest run on numpy alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from g16dump import rotate as R
from g16dump.bundle import BundleError, load, validate
from g16dump.hamiltonian import active_hamiltonian
from g16dump.write import write_fcidump

SOUND = [
    "h2o_rhf",
    "ch2_rohf",
    "ch2_rohf_rotated",
    "nh_rohf",
    "h2o_frozen_virtual",
]


@pytest.fixture
def bundles(fixture_path):
    def _load(name):
        return load(fixture_path(name))

    return _load


# ------------------------------------------------------------- the invariances


@pytest.mark.parametrize("name", SOUND)
def test_every_energy_survives_a_rotation(bundles, name):
    """The claim the whole module exists to make."""
    bundle = bundles(name)
    plain = active_hamiltonian(bundle)
    rotated = active_hamiltonian(
        R.rotate_active_space(bundle, R.random_block_rotation(bundle, 4242))
    )

    assert rotated.e_core == pytest.approx(plain.e_core, abs=1e-9)
    assert rotated.e_ref == pytest.approx(plain.e_ref, abs=1e-9)
    assert rotated.e_act == pytest.approx(plain.e_act, abs=1e-9)
    assert rotated.nelec_act == plain.nelec_act
    assert rotated.ms2 == plain.ms2
    assert rotated.nact == plain.nact


@pytest.mark.parametrize("name", SOUND)
def test_the_reference_energy_still_matches_the_scf_energy(bundles, name):
    """Invariance against an outside number, not merely against ourselves."""
    bundle = bundles(name)
    rotated = R.rotate_active_space(bundle, R.random_block_rotation(bundle, 7))
    assert bundle.e_scf is not None
    assert active_hamiltonian(rotated).e_ref == pytest.approx(bundle.e_scf, abs=1e-9)


@pytest.mark.parametrize("name", SOUND)
def test_the_spectrum_of_h_effective_is_invariant(bundles, name):
    """``h' -> U.T h' U`` is a similarity transform, so the eigenvalues hold."""
    bundle = bundles(name)
    plain = active_hamiltonian(bundle).h_eff
    rotated = active_hamiltonian(
        R.rotate_active_space(bundle, R.random_block_rotation(bundle, 11))
    ).h_eff
    np.testing.assert_allclose(
        np.linalg.eigvalsh(rotated), np.linalg.eigvalsh(plain), atol=1e-9
    )


@pytest.mark.parametrize("name", SOUND)
def test_h_effective_transforms_the_way_it_should(bundles, name):
    """Not just invariant in spectrum: equal to ``U.T h' U`` element by element."""
    bundle = bundles(name)
    rotation = R.random_block_rotation(bundle, 99)
    plain = active_hamiltonian(bundle).h_eff
    rotated = active_hamiltonian(R.rotate_active_space(bundle, rotation)).h_eff
    np.testing.assert_allclose(rotated, rotation.T @ plain @ rotation, atol=1e-9)


@pytest.mark.parametrize("name", SOUND)
def test_the_two_electron_invariants_hold(bundles, name):
    """Traces of the ERI tensor that a four-index orthogonal transform preserves."""
    bundle = bundles(name)
    rotated = R.rotate_active_space(bundle, R.random_block_rotation(bundle, 3))
    for eri in (bundle.eri_act, rotated.eri_act):
        assert np.isfinite(eri).all()
    plain_j = np.einsum("ttuu->", bundle.eri_act)
    plain_k = np.einsum("tuut->", bundle.eri_act)
    assert np.einsum("ttuu->", rotated.eri_act) == pytest.approx(plain_j, abs=1e-9)
    assert np.einsum("tuut->", rotated.eri_act) == pytest.approx(plain_k, abs=1e-9)


def test_a_rotated_fcidump_carries_the_same_header_and_core_energy(
    bundles, tmp_path
):
    """What a solver is handed differs only in representation."""
    bundle = bundles("ch2_rohf")
    plain = active_hamiltonian(bundle)
    rotated = active_hamiltonian(
        R.rotate_active_space(bundle, R.random_block_rotation(bundle, 5))
    )
    first = write_fcidump(plain, tmp_path / "a.FCIDUMP")
    second = write_fcidump(rotated, tmp_path / "b.FCIDUMP")

    head = [p.read_text().split("&END")[0] for p in (first, second)]
    assert head[0] == head[1]
    assert len(first.read_text().splitlines()) == len(second.read_text().splitlines())
    assert rotated.e_core == pytest.approx(plain.e_core, abs=1e-9)


# ------------------------------------------------- a rotated bundle is a bundle


@pytest.mark.parametrize("name", SOUND)
def test_a_rotated_bundle_validates(bundles, name):
    bundle = bundles(name)
    validate(R.rotate_active_space(bundle, R.random_block_rotation(bundle, 1)))


@pytest.mark.parametrize("name", SOUND)
def test_rotations_compose(bundles, name):
    """Rotating twice equals rotating once by the product, so it is a real transform."""
    bundle = bundles(name)
    first = R.random_block_rotation(bundle, 21)
    second = R.random_block_rotation(bundle, 22)

    twice = R.rotate_active_space(R.rotate_active_space(bundle, first), second)
    once = R.rotate_active_space(bundle, first @ second)

    np.testing.assert_allclose(twice.mo_coeff, once.mo_coeff, atol=1e-10)
    np.testing.assert_allclose(twice.eri_act, once.eri_act, atol=1e-8)


def test_the_identity_changes_nothing(bundles):
    bundle = bundles("ch2_rohf")
    rotated = R.rotate_active_space(bundle, np.eye(bundle.nact))
    np.testing.assert_allclose(rotated.mo_coeff, bundle.mo_coeff, atol=1e-14)
    np.testing.assert_allclose(rotated.eri_act, bundle.eri_act, atol=1e-12)


def test_rotating_back_recovers_the_original(bundles):
    """``U`` then ``U.T`` is the identity, which is a round trip with no writer."""
    bundle = bundles("ch2_rohf")
    rotation = R.random_block_rotation(bundle, 31)
    there = R.rotate_active_space(bundle, rotation)
    back = R.rotate_active_space(there, rotation.T)
    np.testing.assert_allclose(back.mo_coeff, bundle.mo_coeff, atol=1e-10)
    np.testing.assert_allclose(back.eri_act, bundle.eri_act, atol=1e-8)


def test_the_orbitals_really_moved(bundles):
    """Guard the guard: an invariance test against an unchanged bundle proves nothing."""
    bundle = bundles("ch2_rohf")
    rotated = R.rotate_active_space(bundle, R.random_block_rotation(bundle, 8))
    active = bundle.active
    change = float(
        np.max(np.abs(rotated.mo_coeff[:, active] - bundle.mo_coeff[:, active]))
    )
    assert change > 0.1, f"the rotation barely moved the orbitals ({change:.3e})"
    assert float(np.max(np.abs(rotated.eri_act - bundle.eri_act))) > 1e-3


def test_the_frozen_core_is_untouched(bundles):
    """A rotation is inside the window; a core orbital that moved would be a bug."""
    bundle = bundles("ch2_rohf")
    assert bundle.act_start > 0, "this fixture has no frozen core to check"
    rotated = R.rotate_active_space(bundle, R.random_block_rotation(bundle, 9))
    np.testing.assert_array_equal(
        rotated.mo_coeff[:, : bundle.act_start],
        bundle.mo_coeff[:, : bundle.act_start],
    )


def test_the_frozen_virtuals_are_untouched(bundles):
    """Uses the one fixture that actually has frozen virtuals.

    Every other fixture windows to the top of the MO space
    (``act_stop == nmo``), so asserting on ``mo_coeff[:, act_stop:]`` there
    compares two empty arrays and passes whatever the code does.
    ``h2o_frozen_virtual`` windows MOs 2-10 of 13, leaving three.
    """
    bundle = bundles("h2o_frozen_virtual")
    frozen_count = bundle.nmo - bundle.act_stop
    assert frozen_count == 3, "this test exists for the non-empty slice"

    rotated = R.rotate_active_space(bundle, R.random_block_rotation(bundle, 9))
    frozen = rotated.mo_coeff[:, bundle.act_stop :]
    assert frozen.size > 0
    np.testing.assert_array_equal(frozen, bundle.mo_coeff[:, bundle.act_stop :])


def test_frozen_virtuals_do_not_change_any_energy(bundles):
    """The frozen-virtual path, end to end, against an outside number."""
    bundle = bundles("h2o_frozen_virtual")
    plain = active_hamiltonian(bundle)
    rotated = active_hamiltonian(
        R.rotate_active_space(bundle, R.random_block_rotation(bundle, 41))
    )
    assert plain.e_ref == pytest.approx(bundle.e_scf, abs=1e-9)
    assert rotated.e_ref == pytest.approx(bundle.e_scf, abs=1e-9)
    assert rotated.e_core == pytest.approx(plain.e_core, abs=1e-9)


def test_the_ao_basis_matrices_are_untouched(bundles):
    """They are stored in the AO basis so that a rotation cannot reach them."""
    bundle = bundles("ch2_rohf")
    rotated = R.rotate_active_space(bundle, R.random_block_rotation(bundle, 10))
    np.testing.assert_array_equal(rotated.hcore_ao, bundle.hcore_ao)
    np.testing.assert_array_equal(rotated.overlap, bundle.overlap)
    np.testing.assert_array_equal(rotated.fock_ao_alpha, bundle.fock_ao_alpha)


def test_stale_orbital_energies_are_dropped(bundles):
    """They described the old orbitals; keeping them would be a lie with a number."""
    bundle = bundles("ch2_rohf")
    assert bundle.fock_ao_alpha is not None
    rotated = R.rotate_active_space(bundle, R.random_block_rotation(bundle, 12))
    assert rotated.mo_energy_alpha is None
    assert rotated.mo_energy_beta is None


def test_the_rotation_is_recorded_in_the_provenance(bundles):
    """Orbital representation is exactly what these results are sensitive to."""
    bundle = bundles("ch2_rohf")
    once = R.rotate_active_space(bundle, R.random_block_rotation(bundle, 13))
    twice = R.rotate_active_space(once, R.random_block_rotation(bundle, 14))
    assert once.provenance["rotated"] == 1
    assert twice.provenance["rotated"] == 2
    assert twice.provenance["rotation_orthogonality"] < R.DEFAULT_TOL


# ------------------------------- mixing occupied with virtual is a different thing


@pytest.mark.parametrize("name", SOUND)
def test_the_blocks_partition_the_window(bundles, name):
    bundle = bundles(name)
    blocks = R.reference_blocks(bundle)
    assert blocks[0][0] == 0
    assert blocks[-1][1] == bundle.nact
    for (_, end), (start, _) in zip(blocks, blocks[1:]):
        assert end == start
    covered = sum(hi - lo for lo, hi in blocks)
    assert covered == bundle.nact


def test_a_closed_shell_has_no_singly_occupied_group(bundles):
    assert len(R.reference_blocks(bundles("h2o_rhf"))) == 2


def test_an_open_shell_has_all_three_groups(bundles):
    bundle = bundles("ch2_rohf")
    blocks = R.reference_blocks(bundle)
    assert len(blocks) == 3
    assert blocks[0] == (0, bundle.nocc_act_beta)
    assert blocks[1] == (bundle.nocc_act_beta, bundle.nocc_act_alpha)


@pytest.mark.parametrize("name", SOUND)
def test_mixing_occupied_with_virtual_is_refused_up_front(bundles, name):
    """It replaces the reference determinant rather than re-expressing it.

    A general rotation is orthogonal and looks entirely reasonable. What it
    breaks is the windowed algebra's convention that the occupied active
    orbitals are the lowest-indexed ones, and the consequence surfaces as an
    alpha/beta disagreement two steps later. Catching it here names the cause.
    """
    bundle = bundles(name)
    with pytest.raises(R.RotationError, match="occupation groups"):
        R.rotate_active_space(bundle, R.random_rotation(bundle.nact, 101))


def test_the_refusal_explains_itself_and_names_the_way_out(bundles):
    bundle = bundles("ch2_rohf")
    with pytest.raises(R.RotationError) as excinfo:
        R.rotate_active_space(bundle, R.random_rotation(bundle.nact, 102))
    message = str(excinfo.value)
    assert "replaces it" in message
    assert "random_block_rotation" in message
    assert "allow_reference_change=True" in message


def test_an_opt_in_general_rotation_is_allowed_and_recorded(bundles):
    """A general change of basis is a real thing to want; it just is not invariant."""
    bundle = bundles("ch2_rohf")
    rotated = R.rotate_active_space(
        bundle, R.random_rotation(bundle.nact, 103), allow_reference_change=True
    )
    validate(rotated)
    assert rotated.provenance["rotation_changed_reference"] is True


@pytest.mark.parametrize("name", SOUND)
def test_a_general_rotation_is_refused_downstream_for_every_shell(bundles, name):
    """Refused by an explicit flag, which is the only thing that works everywhere.

    The tempting shortcut is to let the alpha/beta consistency gate catch this.
    It does for an open shell. For a closed shell it cannot: ``F^alpha`` and
    ``F^beta`` are the same matrix, so the two spin-derived ``h'`` agree
    identically no matter how wrong the orbitals are. Before this was an
    explicit check, ``h2o_rhf`` accepted a general rotation with a spin
    deviation of exactly 0.0 and an ``E_ref`` 5.7 Ha off -- the precise failure
    this package exists to prevent. Parametrised over every fixture so the
    closed-shell case can never be the untested one again.
    """
    from g16dump.hamiltonian import HamiltonianError

    bundle = bundles(name)
    rotated = R.rotate_active_space(
        bundle, R.random_rotation(bundle.nact, 104), allow_reference_change=True
    )
    with pytest.raises(HamiltonianError, match="no longer span the occupied space"):
        active_hamiltonian(rotated)


def test_the_closed_shell_spin_gate_really_is_blind_to_this(bundles):
    """Guard the guard: shows why the explicit flag is load-bearing, not belt-and-braces.

    If this ever starts failing -- if the spin deviation becomes non-zero for a
    closed shell -- then the gate could have caught it after all, and the
    reasoning in rotate.py's docstring needs revisiting.
    """
    from g16dump import hamiltonian as H

    bundle = bundles("h2o_rhf")
    rotated = R.rotate_active_space(
        bundle, R.random_rotation(bundle.nact, 104), allow_reference_change=True
    )
    fock_a, fock_b = H.mo_fock_matrices(rotated)
    from_alpha, from_beta = H.effective_one_electron(
        fock_a, fock_b, rotated.eri_act, rotated.active,
        rotated.nocc_act_alpha, rotated.nocc_act_beta,
    )
    assert float(np.max(np.abs(from_alpha - from_beta))) == 0.0


def test_the_refusal_names_only_the_groups_that_exist(bundles):
    """A closed shell has two groups; reciting three names would misdescribe it."""
    bundle = bundles("h2o_rhf")
    assert len(R.reference_blocks(bundle)) == 2
    with pytest.raises(R.RotationError) as excinfo:
        R.rotate_active_space(bundle, R.random_rotation(bundle.nact, 106))
    message = str(excinfo.value)
    assert "doubly occupied" in message and "virtual" in message
    assert "singly occupied" not in message

    open_shell = bundles("ch2_rohf")
    assert len(R.reference_blocks(open_shell)) == 3
    with pytest.raises(R.RotationError) as excinfo:
        R.rotate_active_space(open_shell, R.random_rotation(open_shell.nact, 106))
    assert "singly occupied" in str(excinfo.value)


def test_group_names_follow_the_boundaries_not_the_count(bundles):
    """Two blocks is not one shape, and naming them by counting gets two wrong.

    ``reference_blocks`` drops whichever groups are empty, so a two-block window
    can be (doubly, singly) when it is fully occupied, (doubly, virtual) for a
    closed shell, or (singly, virtual) when the frozen core already takes every
    beta electron. Deducing the names from ``len(blocks) == 2`` labels the last
    two of those incorrectly.
    """
    from dataclasses import replace

    bundle = bundles("ch2_rohf")  # nocc_act_beta=2, nocc_act_alpha=4, nact=12

    closed = replace(bundle, nalpha=3, nbeta=3, nelec=6, multiplicity=1)
    assert [n for _, _, n in R.named_reference_blocks(closed)] == [
        "doubly occupied", "virtual",
    ]

    # ncore == nbeta: no doubly occupied active orbital at all.
    no_doubly = replace(bundle, nbeta=bundle.ncore)
    assert no_doubly.nocc_act_beta == 0
    assert [n for _, _, n in R.named_reference_blocks(no_doubly)] == [
        "singly occupied", "virtual",
    ]

    # The window is entirely occupied: nothing virtual in it.
    full = replace(bundle, nalpha=bundle.ncore + bundle.nact)
    assert full.nocc_act_alpha == bundle.nact
    assert [n for _, _, n in R.named_reference_blocks(full)] == [
        "doubly occupied", "singly occupied",
    ]

    assert R.reference_blocks(bundle) == tuple(
        (lo, hi) for lo, hi, _ in R.named_reference_blocks(bundle)
    )


def test_a_block_rotation_is_reference_preserving_by_construction(bundles):
    bundle = bundles("ch2_rohf")
    rotation = R.random_block_rotation(bundle, 105)
    R.check_reference_preserving(rotation, R.reference_blocks(bundle))
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(bundle.nact), atol=1e-12)
    # and it is not secretly the identity
    assert not np.allclose(rotation, np.eye(bundle.nact))


# --------------------------------------------------------- refusing a bad U


def test_a_non_orthogonal_rotation_is_refused(bundles):
    """It would change the spectrum rather than fail, so it is caught here."""
    bundle = bundles("ch2_rohf")
    bad = R.random_block_rotation(bundle, 15)
    bad[0, 0] += 0.1
    with pytest.raises(R.RotationError, match="not orthogonal"):
        R.rotate_active_space(bundle, bad)


def test_a_scaled_rotation_is_refused(bundles):
    """The most plausible near-miss: orthogonal directions, wrong normalisation."""
    bundle = bundles("ch2_rohf")
    with pytest.raises(R.RotationError, match="not orthogonal"):
        R.rotate_active_space(bundle, 1.01 * R.random_block_rotation(bundle, 16))


@pytest.mark.parametrize(
    "bad, message",
    [
        (np.zeros((3, 3)), "shape"),
        (np.zeros((4,)), "shape"),
        (np.full((12, 12), np.nan), "non-finite"),
    ],
)
def test_a_malformed_rotation_is_refused_by_name(bundles, bad, message):
    with pytest.raises(R.RotationError, match=message):
        R.rotate_active_space(bundles("ch2_rohf"), bad)


def test_the_tolerance_is_configurable_but_not_a_way_out(bundles):
    bundle = bundles("ch2_rohf")
    slightly_off = R.random_block_rotation(bundle, 17)
    slightly_off[0, 0] += 1e-7
    with pytest.raises(R.RotationError):
        R.rotate_active_space(bundle, slightly_off)
    R.rotate_active_space(bundle, slightly_off, tol=1e-5)


def test_random_rotation_is_reproducible_and_orthogonal():
    first = R.random_rotation(9, 2024)
    np.testing.assert_array_equal(first, R.random_rotation(9, 2024))
    np.testing.assert_allclose(first.T @ first, np.eye(9), atol=1e-12)
    assert not np.allclose(first, R.random_rotation(9, 2025))


def test_transform_eri_keeps_the_permutational_symmetry(bundles):
    """An orthogonal transform of a symmetric tensor is still symmetric."""
    bundle = bundles("ch2_rohf")
    out = R.transform_eri(bundle.eri_act, R.random_block_rotation(bundle, 18))
    np.testing.assert_allclose(out, out.transpose(1, 0, 2, 3), atol=1e-10)
    np.testing.assert_allclose(out, out.transpose(0, 1, 3, 2), atol=1e-10)
    np.testing.assert_allclose(out, out.transpose(2, 3, 0, 1), atol=1e-10)


def test_a_rotation_that_breaks_the_schema_fails_here(bundles, monkeypatch):
    """The returned bundle is validated, so a bad rotation cannot be saved later."""
    bundle = bundles("ch2_rohf")
    monkeypatch.setattr(
        R, "transform_eri", lambda eri, rotation: np.full_like(np.asarray(eri), np.nan)
    )
    with pytest.raises(BundleError):
        R.rotate_active_space(bundle, R.random_block_rotation(bundle, 19))


# ------------------------------------------------- the invariance that matters

@pytest.mark.pyscf
@pytest.mark.parametrize("name", ["h2o_rhf", "ch2_rohf"])
def test_the_fci_ground_state_energy_is_unchanged(bundles, name):
    """The strongest form of the claim: the correlated answer does not move.

    ``h'`` and the active ERIs are exactly what a CI solver is handed, so
    diagonalising them before and after a rotation asks the question a user
    actually cares about -- whether the choice of active orbitals changes the
    energy this pipeline reports.
    """
    fci = pytest.importorskip("pyscf.fci")

    bundle = bundles(name)
    plain = active_hamiltonian(bundle)
    rotated = active_hamiltonian(
        R.rotate_active_space(bundle, R.random_block_rotation(bundle, 2718))
    )

    nalpha = (plain.nelec_act + plain.ms2) // 2
    nbeta = plain.nelec_act - nalpha

    energies = []
    for result in (plain, rotated):
        energy, _ = fci.direct_spin1.kernel(
            np.asarray(result.h_eff),
            np.asarray(result.eri_act),
            result.nact,
            (nalpha, nbeta),
            ecore=result.e_core,
        )
        energies.append(float(energy))

    assert energies[1] == pytest.approx(energies[0], abs=1e-8), (
        f"the FCI ground state moved by {abs(energies[1] - energies[0]):.3e} Ha "
        f"under a rotation that cannot change it"
    )
    # And it must be at or below the reference determinant it was built from.
    assert energies[0] <= plain.e_ref + 1e-9
