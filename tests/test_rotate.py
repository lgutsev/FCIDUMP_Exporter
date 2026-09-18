"""Active-space rotations: algebra, invariants, and FCI invariance.

A rotation inside the active space is a change of basis, so every observable
must be unchanged. The strongest available statement is that the FCI ground
state energy is identical before and after; that is checked here with PySCF
where it is installed, and with an exact dense diagonalization of a small
Hamiltonian otherwise, so the invariance is covered even in a NumPy-only CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from g16dump.errors import ValidationError
from g16dump.hamiltonian import active_space_hamiltonian
from g16dump.rotate import (
    check_orthogonal,
    natural_orbital_rotation,
    random_orthogonal,
    rotate,
    rotate_eri,
    rotate_one_electron,
)
from exact_fci import ground_state_energy
from synthetic import make_system

try:
    from pyscf import fci

    HAVE_PYSCF = True
except ImportError:  # pragma: no cover
    fci = None
    HAVE_PYSCF = False


@pytest.fixture
def built():
    system = make_system(nbasis=8, nalpha=4, nbeta=3, ncore=1, ref_type="ROHF", seed=31)
    return active_space_hamiltonian(system.bundle, tol_scf=None)


# ---------------------------------------------------------------- algebra


def test_identity_rotation_changes_nothing(built):
    u = np.eye(built.h_eff.shape[0])
    h, eri, e_core = rotate(built.h_eff, built.eri_act, built.e_core, u)
    assert np.max(np.abs(h - built.h_eff)) < 1e-13
    assert np.max(np.abs(eri - built.eri_act)) < 1e-13
    assert e_core == built.e_core


def test_rotation_preserves_the_core_energy(built):
    """E_core is untouched: the rotation never leaves the active space."""
    u = random_orthogonal(built.h_eff.shape[0], np.random.default_rng(1))
    _, _, e_core = rotate(built.h_eff, built.eri_act, built.e_core, u)
    assert e_core == built.e_core


def test_rotation_preserves_the_one_electron_spectrum(built):
    """U.T h U is a similarity transform, so the eigenvalues are invariant."""
    u = random_orthogonal(built.h_eff.shape[0], np.random.default_rng(2))
    rotated = rotate_one_electron(built.h_eff, u)
    assert np.allclose(
        np.linalg.eigvalsh(rotated), np.linalg.eigvalsh(built.h_eff), atol=1e-10
    )


def test_rotation_preserves_eri_permutational_symmetry(built):
    u = random_orthogonal(built.h_eff.shape[0], np.random.default_rng(3))
    eri = rotate_eri(built.eri_act, u)
    assert np.max(np.abs(eri - eri.transpose(1, 0, 2, 3))) < 1e-10
    assert np.max(np.abs(eri - eri.transpose(0, 1, 3, 2))) < 1e-10
    assert np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))) < 1e-10


def test_rotations_compose(built):
    """Rotating by U then V equals rotating once by U@V."""
    rng = np.random.default_rng(4)
    n = built.h_eff.shape[0]
    u, v = random_orthogonal(n, rng), random_orthogonal(n, rng)

    h1, eri1, _ = rotate(built.h_eff, built.eri_act, built.e_core, u)
    h2, eri2, _ = rotate(h1, eri1, built.e_core, v)
    h_direct, eri_direct, _ = rotate(built.h_eff, built.eri_act, built.e_core, u @ v)

    assert np.max(np.abs(h2 - h_direct)) < 1e-10
    assert np.max(np.abs(eri2 - eri_direct)) < 1e-10


def test_rotation_is_invertible(built):
    rng = np.random.default_rng(5)
    u = random_orthogonal(built.h_eff.shape[0], rng)
    h, eri, _ = rotate(built.h_eff, built.eri_act, built.e_core, u)
    h_back, eri_back, _ = rotate(h, eri, built.e_core, u.T)
    assert np.max(np.abs(h_back - built.h_eff)) < 1e-10
    assert np.max(np.abs(eri_back - built.eri_act)) < 1e-10


# ------------------------------------------------------------ FCI invariance


@pytest.mark.skipif(not HAVE_PYSCF, reason="pyscf not installed")
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fci_energy_is_invariant_under_rotation(seed):
    """The headline invariance check: FCI energy unchanged by a random rotation."""
    system = make_system(
        nbasis=6, nalpha=3, nbeta=2, ncore=0, nact=6, ref_type="ROHF", seed=40 + seed
    )
    ash = active_space_hamiltonian(system.bundle, tol_scf=None)
    nelec = (ash.nocc_a_act, ash.nocc_b_act)
    norb = ash.h_eff.shape[0]

    e_before = fci.direct_spin1.kernel(
        ash.h_eff, ash.eri_act, norb, nelec, verbose=0
    )[0]

    u = random_orthogonal(norb, np.random.default_rng(seed))
    h_rot, eri_rot, e_core = rotate(ash.h_eff, ash.eri_act, ash.e_core, u)

    e_after = fci.direct_spin1.kernel(h_rot, eri_rot, norb, nelec, verbose=0)[0]

    assert abs(e_before - e_after) < 1e-9, f"FCI energy moved by {e_before - e_after:.3e}"


def test_exact_diagonalization_energy_is_invariant_without_pyscf():
    """Same invariance, by exact diagonalization, so CI covers it without PySCF."""
    system = make_system(
        nbasis=4, nalpha=2, nbeta=1, ncore=0, nact=4, ref_type="ROHF", seed=55
    )
    ash = active_space_hamiltonian(system.bundle, tol_scf=None)

    e_before = ground_state_energy(ash.h_eff, ash.eri_act, ash.nocc_a_act, ash.nocc_b_act)

    u = random_orthogonal(4, np.random.default_rng(9))
    h_rot, eri_rot, _ = rotate(ash.h_eff, ash.eri_act, ash.e_core, u)
    e_after = ground_state_energy(h_rot, eri_rot, ash.nocc_a_act, ash.nocc_b_act)

    assert abs(e_before - e_after) < 1e-9


@pytest.mark.skipif(not HAVE_PYSCF, reason="pyscf not installed")
def test_dense_diagonalization_agrees_with_pyscf_fci():
    """Validate the NumPy-only FCI helper itself against PySCF, once."""
    system = make_system(
        nbasis=4, nalpha=2, nbeta=1, ncore=0, nact=4, ref_type="ROHF", seed=55
    )
    ash = active_space_hamiltonian(system.bundle, tol_scf=None)
    mine = ground_state_energy(ash.h_eff, ash.eri_act, ash.nocc_a_act, ash.nocc_b_act)
    theirs = fci.direct_spin1.kernel(
        ash.h_eff, ash.eri_act, 4, (ash.nocc_a_act, ash.nocc_b_act), verbose=0
    )[0]
    assert abs(mine - theirs) < 1e-9


# ------------------------------------------------------------ natural orbitals


@pytest.mark.skipif(not HAVE_PYSCF, reason="pyscf not installed")
def test_natural_orbitals_diagonalize_the_density():
    """Rotating to natural orbitals must make the 1-RDM diagonal, with the
    eigenvalues as occupations."""
    system = make_system(
        nbasis=6, nalpha=3, nbeta=2, ncore=0, nact=6, ref_type="ROHF", seed=61
    )
    ash = active_space_hamiltonian(system.bundle, tol_scf=None)
    norb = ash.h_eff.shape[0]
    nelec = (ash.nocc_a_act, ash.nocc_b_act)

    _, civec = fci.direct_spin1.kernel(ash.h_eff, ash.eri_act, norb, nelec, verbose=0)
    rdm1 = fci.direct_spin1.make_rdm1(civec, norb, nelec)

    occupations, u = natural_orbital_rotation(rdm1)

    assert np.all(np.diff(occupations) <= 1e-12), "occupations must be descending"
    assert abs(occupations.sum() - sum(nelec)) < 1e-9

    rotated_rdm = u.T @ rdm1 @ u
    off_diagonal = rotated_rdm - np.diag(np.diag(rotated_rdm))
    assert np.max(np.abs(off_diagonal)) < 1e-9
    assert np.allclose(np.diag(rotated_rdm), occupations, atol=1e-9)


@pytest.mark.skipif(not HAVE_PYSCF, reason="pyscf not installed")
def test_natural_orbital_rotation_preserves_the_fci_energy():
    system = make_system(
        nbasis=6, nalpha=3, nbeta=2, ncore=0, nact=6, ref_type="ROHF", seed=61
    )
    ash = active_space_hamiltonian(system.bundle, tol_scf=None)
    norb, nelec = 6, (ash.nocc_a_act, ash.nocc_b_act)

    e_before, civec = fci.direct_spin1.kernel(
        ash.h_eff, ash.eri_act, norb, nelec, verbose=0
    )
    rdm1 = fci.direct_spin1.make_rdm1(civec, norb, nelec)
    _, u = natural_orbital_rotation(rdm1)

    h_rot, eri_rot, _ = rotate(ash.h_eff, ash.eri_act, ash.e_core, u)
    e_after = fci.direct_spin1.kernel(h_rot, eri_rot, norb, nelec, verbose=0)[0]
    assert abs(e_before - e_after) < 1e-9


def test_non_symmetric_rdm_is_refused():
    rdm = np.array([[1.0, 0.5], [0.2, 1.0]])
    with pytest.raises(ValidationError, match="not symmetric"):
        natural_orbital_rotation(rdm)


# -------------------------------------------------------------- bad rotations


def test_non_orthogonal_rotation_is_refused(built):
    n = built.h_eff.shape[0]
    u = np.eye(n)
    u[0, 1] = 0.3  # shears rather than rotates
    with pytest.raises(ValidationError, match="not orthogonal"):
        rotate(built.h_eff, built.eri_act, built.e_core, u)


def test_scaled_rotation_is_refused(built):
    """A scaled orthogonal matrix changes the spectrum; it is not a rotation."""
    n = built.h_eff.shape[0]
    u = 1.01 * random_orthogonal(n, np.random.default_rng(6))
    with pytest.raises(ValidationError, match="not orthogonal"):
        rotate(built.h_eff, built.eri_act, built.e_core, u)


def test_wrong_size_rotation_is_refused(built):
    n = built.h_eff.shape[0]
    with pytest.raises(ValidationError, match="rotations act within|active space"):
        rotate(built.h_eff, built.eri_act, built.e_core, np.eye(n + 1))


def test_non_square_rotation_is_refused():
    with pytest.raises(ValidationError, match="must be square"):
        check_orthogonal(np.ones((2, 3)))


def test_random_orthogonal_is_orthogonal():
    for seed in range(5):
        u = random_orthogonal(7, np.random.default_rng(seed))
        assert check_orthogonal(u) < 1e-12
