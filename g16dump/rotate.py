"""Unitary rotations inside the active space.

If the active orbitals are replaced by ``|t'> = sum_t |t> U[t, t']`` for a real
orthogonal ``U``, then

    h'  ->  U.T h' U
    (pq|rs) -> sum_abcd U[a,p] U[b,q] U[c,r] U[d,s] (ab|cd)
    E_core  ->  unchanged

``E_core`` is untouched because the rotation mixes active orbitals only: the
frozen core, and therefore the scalar folded out of it, is unaffected. Any
observable of the active-space Hamiltonian -- the FCI spectrum above all -- is
invariant, which is what the tests check.

The four-index transform is done as four successive ``tensordot`` contractions,
O(n^5), never by forming an n^8 intermediate.

The intended use is natural orbitals: diagonalize a 1-RDM from Dice or Block2,
feed the eigenvectors in here, and get a new dump in the natural-orbital basis
without going back to Gaussian.

This matters beyond convenience. Active-space composition and orbital
representation materially change apparent multireference character in
transition-metal systems, so being able to re-express a Hamiltonian in a
different orbital basis -- cheaply, and without a new integral transform -- is
part of the scientific workflow, not a utility.
"""

from __future__ import annotations

import numpy as np

from .errors import ValidationError, require

#: Default tolerance on ``max|U.T U - I|``.
DEFAULT_ORTHOGONALITY_TOLERANCE = 1e-10


def check_orthogonal(u: np.ndarray, tol: float = DEFAULT_ORTHOGONALITY_TOLERANCE) -> float:
    """Return ``max|U.T U - I|``, raising if it exceeds ``tol``.

    A non-orthogonal "rotation" silently changes the spectrum, so this is a
    hard error rather than a warning.
    """
    u = np.asarray(u)
    require(
        u.ndim == 2 and u.shape[0] == u.shape[1],
        f"the rotation matrix must be square, got shape {u.shape}.",
    )
    residual = float(np.max(np.abs(u.T @ u - np.eye(u.shape[0]))))
    require(
        residual <= tol,
        f"the rotation matrix is not orthogonal: max|U.T U - I| = "
        f"{residual:.3e} > {tol:.1e}. A non-orthogonal transformation is not a "
        f"change of orbital basis -- it changes the Hamiltonian's spectrum, so "
        f"the resulting FCIDUMP would describe a different problem. If this "
        f"came from diagonalizing a 1-RDM, re-orthogonalize the eigenvectors "
        f"before using them.",
    )
    return residual


def rotate_one_electron(h: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``U.T h U``."""
    return u.T @ h @ u


def rotate_eri(eri: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Four-index transform of ``(pq|rs)`` under ``u``, as four O(n^5) steps."""
    out = np.tensordot(u, eri, axes=(0, 0))  # p b c d
    out = np.tensordot(u, out, axes=(0, 1)).transpose(1, 0, 2, 3)  # p q c d
    out = np.tensordot(u, out, axes=(0, 2)).transpose(1, 2, 0, 3)  # p q r d
    out = np.tensordot(u, out, axes=(0, 3)).transpose(1, 2, 3, 0)  # p q r s
    return np.ascontiguousarray(out)


def rotate(
    h: np.ndarray,
    eri: np.ndarray,
    e_core: float,
    u: np.ndarray,
    *,
    tol: float = DEFAULT_ORTHOGONALITY_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotate an active-space Hamiltonian. Returns ``(h_rot, eri_rot, e_core)``.

    ``e_core`` is returned unchanged, deliberately: a rotation confined to the
    active space cannot alter the frozen-core scalar.
    """
    h = np.asarray(h)
    eri = np.asarray(eri)
    u = np.asarray(u, dtype=float)

    nact = h.shape[0]
    require(
        u.shape == (nact, nact),
        f"the rotation matrix has shape {u.shape} but the active space has "
        f"{nact} orbitals. Rotations act within the active space only.",
    )
    require(
        eri.shape == (nact,) * 4,
        f"eri has shape {eri.shape}, expected {(nact,) * 4}.",
    )
    check_orthogonal(u, tol)

    return rotate_one_electron(h, u), rotate_eri(eri, u), e_core


def natural_orbital_rotation(rdm1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose a spatial 1-RDM into occupations and a rotation.

    Returns ``(occupations, u)`` with occupations in descending order and ``u``
    the matrix whose columns are the natural orbitals in the current active
    basis -- exactly what :func:`rotate` expects.

    Dice writes this matrix as ``spatialRDM``; Block2 produces the equivalent.
    """
    rdm1 = np.asarray(rdm1)
    require(
        rdm1.ndim == 2 and rdm1.shape[0] == rdm1.shape[1],
        f"the 1-RDM must be square, got shape {rdm1.shape}.",
    )
    asymmetry = float(np.max(np.abs(rdm1 - rdm1.T)))
    require(
        asymmetry <= 1e-8,
        f"the 1-RDM is not symmetric (max|D - D.T| = {asymmetry:.3e}). A "
        f"spatial one-particle density matrix must be symmetric; check that the "
        f"solver's output was read in the right index order.",
    )

    occupations, vectors = np.linalg.eigh(rdm1)
    order = np.argsort(occupations)[::-1]
    return occupations[order], np.ascontiguousarray(vectors[:, order])


def random_orthogonal(n: int, rng=None) -> np.ndarray:
    """A random real orthogonal matrix, for invariance testing."""
    rng = rng if rng is not None else np.random.default_rng()
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    # Fix the sign convention so the result is Haar-distributed.
    return q * np.sign(np.diag(r))
