"""Real orthogonal rotations inside the active space.

A rotation ``U`` mixes the active orbitals among themselves and nothing else.
The physics cannot notice: the active space spans the same subspace afterwards,
so every energy -- ``E_core``, ``E_ref``, and the FCI ground state of the
resulting FCIDUMP -- is unchanged. What does change is the *representation*, and
that is the point. Active-space results are sensitive to orbital choice in ways
that are easy to get wrong and hard to see, so a rotation is the cheapest
available test of a pipeline that claims not to care.

What is transformed, and what is deliberately not:

``mo_coeff[:, active] -> mo_coeff[:, active] @ U``
    The orbitals themselves. Doing this, rather than transforming the Fock
    matrices, is what keeps the bundle self-consistent: a rotated bundle is an
    ordinary bundle and can be rotated again, or fed to
    :func:`g16dump.hamiltonian.active_hamiltonian` unchanged.

``eri_act -> U' U' U' U' eri_act``
    The four-index transform, over the active window only. It is the one piece
    of real work here, and it stays inside the window because that is the whole
    premise of the method.

The AO-basis Fock matrices, the overlap and ``hcore_ao`` are **not** touched.
They are stored in the AO basis precisely so that a rotation does not reach
them; ``hamiltonian.py`` does the ``C.T F C`` transform afterwards and so sees
the rotation through ``mo_coeff`` alone. Touching them here would double-count
it.

``U`` is verified orthogonal before anything is transformed. A ``U`` that is not
orthogonal silently changes the spectrum rather than failing, which is the worst
failure this module could have.

Which rotations are allowed, and why it is not all of them
---------------------------------------------------------

The windowed algebra reaches ``h'`` from the Fock matrices, which means
subtracting the contribution of the active orbitals occupied in the reference
determinant -- and it identifies those *by index order*, as the lowest
``n_sigma - ncore`` of the window. That convention is what
:func:`g16dump.bundle.Bundle.nocc_act_alpha` encodes.

A rotation that mixes an occupied active orbital with a virtual one therefore
does not merely re-express the reference determinant, it **replaces** it: the
lowest orbitals of the rotated window no longer span the occupied space, so
``E_ref`` and ``E_core`` are not the same quantities afterwards. The alpha/beta
consistency gate in :mod:`g16dump.hamiltonian` notices and refuses, which is
correct but reports the problem one step downstream of its cause.

So by default a rotation must be block diagonal across the three groups the
reference determinant distinguishes inside the window -- doubly occupied, singly
occupied, virtual -- and :func:`rotate_active_space` checks that up front. Those
are exactly the rotations that leave every energy invariant, which is what makes
them a test. Pass ``allow_reference_change=True`` to perform a general change of
active-orbital basis anyway; it is a legitimate thing to want, and the resulting
bundle will be refused by ``active_hamiltonian`` for the reason above.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .bundle import Bundle, validate

#: Rotating a rotated bundle composes, so this is not the accuracy of one
#: transform but the drift a chain of them may accumulate before it is refused.
DEFAULT_TOL = 1e-10


class RotationError(ValueError):
    """The rotation is not a real orthogonal matrix on the active space."""


def check_orthogonal(rotation, nact: int, tol: float = DEFAULT_TOL) -> np.ndarray:
    """Return ``rotation`` as a float array, or raise :class:`RotationError`.

    Checked rather than assumed: a non-orthogonal ``U`` produces a Hamiltonian
    with a different spectrum and no error anywhere, so every caller would
    otherwise be one typo away from a plausible wrong answer.
    """
    matrix = np.asarray(rotation, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape != (nact, nact):
        raise RotationError(
            f"rotation has shape {matrix.shape}, expected ({nact}, {nact}): one "
            f"row and column per active orbital"
        )
    if not np.all(np.isfinite(matrix)):
        raise RotationError("rotation holds non-finite values")

    deviation = float(np.max(np.abs(matrix.T @ matrix - np.eye(nact))))
    if deviation > tol:
        raise RotationError(
            f"rotation is not orthogonal: max |U.T U - I| = {deviation:.3e}, "
            f"above the tolerance {tol:.1e}.\n"
            f"A non-orthogonal U changes the spectrum of the Hamiltonian "
            f"instead of merely re-expressing it, and nothing downstream would "
            f"report that. If this came from an eigendecomposition or a QR, "
            f"re-orthonormalise it before rotating."
        )
    return matrix


def reference_blocks(bundle: Bundle) -> tuple[tuple[int, int], ...]:
    """The groups inside the window that the reference determinant distinguishes.

    ``(doubly occupied, singly occupied, virtual)`` as half-open index ranges
    into the active window. A rotation may mix orbitals freely within any one of
    them and not at all between them. Empty groups are dropped, so a closed
    shell gives two.
    """
    edges = (0, bundle.nocc_act_beta, bundle.nocc_act_alpha, bundle.nact)
    return tuple(
        (lo, hi) for lo, hi in zip(edges, edges[1:]) if hi > lo
    )


def check_reference_preserving(
    rotation: np.ndarray, blocks, tol: float = DEFAULT_TOL
) -> None:
    """Raise unless ``rotation`` is block diagonal over ``blocks``.

    The check is on the off-block elements rather than on what they do, because
    the consequence -- a different reference determinant -- is not something the
    caller can inspect afterwards.
    """
    leak, where = 0.0, None
    for lo, hi in blocks:
        outside = np.ones(rotation.shape[0], dtype=bool)
        outside[lo:hi] = False
        if not outside.any():
            continue
        block_leak = float(np.max(np.abs(rotation[outside, lo:hi])))
        if block_leak > leak:
            leak, where = block_leak, (lo, hi)

    if leak > tol:
        groups = ", ".join(f"[{lo + 1}-{hi}]" for lo, hi in blocks)
        raise RotationError(
            f"rotation mixes orbitals across the reference determinant's "
            f"occupation groups: {leak:.3e} of block {where[0] + 1}-{where[1]} "
            f"leaks outside it.\n"
            f"Inside the window the groups are (1-based) {groups}: doubly "
            f"occupied, singly occupied, virtual.\n"
            f"Mixing across them does not re-express the reference "
            f"determinant, it replaces it. The windowed algebra identifies the "
            f"occupied active orbitals by index order, so E_ref and E_core stop "
            f"meaning what they meant and the alpha/beta gate will reject the "
            f"result.\n"
            f"Use g16dump.rotate.random_block_rotation for a rotation that "
            f"leaves every energy invariant, or pass "
            f"allow_reference_change=True if a general change of basis really "
            f"is what you want."
        )


def transform_eri(eri_act: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """The four-index transform of the active ERIs, ``(tu|vw)`` in chemist order.

    Done as four successive two-index contractions rather than one einsum over
    eight indices: the same answer at ``O(nact**5)`` instead of ``O(nact**8)``.
    """
    out = np.asarray(eri_act, dtype=np.float64)
    for axis in range(4):
        out = np.tensordot(out, rotation, axes=([axis], [0]))
        out = np.moveaxis(out, -1, axis)
    return out


def rotate_active_space(
    bundle: Bundle,
    rotation,
    *,
    tol: float = DEFAULT_TOL,
    allow_reference_change: bool = False,
) -> Bundle:
    """Return a copy of ``bundle`` with its active space rotated by ``rotation``.

    ``rotation`` is a real orthogonal ``(nact, nact)`` matrix. ``tol`` is the
    tolerance on ``U.T U = I`` and on the off-block leakage.

    By default the rotation must preserve the reference determinant's occupation
    groups; see the module docstring for why, and
    :func:`check_reference_preserving` for what is checked.
    ``allow_reference_change=True`` lifts that and performs a general change of
    active-orbital basis, which is a real operation but produces a bundle
    ``active_hamiltonian`` will refuse.

    The returned bundle is validated before it is handed back, so a rotation
    that produced something the schema would reject fails here rather than at
    the next ``load``.
    """
    matrix = check_orthogonal(rotation, bundle.nact, tol)
    if not allow_reference_change:
        check_reference_preserving(matrix, reference_blocks(bundle), tol)

    mo_coeff = np.array(bundle.mo_coeff, copy=True)
    mo_coeff[:, bundle.active] = mo_coeff[:, bundle.active] @ matrix

    provenance = dict(bundle.provenance)
    provenance["rotated"] = int(provenance.get("rotated", 0)) + 1
    provenance["rotation_orthogonality"] = float(
        np.max(np.abs(matrix.T @ matrix - np.eye(bundle.nact)))
    )
    if allow_reference_change:
        provenance["rotation_changed_reference"] = True

    rotated = replace(
        bundle,
        mo_coeff=mo_coeff,
        eri_act=transform_eri(bundle.eri_act, matrix),
        # The orbitals are no longer the ones these energies described, and a
        # stale diagnostic is worse than an absent one.
        mo_energy_alpha=None,
        mo_energy_beta=None,
        provenance=provenance,
    )
    validate(rotated, source="rotated bundle")
    return rotated


def random_rotation(nact: int, seed: int) -> np.ndarray:
    """A reproducible random orthogonal matrix, for tests and for probing.

    The sign fix on the ``R`` diagonal makes the QR decomposition unique, so the
    same seed gives the same matrix on every platform and numpy version -- which
    an invariance test that is supposed to be reproducible needs.
    """
    generator = np.random.default_rng(seed)
    q, r = np.linalg.qr(generator.normal(size=(nact, nact)))
    return q * np.sign(np.diag(r))


def random_block_rotation(bundle: Bundle, seed: int) -> np.ndarray:
    """A random rotation that leaves ``bundle``'s reference determinant alone.

    Block diagonal over :func:`reference_blocks`, so it mixes orbitals only
    within the doubly occupied, singly occupied and virtual groups. This is the
    rotation an invariance test wants: it changes the orbitals completely and
    every energy not at all.

    A group of one orbital cannot be rotated, so it is left as it is rather than
    multiplied by a random sign -- a sign flip is orthogonal and harmless, but
    it makes "the orbitals moved" harder to assert honestly.
    """
    generator = np.random.default_rng(seed)
    matrix = np.eye(bundle.nact)
    for lo, hi in reference_blocks(bundle):
        size = hi - lo
        if size < 2:
            continue
        q, r = np.linalg.qr(generator.normal(size=(size, size)))
        matrix[lo:hi, lo:hi] = q * np.sign(np.diag(r))
    return matrix


__all__ = [
    "DEFAULT_TOL",
    "RotationError",
    "check_orthogonal",
    "check_reference_preserving",
    "random_block_rotation",
    "random_rotation",
    "reference_blocks",
    "rotate_active_space",
    "transform_eri",
]
