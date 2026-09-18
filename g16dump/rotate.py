"""Real orthogonal rotations inside the active space.

Orbital representation is not a cosmetic choice here. The Ni-porphyrin work that
motivated this package found the apparent multireference character to depend
strongly on which orbitals span the active space, so being able to re-express
one active space in another basis, exactly and reversibly, is both a feature and
the sharpest correctness test the package has: the many-body spectrum of the
active-space Hamiltonian is invariant under any orthogonal ``U``, so a rotation
that moves a FCI eigenvalue is a bug with nowhere to hide.

Two things can be rotated, and the difference matters.

:func:`rotate_hamiltonian` rotates the **Hamiltonian** --- ``h' -> U.T h' U``,
the four-index transform of the ERIs, ``E_core`` untouched. This is exact for
any orthogonal ``U``, needs nothing but numpy, and is what a FCIDUMP wants.

:func:`rotate_active_space` rotates the **bundle** --- the orbitals themselves,
so the result is another bundle that can be validated, dumped or rotated again.
This one has a precondition that :func:`rotate_hamiltonian` does not, and it is
the whole subtlety of this module:

    The stored Fock matrices were built from the reference determinant's own
    density. ``hamiltonian.py`` reaches ``h'`` by subtracting the active
    occupied orbitals' J/K back out of ``f^sigma``, which is only the inverse of
    what went in while the occupied *space* is the same one. A ``U`` that mixes
    an active occupied orbital with an active virtual one changes that space, so
    the stored ``f^sigma`` no longer describes the determinant the orbital
    ordering now implies, and the subtraction removes the wrong thing.

The answer is not to patch the Fock matrices into agreement, which would mean
writing a matrix that is not the Fock operator of anything. It is to drop them:
an occupation-changing rotation returns a bundle with ``fock_source="none"``, a
warning, and the reason in its provenance. Downstream, ``active_hamiltonian``
then refuses it and says to rebuild the Fock matrices through PySCF, which is
the correct route for the determinant that now exists. Nothing produces a
plausible wrong number at any point.

For the common case --- rotating the occupied orbitals among themselves, or the
virtuals among themselves, as a localisation or a natural-orbital
transformation does --- the occupied space is unchanged, the Fock matrices are
kept, and ``active_hamiltonian`` on the rotated bundle reproduces
``U.T h' U`` exactly. ``tests/test_rotate.py`` asserts precisely that.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np

from .bundle import Bundle
from .hamiltonian import ActiveHamiltonian, active_energy

#: ``max |U.T U - I|``. The tolerance is tight because ``U`` is an input, not a
#: computed quantity: a caller who means an orthogonal matrix can supply one to
#: machine precision, and a caller who cannot is telling us something.
ORTHOGONALITY_TOL = 1e-10

#: ``max |U[occupied, virtual]|`` below which the active occupied space counts
#: as unchanged and the bundle's Fock matrices stay valid. See the module
#: docstring.
OCCUPATION_TOL = 1e-10


class RotationError(ValueError):
    """The rotation is not a rotation, or does not fit the space it is given."""


class RotationWarning(UserWarning):
    """A rotation was applied, but it cost the bundle something."""


# ------------------------------------------------------------- the matrix


def check_orthogonal(
    rotation, nact: int, *, tol: float = ORTHOGONALITY_TOL, diagnostic: bool = False
) -> np.ndarray:
    """Return ``rotation`` as a validated ``(nact, nact)`` float array.

    Raises :class:`RotationError` unless ``U.T U = I`` to within ``tol``. A
    non-orthogonal ``U`` changes the spectrum of the Hamiltonian instead of
    re-expressing it, which is a different calculation wearing the same name;
    ``diagnostic=True`` permits it explicitly, for the tests and experiments
    that want to watch an invariant break.
    """
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (nact, nact):
        raise RotationError(
            f"rotation has shape {np.asarray(rotation).shape}, expected "
            f"{(nact, nact)} for an active space of {nact} orbitals"
        )
    if not np.all(np.isfinite(matrix)):
        raise RotationError("rotation holds non-finite values")

    deviation = float(np.max(np.abs(matrix.T @ matrix - np.eye(nact))))
    if deviation <= tol:
        return matrix
    if diagnostic:
        warnings.warn(
            f"rotation is not orthogonal (max |U.T U - I| = {deviation:.3e} > "
            f"{tol:.1e}) and was accepted in diagnostic mode. The Hamiltonian "
            f"spectrum is not invariant under it.",
            RotationWarning,
            stacklevel=3,
        )
        return matrix
    raise RotationError(
        f"rotation is not orthogonal: max |U.T U - I| = {deviation:.3e} > "
        f"{tol:.1e}.\n"
        f"Only an orthogonal U re-expresses the active space rather than "
        f"changing it; under anything else the Hamiltonian's eigenvalues move. "
        f"Pass diagnostic=True if breaking that invariant is the point."
    )


def random_orthogonal(nact: int, seed=None) -> np.ndarray:
    """A random real orthogonal ``(nact, nact)`` matrix, from a QR factorisation.

    Deterministic for a given ``seed``, so a test that fails can be reproduced.
    """
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(nact, nact)))
    # Fix the sign convention of the QR so the result is drawn from O(n) evenly
    # rather than favouring whatever LAPACK's Householder choices favour.
    diagonal = np.diag(r)
    return q * np.where(diagonal == 0.0, 1.0, np.sign(diagonal))


# ---------------------------------------------------------- the transforms


def transform_one_body(matrix, rotation) -> np.ndarray:
    """``U.T h U``."""
    u = np.asarray(rotation, dtype=np.float64)
    return u.T @ np.asarray(matrix, dtype=np.float64) @ u


def transform_eri(eri, rotation) -> np.ndarray:
    """The four-index transform ``(tu|vw) = sum_pqrs U_pt U_qu U_rv U_sw (pq|rs)``.

    One index at a time, ``O(n^5)`` rather than the ``O(n^8)`` of contracting
    all four at once, which is the only reason this is usable past a handful of
    orbitals.
    """
    u = np.asarray(rotation, dtype=np.float64)
    out = np.asarray(eri, dtype=np.float64)
    for _ in range(4):
        # Contract the leading axis and let the result rotate into place, so
        # after four passes the axes are back in their original order.
        out = np.tensordot(out, u, axes=([0], [0]))
    return out


# --------------------------------------------------- rotating a Hamiltonian


def rotate_hamiltonian(
    hamiltonian: ActiveHamiltonian,
    rotation,
    *,
    tol: float = ORTHOGONALITY_TOL,
    diagnostic: bool = False,
) -> ActiveHamiltonian:
    """Re-express ``hamiltonian`` in a rotated active-orbital basis.

    Exact for any orthogonal ``U``: ``E_core`` is a property of the frozen core
    and does not move, ``h'`` and the ERIs transform as tensors, and the
    many-body spectrum is therefore unchanged.

    ``e_act`` and ``e_ref`` are recomputed rather than carried over. They are
    properties of the *reference determinant* --- the first ``nocc`` active
    orbitals --- and a ``U`` that mixes occupied with virtual orbitals makes
    that a different determinant with a genuinely different energy. Only
    ``e_ref`` still equals the SCF energy when the occupied space survives the
    rotation; :func:`preserves_occupied_space` is how to ask.
    """
    nact = int(hamiltonian.nact)
    u = check_orthogonal(rotation, nact, tol=tol, diagnostic=diagnostic)

    h_eff = transform_one_body(hamiltonian.h_eff, u)
    eri = transform_eri(hamiltonian.eri_active, u)

    nelec, ms2 = int(hamiltonian.nelec_active), int(hamiltonian.ms2)
    nocc_alpha = (nelec + ms2) // 2
    nocc_beta = nelec - nocc_alpha
    e_act = active_energy(h_eff, eri, nocc_alpha, nocc_beta)

    return replace(
        hamiltonian,
        h_eff=h_eff,
        eri_active=eri,
        e_act=e_act,
        e_ref=hamiltonian.e_core + e_act,
    )


# -------------------------------------------------------- rotating a bundle


def occupation_leakage(bundle: Bundle, rotation) -> float:
    """``max |U[occupied, virtual]|`` over the active occupied blocks.

    Zero when ``U`` is block diagonal between the orbitals the reference
    determinant occupies and the ones it does not --- for either spin, since an
    open-shell reference has two such partitions and both have to survive.
    """
    u = np.asarray(rotation, dtype=np.float64)
    leakage = 0.0
    for nocc in {bundle.nocc_active_alpha, bundle.nocc_active_beta}:
        if 0 < nocc < u.shape[0]:
            leakage = max(
                leakage,
                float(np.max(np.abs(u[:nocc, nocc:]))),
                float(np.max(np.abs(u[nocc:, :nocc]))),
            )
    return leakage


def preserves_occupied_space(
    bundle: Bundle, rotation, *, tol: float = OCCUPATION_TOL
) -> bool:
    """Whether ``rotation`` leaves the reference determinant's occupied space alone."""
    return occupation_leakage(bundle, rotation) <= tol


def rotate_active_space(
    bundle: Bundle,
    rotation,
    *,
    tol: float = ORTHOGONALITY_TOL,
    occupation_tol: float = OCCUPATION_TOL,
    diagnostic: bool = False,
) -> Bundle:
    """Return a copy of ``bundle`` with its active space rotated by ``rotation``.

    ``rotation`` is a real orthogonal ``(nact, nact)`` matrix; ``tol`` is the
    tolerance on ``U.T U = I`` and ``diagnostic=True`` waives it. The orbitals
    and the active ERIs move; the AO-basis Fock matrices, the overlap, the core
    Hamiltonian, the nuclear repulsion and every electron count do not.

    Two things are deliberately not carried through:

    ``orbital energies`` are dropped. Within a rotated window they are the
    diagonal of nothing, and keeping them invites exactly the
    ``diag(orbital_energies)`` mistake this package exists to correct.

    ``Fock matrices`` are dropped, with a :class:`RotationWarning`, when the
    rotation mixes the reference determinant's occupied orbitals with its
    virtual ones --- see the module docstring. The returned bundle is valid and
    can be rotated again, but it needs its Fock matrices rebuilt through
    :func:`g16dump.hamiltonian.with_rebuilt_fock` before it can produce a
    Hamiltonian.
    """
    u = check_orthogonal(rotation, int(bundle.nact), tol=tol, diagnostic=diagnostic)

    mo_coeff = np.array(bundle.C, dtype=np.float64, copy=True)
    mo_coeff[:, bundle.active] = mo_coeff[:, bundle.active] @ u

    provenance = dict(bundle.provenance)
    rotations = list(provenance.get("active_rotations", []))
    rotations.append(
        {
            "nact": int(bundle.nact),
            "window_1based": [int(bundle.active_first), int(bundle.active_last)],
            "determinant": float(np.linalg.det(u)),
            "orthogonality_deviation": float(
                np.max(np.abs(u.T @ u - np.eye(u.shape[0])))
            ),
            "diagnostic": bool(diagnostic),
        }
    )
    provenance["active_rotations"] = rotations
    provenance["orbital_energies_dropped"] = (
        "orbital energies are not defined in a rotated active space"
    )

    changes = {
        "C": mo_coeff,
        "eri_active": transform_eri(bundle.eri_active, u),
        "orbital_energies": None,
        "orbital_energies_beta": None,
        "provenance": provenance,
    }

    leakage = occupation_leakage(bundle, u)
    if leakage > occupation_tol:
        rotations[-1]["occupation_leakage"] = leakage
        reason = (
            f"the rotation mixes the reference determinant's occupied and "
            f"virtual active orbitals (max |U[occ, virt]| = {leakage:.3e}), so "
            f"the stored Fock matrices no longer describe the determinant the "
            f"orbital ordering implies"
        )
        provenance["fock_dropped_reason"] = reason
        changes.update(
            F_alpha_ao=None, F_beta_ao=None, fock_source="none"
        )
        warnings.warn(
            f"{reason}. They have been dropped and fock_source set to 'none'; "
            f"rebuild them from the rotated orbitals with "
            f"g16dump.hamiltonian.with_rebuilt_fock before building a "
            f"Hamiltonian. To rotate the Hamiltonian itself, which is exact for "
            f"any orthogonal U and needs no Fock matrix, use "
            f"g16dump.rotate.rotate_hamiltonian.",
            RotationWarning,
            stacklevel=2,
        )

    return replace(bundle, **changes)


__all__ = [
    "OCCUPATION_TOL",
    "ORTHOGONALITY_TOL",
    "RotationError",
    "RotationWarning",
    "check_orthogonal",
    "occupation_leakage",
    "preserves_occupied_space",
    "random_orthogonal",
    "rotate_active_space",
    "rotate_hamiltonian",
    "transform_eri",
    "transform_one_body",
]
