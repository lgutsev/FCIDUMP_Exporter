"""Real orthogonal rotations inside the active space.

**Seam only. The implementation lands in M4.** The signature is fixed here
because ``cli.py`` calls it; filling it in is a separate piece of work with its
own invariance tests (random orthogonal matrices, an invariant Hamiltonian
spectrum, and identical PySCF FCI ground-state energies before and after).

What the implementation owes, recorded here so it is not rediscovered:

* ``h' -> U.T h' U`` and the matching four-index transform of ``eri_active``.
* ``C[:, bundle.active] -> C[:, bundle.active] @ U``, so that the rotated
  bundle stays self-consistent and can be rotated again.
* ``U`` verified orthogonal to numerical tolerance before anything is
  transformed, and rejected otherwise, since a non-orthogonal ``U`` silently
  changes the spectrum.
* ``E_core``, ``E_ref`` and the electron counts are invariant; a rotation that
  changes them is a bug and should be caught rather than written out.
* The rotation recorded in the bundle's provenance, because orbital
  representation is precisely what the active-space results are sensitive to.
"""

from __future__ import annotations

from .bundle import Bundle


def rotate_active_space(bundle: Bundle, rotation, *, tol: float = 1e-10) -> Bundle:
    """Return a copy of ``bundle`` with its active space rotated by ``rotation``.

    ``rotation`` is a real orthogonal ``(nact, nact)`` matrix. ``tol`` is the
    tolerance on ``U.T U = I``.
    """
    raise NotImplementedError(
        "active-space rotation is M4 and is not implemented yet. The bundle "
        "schema already stores everything it needs: C, the AO-basis Fock "
        "matrices, and eri_active."
    )


__all__ = ["rotate_active_space"]
