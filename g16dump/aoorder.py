"""Gaussian <-> PySCF atomic-orbital ordering.

A Gaussian ``.mat`` stores AO-basis quantities -- the core Hamiltonian, the
overlap, the MO coefficients -- in Gaussian's AO order. A molecule rebuilt from
the matching ``.fch`` (via MOKIT's ``load_mol_from_fch``) produces integrals in
PySCF's AO order. For s and p functions the two agree, which is why a test suite
built on 6-31G never notices. For spherical d, f, g... functions they do not:

    Gaussian pure order:  m =  0, +1, -1, +2, -2, ..., +l, -l
    PySCF pure order:     m = -l, ..., -1, 0, +1, ..., +l

Mixing the two gives an overlap against which Gaussian-ordered orbitals are not
even orthonormal (max|C S C.T - I| ~ 1.6 for H2O/6-31G*), and a rebuilt Fock
matrix built from densities in the wrong AO basis. Every transition-metal basis
has d functions, so this is not an edge case.

The permutation here was checked against MOKIT's own reordering (``fch2py``) for
d, f and g functions across 6-31G*, cc-pVTZ, def2-TZVP, cc-pVQZ and Ni/def2-SVP,
matching exactly. ``tests/test_aoorder.py`` pins the convention.

Cartesian functions (Gaussian's 6D/10F) are refused rather than handled: they
differ from PySCF's not only in order but in per-function normalization, and
getting that wrong would be silent. Run Gaussian with ``5D 7F`` instead.
"""

from __future__ import annotations

import numpy as np

from .errors import ValidationError, require


def gaussian_pure_order(l: int) -> list:
    """Magnetic quantum numbers in Gaussian's order for a pure shell, l >= 2.

    s and p need no mapping: s is a single function, and p is x, y, z in both
    codes.
    """
    require(l >= 2, f"pure-shell reordering applies to l >= 2, got l={l}.")
    return [0] + [sign * k for k in range(1, l + 1) for sign in (1, -1)]


def gaussian_to_pyscf_permutation(mol) -> np.ndarray:
    """``perm[g]`` is the PySCF AO index of Gaussian AO ``g``.

    ``mol`` is the PySCF ``Mole`` built from the ``.fch`` -- its shells are in
    the same order as Gaussian's, which is what makes a per-shell permutation
    sufficient.
    """
    require(
        not getattr(mol, "cart", False),
        "this basis uses Cartesian d/f functions (Gaussian 6D/10F). Their "
        "ordering *and* normalization differ between Gaussian and PySCF, and a "
        "mistake there would be silent, so g16dump refuses them. Rerun the "
        "Gaussian job with `5D 7F` in the route section.",
    )

    perm = []
    offset = 0
    for shell in range(mol.nbas):
        l = int(mol.bas_angular(shell))
        for _ in range(int(mol.bas_nctr(shell))):
            if l <= 1:
                # s is one function; p is x, y, z in both codes.
                order = list(range(2 * l + 1))
            else:
                order = [m + l for m in gaussian_pure_order(l)]
            perm.extend(offset + o for o in order)
            offset += 2 * l + 1

    perm = np.asarray(perm, dtype=int)
    require(
        offset == mol.nao and sorted(perm.tolist()) == list(range(mol.nao)),
        f"the AO permutation covers {offset} functions but the molecule has "
        f"{mol.nao}; the basis contains something this mapping does not "
        f"understand.",
    )
    return perm


def matrix_to_gaussian(m_pyscf: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """An AO matrix in PySCF order, re-expressed in Gaussian order."""
    return np.ascontiguousarray(np.asarray(m_pyscf)[np.ix_(perm, perm)])


def matrix_to_pyscf(m_gaussian: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """An AO matrix in Gaussian order, re-expressed in PySCF order."""
    m_gaussian = np.asarray(m_gaussian)
    out = np.empty_like(m_gaussian)
    out[np.ix_(perm, perm)] = m_gaussian
    return out


def coefficients_to_pyscf(c_gaussian: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """MO-major coefficients ``(nmo, nbasis)`` from Gaussian to PySCF AO order."""
    c_gaussian = np.asarray(c_gaussian)
    out = np.empty_like(c_gaussian)
    out[:, perm] = c_gaussian
    return out


def check_same_basis(
    name: str,
    m_gaussian: np.ndarray,
    m_pyscf: np.ndarray,
    perm: np.ndarray,
    rtol: float = 1e-8,
) -> float:
    """Confirm a ``.mat`` AO matrix and its PySCF counterpart describe one basis.

    This is the evidence that the ``.fch`` really belongs to this ``.mat`` and
    that the permutation is right: the same one-electron matrix computed two
    ways must agree once reordered. A wrong ``.fch``, a Cartesian basis that
    slipped through, or a ghost-atom mismatch all show up here as an O(1)
    difference instead of as a wrong Hamiltonian downstream.

    The tolerance is *relative* to the largest element. A ``.fch`` stores basis
    exponents to 9 significant figures, so integrals rebuilt from it differ from
    Gaussian's by ~1e-10 relative -- which for the tight core functions of a
    transition metal (``max|h|`` ~ 400 Ha for Ni/def2-SVP) is ~6e-8 absolute.
    An AO-ordering or wrong-file error is O(1) relative, so ``rtol=1e-8`` keeps
    two orders of magnitude clear of both. Returns the relative difference.
    """
    scale = max(float(np.max(np.abs(m_gaussian))), 1.0)
    difference = float(np.max(np.abs(matrix_to_gaussian(m_pyscf, perm) - m_gaussian))) / scale
    tol = rtol
    if difference > tol:
        raise ValidationError(
            f"{name} from the .mat and the same matrix rebuilt from the .fch "
            f"disagree by {difference:.3e} (relative to its largest element) "
            f"after reordering Gaussian -> PySCF AO order (tolerance {tol:.1e}). Either the .fch is not from the "
            f"same Gaussian job as the .mat, or the two use different AO "
            f"conventions (e.g. Cartesian functions). The .fch cannot be used to "
            f"rebuild anything for this .mat."
        )
    return difference
