"""A synthetic "molecule" with exactly consistent integrals, in NumPy alone.

This exists so the frozen-core algebra can be validated without PySCF, without
Gaussian and without gauopen -- which means it runs everywhere, including CI, and
it is the oracle that has to pass before any real-molecule test is worth running.

The construction is deliberately *not* an SCF solution. The orbitals are random
orthonormal vectors, so the MO Fock matrix has large off-diagonal elements and
the orbital energies are meaningless. That is the point: this is precisely the
regime where the legacy ``diag(orbital_energies)`` approach fails, and where a
correct implementation must not.

Every quantity is mutually consistent by construction:

* ``s_ao`` is a genuine positive-definite metric,
* ``eri_ao`` obeys 8-fold permutational symmetry exactly (it is built in a
  density-fitted form, ``(mn|ls) = sum_P B[P,m,n] B[P,l,s]``),
* ``c`` is orthonormal against ``s_ao``,
* the Fock matrices are built from the reference determinant's own densities,
  which is the assumption the derivation in ``hamiltonian.py`` rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g16dump.bundle import Bundle
from g16dump.hamiltonian import build_fock_dense, density_matrix


@dataclass
class SyntheticSystem:
    """A consistent set of integrals plus the bundle they imply."""

    bundle: Bundle
    eri_ao: np.ndarray  # (nbasis,)*4 full AO integrals, chemist's notation
    eri_mo: np.ndarray  # (nmo,)*4 full MO integrals -- the brute-force oracle
    h_mo: np.ndarray  # (nmo, nmo)
    f_a_mo: np.ndarray
    f_b_mo: np.ndarray


def random_metric(n: int, rng: np.random.Generator) -> np.ndarray:
    """A positive-definite overlap-like matrix with unit diagonal."""
    a = rng.normal(size=(n, n)) * 0.25
    s = a @ a.T + n * np.eye(n)
    d = np.sqrt(np.diag(s))
    return s / np.outer(d, d)


def random_eri(n: int, rng: np.random.Generator, naux: int | None = None) -> np.ndarray:
    """A full AO integral tensor with exact 8-fold permutational symmetry.

    Built as ``(mn|ls) = sum_P B[P,m,n] B[P,l,s]`` with ``B[P]`` symmetric, the
    same structure density fitting produces. This guarantees the symmetry
    exactly rather than by symmetrization after the fact.
    """
    naux = naux or (2 * n)
    b = rng.normal(size=(naux, n, n)) / np.sqrt(naux * n)
    b = 0.5 * (b + b.transpose(0, 2, 1))
    return np.einsum("Pmn,Pls->mnls", b, b, optimize=True)


def random_orbitals(nmo: int, nbasis: int, s_ao: np.ndarray, rng) -> np.ndarray:
    """Random MO-major orbitals, orthonormal against ``s_ao``.

    Returns ``c`` of shape ``(nmo, nbasis)`` with ``c s c.T = I``. Random, hence
    non-canonical: the resulting MO Fock matrix is dense.
    """
    eigval, eigvec = np.linalg.eigh(s_ao)
    s_inv_half = eigvec @ np.diag(eigval**-0.5) @ eigvec.T
    q, _ = np.linalg.qr(rng.normal(size=(nbasis, nbasis)))
    return q[:, :nmo].T @ s_inv_half


def transform_eri(eri_ao: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Full AO -> MO four-index transform, MO-major ``c``.

    The expensive thing this whole project exists to avoid. Used here only to
    build the brute-force oracle on systems small enough not to care.
    """
    tmp = np.tensordot(c, eri_ao, axes=(1, 0))  # p n l s
    tmp = np.tensordot(c, tmp, axes=(1, 1)).transpose(1, 0, 2, 3)  # p q l s
    tmp = np.tensordot(c, tmp, axes=(1, 2)).transpose(1, 2, 0, 3)  # p q r s
    return np.tensordot(c, tmp, axes=(1, 3)).transpose(1, 2, 3, 0)


def make_system(
    *,
    nbasis: int = 8,
    nmo: int | None = None,
    nalpha: int = 4,
    nbeta: int = 3,
    ncore: int = 1,
    nact: int | None = None,
    seed: int = 0,
    ref_type: str = "ROHF",
    e_nuc: float = 7.5,
) -> SyntheticSystem:
    """Build a consistent synthetic system and the bundle describing it."""
    rng = np.random.default_rng(seed)
    nmo = nmo if nmo is not None else nbasis
    nact = nact if nact is not None else nmo - ncore
    nfv = nmo - ncore - nact

    s_ao = random_metric(nbasis, rng)
    h_ao = rng.normal(size=(nbasis, nbasis))
    h_ao = 0.5 * (h_ao + h_ao.T) - 2.0 * np.eye(nbasis)
    eri_ao = random_eri(nbasis, rng)

    c = random_orbitals(nmo, nbasis, s_ao, rng)

    p_a = density_matrix(c, nalpha)
    p_b = density_matrix(c, nbeta)
    f_a_ao, f_b_ao = build_fock_dense(h_ao, eri_ao, p_a, p_b)

    eri_mo = transform_eri(eri_ao, c)
    act = slice(ncore, ncore + nact)
    eri_act = np.ascontiguousarray(eri_mo[act, act, act, act])

    bundle = Bundle(
        ref_type=ref_type,
        fock_source="gaussian",
        charge=0,
        multiplicity=nalpha - nbeta + 1,
        nelec=nalpha + nbeta,
        nalpha=nalpha,
        nbeta=nbeta,
        ncore=ncore,
        nact=nact,
        nfv=nfv,
        nmo=nmo,
        nbasis=nbasis,
        e_nuc=e_nuc,
        h_ao=h_ao,
        s_ao=s_ao,
        c_a=c,
        c_b=c.copy(),
        eri_act=eri_act,
        provenance={"source": "tests/synthetic.py", "seed": seed},
        f_a_ao=f_a_ao,
        f_b_ao=f_b_ao,
    )

    return SyntheticSystem(
        bundle=bundle,
        eri_ao=eri_ao,
        eri_mo=eri_mo,
        h_mo=c @ h_ao @ c.T,
        f_a_mo=c @ f_a_ao @ c.T,
        f_b_mo=c @ f_b_ao @ c.T,
    )


# ------------------------------------------------------- brute-force oracles
#
# These implement the textbook frozen-core reduction directly over the FULL MO
# integral tensor -- the route the whole windowed method exists to avoid. They
# share no code with g16dump.hamiltonian, which is what makes them an oracle.


def oracle_h_eff(h_mo: np.ndarray, eri_mo: np.ndarray, ncore: int, nact: int) -> np.ndarray:
    """``h'_tu = h_tu + sum_{i in core} [ 2 (tu|ii) - (ti|iu) ]``, full-space."""
    act = slice(ncore, ncore + nact)
    h = h_mo[act, act].copy()
    if ncore:
        core = slice(0, ncore)
        h += 2.0 * np.einsum("tuii->tu", eri_mo[act, act, core, core])
        h -= np.einsum("tiiu->tu", eri_mo[act, core, core, act])
    return h


def oracle_e_core(
    h_mo: np.ndarray, eri_mo: np.ndarray, ncore: int, e_nuc: float
) -> float:
    """``E_nuc + 2 sum_i h_ii + sum_ij [ 2 (ii|jj) - (ij|ji) ]`` over the core."""
    if ncore == 0:
        return e_nuc
    core = slice(0, ncore)
    e = e_nuc + 2.0 * float(np.trace(h_mo[core, core]))
    block = eri_mo[core, core, core, core]
    e += 2.0 * float(np.einsum("iijj->", block))
    e -= float(np.einsum("ijji->", block))
    return e


def oracle_determinant_energy(
    h_mo: np.ndarray, eri_mo: np.ndarray, nalpha: int, nbeta: int, e_nuc: float
) -> float:
    """Slater's rules for a single determinant, over the full MO space.

    Completely independent of the Fock-matrix route used by
    ``hamiltonian.reference_energy``: this sums the bare one- and two-electron
    integrals directly.
    """
    e = e_nuc
    for nocc in (nalpha, nbeta):
        e += float(np.trace(h_mo[:nocc, :nocc]))

    for nocc in (nalpha, nbeta):
        if nocc:
            block = eri_mo[:nocc, :nocc, :nocc, :nocc]
            e += 0.5 * float(np.einsum("iijj->", block) - np.einsum("ijji->", block))

    if nalpha and nbeta:
        e += float(np.einsum("iijj->", eri_mo[:nalpha, :nalpha, :nbeta, :nbeta]))

    return e
