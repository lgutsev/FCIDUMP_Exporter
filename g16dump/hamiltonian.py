"""Frozen-core reduction to an active-space Hamiltonian.

One implementation serves RHF and ROHF. There is no closed-shell special case
and no spin-coupling coefficients; the spin dependence lives entirely in which
orbitals are occupied for each spin.

The derivation
--------------
Let ``C`` be the frozen core (doubly occupied), ``A`` the active window, and
``O_s`` the active orbitals occupied by spin ``s`` in the reference determinant.
The Fock operator built from the *reference determinant's* densities is

    f^s = h + J[P^a + P^b] - K[P^s]

Writing its active block and splitting each occupied sum into core and active
parts gives the frozen-core effective one-electron Hamiltonian

    h'_tu = h_tu + sum_{i in C} [ 2 (tu|ii) - (ti|iu) ]
          = f^s_tu - sum_{k in O_s} [ (tu|kk) - (tk|ku) ] - sum_{k in O_s'} (tu|kk)

where ``s'`` is the opposite spin. The second form is what we use, because it
needs only the active-window integrals -- never an integral with a core index --
which is the entire point of the method.

``h'`` does not depend on spin. Evaluating the right-hand side for both spins
therefore gives a free and rather strong check on the input: if the two disagree,
the stored matrix is not the UHF-type Fock operator the derivation assumes
(Gaussian's ROHF Roothaan effective operator is the usual culprit). That is a
bug signal. **The two are never averaged.**

Energies
--------
For any single determinant,

    E_ref = E_nuc + 1/2 sum_s sum_{i in occupied_s} ( h_ii + f^s_ii )
    E_act = sum_s sum_{t in O_s} h'_tt
          + 1/2 sum_s sum_{t,u in O_s} [ (tt|uu) - (tu|ut) ]
          + sum_{t in O_a, u in O_b} (tt|uu)
    E_core = E_ref - E_act

so that ``E_core`` plus the active-space energy of the reference determinant
reproduces ``E_ref`` exactly. ``E_core`` is what goes on the FCIDUMP's
``0 0 0 0`` line.

Closed-shell limit
------------------
For RHF, ``O_a = O_b = O`` and ``f^a = f^b = f``, so the two active sums
collapse:

    h'_tu = f_tu - 2 sum_{k in O} (tu|kk) + sum_{k in O} (tk|ku)

which is exactly the expression in ``legacy/FCIDUMP_Write_MOe.py``, differing
only in that ``f_tu`` is the full Fock matrix here and ``diag(orbital energies)``
there. Those agree only for canonical RHF orbitals; ``test_vs_pyscf.py``
demonstrates both the agreement and, for rotated orbitals, the failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .bundle import Bundle
from .errors import (
    ConsistencyError,
    MissingDependencyError,
    ReferenceTypeError,
    SchemaError,
    ValidationError,
    require,
)

#: Default tolerance for max|h'(alpha) - h'(beta)|, in Hartree.
DEFAULT_SPIN_TOLERANCE = 1e-8


@dataclass
class ActiveSpaceHamiltonian:
    """The active-space Hamiltonian and the numbers that prove it is sound."""

    h_eff: np.ndarray  # (nact, nact) effective one-electron Hamiltonian
    eri_act: np.ndarray  # (nact,)*4 chemist's (pq|rs)
    e_core: float  # scalar frozen-core energy, incl. nuclear repulsion
    e_ref: float  # energy of the reference determinant
    e_act: float  # active-space part of e_ref
    nelec_act: int
    ms2: int
    nocc_a_act: int
    nocc_b_act: int
    diagnostics: dict = field(default_factory=dict)


# ------------------------------------------------------------ basic transforms


def mo_transform(m_ao: np.ndarray, c: np.ndarray) -> np.ndarray:
    """AO -> MO for a one-electron matrix, with MO-major ``c`` (nmo, nbasis).

    Returns ``C M C.T``. This is O(N^3) and is applied over the *full* MO space;
    it is the two-electron transform that we keep restricted to the window.
    """
    return c @ m_ao @ c.T


def density_matrix(c: np.ndarray, nocc: int) -> np.ndarray:
    """AO-basis density for the first ``nocc`` orbitals of MO-major ``c``.

    ``P_mn = sum_{i occupied} C[i, m] C[i, n]`` -- one electron per orbital, so
    this is the density for a single spin.
    """
    occ = c[:nocc, :]
    return occ.T @ occ


def build_fock_dense(
    h_ao: np.ndarray, eri_ao: np.ndarray, p_a: np.ndarray, p_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``F^s = h + J[P_a + P_b] - K[P_s]`` from a dense AO integral tensor.

    NumPy only, O(N^4) memory, so this is for small systems and for tests. The
    production fallback for real molecules is :func:`rebuild_fock_with_pyscf`,
    which never forms the full tensor.
    """
    p_total = p_a + p_b
    j = np.einsum("mnls,ls->mn", eri_ao, p_total, optimize=True)
    k_a = np.einsum("mlsn,ls->mn", eri_ao, p_a, optimize=True)
    k_b = np.einsum("mlsn,ls->mn", eri_ao, p_b, optimize=True)
    return h_ao + j - k_a, h_ao + j - k_b


def rebuild_fock_with_pyscf(
    mol, c_a: np.ndarray, c_b: np.ndarray, nalpha: int, nbeta: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the *HF* Fock matrices for the given orbitals using PySCF's JK.

    This is one Fock build -- far cheaper than an ``ao2mo`` -- and it is the
    mandatory path for Kohn-Sham orbitals, whose stored matrix contains
    exchange-correlation and is not a Fock operator at all. The orbitals are
    used as given; only the two-electron operators are rebuilt.

    ``c_a``/``c_b`` are MO-major ``(nmo, nbasis)``, this package's convention.
    """
    try:
        from pyscf import scf  # noqa: F401  (imported for its side effects too)
    except ImportError as exc:  # pragma: no cover - exercised by absence
        raise MissingDependencyError(
            "the rebuilt-Fock path needs PySCF, which is not installed. "
            "Install it with `pip install 'g16dump[validate]'` (or "
            "`pip install pyscf`). It is not required for the stored-Fock "
            "path."
        ) from exc

    p_a = density_matrix(np.asarray(c_a), nalpha)
    p_b = density_matrix(np.asarray(c_b), nbeta)

    h_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    # get_jk on a stacked density returns J and K for each, in one pass.
    from pyscf.scf import hf

    j_mats, k_mats = hf.get_jk(mol, np.array([p_a, p_b]), hermi=1)
    j_total = j_mats[0] + j_mats[1]
    return h_ao + j_total - k_mats[0], h_ao + j_total - k_mats[1]


def load_mol_from_fch(fchname: str):
    """Load a PySCF ``Mole`` from a Gaussian formatted checkpoint, via MOKIT."""
    try:
        from mokit.lib.gaussian import load_mol_from_fch as _load
    except ImportError:
        try:
            from mokit.lib.rwwfn import load_mol_from_fch as _load  # type: ignore
        except ImportError as exc:
            raise MissingDependencyError(
                f"reading {fchname} needs MOKIT (`load_mol_from_fch`), which is "
                f"not installed. Install it with "
                f"`pip install 'g16dump[validate]'`. It is only needed for the "
                f"rebuilt-Fock path and for the validation oracles."
            ) from exc
    return _load(fchname)


def with_rebuilt_fock(bundle: Bundle, mol) -> Bundle:
    """Return a copy of ``bundle`` whose Fock matrices were rebuilt with PySCF.

    Marks ``fock_source='rebuilt-pyscf'`` so the provenance records that these
    are reconstructed HF operators rather than anything Gaussian stored.
    """
    import dataclasses

    f_a, f_b = rebuild_fock_with_pyscf(
        mol, bundle.c_a, bundle.c_b, bundle.nalpha, bundle.nbeta
    )
    provenance = dict(bundle.provenance)
    provenance["fock_rebuilt"] = (
        "F^s = Hcore + J[P_a + P_b] - K[P_s], rebuilt with PySCF from the "
        "stored orbitals; any stored Gaussian Fock matrix was discarded."
    )
    return dataclasses.replace(
        bundle, f_a_ao=f_a, f_b_ao=f_b, fock_source="rebuilt-pyscf", provenance=provenance
    )


# ------------------------------------------------------------------- the math


def effective_one_electron(
    f_act: np.ndarray, eri_act: np.ndarray, nocc_same: int, nocc_other: int
) -> np.ndarray:
    """``h'`` for one spin, from that spin's active Fock block.

        h'_tu = f_tu - sum_{k in O_same} [ (tu|kk) - (tk|ku) ]
                     - sum_{k in O_other} (tu|kk)

    ``nocc_same``/``nocc_other`` count active orbitals occupied by this spin and
    by the opposite spin. Occupied orbitals are the *first* ones in the window.
    """
    h = f_act.copy()

    if nocc_same > 0:
        j_same = np.einsum("tukk->tu", eri_act[:, :, :nocc_same, :nocc_same])
        k_same = np.einsum("tkku->tu", eri_act[:, :nocc_same, :nocc_same, :])
        h -= j_same - k_same
    if nocc_other > 0:
        j_other = np.einsum("tukk->tu", eri_act[:, :, :nocc_other, :nocc_other])
        h -= j_other

    return h


def reference_energy(
    h_mo: np.ndarray,
    f_a_mo: np.ndarray,
    f_b_mo: np.ndarray,
    nalpha: int,
    nbeta: int,
    e_nuc: float,
) -> float:
    """``E_nuc + 1/2 sum_s sum_{i occupied} (h_ii + f^s_ii)``.

    Exact for any single determinant whose density built the Fock matrices --
    no assumption that the orbitals are canonical, or even that they solve any
    SCF equations.
    """
    e = e_nuc
    for f_mo, nocc in ((f_a_mo, nalpha), (f_b_mo, nbeta)):
        if nocc:
            e += 0.5 * float(
                np.trace(h_mo[:nocc, :nocc]) + np.trace(f_mo[:nocc, :nocc])
            )
    return e


def active_reference_energy(
    h_eff: np.ndarray, eri_act: np.ndarray, nocc_a: int, nocc_b: int
) -> float:
    """Energy of the reference determinant *within* the active space.

    Uses ``h'`` and the active integrals only, so ``E_core = E_ref - E_act`` is
    the exact scalar that makes the active-space Hamiltonian reproduce ``E_ref``.
    """
    e = 0.0
    for nocc in (nocc_a, nocc_b):
        if nocc:
            e += float(np.trace(h_eff[:nocc, :nocc]))

    # Same-spin: Coulomb minus exchange, half to avoid double counting.
    for nocc in (nocc_a, nocc_b):
        if nocc:
            block = eri_act[:nocc, :nocc, :nocc, :nocc]
            coulomb = np.einsum("ttuu->", block)
            exchange = np.einsum("tuut->", block)
            e += 0.5 * float(coulomb - exchange)

    # Opposite-spin: Coulomb only, counted once.
    if nocc_a and nocc_b:
        e += float(np.einsum("ttuu->", eri_act[:nocc_a, :nocc_a, :nocc_b, :nocc_b]))

    return e


def active_space_hamiltonian(
    bundle: Bundle,
    *,
    tol_spin: float = DEFAULT_SPIN_TOLERANCE,
    tol_scf: Optional[float] = 1e-8,
) -> ActiveSpaceHamiltonian:
    """Build ``h'``, ``E_core`` and ``E_ref`` for a bundle.

    ``tol_spin`` bounds ``max|h'(alpha) - h'(beta)|``. Exceeding it raises
    :class:`ConsistencyError`; the two are never averaged.

    ``tol_scf`` bounds ``|E_ref - E_scf|`` when the bundle carries an SCF energy.
    This is the single check that would have caught the legacy triplet bug. Pass
    ``None`` to skip it (for orbitals that deliberately do not solve any SCF
    equations, such as the rotated-orbital tests).
    """
    if bundle.is_ks and bundle.fock_source == "gaussian":
        raise ReferenceTypeError(
            f"reference type {bundle.ref_type} is Kohn-Sham, but this bundle's "
            f"Fock matrices came from Gaussian (fock_source='gaussian'). The "
            f"stored KS matrix contains the exchange-correlation potential and "
            f"is not a Fock operator, so using it as f^s would be physically "
            f"wrong. The active-space Hamiltonian is always the HF Hamiltonian "
            f"evaluated in the KS orbitals: rebuild the Fock matrices first "
            f"(`g16dump extract --rebuild-fock`, or "
            f"`hamiltonian.with_rebuilt_fock`)."
        )
    require(
        bundle.has_fock,
        "this bundle carries no Fock matrices, so h' cannot be built. Either "
        "re-extract from a .mat that stores them, or use the rebuilt-Fock path "
        "(`g16dump extract --rebuild-fock --fch JOB.fch`), which reconstructs "
        "F^s = Hcore + J[P_a + P_b] - K[P_s] from the orbitals with one PySCF "
        "Fock build.",
        ValidationError,
    )

    act = bundle.active
    nocc_a, nocc_b = bundle.nocc_a_act, bundle.nocc_b_act

    h_mo = mo_transform(bundle.h_ao, bundle.c_a)
    f_a_mo = mo_transform(bundle.f_a_ao, bundle.c_a)
    f_b_mo = mo_transform(bundle.f_b_ao, bundle.c_b)

    # The same expression for each spin. h' is spin-independent, so agreement
    # here is a genuine check on the input rather than a tautology.
    h_alpha = effective_one_electron(f_a_mo[act, act], bundle.eri_act, nocc_a, nocc_b)
    h_beta = effective_one_electron(f_b_mo[act, act], bundle.eri_act, nocc_b, nocc_a)

    spin_error = float(np.max(np.abs(h_alpha - h_beta)))
    if spin_error > tol_spin:
        raise ConsistencyError(
            f"the alpha- and beta-derived effective one-electron Hamiltonians "
            f"disagree by max|h'(a) - h'(b)| = {spin_error:.3e} Ha, which "
            f"exceeds the tolerance {tol_spin:.1e}.\n"
            f"h' is spin-independent by construction, so this means the stored "
            f"matrices are not the UHF-type Fock operators "
            f"F^s = Hcore + J[P_a + P_b] - K[P_s] that the derivation requires. "
            f"For ROHF this usually means Gaussian stored its Roothaan "
            f"effective operator instead.\n"
            f"Do not average the two -- that silently produces a wrong "
            f"Hamiltonian. Rebuild the Fock matrices instead: "
            f"`g16dump extract --rebuild-fock --fch JOB.fch`."
        )

    h_eff = h_alpha
    e_ref = reference_energy(
        h_mo, f_a_mo, f_b_mo, bundle.nalpha, bundle.nbeta, bundle.e_nuc
    )
    e_act = active_reference_energy(h_eff, bundle.eri_act, nocc_a, nocc_b)
    e_core = e_ref - e_act

    diagnostics = {
        "spin_error": spin_error,
        "e_ref": e_ref,
        "e_act": e_act,
        "e_core": e_core,
        "fock_source": bundle.fock_source,
        "ref_type": bundle.ref_type,
    }

    if bundle.e_scf is not None:
        scf_error = abs(e_ref - bundle.e_scf)
        diagnostics["scf_error"] = scf_error
        if tol_scf is not None and scf_error > tol_scf:
            raise ConsistencyError(
                f"the reference determinant energy rebuilt from h and f, "
                f"E_ref = {e_ref:.12f} Ha, disagrees with the SCF energy stored "
                f"in the bundle, E_scf = {bundle.e_scf:.12f} Ha, by "
                f"{scf_error:.3e} Ha (tolerance {tol_scf:.1e}).\n"
                f"E_ref is exact for the determinant that built these Fock "
                f"matrices, so a mismatch means the occupied orbitals are not "
                f"the first {bundle.nalpha} (alpha) / {bundle.nbeta} (beta) in "
                f"the stored order, the Fock matrices belong to a different "
                f"density, or the SCF was not converged tightly enough."
            )

    return ActiveSpaceHamiltonian(
        h_eff=h_eff,
        eri_act=bundle.eri_act,
        e_core=e_core,
        e_ref=e_ref,
        e_act=e_act,
        nelec_act=bundle.nelec_act,
        ms2=bundle.ms2,
        nocc_a_act=nocc_a,
        nocc_b_act=nocc_b,
        diagnostics=diagnostics,
    )


# ------------------------------------------------- persisting a built Hamiltonian

#: Marker distinguishing a rotated-Hamiltonian file from a .npz bundle.
HAMILTONIAN_SCHEMA_VERSION = 1


def save_hamiltonian(ash: ActiveSpaceHamiltonian, path: str, provenance: dict) -> None:
    """Write a built active-space Hamiltonian to a ``.npz``.

    This is a *different* artifact from a bundle, and deliberately so. A bundle
    holds AO-basis quantities plus orbitals; a Hamiltonian file holds ``h'``,
    the active ERIs and ``E_core`` -- objects that are already reduced to the
    active space.

    The distinction matters for rotations. A general rotation of the active
    orbitals mixes occupied with virtual, which changes the reference
    determinant and therefore the density that built ``f^s``. Rotating a bundle
    would leave its stored Fock matrices describing a determinant that no longer
    exists, and ``h'`` rebuilt from them would be wrong. Rotating the *reduced*
    Hamiltonian has no such problem: ``h' -> U.T h' U`` and the four-index
    transform are exact for any orthogonal ``U``, and ``E_core`` is untouched
    because the frozen core is not involved.
    """
    np.savez_compressed(
        path,
        hamiltonian_schema_version=np.array(HAMILTONIAN_SCHEMA_VERSION),
        h_eff=ash.h_eff,
        eri_act=ash.eri_act,
        e_core=np.array(ash.e_core),
        e_ref=np.array(ash.e_ref),
        e_act=np.array(ash.e_act),
        nelec_act=np.array(ash.nelec_act),
        ms2=np.array(ash.ms2),
        nocc_a_act=np.array(ash.nocc_a_act),
        nocc_b_act=np.array(ash.nocc_b_act),
        provenance=np.array(json.dumps(provenance, default=str)),
    )


def load_hamiltonian(path: str) -> tuple[ActiveSpaceHamiltonian, dict]:
    """Read a Hamiltonian written by :func:`save_hamiltonian`."""
    with np.load(path, allow_pickle=False) as data:
        if "hamiltonian_schema_version" not in data.files:
            raise SchemaError(
                f"{path} is not a g16dump Hamiltonian file (no "
                f"hamiltonian_schema_version key). If it is a .npz bundle, use "
                f"`g16dump dump` on it directly. Keys present: "
                f"{sorted(data.files)}."
            )
        version = int(data["hamiltonian_schema_version"])
        if version != HAMILTONIAN_SCHEMA_VERSION:
            raise SchemaError(
                f"{path} declares Hamiltonian schema version {version}; this "
                f"build understands {HAMILTONIAN_SCHEMA_VERSION}."
            )
        provenance = json.loads(str(data["provenance"])) if "provenance" in data.files else {}
        ash = ActiveSpaceHamiltonian(
            h_eff=np.array(data["h_eff"]),
            eri_act=np.array(data["eri_act"]),
            e_core=float(data["e_core"]),
            e_ref=float(data["e_ref"]),
            e_act=float(data["e_act"]),
            nelec_act=int(data["nelec_act"]),
            ms2=int(data["ms2"]),
            nocc_a_act=int(data["nocc_a_act"]),
            nocc_b_act=int(data["nocc_b_act"]),
            diagnostics=dict(provenance.get("diagnostics", {})),
        )
    return ash, provenance
