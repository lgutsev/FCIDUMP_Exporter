"""The open-shell frozen-core reduction: ``h'``, ``E_core``, ``E_ref``.

One code path serves RHF and ROHF. The distinction that matters is not the
reference label but whether the Fock matrices handed in are the UHF-type
operators built from the reference determinant's own densities,

    F^sigma = Hcore + J[P^alpha + P^beta] - K[P^sigma]

Everything here is exact for those and wrong for anything else, which is why the
alpha/beta agreement is checked rather than assumed. See :func:`active_hamiltonian`.

The algebra, in the README's notation. ``C`` are the frozen core orbitals,
``A`` the active window, ``O_sigma`` the active orbitals occupied in the
reference determinant for spin sigma -- by index order, the lowest
``n_sigma - ncore`` of them:

    h'_tu = f^sigma_tu - sum_{k in O_sigma} [ (tu|kk) - (tk|ku) ]
                       - sum_{k in O_sigmabar} (tu|kk)

Substituting ``F^sigma`` shows why the result cannot depend on spin::

    h'_tu = h_tu + 2 sum_{i in C} (tu|ii) - sum_{i in C} (ti|iu)

which is the ordinary frozen-core effective Hamiltonian. The point of the
windowed route is that it reaches that answer from the active ERIs and the Fock
matrices alone, never touching a core-active integral.

Nothing in this module reads a file or imports pyscf, except
:func:`rebuild_fock` and the ``.fch`` helper beside it, which import lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bundle import Bundle

#: ``max |h'(alpha) - h'(beta)|``. Above this the stored matrices are not
#: ``F^alpha``/``F^beta`` and the result would be meaningless. Never averaged.
SPIN_CONSISTENCY_TOL = 1e-8


class HamiltonianError(ValueError):
    """The active-space Hamiltonian cannot be built from what was supplied."""


class SpinConsistencyError(HamiltonianError):
    """``h'`` came out spin-dependent, so the Fock matrices are not F^alpha/F^beta."""

    def __init__(self, deviation: float, tol: float):
        self.deviation = deviation
        self.tol = tol
        super().__init__(
            f"the alpha- and beta-derived effective one-electron Hamiltonians "
            f"disagree by {deviation:.3e}, above the tolerance {tol:.1e}.\n"
            f"h' is spin-independent by construction, so this means the stored "
            f"matrices are not F^sigma = Hcore + J[Pa+Pb] - K[Psigma] -- most "
            f"often they are Gaussian's Roothaan effective ROHF operator, or a "
            f"Kohn-Sham matrix.\n"
            f"Do not average them. Rebuild the Fock matrices from the orbitals "
            f"with g16dump.hamiltonian.rebuild_fock and try again."
        )


@dataclass
class ActiveHamiltonian:
    """The active-space Hamiltonian, ready for a writer or a solver.

    ``e_core`` is the scalar the FCIDUMP carries; it already includes the
    nuclear repulsion. ``e_ref`` is the energy of the reference determinant and
    is the number to compare against the SCF energy.
    """

    h_eff: np.ndarray
    eri_act: np.ndarray
    e_core: float
    e_ref: float
    e_act: float
    nact: int
    nelec_act: int
    ms2: int
    spin_deviation: float
    fock_source: str

    def reference_energy_from_parts(self) -> float:
        """``E_core + E_act``, i.e. the reference energy this Hamiltonian encodes."""
        return self.e_core + self.e_act


# ------------------------------------------------------------ MO transforms


def to_mo(matrix_ao: np.ndarray, mo_coeff: np.ndarray) -> np.ndarray:
    """``C.T A C`` for the column-wise coefficients the bundle stores."""
    return mo_coeff.T @ matrix_ao @ mo_coeff


def mo_fock_matrices(bundle: Bundle) -> tuple[np.ndarray, np.ndarray]:
    """The alpha and beta Fock matrices of ``bundle``, in the MO basis.

    A closed-shell bundle may legitimately store only the alpha matrix, since
    the two are equal. An open-shell one may not: a missing beta matrix there is
    missing information, not a shortcut, and is refused.
    """
    if bundle.fock_ao_alpha is None:
        raise HamiltonianError(
            f"{bundle.reference} bundle carries no Fock matrix (fock_source="
            f"{bundle.fock_source!r}). Rebuild one from the orbitals with "
            f"g16dump.hamiltonian.rebuild_fock."
        )

    fock_beta_ao = bundle.fock_ao_beta
    if fock_beta_ao is None:
        if bundle.nalpha != bundle.nbeta:
            raise HamiltonianError(
                f"open-shell reference ({bundle.nalpha} alpha, {bundle.nbeta} "
                f"beta electrons) but no beta Fock matrix is stored. F^alpha and "
                f"F^beta differ whenever the spin densities differ; the beta "
                f"matrix cannot be substituted by the alpha one here."
            )
        fock_beta_ao = bundle.fock_ao_alpha

    return (
        to_mo(bundle.fock_ao_alpha, bundle.mo_coeff),
        to_mo(fock_beta_ao, bundle.mo_coeff),
    )


# ------------------------------------------------------- the frozen-core fold


def _coulomb_exchange_over_occupied(
    eri_act: np.ndarray, nocc: int
) -> tuple[np.ndarray, np.ndarray]:
    """``sum_{k<nocc} (tu|kk)`` and ``sum_{k<nocc} (tk|ku)`` over the window."""
    if nocc == 0:
        zero = np.zeros(eri_act.shape[:2])
        return zero, zero.copy()
    coulomb = np.einsum("tukk->tu", eri_act[:, :, :nocc, :nocc], optimize=True)
    exchange = np.einsum("tkku->tu", eri_act[:, :nocc, :nocc, :], optimize=True)
    return coulomb, exchange


def effective_one_electron(
    fock_mo_alpha: np.ndarray,
    fock_mo_beta: np.ndarray,
    eri_act: np.ndarray,
    active: slice,
    nocc_act_alpha: int,
    nocc_act_beta: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Both spin-derived effective one-electron Hamiltonians, unreconciled.

    Returned separately and on purpose: the caller compares them. A function
    that returned one number here would be hiding the only free check this
    method has.
    """
    f_a = np.asarray(fock_mo_alpha)[active, active]
    f_b = np.asarray(fock_mo_beta)[active, active]

    j_a, k_a = _coulomb_exchange_over_occupied(eri_act, nocc_act_alpha)
    j_b, k_b = _coulomb_exchange_over_occupied(eri_act, nocc_act_beta)

    h_from_alpha = f_a - (j_a - k_a) - j_b
    h_from_beta = f_b - (j_b - k_b) - j_a
    return h_from_alpha, h_from_beta


def reference_energy(
    hcore_mo: np.ndarray,
    fock_mo_alpha: np.ndarray,
    fock_mo_beta: np.ndarray,
    e_nuc: float,
    nalpha: int,
    nbeta: int,
) -> float:
    """``E_nuc + 1/2 sum_sigma sum_{i occupied} (h_ii + f^sigma_ii)``."""
    diag = np.diag(hcore_mo)
    energy = e_nuc
    energy += 0.5 * float(
        np.sum(diag[:nalpha] + np.diag(fock_mo_alpha)[:nalpha])
    )
    energy += 0.5 * float(np.sum(diag[:nbeta] + np.diag(fock_mo_beta)[:nbeta]))
    return energy


def active_energy(
    h_eff: np.ndarray, eri_act: np.ndarray, nocc_alpha: int, nocc_beta: int
) -> float:
    """Energy of the reference determinant within the active space alone.

    Counts only what the active-space Hamiltonian itself accounts for; the
    difference from :func:`reference_energy` is ``E_core``.
    """
    energy = float(np.sum(np.diag(h_eff)[:nocc_alpha]))
    energy += float(np.sum(np.diag(h_eff)[:nocc_beta]))

    for nocc in (nocc_alpha, nocc_beta):
        if nocc:
            block = eri_act[:nocc, :nocc, :nocc, :nocc]
            coulomb = np.einsum("ttuu->", block, optimize=True)
            exchange = np.einsum("tuut->", block, optimize=True)
            energy += 0.5 * float(coulomb - exchange)

    if nocc_alpha and nocc_beta:
        energy += float(
            np.einsum(
                "ttuu->", eri_act[:nocc_alpha, :nocc_alpha, :nocc_beta, :nocc_beta],
                optimize=True,
            )
        )
    return energy


def active_hamiltonian(
    bundle: Bundle, *, spin_tol: float = SPIN_CONSISTENCY_TOL
) -> ActiveHamiltonian:
    """Fold the frozen core into ``h'`` and ``E_core`` for one bundle.

    Raises :class:`SpinConsistencyError` when the alpha- and beta-derived
    effective Hamiltonians disagree by more than ``spin_tol``. That disagreement
    is a bug signal about the input, not a number to average away.
    """
    if bundle.is_ks and bundle.fock_source != "pyscf_rebuilt":
        raise HamiltonianError(
            f"{bundle.reference} orbitals with fock_source="
            f"{bundle.fock_source!r}. The Kohn-Sham matrix contains "
            f"exchange-correlation and is not the HF Fock operator; the "
            f"active-space Hamiltonian is always the HF Hamiltonian evaluated "
            f"in the KS orbitals. Use g16dump.hamiltonian.rebuild_fock."
        )

    fock_a, fock_b = mo_fock_matrices(bundle)
    hcore_mo = to_mo(bundle.hcore_ao, bundle.mo_coeff)

    h_from_alpha, h_from_beta = effective_one_electron(
        fock_a,
        fock_b,
        bundle.eri_act,
        bundle.active,
        bundle.nocc_act_alpha,
        bundle.nocc_act_beta,
    )
    deviation = float(np.max(np.abs(h_from_alpha - h_from_beta)))
    if deviation > spin_tol:
        raise SpinConsistencyError(deviation, spin_tol)

    # Below tolerance the two are the same matrix to numerical precision. Take
    # the alpha one rather than the mean, so the returned h' is a quantity that
    # was computed, not a compromise between two of them.
    h_eff = h_from_alpha

    e_ref = reference_energy(
        hcore_mo, fock_a, fock_b, bundle.e_nuc, bundle.nalpha, bundle.nbeta
    )
    e_act = active_energy(
        h_eff, bundle.eri_act, bundle.nocc_act_alpha, bundle.nocc_act_beta
    )

    return ActiveHamiltonian(
        h_eff=h_eff,
        eri_act=np.asarray(bundle.eri_act),
        e_core=e_ref - e_act,
        e_ref=e_ref,
        e_act=e_act,
        nact=bundle.nact,
        nelec_act=bundle.nocc_act_alpha + bundle.nocc_act_beta,
        ms2=bundle.nalpha - bundle.nbeta,
        spin_deviation=deviation,
        fock_source=bundle.fock_source,
    )


# ------------------------------------------------------- rebuilt-Fock path


def densities(mo_coeff: np.ndarray, nalpha: int, nbeta: int) -> tuple[np.ndarray, ...]:
    """``P^sigma = C_occ C_occ.T`` from the orbitals and their occupations."""
    occ_a = mo_coeff[:, :nalpha]
    occ_b = mo_coeff[:, :nbeta]
    return occ_a @ occ_a.T, occ_b @ occ_b.T


def rebuild_fock(
    mol,
    mo_coeff: np.ndarray,
    nalpha: int,
    nbeta: int,
    hcore_ao: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``F^sigma = Hcore + J[Pa+Pb] - K[Psigma]`` through PySCF.

    This is the path that makes Kohn-Sham orbitals usable, and the path an ROHF
    job needs whenever Gaussian stored its Roothaan effective operator instead
    of the two spin Fock matrices. One JK build, far cheaper than an ``ao2mo``.

    ``mol`` is a PySCF ``Mole``; :func:`mol_from_fch` builds one from the
    Gaussian formatted checkpoint that goes with the ``.mat``.
    """
    try:
        from pyscf import scf
    except ImportError as exc:  # pragma: no cover - exercised by absence, not CI
        raise HamiltonianError(
            "the rebuilt-Fock path needs pyscf, which is not installed. "
            "Install it with: pip install 'g16dump[validate]'"
        ) from exc

    dm_a, dm_b = densities(np.asarray(mo_coeff), nalpha, nbeta)
    if hcore_ao is None:
        hcore_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

    j_matrices, k_matrices = scf.hf.get_jk(mol, [dm_a, dm_b], hermi=1)
    coulomb = j_matrices[0] + j_matrices[1]

    fock_a = hcore_ao + coulomb - k_matrices[0]
    fock_b = hcore_ao + coulomb - k_matrices[1]
    return fock_a, fock_b


def with_rebuilt_fock(bundle: Bundle, mol) -> Bundle:
    """A copy of ``bundle`` carrying HF Fock matrices rebuilt from its orbitals.

    The returned bundle records ``fock_source="pyscf_rebuilt"`` in both the
    field and the provenance, so a downstream FCIDUMP can say where its
    Hamiltonian came from.
    """
    from dataclasses import replace

    fock_a, fock_b = rebuild_fock(
        mol, bundle.mo_coeff, bundle.nalpha, bundle.nbeta, bundle.hcore_ao
    )
    provenance = dict(bundle.provenance)
    provenance["fock_source"] = "pyscf_rebuilt"
    provenance["fock_rebuilt_by"] = "pyscf"
    return replace(
        bundle,
        fock_ao_alpha=fock_a,
        fock_ao_beta=fock_b,
        fock_source="pyscf_rebuilt",
        provenance=provenance,
    )


def mol_from_fch(path):
    """Load a PySCF ``Mole`` from a Gaussian formatted checkpoint, through MOKIT.

    MOKIT is not on PyPI; it is installed alongside Gaussian workflows. Its
    absence is reported here rather than as a traceback from an import three
    frames down.
    """
    try:
        from mokit.lib.gaussian import load_mol_from_fch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise HamiltonianError(
            f"reading {path} needs MOKIT (mokit.lib.gaussian.load_mol_from_fch), "
            f"which is not importable. MOKIT is not on PyPI; see "
            f"https://gitlab.com/jxzou/mokit for installation. Alternatively, "
            f"build the pyscf Mole yourself and pass it to "
            f"g16dump.hamiltonian.with_rebuilt_fock."
        ) from exc
    return load_mol_from_fch(str(path))


__all__ = [
    "ActiveHamiltonian",
    "HamiltonianError",
    "SpinConsistencyError",
    "SPIN_CONSISTENCY_TOL",
    "active_hamiltonian",
    "active_energy",
    "densities",
    "effective_one_electron",
    "mo_fock_matrices",
    "mol_from_fch",
    "rebuild_fock",
    "reference_energy",
    "to_mo",
    "with_rebuilt_fock",
]
