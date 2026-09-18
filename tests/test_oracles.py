"""Independent scientific oracles for the active-space Hamiltonian.

The package's claim is that it can build the frozen-core active-space
Hamiltonian from a *windowed* set of two-electron integrals plus the AO Fock
matrices, never touching a core-active integral. That claim is worth exactly as
much as the evidence against an independent construction, so this module builds
the same Hamiltonian a second time by an unrelated route and compares.

Route 1, the oracle, lives entirely in this file
    A full AO->MO transformation of every two-electron integral, followed by
    the textbook frozen-core reduction::

        h'_tu  = h_tu + sum_{i in core} [ 2 (tu|ii) - (ti|iu) ]
        E_core = E_nuc + sum_{i in core} 2 h_ii
                       + sum_{ij in core} [ 2 (ii|jj) - (ij|ji) ]

    and the Slater determinant energy evaluated straight from the full MO
    integrals. It is written with explicit loops and imports nothing from
    ``g16dump``: the production code and the oracle must not share a helper,
    or the comparison degenerates into checking that a function equals itself.

Route 2, the production code
    ``g16dump.hamiltonian.active_hamiltonian``, given the windowed ERIs and the
    AO Fock matrices, exactly as ``matfile.py`` will hand them over.

Four quantities are compared for every system: the effective one-electron
Hamiltonian, the active ERIs, ``E_core``, and the reference determinant energy.
For the Hartree-Fock systems the reference energy is compared against the SCF
energy PySCF's own driver converged to, which is a number this project did not
compute by any route.

The systems are the ones the plan names, transcribed from the benchmark
manifests. Their declared electron counts and basis sizes are asserted against
what PySCF actually builds, so a mis-transcribed window fails as a wrong window
rather than as a mysterious energy discrepancy.

Everything here needs PySCF and is marked accordingly; the core CI job runs
none of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from g16dump import hamiltonian as H
from g16dump.bundle import Bundle, make_provenance

pytestmark = pytest.mark.pyscf


# --------------------------------------------------------------- the systems
#
# Geometries and windows transcribed from benchmarks/systems/*.json. The window
# is given as Gaussian's 1-based inclusive NFIRST/NLAST, which is how a user
# writes it in a route line; the conversion to a 0-based half-open pair happens
# once, in _build, and is asserted against the declared counts.

H2O_GEOMETRY = """
O   0.0000000   0.0000000   0.1173000
H   0.0000000   0.7572000  -0.4692000
H   0.0000000  -0.7572000  -0.4692000
"""
CH2_GEOMETRY = """
C   0.0000000   0.0000000   0.0000000
H   0.9921000   0.0000000   0.4216000
H  -0.9921000   0.0000000   0.4216000
"""
NH_GEOMETRY = """
N   0.0000000   0.0000000   0.0000000
H   0.0000000   0.0000000   1.0362000
"""
O2_GEOMETRY = """
O   0.0000000   0.0000000   0.0000000
O   0.0000000   0.0000000   1.2075000
"""

SYSTEMS = {
    "h2o_rhf": dict(
        atom=H2O_GEOMETRY, basis="sto-3g", spin=0, reference="RHF",
        nfirst=2, nlast=7, nbasis=7, nelec_active=8, rotate=False,
        note="closed shell, canonical orbitals; the case the legacy scripts got right",
    ),
    "ch2_rohf": dict(
        atom=CH2_GEOMETRY, basis="6-31g", spin=2, reference="ROHF",
        nfirst=2, nlast=13, nbasis=13, nelec_active=6, rotate=False,
        note="open-shell triplet; four active alpha and two active beta electrons",
    ),
    "nh_rohf": dict(
        atom=NH_GEOMETRY, basis="6-31g", spin=2, reference="ROHF",
        nfirst=2, nlast=11, nbasis=11, nelec_active=6, rotate=False,
        note="a second open shell, with a different bonding pattern from CH2",
    ),
    "o2_rohf": dict(
        atom=O2_GEOMETRY, basis="6-31g", spin=2, reference="ROHF",
        nfirst=3, nlast=12, nbasis=18, nelec_active=12, rotate=False,
        note="the only system with frozen virtuals as well as frozen core",
    ),
    "ch2_rohf_rotated": dict(
        atom=CH2_GEOMETRY, basis="6-31g", spin=2, reference="ROHF",
        nfirst=2, nlast=13, nbasis=13, nelec_active=6, rotate=True,
        note="the same determinant in rotated orbitals; the MO Fock is not diagonal",
    ),
    "h2o_rks": dict(
        atom=H2O_GEOMETRY, basis="6-31g", spin=0, reference="RKS", xc="b3lyp",
        nfirst=2, nlast=13, nbasis=13, nelec_active=8, rotate=False,
        note="Kohn-Sham orbitals, which are only usable through the rebuilt-Fock path",
    ),
}

#: Systems whose reference determinant energy is the SCF energy of the job.
#: The KS entry is excluded: its Hamiltonian is the HF Hamiltonian evaluated in
#: KS orbitals, which is a different and deliberately larger number.
HARTREE_FOCK = [name for name, s in SYSTEMS.items() if s["reference"] != "RKS"]
OPEN_SHELL = [name for name, s in SYSTEMS.items() if s["spin"] != 0]


# ------------------------------------------------------------- the oracle
#
# Explicit loops, no einsum, and nothing imported from g16dump. Slow and
# obviously correct is the point: these are the equations out of a textbook,
# transcribed index by index.


def _oracle_to_mo(matrix_ao, mo_coeff):
    """``C^T A C`` for column-wise coefficients, spelled out."""
    return np.dot(np.transpose(mo_coeff), np.dot(matrix_ao, mo_coeff))


def _oracle_frozen_core(hcore_mo, eri, e_nuc, ncore, act_start, act_stop):
    """The textbook frozen-core reduction over the *full* MO integral set.

    Returns ``h'`` over the active window and the scalar core energy. Every
    integral used here has at least one core index, which is exactly the class
    of integral the windowed route never computes.
    """
    nact = act_stop - act_start
    h_eff = np.zeros((nact, nact))
    for t in range(nact):
        for u in range(nact):
            element = hcore_mo[act_start + t, act_start + u]
            for i in range(ncore):
                element += 2.0 * eri[act_start + t, act_start + u, i, i]
                element -= eri[act_start + t, i, i, act_start + u]
            h_eff[t, u] = element

    e_core = e_nuc
    for i in range(ncore):
        e_core += 2.0 * hcore_mo[i, i]
    for i in range(ncore):
        for j in range(ncore):
            e_core += 2.0 * eri[i, i, j, j] - eri[i, j, j, i]
    return h_eff, e_core


def _oracle_determinant_energy(one_electron, eri, nalpha, nbeta, constant=0.0):
    """Slater's rules for a single determinant, in whatever basis is handed in.

    Called twice per system with genuinely different arguments: once over the
    full MO set to get the total reference energy, and once over the active
    window with ``h'`` to get the energy the active-space Hamiltonian accounts
    for. That the two differ by exactly ``E_core`` is the whole content of the
    frozen-core reduction.
    """
    energy = constant
    for nocc in (nalpha, nbeta):
        for i in range(nocc):
            energy += one_electron[i, i]
    for nocc in (nalpha, nbeta):
        for i in range(nocc):
            for j in range(nocc):
                energy += 0.5 * (eri[i, i, j, j] - eri[i, j, j, i])
    for i in range(nalpha):
        for j in range(nbeta):
            energy += eri[i, i, j, j]
    return energy


def _legacy_closed_shell_h1(orbital_energies, eri_active, act_start, act_stop, nocc):
    """The legacy expression, transcribed from ``legacy/FCIDUMP_Write_MOe.py``.

    ``h'_tu = eps_tu - 2 sum_k (tu|kk) + sum_k (tk|ku)``, with ``eps`` the
    diagonal matrix of orbital energies standing in for the MO Fock matrix.
    That substitution is the assumption the rewrite exists to remove, and it is
    reproduced here verbatim rather than paraphrased, so the regression test
    compares against the code that actually shipped.
    """
    nact = act_stop - act_start
    h1 = np.diag(orbital_energies)[act_start:act_stop, act_start:act_stop].copy()
    h1 -= 2.0 * np.einsum(
        "acbb->ac", eri_active[:nact, :nact, :nocc, :nocc], optimize=True
    )
    h1 += np.einsum(
        "abbc->ac", eri_active[:nact, :nocc, :nocc, :nact], optimize=True
    )
    return h1


# ------------------------------------------------------------- construction


def _random_orthogonal(n, rng):
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return q * np.sign(np.diag(r))


def _rotate_within_blocks(mo_coeff, boundaries, seed):
    """Rotate inside each occupancy block, leaving the determinant untouched.

    The blocks are frozen core, doubly occupied active, singly occupied, and
    virtual. A rotation confined to one of them spans the same space, so every
    energy is unchanged, but the MO Fock matrix picks up off-diagonal elements
    and the orbital energies stop describing the orbitals they are stored with.
    """
    rng = np.random.default_rng(seed)
    rotated = np.array(mo_coeff, copy=True)
    for lo, hi in zip(boundaries, boundaries[1:]):
        if hi - lo > 1:
            rotated[:, lo:hi] = mo_coeff[:, lo:hi] @ _random_orthogonal(hi - lo, rng)
    return rotated


def _pyscf_fock_matrices(mol, mo_coeff, nalpha, nbeta, hcore_ao):
    """``F^sigma = Hcore + J[Pa+Pb] - K[Psigma]``, built here rather than imported.

    This is the input the bundle carries, so it has to come from somewhere; it
    comes from PySCF directly, so that ``g16dump`` contributes nothing to
    either side of the comparison. ``hamiltonian.rebuild_fock`` is checked
    against this in its own test below.
    """
    from pyscf import scf

    occ_a = mo_coeff[:, :nalpha]
    occ_b = mo_coeff[:, :nbeta]
    dm_a, dm_b = occ_a @ occ_a.T, occ_b @ occ_b.T
    j, k = scf.hf.get_jk(mol, [dm_a, dm_b], hermi=1)
    coulomb = j[0] + j[1]
    return hcore_ao + coulomb - k[0], hcore_ao + coulomb - k[1]


def _windowed_eri(mol, mo_coeff, act_start, act_stop):
    """The active-window MO block, transformed over the active columns only.

    This is what Gaussian's ``Window=`` produces and what the production route
    consumes. The oracle instead slices the same block out of a *full*
    transform, so agreement between the two is itself a check.
    """
    from pyscf import ao2mo

    c_act = mo_coeff[:, act_start:act_stop]
    nact = act_stop - act_start
    return ao2mo.restore(
        1, ao2mo.general(mol, [c_act] * 4, compact=False), nact
    ).reshape((nact,) * 4)


def _full_eri(mol, mo_coeff):
    from pyscf import ao2mo

    nmo = mo_coeff.shape[1]
    return ao2mo.restore(1, ao2mo.full(mol, mo_coeff), nmo).reshape((nmo,) * 4)


class Case:
    """One system, carried through both routes.

    Built once per session; the SCF, the full transform and the oracle are all
    on it, so a test is a comparison and nothing else.
    """

    def __init__(self, name, spec):
        from pyscf import dft, gto, scf

        self.name = name
        self.spec = spec
        mol = gto.M(
            atom=spec["atom"], basis=spec["basis"], spin=spec["spin"], verbose=0
        )
        if spec["reference"] == "RKS":
            mf = dft.RKS(mol)
            mf.xc = spec["xc"]
            mf.conv_tol = 1e-12
        elif spec["spin"] == 0:
            mf = scf.RHF(mol).set(conv_tol=1e-12)
        else:
            mf = scf.ROHF(mol).set(conv_tol=1e-12)
        mf.run()
        assert mf.converged, f"{name}: the SCF did not converge"

        mo = np.asarray(mf.mo_coeff)
        nalpha, nbeta = (int(n) for n in mol.nelec)
        # The one place the 1-based inclusive route window becomes a 0-based
        # half-open pair. Nothing else in this module does index arithmetic.
        act_start, act_stop = spec["nfirst"] - 1, spec["nlast"]

        self.canonical_mo = mo
        self.orbital_energies = np.asarray(mf.mo_energy, dtype=float).ravel()
        if spec["rotate"]:
            mo = _rotate_within_blocks(
                mo, (0, act_start, nbeta, nalpha, act_stop), seed=20260918
            )

        hcore_ao = np.asarray(mf.get_hcore())
        fock_a, fock_b = _pyscf_fock_matrices(mol, mo, nalpha, nbeta, hcore_ao)

        self.mol, self.mf, self.mo = mol, mf, mo
        self.nalpha, self.nbeta = nalpha, nbeta
        self.act_start, self.act_stop = act_start, act_stop
        self.e_scf = float(mf.e_tot)

        self.bundle = Bundle(
            reference=spec["reference"],
            charge=int(mol.charge),
            multiplicity=int(mol.spin) + 1,
            nelec=nalpha + nbeta,
            nalpha=nalpha,
            nbeta=nbeta,
            nao=mo.shape[0],
            nmo=mo.shape[1],
            ncore=act_start,
            nact=act_stop - act_start,
            act_start=act_start,
            act_stop=act_stop,
            e_nuc=float(mol.energy_nuc()),
            mo_coeff=mo,
            overlap=np.asarray(mf.get_ovlp()),
            hcore_ao=hcore_ao,
            eri_act=_windowed_eri(mol, mo, act_start, act_stop),
            fock_source="pyscf_rebuilt",
            fock_ao_alpha=fock_a,
            fock_ao_beta=fock_b,
            atom_charges=mol.atom_charges().astype(float),
            e_scf=float(mf.e_tot),
            provenance=make_provenance(
                basis=spec["basis"],
                method=spec["reference"],
                window_1based=(spec["nfirst"], spec["nlast"]),
                fock_source="pyscf_rebuilt",
                extra={"generated_by": "tests/test_oracles.py", "note": spec["note"]},
            ),
        )

        # -- the oracle route, from the full transform down ------------------
        hcore_mo = _oracle_to_mo(hcore_ao, mo)
        eri_full = _full_eri(mol, mo)
        window = slice(act_start, act_stop)

        self.oracle_h_eff, self.oracle_e_core = _oracle_frozen_core(
            hcore_mo, eri_full, float(mol.energy_nuc()), act_start,
            act_start, act_stop,
        )
        self.oracle_eri_active = eri_full[window, window, window, window]
        self.oracle_e_ref = _oracle_determinant_energy(
            hcore_mo, eri_full, nalpha, nbeta, constant=float(mol.energy_nuc())
        )
        self.oracle_e_act = _oracle_determinant_energy(
            self.oracle_h_eff, self.oracle_eri_active,
            nalpha - act_start, nbeta - act_start,
        )

        # -- the production route --------------------------------------------
        self.result = H.active_hamiltonian(self.bundle)


_CACHE: dict = {}


@pytest.fixture(scope="session")
def case():
    """``case(name)`` -> the :class:`Case` for that system, built once."""

    def _case(name: str) -> Case:
        if name not in _CACHE:
            pytest.importorskip("pyscf")
            _CACHE[name] = Case(name, SYSTEMS[name])
        return _CACHE[name]

    return _case


# --------------------------------------------------- the systems are what we say

@pytest.mark.parametrize("name", list(SYSTEMS))
def test_the_system_is_the_one_the_manifest_describes(case, name):
    """Guard the transcription before trusting any energy computed from it.

    A window copied down wrong produces a plausible-looking Hamiltonian for the
    wrong active space, and every comparison below would still pass because
    both routes would use it. These are the numbers that pin the system down.
    """
    c = case(name)
    spec = SYSTEMS[name]
    assert c.bundle.nao == spec["nbasis"], "basis set size"
    assert c.bundle.multiplicity == spec["spin"] + 1
    active_electrons = c.bundle.nocc_act_alpha + c.bundle.nocc_act_beta
    assert active_electrons == spec["nelec_active"], "active electron count"
    assert c.bundle.nelec == int(sum(c.mol.atom_charges())) - c.bundle.charge
    # Valid on its own terms, so a later failure is about the algebra and not
    # about a bundle that should never have been built.
    from g16dump.bundle import validate

    validate(c.bundle, source=name)


# ------------------------------------------- the four quantities, both routes

@pytest.mark.parametrize("name", list(SYSTEMS))
def test_effective_one_electron_hamiltonian_matches_the_oracle(case, name):
    """``h'`` from the windowed route equals the full-transform reduction.

    The windowed route never forms a core-active integral; the oracle forms
    every one of them. That they agree is the package's central claim.
    """
    c = case(name)
    np.testing.assert_allclose(
        c.result.h_eff, c.oracle_h_eff, atol=1e-10,
        err_msg=f"{name}: windowed h' disagrees with the full-transform reduction",
    )


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_active_eris_match_the_oracle(case, name):
    """The windowed transform equals the active block of the full transform."""
    c = case(name)
    np.testing.assert_allclose(
        c.result.eri_act, c.oracle_eri_active, atol=1e-10,
        err_msg=f"{name}: windowed ERIs disagree with the full transform",
    )


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_core_energy_matches_the_oracle(case, name):
    """``E_core`` from ``E_ref - E_act`` equals the textbook core expression."""
    c = case(name)
    assert c.result.e_core == pytest.approx(c.oracle_e_core, abs=1e-9), name


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_reference_energy_matches_the_oracle(case, name):
    """``E_ref`` from the Fock matrices equals Slater's rules over full integrals.

    The production route reaches this number from the Fock diagonals; the
    oracle sums two-electron integrals directly and never forms a Fock matrix.
    """
    c = case(name)
    assert c.result.e_ref == pytest.approx(c.oracle_e_ref, abs=1e-8), name


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_core_and_active_energies_partition_the_reference_energy(case, name):
    """``E_core + E_act = E_ref``, with both terms taken from the oracle.

    This is what makes the active-space Hamiltonian usable at all: a solver
    adds its correlation energy to ``E_core``, so any leak between the two
    terms shows up as an absolute energy that is quietly wrong.
    """
    c = case(name)
    assert c.oracle_e_core + c.oracle_e_act == pytest.approx(
        c.oracle_e_ref, abs=1e-8
    ), name
    assert c.result.e_core + c.result.e_act == pytest.approx(
        c.result.e_ref, abs=1e-9
    ), name


@pytest.mark.parametrize("name", HARTREE_FOCK)
def test_reference_energy_reproduces_the_scf_energy(case, name):
    """The one number in this file that neither route computed.

    PySCF's SCF driver converged it by iterating to self-consistency, which has
    nothing in common with either of the two routes being compared.
    """
    c = case(name)
    assert c.result.e_ref == pytest.approx(c.e_scf, abs=1e-8), (
        f"{name}: E_ref = {c.result.e_ref:.10f} but the SCF converged to "
        f"{c.e_scf:.10f}"
    )


@pytest.mark.parametrize("name", OPEN_SHELL)
def test_the_alpha_and_beta_routes_agree_on_open_shell_systems(case, name):
    """``h'`` is spin-independent by construction, and measurably so here.

    On these systems the alpha and beta occupations genuinely differ, so the
    two expressions subtract different integrals and arriving at the same
    matrix is informative rather than tautological.
    """
    c = case(name)
    assert c.bundle.nocc_act_alpha != c.bundle.nocc_act_beta, (
        f"{name} would not exercise the check: equal active occupations"
    )
    assert c.result.spin_deviation < 1e-9, (
        f"{name}: alpha- and beta-derived h' differ by "
        f"{c.result.spin_deviation:.3e}"
    )


# ----------------------------------------------- the legacy closed-shell claim

def test_canonical_rhf_reduces_to_the_legacy_closed_shell_expression(case):
    """For canonical RHF the new algebra *is* the old one, so it must agree.

    With ``O_alpha = O_beta`` the general expression collapses to
    ``f - 2J + K``, and for canonical orbitals ``f`` is the diagonal matrix of
    orbital energies. Both of the legacy assumptions hold here, so the legacy
    code was right on this case and the rewrite must not have changed it.
    """
    c = case("h2o_rhf")
    legacy = _legacy_closed_shell_h1(
        c.orbital_energies, c.bundle.eri_act, c.act_start, c.act_stop,
        c.bundle.nocc_act_alpha,
    )
    # Loose against machine precision on purpose: a converged SCF still leaves
    # MO Fock off-diagonals of the order of its own convergence threshold.
    np.testing.assert_allclose(c.result.h_eff, legacy, atol=1e-8)


def test_the_legacy_expression_fails_on_noncanonical_orbitals(case):
    """The mandatory test: ``diag(orbital_energies)`` is not the MO Fock matrix.

    The orbitals here span the same occupied space, so the determinant and
    every energy are unchanged and the new implementation still reproduces the
    SCF energy exactly. Only the legacy substitution breaks, and it breaks by
    almost a hartree -- not by a tolerance-sized amount that could be argued
    away as noise.
    """
    c = case("ch2_rohf_rotated")

    fock_mo = _oracle_to_mo(c.bundle.fock_ao_alpha, c.mo)
    off_diagonal = float(
        np.max(np.abs(fock_mo - np.diag(np.diag(fock_mo))))
    )
    assert off_diagonal > 1e-2, (
        "the fixture no longer has a non-diagonal MO Fock matrix, so it cannot "
        "demonstrate anything"
    )

    # The new route is unharmed.
    assert c.result.e_ref == pytest.approx(c.e_scf, abs=1e-8)
    np.testing.assert_allclose(c.result.h_eff, c.oracle_h_eff, atol=1e-10)

    # The legacy one is not. No diagonal matrix can equal this Fock matrix, so
    # the failure does not depend on which orbital energies are stored with the
    # rotated orbitals.
    legacy = _legacy_closed_shell_h1(
        c.orbital_energies, c.bundle.eri_act, c.act_start, c.act_stop,
        c.bundle.nocc_act_beta,
    )
    discrepancy = float(np.max(np.abs(c.result.h_eff - legacy)))
    assert discrepancy > 0.1, (
        f"the legacy expression was expected to fail here but agreed to "
        f"{discrepancy:.3e}"
    )


def test_the_legacy_expression_also_fails_on_canonical_rohf(case):
    """Open shells break it a second, independent way.

    Even with canonical orbitals, an ROHF job has ``O_alpha != O_beta`` and the
    stored orbital energies are eigenvalues of the Roothaan effective operator,
    which is not ``F^alpha`` or ``F^beta``. So the legacy expression is wrong
    here for a reason that has nothing to do with canonicality.
    """
    c = case("ch2_rohf")
    legacy = _legacy_closed_shell_h1(
        c.orbital_energies, c.bundle.eri_act, c.act_start, c.act_stop,
        c.bundle.nocc_act_beta,
    )
    assert float(np.max(np.abs(c.result.h_eff - legacy))) > 0.1
    assert c.result.e_ref == pytest.approx(c.e_scf, abs=1e-8)


def test_gaussians_roothaan_operator_is_rejected_rather_than_averaged(case):
    """An ROHF ``.mat`` carries one effective operator, and it is not usable.

    Measured here rather than asserted: the disagreement is most of a hartree,
    so the tolerance is nowhere near the decision. This is why the rebuilt-Fock
    path is the ordinary route for ROHF and not a fallback.
    """
    from dataclasses import replace

    c = case("ch2_rohf")
    roothaan = np.asarray(c.mf.get_fock())
    bundle = replace(
        c.bundle, fock_ao_alpha=roothaan, fock_ao_beta=roothaan.copy(),
        fock_source="gaussian",
    )
    with pytest.raises(H.SpinConsistencyError) as excinfo:
        H.active_hamiltonian(bundle)
    assert excinfo.value.deviation > 0.1, (
        f"the Roothaan operator was rejected by only "
        f"{excinfo.value.deviation:.3e}, which is too close to the tolerance "
        f"to be a safe gate"
    )
    assert "do not average" in str(excinfo.value).lower()


# ------------------------------------------------------ Kohn-Sham orbitals

def test_kohn_sham_orbitals_give_the_hf_hamiltonian_not_the_ks_one(case):
    """The KS determinant's HF energy, which is not the KS SCF energy.

    Both routes agree on it, and both differ from the B3LYP total energy by
    several tenths of a hartree. If ``E_ref`` ever came out equal to the KS
    energy, the exchange-correlation potential would have leaked into the
    Hamiltonian being handed to the solver.
    """
    c = case("h2o_rks")
    assert c.result.e_ref == pytest.approx(c.oracle_e_ref, abs=1e-8)
    assert abs(c.result.e_ref - c.e_scf) > 0.1, (
        "the reference energy equals the Kohn-Sham SCF energy, so the "
        "Hamiltonian is not the HF one"
    )


def test_the_stored_ks_matrix_would_have_been_materially_wrong(case):
    """Quantify what the rebuilt-Fock path is protecting against.

    Relabelled RHF so that the reference-type gate does not fire first and the
    numbers actually get computed: this is the damage done by treating the
    stored Kohn-Sham matrix as a Fock matrix, which is what an unguarded reader
    would do.
    """
    from dataclasses import replace

    c = case("h2o_rks")
    ks_matrix = np.asarray(c.mf.get_fock())
    disguised = replace(
        c.bundle, reference="RHF", fock_ao_alpha=ks_matrix,
        fock_ao_beta=ks_matrix.copy(), fock_source="gaussian",
    )
    wrong = H.active_hamiltonian(disguised)
    assert abs(wrong.e_ref - c.result.e_ref) > 0.5, (
        "treating the KS matrix as a Fock matrix should be badly wrong"
    )
    assert float(np.max(np.abs(wrong.h_eff - c.result.h_eff))) > 0.1

    # And with the reference type told truthfully, it never gets this far.
    from g16dump.hamiltonian import HamiltonianError

    with pytest.raises(HamiltonianError, match="exchange-correlation"):
        H.active_hamiltonian(replace(disguised, reference="RKS"))


def test_the_production_rebuild_agrees_with_the_fock_matrices_used_here(case):
    """Tie the oracle's inputs back to the production rebuild path.

    Everything above feeds the bundle Fock matrices this module built from
    PySCF directly. That is deliberate, but it leaves ``rebuild_fock`` itself
    unmeasured, so it is measured here.
    """
    c = case("h2o_rks")
    rebuilt_a, rebuilt_b = H.rebuild_fock(
        c.mol, c.mo, c.nalpha, c.nbeta, c.bundle.hcore_ao
    )
    np.testing.assert_allclose(rebuilt_a, c.bundle.fock_ao_alpha, atol=1e-10)
    np.testing.assert_allclose(rebuilt_b, c.bundle.fock_ao_beta, atol=1e-10)

    rebuilt_bundle = H.with_rebuilt_fock(c.bundle, c.mol)
    assert rebuilt_bundle.fock_source == "pyscf_rebuilt"
    assert H.active_hamiltonian(rebuilt_bundle).e_ref == pytest.approx(
        c.result.e_ref, abs=1e-9
    )
