"""The open-shell frozen-core algebra.

Three things are being asserted, in order of how much they matter:

1. ``E_ref`` computed from the Fock matrices reproduces the SCF energy of the
   job the bundle came from. That number was produced by a different program
   through a different route, so it is a real check and not a restatement.
2. ``h'`` is correct where ``diag(orbital_energies)`` is not. The rotated
   fixture exists solely for this, and the test computes both.
3. The alpha/beta agreement is enforced rather than averaged away. The Roothaan
   fixture exists solely to be rejected.

The committed ``.npz`` fixtures carry real SCF quantities, so all of the above
runs without PySCF. One test compares against a full AO->MO transform and does
need it; it is marked ``pyscf``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from g16dump import hamiltonian as H
from g16dump.bundle import load

SOUND = ["h2o_rhf", "ch2_rohf", "ch2_rohf_rotated"]


@pytest.fixture
def bundles(fixture_path):
    def _load(name):
        return load(fixture_path(name))

    return _load


# --------------------------------------------- the reference energy is right

@pytest.mark.parametrize("name", SOUND)
def test_reference_energy_reproduces_the_scf_energy(bundles, name):
    """The one check that comes from outside this code base entirely."""
    bundle = bundles(name)
    result = H.active_hamiltonian(bundle)
    assert bundle.e_scf is not None
    assert result.e_ref == pytest.approx(bundle.e_scf, abs=1e-9)


@pytest.mark.parametrize("name", SOUND)
def test_core_and_active_energies_partition_the_reference(bundles, name):
    """``E_core`` is defined as the remainder, so the parts must re-add exactly."""
    result = H.active_hamiltonian(bundles(name))
    assert result.reference_energy_from_parts() == pytest.approx(
        result.e_ref, abs=1e-12
    )


@pytest.mark.parametrize("name", SOUND)
def test_h_effective_is_spin_independent(bundles, name):
    """Substituting F^sigma into h' cancels the spin dependence analytically."""
    result = H.active_hamiltonian(bundles(name))
    assert result.spin_deviation < 1e-10


@pytest.mark.parametrize("name", SOUND)
def test_h_effective_is_symmetric(bundles, name):
    h = H.active_hamiltonian(bundles(name)).h_eff
    np.testing.assert_allclose(h, h.T, atol=1e-10)


def test_rotated_orbitals_leave_every_energy_unchanged(bundles):
    """Rotating inside the occupied and virtual blocks cannot move an energy."""
    plain = H.active_hamiltonian(bundles("ch2_rohf"))
    rotated = H.active_hamiltonian(bundles("ch2_rohf_rotated"))
    assert rotated.e_ref == pytest.approx(plain.e_ref, abs=1e-9)
    assert rotated.e_core == pytest.approx(plain.e_core, abs=1e-9)


# ------------------------------------------- where the legacy algebra fails

def _h_from_diagonal_fock_only(bundle):
    """``h'`` as the legacy scripts build it: the Fock matrix's diagonal alone.

    ``legacy/FCIDUMP_Write_MOe_3.py`` reaches this by using
    ``diag(ALPHA ORBITAL ENERGIES)`` in place of the MO Fock matrix. Discarding
    the off-diagonals of the real Fock matrix is the same assumption, expressed
    so that the two can be compared on identical input.
    """
    fock_alpha, fock_beta = H.mo_fock_matrices(bundle)
    diagonal_only = [np.diag(np.diag(f)) for f in (fock_alpha, fock_beta)]
    return H.effective_one_electron(
        diagonal_only[0],
        diagonal_only[1],
        bundle.eri_act,
        bundle.active,
        bundle.nocc_act_alpha,
        bundle.nocc_act_beta,
    )[0]


def test_canonical_rhf_is_where_the_legacy_assumption_holds(bundles):
    """Canonical RHF has a diagonal MO Fock matrix, so the two agree."""
    bundle = bundles("h2o_rhf")
    exact = H.active_hamiltonian(bundle).h_eff
    np.testing.assert_allclose(
        _h_from_diagonal_fock_only(bundle), exact, atol=1e-8
    )


def test_rotated_rohf_is_where_the_legacy_assumption_fails(bundles):
    """Non-canonical orbitals: the discarded off-diagonals are Ha-sized."""
    bundle = bundles("ch2_rohf_rotated")
    exact = H.active_hamiltonian(bundle).h_eff
    legacy = _h_from_diagonal_fock_only(bundle)
    error = float(np.max(np.abs(legacy - exact)))
    assert error > 0.1, (
        f"the rotated fixture is supposed to break the diagonal-Fock "
        f"assumption, but the error is only {error:.3e} Ha"
    )


def test_the_rotated_fixture_really_has_a_nondiagonal_fock(bundles):
    """Guard the guard: a fixture that quietly stayed canonical proves nothing."""
    bundle = bundles("ch2_rohf_rotated")
    fock = H.to_mo(bundle.fock_ao_alpha, bundle.mo_coeff)[bundle.active, bundle.active]
    off_diagonal = float(np.max(np.abs(fock - np.diag(np.diag(fock)))))
    assert off_diagonal > 1e-2


# ------------------------------------------- the alpha/beta consistency gate

def test_roothaan_effective_fock_is_rejected_loudly(bundles):
    """Gaussian's stored ROHF operator is not F^alpha/F^beta, and it shows."""
    with pytest.raises(H.SpinConsistencyError) as excinfo:
        H.active_hamiltonian(bundles("ch2_rohf_roothaan"))
    assert excinfo.value.deviation > 1e-3
    message = str(excinfo.value)
    assert "Do not average" in message
    assert "rebuild_fock" in message


def test_the_consistency_tolerance_is_configurable(bundles):
    """Configurable, but never a way to make a real disagreement disappear."""
    bundle = bundles("ch2_rohf_roothaan")
    with pytest.raises(H.SpinConsistencyError):
        H.active_hamiltonian(bundle, spin_tol=1e-8)
    loose = H.active_hamiltonian(bundle, spin_tol=10.0)
    assert loose.spin_deviation > 1e-3


def test_a_disagreeing_pair_is_never_averaged(bundles):
    """h' must be one of the two computed matrices, not a compromise."""
    bundle = bundles("ch2_rohf_roothaan")
    result = H.active_hamiltonian(bundle, spin_tol=10.0)
    from_alpha, from_beta = H.effective_one_electron(
        *H.mo_fock_matrices(bundle),
        bundle.eri_act,
        bundle.active,
        bundle.nocc_act_alpha,
        bundle.nocc_act_beta,
    )
    np.testing.assert_allclose(result.h_eff, from_alpha, atol=0)
    assert not np.allclose(result.h_eff, 0.5 * (from_alpha + from_beta))


# ------------------------------------------------------ missing information

def test_open_shell_without_a_beta_fock_is_refused(bundles):
    bundle = replace(bundles("ch2_rohf"), fock_ao_beta=None)
    with pytest.raises(H.HamiltonianError, match="no beta Fock matrix"):
        H.active_hamiltonian(bundle)


def test_closed_shell_without_a_beta_fock_is_fine(bundles):
    """For a closed shell the two are the same matrix; h2o_rhf stores only one."""
    bundle = bundles("h2o_rhf")
    assert bundle.fock_ao_beta is None
    result = H.active_hamiltonian(bundle)
    assert result.spin_deviation == pytest.approx(0.0, abs=1e-12)


def test_no_fock_at_all_names_the_way_out(bundles):
    bundle = replace(
        bundles("h2o_rhf"), fock_ao_alpha=None, fock_ao_beta=None,
        fock_source="none",
    )
    with pytest.raises(H.HamiltonianError, match="rebuild_fock"):
        H.active_hamiltonian(bundle)


# ------------------------------------------------------------- the KS gate

def test_ks_orbitals_may_not_use_a_stored_fock(bundles):
    """The KS matrix carries exchange-correlation; it is not the HF Hamiltonian."""
    bundle = replace(
        bundles("h2o_rhf"), reference="RKS", fock_source="none",
    )
    with pytest.raises(H.HamiltonianError, match="exchange-correlation"):
        H.active_hamiltonian(bundle)


def test_ks_orbitals_with_a_rebuilt_fock_are_accepted(bundles):
    """Relabelled only: the point is that the gate opens on fock_source alone."""
    bundle = replace(
        bundles("h2o_rhf"), reference="RKS", fock_source="pyscf_rebuilt",
    )
    assert H.active_hamiltonian(bundle).fock_source == "pyscf_rebuilt"


# --------------------------------------------------- the independent oracle

@pytest.mark.pyscf
def test_against_a_full_ao_to_mo_transform(bundles):
    """Compare with the ordinary frozen-core reduction after a full transform.

    Two genuinely different routes to the same matrix. This one builds every MO
    integral, including the core-active ones the windowed method never touches,
    and folds the core the textbook way::

        h'_tu  = h_tu + sum_{i in core} [ 2 (tu|ii) - (ti|iu) ]
        E_core = E_nuc + sum_i 2 h_ii + sum_ij [ 2 (ii|jj) - (ij|ji) ]

    The windowed route reaches the same answer from the Fock matrices and the
    active ERIs alone, which is the claim the whole package rests on.
    """
    gto = pytest.importorskip("pyscf.gto")
    ao2mo = pytest.importorskip("pyscf.ao2mo")
    from make_fixtures import CH2, H2O

    cases = [
        ("h2o_rhf", dict(atom=H2O, basis="sto-3g", verbose=0)),
        ("ch2_rohf", dict(atom=CH2, basis="6-31g", spin=2, verbose=0)),
    ]
    for name, spec in cases:
        bundle = bundles(name)
        mol = gto.M(**spec)
        assert mol.energy_nuc() == pytest.approx(bundle.e_nuc, abs=1e-9)

        mo = bundle.mo_coeff
        hcore_mo = H.to_mo(bundle.hcore_ao, mo)
        eri = ao2mo.restore(1, ao2mo.full(mol, mo), bundle.nmo)

        active = bundle.active
        h_oracle = hcore_mo[active, active].copy()
        e_core_oracle = bundle.e_nuc
        for i in range(bundle.ncore):
            h_oracle += 2.0 * eri[active, active, i, i] - eri[active, i, i, active]
            e_core_oracle += 2.0 * hcore_mo[i, i]
            for j in range(bundle.ncore):
                e_core_oracle += 2.0 * eri[i, i, j, j] - eri[i, j, j, i]

        result = H.active_hamiltonian(bundle)
        np.testing.assert_allclose(result.h_eff, h_oracle, atol=1e-10, err_msg=name)
        assert result.e_core == pytest.approx(e_core_oracle, abs=1e-9), name


@pytest.mark.pyscf
def test_rebuilt_fock_matches_the_scf_programs_own_fock():
    """For RHF the rebuilt F^alpha is exactly what the SCF program reports."""
    gto = pytest.importorskip("pyscf.gto")
    scf = pytest.importorskip("pyscf.scf")
    from make_fixtures import H2O

    mol = gto.M(atom=H2O, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    fock_a, fock_b = H.rebuild_fock(mol, mf.mo_coeff, 5, 5, mf.get_hcore())
    np.testing.assert_allclose(fock_a, mf.get_fock(), atol=1e-10)
    np.testing.assert_allclose(fock_a, fock_b, atol=1e-12)


@pytest.mark.pyscf
def test_with_rebuilt_fock_records_its_own_provenance(bundles):
    """A Hamiltonian must be able to say where its Fock matrix came from."""
    gto = pytest.importorskip("pyscf.gto")
    from make_fixtures import H2O

    bundle = replace(
        bundles("h2o_rhf"), fock_ao_alpha=None, fock_ao_beta=None,
        fock_source="none",
    )
    rebuilt = H.with_rebuilt_fock(bundle, gto.M(atom=H2O, basis="sto-3g", verbose=0))
    assert rebuilt.fock_source == "pyscf_rebuilt"
    assert rebuilt.provenance["fock_source"] == "pyscf_rebuilt"
    assert H.active_hamiltonian(rebuilt).e_ref == pytest.approx(
        bundle.e_scf, abs=1e-9
    )


def test_densities_are_idempotent_against_the_overlap(bundles):
    """``P S P = P``: the density is a projector onto the occupied space."""
    bundle = bundles("ch2_rohf")
    dm_a, dm_b = H.densities(bundle.mo_coeff, bundle.nalpha, bundle.nbeta)
    for dm, nocc in ((dm_a, bundle.nalpha), (dm_b, bundle.nbeta)):
        np.testing.assert_allclose(dm @ bundle.overlap @ dm, dm, atol=1e-10)
        assert np.trace(dm @ bundle.overlap) == pytest.approx(nocc, abs=1e-10)
