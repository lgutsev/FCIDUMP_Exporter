"""The ``.npz`` schema and its validation.

The bundle is the contract between the Gaussian reader and everything
downstream, so these tests are mostly about what it *refuses*. A bundle that is
merely wrong -- transposed coefficients, a window that does not match the
integrals it carries, an impossible spin state -- produces a plausible number
rather than an error, and a plausible wrong number is the failure mode this
project exists to eliminate.

Nothing here needs PySCF.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from g16dump import bundle as B


def _message(excinfo) -> str:
    return str(excinfo.value)


# ------------------------------------------------------------ the happy path

def test_synthetic_bundle_validates(synthetic):
    """The fixture every rejection test perturbs must itself be clean."""
    B.validate(synthetic)


def test_round_trip_preserves_everything(synthetic, tmp_path):
    path = B.save(synthetic, tmp_path / "round.npz")
    restored = B.load(path)

    for name in ("reference_type", "charge", "multiplicity", "nelec", "nalpha",
                 "nbeta", "nao", "nmo", "ncore", "nact", "active_first",
                 "active_last", "source_program", "source_file", "fock_source",
                 "schema_version"):
        assert getattr(restored, name) == getattr(synthetic, name), name
    assert restored.enuc == pytest.approx(synthetic.enuc)
    assert restored.escf == pytest.approx(synthetic.escf)
    for name in ("C", "S", "Hcore_ao", "eri_active", "F_alpha_ao",
                 "F_beta_ao", "orbital_energies", "atom_charges"):
        np.testing.assert_allclose(
            getattr(restored, name), getattr(synthetic, name), atol=0, rtol=0
        )
    assert restored.provenance == synthetic.provenance


def test_absent_optionals_stay_absent(synthetic, tmp_path):
    """An unknown quantity is missing, never a placeholder that reads as data."""
    sparse = B.validate  # keep the name bound for clarity below
    del sparse
    from dataclasses import replace

    bundle = replace(
        synthetic, escf=None, orbital_energies=None, atom_charges=None
    )
    restored = B.load(B.save(bundle, tmp_path / "sparse.npz"))
    assert restored.escf is None
    assert restored.orbital_energies is None
    assert restored.atom_charges is None


def test_derived_views(synthetic):
    """The accessors other threads use instead of touching raw keys."""
    assert synthetic.active == slice(1, 5)
    assert synthetic.core == slice(0, 1)
    assert synthetic.nocc_active_alpha == 2
    assert synthetic.nocc_active_beta == 2
    assert synthetic.nelec_active == 4
    assert synthetic.ms2 == 0
    assert synthetic.nfrozen_virtual == 1
    assert synthetic.is_ks is False
    assert synthetic.has_fock is True
    assert "active window MOs 2-5" in synthetic.describe()


def test_provenance_records_commit_and_schema():
    record = B.make_provenance(source_file="job.mat", fock_source="gaussian")
    assert record["schema_version"] == B.SCHEMA_VERSION
    assert record["source_file"] == "job.mat"
    assert record["fock_source"] == "gaussian"
    # Unknown fields stay out rather than being invented.
    assert "basis" not in record
    assert "g16dump_commit" in record


# ------------------------------------------------ coefficient orientation

def test_transposed_coefficients_are_named_not_just_rejected(broken):
    """The single most likely reader bug gets a message that says what to do."""
    good = broken()
    bundle = broken(C=np.ascontiguousarray(good.C.T))
    with pytest.raises(B.BundleError) as excinfo:
        B.validate(bundle)
    message = _message(excinfo)
    assert "row-wise" in message
    assert "transpose" in message


def test_non_orthonormal_coefficients_are_rejected(broken):
    good = broken()
    scrambled = good.C.copy()
    scrambled[:, 2] *= 1.5
    with pytest.raises(B.BundleError, match="not orthonormal"):
        B.validate(broken(C=scrambled))


# ---------------------------------------------------- one-electron matrices

@pytest.mark.parametrize("name", ["S", "Hcore_ao", "F_alpha_ao",
                                  "F_beta_ao"])
def test_non_hermitian_one_electron_matrices_are_rejected(broken, name):
    good = broken()
    matrix = np.array(getattr(good, name), copy=True)
    matrix[0, 1] += 1e-3
    with pytest.raises(B.BundleError, match="not symmetric"):
        B.validate(broken(**{name: matrix}))


def test_overlap_must_be_positive_definite(broken):
    good = broken()
    overlap = np.array(good.S, copy=True)
    overlap[0, 0] = -overlap[0, 0]
    with pytest.raises(B.BundleError) as excinfo:
        B.validate(broken(S=overlap))
    message = _message(excinfo)
    assert "positive definite" in message or "orthonormal" in message


def test_non_finite_values_are_rejected(broken):
    good = broken()
    hcore = np.array(good.Hcore_ao, copy=True)
    hcore[2, 2] = np.nan
    with pytest.raises(B.BundleError, match="non-finite"):
        B.validate(broken(Hcore_ao=hcore))


# --------------------------------------------------------- ERI symmetry

@pytest.mark.parametrize(
    "index, expected",
    [
        ((0, 1, 2, 2), r"\(tu\|vw\) = \(ut\|vw\)"),
        ((0, 0, 1, 2), r"\(tu\|vw\) = \(tu\|wv\)"),
    ],
)
def test_corrupted_eri_symmetry_is_caught(broken, index, expected):
    """A corrupted permutation is silent in every energy until it is not."""
    good = broken()
    eri = np.array(good.eri_active, copy=True)
    eri[index] += 0.25
    with pytest.raises(B.BundleError, match=expected):
        B.validate(broken(eri_active=eri))


def test_pair_exchange_asymmetry_is_caught(broken):
    good = broken()
    eri = np.array(good.eri_active, copy=True)
    eri[0, 1, 2, 3] += 0.25
    eri[1, 0, 2, 3] += 0.25
    eri[0, 1, 3, 2] += 0.25
    eri[1, 0, 3, 2] += 0.25
    with pytest.raises(B.BundleError, match=r"\(tu\|vw\) = \(vw\|tu\)"):
        B.validate(broken(eri_active=eri))


# ------------------------------------------------------------ electron counts

@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"nelec": 7}, "but nelec = 7"),
        ({"multiplicity": 2}, "implies 1"),
        ({"nalpha": 2, "nbeta": 4, "multiplicity": 1}, "nalpha .* < nbeta"),
        ({"multiplicity": 0}, "not a positive integer"),
        ({"nalpha": 4, "nbeta": 2, "multiplicity": 3, "nelec": 6,
          "atom_charges": np.array([4.0, 1.0, 1.0])}, None),
        ({"charge": 2}, "implies 4 electrons"),
    ],
)
def test_impossible_electron_counts_are_rejected(broken, changes, expected):
    if expected is None:
        pytest.skip("this combination is consistent; kept to document the boundary")
    with pytest.raises(B.BundleError, match=expected):
        B.validate(broken(**changes))


def test_parity_of_multiplicity_against_electron_count(broken):
    """5 electrons cannot be a singlet, whatever nalpha and nbeta claim."""
    with pytest.raises(B.BundleError) as excinfo:
        B.validate(broken(nelec=5, nalpha=3, nbeta=2, multiplicity=1))
    message = _message(excinfo)
    assert "multiplicity" in message


# ------------------------------------------------------------- active window

@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"nact": 3}, "spans 4 orbitals"),
        ({"ncore": 0}, "window starts at MO 2"),
        ({"active_last": 9, "nact": 8}, "only 6 MOs"),
        ({"active_first": 6, "active_last": 5}, "empty"),
        ({"active_first": 0}, "the window is 1-based"),
        ({"active_first": 5, "ncore": 4, "nact": 1, "active_last": 5},
         "would not be doubly occupied"),
    ],
)
def test_bad_active_windows_are_rejected(broken, changes, expected):
    with pytest.raises(B.BundleError, match=expected):
        B.validate(broken(**changes))


def test_the_window_is_one_based_and_inclusive(synthetic):
    """active_first/active_last are NFIRST/NLAST from the Gaussian route."""
    assert (synthetic.active_first, synthetic.active_last) == (2, 5)
    assert synthetic.nact == synthetic.active_last - synthetic.active_first + 1
    # and the accessors are the only place that becomes 0-based indexing
    assert synthetic.active == slice(1, 5)
    assert (synthetic.active_start, synthetic.active_stop) == (1, 5)
    assert synthetic.core == slice(0, 1)
    assert synthetic.nfrozen_virtual == 1


def test_an_off_by_one_window_does_not_pass_quietly(broken):
    """Reading active_first as 0-based would give ncore one too many."""
    with pytest.raises(B.BundleError, match="frozen core"):
        B.validate(broken(active_first=3))


def test_occupied_frozen_virtual_is_rejected(broken):
    """An electron above the window means the window is not the active space."""
    with pytest.raises(B.BundleError, match="frozen virtual cannot be occupied"):
        B.validate(broken(nelec=12, nalpha=6, nbeta=6))


def test_shape_mismatch_reports_the_dimension_it_expected(broken):
    good = broken()
    with pytest.raises(B.BundleError) as excinfo:
        B.validate(broken(eri_active=good.eri_active[:3, :3, :3, :3]))
    assert "expected (4, 4, 4, 4)" in _message(excinfo)
    assert "nact x nact x nact x nact" in _message(excinfo)


# ---------------------------------------------------------- Fock bookkeeping

def test_ks_reference_may_not_carry_a_gaussian_fock(broken):
    with pytest.raises(B.BundleError, match="exchange-correlation"):
        B.validate(broken(reference_type="RKS"))


def test_ks_reference_with_a_rebuilt_fock_is_fine(broken):
    B.validate(broken(reference_type="RKS", fock_source="pyscf_rebuilt"))


def test_a_fock_matrix_without_a_source_is_rejected(broken):
    with pytest.raises(B.BundleError, match="source of every stored Fock"):
        B.validate(broken(fock_source="none"))


def test_a_source_without_a_fock_matrix_is_rejected(broken):
    with pytest.raises(B.BundleError, match="no F_alpha_ao is stored"):
        B.validate(broken(F_alpha_ao=None, F_beta_ao=None))


def test_unknown_reference_is_rejected(broken):
    with pytest.raises(B.BundleError, match="never inferred"):
        B.validate(broken(reference_type="UHF"))


# --------------------------------------------------------------- file errors

def test_load_rejects_a_foreign_npz(tmp_path):
    path = tmp_path / "not_a_bundle.npz"
    np.savez(path, something=np.zeros(3))
    with pytest.raises(B.BundleError, match="no schema_version"):
        B.load(path)


def test_load_rejects_a_future_schema(synthetic, tmp_path):
    path = B.save(synthetic, tmp_path / "future.npz")
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    payload["schema_version"] = np.int64(B.SCHEMA_VERSION + 1)
    np.savez(path, **payload)
    with pytest.raises(B.BundleError, match="speaks version"):
        B.load(path)


def test_load_reports_missing_keys_by_name(synthetic, tmp_path):
    path = B.save(synthetic, tmp_path / "truncated.npz")
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files if key != "eri_active"}
    np.savez(path, **payload)
    with pytest.raises(B.BundleError, match="missing required keys: eri_act"):
        B.load(path)


def test_load_rejects_unparseable_provenance(synthetic, tmp_path):
    path = B.save(synthetic, tmp_path / "badprov.npz")
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    payload["provenance"] = np.asarray("{not json")
    np.savez(path, **payload)
    with pytest.raises(B.BundleError, match="not valid JSON"):
        B.load(path)


def test_save_refuses_to_write_an_invalid_bundle(broken, tmp_path):
    """A bundle that cannot be loaded back was never worth writing."""
    with pytest.raises(B.BundleError):
        B.save(broken(nact=3), tmp_path / "never.npz")
    assert not (tmp_path / "never.npz").exists()


def test_validation_reports_every_problem_at_once(broken):
    """One run of the reader should surface every fault, not the first one."""
    good = broken()
    hcore = np.array(good.Hcore_ao, copy=True)
    hcore[0, 1] += 1e-3
    with pytest.raises(B.BundleError) as excinfo:
        B.validate(broken(Hcore_ao=hcore, multiplicity=0))
    message = _message(excinfo)
    assert "not symmetric" in message
    assert "not a positive integer" in message


# ------------------------------------------------------- committed fixtures

@pytest.mark.parametrize("name", ["h2o_rhf", "ch2_rohf", "ch2_rohf_roothaan",
                                  "ch2_rohf_rotated"])
def test_committed_fixtures_load_and_validate(fixture_path, name):
    bundle = B.load(fixture_path(name))
    assert bundle.schema_version == B.SCHEMA_VERSION
    assert bundle.nact == bundle.active_last - bundle.active_first + 1
    assert json.dumps(bundle.provenance)  # provenance survives the round trip
