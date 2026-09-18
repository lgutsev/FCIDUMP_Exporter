"""Bundle schema, I/O, and the failure modes validation exists to catch.

Every test here asserts on the *message*, not just the exception type. An error
that says "shapes (7,7) and (6,6) not aligned" is useless to someone holding a
Gaussian job that ran for two days; an error that names the offending quantity,
the measured value and the fix is the actual deliverable.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from g16dump.bundle import (
    SCHEMA_VERSION,
    Bundle,
    ao_major,
    detect_coefficient_orientation,
    eri_symmetry_error,
    load,
    load_and_validate,
    save,
    validate,
)
from g16dump.errors import ReferenceTypeError, SchemaError, ValidationError
from synthetic import make_system


@pytest.fixture
def bundle():
    return make_system(nbasis=8, nalpha=5, nbeta=3, ncore=1, ref_type="ROHF", seed=71).bundle


# ------------------------------------------------------------------- round trip


def test_save_and_load_round_trip(bundle, tmp_path):
    path = tmp_path / "b.npz"
    save(bundle, str(path))
    back = load(str(path))

    for field in ("ref_type", "fock_source", "nelec", "nalpha", "nbeta",
                  "ncore", "nact", "nfv", "nmo", "nbasis", "multiplicity",
                  "charge", "schema_version"):
        assert getattr(back, field) == getattr(bundle, field), field
    assert abs(back.e_nuc - bundle.e_nuc) < 1e-15
    for field in ("h_ao", "s_ao", "c_a", "c_b", "eri_act", "f_a_ao", "f_b_ao"):
        assert np.array_equal(getattr(back, field), getattr(bundle, field)), field
    assert back.provenance["seed"] == bundle.provenance["seed"]


def test_optional_fields_survive_being_absent(bundle, tmp_path):
    stripped = dataclasses.replace(
        bundle, e_scf=None, mo_energies_a=None, f_a_ao=None, f_b_ao=None,
        fock_source="none",
    )
    path = tmp_path / "b.npz"
    save(stripped, str(path))
    back = load(str(path))
    assert back.e_scf is None
    assert back.f_a_ao is None
    assert back.has_fock is False


def test_load_and_validate_is_the_normal_entry_point(bundle, tmp_path):
    path = tmp_path / "b.npz"
    save(bundle, str(path))
    assert load_and_validate(str(path)).nact == bundle.nact


def test_save_refuses_to_write_an_invalid_bundle(bundle, tmp_path):
    broken = dataclasses.replace(bundle, nelec=bundle.nelec + 1)
    with pytest.raises(ValidationError):
        save(broken, str(tmp_path / "b.npz"))
    assert not (tmp_path / "b.npz").exists()


# --------------------------------------------------------- corrupted schemas


def test_missing_schema_version_is_reported(tmp_path):
    path = tmp_path / "notabundle.npz"
    np.savez(path, something=np.zeros(3))
    with pytest.raises(SchemaError, match="no schema_version key"):
        load(str(path))


def test_wrong_schema_version_is_reported(bundle, tmp_path):
    path = tmp_path / "b.npz"
    save(bundle, str(path))
    with np.load(path) as data:
        payload = {k: data[k] for k in data.files}
    payload["schema_version"] = np.array(SCHEMA_VERSION + 99)
    np.savez(path, **payload)

    with pytest.raises(SchemaError, match=f"declares schema version {SCHEMA_VERSION + 99}"):
        load(str(path))


def test_truncated_bundle_names_the_missing_keys(bundle, tmp_path):
    path = tmp_path / "b.npz"
    save(bundle, str(path))
    with np.load(path) as data:
        payload = {k: data[k] for k in data.files if k not in ("eri_act", "c_b")}
    np.savez(path, **payload)

    with pytest.raises(SchemaError) as excinfo:
        load(str(path))
    message = str(excinfo.value)
    assert "c_b" in message and "eri_act" in message
    assert "re-run `g16dump extract`" in message


# ------------------------------------------------- electron and spin counts


def test_electron_counts_must_add_up(bundle):
    with pytest.raises(ValidationError, match="does not equal nelec"):
        validate(dataclasses.replace(bundle, nelec=bundle.nelec + 1))


def test_multiplicity_must_match_the_spin_difference(bundle):
    with pytest.raises(ValidationError, match="inconsistent with multiplicity"):
        validate(dataclasses.replace(bundle, multiplicity=bundle.multiplicity + 1))


def test_multiplicity_must_be_positive(bundle):
    with pytest.raises(ValidationError, match="multiplicity must be >= 1"):
        validate(dataclasses.replace(bundle, multiplicity=0))


def test_charge_is_checked_against_nuclear_charges(bundle):
    with_atoms = dataclasses.replace(
        bundle, atom_charges=np.array([8.0, 1.0, 1.0]), charge=0
    )
    with pytest.raises(ValidationError, match="charge is inconsistent"):
        validate(with_atoms)


# ---------------------------------------------------------- the active window


def test_window_must_tile_the_mo_space(bundle):
    with pytest.raises(ValidationError, match="does not tile the MO space"):
        validate(dataclasses.replace(bundle, nact=bundle.nact + 1))


def test_core_cannot_exceed_the_beta_electrons(bundle):
    """A frozen core is doubly occupied, so ncore <= nbeta."""
    broken = dataclasses.replace(
        bundle, ncore=bundle.nbeta + 1, nact=bundle.nmo - bundle.nbeta - 1, nfv=0
    )
    with pytest.raises(ValidationError, match="frozen core must be doubly occupied"):
        validate(broken)


def test_window_must_contain_the_occupied_orbitals(bundle):
    """An active window narrower than the reference occupation is incoherent."""
    narrow = dataclasses.replace(
        bundle, nact=1, nfv=bundle.nmo - bundle.ncore - 1
    )
    with pytest.raises(ValidationError, match="excludes orbitals that are occupied"):
        validate(narrow)


def test_empty_active_space_is_refused(bundle):
    with pytest.raises(ValidationError, match="active space is empty"):
        validate(
            dataclasses.replace(
                bundle, nact=0, nfv=bundle.nmo - bundle.ncore,
                eri_act=np.zeros((0, 0, 0, 0)),
            )
        )


def test_nmo_cannot_exceed_nbasis(bundle):
    with pytest.raises(ValidationError, match="exceeds nbasis"):
        validate(
            dataclasses.replace(
                bundle, nmo=bundle.nbasis + 1, nfv=bundle.nfv + 1
            )
        )


# ------------------------------------------------------- matrices and shapes


def test_mismatched_shapes_are_named(bundle):
    with pytest.raises(ValidationError, match="h_ao has shape"):
        validate(dataclasses.replace(bundle, h_ao=np.eye(bundle.nbasis + 1)))


def test_non_hermitian_one_electron_matrix_is_refused(bundle):
    broken = bundle.h_ao.copy()
    broken[0, 1] += 0.5
    with pytest.raises(ValidationError, match="h_ao is not symmetric"):
        validate(dataclasses.replace(bundle, h_ao=broken))


def test_non_hermitian_fock_is_refused(bundle):
    broken = bundle.f_a_ao.copy()
    broken[2, 0] += 0.1
    with pytest.raises(ValidationError, match="f_a_ao is not symmetric"):
        validate(dataclasses.replace(bundle, f_a_ao=broken))


def test_overlap_must_be_positive_definite(bundle):
    bad = bundle.s_ao.copy()
    bad[0, 0] = -1.0
    with pytest.raises(ValidationError, match="not positive definite"):
        validate(dataclasses.replace(bundle, s_ao=bad))


# ------------------------------------------------- coefficient orientation


def test_transposed_coefficients_are_detected_and_explained(bundle):
    """The failure mode that yields a plausible but wrong Hamiltonian."""
    transposed = dataclasses.replace(bundle, c_a=ao_major(bundle.c_a))
    with pytest.raises(ValidationError) as excinfo:
        validate(transposed)
    message = str(excinfo.value)
    assert "AO-major" in message
    assert "do not transpose downstream" in message.lower()


def test_orientation_detection_measures_rather_than_assumes(bundle):
    assert detect_coefficient_orientation(bundle.c_a, bundle.s_ao) == "mo-major"
    assert detect_coefficient_orientation(ao_major(bundle.c_a), bundle.s_ao) == "ao-major"
    assert detect_coefficient_orientation(np.eye(bundle.nbasis), bundle.s_ao) == "neither"


def test_non_orthonormal_coefficients_are_refused(bundle):
    scaled = bundle.c_a.copy()
    scaled[0] *= 1.5
    with pytest.raises(ValidationError, match="not orthonormal in either orientation"):
        validate(dataclasses.replace(bundle, c_a=scaled))


def test_ao_major_is_its_own_inverse(bundle):
    assert np.array_equal(ao_major(ao_major(bundle.c_a)), bundle.c_a)


# ------------------------------------------------------------ ERI symmetry


def test_corrupted_eri_symmetry_is_refused(bundle):
    broken = bundle.eri_act.copy()
    broken[0, 1, 2, 3] += 0.25  # breaks (pq|rs) = (qp|rs)
    with pytest.raises(ValidationError, match="8-fold permutational symmetry"):
        validate(dataclasses.replace(bundle, eri_act=broken))


def test_eri_symmetry_error_reports_the_size_of_the_violation(bundle):
    assert eri_symmetry_error(bundle.eri_act) < 1e-12
    broken = bundle.eri_act.copy()
    broken[0, 1, 2, 3] += 0.25
    assert eri_symmetry_error(broken) == pytest.approx(0.25, abs=1e-12)


def test_partial_index_transposition_is_caught(bundle):
    """Swapping two indices is not a symmetry of (pq|rs) and must be detected."""
    with pytest.raises(ValidationError, match="C-order vs Fortran-order"):
        validate(
            dataclasses.replace(
                bundle, eri_act=np.ascontiguousarray(bundle.eri_act.transpose(0, 2, 1, 3))
            )
        )


def test_full_index_reversal_is_invisible_to_the_symmetry_check(bundle):
    """A C-order/Fortran-order reshape is NOT detectable by symmetry alone.

    Reshaping a flat (nact,)*4 array with the wrong memory order is exactly
    ``transpose(3, 2, 1, 0)``, and that permutation is one of the eight
    operations the 8-fold symmetry already guarantees -- so the tensor comes out
    numerically identical and no symmetry check can ever see the mistake.

    This is why index order is verified against an independent code
    (``test_vs_pyscf.py``) rather than being inferred from symmetry. The test is
    here to keep that reasoning from being quietly forgotten.
    """
    flipped = bundle.eri_act.ravel().reshape(bundle.eri_act.shape, order="F")
    assert np.array_equal(flipped, bundle.eri_act.transpose(3, 2, 1, 0))
    assert np.max(np.abs(flipped - bundle.eri_act)) < 1e-15
    validate(dataclasses.replace(bundle, eri_act=np.ascontiguousarray(flipped)))


# ------------------------------------------------------------ reference type


def test_unknown_reference_type_is_refused(bundle):
    with pytest.raises(ValidationError, match="unknown reference type"):
        validate(dataclasses.replace(bundle, ref_type="MP2"))


def test_unknown_fock_source_is_refused(bundle):
    with pytest.raises(ValidationError, match="unknown fock_source"):
        validate(dataclasses.replace(bundle, fock_source="magic"))


def test_restricted_reference_must_share_spatial_orbitals(bundle):
    """Different alpha and beta orbitals mean UHF, which v1 cannot represent."""
    different = bundle.c_b.copy()
    different[0] = -different[0]
    with pytest.raises(ValidationError, match="UHF-like reference"):
        validate(dataclasses.replace(bundle, c_b=different))


def test_bundle_accessors():
    system = make_system(nbasis=8, nalpha=5, nbeta=3, ncore=1, ref_type="ROHF", seed=72)
    b = system.bundle
    assert b.nocc_a_act == 4 and b.nocc_b_act == 2
    assert b.nelec_act == 6 and b.ms2 == 2
    assert b.active == slice(1, 1 + b.nact)
    assert b.has_fock is True
    assert b.is_ks is False
    assert dataclasses.replace(b, ref_type="ROKS").is_ks is True
