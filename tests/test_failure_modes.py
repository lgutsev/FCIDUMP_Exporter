"""Every way a bundle can be wrong, and what the user is told about it.

Two separate requirements are being tested here, and they are not the same one.

The first is that each malformation is *caught*. A bundle that is silently
accepted produces a FCIDUMP that a solver will happily consume for a week
before anyone notices the energies are nonsense, so every check below
corresponds to a way real data has been or could be mangled: a reader that
guessed a storage convention, a route line copied down wrong, a Kohn-Sham job
sent through the Hartree-Fock path.

The second is that the complaint is *scientific and actionable*. A traceback
ending in ``operands could not be broadcast together with shapes (12,12)
(11,11)`` tells a chemist nothing about their calculation. The rule this module
enforces is that the exception is one of the package's own, that it names the
quantity at fault, and that where there is a way out it says so. That rule is
checked twice: once per failure with the specific wording it owes, and once
across all of them in :func:`test_no_failure_mode_escapes_as_a_generic_exception`.

Two of the failure modes the plan lists are not here. A truncated or malformed
FCIDUMP header belongs with ``write.py`` and a non-orthogonal rotation matrix
with ``rotate.py``; both modules are seams at this commit and their failure
modes land with their implementations.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from g16dump import hamiltonian as H
from g16dump.bundle import BundleError, load, save, validate
from g16dump.hamiltonian import HamiltonianError

#: The exceptions this package is allowed to raise at a user. Anything else --
#: a bare ``IndexError`` from a slice, a ``ValueError`` from einsum, a
#: ``LinAlgError`` from a decomposition -- is a leak.
PACKAGE_ERRORS = (BundleError, HamiltonianError)


def assert_actionable(excinfo, *phrases):
    """The exception is one of ours, reads as a sentence, and names the physics."""
    exception = excinfo.value
    assert isinstance(exception, PACKAGE_ERRORS), (
        f"{type(exception).__name__} escaped; the user should see a "
        f"BundleError or a HamiltonianError, not a library's own exception"
    )
    message = str(exception)
    assert len(message) > 40, f"too terse to act on: {message!r}"
    lowered = message.lower()
    missing = [p for p in phrases if p.lower() not in lowered]
    assert not missing, (
        f"the message does not mention {missing}; it reads:\n{message}"
    )


# ---------------------------------------------------- 1. coefficient orientation

def test_transposed_coefficients_are_diagnosed_not_merely_refused(broken):
    """The single most likely reader bug, and the one with a specific remedy.

    ``C S C^T = I`` and ``C^T S C = I`` are both plausible-looking conditions,
    and a reader that reshapes a flat Gaussian array can satisfy either. Saying
    'not orthonormal' would leave the user to guess; the message has to name the
    convention and the fix.
    """
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, C=good.C.T))
    assert_actionable(excinfo, "row-wise", "column-wise", "transpose")


def test_nonorthonormal_coefficients_report_how_far_off_they_are(broken):
    """Not a convention problem: these coefficients are wrong in either reading."""
    good = broken()
    scrambled = good.C.copy()
    scrambled[:, 2] *= 1.7
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, C=scrambled))
    assert_actionable(excinfo, "MO coefficients", "not orthonormal", "overlap")


# ------------------------------------------------------------ 2. the overlap

def test_an_asymmetric_overlap_is_refused(broken):
    """``S`` is symmetric by construction; an asymmetric one is a reader bug."""
    good = broken()
    bad = good.S.copy()
    bad[0, 1] += 0.5
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, S=bad))
    assert_actionable(excinfo, "S is not symmetric")


def test_an_overlap_that_is_not_positive_definite_is_refused(broken):
    """A symmetric matrix that is not an overlap at all.

    Caught here rather than three functions later as a ``LinAlgError`` from
    whatever first tries to invert it.
    """
    good = broken()
    eigenvalues, eigenvectors = np.linalg.eigh(good.S)
    eigenvalues[0] = -1.0
    indefinite = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, S=indefinite))
    assert_actionable(excinfo, "S is not positive definite", "overlap matrix")


# ------------------------------------------- 3. non-Hermitian one-electron data

@pytest.mark.parametrize(
    "field", ["Hcore_ao", "F_alpha_ao", "F_beta_ao"]
)
def test_non_hermitian_one_electron_matrices_are_refused(broken, field):
    """Every one of these is a Hermitian operator in a real basis.

    Gaussian writes them to full double precision, so an asymmetry above the
    tolerance means the array was reshaped wrongly or read from the wrong
    record, not that precision was lost.
    """
    good = broken()
    bad = getattr(good, field).copy()
    bad[0, 1] += 1e-3
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, **{field: bad}))
    assert_actionable(excinfo, field, "symmetric")


# ------------------------------------------------------- 4. electron counts

def test_spin_populations_that_do_not_sum_to_the_electron_count(broken):
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, nalpha=4, nbeta=3, nelec=6, multiplicity=2))
    assert_actionable(excinfo, "nalpha", "nbeta", "nelec")


def test_a_multiplicity_of_the_wrong_parity_is_refused(broken):
    """Six electrons cannot make a doublet, whatever the input file says."""
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, multiplicity=2))
    assert_actionable(excinfo, "multiplicity", "parity")


def test_a_multiplicity_larger_than_the_electron_count_allows(broken):
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, nelec=2, nalpha=2, nbeta=0, multiplicity=9))
    assert_actionable(excinfo, "multiplicity", "unpaired")


def test_nuclear_charges_that_contradict_the_electron_count(broken):
    """A charge and a set of nuclei imply an electron count; it has to match."""
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, charge=3))
    assert_actionable(excinfo, "charge", "electrons")


# ------------------------------------------------- 5. the active window

def test_an_active_orbital_count_that_contradicts_the_window(broken):
    """``nact`` and the window are stored redundantly precisely so this is caught."""
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, nact=3))
    assert_actionable(excinfo, "nact", "window")


def test_a_window_running_past_the_last_orbital(broken):
    """The commonest route-line mistake: NLAST copied from a bigger basis."""
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, active_last=9, nact=8))
    assert_actionable(excinfo, "active window", "mos")


def test_a_frozen_core_count_that_disagrees_with_the_window(broken):
    """Every orbital below the window is frozen core; the two cannot differ."""
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, ncore=0))
    assert_actionable(excinfo, "ncore", "window", "frozen core")


def test_a_zero_based_window_is_caught_as_such(broken):
    """The window is 1-based; a 0 in ``active_first`` means someone converted.

    This is the specific shape of the version-1 mistake, and the one a caller
    is most likely to make by hand, so the message says which convention the
    field is in rather than only that the value is out of range.
    """
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, active_first=0, ncore=0, nact=5))
    assert_actionable(excinfo, "active_first", "1-based", "first mo is 1")


def test_an_empty_window_is_refused(broken):
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, active_first=5, active_last=4, nact=0, ncore=4))
    assert_actionable(excinfo, "active window", "empty")


def test_an_occupied_orbital_frozen_as_a_virtual(broken):
    """A window that closes below the highest occupied orbital.

    Physically impossible rather than merely small: the reference determinant
    would occupy an orbital the Hamiltonian does not contain.
    """
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, active_last=2, nact=1, orbital_energies=None,
                         eri_active=np.zeros((1, 1, 1, 1))))
    assert_actionable(excinfo, "alpha electrons", "window")


def test_a_core_too_deep_to_be_doubly_occupied(broken):
    """``ncore`` beta electrons are needed to fill ``ncore`` core orbitals."""
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, nelec=2, nalpha=2, nbeta=0, multiplicity=3,
                         atom_charges=None))
    assert_actionable(excinfo, "frozen core", "doubly occupied")


# --------------------------------------------------------- 6. the active ERIs

def test_an_eri_tensor_of_the_wrong_size_names_the_dimension_it_wanted(broken):
    """``nact`` says one thing and the two-electron block another.

    This is what a windowed Gaussian job produces when the route's window and
    the requested active space disagree, and the message has to say which
    shape was expected rather than failing later inside an einsum.
    """
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, eri_active=np.zeros((3, 3, 3, 3))))
    assert_actionable(excinfo, "eri_active", "shape", "expected")


def test_an_eri_tensor_of_the_wrong_rank_is_refused(broken):
    """A flat or half-unpacked block, which is how a packed record arrives."""
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, eri_active=np.zeros((4, 4))))
    assert_actionable(excinfo, "eri_active", "shape")


@pytest.mark.parametrize(
    "transpose, generator",
    [
        ((1, 0, 2, 3), "(tu|vw) = (ut|vw)"),
        ((0, 1, 3, 2), "(tu|vw) = (tu|wv)"),
        ((2, 3, 0, 1), "(tu|vw) = (vw|tu)"),
    ],
)
def test_each_permutational_symmetry_generator_is_checked(
    broken, transpose, generator
):
    """Real orbitals give 8-fold symmetry, generated by these three swaps.

    Breaking one at a time proves the check is not passing by accident on a
    tensor that happens to be symmetric under the others. Each corresponds to a
    real unpacking bug: a transposed index pair, a swapped bra and ket.
    """
    good = broken()
    corrupted = np.array(good.eri_active, copy=True)
    corrupted += 0.05 * (
        np.arange(corrupted.size).reshape(corrupted.shape)
        - np.arange(corrupted.size).reshape(corrupted.shape).transpose(transpose)
    )
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, eri_active=corrupted))
    assert_actionable(excinfo, "eri_active", generator)


# ------------------------------------------------------- 7. the Fock matrices

def test_no_fock_matrix_at_all_names_the_way_out(fixture_path):
    """A KS job, or any ``.mat`` without the AO Fock records.

    Not a dead end: the rebuilt-Fock path exists precisely for this, so the
    message has to name it.
    """
    bundle = replace(
        load(fixture_path("h2o_rhf")), F_alpha_ao=None, F_beta_ao=None,
        fock_source="none",
    )
    with pytest.raises(HamiltonianError) as excinfo:
        H.active_hamiltonian(bundle)
    assert_actionable(excinfo, "fock", "rebuild_fock")


def test_an_open_shell_bundle_missing_its_beta_fock_matrix(fixture_path):
    """The alpha matrix is not a substitute when the spin densities differ.

    Silently reusing it would make the alpha/beta consistency check compare a
    matrix with itself, which is the one thing that would disarm the gate
    without any visible symptom.
    """
    bundle = replace(load(fixture_path("ch2_rohf")), F_beta_ao=None)
    with pytest.raises(HamiltonianError) as excinfo:
        H.active_hamiltonian(bundle)
    assert_actionable(excinfo, "beta fock", "open-shell")


def test_a_closed_shell_bundle_may_omit_its_beta_fock_matrix(fixture_path):
    """The complement: when the two are equal, omitting one is not an error."""
    bundle = replace(load(fixture_path("h2o_rhf")), F_beta_ao=None)
    assert bundle.nalpha == bundle.nbeta
    result = H.active_hamiltonian(bundle)
    assert result.e_ref == pytest.approx(bundle.escf, abs=1e-8)


def test_a_stored_fock_matrix_with_no_recorded_source(broken):
    """Provenance is not optional: a Hamiltonian must say where it came from."""
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, fock_source="none"))
    assert_actionable(excinfo, "fock_source", "source")


def test_a_recorded_source_with_no_stored_fock_matrix(broken):
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, F_alpha_ao=None, F_beta_ao=None))
    assert_actionable(excinfo, "fock_source", "F_alpha_ao")


# --------------------------------- 8. the alpha/beta consistency failure itself

def test_the_roothaan_operator_fails_the_consistency_check_loudly(fixture_path):
    """The failure this whole design is built around.

    Gaussian's ROHF ``.mat`` carries one effective operator, not ``F^alpha``
    and ``F^beta``. Substituting it produces a spin-dependent ``h'``, which is
    impossible by construction, so the disagreement is a reliable signal. The
    message must name the cause and forbid the obvious wrong fix.
    """
    with pytest.raises(H.SpinConsistencyError) as excinfo:
        H.active_hamiltonian(load(fixture_path("ch2_rohf_roothaan")))
    assert_actionable(
        excinfo, "disagree", "do not average", "rebuild_fock", "roothaan"
    )
    assert excinfo.value.deviation > 0.1, (
        "a gate that only just fires is not a gate"
    )


def test_the_consistency_tolerance_is_configurable_but_not_bypassable(fixture_path):
    """Loosening it far enough lets the bad bundle through, which is the point.

    The tolerance is a knob for numerical noise, and it is being shown here
    that the Roothaan failure is nowhere near it: the default rejects, and only
    a tolerance larger than the disagreement itself accepts.
    """
    bundle = load(fixture_path("ch2_rohf_roothaan"))
    with pytest.raises(H.SpinConsistencyError) as excinfo:
        H.active_hamiltonian(bundle, spin_tol=1e-8)
    deviation = excinfo.value.deviation
    accepted = H.active_hamiltonian(bundle, spin_tol=10 * deviation)
    assert accepted.spin_deviation == pytest.approx(deviation)


def test_a_disagreeing_pair_is_never_quietly_averaged(fixture_path):
    """``h'`` must be a matrix that was computed, not a compromise.

    Averaging would produce a plausible-looking Hamiltonian from two wrong
    ones, which is worse than either, so the mean is asserted *not* to be what
    comes back when the tolerance is opened up.
    """
    bundle = load(fixture_path("ch2_rohf_roothaan"))
    fock_a, fock_b = H.mo_fock_matrices(bundle)
    from_alpha, from_beta = H.effective_one_electron(
        fock_a, fock_b, bundle.eri_active, bundle.active,
        bundle.nocc_active_alpha, bundle.nocc_active_beta,
    )
    mean = 0.5 * (from_alpha + from_beta)
    result = H.active_hamiltonian(bundle, spin_tol=1e3)
    assert float(np.max(np.abs(result.h_eff - mean))) > 1e-3
    np.testing.assert_allclose(result.h_eff, from_alpha)


# ----------------------------------------------- 9. Kohn-Sham sent the wrong way

def test_a_ks_bundle_carrying_a_gaussian_fock_matrix_is_refused(broken):
    """Caught at validation, before anything reads the matrix."""
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, reference_type="RKS"))
    assert_actionable(excinfo, "kohn-sham", "exchange-correlation", "rebuilt")


def test_ks_orbitals_reaching_the_algebra_with_a_stored_fock_are_refused(broken):
    """The second line of defence, in case a bundle is built in memory.

    ``validate`` guards the file; this guards the function. A KS bundle that
    somehow arrived with ``fock_source="gaussian"`` must not produce numbers.
    """
    good = broken()
    with pytest.raises(HamiltonianError) as excinfo:
        H.active_hamiltonian(replace(good, reference_type="RKS", fock_source="gaussian"))
    assert_actionable(
        excinfo, "exchange-correlation", "not the hf fock", "rebuild_fock"
    )


def test_an_unknown_reference_type_is_refused_rather_than_guessed(broken):
    """Reference types are never inferred; an unrecognised one stops the run."""
    good = broken()
    with pytest.raises(BundleError) as excinfo:
        validate(replace(good, reference_type="UHF"))
    assert_actionable(excinfo, "reference", "never inferred")


# ------------------------------------------------------ 10. the file on disk

@pytest.mark.parametrize("version", [1, 3, 99])
def test_a_bundle_of_another_schema_version_is_refused(synthetic, tmp_path, version):
    """Version 2 is what this code speaks; anything else is a different format.

    Version 1 is not hypothetical. Bundles written before the field rename are
    still on disk, and they store the active window 0-based and half-open where
    version 2 stores it 1-based and inclusive. Nothing about such a file is the
    wrong shape: read as version 2 it would silently drop the first active
    orbital, take a virtual in its place, and produce a Hamiltonian for an
    active space nobody asked for. Refusing on the version is the only thing
    standing between that file and a plausible wrong answer.
    """
    path = save(synthetic, tmp_path / "bundle.npz")
    payload = dict(np.load(path, allow_pickle=False))
    payload["schema_version"] = np.int64(version)
    np.savez_compressed(path, **payload)

    with pytest.raises(BundleError) as excinfo:
        load(path)
    assert_actionable(excinfo, "schema version", str(version))


def test_a_bundle_with_no_schema_version_at_all(synthetic, tmp_path):
    """An ``.npz`` from somewhere else entirely, or from before the schema."""
    path = save(synthetic, tmp_path / "bundle.npz")
    payload = {k: v for k, v in np.load(path, allow_pickle=False).items()
               if k != "schema_version"}
    np.savez_compressed(path, **payload)

    with pytest.raises(BundleError) as excinfo:
        load(path)
    assert_actionable(excinfo, "schema_version", "not a g16dump bundle")


def test_a_truncated_bundle_lists_every_key_it_is_missing(synthetic, tmp_path):
    """Named all at once, so a truncated file takes one round trip to diagnose."""
    path = save(synthetic, tmp_path / "bundle.npz")
    dropped = {"S", "Hcore_ao", "nalpha"}
    payload = {k: v for k, v in np.load(path, allow_pickle=False).items()
               if k not in dropped}
    np.savez_compressed(path, **payload)

    with pytest.raises(BundleError) as excinfo:
        load(path)
    assert_actionable(excinfo, "missing required keys", *sorted(dropped))


def test_unparseable_provenance_is_reported_as_such(synthetic, tmp_path):
    """Provenance is JSON in the file; corruption there is not a numeric fault."""
    path = save(synthetic, tmp_path / "bundle.npz")
    payload = dict(np.load(path, allow_pickle=False))
    payload["provenance"] = np.asarray("{not json")
    np.savez_compressed(path, **payload)

    with pytest.raises(BundleError) as excinfo:
        load(path)
    assert_actionable(excinfo, "provenance", "json")


def test_an_invalid_bundle_is_never_written_to_disk(broken, tmp_path):
    """Writing a file that cannot be read back is never what a caller wanted."""
    path = tmp_path / "never.npz"
    with pytest.raises(BundleError):
        save(replace(broken(), nact=2), path)
    assert not path.exists(), "a rejected bundle left a file behind"


def test_a_bundle_round_trips_through_a_corrupted_provenance_unchanged(synthetic):
    """The complement: valid JSON that happens to be unusual is not an error."""
    odd = replace(synthetic, provenance={"route": "#p rohf/6-31g", "nested": {"a": 1}})
    validate(odd)
    assert json.loads(json.dumps(odd.provenance)) == odd.provenance


# ----------------------------------------- the discipline, across every mode

#: Every corruption above, as (label, mutation) pairs. The individual tests
#: assert what each message says; this table asserts what none of them may be.
CORRUPTIONS = [
    ("transposed coefficients", lambda b: dict(C=b.C.T)),
    ("asymmetric overlap", lambda b: dict(S=b.S + np.triu(
        np.ones_like(b.S), 1))),
    ("non-Hermitian hcore", lambda b: dict(Hcore_ao=b.Hcore_ao + np.triu(
        np.ones_like(b.Hcore_ao), 1))),
    ("electron count", lambda b: dict(nelec=7)),
    ("impossible multiplicity", lambda b: dict(multiplicity=2)),
    ("window contradicts nact", lambda b: dict(nact=2)),
    ("window past the last MO", lambda b: dict(active_last=99)),
    ("eri of the wrong size", lambda b: dict(eri_active=np.zeros((2, 2, 2, 2)))),
    ("eri of the wrong rank", lambda b: dict(eri_active=np.zeros(16))),
    ("eri symmetry broken", lambda b: dict(
        eri_active=b.eri_active + np.arange(b.eri_active.size).reshape(b.eri_active.shape))),
    ("mo_coeff of the wrong shape", lambda b: dict(
        C=np.zeros((b.nao, b.nmo + 1)))),
    ("unknown reference", lambda b: dict(reference_type="CASSCF")),
    ("KS with a stored Fock", lambda b: dict(reference_type="ROKS")),
    ("non-finite integrals", lambda b: dict(
        Hcore_ao=np.full_like(b.Hcore_ao, np.nan))),
]


@pytest.mark.parametrize(
    "label, mutation", CORRUPTIONS, ids=[c[0] for c in CORRUPTIONS]
)
def test_no_failure_mode_escapes_as_a_generic_exception(synthetic, label, mutation):
    """The rule, stated once over every corruption this module knows about.

    Whatever is wrong with the input, what reaches the user is a
    :class:`BundleError` naming the quantity at fault -- never an ``IndexError``
    from a slice, a broadcasting complaint from NumPy, or a ``LinAlgError`` from
    a decomposition that was handed something that is not a matrix.
    """
    with pytest.raises(PACKAGE_ERRORS) as excinfo:
        validate(replace(synthetic, **mutation(synthetic)))
    assert_actionable(excinfo, "failed validation")


def test_validation_reports_every_problem_at_once(synthetic):
    """One round trip per file, not one per mistake.

    A file that was read with the wrong conventions is usually wrong in several
    ways at once, and fixing them one exception at a time is how an afternoon
    disappears.
    """
    with pytest.raises(BundleError) as excinfo:
        validate(replace(synthetic, nelec=99, multiplicity=4, nact=2))
    message = str(excinfo.value)
    assert message.count("\n  - ") >= 3, message
