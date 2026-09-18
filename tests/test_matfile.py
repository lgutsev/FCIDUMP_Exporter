"""The Gaussian matrix-element reader.

gauopen is not installable in CI, so these tests drive :func:`g16dump.matfile.extract`
against an in-process stand-in for a ``MatEl`` object, built from the committed
``h2o_rhf`` fixture. The stand-in reproduces the awkward parts of the real
thing: lower-triangular packing on the one-electron blocks, a flat ``nact**4``
two-electron block, and MO coefficients stored in the orientation the reader
must *not* assume.

The label names themselves remain unconfirmed against a real ``.mat``; that is
the M0 gate. What is tested here is everything around them -- that a missing
block is reported with the file's actual contents, that the coefficient
convention is decided numerically, and that a window disagreeing with the
integrals it is supposed to describe stops the run instead of being absorbed.
"""

from __future__ import annotations

import numpy as np
import pytest

from g16dump import matfile as M
from g16dump.bundle import load


class _Entry:
    """One matlist entry. Packed ones expand; flat ones do not, like the real API."""

    def __init__(self, array, packed_dimension=None):
        self.array = np.asarray(array, dtype=np.float64)
        self._n = packed_dimension
        if packed_dimension is not None:
            self.expand = self._expand

    def _expand(self):
        n = self._n
        out = np.zeros((n, n))
        rows, cols = np.tril_indices(n)
        out[rows, cols] = self.array
        return out + out.T - np.diag(np.diag(out))


def _pack(matrix):
    rows, cols = np.tril_indices(matrix.shape[0])
    return matrix[rows, cols]


class FakeMatEl:
    """A MatEl-shaped object carrying a real H2O RHF job."""

    def __init__(self, bundle, *, transpose_coefficients=True, drop=(), extra=None,
                 scalars=None, nfc=None, nfv=None):
        self.nbasis = bundle.nao
        self.nbsuse = bundle.nmo
        self.ne = bundle.nelec
        self.multip = bundle.multiplicity
        self.nfc = bundle.ncore if nfc is None else nfc
        self.nfv = (bundle.nmo - bundle.act_stop) if nfv is None else nfv
        self.atmchg = np.asarray(bundle.atom_charges)

        coefficients = bundle.mo_coeff
        # Stored row-wise and flattened, which is what the reader must detect
        # rather than assume. The bundle convention is column-wise.
        flat = (coefficients.T if transpose_coefficients else coefficients).ravel()

        self.matlist = {
            "OVERLAP": _Entry(_pack(bundle.overlap), bundle.nao),
            "CORE HAMILTONIAN ALPHA": _Entry(_pack(bundle.hcore_ao), bundle.nao),
            "ALPHA MO COEFFICIENTS": _Entry(flat),
            "ALPHA FOCK MATRIX": _Entry(_pack(bundle.fock_ao_alpha), bundle.nao),
            "ALPHA ORBITAL ENERGIES": _Entry(bundle.mo_energy_alpha),
            "AA MO 2E INTEGRALS": _Entry(bundle.eri_act.ravel()),
        }
        for label in drop:
            self.matlist.pop(label, None)
        if extra:
            self.matlist.update(extra)

        self._scalars = {"ENUCREP": bundle.e_nuc, "ESCF": bundle.e_scf}
        if scalars is not None:
            self._scalars = scalars

    def scalar(self, name):
        if name not in self._scalars:
            raise KeyError(f"no scalar named {name!r}")
        return self._scalars[name]


@pytest.fixture
def h2o(fixture_path):
    return load(fixture_path("h2o_rhf"))


@pytest.fixture
def fake(h2o, monkeypatch):
    """``fake(**options)`` installs a stand-in and returns the bundle it models."""

    def _install(**options):
        matel = FakeMatEl(h2o, **options)
        monkeypatch.setattr(M, "read_matel", lambda path: matel)
        return matel

    return _install


# ------------------------------------------------------------- the happy path

def test_extract_reproduces_the_job(fake, h2o):
    fake()
    bundle = M.extract("job.mat", reference="RHF", window=(2, 7), basis="sto-3g")

    assert bundle.reference == "RHF"
    assert (bundle.nao, bundle.nmo, bundle.nact) == (h2o.nao, h2o.nmo, h2o.nact)
    assert (bundle.nalpha, bundle.nbeta) == (h2o.nalpha, h2o.nbeta)
    assert bundle.charge == 0
    assert bundle.fock_source == "gaussian"
    np.testing.assert_allclose(bundle.mo_coeff, h2o.mo_coeff, atol=1e-12)
    np.testing.assert_allclose(bundle.overlap, h2o.overlap, atol=1e-12)
    np.testing.assert_allclose(bundle.hcore_ao, h2o.hcore_ao, atol=1e-12)
    np.testing.assert_allclose(bundle.eri_act, h2o.eri_act, atol=1e-12)
    assert bundle.e_nuc == pytest.approx(h2o.e_nuc)
    assert bundle.e_scf == pytest.approx(h2o.e_scf)


def test_extract_records_provenance(fake):
    fake()
    bundle = M.extract(
        "job.mat", reference="RHF", window=(2, 7), fch="job.fch",
        route="#P RHF/STO-3G Output=MatrixElement", basis="sto-3g", method="RHF",
    )
    provenance = bundle.provenance
    assert provenance["source_file"] == "job.mat"
    assert provenance["fch_file"] == "job.fch"
    assert provenance["window_1based"] == [2, 7]
    assert provenance["fock_source"] == "gaussian"
    assert provenance["basis"] == "sto-3g"
    assert "route" in provenance


def test_the_window_may_be_taken_from_the_file(fake, h2o):
    """nfc/nfv is the file's own account of the partition, and is usable."""
    fake()
    bundle = M.extract("job.mat", reference="RHF")
    assert (bundle.act_start, bundle.act_stop) == (h2o.act_start, h2o.act_stop)


def test_extracted_bundles_are_valid_and_usable(fake):
    """The reader's output must go straight into the Hamiltonian code."""
    from g16dump.hamiltonian import active_hamiltonian

    fake()
    bundle = M.extract("job.mat", reference="RHF", window=(2, 7))
    result = active_hamiltonian(bundle)
    assert result.e_ref == pytest.approx(bundle.e_scf, abs=1e-9)


# ------------------------------------------------ the coefficient convention

@pytest.mark.parametrize("transposed", [True, False])
def test_either_storage_orientation_is_read_correctly(fake, h2o, transposed):
    """Which one Gaussian used is decided by C.T S C = I, not by documentation."""
    fake(transpose_coefficients=transposed)
    bundle = M.extract("job.mat", reference="RHF", window=(2, 7))
    np.testing.assert_allclose(bundle.mo_coeff, h2o.mo_coeff, atol=1e-12)


def test_orient_mo_coeff_reports_both_scores_when_neither_works(h2o):
    scrambled = h2o.mo_coeff.copy()
    scrambled[:, 0] *= 3.0
    with pytest.raises(M.MatFileError) as excinfo:
        M.orient_mo_coeff(scrambled, h2o.overlap, h2o.nao, h2o.nmo)
    message = str(excinfo.value)
    assert "as stored" in message and "transposed" in message
    assert "NoBasisTransform" in message


def test_orient_mo_coeff_rejects_a_wrong_element_count(h2o):
    with pytest.raises(M.MatFileError, match="expected 49"):
        M.orient_mo_coeff(np.zeros(30), h2o.overlap, h2o.nao, h2o.nmo)


# ---------------------------------------------------------------- the window

def test_a_window_that_contradicts_the_integrals_stops_the_run(fake):
    """The one number that proves Window= did what the route meant."""
    fake()
    with pytest.raises(M.MatFileError) as excinfo:
        M.extract("job.mat", reference="RHF", window=(1, 7))
    message = str(excinfo.value)
    assert "7 active orbitals" in message
    assert "6^4 integrals" in message
    assert "fix the route" in message


def test_a_file_reported_partition_that_contradicts_the_integrals_also_stops(fake):
    fake(nfc=2)
    with pytest.raises(M.MatFileError, match="the file's nfc/nfv"):
        M.extract("job.mat", reference="RHF")


def test_a_two_electron_block_of_an_impossible_size_is_reported(fake):
    fake(extra={"AA MO 2E INTEGRALS": _Entry(np.zeros(1000))})
    with pytest.raises(M.MatFileError, match="not n\\^4"):
        M.extract("job.mat", reference="RHF", window=(2, 7))


# --------------------------------------------------------- missing material

def test_a_missing_block_lists_what_the_file_does_carry(fake):
    fake(drop=["ALPHA FOCK MATRIX", "AA MO 2E INTEGRALS"])
    with pytest.raises(M.MatFileError) as excinfo:
        M.extract("job.mat", reference="RHF", window=(2, 7))
    message = str(excinfo.value)
    assert "looked for" in message
    assert "OVERLAP" in message          # what it does have
    assert "transformation did not run" in message


def test_a_missing_fock_matrix_is_recorded_not_fatal(fake):
    """No Fock matrix means the rebuilt-Fock path, not a failed extraction."""
    fake(drop=["ALPHA FOCK MATRIX"])
    bundle = M.extract("job.mat", reference="RHF", window=(2, 7))
    assert bundle.fock_source == "none"
    assert bundle.fock_ao_alpha is None


def test_a_missing_nuclear_repulsion_names_the_probe(fake):
    fake(scalars={"ESCF": -74.9})
    with pytest.raises(M.MatFileError, match="inspect_mat.py"):
        M.extract("job.mat", reference="RHF", window=(2, 7))


def test_a_missing_scf_energy_is_simply_absent(fake):
    fake(scalars={"ENUCREP": 9.1671})
    bundle = M.extract("job.mat", reference="RHF", window=(2, 7))
    assert bundle.e_scf is None


# ------------------------------------------------------- reference handling

def test_the_reference_type_is_never_guessed(fake):
    fake()
    with pytest.raises(M.MatFileError, match="not inferred from the file"):
        M.extract("job.mat", reference="MP2", window=(2, 7))


def test_a_ks_job_never_carries_a_stored_fock_matrix(fake):
    """The KS matrix is present in the file and is deliberately not read."""
    fake()
    bundle = M.extract("job.mat", reference="RKS", window=(2, 7))
    assert bundle.fock_source == "none"
    assert bundle.fock_ao_alpha is None
    assert "rebuilt" in bundle.provenance["ks_note"]


def test_a_uhf_job_is_refused_with_the_workaround(fake, h2o):
    """Different beta orbitals cannot be represented by a restricted FCIDUMP."""
    # Swap two orbitals, so the beta set spans the same space but is not the
    # same set of orbitals -- which is exactly what a UHF job produces.
    rotation = np.eye(h2o.nmo)
    rotation[2, 2] = rotation[3, 3] = 0.0
    rotation[2, 3] = rotation[3, 2] = 1.0
    fake(extra={"BETA MO COEFFICIENTS": _Entry((h2o.mo_coeff @ rotation).T.ravel())})
    with pytest.raises(M.MatFileError, match="UHF reference"):
        M.extract("job.mat", reference="RHF", window=(2, 7))


def test_identical_beta_coefficients_are_not_a_uhf_job(fake, h2o):
    """A restricted job may write both blocks; only a difference is UHF."""
    fake(extra={"BETA MO COEFFICIENTS": _Entry(h2o.mo_coeff.T.ravel())})
    bundle = M.extract("job.mat", reference="RHF", window=(2, 7))
    np.testing.assert_allclose(bundle.mo_coeff, h2o.mo_coeff, atol=1e-12)


def test_an_impossible_multiplicity_in_the_header_is_caught(fake, h2o, monkeypatch):
    matel = FakeMatEl(h2o)
    matel.multip = 2  # 10 electrons cannot be a doublet
    monkeypatch.setattr(M, "read_matel", lambda path: matel)
    with pytest.raises(M.MatFileError, match="cannot carry multiplicity"):
        M.extract("job.mat", reference="RHF", window=(2, 7))


def test_a_truncated_header_is_caught_before_anything_else(fake, h2o, monkeypatch):
    matel = FakeMatEl(h2o)
    matel.nbasis = 0
    monkeypatch.setattr(M, "read_matel", lambda path: matel)
    with pytest.raises(M.MatFileError, match="truncated"):
        M.extract("job.mat", reference="RHF")


# ----------------------------------------------------------- label matching

def test_find_label_prefers_an_exact_match_over_a_substring():
    keys = ["ALPHA FOCK MATRIX (SAVED)", "ALPHA FOCK MATRIX"]
    assert M.find_label(keys, ["ALPHA FOCK MATRIX"]) == "ALPHA FOCK MATRIX"


def test_find_label_falls_back_to_a_substring():
    keys = ["SCF ALPHA FOCK MATRIX, FINAL"]
    assert M.find_label(keys, ["ALPHA FOCK MATRIX"]) == "SCF ALPHA FOCK MATRIX, FINAL"


def test_find_label_is_insensitive_to_case_and_spacing():
    assert M.find_label(["alpha   fock  matrix"], ["ALPHA FOCK MATRIX"])


def test_find_label_returns_none_rather_than_a_near_miss():
    assert M.find_label(["OVERLAP", "KINETIC ENERGY"], ["ALPHA FOCK MATRIX"]) is None


def test_candidate_order_is_preference_order():
    """The first spelling listed wins when a file carries several."""
    keys = ["CORE HAMILTONIAN", "CORE HAMILTONIAN ALPHA"]
    assert M.find_label(keys, M.LABELS["hcore"]) == "CORE HAMILTONIAN ALPHA"


def test_gauopen_absence_is_a_sentence_not_a_traceback(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_qcmatel(name, *args, **kwargs):
        if name == "QCMatEl":
            raise ImportError("No module named 'QCMatEl'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_qcmatel)
    with pytest.raises(M.MatFileError, match="PYTHONPATH"):
        M.read_matel("job.mat")
