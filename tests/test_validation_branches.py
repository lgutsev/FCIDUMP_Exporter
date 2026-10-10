"""Refusal branches of the writer and the small matfile helpers, pinned with synthetic inputs.

Everything here runs on numpy alone (no pyscf, gauopen, Gaussian or mokit): it uses the committed
``h2o_rhf`` fixture and hand-made stand-ins for gauopen's matrix-element objects.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from g16dump import write
from g16dump.bundle import load
from g16dump.hamiltonian import active_hamiltonian
from g16dump.matfile import (
    MatFileError,
    _attr,
    _expand,
    _scalar,
    _square,
    find_label,
    orient_mo_coeff,
)
from g16dump.write import WriteError, write_fcidump


@pytest.fixture
def ham(fixture_path):
    return active_hamiltonian(load(fixture_path("h2o_rhf")))


def refused(ham, tmp_path, match, **kwargs):
    target = tmp_path / "FCIDUMP"
    with pytest.raises(WriteError, match=match):
        write_fcidump(ham, target, **kwargs)
    assert not target.exists()
    assert not list(tmp_path.glob("*.partial"))


# ------------------------------------------------------------------ writer refusals


def test_no_active_orbitals(ham, tmp_path):
    refused(replace(ham, nact=0), tmp_path, "nothing to write")


def test_wrong_array_shapes_are_all_reported(ham, tmp_path):
    n = ham.nact
    bad = replace(ham, h_eff=np.zeros((n, n + 1)), eri_active=np.zeros((n, n, n)))
    refused(bad, tmp_path, r"(?s)h_eff has shape.*eri_act has shape")


def test_non_finite_values(ham, tmp_path):
    h = ham.h_eff.copy()
    h[0, 0] = np.nan
    eri = ham.eri_active.copy()
    eri[0, 0, 0, 0] = np.inf
    refused(replace(ham, h_eff=h, eri_active=eri, e_core=float("nan")), tmp_path,
            r"(?s)h_eff holds non-finite.*eri_act holds non-finite.*e_core is nan")


def test_asymmetric_one_electron_matrix(ham, tmp_path):
    h = ham.h_eff.copy()
    h[0, 1] += 1e-3
    refused(replace(ham, h_eff=h), tmp_path, "h_eff is not symmetric")


@pytest.mark.parametrize(
    ("nelec", "ms2", "message"),
    [
        (-2, 0, "nelec_act is -2"),
        (4, -2, "high-spin convention"),
        (2, 4, "needs more unpaired electrons"),
        (4, 1, "parity is wrong"),
        (100, 0, "do not fit in"),
        (10, 8, "does not fit in"),  # 9 alpha + 1 beta electrons in too few orbitals
    ],
)
def test_electron_count_and_spin_refusals(ham, tmp_path, nelec, ms2, message):
    refused(replace(ham, nelec_active=nelec, ms2=ms2), tmp_path, message)


def test_orbsym_length_and_range(ham, tmp_path):
    refused(ham, tmp_path, "orbsym has 1 entries", orbsym=[1])
    refused(ham, tmp_path, r"outside 1-8", orbsym=[0] + [1] * (ham.nact - 1))
    refused(ham, tmp_path, r"outside 1-8", orbsym=[9] * ham.nact)


@pytest.mark.parametrize("isym", [0, 9])
def test_isym_range(ham, tmp_path, isym):
    refused(ham, tmp_path, "isym is", isym=isym)


def test_failed_write_leaves_neither_target_nor_partial_file(ham, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(write, "_write_records", boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        write_fcidump(ham, tmp_path / "FCIDUMP")
    assert list(tmp_path.iterdir()) == []


def test_keyboard_interrupt_also_cleans_up(ham, tmp_path, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(write, "_write_records", interrupted)
    with pytest.raises(KeyboardInterrupt):
        write_fcidump(ham, tmp_path / "FCIDUMP")
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ matfile helpers


class _Entry:
    def __init__(self, array=None, expanded=None, fail=False):
        self.array = array
        self._expanded = expanded
        self._fail = fail
        if expanded is not None or fail:
            self.expand = self._expand

    def _expand(self):
        if self._fail:
            raise RuntimeError("bad packing")
        return self._expanded


def test_expand_prefers_expand_then_array():
    assert _expand(_Entry(expanded=[[1, 2], [3, 4]]), "x", "f.mat").dtype == np.float64
    assert _expand(_Entry(array=[1, 2, 3]), "x", "f.mat").tolist() == [1.0, 2.0, 3.0]


def test_expand_failures_become_matfile_errors():
    with pytest.raises(MatFileError, match=r"x\.expand\(\) failed: RuntimeError: bad packing"):
        _expand(_Entry(fail=True), "x", "f.mat")
    with pytest.raises(MatFileError, match="neither expand"):
        _expand(object(), "x", "f.mat")


def test_square_accepts_square_flat_and_packed_triangle():
    full = np.array([[1.0, 2.0], [2.0, 3.0]])
    assert np.array_equal(_square(full, 2, "m", "f"), full)
    assert np.array_equal(_square(full.ravel(), 2, "m", "f"), full)
    packed = np.array([1.0, 2.0, 3.0])  # lower triangle, row-major: (0,0), (1,0), (1,1)
    assert np.array_equal(_square(packed, 2, "m", "f"), full)
    with pytest.raises(MatFileError, match="neither 2x2 nor a 2-dimensional packed triangle"):
        _square(np.arange(5.0), 2, "m", "f")


def test_find_label_exact_beats_substring_and_ignores_case_and_spacing():
    keys = ["ALPHA  FOCK MATRIX (SAVED)", "alpha fock matrix"]
    assert find_label(keys, ["ALPHA FOCK MATRIX"]) == "alpha fock matrix"
    only_saved = ["ALPHA FOCK MATRIX (SAVED)"]
    assert find_label(only_saved, ["ALPHA FOCK MATRIX"]) == "ALPHA FOCK MATRIX (SAVED)"
    assert find_label(keys, ["BETA FOCK MATRIX"]) is None


def test_orient_mo_coeff_errors_and_orientation():
    identity = np.eye(2)
    with pytest.raises(MatFileError, match="expected 6"):
        orient_mo_coeff(np.zeros(4), np.eye(3), 3, 2)
    rotation = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    assert orient_mo_coeff(rotation, np.eye(3), 3, 2).shape == (3, 2)
    assert orient_mo_coeff(rotation.T, np.eye(3), 3, 2).shape == (3, 2)  # stored transposed
    with pytest.raises(MatFileError, match="not orthonormal"):
        orient_mo_coeff(np.ones((3, 2)), np.eye(3), 3, 2)
    assert identity.shape == (2, 2)


def test_scalar_and_attr_lookups_skip_failures():
    class Me:
        def scalar(self, name):
            if name == "boom":
                raise KeyError(name)
            return {"enuc": 9.5, "none": None}.get(name)

        nbasis = 7

        @property
        def broken(self):
            raise RuntimeError("no")

    assert _scalar(Me(), ["boom", "none", "enuc"]) == 9.5
    assert _scalar(Me(), ["boom", "none"]) is None
    assert _scalar(object(), ["enuc"]) is None  # no .scalar at all
    assert _attr(Me(), ["broken", "nbasis"]) == 7
    assert _attr(Me(), ["missing"], default="d") == "d"
