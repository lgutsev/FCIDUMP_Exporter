"""Shared fixtures.

Two kinds. A synthetic bundle built from random matrices, which needs nothing
but numpy and is what the schema and validation tests use -- they care about
structure, not chemistry. And the committed ``.npz`` fixtures under
``tests/data/``, which carry real SCF quantities and are what the frozen-core
algebra is checked against. Neither needs PySCF, Gaussian or gauopen; see
``tests/make_fixtures.py`` for how the committed ones were produced.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from g16dump.bundle import Bundle, make_provenance

DATA = Path(__file__).resolve().parent / "data"

#: Markers for the things CI cannot install. The core CI job deselects these and
#: a separate job installs pyscf and runs the ones marked ``pyscf``. Registered
#: here rather than in pyproject.toml so that packaging and tests stay separable.
MARKERS = {
    "pyscf": "needs pyscf",
    "gaussian": "needs a Gaussian 16 installation",
    "gauopen": "needs gauopen (QCMatEl) on PYTHONPATH",
    "mokit": "needs mokit, which is not installable from PyPI",
}


def pytest_configure(config):
    for name, description in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")

FIXTURES = (
    "h2o_rhf",
    "ch2_rohf",
    "ch2_rohf_roothaan",
    "ch2_rohf_rotated",
    "nh_rohf",
    "h2o_rks",
    "h2o_frozen_virtual",
)

#: The fixtures whose Fock matrices really are F^alpha/F^beta, so the algebra
#: is expected to succeed on them. ch2_rohf_roothaan is excluded on purpose.
SOUND_FIXTURES = ("h2o_rhf", "ch2_rohf", "ch2_rohf_rotated", "nh_rohf")


def _positive_definite_overlap(nao, rng):
    a = rng.normal(size=(nao, nao))
    return a @ a.T + nao * np.eye(nao)


def _orthonormal_against(overlap, nao, nmo, rng):
    """Coefficients satisfying ``C.T S C = I``, via ``S^-1/2``."""
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    s_inv_sqrt = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
    q, _ = np.linalg.qr(rng.normal(size=(nao, nao)))
    return s_inv_sqrt @ q[:, :nmo]


def _eightfold_symmetric_eri(nact, rng):
    """A random ERI tensor with the full permutational symmetry of real orbitals."""
    naux = nact + 2
    factors = rng.normal(size=(nact, nact, naux))
    factors = 0.5 * (factors + factors.transpose(1, 0, 2))
    return np.einsum("tux,vwx->tuvw", factors, factors)


def _symmetric(n, rng):
    a = rng.normal(size=(n, n))
    return 0.5 * (a + a.T)


@pytest.fixture
def synthetic() -> Bundle:
    """A structurally valid bundle whose numbers mean nothing.

    Validation must pass it, so that a test asserting a specific rejection is
    testing that rejection and not some unrelated defect in the fixture.
    """
    rng = np.random.default_rng(1234)
    nao = nmo = 6
    nact = 4
    overlap = _positive_definite_overlap(nao, rng)
    mo_coeff = _orthonormal_against(overlap, nao, nmo, rng)
    fock = _symmetric(nao, rng)
    return Bundle(
        reference_type="RHF",
        charge=0,
        multiplicity=1,
        nelec=6,
        nalpha=3,
        nbeta=3,
        nao=nao,
        nmo=nmo,
        ncore=1,
        nact=nact,
        active_first=2,   # 1-based and inclusive: MOs 2-5 of 6
        active_last=5,
        enuc=9.5,
        C=mo_coeff,
        S=overlap,
        Hcore_ao=_symmetric(nao, rng),
        eri_active=_eightfold_symmetric_eri(nact, rng),
        source_program="synthetic",
        source_file="tests/conftest.py",
        fock_source="gaussian",
        F_alpha_ao=fock,
        F_beta_ao=fock.copy(),
        orbital_energies=rng.normal(size=nmo),
        atom_charges=np.array([4.0, 1.0, 1.0]),
        escf=-42.0,
        provenance=make_provenance(fock_source="gaussian"),
    )


@pytest.fixture
def broken(synthetic):
    """``broken(**changes)`` -> the synthetic bundle with those fields replaced."""

    def _broken(**changes) -> Bundle:
        return replace(synthetic, **changes)

    return _broken


@pytest.fixture(scope="session")
def fixture_path():
    """``fixture_path(name)`` -> the committed ``.npz`` of that name."""

    def _path(name: str) -> Path:
        path = DATA / f"{name}.npz"
        if not path.exists():
            pytest.skip(
                f"{path} is missing; regenerate it with "
                f"python3 tests/make_fixtures.py (needs pyscf)"
            )
        return path

    return _path
