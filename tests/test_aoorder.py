"""The Gaussian -> PySCF AO permutation, without PySCF.

The convention is pinned here with a stand-in molecule object, so it is checked
in the NumPy-only CI. ``test_extract.py`` separately checks the same
permutation against MOKIT's own reordering on real basis sets.
"""

from __future__ import annotations

import numpy as np
import pytest

from g16dump.aoorder import (
    check_same_basis,
    coefficients_to_pyscf,
    gaussian_pure_order,
    gaussian_to_pyscf_permutation,
    matrix_to_gaussian,
    matrix_to_pyscf,
)
from g16dump.errors import ValidationError


class FakeMol:
    """Only the attributes the permutation reads."""

    def __init__(self, shells, cart=False):
        self._shells = shells  # list of (l, nctr)
        self.cart = cart
        self.nbas = len(shells)
        self.nao = sum((2 * l + 1) * n for l, n in shells)

    def bas_angular(self, i):
        return self._shells[i][0]

    def bas_nctr(self, i):
        return self._shells[i][1]


def test_gaussian_pure_d_and_f_order():
    assert gaussian_pure_order(2) == [0, 1, -1, 2, -2]
    assert gaussian_pure_order(3) == [0, 1, -1, 2, -2, 3, -3]


def test_s_and_p_are_left_alone():
    perm = gaussian_to_pyscf_permutation(FakeMol([(0, 1), (1, 1), (0, 1)]))
    assert perm.tolist() == [0, 1, 2, 3, 4]


def test_d_shell_permutation():
    # s, then one d shell at offset 1. PySCF d is m=-2..2, i.e. index m+2.
    perm = gaussian_to_pyscf_permutation(FakeMol([(0, 1), (2, 1)]))
    assert perm.tolist() == [0, 1 + 2, 1 + 3, 1 + 1, 1 + 4, 1 + 0]


def test_general_contraction_repeats_the_pattern():
    perm = gaussian_to_pyscf_permutation(FakeMol([(2, 2)]))
    assert perm.tolist() == [2, 3, 1, 4, 0, 7, 8, 6, 9, 5]


def test_cartesian_basis_is_refused():
    with pytest.raises(ValidationError, match="5D 7F"):
        gaussian_to_pyscf_permutation(FakeMol([(2, 1)], cart=True))


def test_matrix_and_coefficient_conversions_are_consistent():
    """C S C.T must be invariant under the AO reordering."""
    rng = np.random.default_rng(0)
    mol = FakeMol([(0, 1), (1, 1), (2, 1), (3, 1)])
    perm = gaussian_to_pyscf_permutation(mol)
    n = mol.nao

    s_pyscf = rng.normal(size=(n, n))
    s_pyscf = s_pyscf @ s_pyscf.T + n * np.eye(n)
    c_gauss = rng.normal(size=(n, n))

    s_gauss = matrix_to_gaussian(s_pyscf, perm)
    c_pyscf = coefficients_to_pyscf(c_gauss, perm)

    assert np.allclose(c_gauss @ s_gauss @ c_gauss.T, c_pyscf @ s_pyscf @ c_pyscf.T)
    assert np.array_equal(matrix_to_pyscf(s_gauss, perm), s_pyscf)


def test_basis_check_is_relative():
    """Tight metal functions make h_ao large; the tolerance must scale with it."""
    mol = FakeMol([(0, 1), (2, 1)])
    perm = gaussian_to_pyscf_permutation(mol)
    h_pyscf = np.diag([400.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    h_gauss = matrix_to_gaussian(h_pyscf, perm)

    # 6e-8 absolute on a 400 Ha matrix: the .fch precision floor, accepted.
    noisy = h_pyscf.copy()
    noisy[0, 0] += 6e-8
    assert check_same_basis("h", h_gauss, noisy, perm) < 1e-9

    # Wrong order: refused.
    with pytest.raises(ValidationError, match="disagree"):
        check_same_basis("h", h_gauss, h_pyscf, np.arange(mol.nao))
