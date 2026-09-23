"""The parts of the MOKIT bridge that need neither MOKIT nor PySCF.

These run in the numpy-only CI job: reading a ``.fch`` section, the unit-vector
rewrite used to read MOKIT's transformation out of ``fch2py``, the naming of
that transformation in provenance, the refusal to rebuild a Fock matrix in a
``Mole`` whose AO basis is not the bundle's, the error a user without MOKIT
sees, and the ``rebuild-fock`` command's refusal to overwrite anything it
should not. What MOKIT itself computes is tested in ``test_mokit_bridge.py``.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from g16dump import fch as F
from g16dump import hamiltonian as H
from g16dump.bundle import load, save
from g16dump.cli import main

FCH = """\
tiny
SP        RHF                                                         STO-3G
Number of basis functions                  I                3
Number of independent functions            I                2
Alpha MO coefficients                      R   N=           6
  1.00000000E+00  2.00000000E+00  3.00000000E+00  4.00000000E+00  5.00000000E+00
  6.00000000E+00
Total Energy                               R     -1.234567890123456E+00
"""


@pytest.fixture
def tiny_fch(tmp_path):
    path = tmp_path / "tiny.fch"
    path.write_text(FCH)
    return path


def test_fch_sections_are_read_as_printed(tiny_fch):
    assert F.read_fch_scalar(tiny_fch, "Number of basis functions") == 3
    assert F.read_fch_scalar(tiny_fch, "Total Energy") == pytest.approx(-1.234567890123456)
    values = F.read_fch_array(tiny_fch, "Alpha MO coefficients")
    assert values.tolist() == [1, 2, 3, 4, 5, 6]
    # Column-wise, one orbital per column: the file lists orbital by orbital.
    assert F.fch_mo_coeff(tiny_fch).tolist() == [[1, 4], [2, 5], [3, 6]]


def test_missing_fch_sections_are_named(tiny_fch):
    with pytest.raises(F.FchError, match="Beta MO coefficients"):
        F.read_fch_array(tiny_fch, "Beta MO coefficients")
    with pytest.raises(F.FchError, match="Charge"):
        F.read_fch_scalar(tiny_fch, "Charge")


def test_unit_vectors_survive_the_fch_format(tiny_fch):
    """The rewrite that feeds fch2py its probes must be exact."""
    lines = tiny_fch.read_text().splitlines()
    probe = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    rewritten = F._replace_real_array(lines, "Alpha MO coefficients", probe)
    out = tiny_fch.with_name("probe.fch")
    out.write_text("\n".join(rewritten) + "\n")
    assert np.array_equal(F.read_fch_array(out, "Alpha MO coefficients"), probe)
    assert F.read_fch_scalar(out, "Total Energy") == pytest.approx(-1.234567890123456)


def test_transform_kinds_are_named_for_provenance():
    assert F.describe_transform(np.eye(3)) == "identity"
    perm = np.eye(3)[[2, 0, 1]]
    assert F.describe_transform(perm) == "permutation"
    signed = perm.copy()
    signed[0, 2] = -1.0
    assert F.describe_transform(signed) == "signed permutation"
    assert "rescaling" in F.describe_transform(perm @ np.diag([1.0, 0.5, 2.0]))
    assert F.describe_transform(np.ones((2, 2))) == "general linear map"


def test_missing_mokit_is_reported_with_the_way_to_install_it(monkeypatch, tiny_fch):
    monkeypatch.setitem(sys.modules, "mokit", None)
    monkeypatch.setitem(sys.modules, "mokit.lib", None)
    with pytest.raises(F.FchError) as excinfo:
        F.mol_from_fch(tiny_fch)
    message = str(excinfo.value)
    assert "MOKIT" in message and "not on PyPI" in message
    assert "conda install mokit" in message
    # The old entry point says the same thing.
    with pytest.raises(H.HamiltonianError, match="conda install mokit"):
        H.mol_from_fch(tiny_fch)


# ------------------------------------------ the AO-basis guard, without PySCF


class _Mol:
    """Only what the guard reads: the AO count and the overlap."""

    def __init__(self, overlap):
        self._overlap = overlap
        self.nao = overlap.shape[0]

    def intor(self, name):
        assert name == "int1e_ovlp"
        return self._overlap


def test_a_mole_in_another_ao_order_is_refused(synthetic):
    order = np.roll(np.arange(synthetic.nao), 1)
    mol = _Mol(synthetic.S[np.ix_(order, order)])
    with pytest.raises(H.AOBasisMismatchError, match="rebuild-fock"):
        H.with_rebuilt_fock(synthetic, mol)


def test_the_guard_accepts_the_right_transformation(synthetic):
    order = np.roll(np.arange(synthetic.nao), 1)
    mol = _Mol(synthetic.S[np.ix_(order, order)])
    transform = np.eye(synthetic.nao)[:, order].T  # C_mol = X C
    assert H.overlap_mismatch(synthetic, mol, transform) < 1e-14
    assert H.overlap_mismatch(synthetic, mol) > 1e-3


def test_a_mole_of_another_size_is_refused(synthetic):
    mol = _Mol(np.eye(synthetic.nao + 1))
    with pytest.raises(H.AOBasisMismatchError, match="not the same AO basis"):
        H.with_rebuilt_fock(synthetic, mol)


# ------------------------------------------------------- the command line


@pytest.fixture
def bundle_file(tmp_path, fixture_path):
    path = tmp_path / "JOB.npz"
    save(load(fixture_path("ch2_rohf_roothaan")), path)
    return path


def test_rebuild_fock_never_replaces_its_input(bundle_file, tiny_fch, capsys):
    before = bundle_file.read_bytes()
    for extra in ([], ["--force"]):
        code = main(["rebuild-fock", str(bundle_file), "--fch", str(tiny_fch),
                     "--out", str(bundle_file), *extra])
        assert code == 1
        assert "never replaces its input" in capsys.readouterr().err
    assert bundle_file.read_bytes() == before


def test_rebuild_fock_refuses_an_existing_output(bundle_file, tiny_fch, tmp_path, capsys):
    target = tmp_path / "other.npz"
    target.write_bytes(b"keep me")
    assert main(["rebuild-fock", str(bundle_file), "--fch", str(tiny_fch),
                 "--out", str(target)]) == 1
    assert "--force" in capsys.readouterr().err
    assert target.read_bytes() == b"keep me"


def test_rebuild_fock_without_mokit_says_how_to_get_it(
    bundle_file, tiny_fch, tmp_path, monkeypatch, capsys
):
    monkeypatch.setitem(sys.modules, "mokit", None)
    monkeypatch.setitem(sys.modules, "mokit.lib", None)
    out = tmp_path / "rebuilt.npz"
    assert main(["rebuild-fock", str(bundle_file), "--fch", str(tiny_fch),
                 "--out", str(out)]) == 1
    assert "conda install mokit" in capsys.readouterr().err
    assert not out.exists()


def test_extract_help_no_longer_implies_a_rebuild(capsys):
    with pytest.raises(SystemExit):
        main(["extract", "--help"])
    assert "nothing is rebuilt here" in " ".join(capsys.readouterr().out.split())
