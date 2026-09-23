"""``matfile.extract`` end to end, with MOKIT and a synthetic ``.mat``.

Until this file existed, ``extract`` had never read anything: gauopen is not
available outside the cluster. Here a stand-in for ``QCMatEl`` serves a
matrix-element file built from a PySCF calculation, with every AO-basis
quantity in **Gaussian's** AO order, alongside a real ``.fch`` written by MOKIT.
That is the situation the reader faces on real data, including the part that
broke: for any basis with d functions the ``.mat`` and the molecule rebuilt from
the ``.fch`` disagree about AO order.

The Gaussian order used to build the fake ``.mat`` is *not* taken from
``g16dump.aoorder``. It is recovered empirically by matching the orbital
coefficients MOKIT writes into the ``.fch`` (Gaussian order, by construction)
against PySCF's, column by column. So the test checks the permutation instead
of agreeing with itself.

Skipped unless both PySCF and MOKIT are installed.
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("mokit.lib.gaussian")

from pyscf import ao2mo  # noqa: E402

from g16dump import matfile  # noqa: E402
from g16dump.aoorder import (  # noqa: E402
    check_same_basis,
    gaussian_to_pyscf_permutation,
)
from g16dump.bundle import validate  # noqa: E402
from g16dump.errors import ValidationError  # noqa: E402
from g16dump.hamiltonian import active_space_hamiltonian, load_mol_from_fch  # noqa: E402
from pyscf_systems import build_molecule, casci_reference, run_scf  # noqa: E402


# ----------------------------------------------------------- the fake .mat


class _Entry:
    """Just enough of a gauopen matrix object: ``.array`` and ``.expand()``."""

    def __init__(self, array, expanded=None):
        self.array = np.asarray(array)
        self._expanded = self.array if expanded is None else np.asarray(expanded)

    def expand(self):
        return self._expanded


def _lower_triangle(m):
    rows, cols = np.tril_indices(m.shape[0])
    return m[rows, cols]


def _raw_fch_array(path, label):
    """A real-valued .fch section, verbatim -- Gaussian AO order."""
    lines = open(path).read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith(label):
            count = int(line.split()[-1])
            values, j = [], i + 1
            while len(values) < count:
                values += [float(x) for x in lines[j].split()]
                j += 1
            return np.array(values)
    raise KeyError(label)


def _empirical_gaussian_order(fch, mo_pyscf):
    """``g2p[g]`` = PySCF AO index of Gaussian AO ``g``, recovered from data.

    Each AO column of the .fch coefficients (Gaussian order, 9 significant
    digits) is matched to the PySCF column it equals. Independent of
    g16dump.aoorder by construction.
    """
    nbasis, nmo = mo_pyscf.shape
    c_fch = _raw_fch_array(fch, "Alpha MO coefficients").reshape(nmo, nbasis)
    c_pyscf = mo_pyscf.T
    g2p = np.empty(nbasis, dtype=int)
    for g in range(nbasis):
        residual = np.max(np.abs(c_pyscf - c_fch[:, [g]]), axis=0)
        g2p[g] = int(np.argmin(residual))
        assert residual[g2p[g]] < 1e-6, "could not match an AO column"
    assert sorted(g2p.tolist()) == list(range(nbasis)), "not a permutation"
    return g2p


def _fake_mat(mf, ncore, nact, g2p, *, with_overlap=True, escf=True):
    """A MatEl-like object with everything in Gaussian AO order."""
    mol = mf.mol
    nbasis, nmo = mf.mo_coeff.shape
    c_gauss = mf.mo_coeff.T[:, g2p]  # MO-major, Gaussian AO order
    h_gauss = mf.get_hcore()[np.ix_(g2p, g2p)]
    s_gauss = mf.get_ovlp()[np.ix_(g2p, g2p)]

    mo_act = mf.mo_coeff[:, ncore : ncore + nact]
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo_act), nact)

    matlist = {
        "CORE HAMILTONIAN ALPHA": _Entry(_lower_triangle(h_gauss)),
        "ALPHA MO COEFFICIENTS": _Entry(c_gauss.ravel()),
        "ALPHA ORBITAL ENERGIES": _Entry(np.asarray(mf.mo_energy)),
        "AA MO 2E INTEGRALS": _Entry(eri.ravel()),
        "BA MO 2E INTEGRALS": _Entry(eri.ravel()),
        "BB MO 2E INTEGRALS": _Entry(eri.ravel()),
    }
    if with_overlap:
        matlist["OVERLAP"] = _Entry(_lower_triangle(s_gauss))

    scalars = {"ENUCREP": float(mol.energy_nuc())}
    if escf:
        scalars["ESCF"] = float(mf.e_tot)

    def scalar(name):
        if name not in scalars:
            raise KeyError(name)
        return scalars[name]

    return types.SimpleNamespace(
        nbasis=nbasis,
        nbsuse=nmo,
        nfc=ncore,
        nfv=nmo - ncore - nact,
        ne=mol.nelectron,
        multip=mol.spin + 1,
        matlist=matlist,
        scalar=scalar,
    )


@pytest.fixture
def fake_gauopen(monkeypatch):
    """Route matfile's gauopen import to whatever fake .mat the test installs."""
    holder = {}
    module = types.SimpleNamespace(MatEl=lambda file: holder["mat"])
    monkeypatch.setattr(matfile, "_import_qcmatel", lambda: module)
    return holder


# --------------------------------------------------------------- systems

# label, molecule, basis (all with d functions), spin, method, ref_type, ncore, nact
SYSTEMS = [
    ("h2o_rhf_631gs", "h2o", "6-31g*", 0, "RHF", "RHF", 1, 10),
    ("ch2_rohf_def2svp", "ch2", "def2-svp", 2, "ROHF", "ROHF", 1, 10),
    ("o2_rohf_ccpvdz", "o2", "cc-pvdz", 2, "ROHF", "ROHF", 2, 10),
    ("h2o_rks_def2svp", "h2o", "def2-svp", 0, "RKS", "RKS", 1, 10),
]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Converge each system once and write its .fch with MOKIT."""
    from mokit.lib.py2fch_direct import fchk

    out = {}
    root = tmp_path_factory.mktemp("fch")
    for label, name, basis, spin, method, ref_type, ncore, nact in SYSTEMS:
        mol = build_molecule(name, basis, spin=spin)
        mf = run_scf(mol, method)
        fch = str(root / f"{label}.fch")
        fchk(mf, fch)
        g2p = _empirical_gaussian_order(fch, mf.mo_coeff)
        out[label] = (mf, fch, g2p, ref_type, ncore, nact)
    return out


@pytest.fixture(params=[s[0] for s in SYSTEMS])
def case(request, prepared, tmp_path, monkeypatch):
    # MOKIT's load_mol_from_fch writes scratch files into the cwd.
    monkeypatch.chdir(tmp_path)
    return prepared[request.param]


# ----------------------------------------------------------------- tests


def test_the_basis_really_has_non_trivial_ao_order(case):
    """Guard: if Gaussian and PySCF order coincided, this file would test nothing."""
    _, _, g2p, _, _, _ = case
    assert not np.array_equal(g2p, np.arange(len(g2p)))


def test_permutation_matches_the_empirical_gaussian_order(case):
    """aoorder's analytic permutation == the one recovered from MOKIT's .fch."""
    _, fch, g2p, _, _, _ = case
    assert np.array_equal(gaussian_to_pyscf_permutation(load_mol_from_fch(fch)), g2p)


@pytest.mark.parametrize("with_overlap", [True, False], ids=["overlap_in_mat", "overlap_from_fch"])
def test_extract_with_rebuilt_fock_matches_the_oracle(case, fake_gauopen, with_overlap):
    """Full extract -> bundle -> h', E_core, against PySCF's CASCI reduction."""
    mf, fch, g2p, ref_type, ncore, nact = case
    is_ks = ref_type in ("RKS", "ROKS")
    fake_gauopen["mat"] = _fake_mat(mf, ncore, nact, g2p, with_overlap=with_overlap)

    bundle = matfile.extract("job.mat", ref_type=ref_type, fch=fch, rebuild_fock=True)
    validate(bundle)
    assert bundle.fock_source == "rebuilt-pyscf"

    # The Fock matrices were rebuilt from the .fch, whose 9-digit basis data
    # limits J and K to ~1e-9 Ha for these molecules (~1e-7 for Ni complexes).
    # The default E_ref gate knows this from provenance; the comparisons below
    # use 1e-7, still five orders of magnitude below what the AO-ordering bug
    # produced.
    assert bundle.provenance["fock_rebuilt_from_fch"] is True
    ash = active_space_hamiltonian(bundle, tol_scf=None if is_ks else "auto")
    if not is_ks:
        assert ash.diagnostics["scf_tolerance"] == 1e-5
        assert abs(ash.e_ref - mf.e_tot) < 1e-7
    else:
        assert bundle.e_scf is None  # a KS total energy is not a determinant energy

    h1, e_core, eri = casci_reference(
        mf, ncore, nact, (bundle.nocc_a_act, bundle.nocc_b_act)
    )
    assert ash.diagnostics["spin_error"] < 1e-9
    assert np.max(np.abs(ash.h_eff - h1)) < 1e-7
    assert abs(ash.e_core - e_core) < 1e-7
    assert np.max(np.abs(ash.eri_act - eri)) < 1e-12


def test_the_old_behaviour_is_caught_by_the_basis_check(case):
    """Treating Gaussian order as PySCF order (the bug) fails loudly now."""
    mf, fch, g2p, _, _, _ = case
    mol = load_mol_from_fch(fch)
    h_gauss = mf.get_hcore()[np.ix_(g2p, g2p)]
    identity = np.arange(mol.nao)
    with pytest.raises(ValidationError, match="disagree"):
        check_same_basis("core Hamiltonian", h_gauss, mf.get_hcore(), identity)
    # ...while the right permutation passes with room to spare.
    perm = gaussian_to_pyscf_permutation(mol)
    assert check_same_basis("core Hamiltonian", h_gauss, mf.get_hcore(), perm) < 1e-9


def test_a_mismatched_fch_is_refused(prepared, fake_gauopen, tmp_path, monkeypatch):
    """A .fch from a different geometry must not be used to rebuild anything."""
    from mokit.lib.py2fch_direct import fchk

    monkeypatch.chdir(tmp_path)
    mf, _, g2p, ref_type, ncore, nact = prepared["h2o_rhf_631gs"]
    fake_gauopen["mat"] = _fake_mat(mf, ncore, nact, g2p)

    stretched = mf.mol.copy()
    stretched.atom = [[a, [x * 1.05 for x in c]] for a, c in mf.mol._atom]
    stretched.unit = "Bohr"
    stretched.build()
    other = run_scf(stretched, "RHF")
    wrong = str(tmp_path / "wrong.fch")
    fchk(other, wrong)

    with pytest.raises(ValidationError, match="not from the\\s+same Gaussian job|disagree"):
        matfile.extract("job.mat", ref_type=ref_type, fch=wrong, rebuild_fock=True)


def test_extract_without_fch_and_without_fock_is_refused(prepared, fake_gauopen):
    mf, _, g2p, ref_type, ncore, nact = prepared["h2o_rhf_631gs"]
    fake_gauopen["mat"] = _fake_mat(mf, ncore, nact, g2p)
    with pytest.raises(ValidationError, match="no usable Fock matrix"):
        matfile.extract("job.mat", ref_type=ref_type)
