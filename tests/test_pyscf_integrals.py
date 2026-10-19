"""The ROHF route: active integrals built by PySCF over the ``.mat`` orbitals.

Gaussian transforms ROHF integrals over per-spin semicanonical orbitals, and no
overlay-8 switch changes that in G16 C.01 (gaussian/README.md, "The ROHF
route"). ``extract(..., integrals="pyscf")`` builds the active ``(tu|vw)`` and
the Fock matrices with PySCF instead, over the ROHF orbitals the ``.mat``
stores.

These tests run that on the real G16 C.01 probe files committed in
``tests/data/g16_c01_probes``. In production the Gaussian -> PySCF AO map comes
from the ``.fch`` through MOKIT, which CI cannot install, so here the map is
built from the ``.mat`` header instead: for s and p shells it is a pure
permutation (PySCF groups each atom's shells by angular momentum, Gaussian keeps
shell order), and :func:`g16dump.fch.gaussian_to_pyscf` is replaced by a
function returning it. Everything else is the production path.

Two checks give the route its meaning:

* on RHF files, where Gaussian's own integrals are over the stored orbitals,
  PySCF must reproduce them element by element;
* on ROHF files, which Gaussian's integrals cannot serve, the alpha/beta gate
  must close, ``E_ref`` must equal Gaussian's E(ROHF), and ``E_core`` must match
  the independent PySCF fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.pyscf
pytest.importorskip("pyscf")

from g16dump import fch as F  # noqa: E402
from g16dump import hamiltonian as H  # noqa: E402
from g16dump.matfile import MatFileError, extract, read_mat  # noqa: E402

PROBES = Path(__file__).resolve().parent / "data" / "g16_c01_probes"

#: Probe -> (PySCF basis name, reference, E_core from the PySCF fixtures or None).
#: Only RHF has a fixture E_core to compare: canonical RHF orbitals are unique,
#: but ROHF canonicalization is not, and Gaussian's differs from PySCF's by a
#: rotation that mixes the frozen core with the active doubly occupied orbitals.
#: That leaves E_ref alone and moves E_core (by 8.5e-5 Ha on CH2, measured in
#: CI), so ROHF is checked against a full transform in its own orbitals instead.
CASES = {
    "probe_h2o_rhf": ("sto-3g", "RHF", -51.4711696318),
    "probe_h2o_frozen_virtual": ("6-31g", "RHF", -52.1215325375),
    "probe_ch2_rohf": ("6-31g", "ROHF", None),
    "probe_nh_rohf": ("6-31g", "ROHF", None),
}

#: The whole Hamiltonian is PySCF's, over Gaussian's converged orbitals, so it
#: differs from Gaussian's own energy only through the basis-set digits the two
#: programs carry: Gaussian's E(SCF) and the PySCF fixtures' differ by 7e-9 to
#: 5e-8 Ha on these jobs, and CI measured up to 2.7e-7 here (NH, whose N core
#: Hamiltonian differs most). With Gaussian's Hcore mixed in it was 1-2e-6.
CROSS_PROGRAM_ENERGY = 5e-7

#: The full-transform oracle uses the same integral code and the same orbitals,
#: so only the frozen-core algebra separates it from the bundle.
SAME_PROGRAM = 1e-8


def _conversion(path: Path, basis: str):
    """The Gaussian -> PySCF map for an s/p basis, built from the .mat header."""
    from pyscf import gto

    m = read_mat(path)
    ell = np.abs(m.ibftyp) // 1000
    assert ell.max() <= 1, "this map covers s and p shells only"
    order = np.lexsort((np.arange(m.nbasis), ell, m.ibfatm))
    mol = gto.M(
        atom=[(int(z), tuple(xyz)) for z, xyz in zip(m.ian, m.coordinates)],
        unit="Bohr",
        basis=basis,
        charge=m.icharg,
        spin=m.multip - 1,
        verbose=0,
    )
    assert mol.nao == m.nbasis
    transform = np.zeros((m.nbasis, m.nbasis))
    transform[np.arange(m.nbasis), order] = 1.0  # C_pyscf[k] = C_gaussian[order[k]]
    overlap = transform.T @ mol.intor("int1e_ovlp") @ transform
    assert np.max(np.abs(overlap - m.matlist["OVERLAP"].expand())) < 1e-6
    return F.AOConversion(
        fch_file="(built from the .mat header in the test)",
        mol=mol,
        transform=transform,
        kind=F.describe_transform(transform),
        checks={},
    )


@pytest.fixture
def pyscf_extract(monkeypatch):
    def _extract(name):
        basis, reference, _ = CASES[name]
        path = PROBES / f"{name}.mat.gz"
        conversion = _conversion(path, basis)
        monkeypatch.setattr(F, "gaussian_to_pyscf", lambda bundle, fch, **kw: conversion)
        bundle = extract(path, reference=reference, integrals="pyscf", fch="unused.fch")
        return bundle, conversion

    return _extract


@pytest.mark.parametrize("name", ["probe_h2o_rhf", "probe_h2o_frozen_virtual"])
def test_pyscf_reproduces_gaussians_own_rhf_integrals(pyscf_extract, name):
    """Where Gaussian's integrals are trustworthy, PySCF must give the same ones."""
    gaussian = extract(PROBES / f"{name}.mat.gz", reference="RHF")
    pyscf_bundle, _ = pyscf_extract(name)
    assert pyscf_bundle.provenance["eri_source"] == "pyscf"
    assert gaussian.provenance["eri_source"] == "gaussian"
    scale = float(np.max(np.abs(gaussian.eri_active)))
    assert np.max(np.abs(pyscf_bundle.eri_active - gaussian.eri_active)) < 2e-6 * scale


@pytest.mark.parametrize("name", list(CASES))
def test_pyscf_integrals_close_the_algebra(pyscf_extract, name):
    from pyscf import scf
    from test_oracles import (
        _full_eri,
        _oracle_determinant_energy,
        _oracle_frozen_core,
        _oracle_to_mo,
    )

    _, reference, e_core = CASES[name]
    bundle, conversion = pyscf_extract(name)
    assert bundle.fock_source == "pyscf_rebuilt"
    result = H.active_hamiltonian(bundle)
    # Every two-electron quantity comes from PySCF, so alpha and beta agree to
    # rounding: the gate the Gaussian ROHF integrals failed by 0.1-0.4 Ha.
    assert result.spin_deviation < 1e-9
    assert result.e_ref == pytest.approx(bundle.escf, abs=CROSS_PROGRAM_ENERGY)
    if e_core is not None:
        assert result.e_core == pytest.approx(e_core, abs=CROSS_PROGRAM_ENERGY)

    # Independent of the windowed algebra: the textbook frozen-core reduction
    # over the full MO integral set, in the same orbitals.
    mo = conversion.mo_coeff_pyscf(bundle.C)
    hcore_mo = _oracle_to_mo(np.asarray(scf.hf.get_hcore(conversion.mol)), mo)
    eri = _full_eri(conversion.mol, mo)
    start, stop = bundle.active_start, bundle.active_stop
    h_eff, oracle_e_core = _oracle_frozen_core(hcore_mo, eri, bundle.enuc, start, start, stop)
    oracle_e_ref = _oracle_determinant_energy(
        hcore_mo, eri, bundle.nalpha, bundle.nbeta, constant=bundle.enuc
    )
    assert np.max(np.abs(result.h_eff - h_eff)) < SAME_PROGRAM
    assert result.e_core == pytest.approx(oracle_e_core, abs=SAME_PROGRAM)
    assert result.e_ref == pytest.approx(oracle_e_ref, abs=SAME_PROGRAM)


@pytest.mark.parametrize("name", ["probe_ch2_rohf", "probe_nh_rohf"])
def test_rohf_is_refused_without_and_accepted_with_pyscf(pyscf_extract, name):
    with pytest.raises(MatFileError, match="integrals pyscf"):
        extract(PROBES / f"{name}.mat.gz", reference="ROHF")
    bundle, _ = pyscf_extract(name)
    assert bundle.provenance["trans_mo_coeff_deviation"] > 0.1  # recorded, not hidden
    assert bundle.nact == 12 if name == "probe_ch2_rohf" else bundle.nact == 10


def test_pyscf_rohf_integrals_are_close_to_gaussians_semicanonical_ones(pyscf_extract):
    """A link to Gaussian for ROHF too: its AA block, rotated into the ROHF
    orbitals, differs from PySCF's only by the semicanonical drift (~1e-4)."""
    name = "probe_ch2_rohf"
    bundle, _ = pyscf_extract(name)
    m = read_mat(PROBES / f"{name}.mat.gz")
    trans = m.matlist["TRANS MO COEFFICIENTS"].expand()[:, :, 0]
    u = trans.T @ bundle.S @ bundle.C[:, bundle.active]
    aa = np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl", m.matlist["AA MO 2E INTEGRALS"].expand(), u, u, u, u,
        optimize=True,
    )
    deviation = float(np.max(np.abs(aa - bundle.eri_active)))
    assert 1e-7 < deviation < 1e-3


def test_pyscf_integrals_need_the_fch():
    with pytest.raises(MatFileError, match="needs the job's .fch"):
        extract(PROBES / "probe_ch2_rohf.mat.gz", reference="ROHF", integrals="pyscf")
