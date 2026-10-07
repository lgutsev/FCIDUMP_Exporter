"""The numpy ``.mat`` reader on real Gaussian 16 output.

``tests/data/g16_c01_probes/`` holds the ten cheap M0 probe jobs exactly as G16
C.01 wrote them on LONI QB4 (job 1072685, 2026-10-06; ``environment.txt``),
gzipped: each ``.mat`` and the ``.fch`` ``formchk`` made from the same
checkpoint. Nothing here needs Gaussian, gauopen, PySCF or MOKIT.

Three independent references are used:

* the ``.fch`` of the same job, written by a different Gaussian code path --
  orbitals, orbital energies, densities, geometry, SCF energy;
* the committed PySCF fixtures in ``tests/data/`` (same geometry, basis and
  window) -- overlap, core Hamiltonian, Fock matrices and, for RHF, the active
  two-electron integrals element by element;
* the algebra itself: ``E_ref`` must reproduce Gaussian's own SCF energy.

The answers to the M0 questions recorded in ``gaussian/README.md`` ("M0
outcome") are the assertions in this file.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from g16dump import hamiltonian as H
from g16dump.bundle import load
from g16dump.fch import read_fch_array, read_fch_scalar
from g16dump.matfile import MatFileError, MatObject, extract, read_mat, survey

PROBES_DIR = Path(__file__).resolve().parent / "data" / "g16_c01_probes"

#: name -> (nbasis, nelec, multiplicity, NFC, NFV); the window is NFC+1..NBsUse-NFV.
PROBES = {
    "probe_h2_smoke": (2, 2, 1, 0, 0),
    "probe_h2o_rhf": (7, 10, 1, 1, 0),
    "probe_h2o_nowindow": (7, 10, 1, 1, 0),
    "probe_h2o_frozen_virtual": (13, 10, 1, 1, 3),
    "probe_h2o_rks": (13, 10, 1, 1, 0),
    "probe_h2o_rks_631gs": (19, 10, 1, 1, 7),
    "probe_ch2_rohf": (13, 8, 3, 1, 0),
    "probe_ch2_rohf_ccpvdz": (24, 8, 3, 1, 11),
    "probe_ch2_uhf": (13, 8, 3, 1, 0),
    "probe_nh_rohf": (11, 8, 3, 1, 0),
}
ROHF = ("probe_ch2_rohf", "probe_ch2_rohf_ccpvdz", "probe_nh_rohf")
OPEN_SHELL = ROHF + ("probe_ch2_uhf",)

#: Probe job -> committed PySCF fixture with the same geometry, basis, window.
FIXTURES = {
    "probe_h2o_rhf": "h2o_rhf",
    "probe_h2o_frozen_virtual": "h2o_frozen_virtual",
    "probe_h2o_rks": "h2o_rks",
    "probe_ch2_rohf": "ch2_rohf",
    "probe_nh_rohf": "nh_rohf",
}

#: Gaussian against PySCF: different integral codes, basis-set digits and SCF
#: convergence. Measured: at most 3.4e-7 of the largest element (N in NH's
#: 6-31G core Hamiltonian, 8.5e-6 absolute). For orbitals, absolute.
CROSS_PROGRAM = 2e-6


def _assert_cross_program(ours, reference, what=""):
    scale = max(1.0, float(np.max(np.abs(reference))))
    deviation = float(np.max(np.abs(ours - reference)))
    assert deviation < CROSS_PROGRAM * scale, f"{what}: {deviation:.2e} vs scale {scale:.1f}"


@pytest.fixture(scope="module")
def mat():
    cache: dict = {}

    def _mat(name):
        if name not in cache:
            cache[name] = read_mat(PROBES_DIR / f"{name}.mat.gz")
        return cache[name]

    return _mat


@pytest.fixture(scope="module")
def fch(tmp_path_factory):
    """``fch(name)`` -> path of the job's decompressed ``.fch``."""
    directory = tmp_path_factory.mktemp("fch")

    def _fch(name):
        out = directory / f"{name}.fch"
        if not out.exists():
            out.write_bytes(gzip.decompress((PROBES_DIR / f"{name}.fch.gz").read_bytes()))
        return out

    return _fch


def _expand(m, label):
    return m.matlist[label].expand()


def _tri(n, packed):
    out = np.zeros((n, n))
    out[np.tril_indices(n)] = packed
    return out + np.tril(out, -1).T


def _pyscf_order(m) -> np.ndarray:
    """Gaussian AO index for each PySCF AO, for s and p shells.

    PySCF groups each atom's shells by angular momentum; Gaussian keeps them in
    shell order (6-31G's SP shells interleave s and p). Read from the header's
    IBfAtm/IBfTyp, whose type code is 1000*l + component.
    """
    ell = np.abs(m.ibftyp) // 1000
    assert ell.max() <= 1, "this map covers s and p functions only"
    return np.lexsort((np.arange(m.nbasis), ell, m.ibfatm))


def _sign_align(columns, reference):
    signs = np.where(np.sum(columns * reference, axis=0) < 0, -1.0, 1.0)
    return columns * signs, signs


# ------------------------------------------------------------- the format


@pytest.mark.parametrize("name", list(PROBES))
def test_every_record_parses_and_every_object_expands(mat, name):
    m = mat(name)
    assert m.gaussian_version == "ES64L-G16RevC.01"
    assert (m.ivers, m.nlab) == (2, 11)
    assert m.matlist, "no objects read"
    for label, obj in m.matlist.items():
        expanded = obj.expand()
        assert np.all(np.isfinite(expanded)), label
        if not obj.is_sparse:
            assert obj.array.size == obj.ntot, label


@pytest.mark.parametrize("name", list(PROBES))
def test_header_matches_the_fch(mat, fch, name):
    m, f = mat(name), fch(name)
    nbasis, nelec, multiplicity, nfc, nfv = PROBES[name]
    assert (m.nbasis, m.nbsuse, m.ne, m.multip, m.icharg) == (
        nbasis, nbasis, nelec, multiplicity, 0,
    )
    assert m.nbasis == read_fch_scalar(f, "Number of basis functions")
    assert m.ne == read_fch_scalar(f, "Number of electrons")
    assert m.multip == read_fch_scalar(f, "Multiplicity")
    np.testing.assert_array_equal(m.ian, read_fch_array(f, "Atomic numbers"))
    np.testing.assert_allclose(m.atmchg, read_fch_array(f, "Nuclear charges"))
    np.testing.assert_allclose(
        m.coordinates.ravel(), read_fch_array(f, "Current cartesian coordinates"),
        atol=1e-8,
    )
    assert m.nshell == read_fch_array(f, "Shell types").size
    assert m.nprimitive == read_fch_array(f, "Primitive exponents").size
    # IOpCl/ICGU say "two spin blocks", not "unrestricted": ROHF and UHF match.
    assert (m.iopcl, m.icgu) == ((1, 112) if name in OPEN_SHELL else (0, 111))


@pytest.mark.parametrize("name", list(PROBES))
def test_scalars_match_the_fch(mat, fch, name):
    m, f = mat(name), fch(name)
    assert m.scalar("ESCF") == pytest.approx(read_fch_scalar(f, "SCF Energy"), abs=1e-12)
    assert m.scalar("ETOTAL") == pytest.approx(read_fch_scalar(f, "Total Energy"), abs=1e-12)
    assert m.scalar("VIRIAL RATIO") == pytest.approx(
        read_fch_scalar(f, "Virial Ratio"), abs=1e-12
    )
    if name in OPEN_SHELL:
        assert m.scalar("S**2") == pytest.approx(read_fch_scalar(f, "S**2"), abs=1e-12)


def test_nuclear_repulsion_matches_pyscf(mat, fixture_path):
    for name, fixture in FIXTURES.items():
        assert mat(name).scalar("ENUCREP") == pytest.approx(
            load(fixture_path(fixture)).enuc, abs=1e-10
        )


# -------------------------------------------------- M0: ITran and the window


@pytest.mark.parametrize("name", list(PROBES))
def test_itran_is_5_and_the_window_restricts_the_integrals(mat, name):
    """ITran=5 everywhere, and the 2e block spans exactly NBsUse-NFC-NFV."""
    m = mat(name)
    _, _, _, nfc, nfv = PROBES[name]
    assert m.itran == 5
    assert (m.nfc, m.nfv) == (nfc, nfv)
    nact = m.nbsuse - nfc - nfv
    assert _expand(m, "AA MO 2E INTEGRALS").shape == (nact,) * 4
    trans = _expand(m, "TRANS MO COEFFICIENTS")
    assert trans.shape[:2] == (m.nbasis, nact)


def test_the_no_window_control_is_gaussians_default_frozen_core(mat):
    """Without Window= Gaussian still freezes the core (its FC default).

    So probe_h2o_nowindow came back identical to probe_h2o_rhf, not with a
    7-orbital block. It does not prove Window= works; the frozen-virtual jobs
    do (NFV = 3, 11 and 7 above, exactly as their routes asked).
    """
    with_window, without = mat("probe_h2o_rhf"), mat("probe_h2o_nowindow")
    assert without.nfc == 1
    assert with_window.matlist.keys() == without.matlist.keys()
    for label, obj in with_window.matlist.items():
        np.testing.assert_array_equal(obj.expand(), without.matlist[label].expand())


# ------------------------------------------- one-electron blocks vs the .fch


@pytest.mark.parametrize("name", list(PROBES))
def test_orbitals_and_orbital_energies_match_the_fch(mat, fch, name):
    m, f = mat(name), fch(name)
    overlap = _expand(m, "OVERLAP")
    spins = ["Alpha", "Beta"] if name == "probe_ch2_uhf" else ["Alpha"]
    for spin in spins:
        coeff = _expand(m, f"{spin.upper()} MO COEFFICIENTS")  # [ao, mo]
        from_fch = read_fch_array(f, f"{spin} MO coefficients").reshape(-1, m.nbasis).T
        np.testing.assert_allclose(coeff, from_fch, atol=1e-8)
        np.testing.assert_allclose(coeff.T @ overlap @ coeff, np.eye(m.nbsuse), atol=1e-12)
        np.testing.assert_allclose(
            _expand(m, f"{spin.upper()} ORBITAL ENERGIES"),
            read_fch_array(f, f"{spin} Orbital Energies"),
            atol=1e-7,
        )


@pytest.mark.parametrize("name", list(PROBES))
def test_scf_densities_match_the_fch(mat, fch, name):
    m, f = mat(name), fch(name)
    alpha = _expand(m, "ALPHA SCF DENSITY MATRIX")
    beta = _expand(m, "BETA SCF DENSITY MATRIX") if name in OPEN_SHELL else alpha
    np.testing.assert_allclose(
        alpha + beta, _tri(m.nbasis, read_fch_array(f, "Total SCF Density")), atol=1e-8
    )
    if name in OPEN_SHELL:
        np.testing.assert_allclose(
            alpha - beta, _tri(m.nbasis, read_fch_array(f, "Spin SCF Density")), atol=1e-8
        )
    assert np.trace((alpha + beta) @ _expand(m, "OVERLAP")) == pytest.approx(m.ne)


# --------------------------------------- one-electron blocks vs PySCF fixtures


@pytest.mark.parametrize("name", list(FIXTURES))
def test_overlap_and_core_hamiltonian_match_pyscf(mat, fixture_path, name):
    m, fixture = mat(name), load(fixture_path(FIXTURES[name]))
    order = _pyscf_order(m)
    for label, reference in (("OVERLAP", fixture.S), ("CORE HAMILTONIAN ALPHA", fixture.Hcore_ao)):
        ours = _expand(m, label)[np.ix_(order, order)]
        _assert_cross_program(ours, reference, label)


@pytest.mark.parametrize("name", ["probe_h2o_rhf", "probe_h2o_frozen_virtual"])
def test_rhf_fock_matrix_matches_pyscf(mat, fixture_path, name):
    m, fixture = mat(name), load(fixture_path(FIXTURES[name]))
    order = _pyscf_order(m)
    _assert_cross_program(
        _expand(m, "ALPHA FOCK MATRIX")[np.ix_(order, order)], fixture.F_alpha_ao, "F"
    )


@pytest.mark.parametrize("name", ["probe_ch2_rohf", "probe_nh_rohf"])
def test_rohf_stores_genuine_alpha_and_beta_fock_matrices(mat, fixture_path, name):
    """M0: F^alpha and F^beta, not the Roothaan effective operator.

    Three independent signs. They match PySCF's F^sigma = h + J[Pa+Pb] -
    K[Psigma], which depends on the densities only and so not on how either
    program canonicalizes its ROHF orbitals. They differ from each other, as
    F^alpha and F^beta must and one Roothaan operator could not. And they
    satisfy the ROHF stationarity conditions in the MO basis.
    """
    m, fixture = mat(name), load(fixture_path(FIXTURES[name]))
    order = _pyscf_order(m)
    f_alpha, f_beta = _expand(m, "ALPHA FOCK MATRIX"), _expand(m, "BETA FOCK MATRIX")
    _assert_cross_program(f_alpha[np.ix_(order, order)], fixture.F_alpha_ao, "F alpha")
    _assert_cross_program(f_beta[np.ix_(order, order)], fixture.F_beta_ao, "F beta")
    assert np.max(np.abs(f_alpha - f_beta)) > 0.1

    coeff = _expand(m, "ALPHA MO COEFFICIENTS")
    fa, fb = coeff.T @ f_alpha @ coeff, coeff.T @ f_beta @ coeff
    nalpha, nbeta = (m.ne + m.multip - 1) // 2, (m.ne - m.multip + 1) // 2
    docc, socc, virt = slice(0, nbeta), slice(nbeta, nalpha), slice(nalpha, None)
    assert np.max(np.abs((fa + fb)[docc, virt])) < 1e-9
    assert np.max(np.abs(fb[docc, socc])) < 1e-9
    assert np.max(np.abs(fa[socc, virt])) < 1e-9


@pytest.mark.parametrize("name", ["probe_h2o_rks", "probe_h2o_rks_631gs"])
def test_a_kohn_sham_job_stores_the_ks_matrix_under_the_fock_label(mat, name):
    """M0: there is no KS label. The KS matrix is ``ALPHA FOCK MATRIX``."""
    m = mat(name)
    coeff = _expand(m, "ALPHA MO COEFFICIENTS")
    in_mo = coeff.T @ _expand(m, "ALPHA FOCK MATRIX") @ coeff
    np.testing.assert_allclose(in_mo, np.diag(_expand(m, "ALPHA ORBITAL ENERGIES")), atol=1e-10)
    assert "BETA FOCK MATRIX" not in m.matlist


# ------------------------------------------------ the two-electron packing


def test_the_eightfold_packing_order_matches_pyscf_element_by_element(mat, fixture_path):
    """(ij|kl) = eri[i, j, k, l], chemists' order, verified on all 1296 elements."""
    m, fixture = mat("probe_h2o_rhf"), load(fixture_path("h2o_rhf"))
    coeff, signs = _sign_align(_expand(m, "ALPHA MO COEFFICIENTS"), fixture.C)
    np.testing.assert_allclose(coeff, fixture.C, atol=CROSS_PROGRAM)
    s = signs[1:]
    eri = np.einsum("ijkl,i,j,k,l->ijkl", _expand(m, "AA MO 2E INTEGRALS"), s, s, s, s)
    np.testing.assert_allclose(eri, fixture.eri_active, atol=1e-7)
    # The check has teeth: the physicists' reading is wrong by a lot.
    assert np.max(np.abs(eri.transpose(0, 2, 1, 3) - fixture.eri_active)) > 0.1


def test_frozen_virtual_integrals_match_pyscf(mat, fixture_path):
    m, fixture = mat("probe_h2o_frozen_virtual"), load(fixture_path("h2o_frozen_virtual"))
    order = _pyscf_order(m)
    coeff, signs = _sign_align(_expand(m, "ALPHA MO COEFFICIENTS")[order], fixture.C)
    np.testing.assert_allclose(coeff, fixture.C, atol=CROSS_PROGRAM)
    s = signs[1:10]
    eri = np.einsum("ijkl,i,j,k,l->ijkl", _expand(m, "AA MO 2E INTEGRALS"), s, s, s, s)
    np.testing.assert_allclose(eri, fixture.eri_active, atol=1e-7)


def test_a_full_space_job_rebuilds_its_own_fock_matrix(mat):
    """H2 has no frozen orbitals, so F and E(SCF) follow from h and (pq|rs) alone."""
    m = mat("probe_h2_smoke")
    coeff = _expand(m, "ALPHA MO COEFFICIENTS")
    h = coeff.T @ _expand(m, "CORE HAMILTONIAN ALPHA") @ coeff
    eri = _expand(m, "AA MO 2E INTEGRALS")
    occ = slice(0, 1)
    fock = h + 2 * np.einsum("pqii->pq", eri[:, :, occ, occ]) - np.einsum(
        "piiq->pq", eri[:, occ, occ, :]
    )
    np.testing.assert_allclose(fock, coeff.T @ _expand(m, "ALPHA FOCK MATRIX") @ coeff, atol=1e-10)
    energy = m.scalar("ENUCREP") + np.trace(h[occ, occ] + fock[occ, occ])
    assert energy == pytest.approx(m.scalar("ESCF"), abs=1e-10)


# --------------------------------------------------- through extract


@pytest.mark.parametrize(
    "name, e_core",
    [
        ("probe_h2o_rhf", -51.4711696318),
        ("probe_h2o_nowindow", -51.4711696318),
        ("probe_h2o_frozen_virtual", -52.1215325375),
        ("probe_h2_smoke", None),
    ],
)
def test_rhf_jobs_extract_and_reproduce_the_scf_energy(mat, name, e_core):
    """M0: the stored RHF Fock matrix closes the windowed algebra."""
    bundle = extract(PROBES_DIR / f"{name}.mat.gz", reference="RHF")
    assert bundle.fock_source == "gaussian"
    assert bundle.provenance["itran"] == 5
    assert bundle.provenance["trans_mo_coeff_deviation"] == 0.0
    result = H.active_hamiltonian(bundle)
    assert result.spin_deviation == 0.0
    assert result.e_ref == pytest.approx(bundle.escf, abs=1e-9)
    if e_core is not None:  # PySCF's number, from the gaussian/README.md table
        assert result.e_core == pytest.approx(e_core, abs=1e-7)


@pytest.mark.parametrize("name", ["probe_h2o_rks", "probe_h2o_rks_631gs"])
def test_ks_jobs_extract_without_a_fock_matrix(name):
    bundle = extract(PROBES_DIR / f"{name}.mat.gz", reference="RKS")
    assert bundle.fock_source == "none" and not bundle.has_fock
    assert bundle.provenance["trans_mo_coeff_deviation"] == 0.0


# ------------------------------------------------------- beta objects


def test_uhf_beta_objects_are_read_and_uhf_is_refused(mat):
    m = mat("probe_ch2_uhf")
    alpha, beta = _expand(m, "ALPHA MO COEFFICIENTS"), _expand(m, "BETA MO COEFFICIENTS")
    assert np.max(np.abs(alpha - beta)) > 1.0
    trans = _expand(m, "TRANS MO COEFFICIENTS")
    assert trans.shape == (13, 12, 2)
    np.testing.assert_array_equal(trans[:, :, 0], alpha[:, 1:])
    np.testing.assert_array_equal(trans[:, :, 1], beta[:, 1:])
    for label in ("AA MO 2E INTEGRALS", "BA MO 2E INTEGRALS", "BB MO 2E INTEGRALS"):
        assert _expand(m, label).shape == (12,) * 4
    assert m.matlist["BA MO 2E INTEGRALS"].dims[:4] == (-12, 12, -12, 12)
    # Beta orbitals diagonalize the beta Fock matrix: a genuine UHF pair.
    in_mo = beta.T @ _expand(m, "BETA FOCK MATRIX") @ beta
    np.testing.assert_allclose(in_mo, np.diag(_expand(m, "BETA ORBITAL ENERGIES")), atol=1e-10)
    with pytest.raises(MatFileError, match="UHF reference"):
        extract(PROBES_DIR / "probe_ch2_uhf.mat.gz", reference="ROHF")
    notes = " ".join(survey(PROBES_DIR / "probe_ch2_uhf.mat.gz")["notes"])
    assert "UHF reference" in notes and "semicanonicalizes" not in notes


def _rotate(eri, *matrices):
    return np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, *matrices, optimize=True)


@pytest.mark.parametrize("name", ROHF)
def test_rohf_integrals_are_over_semicanonical_orbitals(mat, name):
    """M0: why the ROHF jobs cannot be extracted.

    ROHF writes BETA MO COEFFICIENTS equal to the alpha ones, both Fock
    matrices and all three integral blocks. But TRANS MO COEFFICIENTS, the
    orbitals the transformation used, are per-spin semicanonical: each
    diagonalizes its own F^sigma within the occupied and the virtual block.
    Their active span has drifted from the ROHF one through mixing with the
    frozen orbitals, so the integrals cannot be rotated back exactly.
    """
    m = mat(name)
    coeff = _expand(m, "ALPHA MO COEFFICIENTS")
    np.testing.assert_array_equal(coeff, _expand(m, "BETA MO COEFFICIENTS"))
    overlap = _expand(m, "OVERLAP")
    trans = _expand(m, "TRANS MO COEFFICIENTS")
    nact = trans.shape[1]
    active = coeff[:, m.nfc : m.nfc + nact]
    assert np.max(np.abs(trans[:, :, 0] - active)) > 0.1

    nalpha, nbeta = (m.ne + m.multip - 1) // 2, (m.ne - m.multip + 1) // 2
    rotations = []
    for spin, label, nocc in ((0, "ALPHA FOCK MATRIX", nalpha), (1, "BETA FOCK MATRIX", nbeta)):
        used = trans[:, :, spin]
        in_mo = used.T @ _expand(m, label) @ used
        occ = nocc - m.nfc
        for block in (in_mo[:occ, :occ], in_mo[occ:, occ:]):
            assert np.max(np.abs(block - np.diag(np.diag(block)))) < 1e-12
        u = used.T @ overlap @ active
        defect = np.max(np.abs(u @ u.T - np.eye(nact)))
        assert 1e-7 < defect < 1e-2  # close to the ROHF space, not in it
        rotations.append(u)

    # BA is (beta beta | alpha alpha): the only reading under which all three
    # blocks describe the same integrals, to within that drift. For CH2 that
    # BA block spans two data records, so this also checks their joining.
    ua, ub = rotations
    if name != "probe_nh_rohf":  # 78**2 = 6084 values: two 4000-word records
        obj = m.matlist["BA MO 2E INTEGRALS"]
        assert obj.ntot > obj.lenbuf
    aa = _rotate(_expand(m, "AA MO 2E INTEGRALS"), ua, ua, ua, ua)
    bb = _rotate(_expand(m, "BB MO 2E INTEGRALS"), ub, ub, ub, ub)
    ba = _expand(m, "BA MO 2E INTEGRALS")
    drift = np.max(np.abs(aa - bb))
    assert drift < 0.05
    assert np.max(np.abs(_rotate(ba, ub, ub, ua, ua) - aa)) < 0.05
    assert np.max(np.abs(_rotate(ba, ua, ua, ub, ub) - aa)) > 0.1


@pytest.mark.parametrize("name", ROHF)
def test_extract_refuses_rohf_semicanonical_integrals(name):
    with pytest.raises(MatFileError, match="semicanonicalizes"):
        extract(PROBES_DIR / f"{name}.mat.gz", reference="ROHF")
    report = survey(PROBES_DIR / f"{name}.mat.gz")
    assert any("semicanonicalizes" in note for note in report["notes"])


# ----------------------------------------------------- survey and errors


def test_survey_of_a_real_file(name="probe_h2o_rhf"):
    report = survey(PROBES_DIR / f"{name}.mat.gz")
    assert report["header"]["itran"] == 5
    assert report["eri_nact"] == 6
    assert report["trans_mo_coeff_deviation"] == 0.0
    assert report["objects"]["AA MO 2E INTEGRALS"]["dims"] == [-6, -6, -6, 6, 1]
    assert not any("cannot run" in note for note in report["notes"])


def test_a_truncated_file_is_refused(tmp_path):
    raw = gzip.decompress((PROBES_DIR / "probe_h2_smoke.mat.gz").read_bytes())
    for cut in (len(raw) - 3, len(raw) - 2000, 200):
        path = tmp_path / f"cut{cut}.mat"
        path.write_bytes(raw[:cut])
        with pytest.raises(MatFileError):
            read_mat(path)


def test_a_file_that_is_not_a_mat_is_refused(tmp_path):
    path = tmp_path / "job.fch.mat"
    path.write_bytes(gzip.decompress((PROBES_DIR / "probe_h2_smoke.fch.gz").read_bytes()))
    with pytest.raises(MatFileError):
        read_mat(path)
    with pytest.raises(MatFileError, match="no such file"):
        read_mat(tmp_path / "absent.mat")


# ----------------------------------------------- packing, synthetically


def _packed_object(label, dims, data, asym=0):
    data = np.asarray(data, dtype=np.float64)
    return MatObject(label, 0, 1, data.size, 4000, list(dims) + [0] * (5 - len(dims)),
                     asym, None, data)


def test_fourfold_packing_round_trips():
    rng = np.random.default_rng(7)
    n = 4
    rows, cols = np.tril_indices(n)
    npair = rows.size
    square = rng.normal(size=(npair, npair))  # [ij pair, kl pair], no pair symmetry
    obj = _packed_object("BA MO 2E INTEGRALS", (-n, n, -n, n, 1), square.ravel(order="F"))
    eri = obj.expand()
    for p, (i, j) in enumerate(zip(rows, cols)):
        for q, (k, m) in enumerate(zip(rows, cols)):
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, m), (m, k)):
                    assert eri[a, b, c, d] == square[p, q]


def test_eightfold_packing_round_trips():
    rng = np.random.default_rng(8)
    n = 3
    factors = rng.normal(size=(n, n, 5))
    factors = factors + factors.transpose(1, 0, 2)
    eri = np.einsum("ijx,klx->ijkl", factors, factors)
    rows, cols = np.tril_indices(n)
    pairs = list(zip(rows, cols))
    packed = [eri[pairs[p] + pairs[q]] for p in range(len(pairs)) for q in range(p + 1)]
    obj = _packed_object("AA MO 2E INTEGRALS", (-n, -n, -n, n, 1), packed)
    np.testing.assert_allclose(obj.expand(), eri)


def test_antisymmetric_triangles_and_stacks(mat):
    m = mat("probe_h2o_rhf")
    for label in ("DIP VEL INTEGRALS", "R X DEL INTEGRALS"):
        assert m.matlist[label].asym == -1
        out = _expand(m, label)
        assert out.shape == (7, 7, 3)
        assert np.max(np.abs(out)) > 0.1
        np.testing.assert_array_equal(out, -out.transpose(1, 0, 2))
    dipole = _expand(m, "DIPOLE INTEGRALS")
    np.testing.assert_array_equal(dipole, dipole.transpose(1, 0, 2))


def test_an_unknown_packing_is_refused():
    obj = _packed_object("ODD", (-2, -2, 2), np.zeros(6))
    with pytest.raises(MatFileError, match="packing"):
        obj.expand()
