"""End-to-end scientific gates for the rebuilt-Fock path through MOKIT.

A Gaussian-side job (AO quantities in Gaussian's AO basis, a formchk-layout
``.fch``) goes through ``g16dump.fch.with_rebuilt_fock_from_fch`` and then the
ordinary frozen-core fold, and every output is compared with an independent
oracle: a full AO->MO transform plus the textbook reduction, in the original
PySCF calculation's own basis, sharing no code with ``g16dump`` and never
passing through Gaussian's AO order or MOKIT (``test_oracles.py``).

The systems cover an RHF control whose stored Fock matrix is sound, ROHF jobs
whose stored matrix is Gaussian's Roothaan operator or absent, a Kohn-Sham job,
and pure and Cartesian polarized bases; see ``gaussian_jobs.JOBS``.

The last part drives the same path through the command line exactly as the
README documents it: extract, rebuild-fock, validate, dump.

Tolerances. The scientific gate is 1e-8 Ha. The ``.fch`` stores basis
exponents to nine significant figures, so PySCF's J and K are built in a basis
that differs from the ``.mat``'s by that rounding; the measured effect on every
quantity below is 1e-9 Ha or smaller, so the gate is kept as it is.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from fcidump_oracle import determinant_energy, read_fcidump
from gaussian_jobs import JOBS

from g16dump import fch as F
from g16dump import hamiltonian as H

pytestmark = [pytest.mark.mokit, pytest.mark.pyscf]

GATE = 1e-8


@pytest.fixture(scope="module", params=list(JOBS))
def job(request, gaussian_job):
    return gaussian_job(request.param)


@pytest.fixture(scope="module")
def rebuilt(job):
    bundle, _ = F.with_rebuilt_fock_from_fch(job.bundle, job.fch)
    return bundle


@pytest.fixture(scope="module")
def result(rebuilt):
    return H.active_hamiltonian(rebuilt)


# -------------------------------------------------------- the stored matrices


def test_what_the_mat_carried_is_handled_as_before(job):
    """Control: the rebuild is needed exactly where the stored matrix fails."""
    stored = job.spec["stored_fock"]
    if stored == "gaussian":
        own = H.active_hamiltonian(job.bundle)
        assert np.max(np.abs(own.h_eff - job.oracle_h_eff)) < GATE
    elif stored == "roothaan":
        with pytest.raises(H.SpinConsistencyError) as excinfo:
            H.active_hamiltonian(job.bundle)
        assert excinfo.value.deviation > 1e-2
    else:
        with pytest.raises(H.HamiltonianError):
            H.active_hamiltonian(job.bundle)


# ------------------------------------------------------------- the gates


def test_rebuilt_fock_passes_the_spin_gate(result):
    assert result.spin_deviation < GATE
    assert result.fock_source == "pyscf_rebuilt"


def test_h_eff_matches_the_oracle(job, result):
    assert np.max(np.abs(result.h_eff - job.oracle_h_eff)) < GATE


def test_active_eris_match_the_oracle(job, result):
    assert np.max(np.abs(result.eri_active - job.oracle_eri_active)) < 1e-10


def test_core_energy_matches_the_oracle(job, result):
    assert abs(result.e_core - job.oracle_e_core) < GATE


def test_reference_energy_matches_the_oracle(job, result):
    assert abs(result.e_ref - job.oracle_e_ref) < GATE
    assert abs(result.reference_energy_from_parts() - job.oracle_e_ref) < GATE


def test_reference_energy_is_the_scf_energy_for_hartree_fock(job, result):
    if job.spec["reference"] == "RKS":
        # The HF energy of the KS determinant, which lies above the DFT energy
        # of the same job by a wide margin; equality would be the bug signal.
        assert result.e_ref - job.mf.e_tot > 1e-2
    else:
        assert abs(result.e_ref - job.mf.e_tot) < GATE


def test_rhf_control_rebuild_agrees_with_the_stored_fock(job, result):
    if job.spec["stored_fock"] != "gaussian":
        pytest.skip("only the RHF controls carry a sound stored Fock matrix")
    own = H.active_hamiltonian(job.bundle)
    assert np.max(np.abs(own.h_eff - result.h_eff)) < GATE
    assert abs(own.e_core - result.e_core) < GATE


def test_fcidump_round_trip(job, result, tmp_path):
    from g16dump.write import write_fcidump

    path = write_fcidump(result, tmp_path / "FCIDUMP")
    dump = read_fcidump(path)
    assert (dump["NORB"], dump["NELEC"], dump["MS2"]) == (
        job.bundle.nact, job.bundle.nelec_active, job.bundle.ms2,
    )
    assert np.max(np.abs(dump["H1"] - job.oracle_h_eff)) < GATE
    assert np.max(np.abs(dump["H2"] - job.oracle_eri_active)) < 1e-10
    assert abs(dump["ECORE"] - job.oracle_e_core) < GATE
    energy = determinant_energy(
        dump["H1"], dump["H2"], dump["ECORE"],
        job.bundle.nocc_active_alpha, job.bundle.nocc_active_beta,
    )
    assert abs(energy - job.oracle_e_ref) < GATE


# --------------------------------------------------- the documented CLI route


def _fake_mat(job, monkeypatch):
    """Serve the job's Gaussian-side quantities to ``extract`` as a .mat would."""
    from dataclasses import replace

    from test_matfile import FakeMatEl

    from g16dump import matfile

    bundle = job.bundle
    if bundle.F_alpha_ao is None:
        # A real .mat carries some Fock-labelled block even here: the KS matrix
        # for a KS job. extract must ignore it, so hand it one.
        x = job.x_independent
        stored = x.T @ np.asarray(job.mf.get_fock()) @ x
        bundle = replace(bundle, F_alpha_ao=stored, F_beta_ao=stored)
    matel = FakeMatEl(bundle)
    monkeypatch.setattr(matfile, "read_matel", lambda path: matel)


@pytest.mark.parametrize(
    "name", ["ch2_rohf_ccpvdz", "h2o_rks_ccpvdz", "nh_rohf_631gs_cart"]
)
def test_cli_extract_rebuild_validate_dump(name, gaussian_job, monkeypatch, tmp_path, capsys):
    from g16dump.bundle import load
    from g16dump.cli import main

    job = gaussian_job(name)
    _fake_mat(job, monkeypatch)
    first, last = job.spec["window"]
    extracted = tmp_path / "JOB.npz"
    rebuilt = tmp_path / "JOB_rebuilt.npz"
    fcidump = tmp_path / "FCIDUMP"

    assert main([
        "extract", "JOB.mat", "--reference", job.spec["reference"],
        "--fch", str(job.fch), "--window", str(first), str(last),
        "--out", str(extracted),
    ]) == 0
    digest = hashlib.sha256(extracted.read_bytes()).hexdigest()

    capsys.readouterr()
    assert main([
        "rebuild-fock", str(extracted), "--fch", str(job.fch), "--out", str(rebuilt),
    ]) == 0
    report = capsys.readouterr().out
    assert "MOKIT fch2py" in report
    assert "h' alpha/beta agreement" in report

    # The input bundle is exactly as extract left it.
    assert hashlib.sha256(extracted.read_bytes()).hexdigest() == digest
    assert load(extracted).fock_source != "pyscf_rebuilt"

    assert main(["validate", str(rebuilt), "--hamiltonian"]) == 0
    assert main(["dump", str(rebuilt), "--out", str(fcidump)]) == 0

    dump = read_fcidump(fcidump)
    assert np.max(np.abs(dump["H1"] - job.oracle_h_eff)) < GATE
    assert abs(dump["ECORE"] - job.oracle_e_core) < GATE

    sidecar = json.loads((tmp_path / "FCIDUMP.provenance.json").read_text())
    record = sidecar["fock_rebuild"]
    assert sidecar["fock_source"] == "pyscf_rebuilt"
    assert record["ao_conversion"]["method"].startswith("mokit fch2py")
    assert record["ao_conversion"]["checks"]["overlap_before"] > 0.1
    assert record["ao_conversion"]["checks"]["overlap_after"] < 1e-8
    assert record["input_bundle"] == str(extracted)
    assert record["fch_sha256"] == hashlib.sha256(job.fch.read_bytes()).hexdigest()
    assert record["spin_deviation"] < GATE
    # Provenance from extract survives the rebuild.
    assert sidecar["fch_file"] == str(job.fch)
    assert sidecar["window_1based"] == [first, last]


def test_cli_rebuild_refuses_to_overwrite(gaussian_job, monkeypatch, tmp_path, capsys):
    from g16dump.cli import main

    job = gaussian_job("ch2_rohf_ccpvdz")
    _fake_mat(job, monkeypatch)
    first, last = job.spec["window"]
    extracted = tmp_path / "JOB.npz"
    main([
        "extract", "JOB.mat", "--reference", "ROHF", "--window", str(first), str(last),
        "--out", str(extracted),
    ])
    capsys.readouterr()

    assert main(["rebuild-fock", str(extracted), "--fch", str(job.fch),
                 "--out", str(extracted)]) == 1
    assert "never replaces its input" in capsys.readouterr().err
    assert main(["rebuild-fock", str(extracted), "--fch", str(job.fch),
                 "--out", str(extracted), "--force"]) == 1

    target = tmp_path / "out.npz"
    target.write_bytes(b"something else")
    assert main(["rebuild-fock", str(extracted), "--fch", str(job.fch),
                 "--out", str(target)]) == 1
    assert "exists" in capsys.readouterr().err
    assert target.read_bytes() == b"something else"
    assert main(["rebuild-fock", str(extracted), "--fch", str(job.fch),
                 "--out", str(target), "--force"]) == 0


def test_cli_rebuild_with_the_wrong_fch_writes_nothing(gaussian_job, monkeypatch, tmp_path, capsys):
    from g16dump.cli import main

    job = gaussian_job("ch2_rohf_ccpvdz")
    other = gaussian_job("ch2_rohf_631g")
    _fake_mat(job, monkeypatch)
    first, last = job.spec["window"]
    extracted = tmp_path / "JOB.npz"
    main([
        "extract", "JOB.mat", "--reference", "ROHF", "--window", str(first), str(last),
        "--out", str(extracted),
    ])
    out = tmp_path / "rebuilt.npz"
    assert main(["rebuild-fock", str(extracted), "--fch", str(other.fch),
                 "--out", str(out)]) == 1
    assert "not from the job" in capsys.readouterr().err
    assert not out.exists()
