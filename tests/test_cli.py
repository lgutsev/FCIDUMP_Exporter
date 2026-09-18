"""The command line.

The CLI is where a wrong answer becomes a file someone runs a solver on, so
these tests care about two things: that no path is hardcoded, and that the
things the package refuses to do are refused here too, in words, with a nonzero
exit status.
"""

from __future__ import annotations

import numpy as np
import pytest

from g16dump.bundle import load, save
from g16dump.cli import main


def _run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out + captured.err


# ----------------------------------------------------------------- validate

def test_validate_accepts_a_good_bundle(capsys, fixture_path):
    code, output = _run(capsys, "validate", str(fixture_path("ch2_rohf")))
    assert code == 0
    assert "valid against schema v2" in output
    assert "active window MOs 2-13" in output
    assert "Fock source   pyscf_rebuilt" in output


def test_validate_reports_the_hamiltonian_and_checks_it_against_the_scf(
    capsys, fixture_path
):
    code, output = _run(
        capsys, "validate", str(fixture_path("ch2_rohf")), "--hamiltonian"
    )
    assert code == 0
    assert "E_core" in output and "E_ref" in output
    assert "agrees" in output
    assert "DISAGREES" not in output


def test_validate_surfaces_the_spin_consistency_failure(capsys, fixture_path):
    """The Roothaan fixture must fail here, in words, with a nonzero status."""
    code, output = _run(
        capsys, "validate", str(fixture_path("ch2_rohf_roothaan")), "--hamiltonian"
    )
    assert code == 1
    assert "Do not average" in output
    assert "rebuild_fock" in output


def test_validate_can_print_provenance(capsys, fixture_path):
    code, output = _run(
        capsys, "validate", str(fixture_path("h2o_rhf")), "--provenance"
    )
    assert code == 0
    assert "generated_by" in output
    assert "schema_version" in output


def test_validate_rejects_a_broken_bundle(capsys, tmp_path, synthetic):
    path = save(synthetic, tmp_path / "b.npz")
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    payload["nact"] = np.int64(3)
    np.savez(path, **payload)

    code, output = _run(capsys, "validate", str(path))
    assert code == 1
    assert "failed validation" in output


def test_validate_warns_about_kohn_sham_orbitals(capsys, tmp_path, synthetic):
    from dataclasses import replace

    path = save(
        replace(synthetic, reference_type="RKS", fock_source="pyscf_rebuilt"),
        tmp_path / "ks.npz",
    )
    code, output = _run(capsys, "validate", str(path))
    assert code == 0
    assert "exchange-correlation" in output


# ------------------------------------------------------------------ inspect

def test_inspect_reports_a_readable_mat(capsys, monkeypatch, fixture_path, tmp_path):
    """The M0 gate in miniature: can extract read this file?"""
    import json as _json

    from g16dump import matfile
    from test_matfile import FakeMatEl

    from g16dump.bundle import load as _load

    matel = FakeMatEl(_load(fixture_path("h2o_rhf")))
    monkeypatch.setattr(matfile, "read_matel", lambda path: matel)

    out = tmp_path / "report.json"
    code, output = _run(capsys, "inspect", "job.mat", "--json", str(out))
    assert code == 0
    assert "AA MO 2E INTEGRALS" in output
    assert "cannot run" not in output
    assert "7 AO, 7 MO, 10 electrons" in output
    assert _json.loads(out.read_text())["eri_nact"] == 6


def test_inspect_exits_nonzero_when_extract_could_not_run(
    capsys, monkeypatch, fixture_path
):
    from g16dump import matfile
    from test_matfile import FakeMatEl

    from g16dump.bundle import load as _load

    matel = FakeMatEl(_load(fixture_path("h2o_rhf")), drop=["OVERLAP"])
    monkeypatch.setattr(matfile, "read_matel", lambda path: matel)

    code, output = _run(capsys, "inspect", "job.mat")
    assert code == 1
    assert "!! overlap" in output
    assert "cannot run" in output


def test_inspect_reports_a_reader_failure_as_a_sentence(capsys, monkeypatch):
    from g16dump import matfile

    def _fail(path):
        raise matfile.MatFileError("gauopen is not importable")

    monkeypatch.setattr(matfile, "read_matel", _fail)
    code, output = _run(capsys, "inspect", "job.mat")
    assert code == 1
    assert "gauopen is not importable" in output


# ------------------------------------------------------- dump and rotate

def test_dump_writes_a_fcidump_where_it_was_told_to(capsys, fixture_path, tmp_path):
    target = tmp_path / "somewhere" / "FCIDUMP"
    code, output = _run(
        capsys, "dump", str(fixture_path("ch2_rohf")), "--out", str(target),
    )
    assert code == 0
    assert str(target) in output
    assert target.read_text().lstrip().startswith("&FCI")


def test_dump_refuses_a_hamiltonian_it_cannot_write(capsys, fixture_path, tmp_path):
    """A WriteError reaches the user as a sentence, not as a traceback."""
    from dataclasses import replace

    import g16dump.cli as cli
    from g16dump.hamiltonian import active_hamiltonian

    bundle = load(fixture_path("ch2_rohf"))
    broken = replace(active_hamiltonian(bundle), ms2=3)

    original = cli.active_hamiltonian
    cli.active_hamiltonian = lambda _bundle: broken
    try:
        code, output = _run(
            capsys, "dump", str(fixture_path("ch2_rohf")),
            "--out", str(tmp_path / "FCIDUMP"),
        )
    finally:
        cli.active_hamiltonian = original

    assert code == 1
    assert "opposite parity" in output


def test_dump_still_runs_the_hamiltonian_gate_first(capsys, fixture_path, tmp_path):
    """A bundle that cannot make a Hamiltonian fails before the writer is reached."""
    code, output = _run(
        capsys, "dump", str(fixture_path("ch2_rohf_roothaan")),
        "--out", str(tmp_path / "FCIDUMP"),
    )
    assert code == 1
    assert "Do not average" in output


def test_rotate_writes_a_rotated_bundle(capsys, fixture_path, tmp_path):
    rotation = tmp_path / "u.npy"
    np.save(rotation, np.eye(12))
    out = tmp_path / "rotated.npz"
    code, output = _run(
        capsys, "rotate", str(fixture_path("ch2_rohf")), "--rotation", str(rotation),
        "--out", str(out),
    )
    assert code == 0
    assert str(out) in output
    assert load(out).nact == 12


def test_rotate_refuses_a_matrix_that_is_not_a_rotation(
    capsys, fixture_path, tmp_path
):
    rotation = tmp_path / "u.npy"
    np.save(rotation, 2.0 * np.eye(12))
    code, output = _run(
        capsys, "rotate", str(fixture_path("ch2_rohf")), "--rotation", str(rotation),
        "--out", str(tmp_path / "rotated.npz"),
    )
    assert code == 1
    assert "not orthogonal" in output


# ------------------------------------------------------------------ extract

def test_extract_requires_the_reference_type(capsys, tmp_path):
    """Never inferred, so the parser refuses rather than defaulting."""
    with pytest.raises(SystemExit) as excinfo:
        main(["extract", "job.mat", "--out", str(tmp_path / "out.npz")])
    assert excinfo.value.code == 2
    assert "--reference" in capsys.readouterr().err


def test_extract_rejects_an_unknown_reference_type(capsys, tmp_path):
    with pytest.raises(SystemExit):
        main([
            "extract", "job.mat", "--out", str(tmp_path / "out.npz"),
            "--reference", "UHF",
        ])
    assert "invalid choice" in capsys.readouterr().err


def test_extract_writes_where_it_is_told(capsys, monkeypatch, tmp_path, fixture_path):
    """No hardcoded paths: the output goes exactly where --out says."""
    from g16dump import matfile

    bundle = load(fixture_path("h2o_rhf"))
    monkeypatch.setattr(matfile, "extract", lambda *a, **k: bundle)

    out = tmp_path / "nested" / "job.npz"
    code, output = _run(
        capsys, "extract", "job.mat", "--out", str(out), "--reference", "RHF"
    )
    assert code == 0
    assert out.exists()
    assert f"wrote {out}" in output
    assert load(out).reference_type == "RHF"


def test_extract_reports_a_reader_failure_as_a_sentence(
    capsys, monkeypatch, tmp_path
):
    from g16dump import matfile

    def _fail(*args, **kwargs):
        raise matfile.MatFileError("no matrix-element block for 'eri_act'")

    monkeypatch.setattr(matfile, "extract", _fail)
    code, output = _run(
        capsys, "extract", "job.mat", "--out", str(tmp_path / "o.npz"),
        "--reference", "RHF",
    )
    assert code == 1
    assert "no matrix-element block" in output


# ------------------------------------------------------------------- parser

def test_no_subcommand_prints_help(capsys):
    code, output = _run(capsys)
    assert code == 1
    assert "extract" in output and "dump" in output


def test_an_unexpected_failure_is_not_a_bare_traceback(capsys, monkeypatch):
    """A numpy error must still reach the user as a sentence."""
    from g16dump import cli

    def _explode(args):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(cli, "_run_validate", _explode)
    code, output = _run(capsys, "validate", "anything.npz")
    assert code == 3
    assert "failed unexpectedly with LinAlgError" in output
    assert "bug in g16dump" in output


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "g16dump" in capsys.readouterr().out


def test_every_subcommand_has_help(capsys):
    for command in ("inspect", "extract", "dump", "validate", "rotate"):
        with pytest.raises(SystemExit):
            main([command, "--help"])
        assert capsys.readouterr().out.strip()
