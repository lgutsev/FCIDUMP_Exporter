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
    assert "valid against schema v1" in output
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
        replace(synthetic, reference="RKS", fock_source="pyscf_rebuilt"),
        tmp_path / "ks.npz",
    )
    code, output = _run(capsys, "validate", str(path))
    assert code == 0
    assert "exchange-correlation" in output


# --------------------------------------------------------- the M3/M4 seams

def test_dump_says_plainly_that_it_is_not_implemented(capsys, fixture_path, tmp_path):
    code, output = _run(
        capsys, "dump", str(fixture_path("ch2_rohf")),
        "--out", str(tmp_path / "FCIDUMP"),
    )
    assert code == 2
    assert "M3" in output
    assert not (tmp_path / "FCIDUMP").exists()


def test_dump_still_runs_the_hamiltonian_gate_first(capsys, fixture_path, tmp_path):
    """A bundle that cannot make a Hamiltonian fails before the writer is reached."""
    code, output = _run(
        capsys, "dump", str(fixture_path("ch2_rohf_roothaan")),
        "--out", str(tmp_path / "FCIDUMP"),
    )
    assert code == 1
    assert "Do not average" in output


def test_rotate_says_plainly_that_it_is_not_implemented(
    capsys, fixture_path, tmp_path
):
    rotation = tmp_path / "u.npy"
    np.save(rotation, np.eye(12))
    code, output = _run(
        capsys, "rotate", str(fixture_path("ch2_rohf")), "--rotation", str(rotation),
        "--out", str(tmp_path / "rotated.npz"),
    )
    assert code == 2
    assert "M4" in output


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
    assert load(out).reference == "RHF"


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


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "g16dump" in capsys.readouterr().out


def test_every_subcommand_has_help(capsys):
    for command in ("extract", "dump", "validate", "rotate"):
        with pytest.raises(SystemExit):
            main([command, "--help"])
        assert capsys.readouterr().out.strip()
