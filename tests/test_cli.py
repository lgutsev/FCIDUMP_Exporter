"""End-to-end CLI tests, driven off the committed ``.npz`` fixtures.

These run with NumPy alone: the fixtures in ``tests/data/`` were produced from
PySCF once and committed, so neither Gaussian, gauopen nor PySCF is needed to
exercise the whole pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fcidump_reader import determinant_energy, read_fcidump
from g16dump.bundle import load
from g16dump.cli import main
from g16dump.hamiltonian import load_hamiltonian
from g16dump.rotate import random_orthogonal

DATA = Path(__file__).resolve().parent / "data"
FIXTURES = sorted(p.name for p in DATA.glob("*.npz"))


def test_fixtures_are_present():
    """A missing fixture would silently skip most of this file."""
    assert FIXTURES, f"no .npz fixtures in {DATA}"
    assert "ch2_rohf_631g.npz" in FIXTURES


@pytest.fixture(params=FIXTURES)
def fixture_path(request):
    return str(DATA / request.param)


# -------------------------------------------------------------------- validate


def test_validate_accepts_every_fixture(fixture_path, capsys):
    assert main(["validate", fixture_path, "--check-hamiltonian"]) == 0
    out = capsys.readouterr().out
    assert "valid" in out
    assert "E_core" in out


def test_validate_reports_residuals(capsys):
    main(["validate", str(DATA / "ch2_rohf_631g.npz"), "--check-hamiltonian"])
    out = capsys.readouterr().out
    assert "c_a_orthonormality" in out
    assert "eri_symmetry" in out
    assert "spin_error" in out


def test_validate_warns_about_kohn_sham(capsys):
    main(["validate", str(DATA / "h2o_rks_631g.npz")])
    out = capsys.readouterr().out
    assert "exchange-correlation" in out


def test_validate_reports_a_bad_bundle_without_a_traceback(tmp_path, capsys):
    broken = tmp_path / "broken.npz"
    np.savez(broken, nothing=np.zeros(2))
    assert main(["validate", str(broken)]) == 1
    assert "error:" in capsys.readouterr().err


# ------------------------------------------------------------------------ dump


def test_dump_writes_a_readable_fcidump(fixture_path, tmp_path):
    out = tmp_path / "FCIDUMP"
    assert main(["dump", fixture_path, "--out", str(out)]) == 0

    bundle = load(fixture_path)
    parsed = read_fcidump(str(out))
    assert parsed.norb == bundle.nact
    assert parsed.nelec == bundle.nelec_act
    assert parsed.ms2 == bundle.ms2


def test_dump_reproduces_the_scf_energy_for_hf_fixtures(fixture_path, tmp_path):
    """E_core plus the reference determinant energy must rebuild E_scf.

    Skipped for Kohn-Sham fixtures, which deliberately carry no e_scf: the KS
    total energy is not the HF reference determinant energy.
    """
    bundle = load(fixture_path)
    if bundle.e_scf is None:
        pytest.skip("KS fixture carries no SCF reference energy, by design")

    out = tmp_path / "FCIDUMP"
    main(["dump", fixture_path, "--out", str(out)])
    parsed = read_fcidump(str(out))

    energy = determinant_energy(
        parsed.h1, parsed.eri, parsed.e_core,
        bundle.nocc_a_act, bundle.nocc_b_act,
    )
    assert abs(energy - bundle.e_scf) < 1e-9


def test_dump_writes_the_dice_nocc_block(tmp_path):
    out, dice = tmp_path / "FCIDUMP", tmp_path / "input.dat"
    main([
        "dump", str(DATA / "ch2_rohf_631g.npz"),
        "--out", str(out), "--dice-input", str(dice),
    ])
    lines = dice.read_text().splitlines()
    bundle = load(str(DATA / "ch2_rohf_631g.npz"))
    assert lines[0] == f"nocc {bundle.nelec_act}"
    assert len(lines[1].split()) == bundle.nelec_act
    assert lines[2] == "end"


def test_dump_threshold_reduces_the_line_count(tmp_path):
    dense, sparse = tmp_path / "a", tmp_path / "b"
    src = str(DATA / "ch2_rohf_631g.npz")
    main(["dump", src, "--out", str(dense)])
    main(["dump", src, "--out", str(sparse), "--threshold", "1e-4"])
    assert len(sparse.read_text().splitlines()) < len(dense.read_text().splitlines())


def test_dump_accepts_custom_orbsym(tmp_path):
    out = tmp_path / "FCIDUMP"
    main([
        "dump", str(DATA / "ch2_rohf_631g.npz"),
        "--out", str(out), "--orbsym", "1,2,3,4,1,2,3,4", "--isym", "2",
    ])
    parsed = read_fcidump(str(out))
    assert parsed.orbsym == [1, 2, 3, 4, 1, 2, 3, 4]
    assert parsed.isym == 2


def test_dump_can_save_the_reduced_hamiltonian(tmp_path):
    out, ham = tmp_path / "FCIDUMP", tmp_path / "ham.npz"
    main([
        "dump", str(DATA / "ch2_rohf_631g.npz"),
        "--out", str(out), "--save-hamiltonian", str(ham),
    ])
    ash, provenance = load_hamiltonian(str(ham))
    parsed = read_fcidump(str(out))
    assert np.max(np.abs(ash.h_eff - parsed.h1)) < 1e-12
    assert provenance["command"] == "dump"


# ---------------------------------------------------------------------- rotate


def test_rotate_preserves_the_core_energy_and_dumps(tmp_path):
    src = str(DATA / "ch2_rohf_631g.npz")
    bundle = load(src)
    rotation = tmp_path / "U.npy"
    np.save(rotation, random_orthogonal(bundle.nact, np.random.default_rng(0)))

    rotated = tmp_path / "rot.npz"
    assert main(["rotate", src, "--rotation", str(rotation), "--out", str(rotated)]) == 0

    plain, turned = tmp_path / "A", tmp_path / "B"
    main(["dump", src, "--out", str(plain)])
    main(["dump", str(rotated), "--out", str(turned)])

    a, b = read_fcidump(str(plain)), read_fcidump(str(turned))
    assert abs(a.e_core - b.e_core) < 1e-12
    assert a.norb == b.norb and a.nelec == b.nelec and a.ms2 == b.ms2
    # The Hamiltonian really did change basis.
    assert np.max(np.abs(a.h1 - b.h1)) > 1e-3


def test_rotate_records_why_e_ref_no_longer_applies(tmp_path):
    src = str(DATA / "ch2_rohf_631g.npz")
    bundle = load(src)
    rotation = tmp_path / "U.npy"
    np.save(rotation, random_orthogonal(bundle.nact, np.random.default_rng(1)))
    rotated = tmp_path / "rot.npz"
    main(["rotate", src, "--rotation", str(rotation), "--out", str(rotated)])

    _, provenance = load_hamiltonian(str(rotated))
    assert "rotation_note" in provenance
    assert "E_core is unchanged" in provenance["rotation_note"]
    assert provenance["rotation_orthogonality_residual"] < 1e-10


def test_rotate_refuses_a_non_orthogonal_matrix(tmp_path, capsys):
    src = str(DATA / "ch2_rohf_631g.npz")
    bundle = load(src)
    rotation = tmp_path / "U.npy"
    bad = np.eye(bundle.nact)
    bad[0, 1] = 0.4
    np.save(rotation, bad)

    assert main(["rotate", src, "--rotation", str(rotation),
                 "--out", str(tmp_path / "out.npz")]) == 1
    assert "not orthogonal" in capsys.readouterr().err


def test_rotate_refuses_a_wrongly_sized_matrix(tmp_path, capsys):
    src = str(DATA / "ch2_rohf_631g.npz")
    rotation = tmp_path / "U.npy"
    np.save(rotation, np.eye(3))
    assert main(["rotate", src, "--rotation", str(rotation),
                 "--out", str(tmp_path / "out.npz")]) == 1
    assert "active space" in capsys.readouterr().err


# ------------------------------------------------------------------------ info


def test_info_prints_provenance(capsys):
    assert main(["info", str(DATA / "ch2_rohf_631g.npz")]) == 0
    out = capsys.readouterr().out
    assert "provenance" in out
    payload = json.loads(out[out.index("{"):])
    assert payload["method"] == "ROHF"
    assert "active_window" in payload


def test_info_handles_a_rotated_hamiltonian(tmp_path, capsys):
    src = str(DATA / "ch2_rohf_631g.npz")
    bundle = load(src)
    rotation = tmp_path / "U.npy"
    np.save(rotation, random_orthogonal(bundle.nact, np.random.default_rng(2)))
    rotated = tmp_path / "rot.npz"
    main(["rotate", src, "--rotation", str(rotation), "--out", str(rotated)])

    capsys.readouterr()
    assert main(["info", str(rotated)]) == 0
    out = capsys.readouterr().out
    assert "reduced active-space Hamiltonian" in out
    assert "NORB=" in out


# ------------------------------------------------------- reference-type policy


def test_extract_refuses_kohn_sham_without_rebuild_fock(capsys):
    """The KS guard fires before anything tries to open the file."""
    assert main([
        "extract", "nonexistent.mat", "--out", "out.npz", "--ref-type", "RKS",
    ]) == 2
    err = capsys.readouterr().err
    assert "--rebuild-fock is required" in err
    assert "exchange-correlation" in err


def test_extract_requires_an_explicit_reference_type(capsys):
    """--ref-type has no default: a .mat cannot distinguish HF from KS."""
    with pytest.raises(SystemExit):
        main(["extract", "job.mat", "--out", "out.npz"])
    assert "--ref-type" in capsys.readouterr().err


def test_fixtures_carry_provenance(fixture_path):
    bundle = load(fixture_path)
    assert bundle.provenance["fixture"]
    assert "active_window" in bundle.provenance
    assert bundle.fock_source in ("gaussian", "rebuilt-pyscf", "none")
