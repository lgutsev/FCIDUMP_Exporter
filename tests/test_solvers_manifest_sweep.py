"""Solver adapters, benchmark manifests and active-space sweeps.

NumPy only. The solvers are never executed here -- generated inputs are compared
against golden fixtures, which is enough to catch a format regression, and the
output parsers are exercised against sample text.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from g16dump.bundle import load
from g16dump.errors import ValidationError
from g16dump.hamiltonian import active_space_hamiltonian
from g16dump.manifest import (
    MANIFEST_VERSION,
    count_electrons,
    gaussian_input,
    load_manifest,
    save_manifest,
    summary_table,
    validate_manifest,
    validate_system,
)
from g16dump.solvers import (
    block2_input,
    dice_input,
    parse_block2_output,
    parse_dice_output,
    pyscf_fci_energy,
    write_block2_input,
    write_dice_input,
)
from g16dump.sweep import (
    collect_sweep,
    format_sweep_table,
    natural_orbital_diagnostics,
    spin_splittings,
    spin_state_sweep_manifest,
    sweep_manifest,
    windows_around,
)

DATA = Path(__file__).resolve().parent / "data"
GOLDEN = DATA / "golden"
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ash():
    return active_space_hamiltonian(load(str(DATA / "ch2_rohf_631g.npz")), tol_scf=None)


# ------------------------------------------------------------ solver inputs


def test_dice_input_matches_the_golden_fixture(ash):
    assert dice_input(ash) == (GOLDEN / "ch2_rohf_dice_input.dat").read_text()


def test_block2_input_matches_the_golden_fixture(ash):
    assert block2_input(ash) == (GOLDEN / "ch2_rohf_block2_input.txt").read_text()


def test_dice_input_reference_determinant_follows_the_hamiltonian(ash):
    """The nocc block is derived, so it cannot drift out of step with the dump."""
    text = dice_input(ash)
    lines = text.splitlines()
    assert lines[0] == f"nocc {ash.nelec_act}"
    assert len(lines[1].split()) == ash.nelec_act
    alpha = [2 * t for t in range(ash.nocc_a_act)]
    beta = [2 * t + 1 for t in range(ash.nocc_b_act)]
    assert [int(x) for x in lines[1].split()] == alpha + beta


def test_block2_spin_comes_from_the_hamiltonian(ash):
    assert f"spin {ash.ms2}" in block2_input(ash)
    assert f"nelec {ash.nelec_act}" in block2_input(ash)


def test_solver_inputs_can_be_written(ash, tmp_path):
    write_dice_input(str(tmp_path / "input.dat"), ash)
    write_block2_input(str(tmp_path / "dmrg.conf"), ash)
    assert (tmp_path / "input.dat").read_text() == dice_input(ash)
    assert (tmp_path / "dmrg.conf").read_text() == block2_input(ash)


def test_custom_schedules_appear_in_the_input(ash):
    text = dice_input(ash, schedule=[(0, 1e-3), (5, 1e-5)], maxiter=20)
    assert "0 0.001" in text and "5 1e-05" in text and "maxiter 20" in text

    text = block2_input(ash, schedule=[(0, 100, 1e-4, 1e-3)], maxiter=5)
    assert "0 100 0.0001 0.001" in text and "maxiter 5" in text


def test_empty_schedules_are_refused(ash):
    with pytest.raises(ValidationError, match="schedule is empty"):
        dice_input(ash, schedule=[])
    with pytest.raises(ValidationError, match="schedule is empty"):
        block2_input(ash, schedule=[])


# ---------------------------------------------------------- output parsing


def test_parse_dice_output_extracts_energies():
    sample = """
    Performing variational calculation
    0    9.00e-04    1024    -38.912345678    0.00
    3    4.00e-04    8192    -38.931234567    0.00
    Variational Energy   -38.931234567
    Semistochastic Perturbation Theory
    PT Energy  -38.940123456
    Total Energy  -38.940123456
    Calculation converged
    """
    parsed = parse_dice_output(sample)
    assert len(parsed["iterations"]) == 2
    assert parsed["iterations"][1][1] == 8192
    assert parsed["variational_energy"] == pytest.approx(-38.931234567)
    assert parsed["pt_energy"] == pytest.approx(-38.940123456)
    assert parsed["converged"] is True


def test_parse_block2_output_extracts_sweeps():
    sample = """
    Sweep =    0 | Direction = forward | M = 250 | Energy = -38.9100000000
    Sweep =    4 | Direction = backward | M = 500 | Energy = -38.9310000000
    DMRG Energy = -38.9312345678
    Convergence: converged
    """
    parsed = parse_block2_output(sample)
    assert len(parsed["sweeps"]) == 2
    assert parsed["sweeps"][0][1] == 250
    assert parsed["energy"] == pytest.approx(-38.9312345678)
    assert parsed["converged"] is True


def test_parsers_return_empty_results_rather_than_guessing():
    """Nothing recognizable in, nothing invented out."""
    assert parse_dice_output("no energies here")["iterations"] == []
    assert parse_block2_output("nothing at all")["sweeps"] == []
    assert "energy" not in parse_block2_output("nothing at all")


# --------------------------------------------------------- PySCF as a solver


def test_pyscf_fci_solver_refuses_an_oversized_space(ash):
    with pytest.raises(ValidationError, match="factorially"):
        pyscf_fci_energy(ash, max_orbitals=2)


def test_pyscf_fci_solver_matches_the_written_dump(ash, tmp_path):
    pytest.importorskip("pyscf")
    from pyscf import ao2mo, fci
    from pyscf.tools import fcidump as pyscf_fcidump

    from g16dump.write import write_from_hamiltonian

    path = tmp_path / "FCIDUMP"
    write_from_hamiltonian(str(path), ash)
    data = pyscf_fcidump.read(str(path))
    norb = int(data["NORB"])
    through_file = fci.direct_spin1.kernel(
        np.asarray(data["H1"]),
        ao2mo.restore(1, np.asarray(data["H2"]), norb),
        norb,
        (ash.nocc_a_act, ash.nocc_b_act),
        verbose=0,
    )[0] + float(data["ECORE"])

    assert abs(pyscf_fci_energy(ash) - through_file) < 1e-9


# ---------------------------------------------------------------- manifests


def test_the_committed_manifest_is_valid():
    manifest = load_manifest(str(REPO / "benchmarks" / "systems.json"))
    derived = validate_manifest(manifest)
    assert len(derived) == len(manifest["systems"])


def test_the_manifest_covers_more_than_nickel():
    """The workflow must be demonstrably not Ni-specific."""
    manifest = load_manifest(str(REPO / "benchmarks" / "systems.json"))
    compositions = " ".join(
        str(s.get("composition", "")) + s["name"] for s in manifest["systems"]
    )
    assert "Fe" in compositions or "fe_" in compositions
    assert "Co" in compositions or "co_" in compositions

    tiers = {s.get("tier") for s in manifest["systems"]}
    assert {"validation", "ligand_field", "porphyrin", "regression"} <= tiers


def test_electron_counting():
    assert count_electrons("O 0 0 0\nH 0 0 1\nH 0 1 0", 0) == 10
    assert count_electrons("Ni 0 0 0", 2) == 26
    assert count_electrons("Fe 0 0 0\nN 0 0 1", 0) == 33


def test_unknown_element_is_reported():
    with pytest.raises(ValidationError, match="unknown element"):
        count_electrons("Xx 0 0 0", 0)


def test_impossible_multiplicity_is_refused():
    system = {
        "name": "bad", "geometry": "O 0 0 0\nH 0 0 1\nH 0 1 0", "charge": 0,
        "multiplicity": 2, "reference": "ROHF", "basis": "6-31G", "window": [2, 7],
    }
    with pytest.raises(ValidationError, match="is impossible"):
        validate_system(system)


def test_window_too_small_for_the_reference_is_refused():
    system = {
        "name": "bad", "geometry": "O 0 0 0\nH 0 0 1\nH 0 1 0", "charge": 0,
        "multiplicity": 1, "reference": "RHF", "basis": "6-31G", "window": [2, 3],
    }
    with pytest.raises(ValidationError, match="occupied alpha orbitals inside it"):
        validate_system(system)


def test_kohn_sham_entry_must_name_a_functional():
    system = {
        "name": "bad", "geometry": "O 0 0 0\nH 0 0 1\nH 0 1 0", "charge": 0,
        "multiplicity": 1, "reference": "RKS", "basis": "6-31G", "window": [2, 7],
    }
    with pytest.raises(ValidationError, match="names no.*functional"):
        validate_system(system)
    system["xc"] = "b3lyp"
    validate_system(system)


def test_duplicate_names_are_refused():
    system = {
        "name": "same", "geometry": "O 0 0 0\nH 0 0 1\nH 0 1 0", "charge": 0,
        "multiplicity": 1, "reference": "RHF", "basis": "6-31G", "window": [2, 7],
    }
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "systems": [dict(system), dict(system)],
    }
    with pytest.raises(ValidationError, match="duplicate system names"):
        validate_manifest(manifest)


def test_placeholder_geometry_defers_rather_than_skips():
    system = {
        "name": "future", "geometry": "PLACEHOLDER", "geometry_status": "placeholder",
        "charge": 0, "multiplicity": 3, "reference": "ROHF", "basis": "def2-SVP",
        "window": [80, 109],
    }
    derived = validate_system(system)
    assert derived["nact"] == 30 and derived["ncore"] == 79
    assert derived["nelec"] is None
    assert "validation_deferred" in derived


def test_gaussian_input_generation(tmp_path):
    manifest = load_manifest(str(REPO / "benchmarks" / "systems.json"))
    system = next(s for s in manifest["systems"] if s["name"] == "ch2_triplet")
    text = gaussian_input(system)

    assert "%chk=ch2_triplet.chk" in text
    assert "ROHF/6-31G" in text
    assert "NoSymm" in text and "Int=NoBasisTransform" in text
    assert "5D 7F" in text
    assert "Output=MatrixElement" in text and "Tran=Full" in text
    assert "Window=(2,9)" in text
    assert "0 3" in text
    assert text.rstrip().endswith("ch2_triplet.mat")


def test_manifest_round_trip(tmp_path):
    manifest = load_manifest(str(REPO / "benchmarks" / "systems.json"))
    path = tmp_path / "copy.json"
    save_manifest(manifest, str(path))
    assert load_manifest(str(path)) == manifest


def test_summary_table_renders():
    manifest = load_manifest(str(REPO / "benchmarks" / "systems.json"))
    text = summary_table(manifest)
    assert "ch2_triplet" in text and "nact" in text


# ------------------------------------------------------------------- sweeps


def test_windows_around_centres_on_the_reference():
    windows = windows_around(10, [4, 8, 12])
    assert windows == [[9, 12], [7, 14], [5, 16]]
    for first, last in windows:
        assert first <= 10 <= last


def test_windows_that_would_start_below_orbital_one_are_refused():
    with pytest.raises(ValidationError, match="below the first orbital"):
        windows_around(2, [20])


def test_sweep_manifest_validates_every_window():
    base = {
        "name": "ch2", "geometry": "C 0 0 0\nH 0.99 0 0.42\nH -0.99 0 0.42",
        "charge": 0, "multiplicity": 3, "reference": "ROHF", "basis": "6-31G",
        "window": [2, 9],
    }
    manifest = sweep_manifest(base, windows_around(5, [4, 6, 8]))
    assert len(manifest["systems"]) == 3
    assert [s["name"] for s in manifest["systems"]] == [
        "ch2_w04", "ch2_w06", "ch2_w08"
    ]
    validate_manifest(manifest)


def test_spin_state_sweep_builds_the_cross_product():
    base = {
        "name": "ch2", "geometry": "C 0 0 0\nH 0.99 0 0.42\nH -0.99 0 0.42",
        "charge": 0, "multiplicity": 3, "reference": "ROHF", "basis": "6-31G",
        "window": [2, 9],
    }
    manifest = spin_state_sweep_manifest(base, windows_around(5, [4, 6]), [1, 3])
    assert len(manifest["systems"]) == 4
    assert {s["multiplicity"] for s in manifest["systems"]} == {1, 3}
    # The reference type must follow the multiplicity.
    for system in manifest["systems"]:
        expected = "RHF" if system["multiplicity"] == 1 else "ROHF"
        assert system["reference"] == expected


def test_spin_state_sweep_needs_at_least_two_states():
    base = {
        "name": "ch2", "geometry": "C 0 0 0\nH 0.99 0 0.42\nH -0.99 0 0.42",
        "charge": 0, "multiplicity": 3, "reference": "ROHF", "basis": "6-31G",
        "window": [2, 9],
    }
    with pytest.raises(ValidationError, match="at least two multiplicities"):
        spin_state_sweep_manifest(base, [[2, 9]], [3])


# ----------------------------------------------------------------- analysis


def test_collect_sweep_reports_deviation_from_the_next_larger_space():
    results = [
        {"name": "w6", "nact": 6, "nelec_act": 6, "energy": -38.90},
        {"name": "w10", "nact": 10, "nelec_act": 10, "energy": -38.95},
        {"name": "w8", "nact": 8, "nelec_act": 8, "energy": -38.93},
    ]
    table = collect_sweep(results)
    assert [row["nact"] for row in table] == [6, 8, 10]
    assert table[0]["deviation"] == pytest.approx(-0.03)
    assert table[1]["deviation"] == pytest.approx(-0.02)
    # The largest space has nothing to compare against; that is stated, not zero.
    assert table[2]["deviation"] is None


def test_collect_sweep_requires_the_fields_it_needs():
    with pytest.raises(ValidationError, match="missing"):
        collect_sweep([{"name": "x", "nact": 4}])


def test_natural_orbital_diagnostics():
    closed = natural_orbital_diagnostics([2.0, 2.0, 0.0, 0.0])
    assert closed["n_effectively_unpaired"] == pytest.approx(0.0)
    assert closed["n_partially_occupied"] == 0
    assert closed["n_electrons"] == pytest.approx(4.0)

    diradical = natural_orbital_diagnostics([2.0, 1.0, 1.0, 0.0])
    assert diradical["n_effectively_unpaired"] == pytest.approx(2.0)
    assert diradical["n_partially_occupied"] == 2


def test_spin_splittings_track_convergence():
    results = [
        {"nact": 6, "multiplicity": 1, "nelec_act": 6, "energy": -38.90},
        {"nact": 6, "multiplicity": 3, "nelec_act": 6, "energy": -38.94},
        {"nact": 10, "multiplicity": 1, "nelec_act": 10, "energy": -38.96},
        {"nact": 10, "multiplicity": 3, "nelec_act": 10, "energy": -38.99},
    ]
    rows = spin_splittings(results)
    assert [row["nact"] for row in rows] == [6, 10]
    assert rows[0]["ground_multiplicity"] == 3
    assert rows[0]["splittings"][1] == pytest.approx(0.04 * 27.211386245988, rel=1e-9)
    # The splitting shrank from 0.04 to 0.03 Ha as the space grew: not converged.
    assert rows[0]["splitting_change"][1] < 0
    assert rows[1]["splitting_change"] is None


def test_spin_splittings_need_a_multiplicity():
    with pytest.raises(ValidationError, match="'multiplicity' on every result"):
        spin_splittings([{"nact": 6, "nelec_act": 6, "energy": -1.0}])


def test_sweep_table_renders_and_marks_the_largest_space():
    table = collect_sweep([
        {"nact": 6, "nelec_act": 6, "energy": -38.90, "occupations": [2.0, 1.0, 1.0, 0.0, 0.0, 0.0]},
        {"nact": 8, "nelec_act": 8, "energy": -38.93},
    ])
    text = format_sweep_table(table)
    assert "(largest)" in text
    assert "nact" in text
