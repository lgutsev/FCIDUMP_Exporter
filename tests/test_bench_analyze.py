"""Tests for the sweep analysis utilities.

The analysis is what a reader of the eventual paper actually sees, so the things
under test are the ones that would quietly mislead: a job whose result never
arrived being dropped instead of reported, a splitting formed between two
different active spaces, and a deviation computed against the wrong neighbour.

Everything here runs on synthetic result records. The one test that needs PySCF
is marked, so the numpy-only CI job deselects it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks import analyze, manifest as mf, sweep  # noqa: E402


# ------------------------------------------------------------------- fixtures

@pytest.fixture
def swept(tmp_path):
    """A real two-state, four-window sweep of Ni porphine."""
    manifests = [
        mf.load(mf.systems_dir() / "ni_porphine_singlet.json"),
        mf.load(mf.systems_dir() / "ni_porphine_triplet.json"),
    ]
    windows = sweep.windows_from_policy(manifests[0], homo_1based=94)
    return sweep.generate(manifests, windows, tmp_path / "sweep", repo_root=REPO)


def result(job, energy, **extra):
    record = {
        "schema_version": "1.0", "kind": "active_space_result", "job": job["name"],
        "system": job["system"], "spin_family": job["spin_family"],
        "multiplicity": job["multiplicity"], "solver": "block2",
        "window": {"nfirst_1based": job["active_space"]["nfirst_1based"],
                   "nlast_1based": job["active_space"]["nlast_1based"]},
        "energy_hartree": energy,
    }
    record.update(extra)
    return record


def energies(swept, singlet, triplet):
    """Attach one energy per job, given the per-window series for each state."""
    series = {1: singlet, 3: triplet}
    records = []
    for job in swept["jobs"]:
        index = [w["nfirst_1based"] for w in swept["windows"]].index(job["active_space"]["nfirst_1based"])
        records.append(result(job, series[job["multiplicity"]][index]))
    return records


# ------------------------------------------------------------- result records

def test_a_good_record_validates(swept):
    assert analyze.validate_result(result(swept["jobs"][0], -2244.1)) == []


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"energy_hartree": "low"}, "must be a number"),
        ({"energy_hartree": float("nan")}, "energy_hartree is nan"),
        ({"runtime_seconds": -5}, "must not be negative"),
        ({"natural_occupations": [1.9, 2.5]}, "outside [0, 2]"),
        ({"natural_occupations": []}, "non-empty list"),
        ({"converged": "yes"}, "converged must be a boolean"),
        ({"schema_version": "0.1"}, "schema_version is"),
        ({"enrgy_hartree": 1.0}, "unknown key"),
    ],
)
def test_validator_catches_bad_records(swept, change, expected):
    record = result(swept["jobs"][0], -2244.1, **change)
    assert any(expected in p for p in analyze.validate_result(record))


def test_a_record_without_an_energy_is_rejected(swept):
    record = result(swept["jobs"][0], -2244.1)
    del record["energy_hartree"]
    assert any("energy_hartree" in p for p in analyze.validate_result(record))


def test_load_results_skips_the_sweep_manifest(tmp_path, swept):
    (tmp_path / "sweep.json").write_text(json.dumps({"kind": "active_space_sweep"}), encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps(result(swept["jobs"][0], -1.0)), encoding="utf-8")
    assert [r["job"] for r in analyze.load_results(tmp_path)] == [swept["jobs"][0]["name"]]


def test_load_results_reports_a_bad_record(tmp_path, swept):
    (tmp_path / "a.json").write_text(json.dumps(result(swept["jobs"][0], "low")), encoding="utf-8")
    with pytest.raises(analyze.AnalysisError, match="must be a number"):
        analyze.load_results(tmp_path)


# ------------------------------------------------------------------ collection

def test_collect_joins_jobs_to_results(swept):
    collected = analyze.collect(swept, energies(swept, [-1, -2, -3, -4], [-1.1, -2.1, -3.1, -4.1]))
    assert len(collected["rows"]) == 8
    assert collected["missing"] == []
    row = next(r for r in collected["rows"] if r["job"].endswith("triplet_w85_100"))
    assert (row["n_orbitals"], row["n_electrons"], row["nalpha"], row["nbeta"]) == (16, 20, 11, 9)


def test_a_job_with_no_result_is_reported_not_dropped(swept):
    records = energies(swept, [-1, -2, -3, -4], [-1.1, -2.1, -3.1, -4.1])[:-1]
    collected = analyze.collect(swept, records)
    assert len(collected["rows"]) == 7
    assert collected["missing"] == [swept["jobs"][-1]["name"]]


def test_strict_mode_fails_on_a_hole(swept):
    records = energies(swept, [-1, -2, -3, -4], [-1.1, -2.1, -3.1, -4.1])[:-1]
    with pytest.raises(analyze.AnalysisError, match="do not match"):
        analyze.collect(swept, records, strict=True)


def test_a_result_for_an_unknown_job_is_reported(swept):
    records = energies(swept, [-1, -2, -3, -4], [-1.1, -2.1, -3.1, -4.1])
    stray = dict(records[0])
    stray["job"] = "some_other_run_w1_2"
    collected = analyze.collect(swept, records + [stray])
    assert collected["orphans"] == ["some_other_run_w1_2"]


def test_duplicate_results_are_an_error(swept):
    records = energies(swept, [-1, -2, -3, -4], [-1.1, -2.1, -3.1, -4.1])
    with pytest.raises(analyze.AnalysisError, match="two result records"):
        analyze.collect(swept, records + [records[0]])


def test_a_result_whose_window_disagrees_is_an_error(swept):
    records = energies(swept, [-1, -2, -3, -4], [-1.1, -2.1, -3.1, -4.1])
    records[0]["window"]["nlast_1based"] += 5
    with pytest.raises(analyze.AnalysisError, match="records window"):
        analyze.collect(swept, records)


# ------------------------------------------------------------------ deviations

def test_deviation_is_measured_against_the_next_larger_window(swept):
    records = energies(swept, [-1.0, -2.0, -2.5, -2.6], [-1.0, -2.0, -2.5, -2.6])
    rows = analyze.add_window_deviations(analyze.collect(swept, records)["rows"])
    singlet = sorted((r for r in rows if r["multiplicity"] == 1), key=lambda r: r["n_orbitals"])
    assert singlet[0]["deviation_from_next_hartree"] == pytest.approx(1.0)
    assert singlet[1]["deviation_from_next_hartree"] == pytest.approx(0.5)
    assert singlet[2]["deviation_from_next_hartree"] == pytest.approx(0.1)


def test_the_largest_window_has_no_deviation(swept):
    """Nothing in the sweep says whether the widest window is converged."""
    records = energies(swept, [-1.0, -2.0, -2.5, -2.6], [-1.0, -2.0, -2.5, -2.6])
    rows = analyze.add_window_deviations(analyze.collect(swept, records)["rows"])
    widest = max(rows, key=lambda r: r["n_orbitals"])
    assert widest["deviation_from_next_hartree"] is None
    assert widest["converged_against_next"] is None


def test_convergence_flag_follows_the_tolerance(swept):
    records = energies(swept, [-1.0, -2.0, -2.5, -2.5001], [-1.0, -2.0, -2.5, -2.5001])
    rows = analyze.add_window_deviations(analyze.collect(swept, records)["rows"], tolerance=1e-3)
    by_orbitals = {r["n_orbitals"]: r for r in rows if r["multiplicity"] == 1}
    assert by_orbitals[24]["converged_against_next"] is True
    assert by_orbitals[16]["converged_against_next"] is False


def test_deviations_do_not_mix_spin_states(swept):
    """A singlet must never be compared with a triplet as if it were a larger window."""
    records = energies(swept, [-1.0, -2.0, -3.0, -4.0], [-10.0, -20.0, -30.0, -40.0])
    rows = analyze.add_window_deviations(analyze.collect(swept, records)["rows"])
    for row in rows:
        if row["next_larger_job"]:
            other = next(r for r in rows if r["job"] == row["next_larger_job"])
            assert other["multiplicity"] == row["multiplicity"]


# ------------------------------------------------------------------ splittings

def test_splitting_is_higher_minus_lower_multiplicity(swept):
    records = energies(swept, [-1.0] * 4, [-0.9] * 4)
    rows = analyze.collect(swept, records)["rows"]
    entry = analyze.spin_splittings(rows)[0]
    assert entry["splitting_hartree"] == pytest.approx(0.1)
    assert entry["ground_state_multiplicity"] == 1
    assert entry["splitting_kcal"] == pytest.approx(0.1 * analyze.HARTREE_TO_KCAL)


def test_a_high_spin_ground_state_gives_a_negative_splitting(swept):
    records = energies(swept, [-1.0] * 4, [-1.2] * 4)
    entry = analyze.spin_splittings(analyze.collect(swept, records)["rows"])[0]
    assert entry["splitting_hartree"] < 0
    assert entry["ground_state_multiplicity"] == 3


def test_splittings_are_only_formed_within_one_window(swept):
    records = energies(swept, [-1.0, -2.0, -3.0, -4.0], [-1.1, -2.1, -3.1, -4.1])
    for entry in analyze.spin_splittings(analyze.collect(swept, records)["rows"]):
        assert entry["splitting_hartree"] == pytest.approx(-0.1)


def test_a_window_with_only_one_spin_state_yields_no_splitting(swept):
    records = [r for r in energies(swept, [-1.0] * 4, [-1.1] * 4) if r["multiplicity"] == 1]
    assert analyze.spin_splittings(analyze.collect(swept, records)["rows"]) == []


def test_an_ordering_change_between_windows_is_flagged(swept):
    """The Ni-PAP result the sweep exists for: state ordering moving with the space."""
    records = energies(swept, [-1.00, -2.00, -3.00, -4.00], [-1.05, -1.90, -2.90, -3.90])
    entries = analyze.spin_splittings(analyze.collect(swept, records)["rows"])
    assert entries[0]["ground_state_multiplicity"] == 3
    assert entries[0]["ground_state_changes_at_next"] is True
    assert entries[1]["ground_state_multiplicity"] == 1
    assert entries[1]["ground_state_changes_at_next"] is False
    assert entries[-1]["ground_state_changes_at_next"] is None


def test_splitting_shift_tracks_how_far_the_number_is_still_moving(swept):
    records = energies(swept, [-1.0, -2.0, -3.0, -4.0], [-0.9, -1.95, -2.98, -3.99])
    entries = analyze.spin_splittings(analyze.collect(swept, records)["rows"])
    shifts = [e["splitting_shift_to_next_kcal"] for e in entries[:-1]]
    assert all(abs(shifts[i]) > abs(shifts[i + 1]) for i in range(len(shifts) - 1))


def test_splitting_uncertainty_combines_both_states(swept):
    records = energies(swept, [-1.0] * 4, [-0.9] * 4)
    for record in records:
        record["energy_uncertainty_hartree"] = 3e-5
    entry = analyze.spin_splittings(analyze.collect(swept, records)["rows"])[0]
    assert entry["splitting_uncertainty_hartree"] == pytest.approx(3e-5 * 2 ** 0.5)


# ---------------------------------------------------------------- occupations

def test_occupations_from_a_density_matrix():
    rdm1 = np.diag([2.0, 1.0, 1.0, 0.0])
    assert analyze.occupations_from_rdm1(rdm1) == pytest.approx([2.0, 1.0, 1.0, 0.0])


def test_occupations_symmetrise_a_slightly_asymmetric_rdm():
    rdm1 = np.diag([2.0, 0.0]) + np.array([[0.0, 1e-6], [0.0, 0.0]])
    assert analyze.occupations_from_rdm1(rdm1) == pytest.approx([2.0, 0.0], abs=1e-9)


def test_a_non_square_rdm_is_rejected():
    with pytest.raises(analyze.AnalysisError, match="must be square"):
        analyze.occupations_from_rdm1(np.zeros((2, 3)))


def test_a_closed_shell_looks_closed_shell():
    diagnostics = analyze.occupation_diagnostics([2.0, 2.0, 0.0, 0.0])
    assert diagnostics["n_fractional"] == 0
    assert diagnostics["n_unpaired"] == pytest.approx(0.0)
    assert diagnostics["n_electrons_from_occupations"] == pytest.approx(4.0)


def test_a_perfect_diradical_has_two_unpaired_electrons():
    diagnostics = analyze.occupation_diagnostics([2.0, 1.0, 1.0, 0.0])
    assert diagnostics["n_fractional"] == 2
    assert diagnostics["n_unpaired"] == pytest.approx(2.0)
    assert diagnostics["n_unpaired_nl"] == pytest.approx(2.0)


def test_weak_correlation_counts_less_in_the_nonlinear_measure():
    weak = analyze.occupation_diagnostics([1.9, 0.1])
    assert weak["n_unpaired"] == pytest.approx(0.2)
    assert weak["n_unpaired_nl"] < weak["n_unpaired"]


def test_occupation_diagnostics_reject_an_empty_list():
    with pytest.raises(analyze.AnalysisError, match="no occupations"):
        analyze.occupation_diagnostics([])


# --------------------------------------------------------------- whole reports

def test_summary_and_report(swept):
    records = energies(swept, [-1.00, -2.00, -3.00, -3.10], [-1.05, -1.90, -2.90, -3.00])
    for record in records:
        record["runtime_seconds"] = 100.0
        record["natural_occupations"] = [2.0, 1.4, 0.6, 0.0]
    summary = analyze.summarize(swept, records)
    text = analyze.format_report(summary)

    assert "ni_porphine_singlet" in text
    assert "ordering changes at the next window" in text
    assert "does not yet support a convergence claim" in text
    assert len(summary["rows"]) == 8
    assert len(summary["splittings"]) == 4


def test_report_names_the_jobs_that_are_missing(swept):
    records = energies(swept, [-1.0] * 4, [-1.1] * 4)[:6]
    text = analyze.format_report(analyze.summarize(swept, records))
    assert "2 job(s) with no result yet" in text
    assert swept["jobs"][-1]["name"] in text


def test_csv_has_a_row_per_job(swept):
    records = energies(swept, [-1.0] * 4, [-1.1] * 4)
    csv = analyze.to_csv(analyze.summarize(swept, records))
    lines = csv.strip().split("\n")
    assert len(lines) == 9
    assert lines[0].startswith("system,spin_family")


def test_cli_analyze(tmp_path, swept):
    results = tmp_path / "results"
    results.mkdir()
    for record in energies(swept, [-1.0, -2.0, -3.0, -3.1], [-1.1, -2.1, -3.1, -3.2]):
        (results / f"{record['job']}.json").write_text(json.dumps(record), encoding="utf-8")

    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "table.csv"
    code = analyze.main([
        str(tmp_path / "sweep"),
        "--results", str(results), "--json", str(summary_path), "--csv", str(csv_path),
    ])
    assert code == 0
    assert json.loads(summary_path.read_text())["rows"]
    assert csv_path.read_text().count("\n") == 9


# ------------------------------------------------------- bundle cross-checking

def bundle_for(man, **overrides):
    counts = mf.active_space(man, repo_root=REPO)
    bundle = {
        "act_start": np.array(counts["act_start"]), "act_stop": np.array(counts["act_stop"]),
        "ncore": np.array(counts["ncore"]), "nact": np.array(counts["nact"]),
        "nelec": np.array(counts["nelec"]), "nalpha": np.array(counts["nalpha"]),
        "nbeta": np.array(counts["nbeta"]),
        "charge": np.array(man["molecule"]["charge"]),
        "multiplicity": np.array(man["molecule"]["multiplicity"]),
        "reference": np.array(man["reference"]["type"]),
        "fock_source": np.array(man["reference"]["fock_source"]),
        "eri_act": np.zeros((counts["nact"],) * 4),
    }
    bundle.update(overrides)
    return bundle


def test_a_matching_bundle_passes():
    man = mf.load(mf.systems_dir() / "h2o_rhf_sto3g.json")
    assert analyze.check_bundle_against_manifest(bundle_for(man), man, repo_root=REPO) == []


def test_a_bundle_with_the_wrong_window_is_caught():
    man = mf.load(mf.systems_dir() / "h2o_rhf_sto3g.json")
    bundle = bundle_for(man, act_stop=np.array(6), nact=np.array(5))
    problems = analyze.check_bundle_against_manifest(bundle, man, repo_root=REPO)
    assert any("act_stop" in p for p in problems)
    assert any("nact" in p for p in problems)


def test_a_bundle_whose_eri_does_not_match_its_window_is_caught():
    man = mf.load(mf.systems_dir() / "h2o_rhf_sto3g.json")
    bundle = bundle_for(man, eri_act=np.zeros((5, 5, 5, 5)))
    problems = analyze.check_bundle_against_manifest(bundle, man, repo_root=REPO)
    assert any("eri_act has shape" in p for p in problems)


def test_a_ks_bundle_that_used_the_stored_fock_matrix_is_caught():
    man = mf.load(mf.systems_dir() / "h2o_rks_b3lyp_631g.json")
    bundle = bundle_for(man, fock_source=np.array("gaussian"))
    problems = analyze.check_bundle_against_manifest(bundle, man, repo_root=REPO)
    assert any("not the HF Fock operator" in p for p in problems)


def test_a_bundle_missing_a_key_is_caught():
    man = mf.load(mf.systems_dir() / "h2o_rhf_sto3g.json")
    bundle = bundle_for(man)
    del bundle["nalpha"]
    problems = analyze.check_bundle_against_manifest(bundle, man, repo_root=REPO)
    assert any("no 'nalpha'" in p for p in problems)


# ---------------------------------------------------------------- pyscf oracle

@pytest.mark.pyscf
def test_fci_energy_from_fcidump_reproduces_pyscf(tmp_path):
    """Round-trip: an FCIDUMP this helper writes back the energy PySCF gets directly."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import fci, gto, scf
    from pyscf.tools import fcidump

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22", basis="sto-3g", verbose=0)
    mean_field = scf.RHF(mol).run()
    reference = fci.FCI(mean_field).kernel()[0]

    path = tmp_path / "FCIDUMP"
    fcidump.from_scf(mean_field, str(path))
    assert analyze.fci_energy_from_fcidump(path) == pytest.approx(reference, abs=1e-9)
