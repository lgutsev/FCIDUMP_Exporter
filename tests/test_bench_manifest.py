"""Tests for the benchmark manifest format.

Two things are being protected here. The first is the validator itself: a
manifest suite is only worth having if a wrong manifest is rejected, so every
failure mode gets a test that starts from a manifest known to be good and breaks
exactly one thing. The second is the committed suite in ``benchmarks/systems/``,
which must validate as a whole -- that is the check that catches an entry edited
into an inconsistent state.

No Gaussian, no gauopen and no solver: manifests are data.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks import manifest  # noqa: E402


# ------------------------------------------------------------- the real suite

def _suite():
    return manifest.load_all(manifest.systems_dir())


def test_suite_is_not_empty():
    assert len(_suite()) >= 10


@pytest.mark.parametrize("man", _suite(), ids=lambda m: m["id"])
def test_committed_manifest_validates(man):
    problems = manifest.validate(man, repo_root=REPO)
    assert problems == [], "\n".join(problems)


def test_suite_ids_are_unique():
    ids = [m["id"] for m in _suite()]
    assert len(ids) == len(set(ids))


def test_suite_covers_the_required_progression():
    """The suite has to span the whole ladder, not just the easy end."""
    roles = {m["role"] for m in _suite()}
    assert {"validation", "ligand_field", "macrocycle", "regression", "stress"} <= roles


def test_suite_has_a_non_nickel_transition_metal():
    """The workflow is meant to be demonstrably not Ni-specific."""
    metals = set()
    for man in _suite():
        if man.get("status") == "blocked":
            continue
        for atom in manifest.geometry_atoms(man, REPO):
            if 21 <= manifest.ELEMENTS[atom[0]] <= 30:
                metals.add(atom[0])
    assert metals - {"Ni"}, f"only nickel found: {metals}"


def test_every_blocked_entry_says_what_it_is_blocked_on():
    for man in _suite():
        if man["status"] == "blocked":
            assert man.get("blocked_on"), man["id"]


def test_ready_entries_resolve_their_geometry():
    for man in _suite():
        if man["status"] == "ready":
            assert manifest.geometry_atoms(man, REPO), man["id"]


def test_spin_families_share_a_geometry_and_basis():
    """A splitting is only meaningful if the two states differ in spin alone."""
    families: dict = {}
    for man in _suite():
        families.setdefault(man["spin_family"], []).append(man)
    for family, members in families.items():
        if len(members) < 2:
            continue
        bases = {json.dumps(m["reference"]["basis"], sort_keys=True) for m in members}
        assert len(bases) == 1, f"{family} mixes bases: {bases}"
        if all(m["status"] == "ready" for m in members):
            geometries = {
                tuple(tuple(round(c, 8) if isinstance(c, float) else c for c in atom)
                      for atom in manifest.geometry_atoms(m, REPO))
                for m in members
            }
            assert len(geometries) == 1, f"{family} mixes geometries"


def test_ks_entries_never_use_the_stored_fock_matrix():
    for man in _suite():
        if man["reference"]["type"] in manifest.KS_REFERENCE_TYPES:
            assert man["reference"]["fock_source"] == "pyscf_rebuilt", man["id"]


# --------------------------------------------------------------- window maths

def test_bundle_window_is_zero_based_half_open():
    """Gaussian Window=(2,7) is bundle act_start=1, act_stop=7, six orbitals."""
    act_start, act_stop = manifest.bundle_window(2, 7)
    assert (act_start, act_stop) == (1, 7)
    assert act_stop - act_start == 6


@pytest.mark.parametrize("nfirst, nlast", [(0, 7), (-1, 3), (5, 4)])
def test_bundle_window_rejects_impossible_windows(nfirst, nlast):
    with pytest.raises(manifest.ManifestError):
        manifest.bundle_window(nfirst, nlast)


def test_spin_counts():
    assert manifest.spin_counts(10, 1) == (5, 5)
    assert manifest.spin_counts(8, 3) == (5, 3)
    assert manifest.spin_counts(96, 5) == (50, 46)


def test_spin_counts_rejects_impossible_multiplicity():
    with pytest.raises(manifest.ManifestError):
        manifest.spin_counts(10, 2)  # 10 electrons cannot be a doublet


def test_n_determinants():
    assert manifest.n_determinants(6, 4, 4) == 225
    assert manifest.n_determinants(4, 5, 1) == 0  # more electrons than orbitals


# ------------------------------------------------------------ window policies

BASE_POLICY = {
    "schema_version": "1.0", "id": "policy_case", "status": "ready", "role": "validation",
    "molecule": {"charge": 0, "multiplicity": 1, "geometry": {"units": "angstrom", "xyz": "H 0 0 0\nH 0 0 0.74"}},
    "reference": {"type": "RHF", "basis": "STO-3G", "fock_source": "gaussian"},
    "active_space": {"window_policy": {"reference_orbital": "homo", "n_below": 2, "n_above": 3}},
    "provenance": {"added": "2026-09-18", "geometry_source": "test fixture"},
}


def test_policy_resolves_against_a_known_homo():
    assert manifest.resolve_window(BASE_POLICY, homo_1based=10) == (8, 13)


def test_policy_on_lumo_shifts_by_one():
    man = copy.deepcopy(BASE_POLICY)
    man["active_space"]["window_policy"]["reference_orbital"] = "lumo"
    assert manifest.resolve_window(man, homo_1based=10) == (9, 14)


def test_policy_with_an_absolute_index_needs_no_scf():
    man = copy.deepcopy(BASE_POLICY)
    man["active_space"]["window_policy"]["reference_orbital"] = 20
    assert manifest.resolve_window(man) == (18, 23)


def test_policy_refuses_to_guess_the_homo():
    with pytest.raises(manifest.ManifestError, match="homo_1based"):
        manifest.resolve_window(BASE_POLICY)


def test_policy_that_runs_off_the_bottom_is_rejected():
    with pytest.raises(manifest.ManifestError, match="below orbital 1"):
        manifest.resolve_window(BASE_POLICY, homo_1based=2)


def test_policy_that_runs_off_the_top_is_rejected():
    with pytest.raises(manifest.ManifestError, match="above the"):
        manifest.resolve_window(BASE_POLICY, homo_1based=10, nbasis=12)


# ----------------------------------------------------------------- geometries

def test_parse_xyz_accepts_a_real_xyz_file():
    atoms = manifest.parse_xyz("2\nsome comment\nH 0 0 0\nH 0 0 0.74\n")
    assert atoms == [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)]


def test_parse_xyz_rejects_a_wrong_atom_count():
    with pytest.raises(manifest.ManifestError, match="3 atoms"):
        manifest.parse_xyz("3\nc\nH 0 0 0\nH 0 0 0.74\n")


def test_parse_xyz_rejects_a_non_numeric_coordinate():
    with pytest.raises(manifest.ManifestError, match="non-numeric"):
        manifest.parse_xyz("H 0 0 x\n")


def test_electron_count_comes_from_the_geometry():
    man = copy.deepcopy(BASE_POLICY)
    man["molecule"]["geometry"]["xyz"] = "O 0 0 0\nH 0 0.757 0.587\nH 0 -0.757 0.587"
    assert manifest.electron_count(man) == 10
    man["molecule"]["charge"] = -2
    assert manifest.electron_count(man) == 12


def test_ni_porphine_geometry_is_the_right_molecule():
    atoms = manifest.parse_xyz((REPO / "benchmarks/geometries/ni_porphine.xyz").read_text())
    counts: dict = {}
    for symbol, *_ in atoms:
        counts[symbol] = counts.get(symbol, 0) + 1
    assert counts == {"Ni": 1, "C": 20, "N": 4, "H": 12}
    assert all(abs(a[3]) < 1e-9 for a in atoms), "porphine must be planar"


# ------------------------------------------------------------ validator gates

GOOD = {
    "schema_version": "1.0", "id": "good_case", "status": "ready", "role": "validation", "tier": 0,
    "spin_family": "good", "title": "t", "description": "d",
    "molecule": {"charge": 0, "multiplicity": 3,
                 "geometry": {"units": "angstrom", "xyz": "C 0 0 0\nH 0.9921 0 0.4216\nH -0.9921 0 0.4216"}},
    "reference": {"type": "ROHF", "basis": "6-31G", "expected_nbasis": 13, "fock_source": "gaussian"},
    "active_space": {"window": {"nfirst_1based": 2, "nlast_1based": 13},
                     "frozen_core": 1, "n_orbitals": 12, "n_electrons": 6, "ms2": 2},
    "solver": {"kind": "pyscf_fci"},
    "provenance": {"added": "2026-09-18", "geometry_source": "test fixture"},
}


def test_the_good_case_is_good():
    assert manifest.validate(GOOD) == []


def _broken(**changes):
    man = copy.deepcopy(GOOD)
    for dotted, value in changes.items():
        keys = dotted.split(".")
        target = man
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return man


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"active_space.frozen_core": 2}, "declared frozen_core=2"),
        ({"active_space.n_orbitals": 11}, "declared n_orbitals=11"),
        ({"active_space.n_electrons": 8}, "declared n_electrons=8"),
        ({"active_space.ms2": 0}, "declared ms2=0"),
        ({"molecule.multiplicity": 2}, "incompatible parity"),
        ({"active_space.window": {"nfirst_1based": 2, "nlast_1based": 40}}, "exceeds the 13 orbitals"),
        ({"active_space.window": {"nfirst_1based": 6, "nlast_1based": 13}}, "active beta electrons"),
        ({"reference.type": "RKS", "reference.functional": "B3LYP"}, "must use fock_source 'pyscf_rebuilt'"),
        ({"reference.type": "RKS"}, "needs a 'functional'"),
        ({"reference.functional": "B3LYP"}, "Hartree-Fock but a functional"),
        ({"reference.fock_source": "magic"}, "fock_source must be one of"),
        ({"reference.type": "UHF"}, "type must be one of"),
        ({"id": "Good Case"}, "lowercase words"),
        ({"status": "blocked"}, "does not say what is missing"),
        ({"role": "misc"}, "role must be one of"),
        ({"schema_version": "0.9"}, "schema_version is"),
        ({"solver.kind": "quantum_oracle"}, "kind must be one of"),
        ({"provenance.added": "18/09/2026"}, "must be an ISO date"),
        ({"molecule.geometry.units": "furlongs"}, "units must be one of"),
    ],
)
def test_validator_catches(changes, expected):
    problems = manifest.validate(_broken(**changes))
    assert any(expected in p for p in problems), f"expected {expected!r} in {problems}"


def test_validator_catches_overlapping_nuclei():
    man = _broken(**{"molecule.geometry.xyz": "C 0 0 0\nH 0 0 0.1\nH -0.9921 0 0.4216"})
    problems = manifest.validate(man)
    assert any("sanity floor" in p for p in problems)


def test_validator_catches_an_unknown_element():
    man = _broken(**{"molecule.geometry.xyz": "C 0 0 0\nXx 0.9921 0 0.4216\nH -0.9921 0 0.4216"})
    problems = manifest.validate(man)
    assert any("unknown element" in p for p in problems)


def test_validator_catches_a_typo_in_a_key_name():
    """An unknown key is a typo, and a typo that is silently ignored is a wrong run."""
    man = copy.deepcopy(GOOD)
    man["active_space"]["frozen_cores"] = 1
    problems = manifest.validate(man)
    assert any("unknown key 'frozen_cores'" in p for p in problems)


def test_validator_catches_both_window_and_policy():
    man = copy.deepcopy(GOOD)
    man["active_space"]["window_policy"] = {"reference_orbital": "homo", "n_below": 1, "n_above": 1}
    problems = manifest.validate(man)
    assert any("exactly one" in p for p in problems)


def test_validator_catches_a_policy_width_mismatch():
    man = copy.deepcopy(GOOD)
    del man["active_space"]["window"]
    del man["active_space"]["frozen_core"]
    del man["active_space"]["n_electrons"]
    del man["active_space"]["ms2"]
    man["active_space"]["window_policy"] = {"reference_orbital": "homo", "n_below": 2, "n_above": 3}
    problems = manifest.validate(man)
    assert any("the policy spans" in p for p in problems)


def test_validator_catches_an_intractable_fci_request():
    man = copy.deepcopy(GOOD)
    man["reference"]["expected_nbasis"] = 60
    man["active_space"]["window"] = {"nfirst_1based": 2, "nlast_1based": 41}
    man["active_space"]["n_orbitals"] = 40
    problems = manifest.validate(man)
    assert any("treats as tractable" in p for p in problems)


def test_validator_reports_every_problem_at_once():
    man = _broken(**{"active_space.frozen_core": 2, "role": "misc", "provenance.added": "yesterday"})
    assert len(manifest.validate(man)) >= 3


def test_check_raises_with_all_problems_in_the_message():
    man = _broken(**{"role": "misc"})
    with pytest.raises(manifest.ManifestError, match="role must be one of"):
        manifest.check(man)


def test_check_returns_a_good_manifest():
    assert manifest.check(copy.deepcopy(GOOD)) is not None


def test_load_rejects_broken_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(manifest.ManifestError, match="not valid JSON"):
        manifest.load(path)


def test_load_records_the_path_for_the_id_check(tmp_path):
    path = tmp_path / "some_name.json"
    path.write_text(json.dumps({"id": "other_name"}), encoding="utf-8")
    man = manifest.load(path)
    problems = manifest.validate(man)
    assert any("does not match filename" in p for p in problems)
