"""Tests for the active-space sweep generator.

A sweep is a claim that one active space is big enough, so the generator has to
be trustworthy about the two things that would silently invalidate that claim:
windows that are not nested, and two spin states run in different active spaces.
Both are errors here rather than notes in a docstring.

Everything runs off the committed manifests and the committed Gaussian
templates. No Gaussian and no gauopen.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks import manifest as mf, sweep  # noqa: E402


def system(name):
    return mf.load(mf.systems_dir() / f"{name}.json")


# ---------------------------------------------------------------- window maths

def test_parse_windows():
    assert sweep.parse_windows("40-50, 38-52") == [("40-50", 40, 50), ("38-52", 38, 52)]


@pytest.mark.parametrize("text", ["", "40", "50-40", "a-b"])
def test_parse_windows_rejects_nonsense(text):
    with pytest.raises((sweep.SweepError, mf.ManifestError)):
        sweep.parse_windows(text)


def test_nested_windows_are_accepted():
    windows = [("a", 40, 50), ("b", 38, 52), ("c", 30, 60)]
    assert sweep.check_nested(windows)[0][0] == "a"


def test_unnested_windows_are_rejected():
    """A wider window that drops orbitals the narrower one had is not a sweep."""
    with pytest.raises(sweep.SweepError, match="not contained in"):
        sweep.check_nested([("a", 40, 50), ("b", 42, 60)])


def test_policy_expands_into_nested_windows():
    windows = sweep.windows_from_policy(system("ni_porphine_singlet"), homo_1based=94)
    assert [(w[1], w[2]) for w in windows] == [(90, 95), (85, 100), (81, 104), (77, 108)]
    sweep.check_nested(windows)


def test_policy_expansion_respects_the_basis_size():
    with pytest.raises(mf.ManifestError, match="above the"):
        sweep.windows_from_policy(system("ni_porphine_singlet"), homo_1based=94, nbasis=100)


def test_an_explicit_window_expands_to_itself():
    windows = sweep.windows_from_policy(system("h2o_rhf_sto3g"))
    assert windows == [("as declared", 2, 7)]


# -------------------------------------------------------------------- expansion

def test_expansion_produces_valid_manifests():
    man = system("ni_porphine_triplet")
    for label, counts, child in sweep.expand(man, [("x", 85, 100)], repo_root=REPO):
        assert mf.validate(child, repo_root=REPO) == []
        assert child["id"] == "ni_porphine_triplet_w85_100"
        assert child["active_space"]["window"] == {"nfirst_1based": 85, "nlast_1based": 100}
        assert counts["nact"] == 16


def test_expansion_inlines_the_geometry():
    """A sweep directory has to be copyable to a cluster on its own."""
    man = system("ni_porphine_singlet")
    assert "file" in man["molecule"]["geometry"]
    _, _, child = sweep.expand(man, [("x", 85, 100)], repo_root=REPO)[0]
    assert "file" not in child["molecule"]["geometry"]
    assert len(mf.parse_xyz(child["molecule"]["geometry"]["xyz"])) == 37


def test_expansion_records_where_it_came_from():
    _, _, child = sweep.expand(system("h2o_rhf_sto3g"), [("full valence", 2, 7)], repo_root=REPO)[0]
    assert "h2o_rhf_sto3g" in child["provenance"]["notes"]
    assert "full valence" in child["provenance"]["notes"]


# -------------------------------------------------------------------- rendering

def test_route_line_carries_the_window_and_the_method():
    man = system("ch2_rohf_631g")
    template = (REPO / "gaussian/rohf_window.gjf").read_text()
    text = sweep.render_input(man, 2, 13, "job", template, repo_root=REPO)
    assert "#P ROHF/6-31G" in text
    assert "Window=(2,13)" in text
    assert "%chk=job.chk" in text
    assert text.rstrip().endswith("job.mat")


def test_charge_and_multiplicity_line_is_right_for_an_anion():
    man = system("nicl4_td_triplet")
    template = (REPO / "gaussian/rohf_window.gjf").read_text()
    text = sweep.render_input(man, 40, 50, "job", template, repo_root=REPO)
    assert "\n-2 3\n" in text


def test_closed_shell_template_keeps_its_fixed_multiplicity():
    man = system("h2o_rhf_sto3g")
    template = (REPO / "gaussian/rhf_window.gjf").read_text()
    text = sweep.render_input(man, 2, 7, "job", template, repo_root=REPO)
    assert "\n0 1\n" in text
    assert "#P HF/STO-3G" in text


def test_geometry_lands_in_the_input():
    man = system("ch2_rohf_631g")
    template = (REPO / "gaussian/rohf_window.gjf").read_text()
    text = sweep.render_input(man, 2, 13, "job", template, repo_root=REPO)
    atoms = mf.parse_xyz(text.split("\n0 3\n", 1)[1].split("\n\n", 1)[0])
    assert atoms == mf.geometry_atoms(man, REPO)


def test_no_placeholder_survives_rendering():
    template = (REPO / "gaussian/rohf_window.gjf").read_text()
    text = sweep.render_input(system("ch2_rohf_631g"), 2, 13, "job", template, repo_root=REPO)
    for placeholder in ("NAME", "BASIS", "CHARGE", "MULTIPLICITY", "GEOMETRY_HERE", "NFIRST", "NLAST"):
        assert placeholder not in text


def test_a_template_that_lost_its_window_keyword_is_an_error():
    """The failure this guards against costs days of queue time and looks fine."""
    template = (REPO / "gaussian/rohf_window.gjf").read_text()
    without_window = template.replace(" Window=(NFIRST,NLAST)", "")
    with pytest.raises(sweep.SweepError, match="NFIRST, NLAST"):
        sweep.render_input(system("ch2_rohf_631g"), 2, 13, "job", without_window, repo_root=REPO)


def test_a_template_missing_the_geometry_is_an_error():
    template = (REPO / "gaussian/rohf_window.gjf").read_text().replace("GEOMETRY_HERE", "")
    with pytest.raises(sweep.SweepError, match="GEOMETRY_HERE"):
        sweep.render_input(system("ch2_rohf_631g"), 2, 13, "job", template, repo_root=REPO)


def test_the_committed_templates_have_every_required_placeholder():
    for name in ("rhf_window.gjf", "rohf_window.gjf"):
        template = (REPO / "gaussian" / name).read_text()
        for placeholder in sweep.REQUIRED_PLACEHOLDERS:
            assert placeholder in template, f"{name} lost {placeholder}"


def test_kohn_sham_reference_swaps_the_method_token():
    man = system("h2o_rks_b3lyp_631g")
    template = (REPO / "gaussian/rhf_window.gjf").read_text()
    text = sweep.render_input(man, 2, 13, "job", template, repo_root=REPO)
    assert "#P B3LYP/6-31G" in text


def test_method_token():
    assert sweep.method_token({"type": "RHF"}) == "HF"
    assert sweep.method_token({"type": "ROHF"}) == "ROHF"
    assert sweep.method_token({"type": "RKS", "functional": "B3LYP"}) == "B3LYP"
    assert sweep.method_token({"type": "ROKS", "functional": "B3LYP"}) == "ROB3LYP"


def test_per_element_basis_is_refused_rather_than_mis_rendered():
    with pytest.raises(sweep.SweepError, match="Gen basis section"):
        sweep.basis_token({"basis": {"Ni": "anoroostz"}})


def test_job_script_is_named_after_the_job():
    template = (REPO / "gaussian/job_template.sh").read_text()
    text = sweep.render_job_script(template, "my_job_w2_7")
    assert "#PBS -N my_job_w2_7" in text
    assert '"${NAME:-my_job_w2_7}"' in text
    assert "CHANGEME" not in text


# ------------------------------------------------------------------- generation

def test_generate_writes_a_job_per_system_and_window(tmp_path):
    manifests = [system("ni_porphine_singlet"), system("ni_porphine_triplet")]
    windows = sweep.windows_from_policy(manifests[0], homo_1based=94)
    result = sweep.generate(manifests, windows, tmp_path, repo_root=REPO)

    assert len(result["jobs"]) == 8
    for job in result["jobs"]:
        assert (tmp_path / job["gjf"]).is_file()
        assert (tmp_path / job["manifest"]).is_file()
        assert (tmp_path / f"{job['name']}.pbs").is_file()
    assert (tmp_path / "sweep.json").is_file()


def test_generated_manifests_all_validate(tmp_path):
    manifests = [system("ni_porphine_singlet")]
    sweep.generate(manifests, sweep.windows_from_policy(manifests[0], homo_1based=94), tmp_path, repo_root=REPO)
    for path in (tmp_path / "manifests").glob("*.json"):
        assert mf.validate(mf.load(path), repo_root=REPO) == [], path.name


def test_generate_keeps_both_spin_states_in_the_same_window(tmp_path):
    manifests = [system("ni_porphine_singlet"), system("ni_porphine_triplet")]
    sweep.generate(manifests, [("x", 85, 100)], tmp_path, repo_root=REPO)
    windows = {
        (job["active_space"]["nfirst_1based"], job["active_space"]["nlast_1based"])
        for job in json.loads((tmp_path / "sweep.json").read_text())["jobs"]
    }
    assert windows == {(85, 100)}


def test_generate_refuses_to_mix_spin_families(tmp_path):
    with pytest.raises(sweep.SweepError, match="one spin family|cannot sweep"):
        sweep.generate([system("ni_porphine_singlet"), system("nicl4_td_singlet")],
                       [("x", 40, 50)], tmp_path, repo_root=REPO)


def test_generate_refuses_a_blocked_entry(tmp_path):
    with pytest.raises(sweep.SweepError, match="is blocked"):
        sweep.generate([system("ni_pap_full")], [("x", 40, 50)], tmp_path, repo_root=REPO)


def test_generate_refuses_unnested_windows(tmp_path):
    with pytest.raises(sweep.SweepError, match="not contained in"):
        sweep.generate([system("ni_porphine_singlet")], [("a", 85, 100), ("b", 88, 110)],
                       tmp_path, repo_root=REPO)


def test_generate_refuses_a_window_that_leaves_no_electrons(tmp_path):
    """A window starting above the occupied orbitals freezes electrons that are not there."""
    man = copy.deepcopy(system("h2o_rhf_sto3g"))
    man["active_space"] = {"window_policy": {"reference_orbital": 7, "n_below": 0, "n_above": 0}}
    with pytest.raises(sweep.SweepError, match="does not validate"):
        sweep.generate([man], [("x", 7, 7)], tmp_path, repo_root=REPO)


def test_sweep_json_records_every_active_space(tmp_path):
    manifests = [system("ni_porphine_triplet")]
    sweep.generate(manifests, sweep.windows_from_policy(manifests[0], homo_1based=94), tmp_path, repo_root=REPO)
    recorded = json.loads((tmp_path / "sweep.json").read_text())
    assert [job["active_space"]["nact"] for job in recorded["jobs"]] == [6, 16, 24, 32]
    for job in recorded["jobs"]:
        space = job["active_space"]
        assert space["act_start"] == space["nfirst_1based"] - 1
        assert space["act_stop"] == space["nlast_1based"]
        assert space["nalpha_act"] - space["nbeta_act"] == space["ms2"]


def test_cli_runs_a_sweep(tmp_path):
    code = sweep.main([
        "ni_porphine_singlet", "ni_porphine_triplet",
        "--homo", "94", "--out", str(tmp_path), "--repo-root", str(REPO),
    ])
    assert code == 0
    assert len(list(tmp_path.glob("*.gjf"))) == 8


def test_cli_accepts_explicit_windows(tmp_path):
    code = sweep.main([
        "h2o_rhf_sto3g", "--windows", "2-7,2-7", "--out", str(tmp_path),
        "--repo-root", str(REPO), "--no-job-script",
    ])
    assert code == 0
    assert not list(tmp_path.glob("*.pbs"))
