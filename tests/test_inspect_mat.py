"""Tests for scripts/inspect_mat.py.

The probe is the M0 deliverable and it runs on a cluster we cannot reach, so a
crash in it costs a full Gaussian round-trip. These tests exercise it against
``tests/fake_gauopen/``, a mock that deliberately reproduces the awkward parts of
the real thing (lower-triangular packing, a property that raises, a ``scalar()``
that rejects unknown names). No gauopen and no Gaussian required, so this runs in
CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "inspect_mat.py"
FAKE_GAUOPEN = Path(__file__).resolve().parent / "fake_gauopen"

sys.path.insert(0, str(SCRIPT.parent))
import inspect_mat  # noqa: E402


# --------------------------------------------------------------- packing math

@pytest.mark.parametrize(
    "length, expected",
    [
        (28, "lower-triangular packed with n=7"),  # 7*8/2, an AO 1e block
        (49, "square n x n with n=7"),  # nbasis**2
        (1296, "full 4-index n^4 with n=6"),  # nact**4 for a 6-orbital window
        (20736, "full 4-index n^4 with n=12"),  # nact**4 for a 12-orbital window
    ],
)
def test_packing_hypotheses_includes_truth(length, expected):
    """The correct packing must be among the candidates the probe reports."""
    assert expected in inspect_mat.packing_hypotheses(length)


def test_packing_hypotheses_eightfold():
    """8-fold packed (pq|rs) for n orbitals: npair=n(n+1)/2, then npair(npair+1)/2."""
    n = 6
    npair = n * (n + 1) // 2
    length = npair * (npair + 1) // 2
    assert f"8-fold packed (pq|rs) with n={n}" in inspect_mat.packing_hypotheses(length)


def test_packing_hypotheses_reports_nothing_rather_than_guessing():
    """A length matching no simple packing must say so, not invent a dimension."""
    assert inspect_mat.packing_hypotheses(7) == ["no simple packing matches this length"]


@pytest.mark.parametrize("n", range(1, 40))
def test_roots_are_exact(n):
    """The root helpers must never round a near-miss into a false positive."""
    assert inspect_mat._triangular_root(n * (n + 1) // 2) == n
    assert inspect_mat._integer_root(n**4, 4) == n
    assert inspect_mat._integer_root(n**2, 2) == n
    # One element off a perfect square is not a square.
    if n > 2:
        assert inspect_mat._integer_root(n**2 + 1, 2) is None


# ------------------------------------------------------- attribute robustness

def test_safe_getattr_swallows_non_attribute_errors():
    """getattr(obj, name, default) only swallows AttributeError; ours must not."""

    class Exploding:
        @property
        def boom(self):
            raise RuntimeError("nope")

    assert inspect_mat._safe_getattr(Exploding(), "boom") is inspect_mat._MISSING
    assert inspect_mat._is_callable_attr(Exploding(), "boom") is False


# --------------------------------------------------------------- end-to-end

@pytest.fixture(scope="module")
def probe_report(tmp_path_factory):
    """Run the probe as a subprocess against the mock, exactly as on the cluster."""
    out = tmp_path_factory.mktemp("probe") / "report.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "fake.mat", "--json", str(out)],
        cwd=FAKE_GAUOPEN,
        env={"PYTHONPATH": str(FAKE_GAUOPEN), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(out.read_text()), result.stdout


def test_probe_describes_every_matlist_entry(probe_report):
    """Every entry must be described. A silent 'error' key means the probe broke."""
    report, _ = probe_report
    entries = report["matlist_entries"]
    assert len(entries) == len(report["matlist_keys"])
    for entry in entries:
        assert "error" not in entry, f"{entry['label']}: {entry.get('error')}"
        assert entry["python_type"] == "_Mat"
        assert isinstance(entry["array"], dict), entry["array"]


def test_probe_survives_a_raising_property(probe_report):
    """A property that raises is evidence to report, not a reason to die."""
    report, _ = probe_report
    entry = next(e for e in report["matlist_entries"] if e["label"] == "OVERLAP")
    assert "raised RuntimeError" in entry["attributes"]["exploding"]
    assert entry["methods"] == ["expand", "lenarray"]


def test_probe_infers_the_real_packing(probe_report):
    """The mock's true layouts must appear among the reported candidates."""
    report, _ = probe_report
    by_label = {e["label"]: e for e in report["matlist_entries"]}

    core = by_label["CORE HAMILTONIAN ALPHA"]
    assert "lower-triangular packed with n=7" in core["array"]["packing_candidates"]
    assert core["expand"]["shape"] == "(7, 7)"

    eri = by_label["AA MO 2E INTEGRALS"]
    assert "full 4-index n^4 with n=6" in eri["array"]["packing_candidates"]


def test_probe_answers_the_fock_question(probe_report):
    """The whole point of M0: say plainly whether a Fock matrix is present."""
    report, stdout = probe_report
    # The mock has no Fock matrix, so the probe must report its absence.
    assert report["answers"]["fock"] == []
    assert "[ABSENT ] fock" in stdout
    assert report["answers"]["beta_mo_coefficients"] == []


def test_probe_reports_scalar_hits_and_misses(probe_report):
    """Both a found scalar and a rejected name are information; keep both."""
    report, _ = probe_report
    probes = report["scalars"]["me.scalar(...) probes"]
    assert probes["ENUCREP"] == "9.1671"
    assert "raised KeyError" in probes["TOTAL ENERGY"]
