"""Collect and compare the results of an active-space sweep.

A sweep produces one number per (system, spin state, window). On its own that is
a list; what makes it an argument is the comparison between windows, and that is
what this module builds:

* what each active space actually was -- orbitals, electrons, alpha/beta split;
* the solver energy for it, with its uncertainty when the solver reports one;
* the spin-state splitting at each window, which is usually the quantity the
  chemistry turns on;
* how far each window is from the next larger one, which is the only evidence
  that the active space was big enough;
* natural-orbital occupations and the multireference diagnostics that follow
  from them, when the solver wrote a 1-RDM;
* runtime and peak memory, when the run recorded them.

This module reads *result records*: one small JSON file per finished job, whose
shape is defined by :data:`RESULT_KEYS` and checked by :func:`validate_result`.
Nothing here runs a solver or parses solver output -- the solver adapters own
that, and they emit these records. Keeping the boundary here means the analysis
runs on a laptop against a directory of JSON copied off the cluster.

Numpy is the only import at module scope. PySCF is used lazily by
:func:`fci_energy_from_fcidump` and by nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import manifest as mf

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.509474063
HARTREE_TO_CM = 219474.6313632

#: Result-record fields. The solver adapters produce these; everything optional
#: is *absent* rather than null when the solver did not report it, matching the
#: convention the .npz bundle uses.
RESULT_KEYS = {
    "required": ("schema_version", "job", "solver", "energy_hartree"),
    "optional": (
        "kind", "system", "spin_family", "multiplicity", "window", "converged",
        "energy_uncertainty_hartree", "variational_energy_hartree",
        "natural_occupations", "runtime_seconds", "peak_memory_gb", "provenance", "notes",
    ),
}

#: Occupations this close to 0 or 2 count as closed or empty rather than active.
OCCUPATION_THRESHOLD = 0.02

#: A window is called converged when it sits this close to the next larger one.
#: 1 mHa is the usual "chemically irrelevant" line for a total energy; for a
#: splitting the interesting threshold is smaller, so both are reported and the
#: caller picks.
DEFAULT_CONVERGENCE_HARTREE = 1.0e-3


class AnalysisError(ValueError):
    """Raised when results and the sweep they claim to belong to do not line up."""


# -------------------------------------------------------------- result records

def validate_result(record: dict) -> list:
    """Return a list of problems with one result record."""
    problems = []
    for key in RESULT_KEYS["required"]:
        if key not in record:
            problems.append(f"missing required key {key!r}")
    allowed = set(RESULT_KEYS["required"]) | set(RESULT_KEYS["optional"]) | {"_path"}
    for key in sorted(set(record) - allowed):
        problems.append(f"unknown key {key!r}")

    if record.get("schema_version") not in (None, mf.SCHEMA_VERSION):
        problems.append(f"schema_version is {record['schema_version']!r}, expected {mf.SCHEMA_VERSION!r}")

    energy = record.get("energy_hartree")
    if energy is not None and not isinstance(energy, (int, float)):
        problems.append("energy_hartree must be a number")
    elif isinstance(energy, float) and not math.isfinite(energy):
        problems.append(f"energy_hartree is {energy}")

    for key in ("energy_uncertainty_hartree", "variational_energy_hartree", "runtime_seconds", "peak_memory_gb"):
        value = record.get(key)
        if value is not None and not isinstance(value, (int, float)):
            problems.append(f"{key} must be a number")
    if isinstance(record.get("runtime_seconds"), (int, float)) and record["runtime_seconds"] < 0:
        problems.append("runtime_seconds must not be negative")

    occupations = record.get("natural_occupations")
    if occupations is not None:
        if not isinstance(occupations, (list, tuple)) or not occupations:
            problems.append("natural_occupations must be a non-empty list")
        else:
            for value in occupations:
                if not isinstance(value, (int, float)):
                    problems.append("natural_occupations must contain numbers")
                    break
                if not -1e-6 <= value <= 2.0 + 1e-6:
                    problems.append(f"natural occupation {value} is outside [0, 2]")
                    break

    if record.get("converged") is not None and not isinstance(record["converged"], bool):
        problems.append("converged must be a boolean")
    return problems


def load_results(directory) -> list:
    """Load every ``*.json`` result record in *directory*, checking each one."""
    directory = Path(directory)
    records = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("kind") == "active_space_sweep":
            continue  # the sweep manifest itself may sit in the same directory
        record.setdefault("_path", str(path))
        problems = validate_result(record)
        if problems:
            raise AnalysisError(f"{path}: " + "; ".join(problems))
        records.append(record)
    return records


def load_sweep(path) -> dict:
    """Load a ``sweep.json`` written by :mod:`benchmarks.sweep`."""
    path = Path(path)
    if path.is_dir():
        path = path / "sweep.json"
    sweep = json.loads(path.read_text(encoding="utf-8"))
    if sweep.get("kind") != "active_space_sweep":
        raise AnalysisError(f"{path} is not a sweep manifest")
    return sweep


# ------------------------------------------------------------ RDM diagnostics

def occupations_from_rdm1(rdm1) -> list:
    """Natural-orbital occupations from a spin-summed one-particle density matrix.

    Symmetrised before diagonalising, because a solver's RDM is usually symmetric
    only to its own convergence threshold, and returned largest first.
    """
    rdm1 = np.asarray(rdm1, dtype=float)
    if rdm1.ndim != 2 or rdm1.shape[0] != rdm1.shape[1]:
        raise AnalysisError(f"a 1-RDM must be square, got shape {rdm1.shape}")
    occupations = np.linalg.eigvalsh(0.5 * (rdm1 + rdm1.T))
    return sorted((float(v) for v in occupations), reverse=True)


def occupation_diagnostics(occupations, threshold: float = OCCUPATION_THRESHOLD) -> dict:
    """Multireference diagnostics from natural-orbital occupations.

    ``n_unpaired`` is Head-Gordon's ``sum min(n_i, 2 - n_i)``; ``n_unpaired_nl``
    is his nonlinear form ``sum n_i^2 (2 - n_i)^2``, which suppresses the weakly
    correlated tail and so tracks genuine open-shell character more sharply.

    ``n_fractional`` is the count of occupations that are neither closed nor
    empty, and is the number to watch across a sweep: if the largest occupation
    in the virtual set is still well above the threshold at the widest window,
    the active space has not yet caught everything that is correlated.
    """
    values = [float(v) for v in occupations]
    if not values:
        raise AnalysisError("no occupations given")
    ordered = sorted(values, reverse=True)
    near_two = [v for v in ordered if v >= 2.0 - threshold]
    near_zero = [v for v in ordered if v <= threshold]
    fractional = [v for v in ordered if threshold < v < 2.0 - threshold]
    return {
        "n_orbitals": len(ordered),
        "n_electrons_from_occupations": sum(ordered),
        "n_closed": len(near_two),
        "n_empty": len(near_zero),
        "n_fractional": len(fractional),
        "n_unpaired": sum(min(v, 2.0 - v) for v in ordered),
        "n_unpaired_nl": sum(v * v * (2.0 - v) ** 2 for v in ordered),
        "max_deviation_from_integer": max(min(v, abs(2.0 - v)) for v in ordered),
        "largest_fractional": max(fractional) if fractional else None,
        "smallest_fractional": min(fractional) if fractional else None,
    }


# ------------------------------------------------------------------- assembly

def collect(sweep: dict, results, strict: bool = False) -> dict:
    """Join a sweep manifest to its result records.

    Returns ``{"rows": [...], "missing": [...], "orphans": [...]}``. A job with no
    result is reported rather than dropped: a sweep with a hole in it is exactly
    the case where a quiet omission would look like a converged series.
    """
    by_job = {}
    for record in results:
        job = record["job"]
        if job in by_job:
            raise AnalysisError(f"two result records claim job {job!r}")
        by_job[job] = record

    rows, missing = [], []
    for job in sweep["jobs"]:
        record = by_job.pop(job["name"], None)
        if record is None:
            missing.append(job["name"])
            continue
        window = record.get("window")
        if window and (
            window.get("nfirst_1based") != job["active_space"]["nfirst_1based"]
            or window.get("nlast_1based") != job["active_space"]["nlast_1based"]
        ):
            raise AnalysisError(
                f"{job['name']}: the result records window "
                f"{window.get('nfirst_1based')}-{window.get('nlast_1based')} but the sweep generated "
                f"{job['active_space']['nfirst_1based']}-{job['active_space']['nlast_1based']}"
            )
        space = job["active_space"]
        row = {
            "job": job["name"],
            "system": job["system"],
            "spin_family": job.get("spin_family"),
            "label": job.get("label"),
            "multiplicity": job["multiplicity"],
            "nfirst_1based": space["nfirst_1based"],
            "nlast_1based": space["nlast_1based"],
            "n_orbitals": space["nact"],
            "n_electrons": space["nelec_act"],
            "nalpha": space["nalpha_act"],
            "nbeta": space["nbeta_act"],
            "n_determinants": space["n_determinants"],
            "solver": record["solver"],
            "energy_hartree": float(record["energy_hartree"]),
            "energy_uncertainty_hartree": record.get("energy_uncertainty_hartree"),
            "converged": record.get("converged"),
            "runtime_seconds": record.get("runtime_seconds"),
            "peak_memory_gb": record.get("peak_memory_gb"),
        }
        occupations = record.get("natural_occupations")
        row["occupations"] = occupation_diagnostics(occupations) if occupations else None
        rows.append(row)

    orphans = sorted(by_job)
    if strict and (missing or orphans):
        raise AnalysisError(
            f"sweep and results do not match: {len(missing)} job(s) without a result "
            f"({', '.join(missing[:3])}...), {len(orphans)} result(s) without a job"
        )
    return {"rows": rows, "missing": missing, "orphans": orphans}


def add_window_deviations(rows, tolerance: float = DEFAULT_CONVERGENCE_HARTREE):
    """Attach each row's deviation from the next larger active space.

    Grouped by system, so a singlet is compared with a singlet. The largest
    window in each group has nothing above it and gets ``None``, which is the
    honest answer: nothing in the sweep says whether it is converged.
    """
    groups: dict = {}
    for row in rows:
        groups.setdefault(row["system"], []).append(row)

    for group in groups.values():
        group.sort(key=lambda r: (r["n_orbitals"], r["n_electrons"]))
        for row, larger in zip(group, group[1:]):
            delta = row["energy_hartree"] - larger["energy_hartree"]
            row["deviation_from_next_hartree"] = delta
            row["deviation_from_next_mhartree"] = delta * 1000.0
            row["deviation_from_next_kcal"] = delta * HARTREE_TO_KCAL
            row["converged_against_next"] = abs(delta) < tolerance
            row["next_larger_job"] = larger["job"]
        group[-1].setdefault("deviation_from_next_hartree", None)
        group[-1].setdefault("converged_against_next", None)
        group[-1].setdefault("next_larger_job", None)
    return rows


def spin_splittings(rows) -> list:
    """Spin-state splittings, one per (family, window) that has two states.

    Reported as the higher multiplicity minus the lower one, so a negative
    splitting means the higher-spin state is the ground state. A splitting is
    only formed between two rows with the *same* window; two different active
    spaces are two different Hamiltonians.
    """
    groups: dict = {}
    for row in rows:
        key = (row["spin_family"], row["nfirst_1based"], row["nlast_1based"])
        groups.setdefault(key, []).append(row)

    splittings = []
    for (family, nfirst, nlast), members in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if len({m["multiplicity"] for m in members}) < 2:
            continue
        members = sorted(members, key=lambda r: r["multiplicity"])
        low, high = members[0], members[-1]
        delta = high["energy_hartree"] - low["energy_hartree"]
        uncertainty = None
        if low.get("energy_uncertainty_hartree") is not None and high.get("energy_uncertainty_hartree") is not None:
            uncertainty = math.hypot(low["energy_uncertainty_hartree"], high["energy_uncertainty_hartree"])
        splittings.append({
            "spin_family": family,
            "nfirst_1based": nfirst,
            "nlast_1based": nlast,
            "n_orbitals": low["n_orbitals"],
            "n_electrons": low["n_electrons"],
            "label": low.get("label"),
            "lower_multiplicity": low["multiplicity"],
            "higher_multiplicity": high["multiplicity"],
            "lower_job": low["job"],
            "higher_job": high["job"],
            "splitting_hartree": delta,
            "splitting_ev": delta * HARTREE_TO_EV,
            "splitting_kcal": delta * HARTREE_TO_KCAL,
            "splitting_cm": delta * HARTREE_TO_CM,
            "splitting_uncertainty_hartree": uncertainty,
            "ground_state_multiplicity": high["multiplicity"] if delta < 0 else low["multiplicity"],
        })

    # How much the splitting itself moves between windows is the number the
    # chemistry rests on, and it converges far more slowly than a total energy.
    by_family: dict = {}
    for entry in splittings:
        by_family.setdefault(entry["spin_family"], []).append(entry)
    for family in by_family.values():
        family.sort(key=lambda e: e["n_orbitals"])
        for entry, larger in zip(family, family[1:]):
            entry["splitting_shift_to_next_kcal"] = (larger["splitting_hartree"] - entry["splitting_hartree"]) * HARTREE_TO_KCAL
            entry["ground_state_changes_at_next"] = entry["ground_state_multiplicity"] != larger["ground_state_multiplicity"]
        family[-1].setdefault("splitting_shift_to_next_kcal", None)
        family[-1].setdefault("ground_state_changes_at_next", None)
    # Smallest active space first, so the table reads in the direction the sweep
    # was run and a shift toward convergence reads downward.
    splittings.sort(key=lambda e: (str(e["spin_family"]), e["n_orbitals"]))
    return splittings


def summarize(sweep: dict, results, tolerance: float = DEFAULT_CONVERGENCE_HARTREE, strict: bool = False) -> dict:
    """The whole analysis: rows, deviations, splittings and what is missing."""
    collected = collect(sweep, results, strict=strict)
    rows = add_window_deviations(collected["rows"], tolerance=tolerance)
    return {
        "spin_family": sweep.get("spin_family"),
        "systems": sweep.get("systems"),
        "tolerance_hartree": tolerance,
        "rows": sorted(rows, key=lambda r: (r["system"], r["n_orbitals"])),
        "splittings": spin_splittings(rows),
        "missing": collected["missing"],
        "orphans": collected["orphans"],
    }


# -------------------------------------------------------------------- reports

def _cell(value, spec="{:.6f}", dash="-"):
    return dash if value is None else spec.format(value)


def format_report(summary: dict) -> str:
    """A plain-text report: one table of active spaces, one of splittings."""
    lines = []
    systems = ", ".join(summary.get("systems") or [])
    lines.append(f"Active-space sweep: {systems}")
    lines.append(f"convergence threshold {summary['tolerance_hartree']:.1e} Ha between adjacent windows")
    lines.append("")

    header = (f"{'window':>9s}  {'orb':>3s}  {'elec':>4s}  {'mult':>4s}  {'energy / Ha':>16s}  "
              f"{'+/-':>9s}  {'d(next) / mHa':>13s}  {'nfrac':>5s}  {'Nu':>6s}  {'time / s':>9s}  label")
    lines.append(header)
    lines.append("-" * len(header))
    for row in summary["rows"]:
        occupations = row.get("occupations") or {}
        lines.append(
            f"{row['nfirst_1based']:>4d}-{row['nlast_1based']:<4d} "
            f"{row['n_orbitals']:>3d}  {row['n_electrons']:>4d}  {row['multiplicity']:>4d}  "
            f"{row['energy_hartree']:>16.8f}  "
            f"{_cell(row.get('energy_uncertainty_hartree'), '{:.2e}'):>9s}  "
            f"{_cell(row.get('deviation_from_next_mhartree'), '{:+.4f}'):>13s}  "
            f"{_cell(occupations.get('n_fractional'), '{:d}'):>5s}  "
            f"{_cell(occupations.get('n_unpaired'), '{:.3f}'):>6s}  "
            f"{_cell(row.get('runtime_seconds'), '{:.0f}'):>9s}  "
            f"{row.get('label') or ''}"
        )

    if summary["splittings"]:
        lines.append("")
        header = (f"{'window':>9s}  {'orb':>3s}  {'states':>7s}  {'splitting / kcal':>17s}  "
                  f"{'shift to next':>13s}  ground state")
        lines.append(header)
        lines.append("-" * len(header))
        for entry in summary["splittings"]:
            flag = "  <-- ordering changes at the next window" if entry.get("ground_state_changes_at_next") else ""
            lines.append(
                f"{entry['nfirst_1based']:>4d}-{entry['nlast_1based']:<4d} "
                f"{entry['n_orbitals']:>3d}  "
                f"{entry['lower_multiplicity']:>3d}/{entry['higher_multiplicity']:<3d}  "
                f"{entry['splitting_kcal']:>17.3f}  "
                f"{_cell(entry.get('splitting_shift_to_next_kcal'), '{:+.3f}'):>13s}  "
                f"multiplicity {entry['ground_state_multiplicity']}{flag}"
            )

    if summary["missing"]:
        lines.append("")
        lines.append(f"{len(summary['missing'])} job(s) with no result yet:")
        lines.extend(f"  {name}" for name in summary["missing"])
    if summary["orphans"]:
        lines.append("")
        lines.append(f"{len(summary['orphans'])} result(s) with no job in this sweep:")
        lines.extend(f"  {name}" for name in summary["orphans"])

    largest_unconverged = [r for r in summary["rows"] if r.get("converged_against_next") is False]
    if largest_unconverged:
        lines.append("")
        lines.append(
            f"{len(largest_unconverged)} window(s) are further than the threshold from the next larger "
            "active space, so the sweep does not yet support a convergence claim."
        )
    return "\n".join(lines)


def to_csv(summary: dict) -> str:
    """The active-space table as CSV, for anything that wants to plot it."""
    columns = [
        "system", "spin_family", "label", "multiplicity", "nfirst_1based", "nlast_1based",
        "n_orbitals", "n_electrons", "nalpha", "nbeta", "n_determinants", "solver",
        "energy_hartree", "energy_uncertainty_hartree", "deviation_from_next_hartree",
        "converged_against_next", "runtime_seconds", "peak_memory_gb",
        "occ_n_fractional", "occ_n_unpaired", "occ_max_deviation_from_integer",
    ]
    lines = [",".join(columns)]
    for row in summary["rows"]:
        occupations = row.get("occupations") or {}
        values = []
        for column in columns:
            if column.startswith("occ_"):
                value = occupations.get(column[4:])
            else:
                value = row.get(column)
            values.append("" if value is None else str(value))
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


# ----------------------------------------------------- cross-checks and oracles

def check_bundle_against_manifest(bundle, man: dict, repo_root=None) -> list:
    """Confirm a ``.npz`` bundle is the active space its manifest asked for.

    *bundle* is anything indexable by the bundle's key names -- the object
    ``numpy.load`` returns, or a plain dict. This reads the interchange schema
    only, so it does not import ``g16dump``, and it is the check worth running
    before a sweep's numbers are believed: a window that silently came back
    different from the one requested is invisible in the energies.
    """
    def value(key):
        try:
            item = bundle[key]
        except (KeyError, IndexError):
            return None
        return item.item() if hasattr(item, "item") and getattr(item, "ndim", None) == 0 else item

    counts = mf.active_space(man, repo_root=repo_root)
    problems = []
    for key, expected in (
        ("act_start", counts["act_start"]),
        ("act_stop", counts["act_stop"]),
        ("ncore", counts["ncore"]),
        ("nact", counts["nact"]),
        ("nelec", counts["nelec"]),
        ("nalpha", counts["nalpha"]),
        ("nbeta", counts["nbeta"]),
        ("charge", man["molecule"]["charge"]),
        ("multiplicity", man["molecule"]["multiplicity"]),
    ):
        found = value(key)
        if found is None:
            problems.append(f"bundle has no {key!r}")
        elif int(found) != int(expected):
            problems.append(f"bundle {key}={int(found)} but {man['id']} asks for {int(expected)}")

    reference = value("reference")
    if reference is not None and str(reference) != man["reference"]["type"]:
        problems.append(f"bundle reference={reference!r} but {man['id']} is {man['reference']['type']!r}")

    fock_source = value("fock_source")
    if fock_source is not None:
        fock_source = str(fock_source)
        if man["reference"]["type"] in mf.KS_REFERENCE_TYPES and fock_source != "pyscf_rebuilt":
            problems.append(
                f"bundle fock_source={fock_source!r} for a Kohn-Sham reference; the stored matrix is "
                "not the HF Fock operator, so this must be 'pyscf_rebuilt'"
            )
        elif man["reference"].get("fock_source") == "pyscf_rebuilt" and fock_source == "gaussian":
            problems.append(f"bundle used the stored Fock matrix but {man['id']} asks for the rebuilt one")

    eri = value("eri_act")
    if eri is not None and getattr(eri, "shape", None) is not None:
        expected_shape = (counts["nact"],) * 4
        if tuple(eri.shape) != expected_shape:
            problems.append(f"bundle eri_act has shape {tuple(eri.shape)}, expected {expected_shape}")
    return problems


def fci_energy_from_fcidump(path, nroots: int = 1):
    """Ground-state FCI energy of a written FCIDUMP, via PySCF.

    The validation solver for a sweep step small enough to diagonalise exactly:
    no Dice, no Block2, and no trust required. PySCF is imported here and nowhere
    else in this module, so the rest of the analysis runs without it.
    """
    try:
        from pyscf import ao2mo, fci  # noqa: PLC0415 -- optional dependency
        from pyscf.tools import fcidump  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise AnalysisError(
            "pyscf is needed to diagonalise an FCIDUMP; install it with "
            "`pip install -e '.[validate]'`"
        ) from exc

    data = fcidump.read(str(path))
    norb = int(data["NORB"])
    nelec = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    nalpha = (nelec + ms2) // 2
    nbeta = nelec - nalpha
    eri = ao2mo.restore(1, np.asarray(data["H2"]), norb)
    solver = fci.direct_spin1.FCI()
    energy, _ = solver.kernel(np.asarray(data["H1"]), eri, norb, (nalpha, nbeta), nroots=nroots)
    energy = np.atleast_1d(energy) + float(data["ECORE"])
    return float(energy[0]) if nroots == 1 else [float(v) for v in energy]


# ------------------------------------------------------------------------ CLI

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks analyze",
        description="Collect an active-space sweep's results and compare its windows.",
    )
    parser.add_argument("sweep", help="the sweep directory, or its sweep.json")
    parser.add_argument("--results", required=True, help="directory of result records, one JSON per job")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_CONVERGENCE_HARTREE,
                        help="Ha between adjacent windows below which a window counts as converged")
    parser.add_argument("--json", default=None, help="also write the full summary here")
    parser.add_argument("--csv", default=None, help="also write the active-space table here")
    parser.add_argument("--strict", action="store_true", help="fail if any job has no result")
    args = parser.parse_args(argv)

    summary = summarize(load_sweep(args.sweep), load_results(args.results),
                        tolerance=args.tolerance, strict=args.strict)
    print(format_report(summary))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.csv:
        Path(args.csv).write_text(to_csv(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
