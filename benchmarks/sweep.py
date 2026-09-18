"""Active-space sweeps: turn one benchmark entry into a series of Gaussian jobs.

An active-space calculation is only as good as the argument that the active
space was big enough, and the only honest form of that argument is a sweep: run
the same system in a series of nested windows and show the quantity of interest
stop moving. The Ni-PAP work is the reason this module exists -- correlation
there reaches past the nominal Ni 3d orbitals into the porphyrin framework, and
state ordering moved when the active space was expanded, so a single window is
not evidence of anything.

What this module does:

* expands one benchmark manifest into one derived manifest per window, each with
  an *explicit* window, so every derived job is itself a valid manifest and goes
  through exactly the same validator;
* renders the Gaussian input for each from the committed templates in
  ``gaussian/``, so the route section stays defined in one place and a fix there
  reaches every sweep;
* writes a sweep manifest recording every active space it generated, which is
  what :mod:`benchmarks.analyze` reads back.

Windows are always the 1-based inclusive Gaussian numbers. Two rules are
enforced rather than documented, because breaking either silently produces
numbers that look fine:

* **A sweep must be nested.** Comparing a window to the next larger one is only
  a convergence statement if the smaller is contained in the larger.
* **Every spin state of a family gets the same window.** A splitting between two
  different active spaces is not a splitting.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import date
from pathlib import Path

from . import manifest as mf

#: Which committed template each reference type is rendered from. Kohn-Sham
#: references reuse the Hartree-Fock templates with the method token swapped;
#: everything else about the route, the window keyword included, stays in
#: gaussian/ so there is one place to fix once the route is verified.
TEMPLATE_FOR_REFERENCE = {
    "RHF": "rhf_window.gjf",
    "RKS": "rhf_window.gjf",
    "ROHF": "rohf_window.gjf",
    "ROKS": "rohf_window.gjf",
}

_ROUTE_METHOD = re.compile(r"(#P\s+)(\S+?)(/BASIS)")
_LEFTOVER_PLACEHOLDER = re.compile(r"\b(NAME|BASIS|CHARGE|MULTIPLICITY|GEOMETRY_HERE|NFIRST|NLAST)\b")

#: Placeholders a window template must contain before it is filled in. NFIRST and
#: NLAST are the important ones: a template edited into a state where the window
#: keyword is gone would still render, and every job would silently transform the
#: full MO space instead of the window -- days of queue time for the wrong answer.
REQUIRED_PLACEHOLDERS = ("NAME", "BASIS", "CHARGE", "GEOMETRY_HERE", "NFIRST", "NLAST")


class SweepError(ValueError):
    """Raised when a sweep is not a sweep: unnested windows, mismatched states."""


# ----------------------------------------------------------------- the windows

def windows_from_policy(man: dict, homo_1based=None, nbasis=None):
    """Expand a manifest's ``window_policy`` into ``[(label, nfirst, nlast), ...]``.

    Uses the policy's ``sweep`` steps when it has them, and falls back to the
    policy's own single ``n_below``/``n_above`` when it does not.
    """
    policy = (man.get("active_space") or {}).get("window_policy")
    if policy is None:
        window = (man.get("active_space") or {}).get("window")
        if window is None:
            raise SweepError(f"{man.get('id')}: no window and no window_policy to sweep")
        return [("as declared", int(window["nfirst_1based"]), int(window["nlast_1based"]))]

    steps = policy.get("sweep") or [{"n_below": policy["n_below"], "n_above": policy["n_above"]}]
    windows = []
    for index, step in enumerate(steps):
        probe = copy.deepcopy(man)
        probe["active_space"]["window_policy"] = {
            "reference_orbital": policy["reference_orbital"],
            "n_below": step["n_below"],
            "n_above": step["n_above"],
        }
        nfirst, nlast = mf.resolve_window(probe, homo_1based=homo_1based, nbasis=nbasis)
        label = step.get("label") or f"step {index + 1}"
        windows.append((label, nfirst, nlast))
    return windows


def parse_windows(text: str):
    """Parse ``"40-50,38-52"`` into ``[("40-50", 40, 50), ("38-52", 38, 52)]``."""
    windows = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.fullmatch(r"(\d+)\s*[-:]\s*(\d+)", chunk)
        if not match:
            raise SweepError(f"{chunk!r} is not a window; write it as NFIRST-NLAST")
        nfirst, nlast = int(match.group(1)), int(match.group(2))
        mf.bundle_window(nfirst, nlast)  # reuse the one place windows are checked
        windows.append((chunk, nfirst, nlast))
    if not windows:
        raise SweepError("no windows given")
    return windows


def check_nested(windows):
    """Raise unless each window contains the one before it."""
    ordered = sorted(windows, key=lambda w: w[2] - w[1])
    for (small_label, small_first, small_last), (big_label, big_first, big_last) in zip(ordered, ordered[1:]):
        if big_first > small_first or big_last < small_last:
            raise SweepError(
                f"window {small_first}-{small_last} ({small_label}) is not contained in "
                f"{big_first}-{big_last} ({big_label}); a sweep of unnested windows cannot "
                "support a convergence claim"
            )
    return ordered


# ------------------------------------------------------------------ expansion

def derived_id(base_id: str, nfirst: int, nlast: int) -> str:
    return f"{base_id}_w{nfirst}_{nlast}"


def expand(man: dict, windows, repo_root=None) -> list:
    """One manifest plus N windows -> N derived manifests with explicit windows.

    The derived manifests inline their geometry, so a sweep directory is
    self-contained and can be copied to a cluster on its own.
    """
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    atoms = mf.geometry_atoms(man, repo_root)
    inline = mf.xyz_text(atoms, f"inlined from {man['id']}").split("\n", 2)[2].rstrip("\n")

    derived = []
    for label, nfirst, nlast in windows:
        child = copy.deepcopy(man)
        child.pop("_path", None)
        child["id"] = derived_id(man["id"], nfirst, nlast)
        child["title"] = f"{man.get('title', man['id'])}, window {nfirst}-{nlast}"
        child["description"] = (
            f"Sweep step '{label}' of {man['id']}. "
            + man.get("description", "")
        ).strip()

        geometry = dict(child["molecule"]["geometry"])
        geometry.pop("file", None)
        geometry["xyz"] = inline
        geometry["comment"] = f"Inlined from {man['id']}; see that entry for provenance."
        child["molecule"]["geometry"] = geometry

        counts = mf.active_space(man, nfirst, nlast, repo_root=repo_root)
        child["active_space"] = {
            "window": {"nfirst_1based": nfirst, "nlast_1based": nlast},
            "frozen_core": counts["ncore"],
            "n_orbitals": counts["nact"],
            "n_electrons": counts["nelec_act"],
            "ms2": counts["ms2"],
            "notes": f"Generated by benchmarks/sweep.py from the '{label}' step of {man['id']}.",
        }

        provenance = dict(child.get("provenance") or {})
        provenance["notes"] = (
            f"Derived from benchmarks/systems/{man['id']}.json on {date.today().isoformat()}; "
            f"sweep step '{label}'. " + (provenance.get("notes") or "")
        ).strip()
        child["provenance"] = provenance

        derived.append((label, counts, child))
    return derived


# ------------------------------------------------------------------ rendering

def method_token(reference: dict) -> str:
    """The Gaussian method keyword for a reference type."""
    ref_type = reference["type"]
    if ref_type == "RHF":
        return "HF"
    if ref_type == "ROHF":
        return "ROHF"
    functional = reference.get("functional")
    if not functional:
        raise SweepError(f"{ref_type} needs a functional to build a route line")
    return functional if ref_type == "RKS" else f"RO{functional}"


def basis_token(reference: dict) -> str:
    """The Gaussian basis keyword, or ``Gen`` when the basis is per-element."""
    basis = reference["basis"]
    if isinstance(basis, dict):
        # A per-element basis needs a Gen section this module does not write; say
        # so rather than emitting a route that will fail on the cluster.
        raise SweepError(
            "per-element bases need a Gen basis section, which sweep.py does not generate. "
            "Render this job by hand, or give the entry a single basis keyword."
        )
    return basis


def render_input(man: dict, nfirst: int, nlast: int, name: str, template: str, repo_root=None) -> str:
    """Fill a ``gaussian/*_window.gjf`` template for one derived job."""
    reference = man["reference"]
    molecule = man["molecule"]
    atoms = mf.geometry_atoms(man, repo_root)

    missing = [p for p in REQUIRED_PLACEHOLDERS if p not in template]
    if missing:
        raise SweepError(
            f"the route template is missing {', '.join(missing)}; it cannot produce a windowed job"
        )

    text = _ROUTE_METHOD.sub(lambda m: m.group(1) + method_token(reference) + m.group(3), template, count=1)
    # Longest placeholders first, so MULTIPLICITY is consumed before CHARGE can
    # match inside a line that holds both.
    for placeholder, value in (
        ("GEOMETRY_HERE", mf.gaussian_geometry_block(atoms)),
        ("MULTIPLICITY", str(molecule["multiplicity"])),
        ("NFIRST", str(nfirst)),
        ("NLAST", str(nlast)),
        ("CHARGE", str(molecule["charge"])),
        ("BASIS", basis_token(reference)),
        ("NAME", name),
    ):
        text = text.replace(placeholder, value)

    leftover = _LEFTOVER_PLACEHOLDER.search(text)
    if leftover:
        raise SweepError(f"template placeholder {leftover.group(0)!r} was not substituted")
    return text


def render_job_script(template: str, name: str) -> str:
    """Fill ``gaussian/job_template.sh`` for one derived job."""
    return template.replace("#PBS -N GJOB", f"#PBS -N {name}").replace('"${NAME:-CHANGEME}"', f'"${{NAME:-{name}}}"')


def templates_dir(repo_root=None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    return root / "gaussian"


# ------------------------------------------------------------------ generation

def generate(manifests, windows, out_dir, repo_root=None, job_template=True):
    """Write the sweep: a ``.gjf`` and a manifest per (system, window).

    *manifests* is one entry, or every spin state of one family. Each gets the
    same windows, which is what makes the spin-state splittings in
    :mod:`benchmarks.analyze` comparisons of one Hamiltonian against another.
    """
    manifests = list(manifests)
    if not manifests:
        raise SweepError("no manifests to sweep")
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    out_dir = Path(out_dir)

    families = {m.get("spin_family") for m in manifests}
    if len(families) > 1:
        raise SweepError(
            f"cannot sweep {sorted(families)} together: a sweep applies one set of windows, and "
            "windows are only comparable within a spin family"
        )
    for man in manifests:
        if man.get("status") == "blocked":
            raise SweepError(f"{man['id']} is blocked: {man.get('blocked_on')}")

    ordered = check_nested(windows)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = out_dir / "manifests"
    manifests_dir.mkdir(exist_ok=True)

    job_template_text = None
    if job_template:
        path = templates_dir(repo_root) / "job_template.sh"
        if path.is_file():
            job_template_text = path.read_text(encoding="utf-8")

    jobs = []
    for man in manifests:
        template_name = TEMPLATE_FOR_REFERENCE[man["reference"]["type"]]
        template_text = (templates_dir(repo_root) / template_name).read_text(encoding="utf-8")

        for label, counts, child in expand(man, ordered, repo_root=repo_root):
            problems = mf.validate(child, repo_root=repo_root)
            if problems:
                raise SweepError(
                    f"{child['id']}: the generated manifest does not validate\n  - "
                    + "\n  - ".join(problems)
                )
            name = child["id"]
            (out_dir / f"{name}.gjf").write_text(
                render_input(child, counts["nfirst_1based"], counts["nlast_1based"], name,
                             template_text, repo_root=repo_root),
                encoding="utf-8",
            )
            (manifests_dir / f"{name}.json").write_text(
                json.dumps(child, indent=2) + "\n", encoding="utf-8"
            )
            if job_template_text is not None:
                (out_dir / f"{name}.pbs").write_text(render_job_script(job_template_text, name), encoding="utf-8")

            jobs.append({
                "name": name,
                "system": man["id"],
                "spin_family": man.get("spin_family"),
                "label": label,
                "multiplicity": man["molecule"]["multiplicity"],
                "reference": man["reference"]["type"],
                "route_template": f"gaussian/{template_name}",
                "gjf": f"{name}.gjf",
                "manifest": f"manifests/{name}.json",
                "active_space": counts,
            })

    sweep = {
        "schema_version": mf.SCHEMA_VERSION,
        "kind": "active_space_sweep",
        "generated": date.today().isoformat(),
        "spin_family": manifests[0].get("spin_family"),
        "systems": [m["id"] for m in manifests],
        "windows": [{"label": label, "nfirst_1based": nfirst, "nlast_1based": nlast}
                    for label, nfirst, nlast in ordered],
        "jobs": jobs,
        "notes": (
            "Windows are the 1-based inclusive numbers in each route's Window=(NFIRST,NLAST). "
            "Every job here is a full benchmark manifest in manifests/ and validates against "
            "benchmarks/manifest.py. Run each .gjf, formchk the .chk, then g16dump extract."
        ),
    }
    (out_dir / "sweep.json").write_text(json.dumps(sweep, indent=2) + "\n", encoding="utf-8")
    return sweep


# ------------------------------------------------------------------------ CLI

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks sweep",
        description="Generate the Gaussian jobs for an active-space sweep.",
    )
    parser.add_argument("system", nargs="+", help="manifest id(s) from benchmarks/systems/, or paths to manifests")
    parser.add_argument("--out", required=True, help="directory to write the sweep into")
    parser.add_argument("--homo", type=int, default=None,
                        help="1-based index of the HOMO from a preliminary SCF; required by any "
                             "entry whose window_policy is relative to it")
    parser.add_argument("--nbasis", type=int, default=None, help="orbital count, so a window past the end is caught here")
    parser.add_argument("--windows", default=None,
                        help="explicit windows as NFIRST-NLAST,NFIRST-NLAST,... , overriding the entry's sweep steps")
    parser.add_argument("--no-job-script", action="store_true", help="write only the .gjf files")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parent.parent

    manifests = []
    for name in args.system:
        path = Path(name)
        if not path.is_file():
            path = mf.systems_dir() / f"{name}.json"
        manifests.append(mf.check(mf.load(path), repo_root=repo_root))

    if args.windows:
        windows = parse_windows(args.windows)
    else:
        windows = windows_from_policy(manifests[0], homo_1based=args.homo, nbasis=args.nbasis)
        for man in manifests[1:]:
            other = windows_from_policy(man, homo_1based=args.homo, nbasis=args.nbasis)
            if [w[1:] for w in other] != [w[1:] for w in windows]:
                raise SweepError(
                    f"{man['id']} and {manifests[0]['id']} expand to different windows; give "
                    "--windows explicitly so both spin states share an active space"
                )

    sweep = generate(manifests, windows, args.out, repo_root=repo_root, job_template=not args.no_job_script)
    print(f"{len(sweep['jobs'])} job(s) in {args.out}")
    for job in sweep["jobs"]:
        space = job["active_space"]
        print(f"  {job['name']:<40s} {space['nact']:>3d} orbitals, "
              f"{space['nelec_act']:>3d} electrons ({space['nalpha_act']}a/{space['nbeta_act']}b)  [{job['label']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
