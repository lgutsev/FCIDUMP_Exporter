"""Benchmark manifests: one JSON file describing a set of systems to run.

A manifest entry fully specifies a calculation -- geometry, charge,
multiplicity, reference, basis, active window, frozen core, solver and
convergence parameters -- so a benchmark is reproducible from the file alone
rather than from a directory of hand-edited inputs.

JSON rather than YAML, so the core stays NumPy-only.

Validation here is real, not schema box-ticking: the electron count is computed
from the geometry and the charge, and checked against the multiplicity. A
manifest asking for a triplet with an odd number of electrons is rejected before
anything is submitted to a queue.
"""

from __future__ import annotations

import json
from typing import Optional

from .errors import ValidationError, require

MANIFEST_VERSION = 1

#: Atomic numbers through Kr -- enough for organic ligands and the first-row
#: transition metals this workflow targets.
ELEMENTS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
}

REQUIRED_KEYS = (
    "name",
    "geometry",
    "charge",
    "multiplicity",
    "reference",
    "basis",
    "window",
)

VALID_REFERENCES = ("RHF", "ROHF", "RKS", "ROKS")


def count_electrons(geometry: str, charge: int) -> int:
    """Total electrons from a Cartesian geometry block and a charge."""
    total = 0
    for line in geometry.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        symbol = parts[0].strip().capitalize()
        require(
            symbol in ELEMENTS,
            f"unknown element {parts[0]!r} in the geometry. Known elements run "
            f"from H to Kr; add it to manifest.ELEMENTS if you need it.",
        )
        total += ELEMENTS[symbol]
    return total - charge


def validate_system(system: dict) -> dict:
    """Check one manifest entry. Returns derived quantities."""
    name = system.get("name", "<unnamed>")
    missing = [key for key in REQUIRED_KEYS if key not in system]
    require(
        not missing,
        f"system {name!r} is missing required keys: {missing}. Required: "
        f"{list(REQUIRED_KEYS)}.",
    )

    reference = system["reference"]
    require(
        reference in VALID_REFERENCES,
        f"system {name!r} has reference {reference!r}; expected one of "
        f"{', '.join(VALID_REFERENCES)}. UHF references are out of scope for "
        f"v1 -- use UHF natural orbitals through a restricted reference.",
    )

    multiplicity = int(system["multiplicity"])
    require(
        multiplicity >= 1,
        f"system {name!r} has multiplicity {multiplicity}; it must be >= 1.",
    )

    # A system may be fully specified except for its geometry -- a structure
    # that still needs optimizing. Everything that does not depend on the atom
    # list is still checked; the electron-count and window checks are reported
    # as deferred rather than silently skipped.
    placeholder = system.get("geometry_status") == "placeholder"
    if placeholder and "nelec" not in system:
        require(
            "window" in system,
            f"system {name!r} is a placeholder but still needs a window.",
        )
        first, last = int(system["window"][0]), int(system["window"][1])
        require(
            1 <= first <= last,
            f"system {name!r} has window [{first}, {last}], which is empty or "
            f"starts below orbital 1.",
        )
        return {
            "name": name,
            "nelec": None,
            "nalpha": None,
            "nbeta": None,
            "ncore": first - 1,
            "nact": last - first + 1,
            "nelec_act": None,
            "ms2": multiplicity - 1,
            "window": [first, last],
            "validation_deferred": "geometry is a placeholder",
        }

    nelec = (
        int(system["nelec"])
        if "nelec" in system
        else count_electrons(system["geometry"], int(system["charge"]))
    )
    unpaired = multiplicity - 1
    require(
        nelec >= unpaired and (nelec - unpaired) % 2 == 0,
        f"system {name!r} is impossible: {nelec} electrons cannot give "
        f"multiplicity {multiplicity}. An even electron count needs an odd "
        f"multiplicity and vice versa.",
    )
    nalpha = (nelec + unpaired) // 2
    nbeta = nelec - nalpha

    window = system["window"]
    require(
        isinstance(window, (list, tuple)) and len(window) == 2,
        f"system {name!r} has window {window!r}; expected [first, last] as "
        f"1-based MO indices.",
    )
    first, last = int(window[0]), int(window[1])
    require(
        1 <= first <= last,
        f"system {name!r} has window [{first}, {last}], which is empty or "
        f"starts below orbital 1.",
    )

    ncore = first - 1
    nact = last - first + 1
    require(
        ncore <= nbeta,
        f"system {name!r} freezes {ncore} core orbitals but has only {nbeta} "
        f"beta electrons; the frozen core must be doubly occupied.",
    )
    require(
        nalpha - ncore <= nact,
        f"system {name!r} has an active window of {nact} orbitals but the "
        f"reference needs {nalpha - ncore} occupied alpha orbitals inside it.",
    )

    if reference in ("RKS", "ROKS"):
        require(
            "xc" in system,
            f"system {name!r} uses a Kohn-Sham reference but names no "
            f"functional; add an 'xc' key.",
        )

    return {
        "name": name,
        "nelec": nelec,
        "nalpha": nalpha,
        "nbeta": nbeta,
        "ncore": ncore,
        "nact": nact,
        "nelec_act": (nalpha - ncore) + (nbeta - ncore),
        "ms2": nalpha - nbeta,
        "window": [first, last],
    }


def validate_manifest(manifest: dict) -> list:
    """Check a whole manifest. Returns the derived quantities per system."""
    require(
        manifest.get("manifest_version") == MANIFEST_VERSION,
        f"manifest_version is {manifest.get('manifest_version')!r}; this build "
        f"understands version {MANIFEST_VERSION}.",
    )
    systems = manifest.get("systems")
    require(
        isinstance(systems, list) and systems,
        "the manifest has no 'systems' list, or it is empty.",
    )

    names = [s.get("name") for s in systems]
    duplicates = {n for n in names if names.count(n) > 1}
    require(
        not duplicates,
        f"duplicate system names in the manifest: {sorted(duplicates)}. Names "
        f"are used for output filenames, so they must be unique.",
    )

    return [validate_system(system) for system in systems]


def load_manifest(path: str) -> dict:
    with open(path) as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def save_manifest(manifest: dict, path: str) -> None:
    validate_manifest(manifest)
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


# ------------------------------------------------------- Gaussian generation


def gaussian_input(system: dict, *, memory: str = "8GB", nproc: int = 8) -> str:
    """The ``.gjf`` for one manifest entry.

    The route section carries the keywords the workflow needs and nothing else:
    ``Output=MatrixElement`` to write the ``.mat``, ``Tran=Full`` with
    ``Window=`` so only the active block is transformed, ``NoSymm`` because the
    ``.fch`` orbital transfer assumes no symmetry reordering,
    ``Int=NoBasisTransform`` so the ``.mat`` and ``.fch`` share an AO set,
    ``5D 7F`` because g16dump refuses Cartesian d/f functions (their ordering
    and normalization differ from PySCF's; see aoorder.py), and
    tight SCF convergence because the E_ref gate is at 1e-8 Ha.

    The route remains UNVERIFIED until a real Gaussian job has run -- see
    gaussian/README.md.
    """
    derived = validate_system(system)
    name = system["name"]
    first, last = derived["window"]

    method = system["reference"]
    if method in ("RKS", "ROKS"):
        prefix = "RO" if system["multiplicity"] > 1 else "R"
        method = f"{prefix}{system['xc']}"

    lines = [
        f"%chk={name}.chk",
        f"%mem={memory}",
        f"%nprocshared={nproc}",
        f"#P {method}/{system['basis']} 5D 7F NoSymm Int=NoBasisTransform "
        f"SCF=(Conver=10)",
        f"# Output=MatrixElement Tran=Full Window=({first},{last})",
        "",
        f"{name}: {derived['nelec_act']} electrons in {derived['nact']} "
        f"orbitals, {derived['ncore']} frozen core",
        "",
        f"{system['charge']} {system['multiplicity']}",
    ]
    lines.extend(
        line.strip() for line in system["geometry"].strip().splitlines() if line.strip()
    )
    lines += ["", f"{name}.mat", ""]
    return "\n".join(lines)


def write_gaussian_inputs(manifest: dict, outdir: str, **kwargs) -> list:
    """Write one ``.gjf`` per system. Returns the paths written."""
    import os

    validate_manifest(manifest)
    os.makedirs(outdir, exist_ok=True)

    written = []
    for system in manifest["systems"]:
        path = os.path.join(outdir, f"{system['name']}.gjf")
        with open(path, "w") as handle:
            handle.write(gaussian_input(system, **kwargs))
        written.append(path)
    return written


def summary_table(manifest: dict) -> str:
    """A readable table of what a manifest will run."""
    derived = validate_manifest(manifest)
    header = (
        f"{'system':28s} {'ref':6s} {'basis':12s} {'mult':>4s} "
        f"{'ncore':>5s} {'nact':>5s} {'nelec_act':>9s} {'MS2':>4s}"
    )
    rows = [header, "-" * len(header)]
    for system, d in zip(manifest["systems"], derived):
        nelec_act = "?" if d["nelec_act"] is None else str(d["nelec_act"])
        rows.append(
            f"{d['name']:28s} {system['reference']:6s} {system['basis']:12s} "
            f"{system['multiplicity']:4d} {d['ncore']:5d} {d['nact']:5d} "
            f"{nelec_act:>9s} {d['ms2']:4d}"
            + ("   (geometry placeholder)" if "validation_deferred" in d else "")
        )
    return "\n".join(rows)
