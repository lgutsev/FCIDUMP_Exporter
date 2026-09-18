"""The benchmark manifest format: load, resolve, validate.

A *manifest* is one JSON file describing one Gaussian job that the workflow can
run end to end: geometry, charge, multiplicity, reference type, basis, active
window, frozen core, solver and its convergence parameters. One file is one
(molecule, spin state, basis, window) combination, because that is exactly what
one Gaussian job is. Related spin states are tied together by ``spin_family`` so
that :mod:`benchmarks.analyze` can form spin-state splittings.

Three conventions, fixed here so they are converted in exactly one place:

* **Windows are the 1-based inclusive numbers you write in a Gaussian route**,
  which is why the fields are called ``nfirst_1based`` / ``nlast_1based``. The
  ``.npz`` bundle stores a 0-based half-open window instead; :func:`bundle_window`
  is the only conversion.
* **Geometries are Cartesian, in angstrom unless ``units`` says otherwise**, and
  are either inline or in a sibling ``.xyz`` file.
* **Electron counts are never taken on trust.** They are recomputed from the
  nuclear charges and the molecular charge, and a manifest that disagrees with
  its own geometry is rejected.

Validation collects *every* problem it finds rather than raising on the first,
because a manifest is usually edited by hand and one error per run is a slow way
to fix six.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

SCHEMA_VERSION = "1.0"

#: Reference types the workflow understands. Kohn-Sham types are listed because
#: they are legitimate *orbital sources*; the Hamiltonian built from them is
#: still the HF Hamiltonian, which is why they are constrained below.
REFERENCE_TYPES = ("RHF", "ROHF", "RKS", "ROKS")
KS_REFERENCE_TYPES = ("RKS", "ROKS")

#: Where the Fock matrix used by ``g16dump`` comes from. ``gaussian`` means the
#: matrix stored in the ``.mat``; ``pyscf_rebuilt`` means rebuilt as
#: ``F_sigma = Hcore + J[Pa + Pb] - K[Psigma]`` from the Gaussian orbitals.
FOCK_SOURCES = ("gaussian", "pyscf_rebuilt")

SOLVERS = ("pyscf_fci", "dice", "block2", "none")

#: ``ready`` means every input needed to run it is in this repository.
#: ``blocked`` means something is genuinely missing and ``blocked_on`` says what.
STATUSES = ("ready", "blocked")

#: What the entry is for. Used to order the suite and to decide what CI may run.
ROLES = ("validation", "ligand_field", "macrocycle", "regression", "stress")

UNITS = ("angstrom", "bohr")

BOHR_PER_ANGSTROM = 1.8897261246257702

_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Nuclear charges. Only through the first transition series plus the light
#: main-group elements -- enough for everything this workflow targets, and a
#: missing symbol is an error rather than a silent zero.
ELEMENTS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
}

#: Below this separation two nuclei are a typo, not a molecule.
MIN_NUCLEAR_SEPARATION_ANGSTROM = 0.5

#: Determinant count above which ``pyscf_fci`` is not a realistic solver choice.
#: Chosen so a validation entry that quietly grew past tractability is caught by
#: the validator rather than by an out-of-memory kill hours later.
MAX_FCI_DETERMINANTS = 5_000_000


class ManifestError(ValueError):
    """Raised by :func:`check` with every problem found, one per line."""


# --------------------------------------------------------------------- loading

def load(path) -> dict:
    """Read one manifest.

    JSON is the on-disk format. YAML is accepted when PyYAML happens to be
    installed, because it is pleasant to hand-edit, but nothing in this
    repository requires it: the committed suite is JSON so that the core
    dependency stays numpy-only.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415 -- optional, imported lazily on purpose
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ManifestError(
                f"{path} is YAML but PyYAML is not installed. Convert it to JSON, "
                "or `pip install pyyaml`."
            ) from exc
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"{path} must contain a JSON object, got {type(data).__name__}")
    data.setdefault("_path", str(path))
    return data


def load_all(directory) -> list:
    """Load every manifest in *directory*, sorted by ``tier`` then ``id``."""
    directory = Path(directory)
    manifests = [load(p) for p in sorted(directory.glob("*.json"))]
    for suffix in ("*.yaml", "*.yml"):
        manifests.extend(load(p) for p in sorted(directory.glob(suffix)))
    return sorted(manifests, key=lambda m: (m.get("tier", 99), m.get("id", "")))


def systems_dir() -> Path:
    """The committed suite, resolved relative to this file rather than to cwd."""
    return Path(__file__).resolve().parent / "systems"


# ------------------------------------------------------------------- geometry

def parse_xyz(text: str) -> list:
    """Parse an XYZ file body into ``[(symbol, x, y, z), ...]``.

    Accepts both a real XYZ file (count line, comment line, atoms) and a bare
    block of atom lines, because the bare form is what a manifest carries
    inline.
    """
    lines = [ln for ln in (raw.strip() for raw in text.splitlines()) if ln and not ln.startswith("#")]
    if lines:
        first = lines[0].split()
        if len(first) == 1 and first[0].isdigit():
            count = int(first[0])
            # Drop the count line and the comment line that follows it.
            lines = lines[2:] if len(lines) > 1 else []
            if len(lines) != count:
                raise ManifestError(
                    f"XYZ header says {count} atoms but {len(lines)} atom lines follow"
                )
    atoms = []
    for lineno, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ManifestError(f"geometry line {lineno} is not 'SYMBOL X Y Z': {line!r}")
        symbol = parts[0].capitalize()
        try:
            xyz = tuple(float(v) for v in parts[1:4])
        except ValueError as exc:
            raise ManifestError(f"geometry line {lineno} has a non-numeric coordinate: {line!r}") from exc
        atoms.append((symbol,) + xyz)
    return atoms


def geometry_atoms(man: dict, repo_root=None) -> list:
    """Resolve a manifest's geometry to ``[(symbol, x, y, z), ...]``.

    ``geometry`` is either ``{"units": ..., "xyz": "<inline block>"}`` or
    ``{"units": ..., "file": "benchmarks/geometries/foo.xyz"}``. A file path is
    relative to the repository root so that a manifest reads the same wherever
    it is loaded from.
    """
    molecule = man.get("molecule") or {}
    geometry = molecule.get("geometry") or {}
    if "xyz" in geometry and "file" in geometry:
        raise ManifestError("geometry has both 'xyz' and 'file'; give exactly one")
    if "xyz" in geometry:
        return parse_xyz(geometry["xyz"])
    if "file" in geometry:
        root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
        path = root / geometry["file"]
        if not path.is_file():
            raise ManifestError(f"geometry file does not exist: {path}")
        return parse_xyz(path.read_text(encoding="utf-8"))
    raise ManifestError("geometry has neither 'xyz' nor 'file'")


def nuclear_charge(atoms) -> int:
    """Sum of nuclear charges; raises on a symbol not in :data:`ELEMENTS`."""
    total = 0
    for atom in atoms:
        symbol = atom[0]
        if symbol not in ELEMENTS:
            raise ManifestError(f"unknown element symbol {symbol!r}")
        total += ELEMENTS[symbol]
    return total


def electron_count(man: dict, repo_root=None) -> int:
    """Total electrons, computed from the geometry and charge -- never declared."""
    atoms = geometry_atoms(man, repo_root)
    return nuclear_charge(atoms) - int(man["molecule"]["charge"])


def spin_counts(nelec: int, multiplicity: int):
    """``(nalpha, nbeta)`` for a spin-restricted reference."""
    two_s = multiplicity - 1
    nbeta, remainder = divmod(nelec - two_s, 2)
    if remainder:
        raise ManifestError(
            f"{nelec} electrons cannot give multiplicity {multiplicity}: "
            "electron count and 2S+1 have incompatible parity"
        )
    return nbeta + two_s, nbeta


def gaussian_geometry_block(atoms, decimals: int = 7) -> str:
    """Format atoms as the Cartesian block of a Gaussian input."""
    width = decimals + 5
    return "\n".join(
        f"{symbol:<3s} {x:>{width}.{decimals}f} {y:>{width}.{decimals}f} {z:>{width}.{decimals}f}"
        for symbol, x, y, z in atoms
    )


def xyz_text(atoms, comment: str = "") -> str:
    """Format atoms as a standalone ``.xyz`` file."""
    body = "\n".join(
        f"{symbol:<3s} {x:>14.8f} {y:>14.8f} {z:>14.8f}" for symbol, x, y, z in atoms
    )
    return f"{len(atoms)}\n{comment}\n{body}\n"


# --------------------------------------------------------------------- windows

def bundle_window(nfirst_1based: int, nlast_1based: int):
    """Convert a Gaussian ``Window=(NFIRST,NLAST)`` to the bundle's window.

    The ``.npz`` bundle stores ``act_start`` / ``act_stop`` as a 0-based
    half-open range. This is the only place the two conventions meet.
    """
    if nfirst_1based < 1:
        raise ManifestError(f"NFIRST is 1-based and must be >= 1, got {nfirst_1based}")
    if nlast_1based < nfirst_1based:
        raise ManifestError(f"NLAST ({nlast_1based}) is below NFIRST ({nfirst_1based})")
    return nfirst_1based - 1, nlast_1based


def resolve_window(man: dict, homo_1based=None, nbasis=None):
    """Return the ``(nfirst, nlast)`` 1-based window for a manifest.

    An entry either states its window outright, or carries a ``window_policy``
    naming a reference orbital and how many orbitals to take below and above it.
    The policy form exists because for anything the size of a metalloporphyrin
    the absolute MO indices are not known until a preliminary SCF has run; the
    sweep workflow resolves the policy once ``homo_1based`` is known.
    """
    active = man.get("active_space") or {}
    window = active.get("window")
    if window is not None:
        return int(window["nfirst_1based"]), int(window["nlast_1based"])

    policy = active.get("window_policy")
    if policy is None:
        raise ManifestError("active_space has neither 'window' nor 'window_policy'")

    reference = policy["reference_orbital"]
    if isinstance(reference, str):
        if homo_1based is None:
            raise ManifestError(
                f"window_policy references the {reference.upper()}, so the window cannot be "
                "resolved until a preliminary SCF has given the HOMO index; "
                "pass homo_1based="
            )
        if reference == "homo":
            centre = int(homo_1based)
        elif reference == "lumo":
            centre = int(homo_1based) + 1
        else:
            raise ManifestError(f"unknown reference_orbital {reference!r}")
    else:
        centre = int(reference)

    nfirst = centre - int(policy["n_below"])
    nlast = centre + int(policy["n_above"])
    if nfirst < 1:
        raise ManifestError(
            f"window_policy puts NFIRST at {nfirst}; n_below={policy['n_below']} reaches "
            f"below orbital 1 from reference orbital {centre}"
        )
    if nbasis is not None and nlast > nbasis:
        raise ManifestError(
            f"window_policy puts NLAST at {nlast}, above the {nbasis} available orbitals"
        )
    return nfirst, nlast


def active_space(man: dict, nfirst=None, nlast=None, repo_root=None, homo_1based=None) -> dict:
    """Everything that follows from a manifest plus a window.

    Returned keys line up with the ``.npz`` bundle so that a bundle can be
    checked against the manifest that produced it field by field.
    """
    if nfirst is None or nlast is None:
        nfirst, nlast = resolve_window(man, homo_1based=homo_1based)
    act_start, act_stop = bundle_window(nfirst, nlast)

    multiplicity = int(man["molecule"]["multiplicity"])
    nelec = electron_count(man, repo_root)
    nalpha, nbeta = spin_counts(nelec, multiplicity)

    ncore = nfirst - 1
    nact = nlast - nfirst + 1
    nelec_act = nelec - 2 * ncore
    nalpha_act = nalpha - ncore
    nbeta_act = nbeta - ncore

    return {
        "nfirst_1based": nfirst,
        "nlast_1based": nlast,
        "act_start": act_start,
        "act_stop": act_stop,
        "ncore": ncore,
        "nact": nact,
        "nelec": nelec,
        "nalpha": nalpha,
        "nbeta": nbeta,
        "nelec_act": nelec_act,
        "nalpha_act": nalpha_act,
        "nbeta_act": nbeta_act,
        "ms2": multiplicity - 1,
        "n_determinants": n_determinants(nact, nalpha_act, nbeta_act),
    }


def n_determinants(norb: int, nalpha: int, nbeta: int) -> int:
    """Size of the FCI space, used to keep ``pyscf_fci`` entries honest."""
    if min(norb, nalpha, nbeta) < 0 or nalpha > norb or nbeta > norb:
        return 0
    return math.comb(norb, nalpha) * math.comb(norb, nbeta)


# ------------------------------------------------------------------ validation

_TOP_LEVEL = {
    "schema_version", "id", "title", "status", "role", "tier", "description",
    "spin_family", "molecule", "reference", "active_space", "solver",
    "expected", "provenance", "blocked_on", "_path",
}
_MOLECULE = {"geometry", "charge", "multiplicity"}
_GEOMETRY = {"units", "xyz", "file", "comment"}
_REFERENCE = {
    "type", "functional", "basis", "spherical_d", "expected_nbasis",
    "fock_source", "scf", "orbital_transform",
}
_ACTIVE = {"window", "window_policy", "frozen_core", "n_orbitals", "n_electrons", "ms2", "notes"}
_WINDOW = {"nfirst_1based", "nlast_1based"}
_POLICY = {"reference_orbital", "n_below", "n_above", "sweep"}
_SOLVER = {"kind", "parameters", "convergence", "notes"}
_EXPECTED = {
    "scf_energy_hartree", "e_core_hartree", "reference_energy_hartree",
    "total_energy_hartree", "tolerance_hartree", "source", "trusted", "notes",
}
_PROVENANCE = {"added", "geometry_source", "basis_source", "notes", "artifacts"}


def _unknown(problems, where, mapping, allowed):
    for key in sorted(set(mapping) - allowed):
        problems.append(f"{where}: unknown key {key!r}")


def _require(problems, where, mapping, keys):
    for key in keys:
        if key not in mapping:
            problems.append(f"{where}: missing required key {key!r}")


def validate(man: dict, repo_root=None) -> list:
    """Return a list of problems. An empty list means the manifest is usable."""
    problems: list = []

    _require(problems, "manifest", man, ["schema_version", "id", "status", "role", "molecule", "reference", "active_space"])
    _unknown(problems, "manifest", man, _TOP_LEVEL)

    if man.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"manifest: schema_version is {man.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )

    ident = man.get("id")
    if isinstance(ident, str):
        if not _ID_RE.match(ident):
            problems.append(f"manifest: id {ident!r} must be lowercase words joined by underscores")
        path = man.get("_path")
        if path and Path(path).stem != ident:
            problems.append(f"manifest: id {ident!r} does not match filename {Path(path).name!r}")
    elif ident is not None:
        problems.append("manifest: id must be a string")

    status = man.get("status")
    if status not in STATUSES:
        problems.append(f"manifest: status must be one of {STATUSES}, got {status!r}")
    if status == "blocked" and not man.get("blocked_on"):
        problems.append("manifest: status is 'blocked' but 'blocked_on' does not say what is missing")
    if status == "ready" and man.get("blocked_on"):
        problems.append("manifest: status is 'ready' but 'blocked_on' is set")

    if man.get("role") not in ROLES:
        problems.append(f"manifest: role must be one of {ROLES}, got {man.get('role')!r}")
    if not isinstance(man.get("tier", 0), int):
        problems.append("manifest: tier must be an integer")

    atoms = _validate_molecule(problems, man, repo_root)
    _validate_reference(problems, man, atoms)
    _validate_active_space(problems, man, atoms, repo_root)
    _validate_solver(problems, man, atoms, repo_root)
    _validate_expected(problems, man)
    _validate_provenance(problems, man)
    return problems


def _validate_molecule(problems, man, repo_root):
    molecule = man.get("molecule")
    if not isinstance(molecule, dict):
        problems.append("molecule: must be an object")
        return None
    _require(problems, "molecule", molecule, ["geometry", "charge", "multiplicity"])
    _unknown(problems, "molecule", molecule, _MOLECULE)

    charge = molecule.get("charge")
    if not isinstance(charge, int) or isinstance(charge, bool):
        problems.append(f"molecule: charge must be an integer, got {charge!r}")
    multiplicity = molecule.get("multiplicity")
    if not isinstance(multiplicity, int) or isinstance(multiplicity, bool) or multiplicity < 1:
        problems.append(f"molecule: multiplicity must be an integer >= 1, got {multiplicity!r}")

    geometry = molecule.get("geometry")
    if not isinstance(geometry, dict):
        problems.append("geometry: must be an object")
        return None
    _unknown(problems, "geometry", geometry, _GEOMETRY)
    units = geometry.get("units", "angstrom")
    if units not in UNITS:
        problems.append(f"geometry: units must be one of {UNITS}, got {units!r}")

    if man.get("status") == "blocked" and not ("xyz" in geometry or "file" in geometry):
        # A blocked entry is allowed to have no geometry yet; that is often what
        # it is blocked on. Everything downstream of the geometry is skipped.
        return None

    try:
        atoms = geometry_atoms(man, repo_root)
    except ManifestError as exc:
        problems.append(f"geometry: {exc}")
        return None
    if not atoms:
        problems.append("geometry: no atoms")
        return None

    try:
        nuclear_charge(atoms)
    except ManifestError as exc:
        problems.append(f"geometry: {exc}")
        return None

    scale = 1.0 / BOHR_PER_ANGSTROM if units == "bohr" else 1.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dx = (atoms[i][1] - atoms[j][1]) * scale
            dy = (atoms[i][2] - atoms[j][2]) * scale
            dz = (atoms[i][3] - atoms[j][3]) * scale
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance < MIN_NUCLEAR_SEPARATION_ANGSTROM:
                problems.append(
                    f"geometry: atoms {i + 1} ({atoms[i][0]}) and {j + 1} ({atoms[j][0]}) are "
                    f"{distance:.3f} A apart, below the {MIN_NUCLEAR_SEPARATION_ANGSTROM} A sanity floor"
                )

    if isinstance(charge, int) and isinstance(multiplicity, int) and multiplicity >= 1:
        nelec = nuclear_charge(atoms) - charge
        if nelec < 0:
            problems.append(f"molecule: charge {charge:+d} leaves {nelec} electrons")
        else:
            try:
                spin_counts(nelec, multiplicity)
            except ManifestError as exc:
                problems.append(f"molecule: {exc}")
    return atoms


def _validate_reference(problems, man, atoms):
    reference = man.get("reference")
    if not isinstance(reference, dict):
        problems.append("reference: must be an object")
        return
    _require(problems, "reference", reference, ["type", "basis", "fock_source"])
    _unknown(problems, "reference", reference, _REFERENCE)

    ref_type = reference.get("type")
    if ref_type not in REFERENCE_TYPES:
        problems.append(f"reference: type must be one of {REFERENCE_TYPES}, got {ref_type!r}")

    functional = reference.get("functional")
    if ref_type in KS_REFERENCE_TYPES and not functional:
        problems.append(f"reference: {ref_type} needs a 'functional'")
    if ref_type in ("RHF", "ROHF") and functional:
        problems.append(f"reference: {ref_type} is Hartree-Fock but a functional {functional!r} is set")

    fock_source = reference.get("fock_source")
    if fock_source not in FOCK_SOURCES:
        problems.append(f"reference: fock_source must be one of {FOCK_SOURCES}, got {fock_source!r}")
    elif ref_type in KS_REFERENCE_TYPES and fock_source != "pyscf_rebuilt":
        # The whole point of the KS warning in the README: the stored KS matrix
        # carries exchange-correlation and is not the HF Fock operator.
        problems.append(
            f"reference: {ref_type} orbitals must use fock_source 'pyscf_rebuilt'; the stored "
            "Kohn-Sham matrix is not the HF Fock operator"
        )

    basis = reference.get("basis")
    if isinstance(basis, dict):
        if atoms:
            missing = sorted({a[0] for a in atoms} - set(basis))
            if missing:
                problems.append(f"reference: per-element basis is missing {', '.join(missing)}")
        extra = sorted(set(basis) - {a[0] for a in atoms}) if atoms else []
        if extra:
            problems.append(f"reference: per-element basis names absent elements {', '.join(extra)}")
    elif not isinstance(basis, str) or not basis:
        problems.append("reference: basis must be a non-empty string or a per-element object")

    nbasis = reference.get("expected_nbasis")
    if nbasis is not None and (not isinstance(nbasis, int) or nbasis < 1):
        problems.append(f"reference: expected_nbasis must be a positive integer, got {nbasis!r}")

    transform = reference.get("orbital_transform")
    if transform is not None:
        if not isinstance(transform, dict):
            problems.append("reference: orbital_transform must be an object")
        else:
            kind = transform.get("kind")
            if kind not in ("random_orthogonal",):
                problems.append(f"reference: unknown orbital_transform kind {kind!r}")
            if kind == "random_orthogonal" and not isinstance(transform.get("seed"), int):
                problems.append("reference: a random_orthogonal transform needs an integer 'seed' to be reproducible")

    scf = reference.get("scf")
    if scf is not None and not isinstance(scf, dict):
        problems.append("reference: scf must be an object")


def _validate_active_space(problems, man, atoms, repo_root):
    active = man.get("active_space")
    if not isinstance(active, dict):
        problems.append("active_space: must be an object")
        return
    _unknown(problems, "active_space", active, _ACTIVE)

    window = active.get("window")
    policy = active.get("window_policy")
    if (window is None) == (policy is None):
        problems.append("active_space: give exactly one of 'window' and 'window_policy'")

    if policy is not None:
        if not isinstance(policy, dict):
            problems.append("window_policy: must be an object")
        else:
            _require(problems, "window_policy", policy, ["reference_orbital", "n_below", "n_above"])
            _unknown(problems, "window_policy", policy, _POLICY)
            reference_orbital = policy.get("reference_orbital")
            if isinstance(reference_orbital, str):
                if reference_orbital not in ("homo", "lumo"):
                    problems.append(f"window_policy: reference_orbital {reference_orbital!r} must be 'homo', 'lumo' or a 1-based index")
            elif not isinstance(reference_orbital, int) or reference_orbital < 1:
                problems.append("window_policy: reference_orbital must be 'homo', 'lumo' or a 1-based index")
            for key in ("n_below", "n_above"):
                value = policy.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    problems.append(f"window_policy: {key} must be an integer >= 0, got {value!r}")
            sweep = policy.get("sweep")
            if sweep is not None:
                if not isinstance(sweep, list) or not sweep:
                    problems.append("window_policy: sweep must be a non-empty list of {n_below, n_above} steps")
                else:
                    for i, step in enumerate(sweep):
                        if not isinstance(step, dict) or set(step) - {"n_below", "n_above", "label"}:
                            problems.append(f"window_policy: sweep[{i}] must be an object with n_below/n_above (and an optional label)")
                            continue
                        for key in ("n_below", "n_above"):
                            value = step.get(key)
                            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                                problems.append(f"window_policy: sweep[{i}].{key} must be an integer >= 0, got {value!r}")

            # A policy has no absolute indices, so the only cross-check available
            # is that a declared orbital count matches the width the policy spans.
            declared_norb = active.get("n_orbitals")
            below, above = policy.get("n_below"), policy.get("n_above")
            if isinstance(declared_norb, int) and isinstance(below, int) and isinstance(above, int):
                width = below + above + 1
                if declared_norb != width:
                    problems.append(
                        f"active_space: declared n_orbitals={declared_norb} but the policy spans "
                        f"{below} + 1 + {above} = {width} orbitals"
                    )

    if window is not None:
        if not isinstance(window, dict):
            problems.append("window: must be an object")
            return
        _require(problems, "window", window, ["nfirst_1based", "nlast_1based"])
        _unknown(problems, "window", window, _WINDOW)
        nfirst = window.get("nfirst_1based")
        nlast = window.get("nlast_1based")
        if not isinstance(nfirst, int) or isinstance(nfirst, bool) or nfirst < 1:
            problems.append(f"window: nfirst_1based must be an integer >= 1, got {nfirst!r}")
            return
        if not isinstance(nlast, int) or isinstance(nlast, bool) or nlast < nfirst:
            problems.append(f"window: nlast_1based must be an integer >= nfirst_1based, got {nlast!r}")
            return

        nbasis = (man.get("reference") or {}).get("expected_nbasis")
        if isinstance(nbasis, int) and nlast > nbasis:
            problems.append(f"window: NLAST {nlast} exceeds the {nbasis} orbitals the basis provides")

        if atoms is None:
            return
        try:
            counts = active_space(man, nfirst, nlast, repo_root=repo_root)
        except ManifestError as exc:
            problems.append(f"active_space: {exc}")
            return

        _cross_check(problems, active, counts)


def _cross_check(problems, active, counts):
    """Declared numbers must agree with what the window and geometry imply."""
    declared = {
        "frozen_core": ("ncore", "frozen core"),
        "n_orbitals": ("nact", "active orbital count"),
        "n_electrons": ("nelec_act", "active electron count"),
        "ms2": ("ms2", "MS2"),
    }
    for key, (derived_key, label) in declared.items():
        if key in active and active[key] != counts[derived_key]:
            problems.append(
                f"active_space: declared {key}={active[key]} but the window and geometry give "
                f"{label} {counts[derived_key]}"
            )

    if counts["nelec_act"] < 0:
        problems.append(
            f"active_space: {counts['ncore']} frozen core orbitals hold more electrons than the "
            f"molecule has ({counts['nelec']})"
        )
    for spin in ("alpha", "beta"):
        n = counts[f"n{spin}_act"]
        if n < 0:
            problems.append(
                f"active_space: NFIRST freezes {counts['ncore']} orbitals but only {counts[f'n{spin}']} "
                f"{spin} electrons exist, so the reference determinant has {n} active {spin} electrons"
            )
        elif n > counts["nact"]:
            problems.append(
                f"active_space: {n} active {spin} electrons do not fit in {counts['nact']} active orbitals"
            )


def _validate_solver(problems, man, atoms, repo_root):
    solver = man.get("solver")
    if solver is None:
        return
    if not isinstance(solver, dict):
        problems.append("solver: must be an object")
        return
    _require(problems, "solver", solver, ["kind"])
    _unknown(problems, "solver", solver, _SOLVER)
    kind = solver.get("kind")
    if kind not in SOLVERS:
        problems.append(f"solver: kind must be one of {SOLVERS}, got {kind!r}")
    for key in ("parameters", "convergence"):
        if key in solver and not isinstance(solver[key], dict):
            problems.append(f"solver: {key} must be an object")

    if kind == "pyscf_fci" and atoms is not None and (man.get("active_space") or {}).get("window"):
        try:
            counts = active_space(man, repo_root=repo_root)
        except ManifestError:
            return
        if counts["n_determinants"] > MAX_FCI_DETERMINANTS:
            problems.append(
                f"solver: pyscf_fci on {counts['nact']} orbitals with "
                f"{counts['nalpha_act']}a/{counts['nbeta_act']}b electrons is "
                f"{counts['n_determinants']:,} determinants, above the "
                f"{MAX_FCI_DETERMINANTS:,} this suite treats as tractable"
            )


def _validate_expected(problems, man):
    expected = man.get("expected")
    if expected is None:
        return
    if not isinstance(expected, dict):
        problems.append("expected: must be an object")
        return
    _unknown(problems, "expected", expected, _EXPECTED)
    for key in ("scf_energy_hartree", "e_core_hartree", "reference_energy_hartree", "total_energy_hartree"):
        if key in expected and not isinstance(expected[key], (int, float)):
            problems.append(f"expected: {key} must be a number")
    tolerance = expected.get("tolerance_hartree")
    if tolerance is not None and (not isinstance(tolerance, (int, float)) or tolerance <= 0):
        problems.append("expected: tolerance_hartree must be a positive number")
    if any(k.endswith("_hartree") and k != "tolerance_hartree" for k in expected) and not expected.get("source"):
        problems.append("expected: reference numbers need a 'source' saying where they came from")
    if "trusted" in expected and not isinstance(expected["trusted"], bool):
        problems.append("expected: 'trusted' must be a boolean")


def _validate_provenance(problems, man):
    provenance = man.get("provenance")
    if provenance is None:
        problems.append("manifest: missing 'provenance'")
        return
    if not isinstance(provenance, dict):
        problems.append("provenance: must be an object")
        return
    _unknown(problems, "provenance", provenance, _PROVENANCE)
    _require(problems, "provenance", provenance, ["added", "geometry_source"])
    added = provenance.get("added")
    if isinstance(added, str) and not _DATE_RE.match(added):
        problems.append(f"provenance: added must be an ISO date (YYYY-MM-DD), got {added!r}")


def check(man: dict, repo_root=None) -> dict:
    """Validate and return the manifest, or raise with every problem at once."""
    problems = validate(man, repo_root)
    if problems:
        name = man.get("id") or man.get("_path") or "<manifest>"
        raise ManifestError(f"{name}: {len(problems)} problem(s)\n  - " + "\n  - ".join(problems))
    return man
