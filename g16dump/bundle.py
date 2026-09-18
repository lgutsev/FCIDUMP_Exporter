"""The ``.npz`` interchange bundle: schema, I/O and validation.

Everything downstream of the Gaussian reader speaks this format and nothing
else. ``matfile.py`` is the only module that produces one from a ``.mat``; the
Hamiltonian code, the writer, the rotations and the tests all read one.

Reach for the accessors on :class:`Bundle` -- :attr:`Bundle.active`,
:attr:`Bundle.core`, :attr:`Bundle.nocc_active_alpha` and the rest -- rather than
doing index arithmetic on the stored fields. The stored window is 1-based and
the arrays are 0-based, and the accessors are the single place that conversion
happens.

Conventions
-----------
``C`` is stored **column-wise**, ``C[ao, mo]``, normalised so that

    C.T @ S @ C == I

This is PySCF's convention, and every oracle in this project is PySCF, so the
bundle speaks it. The README states the same algebra in the row convention
(``h = C Hcore C.T``); ``matfile.py`` determines numerically which one Gaussian
handed it and transposes if needed, rather than trusting a storage convention.

``active_first`` and ``active_last`` are the **1-based, inclusive** MO indices
of the active window -- the same ``NFIRST`` and ``NLAST`` that go into the
Gaussian route, stored unchanged so that a bundle records the window exactly as
the job was asked for. ``ncore`` and ``nact`` are stored redundantly and
validation requires all four to agree, so a window that does not mean what the
route meant cannot pass quietly. Use :attr:`Bundle.active` for a 0-based slice.

Fock matrices are stored in the **AO** basis. Storing them in the MO basis would
make every active-space rotation have to touch them; ``hamiltonian.py`` does the
``C.T F C`` transform itself.

``orbital_energies`` is a **diagnostic** and is never used as a substitute for a
Fock matrix. That substitution is the defect this package was written to
replace.

A Kohn-Sham bundle never carries a stored Fock matrix. The KS matrix contains
exchange-correlation and is not the HF Fock operator, so ``matfile.py`` refuses
to write it into ``F_alpha_ao``; a KS bundle arrives with ``fock_source="none"``
and the rebuilt-HF-Fock path is the only way to use it. Validation rejects a KS
reference carrying ``fock_source="gaussian"`` as the second line of defence.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

#: Bumped from 1 when the field names were reconciled with the project plan.
SCHEMA_VERSION = 2

#: References whose orbitals are Kohn-Sham. The stored matrix for these is the
#: KS matrix, never a Fock matrix; see the module docstring.
KS_REFERENCES = frozenset({"RKS", "ROKS"})
HF_REFERENCES = frozenset({"RHF", "ROHF"})
REFERENCES = KS_REFERENCES | HF_REFERENCES

FOCK_SOURCES = frozenset({"gaussian", "pyscf_rebuilt", "none"})

#: Hermiticity of the one-electron AO matrices. Gaussian writes these to full
#: double precision, so a violation above this is a reader bug, not noise.
HERMITICITY_TOL = 1e-8
#: ``C.T S C - I``. Looser than the above because MO coefficients often make the
#: trip through a formatted checkpoint file, which truncates them.
ORTHONORMALITY_TOL = 1e-6
#: Permutational symmetry of the active ERIs.
ERI_SYMMETRY_TOL = 1e-8


class BundleError(ValueError):
    """A bundle is malformed, internally inconsistent or physically impossible."""


# Arrays, with their expected shapes written as a tuple of attribute names.
_REQUIRED_ARRAYS = {
    "C": ("nao", "nmo"),
    "S": ("nao", "nao"),
    "Hcore_ao": ("nao", "nao"),
    "eri_active": ("nact", "nact", "nact", "nact"),
}
_OPTIONAL_ARRAYS = {
    "F_alpha_ao": ("nao", "nao"),
    "F_beta_ao": ("nao", "nao"),
    "orbital_energies": ("nmo",),
    "orbital_energies_beta": ("nmo",),
    "atom_charges": None,  # (natom,), and natom is not otherwise known
}
_INT_SCALARS = (
    "schema_version",
    "charge",
    "multiplicity",
    "nelec",
    "nalpha",
    "nbeta",
    "nao",
    "nmo",
    "ncore",
    "nact",
    "active_first",
    "active_last",
)
_STR_SCALARS = (
    "source_program",
    "source_file",
    "reference_type",
    "fock_source",
    "provenance",
)


@dataclass
class Bundle:
    """One Gaussian job, reduced to what the active-space Hamiltonian needs.

    Construct it directly in tests, or through :func:`load`. Nothing here is
    checked on construction; call :func:`validate` (which :func:`load` does by
    default) to assert the invariants.
    """

    reference_type: str
    charge: int
    multiplicity: int
    nelec: int
    nalpha: int
    nbeta: int
    nao: int
    nmo: int
    ncore: int
    nact: int
    active_first: int
    active_last: int
    enuc: float

    C: np.ndarray
    S: np.ndarray
    Hcore_ao: np.ndarray
    eri_active: np.ndarray

    source_program: str = "unknown"
    source_file: str = "unknown"
    fock_source: str = "none"
    F_alpha_ao: Optional[np.ndarray] = None
    F_beta_ao: Optional[np.ndarray] = None
    orbital_energies: Optional[np.ndarray] = None
    orbital_energies_beta: Optional[np.ndarray] = None
    atom_charges: Optional[np.ndarray] = None
    escf: Optional[float] = None

    provenance: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    # ---- accessors -------------------------------------------------------
    # Every conversion between the stored 1-based window and 0-based array
    # indices happens here and nowhere else. Downstream code goes through these
    # rather than recomputing active_first - 1 in a dozen places.

    @property
    def active(self) -> slice:
        """The active window as a 0-based slice into the MO index range."""
        return slice(self.active_first - 1, self.active_last)

    @property
    def core(self) -> slice:
        """The frozen core as a 0-based slice. Doubly occupied by construction."""
        return slice(0, self.ncore)

    @property
    def active_start(self) -> int:
        """0-based index of the first active MO."""
        return self.active_first - 1

    @property
    def active_stop(self) -> int:
        """0-based index one past the last active MO."""
        return self.active_last

    @property
    def nocc_active_alpha(self) -> int:
        """Active orbitals occupied by an alpha electron in the reference."""
        return self.nalpha - self.ncore

    @property
    def nocc_active_beta(self) -> int:
        return self.nbeta - self.ncore

    @property
    def nelec_active(self) -> int:
        """Electrons inside the active space; the FCIDUMP's ``NELEC``."""
        return self.nocc_active_alpha + self.nocc_active_beta

    @property
    def ms2(self) -> int:
        """``nalpha - nbeta``; the FCIDUMP's ``MS2``."""
        return self.nalpha - self.nbeta

    @property
    def nfrozen_virtual(self) -> int:
        return self.nmo - self.active_last

    @property
    def is_ks(self) -> bool:
        return self.reference_type in KS_REFERENCES

    @property
    def has_fock(self) -> bool:
        return self.F_alpha_ao is not None

    def describe(self) -> str:
        """One-paragraph human summary, used by ``g16dump validate``."""
        scf = "unknown" if self.escf is None else f"{self.escf:.10f} Ha"
        return (
            f"{self.reference_type}  charge {self.charge:+d}  multiplicity "
            f"{self.multiplicity}\n"
            f"  source        {self.source_program}: {self.source_file}\n"
            f"  electrons     {self.nelec} ({self.nalpha} alpha, {self.nbeta} beta)\n"
            f"  orbitals      {self.nao} AO, {self.nmo} MO\n"
            f"  active window MOs {self.active_first}-{self.active_last} "
            f"(1-based), {self.nact} orbitals\n"
            f"  frozen        {self.ncore} core, {self.nfrozen_virtual} virtual\n"
            f"  in the window {self.nocc_active_alpha} alpha and "
            f"{self.nocc_active_beta} beta electrons\n"
            f"  enuc          {self.enuc:.10f} Ha\n"
            f"  escf          {scf}\n"
            f"  Fock source   {self.fock_source}\n"
            f"  schema        v{self.schema_version}"
        )


# --------------------------------------------------------------------- I/O


def save(bundle: Bundle, path) -> Path:
    """Write ``bundle`` to ``path`` as a compressed ``.npz``.

    The bundle is validated first: writing a bundle that cannot be loaded back
    is never what the caller wanted.
    """
    validate(bundle)
    path = Path(path)
    payload = {}

    for name in _INT_SCALARS:
        payload[name] = np.int64(getattr(bundle, name))
    payload["enuc"] = np.float64(bundle.enuc)
    for name in ("source_program", "source_file", "reference_type", "fock_source"):
        payload[name] = np.asarray(getattr(bundle, name))
    payload["provenance"] = np.asarray(json.dumps(bundle.provenance, default=str))

    if bundle.escf is not None:
        payload["escf"] = np.float64(bundle.escf)

    for name in list(_REQUIRED_ARRAYS) + list(_OPTIONAL_ARRAYS):
        value = getattr(bundle, name)
        if value is not None:
            payload[name] = np.ascontiguousarray(value, dtype=np.float64)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load(path, validate_bundle: bool = True) -> Bundle:
    """Read a ``.npz`` bundle back into a :class:`Bundle`.

    ``validate_bundle=False`` is for tests that want to inspect a deliberately
    broken bundle; every other caller should leave it alone.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)

        if "schema_version" not in keys:
            raise BundleError(
                f"{path}: no schema_version. This is not a g16dump bundle, or it "
                f"predates the schema."
            )
        version = int(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise BundleError(
                f"{path}: schema version {version}, but this g16dump speaks "
                f"version {SCHEMA_VERSION}."
            )

        missing = [
            name
            for name in (*_INT_SCALARS, *_STR_SCALARS, "enuc", *_REQUIRED_ARRAYS)
            if name not in keys
        ]
        if missing:
            raise BundleError(f"{path}: missing required keys: {', '.join(missing)}")

        kwargs = {name: int(data[name]) for name in _INT_SCALARS}
        kwargs["enuc"] = float(data["enuc"])
        for name in ("source_program", "source_file", "reference_type",
                     "fock_source"):
            kwargs[name] = str(data[name].item())
        kwargs["escf"] = float(data["escf"]) if "escf" in keys else None

        try:
            kwargs["provenance"] = json.loads(str(data["provenance"].item()))
        except json.JSONDecodeError as exc:
            raise BundleError(f"{path}: provenance is not valid JSON: {exc}") from exc

        for name in _REQUIRED_ARRAYS:
            kwargs[name] = np.asarray(data[name], dtype=np.float64)
        for name in _OPTIONAL_ARRAYS:
            if name in keys:
                kwargs[name] = np.asarray(data[name], dtype=np.float64)

    bundle = Bundle(**kwargs)
    if validate_bundle:
        validate(bundle, source=str(path))
    return bundle


# ---------------------------------------------------------------- validation


def validate(
    bundle: Bundle,
    *,
    source: str = "bundle",
    hermiticity_tol: float = HERMITICITY_TOL,
    orthonormality_tol: float = ORTHONORMALITY_TOL,
    eri_symmetry_tol: float = ERI_SYMMETRY_TOL,
) -> None:
    """Assert every invariant the schema promises. Raise :class:`BundleError`.

    The checks are ordered so that the first failure is the most informative
    one: metadata before shapes, shapes before numerics, because a numerical
    check on a wrongly shaped array reports the wrong problem.
    """
    problems: list[str] = []
    problems += _check_metadata(bundle)
    problems += _check_counts(bundle)
    problems += _check_window(bundle)

    # A shape mismatch makes every numerical check below meaningless, so stop.
    shape_problems = _check_shapes(bundle)
    if shape_problems:
        problems += shape_problems
        _raise(source, problems)

    problems += _check_finite(bundle)
    problems += _check_hermiticity(bundle, hermiticity_tol)
    problems += _check_orthonormality(bundle, orthonormality_tol)
    problems += _check_eri_symmetry(bundle, eri_symmetry_tol)

    if problems:
        _raise(source, problems)


def _raise(source: str, problems: list[str]) -> None:
    body = "\n".join(f"  - {p}" for p in problems)
    raise BundleError(f"{source} failed validation:\n{body}")


def _check_metadata(b: Bundle) -> list[str]:
    problems = []
    if b.schema_version != SCHEMA_VERSION:
        problems.append(
            f"schema_version is {b.schema_version}, expected {SCHEMA_VERSION}"
        )
    if b.reference_type not in REFERENCES:
        problems.append(
            f"reference_type {b.reference_type!r} is not one of "
            f"{sorted(REFERENCES)}. The reference type is never inferred; pass "
            f"it explicitly."
        )
    if b.fock_source not in FOCK_SOURCES:
        problems.append(
            f"fock_source {b.fock_source!r} is not one of {sorted(FOCK_SOURCES)}"
        )
    if b.reference_type in KS_REFERENCES and b.fock_source == "gaussian":
        problems.append(
            "reference_type is Kohn-Sham but fock_source is 'gaussian'. The "
            "stored KS matrix contains exchange-correlation and is not the HF "
            "Fock operator; KS orbitals require the rebuilt-Fock path."
        )
    if b.fock_source == "none" and (
        b.F_alpha_ao is not None or b.F_beta_ao is not None
    ):
        problems.append(
            "fock_source is 'none' but a Fock matrix is present; the source of "
            "every stored Fock matrix must be recorded"
        )
    if b.fock_source != "none" and b.F_alpha_ao is None:
        problems.append(
            f"fock_source is {b.fock_source!r} but no F_alpha_ao is stored"
        )
    for name in ("source_program", "source_file"):
        if not str(getattr(b, name)).strip():
            problems.append(
                f"{name} is empty; provenance must say where this came from"
            )
    if not isinstance(b.provenance, dict):
        problems.append(f"provenance must be a dict, got {type(b.provenance).__name__}")
    return problems


def _check_counts(b: Bundle) -> list[str]:
    problems = []
    if b.multiplicity < 1:
        problems.append(f"multiplicity {b.multiplicity} is not a positive integer")
    if b.nelec < 0:
        problems.append(f"nelec {b.nelec} is negative")
    if b.nalpha < 0 or b.nbeta < 0:
        problems.append(
            f"negative spin populations: nalpha={b.nalpha}, nbeta={b.nbeta}"
        )
    if b.nalpha + b.nbeta != b.nelec:
        problems.append(f"nalpha + nbeta = {b.nalpha + b.nbeta} but nelec = {b.nelec}")
    if b.nalpha < b.nbeta:
        problems.append(
            f"nalpha ({b.nalpha}) < nbeta ({b.nbeta}); this bundle format takes "
            f"the high-spin convention"
        )
    if b.nalpha - b.nbeta != b.multiplicity - 1:
        problems.append(
            f"multiplicity {b.multiplicity} incompatible with electron count: "
            f"nalpha - nbeta = {b.nalpha - b.nbeta} but multiplicity "
            f"{b.multiplicity} implies {b.multiplicity - 1}"
        )
    if b.multiplicity - 1 > b.nelec:
        problems.append(
            f"multiplicity {b.multiplicity} incompatible with electron count: it "
            f"needs {b.multiplicity - 1} unpaired electrons but there are only "
            f"{b.nelec}"
        )
    if (b.nelec - (b.multiplicity - 1)) % 2 != 0:
        problems.append(
            f"multiplicity {b.multiplicity} incompatible with electron count: "
            f"{b.nelec} electrons cannot give it, the parity is wrong"
        )
    if b.nao < 1 or b.nmo < 1:
        problems.append(f"nao={b.nao}, nmo={b.nmo}; both must be positive")
    if b.nmo > b.nao:
        problems.append(
            f"nmo ({b.nmo}) exceeds nao ({b.nao}); an MO basis cannot be larger "
            f"than the AO basis it is built from"
        )
    if b.nalpha > b.nmo:
        problems.append(f"nalpha ({b.nalpha}) exceeds nmo ({b.nmo})")
    if b.atom_charges is not None:
        implied = int(round(float(np.sum(b.atom_charges)))) - b.charge
        if implied != b.nelec:
            problems.append(
                f"nuclear charges sum to {float(np.sum(b.atom_charges)):g} which "
                f"with charge {b.charge:+d} implies {implied} electrons, but "
                f"nelec is {b.nelec}"
            )
    return problems


def _check_window(b: Bundle) -> list[str]:
    """The window is 1-based and inclusive; ncore and nact must agree with it."""
    problems = []
    if b.active_first < 1:
        problems.append(
            f"active_first is {b.active_first}; the window is 1-based, so the "
            f"first MO is 1"
        )
    if b.active_last < b.active_first:
        problems.append(
            f"active window {b.active_first}-{b.active_last} is empty; "
            f"active_last is inclusive and must not precede active_first"
        )
        return problems  # everything below divides by this window
    if b.active_last > b.nmo:
        problems.append(
            f"active window ends at MO {b.active_last} but there are only "
            f"{b.nmo} MOs"
        )
    span = b.active_last - b.active_first + 1
    if b.nact != span:
        problems.append(
            f"active ERI dimension inconsistent with nact: nact is {b.nact} but "
            f"the window {b.active_first}-{b.active_last} spans {span} orbitals"
        )
    if b.ncore != b.active_first - 1:
        problems.append(
            f"ncore is {b.ncore} but the window starts at MO {b.active_first}, "
            f"so {b.active_first - 1} orbitals below it are frozen core"
        )
    if b.ncore > b.nbeta:
        problems.append(
            f"{b.ncore} frozen core orbitals need {2 * b.ncore} electrons but "
            f"there are only {b.nbeta} beta electrons; the core would not be "
            f"doubly occupied"
        )
    if b.nalpha > b.active_last:
        problems.append(
            f"{b.nalpha} alpha electrons occupy MOs past the end of the window "
            f"(MO {b.active_last}); a frozen virtual cannot be occupied"
        )
    if b.nalpha - b.ncore > b.nact:
        problems.append(
            f"{b.nalpha - b.ncore} active alpha electrons do not fit in "
            f"{b.nact} active orbitals"
        )
    return problems


def _check_shapes(b: Bundle) -> list[str]:
    problems = []
    dims = {"nao": b.nao, "nmo": b.nmo, "nact": b.nact}
    for name, spec in {**_REQUIRED_ARRAYS, **_OPTIONAL_ARRAYS}.items():
        value = getattr(b, name)
        if value is None:
            continue
        value = np.asarray(value)
        if spec is None:
            if value.ndim != 1:
                problems.append(f"{name} has shape {value.shape}, expected 1-D")
            continue
        expected = tuple(dims[d] for d in spec)
        if value.shape != expected:
            problems.append(
                f"{name} has shape {value.shape}, expected {expected} "
                f"({' x '.join(spec)})"
            )
    return problems


def _check_finite(b: Bundle) -> list[str]:
    problems = []
    for name in (*_REQUIRED_ARRAYS, *_OPTIONAL_ARRAYS):
        value = getattr(b, name)
        if value is not None and not np.all(np.isfinite(value)):
            count = int(np.count_nonzero(~np.isfinite(value)))
            problems.append(f"{name} holds {count} non-finite values")
    for name in ("enuc", "escf"):
        value = getattr(b, name)
        if value is not None and not np.isfinite(value):
            problems.append(f"{name} is {value}")
    return problems


def _check_hermiticity(b: Bundle, tol: float) -> list[str]:
    problems = []
    for name in ("S", "Hcore_ao", "F_alpha_ao", "F_beta_ao"):
        matrix = getattr(b, name)
        if matrix is None:
            continue
        deviation = float(np.max(np.abs(matrix - matrix.T))) if matrix.size else 0.0
        if deviation > tol:
            problems.append(
                f"{name} is not symmetric: max |A - A.T| = {deviation:.3e} > {tol:.1e}"
            )
    if b.S is not None and b.S.size:
        try:
            np.linalg.cholesky(b.S)
        except np.linalg.LinAlgError:
            problems.append(
                "S is not positive definite; it is not an AO overlap matrix"
            )
    return problems


def _check_orthonormality(b: Bundle, tol: float) -> list[str]:
    """Pin down the coefficient convention numerically, never by assumption."""
    column = float(np.max(np.abs(b.C.T @ b.S @ b.C - np.eye(b.nmo))))
    if column <= tol:
        return []

    # Before blaming the coefficients, ask whether they are simply transposed.
    # That is the single most likely reader bug, and it has a specific fix.
    if b.nao == b.nmo:
        row = float(np.max(np.abs(b.C @ b.S @ b.C.T - np.eye(b.nao))))
        if row <= tol:
            return [
                f"MO coefficients are not orthonormal in the supplied AO "
                f"overlap: C satisfies C S C.T = I but not C.T S C = I "
                f"(deviation {column:.3e}), so they are stored row-wise. The "
                f"bundle convention is column-wise, C[ao, mo]; transpose them."
            ]
    return [
        f"MO coefficients are not orthonormal in the supplied AO overlap: "
        f"max |C.T S C - I| = {column:.3e} > {tol:.1e}"
    ]


def _check_eri_symmetry(b: Bundle, tol: float) -> list[str]:
    """Real orbitals give 8-fold permutational symmetry; check its generators."""
    eri = b.eri_active
    checks = {
        "(tu|vw) = (ut|vw)": eri - eri.transpose(1, 0, 2, 3),
        "(tu|vw) = (tu|wv)": eri - eri.transpose(0, 1, 3, 2),
        "(tu|vw) = (vw|tu)": eri - eri.transpose(2, 3, 0, 1),
    }
    problems = []
    for label, difference in checks.items():
        deviation = float(np.max(np.abs(difference))) if difference.size else 0.0
        if deviation > tol:
            problems.append(
                f"eri_active violates {label}: max deviation {deviation:.3e} > "
                f"{tol:.1e}"
            )
    return problems


# --------------------------------------------------------------- provenance


def git_commit(repo: Optional[Path] = None) -> Optional[str]:
    """The commit this bundle was produced at, or ``None`` outside a checkout."""
    root = Path(repo) if repo else Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def make_provenance(
    *,
    source_file: Optional[str] = None,
    fch_file: Optional[str] = None,
    route: Optional[str] = None,
    basis: Optional[str] = None,
    method: Optional[str] = None,
    window_1based: Optional[tuple] = None,
    fock_source: str = "none",
    extra: Optional[dict] = None,
) -> dict:
    """Assemble the provenance record every bundle carries.

    Whatever is unknown stays out; a provenance record that invents a basis set
    name is worse than one that admits it does not know.
    """
    record = {
        "schema_version": SCHEMA_VERSION,
        "g16dump_commit": git_commit(),
        "fock_source": fock_source,
    }
    optional = {
        "source_file": source_file,
        "fch_file": fch_file,
        "route": route,
        "basis": basis,
        "method": method,
        "window_1based": list(window_1based) if window_1based else None,
    }
    record.update({k: v for k, v in optional.items() if v is not None})
    if extra:
        record.update(extra)
    return record


__all__ = [
    "Bundle",
    "BundleError",
    "SCHEMA_VERSION",
    "REFERENCES",
    "KS_REFERENCES",
    "HF_REFERENCES",
    "FOCK_SOURCES",
    "load",
    "save",
    "validate",
    "make_provenance",
    "git_commit",
]
