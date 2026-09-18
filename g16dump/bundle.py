"""The ``.npz`` interchange bundle: schema, I/O and validation.

Everything downstream of the Gaussian reader speaks this format and nothing
else. ``matfile.py`` is the only module that imports gauopen; it writes a bundle,
and every other module -- and every test -- reads one. That split exists because
gauopen needs a compiled component and may only exist on the cluster.

Conventions, fixed here and relied on everywhere
------------------------------------------------
**MO coefficients are MO-major**: ``C`` has shape ``(nmo, nbasis)`` and row ``p``
is orbital ``p`` expanded in AOs. So the MO-basis transform is ``C M C.T`` and
orthonormality reads ``C S C.T = I``. This matches gauopen's storage and the
derivation in the README. PySCF uses the transpose (AO-major, ``C.T S C = I``);
:func:`ao_major` converts, and validation reports the orientation by *measuring*
it rather than trusting either convention.

**Two-electron integrals are in chemist's notation** ``(pq|rs)``, indexed
``eri_act[p, q, r, s]``, over active orbitals only.

**The active space is a contiguous window**: orbitals ``[ncore, ncore + nact)``
in the stored MO order, and within that window the orbitals occupied in the
reference determinant come first. Both facts are asserted, not assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .errors import SchemaError, ValidationError, require

SCHEMA_VERSION = 1

#: Reference types we understand. The ``*KS`` entries are accepted in a bundle
#: but are barred from the stored-Fock path (see hamiltonian.py).
REFERENCE_TYPES = ("RHF", "ROHF", "UHF", "RKS", "ROKS", "UKS")

#: Reference types whose stored "Fock" matrix is a Kohn-Sham matrix and
#: therefore carries exchange-correlation, not HF exchange.
KS_REFERENCES = ("RKS", "ROKS", "UKS")

#: Where the Fock matrices in a bundle came from. This is provenance that
#: changes the physics, so it is a required field, not a note.
FOCK_SOURCES = ("gaussian", "rebuilt-pyscf", "none")


@dataclass
class Bundle:
    """A validated, self-describing active-space export.

    Plain data. The only methods are trivial accessors; all the real work lives
    in module-level functions.
    """

    # --- identity and provenance ----------------------------------------
    ref_type: str
    fock_source: str
    charge: int
    multiplicity: int

    # --- electron and orbital counts ------------------------------------
    nelec: int
    nalpha: int
    nbeta: int
    ncore: int
    nact: int
    nfv: int
    nmo: int
    nbasis: int

    # --- energies --------------------------------------------------------
    e_nuc: float

    # --- arrays ----------------------------------------------------------
    h_ao: np.ndarray  # (nbasis, nbasis) bare one-electron AO Hamiltonian
    s_ao: np.ndarray  # (nbasis, nbasis) AO overlap
    c_a: np.ndarray  # (nmo, nbasis) alpha MO coefficients, MO-major
    c_b: np.ndarray  # (nmo, nbasis) beta; equal to c_a for RHF/ROHF
    eri_act: np.ndarray  # (nact,)*4 chemist's (pq|rs) over the active window

    # --- optional --------------------------------------------------------
    e_scf: Optional[float] = None
    f_a_ao: Optional[np.ndarray] = None  # (nbasis, nbasis)
    f_b_ao: Optional[np.ndarray] = None  # (nbasis, nbasis)
    mo_energies_a: Optional[np.ndarray] = None  # (nmo,) DIAGNOSTIC ONLY
    mo_energies_b: Optional[np.ndarray] = None  # (nmo,) DIAGNOSTIC ONLY
    irrep_labels: Optional[np.ndarray] = None  # (nmo,) strings
    atom_charges: Optional[np.ndarray] = None  # (natom,) nuclear charges
    provenance: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    # --- trivial accessors ------------------------------------------------

    @property
    def active(self) -> slice:
        """The active window as a slice into the full MO range."""
        return slice(self.ncore, self.ncore + self.nact)

    @property
    def nocc_a_act(self) -> int:
        """Active orbitals occupied by alpha electrons in the reference."""
        return self.nalpha - self.ncore

    @property
    def nocc_b_act(self) -> int:
        """Active orbitals occupied by beta electrons in the reference."""
        return self.nbeta - self.ncore

    @property
    def nelec_act(self) -> int:
        """Electrons inside the active space."""
        return self.nocc_a_act + self.nocc_b_act

    @property
    def ms2(self) -> int:
        """2*S_z for the active space: the FCIDUMP MS2 field."""
        return self.nocc_a_act - self.nocc_b_act

    @property
    def is_ks(self) -> bool:
        return self.ref_type in KS_REFERENCES

    @property
    def has_fock(self) -> bool:
        return self.f_a_ao is not None and self.f_b_ao is not None


def ao_major(c: np.ndarray) -> np.ndarray:
    """Convert our MO-major ``(nmo, nbasis)`` coefficients to PySCF's layout.

    PySCF's ``mo_coeff`` is ``(nbasis, nmo)`` with AOs down the rows. Anything
    handed to or taken from PySCF goes through here, so the transpose happens in
    exactly one place instead of being rediscovered at every call site.
    """
    return np.ascontiguousarray(np.asarray(c).T)


# ---------------------------------------------------------------- validation


def check_orthonormal(c: np.ndarray, s: np.ndarray, tol: float) -> float:
    """Return ``max|C S C.T - I|``. Does not raise; callers decide."""
    gram = c @ s @ c.T
    return float(np.max(np.abs(gram - np.eye(gram.shape[0]))))


def detect_coefficient_orientation(c: np.ndarray, s: np.ndarray, tol: float = 1e-8) -> str:
    """Measure which storage convention ``c`` is in, rather than assuming one.

    Returns ``"mo-major"`` (ours, ``C S C.T = I``), ``"ao-major"`` (PySCF's,
    ``C.T S C = I``), or ``"neither"``. For a square ``C`` both orientations are
    conformable, which is exactly why this has to be measured: a silently
    transposed coefficient matrix produces a plausible-looking but wrong
    Hamiltonian.
    """
    c = np.asarray(c)
    s = np.asarray(s)

    def _residual(mat):
        if mat.shape[1] != s.shape[0]:
            return np.inf
        return check_orthonormal(mat, s, tol)

    res_mo = _residual(c)
    res_ao = _residual(c.T)

    if res_mo <= tol and res_mo <= res_ao:
        return "mo-major"
    if res_ao <= tol:
        return "ao-major"
    return "neither"


def eri_symmetry_error(eri: np.ndarray) -> float:
    """Return the largest violation of 8-fold permutational symmetry.

    Checks ``(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)``; the remaining four
    relations follow by composition. A nonzero result here on data that should
    be restricted-reference integrals usually means a C-order/Fortran-order
    mix-up in the reader, so the number is reported rather than swallowed.
    """
    eri = np.asarray(eri)
    return max(
        float(np.max(np.abs(eri - eri.transpose(1, 0, 2, 3)))),
        float(np.max(np.abs(eri - eri.transpose(0, 1, 3, 2)))),
        float(np.max(np.abs(eri - eri.transpose(2, 3, 0, 1)))),
    )


def _check_hermitian(name: str, mat: np.ndarray, tol: float) -> None:
    err = float(np.max(np.abs(mat - mat.T))) if mat.size else 0.0
    require(
        err <= tol,
        f"{name} is not symmetric: max|M - M.T| = {err:.3e} > {tol:.1e}. "
        f"A one-electron matrix in a real basis must be symmetric; this usually "
        f"means the packed lower triangle was unpacked into the wrong shape.",
    )


def validate(
    bundle: Bundle,
    *,
    tol_orthonormal: float = 1e-8,
    tol_hermitian: float = 1e-10,
    tol_eri: float = 1e-10,
) -> dict:
    """Check a bundle for physical and numerical consistency.

    Raises :class:`ValidationError` (or :class:`SchemaError`) with a message
    naming the offending quantity and the measured value. Returns a dict of the
    measured residuals so callers can report them.
    """
    require(
        bundle.schema_version == SCHEMA_VERSION,
        f"bundle schema version is {bundle.schema_version!r}, this build of "
        f"g16dump understands version {SCHEMA_VERSION}. Re-export the bundle "
        f"with `g16dump extract`, or install the matching g16dump version.",
        SchemaError,
    )
    require(
        bundle.ref_type in REFERENCE_TYPES,
        f"unknown reference type {bundle.ref_type!r}; expected one of "
        f"{', '.join(REFERENCE_TYPES)}. The reference type is never guessed -- "
        f"pass it explicitly with --ref-type if the file does not state it.",
        ValidationError,
    )
    require(
        bundle.fock_source in FOCK_SOURCES,
        f"unknown fock_source {bundle.fock_source!r}; expected one of "
        f"{', '.join(FOCK_SOURCES)}.",
    )

    # --- electron counts ------------------------------------------------
    require(
        bundle.multiplicity >= 1,
        f"multiplicity must be >= 1, got {bundle.multiplicity}.",
    )
    require(
        bundle.nalpha >= 0 and bundle.nbeta >= 0,
        f"electron counts must be non-negative, got nalpha={bundle.nalpha}, "
        f"nbeta={bundle.nbeta}.",
    )
    require(
        bundle.nalpha + bundle.nbeta == bundle.nelec,
        f"nalpha + nbeta = {bundle.nalpha} + {bundle.nbeta} = "
        f"{bundle.nalpha + bundle.nbeta} does not equal nelec = {bundle.nelec}.",
    )
    require(
        bundle.nalpha - bundle.nbeta == bundle.multiplicity - 1,
        f"nalpha - nbeta = {bundle.nalpha - bundle.nbeta} is inconsistent with "
        f"multiplicity {bundle.multiplicity} (which requires "
        f"{bundle.multiplicity - 1}). A multiplicity of {bundle.multiplicity} "
        f"with {bundle.nelec} electrons is impossible if the parity differs.",
    )
    require(
        bundle.nalpha >= bundle.nbeta,
        f"nalpha ({bundle.nalpha}) < nbeta ({bundle.nbeta}); by convention the "
        f"alpha set is the majority spin.",
    )
    if bundle.atom_charges is not None:
        total_z = int(round(float(np.sum(bundle.atom_charges))))
        require(
            total_z - bundle.charge == bundle.nelec,
            f"charge is inconsistent: sum(Z) - charge = {total_z} - "
            f"{bundle.charge} = {total_z - bundle.charge}, but nelec = "
            f"{bundle.nelec}.",
        )

    # --- orbital counts and the active window ----------------------------
    require(
        bundle.ncore >= 0 and bundle.nact >= 0 and bundle.nfv >= 0,
        f"orbital counts must be non-negative: ncore={bundle.ncore}, "
        f"nact={bundle.nact}, nfv={bundle.nfv}.",
    )
    require(
        bundle.ncore + bundle.nact + bundle.nfv == bundle.nmo,
        f"the active window does not tile the MO space: ncore + nact + nfv = "
        f"{bundle.ncore} + {bundle.nact} + {bundle.nfv} = "
        f"{bundle.ncore + bundle.nact + bundle.nfv}, but nmo = {bundle.nmo}.",
    )
    require(
        bundle.nmo <= bundle.nbasis,
        f"nmo ({bundle.nmo}) exceeds nbasis ({bundle.nbasis}); linear "
        f"dependence can only remove orbitals, never add them.",
    )
    require(
        bundle.ncore <= bundle.nbeta,
        f"ncore ({bundle.ncore}) exceeds nbeta ({bundle.nbeta}): the frozen "
        f"core must be doubly occupied, so it cannot hold more orbitals than "
        f"there are beta electrons.",
    )
    require(
        bundle.nact > 0,
        "the active space is empty (nact = 0); there is nothing to dump.",
    )
    require(
        bundle.nocc_a_act <= bundle.nact,
        f"the reference determinant needs {bundle.nocc_a_act} occupied alpha "
        f"orbitals inside a window that is only {bundle.nact} wide. The active "
        f"window excludes orbitals that are occupied in the reference.",
    )

    # --- shapes ------------------------------------------------------------
    nb, nmo, nact = bundle.nbasis, bundle.nmo, bundle.nact
    for name, arr, shape in (
        ("h_ao", bundle.h_ao, (nb, nb)),
        ("s_ao", bundle.s_ao, (nb, nb)),
        ("c_a", bundle.c_a, (nmo, nb)),
        ("c_b", bundle.c_b, (nmo, nb)),
        ("eri_act", bundle.eri_act, (nact, nact, nact, nact)),
    ):
        require(
            arr.shape == shape,
            f"{name} has shape {arr.shape}, expected {shape}. Note that MO "
            f"coefficients are stored MO-major here: C is (nmo, nbasis), so "
            f"C S C.T = I.",
        )
    for name, arr in (("f_a_ao", bundle.f_a_ao), ("f_b_ao", bundle.f_b_ao)):
        if arr is not None:
            require(
                arr.shape == (nb, nb),
                f"{name} has shape {arr.shape}, expected {(nb, nb)}.",
            )
    for name, arr in (
        ("mo_energies_a", bundle.mo_energies_a),
        ("mo_energies_b", bundle.mo_energies_b),
    ):
        if arr is not None:
            require(
                arr.shape == (nmo,),
                f"{name} has shape {arr.shape}, expected {(nmo,)}.",
            )

    # --- hermiticity --------------------------------------------------------
    _check_hermitian("h_ao", bundle.h_ao, tol_hermitian)
    _check_hermitian("s_ao", bundle.s_ao, tol_hermitian)
    if bundle.f_a_ao is not None:
        _check_hermitian("f_a_ao", bundle.f_a_ao, tol_hermitian)
    if bundle.f_b_ao is not None:
        _check_hermitian("f_b_ao", bundle.f_b_ao, tol_hermitian)

    # --- overlap must be a metric -------------------------------------------
    eigvals = np.linalg.eigvalsh(bundle.s_ao)
    require(
        eigvals.min() > 0,
        f"the AO overlap matrix is not positive definite (smallest eigenvalue "
        f"{eigvals.min():.3e}). It is not a valid metric, so orthonormality "
        f"cannot be checked against it.",
    )

    # --- coefficient orientation, measured not assumed -----------------------
    residuals = {}
    for name, c in (("c_a", bundle.c_a), ("c_b", bundle.c_b)):
        orientation = detect_coefficient_orientation(c, bundle.s_ao, tol_orthonormal)
        res = check_orthonormal(c, bundle.s_ao, tol_orthonormal)
        residuals[f"{name}_orthonormality"] = res
        if orientation == "ao-major":
            raise ValidationError(
                f"{name} appears to be stored AO-major: C.T S C = I holds but "
                f"C S C.T = I does not (max deviation {res:.3e}). This bundle's "
                f"convention is MO-major, C of shape (nmo, nbasis). Transpose "
                f"the coefficients before building the bundle -- do not "
                f"transpose downstream, or the one- and two-electron integrals "
                f"will disagree about which index is which."
            )
        if orientation == "neither":
            raise ValidationError(
                f"{name} is not orthonormal in either orientation: "
                f"max|C S C.T - I| = {res:.3e} > {tol_orthonormal:.1e}. Either "
                f"the overlap does not match these coefficients, or the "
                f"coefficient array was reshaped with the wrong row/column "
                f"order."
            )

    # --- two-electron symmetry ----------------------------------------------
    sym_err = eri_symmetry_error(bundle.eri_act)
    residuals["eri_symmetry"] = sym_err
    require(
        sym_err <= tol_eri,
        f"eri_act violates 8-fold permutational symmetry by {sym_err:.3e} > "
        f"{tol_eri:.1e}. For a restricted reference (pq|rs) must equal "
        f"(qp|rs), (pq|sr) and (rs|pq). A violation of this size usually means "
        f"the flat integral array was reshaped in the wrong index order "
        f"(C-order vs Fortran-order), not that the integrals are wrong.",
    )

    # --- restricted references share spatial orbitals -------------------------
    if bundle.ref_type in ("RHF", "ROHF", "RKS", "ROKS"):
        diff = float(np.max(np.abs(bundle.c_a - bundle.c_b)))
        residuals["c_a_minus_c_b"] = diff
        require(
            diff <= 1e-12,
            f"reference type {bundle.ref_type} is restricted, so alpha and beta "
            f"must share spatial orbitals, but max|c_a - c_b| = {diff:.3e}. If "
            f"the orbitals really differ this is a UHF-like reference, which a "
            f"spin-restricted FCIDUMP cannot represent (see the v1 limitations "
            f"in the README); use UHF natural orbitals instead.",
        )

    return residuals


# ------------------------------------------------------------------ file I/O

_SCALAR_FIELDS = (
    "ref_type",
    "fock_source",
    "charge",
    "multiplicity",
    "nelec",
    "nalpha",
    "nbeta",
    "ncore",
    "nact",
    "nfv",
    "nmo",
    "nbasis",
    "e_nuc",
    "e_scf",
    "schema_version",
)

_ARRAY_FIELDS = (
    "h_ao",
    "s_ao",
    "c_a",
    "c_b",
    "eri_act",
    "f_a_ao",
    "f_b_ao",
    "mo_energies_a",
    "mo_energies_b",
    "irrep_labels",
    "atom_charges",
)


def save(bundle: Bundle, path: str, *, validate_first: bool = True) -> None:
    """Write a bundle to ``path`` as a compressed ``.npz``.

    Validates before writing by default: an invalid bundle on disk is a trap for
    whoever picks it up next.
    """
    if validate_first:
        validate(bundle)

    payload = {}
    for name in _SCALAR_FIELDS:
        value = getattr(bundle, name)
        if value is None:
            continue
        payload[name] = np.array(value)
    for name in _ARRAY_FIELDS:
        value = getattr(bundle, name)
        if value is None:
            continue
        payload[name] = np.asarray(value)
    payload["provenance"] = np.array(json.dumps(bundle.provenance, default=str))

    np.savez_compressed(path, **payload)


def load(path: str) -> Bundle:
    """Read a bundle from ``path``.

    Missing required keys and unsupported schema versions are reported as
    schema errors naming the key, rather than as a ``KeyError`` from deep
    inside numpy.
    """
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)

        if "schema_version" not in keys:
            raise SchemaError(
                f"{path} has no schema_version key, so it is not a g16dump "
                f"bundle (or predates the schema). Keys present: "
                f"{sorted(keys)}."
            )
        version = int(data["schema_version"])
        if version != SCHEMA_VERSION:
            raise SchemaError(
                f"{path} declares schema version {version}; this build of "
                f"g16dump understands version {SCHEMA_VERSION}. Re-export it "
                f"with `g16dump extract`."
            )

        required = {
            "ref_type",
            "fock_source",
            "charge",
            "multiplicity",
            "nelec",
            "nalpha",
            "nbeta",
            "ncore",
            "nact",
            "nfv",
            "nmo",
            "nbasis",
            "e_nuc",
            "h_ao",
            "s_ao",
            "c_a",
            "c_b",
            "eri_act",
        }
        missing = sorted(required - keys)
        if missing:
            raise SchemaError(
                f"{path} is missing required bundle keys: {missing}. Present "
                f"keys: {sorted(keys)}. A bundle truncated this way cannot be "
                f"repaired here -- re-run `g16dump extract`."
            )

        def _get(name, cast=None):
            if name not in keys:
                return None
            value = data[name]
            return cast(value) if cast is not None else np.array(value)

        provenance = {}
        if "provenance" in keys:
            provenance = json.loads(str(data["provenance"]))

        return Bundle(
            ref_type=str(data["ref_type"]),
            fock_source=str(data["fock_source"]),
            charge=int(data["charge"]),
            multiplicity=int(data["multiplicity"]),
            nelec=int(data["nelec"]),
            nalpha=int(data["nalpha"]),
            nbeta=int(data["nbeta"]),
            ncore=int(data["ncore"]),
            nact=int(data["nact"]),
            nfv=int(data["nfv"]),
            nmo=int(data["nmo"]),
            nbasis=int(data["nbasis"]),
            e_nuc=float(data["e_nuc"]),
            e_scf=_get("e_scf", float),
            h_ao=np.array(data["h_ao"]),
            s_ao=np.array(data["s_ao"]),
            c_a=np.array(data["c_a"]),
            c_b=np.array(data["c_b"]),
            eri_act=np.array(data["eri_act"]),
            f_a_ao=_get("f_a_ao"),
            f_b_ao=_get("f_b_ao"),
            mo_energies_a=_get("mo_energies_a"),
            mo_energies_b=_get("mo_energies_b"),
            irrep_labels=_get("irrep_labels"),
            atom_charges=_get("atom_charges"),
            provenance=provenance,
            schema_version=version,
        )

def load_and_validate(path: str, **tolerances) -> Bundle:
    """Load a bundle and validate it, the normal entry point for consumers."""
    bundle = load(path)
    validate(bundle, **tolerances)
    return bundle
