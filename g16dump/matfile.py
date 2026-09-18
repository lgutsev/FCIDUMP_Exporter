"""Read a Gaussian 16 matrix-element file into a :class:`~g16dump.bundle.Bundle`.

This is the only module that imports gauopen, and the only one that runs on the
cluster. Everything downstream reads the ``.npz`` it writes.

**The matrix-element label names below are not yet confirmed against a real
``.mat``.** The M0 gate is exactly that confirmation: run
``scripts/inspect_mat.py`` on a Gaussian job and compare its ``matlist_keys``
against :data:`LABELS`. Nothing else in this module hardcodes a label, so when
the probe output arrives, correcting :data:`LABELS` is the whole change. Until
then a missing quantity raises an error that prints every label the file did
carry, which turns a failed cluster run into a usable bug report instead of a
traceback.

Two things are resolved numerically rather than assumed:

* the MO coefficient convention, by testing ``C.T S C = I`` against both
  orientations (see :func:`orient_mo_coeff`);
* the active window, by cross-checking the frozen-core partition the file
  reports against the dimension of the two-electron block it actually contains.

The reference type is never inferred. ``RHF``, ``ROHF``, ``RKS`` and ``ROKS``
leave overlapping fingerprints in a ``.mat``, and guessing wrong silently
produces a plausible, wrong Hamiltonian, so the caller states it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .bundle import Bundle, KS_REFERENCES, REFERENCES, make_provenance

#: Candidate spellings for each quantity, matched case-insensitively as whole
#: labels first and as substrings second. UNCONFIRMED -- see the module
#: docstring. Order is preference order.
LABELS = {
    "overlap": ["OVERLAP"],
    "hcore": ["CORE HAMILTONIAN ALPHA", "CORE HAMILTONIAN"],
    "mo_coeff_alpha": ["ALPHA MO COEFFICIENTS", "ALPHA ORBITAL COEFFICIENTS"],
    "mo_coeff_beta": ["BETA MO COEFFICIENTS", "BETA ORBITAL COEFFICIENTS"],
    "fock_alpha": ["ALPHA FOCK MATRIX", "ALPHA FOCK"],
    "fock_beta": ["BETA FOCK MATRIX", "BETA FOCK"],
    "orbital_energies": ["ALPHA ORBITAL ENERGIES"],
    "orbital_energies_beta": ["BETA ORBITAL ENERGIES"],
    "eri_active": ["AA MO 2E INTEGRALS", "ALPHA-ALPHA MO 2E INTEGRALS"],
}

#: Scalar names to try, in order, for each energy. A miss is expected; only an
#: all-miss is a problem, and only for the nuclear repulsion.
SCALARS = {
    "enuc": ["ENUCREP"],
    "escf": ["ESCF", "SCF ENERGY", "ETOTAL"],
}

#: The blocks ``extract`` cannot do without. The rest are optional: a missing
#: Fock matrix means the rebuilt-Fock path, not a failed extraction.
_REQUIRED_QUANTITIES = ("overlap", "hcore", "mo_coeff_alpha", "eri_active")

#: ``C.T S C - I`` must come in under this for an orientation to be accepted.
ORIENTATION_TOL = 1e-6


class MatFileError(ValueError):
    """A ``.mat`` file is missing something, or does not mean what we assumed."""


# ------------------------------------------------------------ label lookup


def find_label(matlist_keys: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """The first candidate present in ``matlist_keys``, or ``None``.

    Exact matches (case- and whitespace-insensitive) win over substring
    matches, so a file carrying both ``ALPHA FOCK MATRIX`` and
    ``ALPHA FOCK MATRIX (SAVED)`` resolves to the plain one.
    """
    normalised = {" ".join(key.upper().split()): key for key in matlist_keys}
    for candidate in candidates:
        wanted = " ".join(candidate.upper().split())
        if wanted in normalised:
            return normalised[wanted]
    for candidate in candidates:
        wanted = " ".join(candidate.upper().split())
        for norm, original in normalised.items():
            if wanted in norm:
                return original
    return None


def _require_label(matlist_keys, quantity: str, path) -> str:
    label = find_label(matlist_keys, LABELS[quantity])
    if label is None:
        raise MatFileError(
            f"{path}: no matrix-element block for {quantity!r}.\n"
            f"  looked for: {LABELS[quantity]}\n"
            f"  file carries: {sorted(matlist_keys)}\n"
            f"If one of those is the right block under a different name, add its "
            f"spelling to g16dump.matfile.LABELS[{quantity!r}]. Do not work "
            f"around a genuinely absent block: a missing Fock matrix means the "
            f"rebuilt-Fock path, and missing 2e integrals mean the "
            f"transformation did not run."
        )
    return label


def _expand(entry, name: str, path) -> np.ndarray:
    """Whatever the entry holds, as a plain float64 ndarray.

    gauopen packs one-electron blocks lower-triangularly and unpacks them
    through ``expand()``. Objects without it are taken as already expanded.
    """
    if hasattr(entry, "expand"):
        try:
            return np.asarray(entry.expand(), dtype=np.float64)
        except Exception as exc:
            raise MatFileError(
                f"{path}: {name}.expand() failed: {type(exc).__name__}: {exc}"
            ) from exc
    try:
        return np.asarray(entry.array, dtype=np.float64)
    except Exception as exc:
        raise MatFileError(
            f"{path}: {name} has neither expand() nor a usable .array"
        ) from exc


def _square(array: np.ndarray, n: int, name: str, path) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.shape == (n, n):
        return array
    if array.size == n * n:
        return array.reshape(n, n)
    if array.size == n * (n + 1) // 2:
        out = np.zeros((n, n))
        rows, cols = np.tril_indices(n)
        out[rows, cols] = array.ravel()
        return out + out.T - np.diag(np.diag(out))
    raise MatFileError(
        f"{path}: {name} has {array.size} elements, which is neither {n}x{n} "
        f"nor a {n}-dimensional packed triangle. The dimension the file reports "
        f"and the block it stores disagree."
    )


# ------------------------------------------------- the coefficient convention


def orient_mo_coeff(
    raw: np.ndarray,
    overlap: np.ndarray,
    nao: int,
    nmo: int,
    tol: float = ORIENTATION_TOL,
) -> np.ndarray:
    """Return the coefficients as ``C[ao, mo]``, decided by ``C.T S C = I``.

    Gaussian's storage order is a documented detail that has changed and that
    differs between quantities; reading it out of the numbers costs one matrix
    product and cannot be wrong in the way an assumption can.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.size != nao * nmo:
        raise MatFileError(
            f"MO coefficients have {raw.size} elements, expected {nao * nmo} "
            f"= nao {nao} x nmo {nmo}"
        )

    if raw.ndim == 2:
        candidates = {"as stored": raw, "transposed": raw.T}
    else:
        candidates = {
            "reshaped (nao, nmo)": raw.reshape(nao, nmo),
            "reshaped (nmo, nao) then transposed": raw.reshape(nmo, nao).T,
        }

    identity = np.eye(nmo)
    scores = {}
    for name, candidate in candidates.items():
        if candidate.shape != (nao, nmo):
            continue
        scores[name] = float(
            np.max(np.abs(candidate.T @ overlap @ candidate - identity))
        )

    if not scores:
        raise MatFileError(
            f"no reshape of the MO coefficients gives shape ({nao}, {nmo})"
        )

    best = min(scores, key=scores.get)
    if scores[best] > tol:
        detail = "\n".join(
            f"    {k}: max |C.T S C - I| = {v:.3e}" for k, v in scores.items()
        )
        raise MatFileError(
            f"MO coefficients are not orthonormal in the supplied AO overlap, "
            f"in either orientation:\n{detail}\n"
            f"  tolerance: {tol:.1e}\n"
            f"Either the overlap and the coefficients come from different AO "
            f"sets (check Int=NoBasisTransform in the route), or the block read "
            f"is not the MO coefficient block."
        )
    return np.ascontiguousarray(candidates[best])


# --------------------------------------------------------------- the reader


def _scalar(me, names: Sequence[str]) -> Optional[float]:
    if not hasattr(me, "scalar"):
        return None
    for name in names:
        try:
            value = me.scalar(name)
        except Exception:
            continue
        if value is not None:
            return float(value)
    return None


def _attr(me, names: Sequence[str], default=None):
    for name in names:
        try:
            value = getattr(me, name)
        except Exception:
            continue
        if value is not None:
            return value
    return default


def read_matel(path):
    """Open a ``.mat`` through gauopen, reporting its absence as a plain error."""
    try:
        import QCMatEl
    except ImportError as exc:
        raise MatFileError(
            "gauopen is not importable (no module QCMatEl). It ships with "
            "Gaussian and is not on PyPI; put its directory on PYTHONPATH:\n"
            "  export PYTHONPATH=/path/to/gauopen:$PYTHONPATH\n"
            "Only 'g16dump extract' needs it."
        ) from exc
    return QCMatEl.MatEl(file=str(path))


def extract(
    path,
    *,
    reference: str,
    window: Optional[tuple] = None,
    fch: Optional[str] = None,
    route: Optional[str] = None,
    basis: Optional[str] = None,
    method: Optional[str] = None,
) -> Bundle:
    """Build a bundle from the Gaussian matrix-element file at ``path``.

    ``reference`` is required and is never guessed; see the module docstring.
    ``window`` is the 1-based ``(NFIRST, NLAST)`` pair from the Gaussian route.
    Omit it to take the frozen-core partition the file reports, which is then
    cross-checked against the dimension of the two-electron block regardless.
    """
    path = Path(path)
    reference = reference.upper()
    if reference not in REFERENCES:
        raise MatFileError(
            f"reference {reference!r} is not one of {sorted(REFERENCES)}. The "
            f"reference type is not inferred from the file: RHF, ROHF and their "
            f"Kohn-Sham counterparts leave overlapping fingerprints in a .mat, "
            f"and the wrong choice produces a plausible, wrong Hamiltonian."
        )

    me = read_matel(path)
    keys = list(me.matlist.keys())

    nao = int(_attr(me, ["nbasis"], 0))
    nmo = int(_attr(me, ["nbsuse"], nao) or nao)
    nelec = int(_attr(me, ["ne"], 0))
    multiplicity = int(_attr(me, ["multip", "multiplicity"], 1))
    if nao < 1 or nelec < 1:
        raise MatFileError(
            f"{path}: the header reports nbasis={nao}, ne={nelec}. The file is "
            f"truncated or is not a matrix-element file."
        )

    nunpaired = multiplicity - 1
    if (nelec - nunpaired) % 2 or nunpaired > nelec:
        raise MatFileError(
            f"{path}: {nelec} electrons cannot carry multiplicity "
            f"{multiplicity}."
        )
    nalpha = (nelec + nunpaired) // 2
    nbeta = nelec - nalpha

    # UHF is out of scope: a spin-restricted FCIDUMP cannot represent it, and a
    # beta coefficient block that differs from alpha is exactly that case.
    beta_label = find_label(keys, LABELS["mo_coeff_beta"])
    overlap = _square(
        _expand(me.matlist[_require_label(keys, "overlap", path)], "overlap", path),
        nao, "overlap", path,
    )
    hcore = _square(
        _expand(me.matlist[_require_label(keys, "hcore", path)], "hcore", path),
        nao, "core Hamiltonian", path,
    )

    mo_label = _require_label(keys, "mo_coeff_alpha", path)
    mo_coeff = orient_mo_coeff(
        _expand(me.matlist[mo_label], mo_label, path), overlap, nao, nmo
    )
    if beta_label is not None:
        beta_coeff = orient_mo_coeff(
            _expand(me.matlist[beta_label], beta_label, path), overlap, nao, nmo
        )
        if not np.allclose(beta_coeff, mo_coeff, atol=1e-8):
            raise MatFileError(
                f"{path}: the beta MO coefficients differ from the alpha ones, "
                f"so this is a UHF reference. A spin-restricted FCIDUMP cannot "
                f"represent it. Generate UHF natural orbitals in Gaussian and "
                f"send those through instead."
            )

    # ------------------------------------------------------ active window
    eri_label = _require_label(keys, "eri_active", path)
    eri_flat = _expand(me.matlist[eri_label], eri_label, path)
    nact_from_eri = int(round(eri_flat.size ** 0.25))
    if nact_from_eri**4 != eri_flat.size:
        raise MatFileError(
            f"{path}: {eri_label} holds {eri_flat.size} elements, which is not "
            f"n^4 for any integer n. The two-electron block is packed in a way "
            f"this reader does not know; run scripts/inspect_mat.py and read its "
            f"packing candidates."
        )
    eri_active = np.ascontiguousarray(eri_flat.reshape((nact_from_eri,) * 4))

    # The window is stored exactly as the Gaussian route stated it: 1-based and
    # inclusive, so that a bundle records the job that was actually asked for.
    if window is not None:
        active_first, active_last = int(window[0]), int(window[1])
    else:
        nfc = int(_attr(me, ["nfc"], 0) or 0)
        nfv = int(_attr(me, ["nfv"], 0) or 0)
        active_first, active_last = nfc + 1, nmo - nfv

    nact = active_last - active_first + 1
    if nact != nact_from_eri:
        source = "the --window given" if window else "the file's nfc/nfv"
        raise MatFileError(
            f"{path}: active ERI dimension inconsistent with nact. {source} "
            f"implies {nact} active orbitals (MOs {active_first}-{active_last} "
            f"of {nmo}), but {eri_label} holds {nact_from_eri}^4 integrals, i.e. "
            f"{nact_from_eri} orbitals. The Window= keyword did not apply the "
            f"way the route intended. Do not work around this; fix the route."
        )

    # --------------------------------------------------------- Fock matrices
    fock_alpha = fock_beta = None
    fock_source = "none"
    if reference in KS_REFERENCES:
        # Deliberate: the stored KS matrix carries exchange-correlation and is
        # not the HF Fock operator. Storing it here would make it reachable by
        # accident, so a KS bundle carries none at all.
        ks_note = (
            "stored KS matrix not read: the active-space Hamiltonian is the HF "
            "Hamiltonian in the KS orbitals, so the Fock matrices must be "
            "rebuilt (see g16dump.hamiltonian.with_rebuilt_fock)"
        )
    else:
        ks_note = None
        alpha_label = find_label(keys, LABELS["fock_alpha"])
        if alpha_label is not None:
            fock_alpha = _square(
                _expand(me.matlist[alpha_label], alpha_label, path),
                nao, alpha_label, path,
            )
            fock_source = "gaussian"
            beta_fock_label = find_label(keys, LABELS["fock_beta"])
            if beta_fock_label is not None:
                fock_beta = _square(
                    _expand(me.matlist[beta_fock_label], beta_fock_label, path),
                    nao, beta_fock_label, path,
                )

    # ------------------------------------------------- diagnostics and scalars
    def _orbital_energies(quantity):
        label = find_label(keys, LABELS[quantity])
        if label is None:
            return None
        values = np.asarray(
            _expand(me.matlist[label], label, path), dtype=np.float64
        ).ravel()
        return values[:nmo] if values.size >= nmo else None

    enuc = _scalar(me, SCALARS["enuc"])
    if enuc is None:
        raise MatFileError(
            f"{path}: no nuclear repulsion energy (tried {SCALARS['enuc']}). "
            f"Run scripts/inspect_mat.py to see which scalar names this file "
            f"carries."
        )

    atom_charges = _attr(me, ["atmchg", "atomic_numbers", "ian"])
    if atom_charges is not None:
        atom_charges = np.asarray(atom_charges, dtype=np.float64).ravel()
    charge = int(_attr(me, ["icharg", "charge"], 0) or 0)
    if atom_charges is not None and atom_charges.size:
        charge = int(round(float(np.sum(atom_charges)))) - nelec

    provenance = make_provenance(
        source_file=str(path),
        fch_file=str(fch) if fch else None,
        route=route,
        basis=basis,
        method=method,
        window_1based=(active_first, active_last),
        fock_source=fock_source,
        extra={"ks_note": ks_note} if ks_note else None,
    )

    return Bundle(
        reference_type=reference,
        charge=charge,
        multiplicity=multiplicity,
        nelec=nelec,
        nalpha=nalpha,
        nbeta=nbeta,
        nao=nao,
        nmo=nmo,
        ncore=active_first - 1,
        nact=nact,
        active_first=active_first,
        active_last=active_last,
        enuc=enuc,
        C=mo_coeff,
        S=overlap,
        Hcore_ao=hcore,
        eri_active=eri_active,
        source_program="gaussian16",
        source_file=str(path),
        fock_source=fock_source,
        F_alpha_ao=fock_alpha,
        F_beta_ao=fock_beta,
        orbital_energies=_orbital_energies("orbital_energies"),
        orbital_energies_beta=_orbital_energies("orbital_energies_beta"),
        atom_charges=(
            atom_charges if atom_charges is not None and atom_charges.size else None
        ),
        escf=_scalar(me, SCALARS["escf"]),
        provenance=provenance,
    )


# ----------------------------------------------------------------- survey


def survey(path) -> dict:
    """Report whether ``extract`` can read this ``.mat``, and if not, why.

    Answers one question: are the blocks g16dump needs present, under what
    names, and are their dimensions consistent with each other? That is the M0
    gate in miniature, and it is what to run before an extraction.

    It is deliberately narrower than ``scripts/inspect_mat.py``, which dumps
    everything a ``.mat`` contains and assumes nothing about labels at all. Run
    that one when this reports something missing.
    """
    path = Path(path)
    me = read_matel(path)
    keys = list(me.matlist.keys())

    nao = int(_attr(me, ["nbasis"], 0) or 0)
    nmo = int(_attr(me, ["nbsuse"], nao) or nao)
    nfc = int(_attr(me, ["nfc"], 0) or 0)
    nfv = int(_attr(me, ["nfv"], 0) or 0)

    report = {
        "file": str(path),
        "header": {
            "nao": nao,
            "nmo": nmo,
            "nelec": int(_attr(me, ["ne"], 0) or 0),
            "multiplicity": int(_attr(me, ["multip", "multiplicity"], 0) or 0),
            "nfc": nfc,
            "nfv": nfv,
            "window_1based": [nfc + 1, nmo - nfv],
        },
        "blocks": {},
        "scalars": {},
        "matlist_keys": sorted(keys),
        "notes": [],
    }

    for quantity, candidates in LABELS.items():
        label = find_label(keys, candidates)
        entry = {"label": label, "required": quantity in _REQUIRED_QUANTITIES}
        if label is not None:
            try:
                entry["elements"] = int(np.asarray(me.matlist[label].array).size)
            except Exception:
                entry["elements"] = None
        report["blocks"][quantity] = entry

    for name, candidates in SCALARS.items():
        report["scalars"][name] = _scalar(me, candidates)

    missing = [q for q in _REQUIRED_QUANTITIES if report["blocks"][q]["label"] is None]
    if missing:
        report["notes"].append(
            f"extract cannot run: no block found for {', '.join(missing)}. If the "
            f"file carries them under other names, add those spellings to "
            f"g16dump.matfile.LABELS; run scripts/inspect_mat.py to see them all."
        )
    if report["scalars"].get("enuc") is None:
        report["notes"].append(
            "extract cannot run: no nuclear repulsion scalar found."
        )
    if report["blocks"]["fock_alpha"]["label"] is None:
        report["notes"].append(
            "no Fock matrix present, so the rebuilt-Fock path will be required. "
            "That is the normal situation for Kohn-Sham orbitals."
        )
    elif report["blocks"]["fock_beta"]["label"] is None:
        report["notes"].append(
            "an alpha Fock matrix is present but no beta one. That is fine for a "
            "closed shell and is missing information for an open shell."
        )
    if report["blocks"]["mo_coeff_beta"]["label"] is not None:
        report["notes"].append(
            "beta MO coefficients are present. If they differ from the alpha "
            "ones this is a UHF reference, which v1 cannot represent."
        )

    eri = report["blocks"]["eri_active"]
    if eri["label"] is not None and eri.get("elements"):
        nact_from_eri = int(round(eri["elements"] ** 0.25))
        if nact_from_eri**4 == eri["elements"]:
            report["eri_nact"] = nact_from_eri
            from_header = nmo - nfv - nfc
            if nact_from_eri != from_header:
                report["notes"].append(
                    f"active ERI dimension inconsistent with the header: the "
                    f"integrals hold {nact_from_eri} orbitals but nfc/nfv imply "
                    f"{from_header}. Window= did not apply the way the route "
                    f"intended; pass --window explicitly only if you know which "
                    f"is right."
                )
        else:
            report["notes"].append(
                f"the two-electron block holds {eri['elements']} elements, which "
                f"is not n^4 for any integer n; its packing is not one this "
                f"reader knows."
            )

    if not report["notes"]:
        report["notes"].append("every block extract needs is present and consistent.")
    return report


__all__ = [
    "LABELS",
    "survey",
    "SCALARS",
    "MatFileError",
    "extract",
    "find_label",
    "orient_mo_coeff",
    "read_matel",
]
