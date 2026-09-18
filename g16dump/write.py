"""FCIDUMP writer.

Takes the :class:`~g16dump.hamiltonian.ActiveHamiltonian` that the frozen-core
fold produces and writes the file Dice and Block2 read. Nothing here knows about
Gaussian, about ``.mat`` files or about where anything lives on disk: ``path``
is whatever the caller passes, and it is the only place a path comes from.

The format
----------
A Fortran namelist header, then the integrals, one per line, as a value and four
1-based orbital indices::

     &FCI NORB=  12,NELEC= 6,MS2= 2,
      ORBSYM=1,1,1,1,1,1,1,1,1,1,1,1,
      ISYM=1,
     &END
      <(tu|vw)>   t u v w      two-electron, symmetry-unique only
      <h'_tu>     t u 0 0      one-electron, lower triangle only
      <E_core>    0 0 0 0      the scalar, last

Real orbitals give the two-electron integrals eightfold permutational symmetry,
so only ``t >= u``, ``v >= w`` and ``(tu) >= (vw)`` are written: one entry in
eight. A reader reconstructs the rest. Writing all of them would not be wrong,
only four to eight times larger, and no reader expects it.

Precision
---------
Integrals are written with :data:`DEFAULT_FLOAT_FORMAT`, seventeen significant
digits, which round-trips an IEEE double exactly. The usual ``%.16g`` loses the
last bit, and a correlated calculation on a transition-metal active space is
precisely the setting where the last bits of the integrals are being asked to
carry a chemically meaningful energy difference.

Provenance
----------
FCIDUMP has no comment syntax. Readers in the field, PySCF's included, parse the
header by joining the first lines up to ``&END`` and splitting them on ``=``, so
a comment line above the namelist makes the file unreadable rather than merely
annotated. Provenance therefore goes in a sidecar JSON file beside the FCIDUMP,
named ``<path>.provenance.json``, and the FCIDUMP itself stays strictly
standard. :func:`provenance_path_for` names the sidecar for a given FCIDUMP.

Symmetry
--------
``orbsym`` defaults to all-``1``, i.e. C1. That is the honest default: the
windowed route does not carry Gaussian's irreducible-representation labels
through, and an active space that has been rotated (see
:mod:`g16dump.rotate`) has no symmetry left to label. Pass ``orbsym``
explicitly when the labels are known and the orbitals really are symmetry
adapted; a wrong ORBSYM makes a solver silently search the wrong symmetry
sector.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .hamiltonian import ActiveHamiltonian

#: Seventeen significant digits: enough to round-trip a float64 exactly.
DEFAULT_FLOAT_FORMAT = "%.16e"

#: ``max |h' - h'.T|`` tolerated before the writer refuses. ``h'`` is built from
#: symmetric matrices and a symmetric ERI contraction, so anything above this is
#: a bug upstream, not accumulated noise.
HERMITICITY_TOL = 1e-10

#: Permutational symmetry of the active ERIs. The writer keeps one entry in
#: eight and lets the reader rebuild the other seven, so a tensor that does not
#: actually have the symmetry would be silently truncated.
ERI_SYMMETRY_TOL = 1e-10


class WriteError(ValueError):
    """The Hamiltonian, or the options given with it, cannot produce a FCIDUMP."""


def provenance_path_for(path) -> Path:
    """The sidecar that carries ``path``'s provenance record."""
    return Path(str(path) + ".provenance.json")


# ----------------------------------------------------------------- checking


def _check_hamiltonian(h: ActiveHamiltonian) -> list:
    problems = []
    nact = int(h.nact)
    if nact < 1:
        problems.append(f"nact is {nact}; a FCIDUMP needs at least one orbital")
        return problems  # every shape below is expressed in terms of it

    h_eff = np.asarray(h.h_eff)
    eri = np.asarray(h.eri_active)
    if h_eff.shape != (nact, nact):
        problems.append(f"h_eff has shape {h_eff.shape}, expected {(nact, nact)}")
    if eri.shape != (nact,) * 4:
        problems.append(f"eri_active has shape {eri.shape}, expected {(nact,) * 4}")
    if problems:
        return problems  # numerics on a wrong shape report the wrong problem

    if not np.all(np.isfinite(h_eff)):
        problems.append("h_eff holds non-finite values")
    if not np.all(np.isfinite(eri)):
        problems.append("eri_active holds non-finite values")
    for name in ("e_core", "e_ref"):
        value = float(getattr(h, name))
        if not np.isfinite(value):
            problems.append(f"{name} is {value}")
    if problems:
        return problems

    deviation = float(np.max(np.abs(h_eff - h_eff.T)))
    if deviation > HERMITICITY_TOL:
        problems.append(
            f"h_eff is not symmetric: max |h - h.T| = {deviation:.3e} > "
            f"{HERMITICITY_TOL:.1e}. Only its lower triangle would be written, "
            f"so the upper one would be silently discarded."
        )
    for label, difference in {
        "(tu|vw) = (ut|vw)": eri - eri.transpose(1, 0, 2, 3),
        "(tu|vw) = (tu|wv)": eri - eri.transpose(0, 1, 3, 2),
        "(tu|vw) = (vw|tu)": eri - eri.transpose(2, 3, 0, 1),
    }.items():
        deviation = float(np.max(np.abs(difference))) if difference.size else 0.0
        if deviation > ERI_SYMMETRY_TOL:
            problems.append(
                f"eri_active violates {label} by {deviation:.3e} > "
                f"{ERI_SYMMETRY_TOL:.1e}. Only symmetry-unique entries are "
                f"written, so the violating ones would be discarded."
            )
    return problems


def _check_occupation(h: ActiveHamiltonian) -> list:
    problems = []
    nact, nelec, ms2 = int(h.nact), int(h.nelec_active), int(h.ms2)
    if nelec < 0:
        problems.append(f"NELEC would be {nelec}; the active space has no electrons")
    if nelec > 2 * nact:
        problems.append(
            f"NELEC {nelec} exceeds the {2 * nact} electrons {nact} spatial "
            f"orbitals can hold"
        )
    if abs(ms2) > nelec:
        problems.append(
            f"MS2 {ms2} needs {abs(ms2)} unpaired electrons but NELEC is {nelec}"
        )
    elif (nelec - ms2) % 2 != 0:
        problems.append(
            f"NELEC {nelec} and MS2 {ms2} have opposite parity, so they describe "
            f"no determinant"
        )
    else:
        nalpha = (nelec + ms2) // 2
        if nalpha > nact or nelec - nalpha > nact:
            problems.append(
                f"NELEC {nelec} with MS2 {ms2} puts {nalpha} alpha and "
                f"{nelec - nalpha} beta electrons in {nact} orbitals"
            )
    return problems


def _check_options(nact: int, orbsym, isym, threshold: float) -> list:
    problems = []
    if threshold < 0:
        problems.append(
            f"threshold {threshold} is negative; it is a magnitude below which "
            f"an integral is omitted, and 0.0 omits nothing"
        )
    if not np.isfinite(threshold):
        problems.append(f"threshold is {threshold}")
    if orbsym is not None:
        labels = list(orbsym)
        if len(labels) != nact:
            problems.append(
                f"orbsym has {len(labels)} labels for {nact} active orbitals"
            )
        elif any(int(label) < 1 for label in labels):
            problems.append(
                "orbsym labels are 1-based irrep numbers; all must be >= 1"
            )
    try:
        int(isym)
    except (TypeError, ValueError):
        problems.append(f"isym must be an integer, got {isym!r}")
    return problems


def _raise(problems: list) -> None:
    body = "\n".join(f"  - {p}" for p in problems)
    raise WriteError(f"cannot write a FCIDUMP:\n{body}")


# ------------------------------------------------------------------ writing


def fcidump_header(
    nact: int, nelec: int, ms2: int, orbsym: Sequence[int] | None, isym: int
) -> str:
    """The namelist, as the four lines every FCIDUMP reader expects to find.

    ORBSYM goes on one line however long it is. Readers cap the header at a
    fixed number of lines, so wrapping it is how a large active space becomes
    unreadable.
    """
    labels = [1] * nact if orbsym is None else [int(label) for label in orbsym]
    return (
        f" &FCI NORB={nact:4d},NELEC={nelec:2d},MS2={ms2:d},\n"
        f"  ORBSYM={','.join(str(label) for label in labels)},\n"
        f"  ISYM={int(isym):d},\n"
        f" &END\n"
    )


def _two_electron_lines(eri: np.ndarray, threshold: float, fmt: str):
    """Yield the symmetry-unique two-electron lines, in canonical order.

    One composite-index row at a time: an ``nact`` of 50 has 1275 pairs and
    813 000 unique integrals, and materialising the whole pair-by-pair matrix
    to filter it costs more memory than the file costs disk.
    """
    rows, cols = np.tril_indices(eri.shape[0])  # t >= u, in FCIDUMP's own order
    for ij in range(rows.size):
        t, u = int(rows[ij]), int(cols[ij])
        take = slice(0, ij + 1)  # (tu) >= (vw)
        values = eri[t, u][rows[take], cols[take]]
        if threshold > 0.0:
            keep = np.flatnonzero(np.abs(values) >= threshold)
        else:
            keep = np.arange(values.size)
        if keep.size == 0:
            continue
        t1, u1 = t + 1, u + 1
        yield "".join(
            f"{fmt % values[kl]} {t1:4d} {u1:4d} {rows[kl] + 1:4d} "
            f"{cols[kl] + 1:4d}\n"
            for kl in keep
        )


def _one_electron_lines(h_eff: np.ndarray, fmt: str) -> str:
    """The lower triangle of ``h'``, written in full.

    ``threshold`` deliberately does not reach here. The one-electron block is
    ``nact(nact+1)/2`` entries against the two-electron block's millions, so
    dropping any of it buys no file size and changes the Hamiltonian.
    """
    rows, cols = np.tril_indices(h_eff.shape[0])
    return "".join(
        f"{fmt % h_eff[t, u]} {t + 1:4d} {u + 1:4d} {0:4d} {0:4d}\n"
        for t, u in zip(rows, cols)
    )


def write_fcidump(
    hamiltonian: ActiveHamiltonian,
    path,
    *,
    orbsym: Sequence[int] | None = None,
    isym: int = 1,
    threshold: float = 0.0,
    provenance: dict | None = None,
    float_format: str = DEFAULT_FLOAT_FORMAT,
) -> Path:
    """Write ``hamiltonian`` to ``path`` in FCIDUMP format. Return ``path``.

    ``orbsym`` defaults to all-``1`` (C1); see the module docstring on why that
    is the honest default here. ``threshold`` omits **two-electron** integrals
    smaller than it in magnitude, and ``0.0`` writes every symmetry-unique one.
    ``provenance`` is written to the sidecar :func:`provenance_path_for` names,
    because FCIDUMP has no comment syntax that readers agree on.

    Raises :class:`WriteError` when the Hamiltonian could not produce a file
    that means what it says: a non-symmetric ``h'`` or an ERI tensor without its
    permutational symmetry would lose exactly the entries this writer does not
    put on disk.
    """
    problems = _check_hamiltonian(hamiltonian)
    problems += _check_occupation(hamiltonian)
    problems += _check_options(int(hamiltonian.nact), orbsym, isym, float(threshold))
    if problems:
        _raise(problems)

    h_eff = np.ascontiguousarray(hamiltonian.h_eff, dtype=np.float64)
    eri = np.ascontiguousarray(hamiltonian.eri_active, dtype=np.float64)

    path = Path(path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as out:
        out.write(
            fcidump_header(
                int(hamiltonian.nact),
                int(hamiltonian.nelec_active),
                int(hamiltonian.ms2),
                orbsym,
                isym,
            )
        )
        for chunk in _two_electron_lines(eri, float(threshold), float_format):
            out.write(chunk)
        out.write(_one_electron_lines(h_eff, float_format))
        out.write(f"{float_format % float(hamiltonian.e_core)} {0:4d} {0:4d} "
                  f"{0:4d} {0:4d}\n")

    if provenance is not None:
        write_provenance(path, hamiltonian, provenance, threshold=float(threshold))
    return path


def write_provenance(
    path, hamiltonian: ActiveHamiltonian, provenance: dict, *, threshold: float = 0.0
) -> Path:
    """Write the sidecar record for the FCIDUMP at ``path``. Return its path.

    The bundle's own provenance is carried through unchanged and the facts about
    this particular file are added beside it, so a FCIDUMP found on a cluster
    six months from now still says which Gaussian job it came from, whether its
    Fock matrices were Gaussian's or rebuilt, and what was thrown away.
    """
    record = dict(provenance)
    record["fcidump"] = {
        "file": Path(path).name,
        "norb": int(hamiltonian.nact),
        "nelec": int(hamiltonian.nelec_active),
        "ms2": int(hamiltonian.ms2),
        "e_core": float(hamiltonian.e_core),
        "e_ref": float(hamiltonian.e_ref),
        "fock_source": hamiltonian.fock_source,
        "spin_deviation": float(hamiltonian.spin_deviation),
        "eri_threshold": float(threshold),
    }
    sidecar = provenance_path_for(path)
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
    return sidecar


__all__ = [
    "DEFAULT_FLOAT_FORMAT",
    "ERI_SYMMETRY_TOL",
    "HERMITICITY_TOL",
    "WriteError",
    "fcidump_header",
    "provenance_path_for",
    "write_fcidump",
    "write_provenance",
]
