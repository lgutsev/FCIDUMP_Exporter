"""FCIDUMP writer.

The format is Knowles and Handy's (Comput. Phys. Commun. **54**, 75 (1989)): a
Fortran namelist header followed by one record per integral, each a value and
four 1-based orbital indices. This writer emits the **spin-restricted** form --
one set of spatial orbitals, no ``IUHF`` -- which is the right form here because
``h'`` is spin-independent by construction and ``hamiltonian.py`` refuses to
build it when it is not.

Three things about the output are worth stating plainly, because each is a place
where a wrong file still looks like a right one:

*Only symmetry-unique two-electron integrals are written.* Real orbitals give
``(tu|vw)`` the full 8-fold permutational symmetry, so the file carries one
element per orbit and the reader scatters it back. The enumeration is
``t >= u``, ``v >= w`` and ``(tu) >= (vw)`` on the composite pair index -- the
same set the legacy script wrote, reached here without a Python loop over
integrals.

*``E_core`` is the last record, with all four indices zero.* It already contains
the nuclear repulsion; see :func:`g16dump.hamiltonian.active_hamiltonian`.

*A threshold changes what the file means.* Dropping small two-electron integrals
makes the file no longer reproduce ``E_ref`` exactly, and the error is not
bounded by the threshold alone. The default is ``0.0`` for that reason.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .bundle import ERI_SYMMETRY_TOL
from .hamiltonian import ActiveHamiltonian

#: ``max |h' - h'.T|`` tolerated before the one-electron block is refused. ``h'``
#: is symmetric by construction and is computed here, not read from a file, so
#: this is tighter than the ERI tolerance. It is not zero: the transforms behind
#: ``h'`` leave asymmetries of order machine epsilon, and the file stores one
#: triangle, so the writer's job is to confirm the discarded triangle was
#: redundant, not to demand it be bit-identical.
H_SYMMETRY_TOL = 1e-10

#: Digits after the decimal point that make an IEEE-754 double round-trip
#: exactly. A ``%.16E`` field carries 17 significant digits, which is the
#: number that distinguishes every double from its neighbours.
ROUND_TRIP_PRECISION = 16


class WriteError(ValueError):
    """The Hamiltonian cannot be written as a valid FCIDUMP."""


def _check(
    hamiltonian: ActiveHamiltonian,
    orbsym: Sequence[int],
    isym: int,
    threshold: float,
    precision: int,
) -> None:
    """Refuse anything that would produce a file a solver misreads."""
    problems: list[str] = []
    nact = int(hamiltonian.nact)

    if nact < 1:
        raise WriteError(f"nact is {nact}; there is nothing to write")

    h = np.asarray(hamiltonian.h_eff)
    eri = np.asarray(hamiltonian.eri_active)
    if h.shape != (nact, nact):
        problems.append(f"h_eff has shape {h.shape}, expected {(nact, nact)}")
    if eri.shape != (nact,) * 4:
        problems.append(f"eri_act has shape {eri.shape}, expected {(nact,) * 4}")
    if problems:
        _raise(problems)

    if not np.all(np.isfinite(h)):
        problems.append("h_eff holds non-finite values")
    if not np.all(np.isfinite(eri)):
        problems.append("eri_act holds non-finite values")
    if not np.isfinite(hamiltonian.e_core):
        problems.append(f"e_core is {hamiltonian.e_core}")

    deviation = float(np.max(np.abs(h - h.T))) if h.size else 0.0
    if deviation > H_SYMMETRY_TOL:
        problems.append(
            f"h_eff is not symmetric: max |h - h.T| = {deviation:.3e}. Only the "
            f"lower triangle is written, so the upper one would be silently lost"
        )

    nelec, ms2 = int(hamiltonian.nelec_active), int(hamiltonian.ms2)
    if nelec < 0:
        problems.append(f"nelec_act is {nelec}")
    if ms2 < 0:
        problems.append(
            f"ms2 is {ms2}; this bundle format takes the high-spin convention"
        )
    if ms2 > nelec:
        problems.append(f"ms2 {ms2} needs more unpaired electrons than {nelec}")
    if (nelec - ms2) % 2 != 0:
        problems.append(
            f"{nelec} active electrons cannot carry MS2 {ms2}: the parity is wrong"
        )
    if nelec > 2 * nact:
        problems.append(
            f"{nelec} active electrons do not fit in {nact} active orbitals"
        )
    # The Pauli bound on each spin string separately. The total-electron check
    # above passes things this one catches: 10 electrons with MS2 = 8 is 9 alpha
    # and 1 beta, which fits in 6 orbitals on the total count and not at all on
    # the alpha one.
    nalpha, nbeta = (nelec + ms2) // 2, (nelec - ms2) // 2
    if nalpha > nact or nbeta > nact:
        problems.append(
            f"NELEC {nelec} with MS2 {ms2} is {nalpha} alpha and {nbeta} beta "
            f"electrons, which does not fit in {nact} spatial orbitals: at most "
            f"one electron of each spin per orbital"
        )

    if len(orbsym) != nact:
        problems.append(
            f"orbsym has {len(orbsym)} entries but there are {nact} active "
            f"orbitals; ORBSYM must name an irrep for every one"
        )
    elif not all(1 <= int(s) <= 8 for s in orbsym):
        problems.append(
            f"orbsym holds {sorted({int(s) for s in orbsym if not 1 <= int(s) <= 8})}, "
            f"outside 1-8. FCIDUMP irrep labels are 1-based Molpro indices into a "
            f"table of at most eight irreps, and C1 is all ones; 0 in particular "
            f"is read as an out-of-range index rather than as 'no symmetry'"
        )
    if not 1 <= int(isym) <= 8:
        problems.append(
            f"isym is {isym}, outside 1-8. It is a 1-based Molpro irrep label "
            f"naming the target state's symmetry, from the same table as "
            f"orbsym, and C1 is 1"
        )

    if not np.isfinite(threshold) or threshold < 0.0:
        problems.append(f"threshold must be finite and non-negative, got {threshold}")
    if not 1 <= int(precision) <= 17:
        problems.append(
            f"precision must be between 1 and 17 digits, got {precision} "
            f"({ROUND_TRIP_PRECISION} is the smallest that round-trips a double)"
        )

    if problems:
        _raise(problems)


def _raise(problems: list[str]) -> None:
    body = "\n".join(f"  - {p}" for p in problems)
    raise WriteError(f"refusing to write an FCIDUMP:\n{body}")


def _header(
    nact: int, nelec: int, ms2: int, orbsym: Sequence[int], isym: int
) -> str:
    """The ``&FCI`` namelist, spelled the way the file Dice already read spells it.

    Three details are load-bearing and none of them is cosmetic.

    ``NORB``, ``NELEC`` and ``MS2`` share the first line, **separated by
    whitespace**. Dice does not parse the namelist; it splits the line on
    whitespace and reads the three values out of tokens 1-6. Written compactly
    as ``NORB=6,NELEC=8,MS2=0,`` the whole thing is a single token and Dice
    leaves ``norbs`` unset. ``legacy/FePorph_3_Window.dat``, which Dice
    consumed, has the spaces; so does this.

    ``ORBSYM`` carries a **trailing comma** after its last entry and sits on one
    physical line. Readers that match ``(\\d+),`` per orbital need the final
    comma, and one that scans only the first few lines for the terminator can
    lose a wrapped list.

    ``&END`` is alone on its line and uppercase: Dice requires the terminator to
    be the only token on it, and MOLPRO compares case-sensitively. A bare ``/``
    would also be legal but is avoided, because a reader that stops at any line
    containing a slash is then at the mercy of every other line.
    """
    return (
        f"&FCI NORB= {nact:d}, NELEC= {nelec:d}, MS2= {ms2:d},\n"
        f" ORBSYM={','.join(str(int(s)) for s in orbsym)},\n"
        f" ISYM={int(isym):d},\n"
        f"&END\n"
    )


def _pair_indices(nact: int) -> tuple[np.ndarray, np.ndarray]:
    """``(t, u)`` with ``t >= u``, ordered by the composite index ``t(t+1)/2+u``.

    ``np.tril_indices`` walks the lower triangle row by row, which is exactly
    that order, so the position in these arrays *is* the composite pair index.
    """
    return np.tril_indices(nact)


def _record_lines(
    values: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    value_format: str,
) -> list[str]:
    """Format one chunk of records. Indices arrive 0-based and are written 1-based."""
    return [
        f"{value_format % value} {ti + 1:4d} {ui + 1:4d} {vi + 1:4d} {wi + 1:4d}"
        for value, ti, ui, vi, wi in zip(values, t, u, v, w)
    ]


def provenance_path(path) -> Path:
    """Where the provenance for the FCIDUMP at ``path`` is written."""
    path = Path(path)
    return path.with_name(path.name + ".provenance.json")


def _write_provenance(path, provenance: dict) -> Path:
    """Write provenance beside the FCIDUMP, never inside it.

    FCIDUMP has no comment syntax, and both obvious places to smuggle one in are
    unsafe. Above the namelist: readers find the end of the header by scanning
    the first few lines for ``&END`` or a bare ``/``, and a provenance record is
    full of ``/`` in paths and dates, so a comment there can truncate the header
    on a reader that scans that way -- silently, because the file still looks
    well formed. Below the last record: readers that parse every remaining line
    as ``value i j k l`` do not stop at the scalar record, so a trailer makes
    them fail on a line that is not an integral.

    A sidecar cannot break either. The cost is that it is a second file, so the
    CLI names it on the way out and the path is derived from the FCIDUMP's, not
    chosen independently.
    """
    target = provenance_path(path)
    target.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def write_fcidump(
    hamiltonian: ActiveHamiltonian,
    path,
    *,
    orbsym: Sequence[int] | None = None,
    isym: int = 1,
    threshold: float = 0.0,
    precision: int = ROUND_TRIP_PRECISION,
    provenance: dict | None = None,
) -> Path:
    """Write ``hamiltonian`` to ``path`` in FCIDUMP format. Return ``path``.

    ``orbsym`` defaults to all-``1`` (C1) when not given; ``threshold`` omits
    two-electron integrals smaller than it in magnitude, and ``0.0`` writes
    every symmetry-unique one. ``provenance``, when given, is written as JSON to
    :func:`provenance_path` beside the FCIDUMP -- never into it, for the reasons
    in :func:`_write_provenance` -- so a FCIDUMP can be traced back to the job
    that produced it.

    ``precision`` is the number of digits after the decimal point in each value.
    The default, ``16``, is the smallest that reproduces a double exactly; lower
    values make a smaller file that no longer round-trips.

    The threshold applies to the two-electron block alone. The one-electron
    block is ``nact(nact+1)/2`` numbers however large the active space gets, so
    thinning it saves nothing worth the risk of dropping an element of ``h'``.
    """
    nact = int(hamiltonian.nact)
    if orbsym is None:
        orbsym = [1] * nact
    else:
        orbsym = [int(s) for s in orbsym]

    _check(hamiltonian, orbsym, isym, threshold, precision)

    h = np.ascontiguousarray(hamiltonian.h_eff, dtype=np.float64)
    eri = np.ascontiguousarray(hamiltonian.eri_active, dtype=np.float64)
    value_format = f"%{int(precision) + 8}.{int(precision)}E"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pair_t, pair_u = _pair_indices(nact)
    npair = pair_t.size

    # Written to a neighbouring temporary and moved into place only once it is
    # complete. Streaming straight into `path` would truncate whatever was
    # there on the first write and leave a half-written file behind on any
    # failure -- and the symmetry check inside the loop below can fail at any
    # bra pair. A truncated FCIDUMP is not obviously broken to a reader: it
    # parses, and the integrals it does contain are right.
    scratch = path.with_name(path.name + ".partial")
    try:
        _write_records(
            scratch, hamiltonian, h, eri, orbsym, isym, threshold,
            value_format, pair_t, pair_u, npair, nact,
        )
        scratch.replace(path)
    except BaseException:
        try:
            scratch.unlink()
        except OSError:
            pass
        raise

    if provenance:
        _write_provenance(path, provenance)
    else:
        # A sidecar left over from an earlier write describes orbitals this file
        # no longer contains. Removing it is the honest outcome: no provenance
        # is recoverable, and stale provenance is worse than none.
        stale = provenance_path(path)
        if stale.exists():
            stale.unlink()

    return path


def _write_records(
    path, hamiltonian, h, eri, orbsym, isym, threshold, value_format,
    pair_t, pair_u, npair, nact,
) -> None:
    """Stream the namelist and every record to ``path``."""
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(
            _header(nact, int(hamiltonian.nelec_active), int(hamiltonian.ms2),
                    orbsym, isym)
        )

        # Two-electron block. One chunk per bra pair, so the largest temporary
        # is npair long rather than npair^2: an active space big enough to
        # matter would not fit the whole unique set in memory at once.
        for bra in range(npair):
            t, u = pair_t[bra], pair_u[bra]
            v, w = pair_t[: bra + 1], pair_u[: bra + 1]
            values = eri[t, u, v, w]

            _assert_eightfold(eri, t, u, v, w, values)

            if threshold > 0.0:
                keep = np.abs(values) >= threshold
                if not keep.any():
                    continue
                values, v, w = values[keep], v[keep], w[keep]

            lines = _record_lines(
                values, np.full(v.shape, t), np.full(v.shape, u), v, w,
                value_format,
            )
            handle.write("\n".join(lines) + "\n")

        # One-electron block: the lower triangle of h', every element of it.
        values = h[pair_t, pair_u]
        lines = [
            f"{value_format % value} {t + 1:4d} {u + 1:4d}    0    0"
            for value, t, u in zip(values, pair_t, pair_u)
        ]
        handle.write("\n".join(lines) + "\n")

        # The scalar, last, with four zero indices.
        handle.write(f"{value_format % hamiltonian.e_core}    0    0    0    0\n")


def _assert_eightfold(
    eri: np.ndarray,
    t: int,
    u: int,
    v: np.ndarray,
    w: np.ndarray,
    values: np.ndarray,
) -> None:
    """Check the three generators of the 8-fold group at the elements written.

    Only one element of each orbit reaches the file, so a tensor that is not
    symmetric loses information silently -- the worst way for this to fail. The
    check costs three gathers per bra pair and no extra memory, and it runs on
    exactly the elements whose values are about to be committed.

    This is the generators at each symmetry-unique element, not the full
    ``nact**4`` comparison; that one belongs to
    :func:`g16dump.bundle.validate`, which every loaded bundle passes.

    The tolerance is the schema's own :data:`g16dump.bundle.ERI_SYMMETRY_TOL`,
    imported rather than restated. A writer stricter than ``validate`` would
    refuse bundles this project calls valid.
    """
    for label, other in (
        ("(tu|vw) = (ut|vw)", eri[u, t, v, w]),
        ("(tu|vw) = (tu|wv)", eri[t, u, w, v]),
        ("(tu|vw) = (vw|tu)", eri[v, w, t, u]),
    ):
        deviation = float(np.max(np.abs(values - other))) if values.size else 0.0
        if deviation > ERI_SYMMETRY_TOL:
            raise WriteError(
                f"eri_act violates {label} by {deviation:.3e} at bra pair "
                f"({t + 1}, {u + 1}). Only symmetry-unique integrals are "
                f"written, so this would be lost rather than reported by the "
                f"solver."
            )


def dice_occupation_line(hamiltonian: ActiveHamiltonian) -> str:
    """Dice's ``nocc`` block for the reference determinant of ``hamiltonian``.

    This is part of Dice's ``input.dat``, not of the FCIDUMP, and it is here
    because it is the one piece of Dice's input that depends on the active space
    rather than on the calculation. Dice indexes **spin** orbitals: spatial
    orbital ``p`` is alpha ``2p`` and beta ``2p+1``, both 0-based. The reference
    determinant fills the lowest spatial orbitals, alpha first::

        nocc 5
        0 2 4 1 3
        end

    is five electrons in three spatial orbitals with MS2 = 1 -- the alpha string
    then the beta string, which is the order the legacy ``input.dat`` uses.
    """
    nelec, ms2 = int(hamiltonian.nelec_active), int(hamiltonian.ms2)
    nalpha = (nelec + ms2) // 2
    nbeta = nelec - nalpha
    spin_orbitals = [2 * p for p in range(nalpha)] + [2 * p + 1 for p in range(nbeta)]
    return (
        f"nocc {nelec}\n"
        f"{' '.join(str(s) for s in spin_orbitals)}\n"
        f"end\n"
    )


__all__ = [
    "H_SYMMETRY_TOL",
    "ROUND_TRIP_PRECISION",
    "WriteError",
    "dice_occupation_line",
    "provenance_path",
    "write_fcidump",
]
