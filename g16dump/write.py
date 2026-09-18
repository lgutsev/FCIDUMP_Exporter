"""FCIDUMP output, and the Dice reference-determinant line that goes with it.

Format, in the order solvers expect it:

1. the ``&FCI ... &END`` namelist header (``NORB``, ``NELEC``, ``MS2``,
   ``ORBSYM``, ``ISYM``),
2. the symmetry-unique two-electron integrals, ``value i j k l``,
3. the one-electron integrals, ``value i j 0 0``,
4. the scalar core energy, ``value 0 0 0 0``.

Indices are 1-based in the file and 0-based everywhere in this package.

Only the symmetry-unique two-electron integrals are written: ``i >= j``,
``k >= l``, and ``ij >= kl`` as composite pair indices, which is
``npair*(npair+1)/2`` entries for ``npair = norb*(norb+1)/2``. For 21 orbitals
that is 26796 entries, matching ``legacy/FePorph_1_Window.dat`` exactly.

Writing is vectorized and chunked. Index arrays are built with NumPy and
formatted in bulk; for the 100-orbital target (12,753,775 two-electron lines)
this runs in roughly ten seconds rather than the hours a per-element Python loop
would take, and peak memory stays bounded by the chunk size rather than the
total line count.
"""

from __future__ import annotations

from typing import Iterator, Optional, Sequence

import numpy as np

from .errors import ValidationError, require
from .hamiltonian import ActiveSpaceHamiltonian

#: Two-electron and one-electron line formats. Whitespace-separated, so the
#: field widths are cosmetic; the 16-digit mantissa is not.
_ERI_FMT = "%23.16E %4d %4d %4d %4d\n"
_ONE_FMT = "%23.16E %4d %4d %4d %4d\n"

#: Rows formatted per chunk. Bounds peak memory independently of norb.
_CHUNK_ROWS = 500_000


def unique_pair_indices(norb: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/column indices of the lower triangle, in composite-pair order.

    Pair ``p`` is ``(i, j)`` with ``i >= j`` and ``p = i*(i+1)/2 + j``, which is
    the ordering the composite-index inequality ``ij >= kl`` refers to.
    """
    return np.tril_indices(norb)


def count_unique_eri(norb: int) -> int:
    """Number of symmetry-unique two-electron integrals for ``norb`` orbitals."""
    npair = norb * (norb + 1) // 2
    return npair * (npair + 1) // 2


def _pair_chunks(npair: int, chunk_rows: int) -> Iterator[np.ndarray]:
    """Yield blocks of pair indices P whose expanded row count stays bounded.

    Pair ``P`` expands to ``P + 1`` rows (every ``Q <= P``), so a fixed number of
    P values per chunk would give wildly uneven memory use. This picks the block
    size from the running row count instead.
    """
    start = 0
    while start < npair:
        # Pair P expands to P+1 rows, so a block [start, start+b) costs
        # sum_{P=start}^{start+b-1} (P+1) <= b*(start+b) rows. Bound that by
        # chunk_rows and solve the quadratic for b.
        block = int((-start + (start * start + 4 * chunk_rows) ** 0.5) / 2)
        block = max(1, min(npair - start, block))
        end = start + block
        yield np.arange(start, end, dtype=np.int64)
        start = end


def _expand_pair_block(p_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each P in the block, produce every Q with 0 <= Q <= P.

    Vectorized ragged arange: no Python loop over pairs.
    """
    counts = p_block + 1
    total = int(counts.sum())
    offsets = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    q = np.arange(total, dtype=np.int64) - np.repeat(offsets, counts)
    p = np.repeat(p_block, counts)
    return p, q


def iter_unique_eri(
    eri: np.ndarray, threshold: float = 0.0, chunk_rows: int = _CHUNK_ROWS
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(values, i, j, k, l)`` blocks of symmetry-unique integrals.

    Indices are 0-based. ``threshold`` drops entries with ``|v| < threshold``;
    the default of 0.0 writes everything, including exact zeros, because some
    solvers use the line count as a consistency check.
    """
    norb = eri.shape[0]
    ip, jp = unique_pair_indices(norb)
    npair = len(ip)

    for p_block in _pair_chunks(npair, chunk_rows):
        p, q = _expand_pair_block(p_block)
        i, j, k, l = ip[p], jp[p], ip[q], jp[q]
        values = eri[i, j, k, l]
        if threshold > 0.0:
            keep = np.abs(values) >= threshold
            if not keep.any():
                continue
            values, i, j, k, l = values[keep], i[keep], j[keep], k[keep], l[keep]
        yield values, i, j, k, l


def format_header(
    norb: int,
    nelec: int,
    ms2: int,
    orbsym: Optional[Sequence[int]] = None,
    isym: int = 1,
) -> str:
    """The ``&FCI ... &END`` namelist block.

    ``orbsym`` defaults to all ones, which is what a ``NoSymm`` Gaussian job
    implies and what the legacy dumps carry.
    """
    if orbsym is None:
        orbsym = [1] * norb
    require(
        len(orbsym) == norb,
        f"ORBSYM has {len(orbsym)} entries but NORB is {norb}. Every orbital "
        f"needs a symmetry label.",
    )
    lines = [
        f"&FCI NORB={norb:4d},NELEC={nelec:3d},MS2={ms2:2d},\n",
        "  ORBSYM=" + ",".join(str(int(s)) for s in orbsym) + ",\n",
        f"  ISYM={int(isym)},\n",
        "&END\n",
    ]
    return "".join(lines)


def write_fcidump(
    path: str,
    h_eff: np.ndarray,
    eri: np.ndarray,
    e_core: float,
    nelec: int,
    ms2: int,
    *,
    orbsym: Optional[Sequence[int]] = None,
    isym: int = 1,
    threshold: float = 0.0,
    chunk_rows: int = _CHUNK_ROWS,
) -> int:
    """Write an FCIDUMP. Returns the number of integral lines written.

    No path is constructed here: ``path`` is used exactly as given.
    """
    h_eff = np.asarray(h_eff)
    eri = np.asarray(eri)
    norb = h_eff.shape[0]

    require(
        h_eff.shape == (norb, norb),
        f"h_eff must be square, got {h_eff.shape}.",
    )
    require(
        eri.shape == (norb, norb, norb, norb),
        f"eri has shape {eri.shape}, which does not match h_eff's {norb} "
        f"orbitals. The one- and two-electron integrals must span the same "
        f"active space.",
    )
    require(
        nelec >= 0,
        f"NELEC must be non-negative, got {nelec}.",
    )
    require(
        abs(ms2) <= nelec and (nelec - abs(ms2)) % 2 == 0,
        f"MS2={ms2} is impossible for NELEC={nelec}: |MS2| cannot exceed NELEC "
        f"and NELEC - |MS2| must be even (it is twice the number of paired "
        f"electrons).",
    )

    written = 0
    with open(path, "w") as handle:
        handle.write(format_header(norb, nelec, ms2, orbsym, isym))

        for values, i, j, k, l in iter_unique_eri(eri, threshold, chunk_rows):
            handle.write(
                "".join(
                    [
                        _ERI_FMT % row
                        for row in zip(
                            values.tolist(),
                            (i + 1).tolist(),
                            (j + 1).tolist(),
                            (k + 1).tolist(),
                            (l + 1).tolist(),
                        )
                    ]
                )
            )
            written += len(values)

        ih, jh = np.tril_indices(norb)
        one_values = h_eff[ih, jh]
        if threshold > 0.0:
            keep = np.abs(one_values) >= threshold
            one_values, ih, jh = one_values[keep], ih[keep], jh[keep]
        handle.write(
            "".join(
                [
                    _ONE_FMT % row
                    for row in zip(
                        one_values.tolist(),
                        (ih + 1).tolist(),
                        (jh + 1).tolist(),
                        [0] * len(ih),
                        [0] * len(ih),
                    )
                ]
            )
        )
        written += len(one_values)

        handle.write(_ONE_FMT % (float(e_core), 0, 0, 0, 0))
        written += 1

    return written


def write_from_hamiltonian(
    path: str, ash: ActiveSpaceHamiltonian, **kwargs
) -> int:
    """Convenience wrapper: write the FCIDUMP for a built active-space Hamiltonian."""
    return write_fcidump(
        path,
        ash.h_eff,
        ash.eri_act,
        ash.e_core,
        ash.nelec_act,
        ash.ms2,
        **kwargs,
    )


# ------------------------------------------------------ Dice reference line


def dice_nocc_block(nocc_a: int, nocc_b: int) -> str:
    """The Dice ``input.dat`` ``nocc`` block for the reference determinant.

    Spin-orbital indices: alpha orbital ``t`` is ``2t``, beta orbital ``t`` is
    ``2t + 1``, matching ``legacy/input.dat``. Writing this alongside the
    FCIDUMP removes a silent failure mode -- a reference determinant
    inconsistent with the Hamiltonian converges to the wrong thing quietly.
    """
    require(
        nocc_a >= 0 and nocc_b >= 0,
        f"occupation counts must be non-negative, got {nocc_a}, {nocc_b}.",
    )
    alpha = [2 * t for t in range(nocc_a)]
    beta = [2 * t + 1 for t in range(nocc_b)]
    occupied = " ".join(str(x) for x in alpha + beta)
    return f"nocc {nocc_a + nocc_b}\n{occupied}\nend\n"


def write_dice_nocc(path: str, ash: ActiveSpaceHamiltonian) -> None:
    """Write just the ``nocc`` block to ``path``."""
    with open(path, "w") as handle:
        handle.write(dice_nocc_block(ash.nocc_a_act, ash.nocc_b_act))
