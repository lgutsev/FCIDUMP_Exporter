"""FCIDUMP writer.

**Seam only. The implementation lands in M3.** The signature is fixed here
because ``cli.py`` and the bundle schema are built against it; filling it in is
a separate piece of work with its own round-trip tests (write a FCIDUMP, read it
back with an independent parser, compare tensors, and confirm the reference
determinant energy evaluated from the file reproduces the SCF energy).

What the implementation owes, recorded here so it is not rediscovered:

* ``NORB``, ``NELEC``, ``MS2``, ``ORBSYM``, ``ISYM`` in the namelist header.
* Symmetry-unique two-electron entries only: for real orbitals the 8-fold
  permutational symmetry means writing ``t >= u``, ``v >= w``, ``(tu) >= (vw)``.
* An optional magnitude threshold below which an integral is omitted.
* The scalar core energy last, with all four indices zero.
* No hardcoded paths; ``path`` is whatever the caller passes.
"""

from __future__ import annotations

from collections.abc import Sequence

from .hamiltonian import ActiveHamiltonian


def write_fcidump(
    hamiltonian: ActiveHamiltonian,
    path,
    *,
    orbsym: Sequence[int] | None = None,
    isym: int = 1,
    threshold: float = 0.0,
    provenance: dict | None = None,
):
    """Write ``hamiltonian`` to ``path`` in FCIDUMP format.

    ``orbsym`` defaults to all-``1`` (C1) when not given; ``threshold`` omits
    two-electron integrals smaller than it in magnitude, and ``0.0`` writes
    every symmetry-unique one. ``provenance`` is written as comment lines above
    the namelist so a FCIDUMP can be traced back to the job that produced it.
    """
    raise NotImplementedError(
        "the FCIDUMP writer is M3 and is not implemented yet. The bundle and "
        "the active-space Hamiltonian are ready for it: see "
        "g16dump.hamiltonian.active_hamiltonian."
    )


__all__ = ["write_fcidump"]
