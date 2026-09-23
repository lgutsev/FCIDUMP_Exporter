"""Independent readers and solvers for the FCIDUMP round-trip and rotation gates.

Nothing here imports :mod:`g16dump`. That is the point: a round-trip test whose
reader shares code with the writer tests that the code is self-consistent, not
that the file says what it claims. These are written from the FCIDUMP format
description and from the Slater-Condon rules, and they exist only to disagree
with the production code if the production code is wrong.

:func:`read_fcidump` parses the file. :func:`determinant_energy` evaluates the
reference determinant from the parsed tensors. :func:`exact_spectrum`
diagonalises the many-electron Hamiltonian in the full determinant basis, which
is what the rotation-invariance gate needs and what makes it independent of
PySCF as well: it runs in the numpy-only CI job.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

_HEADER_KEYS_INT = ("NORB", "NELEC", "MS2", "ISYM")


def read_fcidump(path) -> dict:
    """Parse a FCIDUMP into ``{NORB, NELEC, MS2, ISYM, ORBSYM, H1, H2, ECORE}``.

    ``H1`` comes back as a full symmetric ``(norb, norb)`` array and ``H2`` as a
    full ``(norb,) * 4`` tensor in chemist notation, both rebuilt from the
    symmetry-unique entries the file carries. Anything the file leaves out is
    zero, which is exactly what a solver would assume.
    """
    lines = Path(path).read_text().splitlines()

    header_tokens: list = []
    body_start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        header_tokens.append(stripped.replace("&FCI", "").replace("&END", ""))
        if "&END" in stripped or stripped == "/":
            body_start = index + 1
            break
    if body_start is None:
        raise ValueError(f"{path}: no &END terminating the FCIDUMP namelist")

    header: dict = {}
    key = None
    for token in ",".join(header_tokens).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            key, _, value = token.partition("=")
            key = key.strip().upper()
            header[key] = [value.strip()] if value.strip() else []
        elif key is not None:
            header[key].append(token)

    result = {name: int(header[name][0]) for name in _HEADER_KEYS_INT if name in header}
    result["ORBSYM"] = [int(value) for value in header.get("ORBSYM", [])]

    norb = result["NORB"]
    h1 = np.zeros((norb, norb))
    h2 = np.zeros((norb,) * 4)
    result["ECORE"] = 0.0

    for line in lines[body_start:]:
        fields = line.split()
        if not fields:
            continue
        value = float(fields[0])
        t, u, v, w = (int(field) for field in fields[1:5])
        if v != 0 or w != 0:
            # Rebuild all eight permutations; assigning the same entry twice
            # with the same value is harmless and simpler than deduplicating.
            for a, b, c, d in (
                (t, u, v, w), (u, t, v, w), (t, u, w, v), (u, t, w, v),
                (v, w, t, u), (w, v, t, u), (v, w, u, t), (w, v, u, t),
            ):
                h2[a - 1, b - 1, c - 1, d - 1] = value
        elif u != 0:
            h1[t - 1, u - 1] = h1[u - 1, t - 1] = value
        else:
            result["ECORE"] = value

    result["H1"] = h1
    result["H2"] = h2
    return result


def spin_occupations(nelec: int, ms2: int) -> tuple:
    """``(nalpha, nbeta)`` from a FCIDUMP's NELEC and MS2."""
    nalpha = (nelec + ms2) // 2
    return nalpha, nelec - nalpha


def determinant_energy(h1, h2, ecore: float, nalpha: int, nbeta: int) -> float:
    """Energy of the determinant occupying the lowest ``nalpha``/``nbeta`` orbitals.

    Written out term by term from the Slater rules rather than contracted with
    einsum, so that it shares no structure with the production ``active_energy``
    it is checked against.
    """
    energy = float(ecore)
    for i in range(nalpha):
        energy += h1[i, i]
    for i in range(nbeta):
        energy += h1[i, i]
    for nocc in (nalpha, nbeta):
        for i in range(nocc):
            for j in range(nocc):
                energy += 0.5 * (h2[i, i, j, j] - h2[i, j, j, i])
    for i in range(nalpha):
        for j in range(nbeta):
            energy += h2[i, i, j, j]
    return energy


# ------------------------------------------------- exact many-body spectrum


def _strings(norb: int, nelec: int) -> list:
    """Occupation bitmasks for every way of putting ``nelec`` in ``norb`` orbitals."""
    return [
        sum(1 << orbital for orbital in occupied)
        for occupied in combinations(range(norb), nelec)
    ]


def _annihilate(mask: int, spin_orbital: int):
    """``a_p |mask>`` as ``(sign, mask)``, or ``None`` when the orbital is empty."""
    bit = 1 << spin_orbital
    if not mask & bit:
        return None
    sign = -1 if bin(mask & (bit - 1)).count("1") % 2 else 1
    return sign, mask ^ bit


def _create(mask: int, spin_orbital: int):
    """``a^dagger_p |mask>`` as ``(sign, mask)``, or ``None`` when already occupied."""
    bit = 1 << spin_orbital
    if mask & bit:
        return None
    sign = -1 if bin(mask & (bit - 1)).count("1") % 2 else 1
    return sign, mask | bit


def exact_spectrum(h1, h2, ecore: float, nalpha: int, nbeta: int) -> np.ndarray:
    """Every eigenvalue of the many-electron Hamiltonian, in the full FCI space.

    Spin orbital ``p`` is alpha and ``norb + p`` is beta, so a determinant is one
    bitmask over ``2 * norb`` bits. The Hamiltonian is assembled by applying

        sum_pq h_pq a+_p a_q  +  1/2 sum_pqrs (pq|rs) a+_p a+_r a_s a_q

    one operator string at a time. Deliberately the most literal construction
    available: it is slow, it is obviously right, and the active spaces it is
    asked to handle are the small ones.
    """
    h1 = np.asarray(h1, dtype=np.float64)
    h2 = np.asarray(h2, dtype=np.float64)
    norb = h1.shape[0]

    determinants = [
        alpha | (beta << norb)
        for alpha in _strings(norb, nalpha)
        for beta in _strings(norb, nbeta)
    ]
    index = {mask: position for position, mask in enumerate(determinants)}
    dimension = len(determinants)
    hamiltonian = np.zeros((dimension, dimension))

    offsets = (0, norb)  # alpha spin orbitals, then beta

    for column, ket in enumerate(determinants):
        for p in range(norb):
            for q in range(norb):
                value = h1[p, q]
                if value == 0.0:
                    continue
                for offset in offsets:
                    step = _annihilate(ket, q + offset)
                    if step is None:
                        continue
                    sign, mask = step
                    step = _create(mask, p + offset)
                    if step is None:
                        continue
                    hamiltonian[index[step[1]], column] += sign * step[0] * value

        for p in range(norb):
            for q in range(norb):
                for r in range(norb):
                    for s in range(norb):
                        value = 0.5 * h2[p, q, r, s]
                        if value == 0.0:
                            continue
                        for sigma in offsets:
                            for tau in offsets:
                                step = _annihilate(ket, q + sigma)
                                if step is None:
                                    continue
                                sign, mask = step
                                step = _annihilate(mask, s + tau)
                                if step is None:
                                    continue
                                sign *= step[0]
                                step = _create(step[1], r + tau)
                                if step is None:
                                    continue
                                sign *= step[0]
                                step = _create(step[1], p + sigma)
                                if step is None:
                                    continue
                                sign *= step[0]
                                hamiltonian[index[step[1]], column] += sign * value

    asymmetry = float(np.max(np.abs(hamiltonian - hamiltonian.T)))
    if asymmetry > 1e-10:
        # Symmetrising here would hide a sign error in the operator strings,
        # which is the one thing this oracle exists to not do.
        raise AssertionError(
            f"the assembled many-electron Hamiltonian is not symmetric "
            f"({asymmetry:.3e}); the oracle itself is wrong"
        )
    return np.linalg.eigvalsh(hamiltonian) + float(ecore)


__all__ = [
    "determinant_energy",
    "exact_spectrum",
    "read_fcidump",
    "spin_occupations",
]


# ------------------------------------------------------- the PySCF variants


def fci_energy_from_fcidump(path, nroots: int = 1):
    """Ground-state FCI energy of a written FCIDUMP, through PySCF's own reader.

    Adapted from the parked ``claude/benchmarks-and-sweeps`` branch, where it was
    the validation solver for a sweep step. Kept here because it is a second,
    entirely foreign path from a file on disk to an energy: PySCF parses it and
    PySCF diagonalises it, so agreement with :func:`exact_spectrum` is agreement
    between two codes rather than between two spellings of one.

    The import is inside the function so the numpy-only CI job can import this
    module.
    """
    from pyscf import ao2mo, fci  # noqa: PLC0415 -- optional dependency
    from pyscf.tools import fcidump  # noqa: PLC0415

    data = fcidump.read(str(path), verbose=False)
    norb = int(data["NORB"])
    nelec = int(data["NELEC"])
    nalpha, nbeta = spin_occupations(nelec, int(data.get("MS2", 0)))
    eri = ao2mo.restore(1, np.asarray(data["H2"]), norb)

    energy, _ = fci.direct_spin1.FCI().kernel(
        np.asarray(data["H1"]), eri, norb, (nalpha, nbeta), nroots=nroots
    )
    energy = np.atleast_1d(energy) + float(data["ECORE"])
    return float(energy[0]) if nroots == 1 else [float(value) for value in energy]


__all__.append("fci_energy_from_fcidump")
