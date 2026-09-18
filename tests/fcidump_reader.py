"""A deliberately independent FCIDUMP parser, for round-trip testing.

Written from the FCIDUMP format description rather than from ``g16dump.write``,
and sharing no code with it, so that "write then read gives back what we started
with" is evidence rather than a tautology. Where PySCF is installed the tests
also round-trip through ``pyscf.tools.fcidump.read``; two independent readers
agreeing is better still.

Kept in ``tests/`` rather than in the package: g16dump writes FCIDUMPs, it does
not need to read them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class ParsedFcidump:
    norb: int
    nelec: int
    ms2: int
    isym: int
    orbsym: list
    h1: np.ndarray  # (norb, norb), symmetrized
    eri: np.ndarray  # (norb,)*4, all 8 permutations filled
    e_core: float
    n_two_electron: int
    n_one_electron: int


def _parse_header(text: str) -> dict:
    """Pull NORB/NELEC/MS2/ISYM/ORBSYM out of the &FCI namelist."""
    header = text.split("&END")[0]

    def _scalar(name, default=None):
        match = re.search(rf"{name}\s*=\s*(-?\d+)", header, re.IGNORECASE)
        if match is None:
            if default is None:
                raise ValueError(f"{name} missing from FCIDUMP header")
            return default
        return int(match.group(1))

    orbsym_match = re.search(r"ORBSYM\s*=\s*([0-9,\s]+)", header, re.IGNORECASE)
    orbsym = []
    if orbsym_match:
        orbsym = [int(x) for x in orbsym_match.group(1).replace(",", " ").split()]

    return {
        "norb": _scalar("NORB"),
        "nelec": _scalar("NELEC"),
        "ms2": _scalar("MS2", 0),
        "isym": _scalar("ISYM", 1),
        "orbsym": orbsym,
    }


def read_fcidump(path: str) -> ParsedFcidump:
    """Parse an FCIDUMP into dense tensors, expanding all permutations."""
    with open(path) as handle:
        text = handle.read()

    head = _parse_header(text)
    norb = head["norb"]

    h1 = np.zeros((norb, norb))
    eri = np.zeros((norb,) * 4)
    e_core = 0.0
    n_two = n_one = 0

    body = text.split("&END", 1)[1]
    for line in body.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        value = float(parts[0])
        i, j, k, l = (int(x) for x in parts[1:])

        if i == 0 and j == 0 and k == 0 and l == 0:
            e_core = value
            continue

        if k == 0 and l == 0:
            a, b = i - 1, j - 1
            h1[a, b] = h1[b, a] = value
            n_one += 1
            continue

        a, b, c, d = i - 1, j - 1, k - 1, l - 1
        for p, q, r, s in (
            (a, b, c, d),
            (b, a, c, d),
            (a, b, d, c),
            (b, a, d, c),
            (c, d, a, b),
            (d, c, a, b),
            (c, d, b, a),
            (d, c, b, a),
        ):
            eri[p, q, r, s] = value
        n_two += 1

    return ParsedFcidump(
        norb=norb,
        nelec=head["nelec"],
        ms2=head["ms2"],
        isym=head["isym"],
        orbsym=head["orbsym"],
        h1=h1,
        eri=eri,
        e_core=e_core,
        n_two_electron=n_two,
        n_one_electron=n_one,
    )


def determinant_energy(
    h1: np.ndarray, eri: np.ndarray, e_core: float, nocc_a: int, nocc_b: int
) -> float:
    """Energy of the aufbau reference determinant from FCIDUMP tensors.

    Slater's rules over the first ``nocc_a``/``nocc_b`` orbitals, plus the
    scalar. Independent of ``g16dump.hamiltonian``.
    """
    e = e_core
    for nocc in (nocc_a, nocc_b):
        e += float(np.trace(h1[:nocc, :nocc]))
    for nocc in (nocc_a, nocc_b):
        if nocc:
            block = eri[:nocc, :nocc, :nocc, :nocc]
            e += 0.5 * float(np.einsum("iijj->", block) - np.einsum("ijji->", block))
    if nocc_a and nocc_b:
        e += float(np.einsum("iijj->", eri[:nocc_a, :nocc_a, :nocc_b, :nocc_b]))
    return e
