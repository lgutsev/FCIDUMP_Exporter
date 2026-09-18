"""Exact diagonalization of a small active space, in NumPy alone.

Exists so that FCI invariance under active-space rotation is covered in a
NumPy-only CI, where PySCF is not installed. Where PySCF *is* installed, a test
checks this implementation against ``pyscf.fci`` directly, so the helper itself
is validated rather than trusted.

The Hamiltonian is built by applying second-quantized operators to determinants
represented as occupation bitmasks:

    H = sum_pq h[p,q] a+_p a_q
      + 1/2 sum_pqrs <pq|rs> a+_p a+_q a_s a_r

in spin-orbital basis, with ``<pq|rs>`` the physicist-ordered integral. Applying
operators one at a time and tracking the fermionic sign is mechanical; the
alternative -- enumerating Slater-Condon cases by excitation rank -- is where
sign errors live.

Only suitable for a handful of orbitals: the determinant basis is built in full.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _annihilate(det: int, p: int):
    """Apply ``a_p``. Returns ``(sign, det)`` or ``None`` if ``p`` is empty."""
    bit = 1 << p
    if not det & bit:
        return None
    # Sign is (-1)^(number of occupied spin-orbitals below p).
    sign = -1 if bin(det & (bit - 1)).count("1") % 2 else 1
    return sign, det ^ bit


def _create(det: int, p: int):
    """Apply ``a+_p``. Returns ``(sign, det)`` or ``None`` if ``p`` is filled."""
    bit = 1 << p
    if det & bit:
        return None
    sign = -1 if bin(det & (bit - 1)).count("1") % 2 else 1
    return sign, det | bit


def spin_orbital_integrals(h: np.ndarray, eri: np.ndarray):
    """Expand spatial integrals to spin orbitals.

    Spin orbital ``2t`` is spatial ``t`` with alpha spin, ``2t+1`` with beta.
    ``eri`` is chemist's ``(pq|rs)``; the returned two-electron array is
    physicist's ``<pq|rs>``, which is what the operator form above uses.
    """
    norb = h.shape[0]
    nso = 2 * norb

    h_so = np.zeros((nso, nso))
    for p in range(nso):
        for q in range(nso):
            if p % 2 == q % 2:
                h_so[p, q] = h[p // 2, q // 2]

    g_so = np.zeros((nso,) * 4)
    for p in range(nso):
        for q in range(nso):
            for r in range(nso):
                for s in range(nso):
                    # <pq|rs> = (pr|qs), nonzero only when spins pair up.
                    if p % 2 == r % 2 and q % 2 == s % 2:
                        g_so[p, q, r, s] = eri[p // 2, r // 2, q // 2, s // 2]
    return h_so, g_so


def determinant_basis(norb: int, nocc_a: int, nocc_b: int):
    """Every determinant with the given alpha and beta occupations, as bitmasks."""
    dets = []
    for alpha in combinations(range(norb), nocc_a):
        for beta in combinations(range(norb), nocc_b):
            mask = 0
            for t in alpha:
                mask |= 1 << (2 * t)
            for t in beta:
                mask |= 1 << (2 * t + 1)
            dets.append(mask)
    return sorted(dets)


def build_hamiltonian(h: np.ndarray, eri: np.ndarray, nocc_a: int, nocc_b: int):
    """Dense many-body Hamiltonian in the determinant basis."""
    norb = h.shape[0]
    nso = 2 * norb
    h_so, g_so = spin_orbital_integrals(h, eri)

    dets = determinant_basis(norb, nocc_a, nocc_b)
    index = {d: n for n, d in enumerate(dets)}
    size = len(dets)
    ham = np.zeros((size, size))

    for n, det in enumerate(dets):
        # One-body: a+_p a_q
        for q in range(nso):
            step = _annihilate(det, q)
            if step is None:
                continue
            sign_q, det_q = step
            for p in range(nso):
                if h_so[p, q] == 0.0:
                    continue
                step2 = _create(det_q, p)
                if step2 is None:
                    continue
                sign_p, det_p = step2
                m = index.get(det_p)
                if m is not None:
                    ham[m, n] += sign_q * sign_p * h_so[p, q]

        # Two-body: 1/2 <pq|rs> a+_p a+_q a_s a_r
        for r in range(nso):
            step_r = _annihilate(det, r)
            if step_r is None:
                continue
            sign_r, det_r = step_r
            for s in range(nso):
                step_s = _annihilate(det_r, s)
                if step_s is None:
                    continue
                sign_s, det_s = step_s
                for q in range(nso):
                    step_q = _create(det_s, q)
                    if step_q is None:
                        continue
                    sign_q, det_q = step_q
                    for p in range(nso):
                        value = g_so[p, q, r, s]
                        if value == 0.0:
                            continue
                        step_p = _create(det_q, p)
                        if step_p is None:
                            continue
                        sign_p, det_p = step_p
                        m = index.get(det_p)
                        if m is not None:
                            ham[m, n] += (
                                0.5
                                * sign_r
                                * sign_s
                                * sign_q
                                * sign_p
                                * value
                            )

    return ham


def ground_state_energy(
    h: np.ndarray, eri: np.ndarray, nocc_a: int, nocc_b: int
) -> float:
    """Lowest eigenvalue of the active-space Hamiltonian (core energy excluded)."""
    ham = build_hamiltonian(h, eri, nocc_a, nocc_b)
    return float(np.linalg.eigvalsh(ham)[0])
