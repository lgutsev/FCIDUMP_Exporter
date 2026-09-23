"""Gaussian's AO conventions and ``.fch`` layout, written down independently of MOKIT.

The MOKIT tests need Gaussian-side data for systems nobody here can run Gaussian
on: AO matrices in Gaussian's AO order, and a formatted checkpoint that MOKIT
will read exactly as it reads a real one. This module produces both from a
PySCF calculation, from Gaussian's documented conventions alone. It imports
nothing from MOKIT and nothing from ``g16dump``, so when the package's MOKIT
route agrees with it, two independent derivations of the same transformation
agree -- which is the point.

The conventions, per shell, in the order Gaussian lists the functions:

* ``s``; ``p`` as x, y, z; an ``SP`` shell (Pople's shared-exponent s and p)
  as s, x, y, z.
* Pure shells, ``l >= 2``: m = 0, +1, -1, +2, -2, ..., +l, -l. PySCF orders
  them m = -l..+l.
* Cartesian shells: d as xx, yy, zz, xy, xz, yz; f as xxx, yyy, zzz, xyy, xxy,
  xxz, xzz, yzz, yyz, xyz; g and above in the reverse of PySCF's order. Every
  Cartesian function is normalized to one in Gaussian, whereas PySCF's are not,
  so these differ by a per-function scale as well as by order.
* Within an atom, shells are listed in basis-set order with ``SP`` counting as
  ``l = 0``. PySCF splits an ``SP`` shell into an s and a p shell and sorts an
  atom's shells by ``l``, so a Pople basis is reordered even when it has no
  polarization functions at all (6-31G: Gaussian s, sp, sp; PySCF s, s, s, p, p).

These were checked against real Gaussian ``.fch`` files (O2/cc-pVTZ pure and
Cartesian, H2O/cc-pVDZ, H2O/3-21G, He/spdfgh Cartesian): the orbitals only come
out orthonormal, and the Gaussian SCF energy is only reproduced, once this
mapping is applied.
"""

from __future__ import annotations

import numpy as np

_CART_ORDER = {
    2: ["xx", "yy", "zz", "xy", "xz", "yz"],
    3: ["xxx", "yyy", "zzz", "xyy", "xxy", "xxz", "xzz", "yzz", "yyz", "xyz"],
}


def _pyscf_cart_labels(ang: int) -> list:
    """PySCF's Cartesian component order: x power descending, then y."""
    labels = []
    for lx in range(ang, -1, -1):
        for ly in range(ang - lx, -1, -1):
            lz = ang - lx - ly
            labels.append("x" * lx + "y" * ly + "z" * lz)
    return labels


def _gaussian_cart_labels(ang: int) -> list:
    if ang in _CART_ORDER:
        return _CART_ORDER[ang]
    return _pyscf_cart_labels(ang)[::-1]


def _gaussian_pure_m(ang: int) -> list:
    return [0] + [sign * k for k in range(1, ang + 1) for sign in (1, -1)]


class GaussianShell:
    """One shell as Gaussian lists it, with where its functions live in PySCF."""

    def __init__(self, atom, ang, exponents, coefficients, sp_coefficients=None):
        self.atom = atom
        self.l = ang  # -1 for SP
        self.exponents = np.asarray(exponents, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.sp_coefficients = (
            None if sp_coefficients is None else np.asarray(sp_coefficients, dtype=float)
        )
        # (pyscf AO index, component) for each function, in Gaussian order.
        self.functions: list = []


def gaussian_shells(mol) -> list:
    """The shells of ``mol`` in Gaussian's order, each knowing its PySCF AOs."""
    cart = bool(mol.cart)
    per_atom: dict = {}
    offset = 0
    for ib in range(mol.nbas):
        ang = int(mol.bas_angular(ib))
        atom = int(mol.bas_atom(ib))
        exps = np.asarray(mol.bas_exp(ib))
        coeffs = np.asarray(mol.bas_ctr_coeff(ib))
        size = (ang + 1) * (ang + 2) // 2 if cart else 2 * ang + 1
        for column in range(int(mol.bas_nctr(ib))):
            keep = np.abs(coeffs[:, column]) > 0
            per_atom.setdefault(atom, []).append(
                dict(l=ang, exps=exps[keep], coefs=coeffs[keep, column], offset=offset)
            )
            offset += size
    assert offset == mol.nao

    shells = []
    for atom in sorted(per_atom):
        entries = per_atom[atom]
        used = set()
        merged = []
        for i, entry in enumerate(entries):
            if i in used:
                continue
            if entry["l"] == 0:
                partner = next(
                    (
                        j for j, other in enumerate(entries)
                        if j not in used and j != i and other["l"] == 1
                        and other["exps"].shape == entry["exps"].shape
                        and np.array_equal(other["exps"], entry["exps"])
                    ),
                    None,
                )
                if partner is not None:
                    used.update((i, partner))
                    merged.append(("sp", entry, entries[partner]))
                    continue
            used.add(i)
            merged.append(("plain", entry, None))
        merged.sort(key=lambda item: 0 if item[0] == "sp" else item[1]["l"])

        for kind, entry, p_entry in merged:
            if kind == "sp":
                shell = GaussianShell(
                    atom, -1, entry["exps"], entry["coefs"], p_entry["coefs"]
                )
                shell.functions.append((entry["offset"], "s"))
                for k, axis in enumerate("xyz"):
                    shell.functions.append((p_entry["offset"] + k, axis))
            else:
                ang = entry["l"]
                shell = GaussianShell(atom, ang, entry["exps"], entry["coefs"])
                base = entry["offset"]
                if ang <= 1:
                    for k in range(2 * ang + 1):
                        shell.functions.append((base + k, k))
                elif cart:
                    pyscf_labels = _pyscf_cart_labels(ang)
                    for label in _gaussian_cart_labels(ang):
                        shell.functions.append((base + pyscf_labels.index(label), label))
                else:
                    for m in _gaussian_pure_m(ang):
                        shell.functions.append((base + m + ang, m))
            shells.append(shell)
    return shells


def gaussian_to_pyscf(mol) -> np.ndarray:
    """``X`` with ``C_pyscf = X @ C_gaussian``, from the conventions above.

    AO matrices go the other way: ``A_gaussian = X.T @ A_pyscf @ X``.
    """
    overlap_diag = np.diag(mol.intor("int1e_ovlp"))
    order = [index for shell in gaussian_shells(mol) for index, _ in shell.functions]
    assert sorted(order) == list(range(mol.nao))
    transform = np.zeros((mol.nao, mol.nao))
    for g, p in enumerate(order):
        # Gaussian normalizes every function; PySCF's Cartesian ones are not.
        transform[p, g] = 1.0 / np.sqrt(overlap_diag[p])
    return transform


# ------------------------------------------------------------------ writer


def _int_scalar(name, value):
    return f"{name:<43}I{value:>17d}"


def _real_scalar(name, value):
    return f"{name:<43}R{value:>27.15E}"


def _int_array(name, values):
    values = [int(v) for v in values]
    lines = [f"{name:<43}I   N={len(values):>12d}"]
    for k in range(0, len(values), 6):
        lines.append("".join(f"{v:>12d}" for v in values[k:k + 6]))
    return lines


def _real_array(name, values):
    values = np.asarray(values, dtype=float).ravel()
    lines = [f"{name:<43}R   N={values.size:>12d}"]
    for k in range(0, values.size, 5):
        lines.append("".join(f"{v:16.8E}" for v in values[k:k + 5]))
    return lines


def write_fch(path, mol, mo_coeff_gaussian, mo_energy, *, method, total_energy,
              nalpha, nbeta, title="written by tests/gaussian_fch.py"):
    """A formatted checkpoint in the layout ``formchk`` produces.

    ``mo_coeff_gaussian`` is column-wise, in Gaussian's AO order. Section order
    matters: MOKIT reads the file sequentially.
    """
    shells = gaussian_shells(mol)
    coords = np.asarray(mol.atom_coords())  # bohr
    charges = mol.atom_charges()
    nbf = mol.nao
    nmo = mo_coeff_gaussian.shape[1]

    shell_types = []
    for shell in shells:
        if shell.l < 2:
            shell_types.append(shell.l)
        else:
            shell_types.append(shell.l if mol.cart else -shell.l)
    cart_flag = 1 if mol.cart else 0

    lines = [title[:72], f"{'SP':<10}{method:<60}{'Gen':<20}"]
    lines += [
        _int_scalar("Number of atoms", mol.natm),
        _int_scalar("Charge", int(mol.charge)),
        _int_scalar("Multiplicity", int(mol.spin) + 1),
        _int_scalar("Number of electrons", nalpha + nbeta),
        _int_scalar("Number of alpha electrons", nalpha),
        _int_scalar("Number of beta electrons", nbeta),
        _int_scalar("Number of basis functions", nbf),
        _int_scalar("Number of independent functions", nmo),
    ]
    lines += _int_array("Atomic numbers", charges)
    lines += _real_array("Nuclear charges", charges.astype(float))
    lines += _real_array("Current cartesian coordinates", coords)
    lines += _int_array("Integer atomic weights", [2 * z for z in charges])
    lines += _real_array("Real atomic weights", [2.0 * z for z in charges])
    lines += [
        _int_scalar("Number of contracted shells", len(shells)),
        _int_scalar("Number of primitive shells", sum(s.exponents.size for s in shells)),
        _int_scalar("Pure/Cartesian d shells", cart_flag),
        _int_scalar("Pure/Cartesian f shells", cart_flag),
        _int_scalar("Highest angular momentum", max(abs(t) for t in shell_types)),
        _int_scalar("Largest degree of contraction", max(s.exponents.size for s in shells)),
    ]
    lines += _int_array("Shell types", shell_types)
    lines += _int_array("Number of primitives per shell", [s.exponents.size for s in shells])
    lines += _int_array("Shell to atom map", [s.atom + 1 for s in shells])
    lines += _real_array("Primitive exponents", np.concatenate([s.exponents for s in shells]))
    lines += _real_array(
        "Contraction coefficients", np.concatenate([s.coefficients for s in shells])
    )
    if any(s.sp_coefficients is not None for s in shells):
        lines += _real_array(
            "P(S=P) Contraction coefficients",
            np.concatenate([
                s.sp_coefficients if s.sp_coefficients is not None
                else np.zeros_like(s.coefficients)
                for s in shells
            ]),
        )
    lines += _real_array(
        "Coordinates of each shell", np.array([coords[s.atom] for s in shells])
    )
    lines += [
        _real_scalar("SCF Energy", total_energy),
        _real_scalar("Total Energy", total_energy),
    ]
    lines += _real_array("Alpha Orbital Energies", mo_energy)
    lines += _real_array("Alpha MO coefficients", np.asarray(mo_coeff_gaussian).T)
    occ_a = mo_coeff_gaussian[:, :nalpha]
    occ_b = mo_coeff_gaussian[:, :nbeta]
    density = occ_a @ occ_a.T + occ_b @ occ_b.T
    rows, cols = np.tril_indices(nbf)
    lines += _real_array("Total SCF Density", density[rows, cols])
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path
