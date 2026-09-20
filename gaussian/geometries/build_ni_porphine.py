#!/usr/bin/env python3
"""Construct an idealised D4h Ni(II) porphine geometry and check it.

NiC20H12N4, 37 atoms, planar, metal at the origin, ring in the xy-plane.

This is a *model* structure built from standard metalloporphine bond lengths and
the pyrrole Ca-N-Ca angle, not an optimised one and not an experimental one.
It exists so that `ni_porphine_s1` / `ni_porphine_s3` are runnable from this
repository alone; production numbers should come from an optimised geometry, and
the manifests say so.

Construction. D4h leaves six free parameters for the heavy-atom skeleton: the
metal-nitrogen distance, the pyrrole carbons' positions, and the meso carbon's
radius. Five bond lengths plus the Ca-N-Ca angle fix all six in closed form --
no optimiser, so the result is exactly reproducible:

    Ni-N      1.950 A     Ca-Cb     1.437 A     Ca-Cm     1.383 A
    N-Ca      1.379 A     Cb-Cb     1.350 A     C-H       1.080 A
    Ca-N-Ca   105.5 deg

One pyrrole (N, 2 Ca, 2 Cb, 2 Hb) and one meso bridge (Cm, Hm) are built on and
about the +x axis; the other three of each follow from the C4 axis. Hydrogens sit
on the external bisector at their carbon.

Usage:  python3 build_ni_porphine.py [--out ni_porphine.xyz]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

R_NI_N = 1.950
R_N_CA = 1.379
R_CA_CB = 1.437
R_CB_CB = 1.350
R_CA_CM = 1.383
R_C_H = 1.080
ANGLE_CA_N_CA = 105.5  # degrees

TOLERANCE = 1e-9


def skeleton():
    """Symmetry-unique heavy atoms, in the xy-plane, as (symbol, x, y)."""
    half_angle = math.radians(ANGLE_CA_N_CA / 2.0)

    # N sits on the +x axis at the metal-nitrogen distance.
    x_n = R_NI_N

    # Ca is one N-Ca bond from N, at half the Ca-N-Ca angle off the axis, and
    # further from the metal than N (the pyrrole points outward).
    x_ca = x_n + R_N_CA * math.cos(half_angle)
    y_ca = R_N_CA * math.sin(half_angle)

    # The two Cb share an x and straddle the axis at half the Cb-Cb bond.
    y_cb = R_CB_CB / 2.0
    dy = y_ca - y_cb
    dx_squared = R_CA_CB**2 - dy**2
    if dx_squared <= 0:
        raise ValueError("Ca-Cb bond is too short to span the Ca/Cb y-offset")
    x_cb = x_ca + math.sqrt(dx_squared)  # outward again, away from the metal

    # Cm lies on the 45 deg diagonal, one Ca-Cm bond from the Ca of each
    # neighbouring pyrrole. With Cm = (t, t): 2t^2 - 2t(x_ca + y_ca) + |Ca|^2 = R^2.
    b = x_ca + y_ca
    c = x_ca**2 + y_ca**2 - R_CA_CM**2
    discriminant = b**2 - 2.0 * c
    if discriminant < 0:
        raise ValueError("no meso carbon satisfies the Ca-Cm bond length on the diagonal")
    root = math.sqrt(discriminant)
    # Two roots: the inner one would put the meso carbon inside the N4 cavity.
    t = (b + root) / 2.0

    return {
        "N": (x_n, 0.0),
        "Ca+": (x_ca, y_ca),
        "Ca-": (x_ca, -y_ca),
        "Cb+": (x_cb, y_cb),
        "Cb-": (x_cb, -y_cb),
        "Cm": (t, t),
    }


def _hydrogen(site, neighbours, length=R_C_H):
    """Place H on the external bisector at *site*, away from its neighbours."""
    sx, sy = site
    ux = uy = 0.0
    for nx, ny in neighbours:
        dx, dy = nx - sx, ny - sy
        norm = math.hypot(dx, dy)
        ux += dx / norm
        uy += dy / norm
    norm = math.hypot(ux, uy)
    if norm < TOLERANCE:
        raise ValueError("neighbours are collinear; the bisector is undefined")
    return (sx - length * ux / norm, sy - length * uy / norm)


def _rotate90(point, times):
    """Apply the C4 axis: (x, y) -> (-y, x), *times* times."""
    x, y = point
    for _ in range(times % 4):
        x, y = -y, x
    return (x, y)


def build():
    """Return the full 37-atom molecule as [(symbol, x, y, z), ...]."""
    s = skeleton()
    h_b_plus = _hydrogen(s["Cb+"], [s["Ca+"], s["Cb-"]])
    h_b_minus = _hydrogen(s["Cb-"], [s["Ca-"], s["Cb+"]])
    # The meso carbon bridges the Ca of the +x pyrrole and the Ca of the +y
    # pyrrole; the latter is the former's partner rotated a quarter turn.
    ca_next = _rotate90(s["Ca-"], 1)
    h_m = _hydrogen(s["Cm"], [s["Ca+"], ca_next])

    unique = [
        ("N", s["N"]),
        ("C", s["Ca+"]), ("C", s["Ca-"]),
        ("C", s["Cb+"]), ("C", s["Cb-"]),
        ("H", h_b_plus), ("H", h_b_minus),
        ("C", s["Cm"]),
        ("H", h_m),
    ]

    atoms = [("Ni", 0.0, 0.0, 0.0)]
    for quarter in range(4):
        for symbol, point in unique:
            x, y = _rotate90(point, quarter)
            atoms.append((symbol, x, y, 0.0))
    return atoms


# ------------------------------------------------------------------- checking

def _distance(a, b):
    return math.dist(a[1:4], b[1:4])


def report(atoms):
    """Print the checks that make this geometry believable, and return them."""
    formula = {}
    for atom in atoms:
        formula[atom[0]] = formula.get(atom[0], 0) + 1

    pairs = [(i, j) for i in range(len(atoms)) for j in range(i + 1, len(atoms))]
    distances = sorted((_distance(atoms[i], atoms[j]), i, j) for i, j in pairs)

    # Bonds, by the usual "shorter than 1.75 A, or 2.1 A to the metal" rule.
    bonds = []
    for d, i, j in distances:
        metal = "Ni" in (atoms[i][0], atoms[j][0])
        if d < (2.10 if metal else 1.75):
            bonds.append((d, atoms[i][0], atoms[j][0]))

    kinds = {}
    for d, si, sj in bonds:
        key = "-".join(sorted((si, sj)))
        kinds.setdefault(key, []).append(d)

    lines = [
        f"atoms          {len(atoms)}",
        "formula        "
        + "".join(f"{s}{formula[s]}" for s in ("Ni", "C", "H", "N") if s in formula),
        f"planar         max |z| = {max(abs(a[3]) for a in atoms):.2e} A",
        f"closest pair   {distances[0][0]:.4f} A "
        f"({atoms[distances[0][1]][0]}-{atoms[distances[0][2]][0]})",
        f"bonds found    {len(bonds)}",
    ]
    for key in sorted(kinds):
        values = kinds[key]
        lines.append(
            f"  {key:<6s} x{len(values):<3d} {min(values):.4f} - {max(values):.4f} A"
        )
    return "\n".join(lines)


def to_xyz(atoms, comment):
    body = "\n".join(f"{s:<3s} {x:>14.8f} {y:>14.8f} {z:>14.8f}" for s, x, y, z in atoms)
    return f"{len(atoms)}\n{comment}\n{body}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="ni_porphine.xyz", help="output .xyz path")
    parser.add_argument("--check-only", action="store_true", help="print the checks, write nothing")
    args = parser.parse_args()

    atoms = build()
    print(report(atoms))
    if args.check_only:
        return
    comment = (
        "Ni(II) porphine, idealised D4h model geometry: "
        f"Ni-N {R_NI_N}, N-Ca {R_N_CA}, Ca-Cb {R_CA_CB}, Cb-Cb {R_CB_CB}, "
        f"Ca-Cm {R_CA_CM}, C-H {R_C_H} A, Ca-N-Ca {ANGLE_CA_N_CA} deg. "
        "Built by benchmarks/geometries/build_ni_porphine.py -- not optimised."
    )
    path = Path(args.out)
    path.write_text(to_xyz(atoms, comment), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
