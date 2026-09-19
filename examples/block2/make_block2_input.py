#!/usr/bin/env python3
"""Write a Block2 ``dmrg.conf`` for an FCIDUMP that g16dump produced.

Deliberately thin, and deliberately not part of the ``g16dump`` package: it
reads the FCIDUMP header and nothing else, so it has no import of g16dump, no
dependency beyond the standard library, and no knowledge of bundles. Solver
input files are the user's to own; this only saves transcribing three numbers.

    python3 make_block2_input.py FCIDUMP --out dmrg.conf

Everything it emits is explained in README.md next to this file.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEADER_END = re.compile(r"^\s*(&END|/END|/)\s*$", re.IGNORECASE)


class FcidumpError(ValueError):
    """The FCIDUMP header is missing, truncated or not what it claims."""


def read_header(path: Path) -> dict:
    """Return the FCIDUMP namelist header as ``{KEY: value}``, keys upper-cased.

    Only the header is read; the integrals below it are the solver's business.
    """
    text = []
    with path.open() as handle:
        for line in handle:
            if HEADER_END.match(line):
                break
            text.append(line)
        else:
            raise FcidumpError(
                f"{path}: the namelist header is not terminated. An FCIDUMP "
                f"header ends with '&END' or '/'; this file has neither, so it "
                f"is truncated or is not an FCIDUMP."
            )

    joined = " ".join(text).strip()
    if not joined.upper().lstrip().startswith("&FCI"):
        raise FcidumpError(
            f"{path}: does not start with the '&FCI' namelist marker, so it is "
            f"not an FCIDUMP."
        )
    joined = joined.lstrip()[len("&FCI"):]

    # Split on "KEY=" rather than on commas: ORBSYM's own value is a
    # comma-separated list, so comma splitting silently truncates it to its
    # first entry and every orbital after the first loses its irrep.
    parts = re.split(r"([A-Za-z_][A-Za-z_0-9]*)\s*=", joined)
    header: dict = {}
    for key, value in zip(parts[1::2], parts[2::2]):
        header[key.upper()] = value.strip().rstrip(",").strip()
    return header


def required_int(header: dict, key: str, path: Path) -> int:
    if key not in header:
        raise FcidumpError(
            f"{path}: the header has no {key}. NORB, NELEC and MS2 are all "
            f"required to set up a DMRG calculation."
        )
    try:
        return int(header[key])
    except ValueError as exc:
        raise FcidumpError(f"{path}: {key} is {header[key]!r}, not an integer") from exc


def point_group(header: dict, norb: int, path: Path) -> str:
    """``c1`` when ORBSYM is all ones, which is what g16dump writes.

    Anything else is refused rather than guessed: the FCIDUMP records irrep
    *numbers*, not which point group numbered them, and a wrong ``sym`` line
    makes Block2 target the wrong state while converging perfectly happily.
    """
    raw = header.get("ORBSYM", "")
    labels = [item for item in raw.replace(" ", "").split(",") if item]
    if labels and len(labels) != norb:
        raise FcidumpError(
            f"{path}: ORBSYM lists {len(labels)} orbitals but NORB is {norb}."
        )
    if all(label == "1" for label in labels):
        return "c1"
    raise FcidumpError(
        f"{path}: ORBSYM is not all ones, so this FCIDUMP carries point-group "
        f"symmetry. This script will not guess which group numbered those "
        f"irreps; set 'sym' and 'irrep' by hand. g16dump writes C1 dumps, so "
        f"this file did not come from g16dump."
    )


TEMPLATE = """\
! Generated from {source} by make_block2_input.py.
! Run with:   block2main {out} > dmrg.out
!
! Sanity check: the DMRG energy must come out below the reference determinant
! energy that 'g16dump validate --hamiltonian' prints for the same bundle. A
! DMRG energy above it means the sweeps are stuck, not that the dump is wrong.

sym {sym}
orbitals {orbitals}

nelec {nelec}
spin {spin}
irrep 1

hf_occ integral
schedule default
maxM {maxm}
maxiter {maxiter}
sweep_tol {sweep_tol}
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a Block2 dmrg.conf for a g16dump FCIDUMP.",
        epilog="The generated file is a starting point, not a converged "
        "calculation; maxM in particular is a scientific choice.",
    )
    parser.add_argument("fcidump", type=Path, help="the FCIDUMP to read")
    parser.add_argument(
        "--out", type=Path, default=Path("dmrg.conf"), help="path for the conf file"
    )
    parser.add_argument(
        "--orbitals",
        help="the 'orbitals' path written into the conf; defaults to the "
        "FCIDUMP's own name, which assumes Block2 runs beside it",
    )
    parser.add_argument("--maxm", type=int, default=500, help="maximum bond dimension")
    parser.add_argument("--maxiter", type=int, default=30, help="maximum sweeps")
    parser.add_argument(
        "--sweep-tol", default="1E-10", help="DMRG energy convergence tolerance"
    )
    args = parser.parse_args(argv)

    try:
        header = read_header(args.fcidump)
        norb = required_int(header, "NORB", args.fcidump)
        nelec = required_int(header, "NELEC", args.fcidump)
        ms2 = required_int(header, "MS2", args.fcidump)
        sym = point_group(header, norb, args.fcidump)
    except (FcidumpError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if nelec > 2 * norb:
        print(
            f"error: {args.fcidump}: NELEC {nelec} does not fit in NORB {norb} "
            f"spatial orbitals.",
            file=sys.stderr,
        )
        return 1
    if ms2 > nelec or (nelec - ms2) % 2 != 0:
        print(
            f"error: {args.fcidump}: MS2 {ms2} is impossible for {nelec} "
            f"electrons; the parity is wrong or there are not enough electrons.",
            file=sys.stderr,
        )
        return 1

    args.out.write_text(
        TEMPLATE.format(
            source=args.fcidump,
            out=args.out,
            sym=sym,
            orbitals=args.orbitals or args.fcidump.name,
            nelec=nelec,
            spin=ms2,
            maxm=args.maxm,
            maxiter=args.maxiter,
            sweep_tol=args.sweep_tol,
        )
    )
    print(f"wrote {args.out}: {norb} orbitals, {nelec} electrons, 2S = {ms2}, {sym}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
