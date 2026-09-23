#!/usr/bin/env python3
"""Write a Dice ``input.dat`` for an FCIDUMP that g16dump produced.

Deliberately thin, and deliberately not part of the ``g16dump`` package: it
reads the FCIDUMP header and nothing else, so it has no import of g16dump, no
dependency beyond the standard library, and no knowledge of bundles. Solver
input files are the user's to own; this only saves transcribing the reference
determinant's spin-orbital list, which is the one part that is easy to get
quietly wrong.

    python3 make_dice_input.py FCIDUMP --out input.dat

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
    """Return the FCIDUMP namelist header as ``{KEY: value}``, keys upper-cased."""
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
    # comma-separated list, so comma splitting silently truncates it.
    parts = re.split(r"([A-Za-z_][A-Za-z_0-9]*)\s*=", joined)
    header: dict = {}
    for key, value in zip(parts[1::2], parts[2::2]):
        header[key.upper()] = value.strip().rstrip(",").strip()
    return header


def required_int(header: dict, key: str, path: Path) -> int:
    if key not in header:
        raise FcidumpError(
            f"{path}: the header has no {key}. NORB, NELEC and MS2 are all "
            f"required to write the reference determinant."
        )
    try:
        return int(header[key])
    except ValueError as exc:
        raise FcidumpError(f"{path}: {key} is {header[key]!r}, not an integer") from exc


def aufbau_occupation(nalpha: int, nbeta: int) -> list:
    """The reference determinant as Dice spin-orbital indices.

    Dice numbers spin orbitals so that spatial orbital ``i`` (0-based) is alpha
    ``2i`` and beta ``2i + 1``. This fills the lowest-numbered active orbitals,
    which is the reference determinant only if the active orbitals arrive in
    the order that makes it so. For a window taken straight out of Gaussian
    they do; after an active-space rotation they need not, and the list has to
    be written by hand.
    """
    return [2 * i for i in range(nalpha)] + [2 * i + 1 for i in range(nbeta)]


TEMPLATE = """\
nocc {nelec}
{occupation}
end

nroots 1

#var keywords
schedule
0 1e-3
3 1e-4
6 1e-5
end
davidsonTol 5e-5
dE 1e-8
maxiter 10

#PT keywords
epsilon2 1e-8
sampleN 200
targetError 1e-4

# Generated from {source} by make_dice_input.py:
#   {norb} active orbitals, {nelec} electrons, {nalpha} alpha and {nbeta} beta.
# Dice reads the integrals from a file named FCIDUMP in the directory it runs
# in. The thresholds above are a starting point, not a converged calculation.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a Dice input.dat for a g16dump FCIDUMP.",
        epilog="The reference determinant is filled aufbau in the active "
        "orbital order; check it against the occupation your job actually has.",
    )
    parser.add_argument("fcidump", type=Path, help="the FCIDUMP to read")
    parser.add_argument(
        "--out", type=Path, default=Path("input.dat"), help="path for the input file"
    )
    args = parser.parse_args(argv)

    try:
        header = read_header(args.fcidump)
        norb = required_int(header, "NORB", args.fcidump)
        nelec = required_int(header, "NELEC", args.fcidump)
        ms2 = required_int(header, "MS2", args.fcidump)
    except (FcidumpError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if ms2 > nelec or (nelec - ms2) % 2 != 0:
        print(
            f"error: {args.fcidump}: MS2 {ms2} is impossible for {nelec} "
            f"electrons; the parity is wrong or there are not enough electrons.",
            file=sys.stderr,
        )
        return 1
    nalpha, nbeta = (nelec + ms2) // 2, (nelec - ms2) // 2
    if nalpha > norb:
        print(
            f"error: {args.fcidump}: {nalpha} alpha electrons do not fit in "
            f"{norb} active orbitals.",
            file=sys.stderr,
        )
        return 1

    occupation = aufbau_occupation(nalpha, nbeta)
    args.out.write_text(
        TEMPLATE.format(
            source=args.fcidump,
            norb=norb,
            nelec=nelec,
            nalpha=nalpha,
            nbeta=nbeta,
            occupation=" ".join(str(index) for index in occupation),
        )
    )
    print(
        f"wrote {args.out}: {norb} orbitals, {nelec} electrons "
        f"({nalpha} alpha, {nbeta} beta), 2S = {ms2}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
