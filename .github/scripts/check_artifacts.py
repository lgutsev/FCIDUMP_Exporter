#!/usr/bin/env python3
"""Inspect the built sdist and wheel before they can ever be published.

Two mistakes are cheap to make and expensive to discover after upload:

* shipping ``legacy/`` or Gaussian output (``.mat``, ``.fch``, ``.chk``, big
  ``.log`` files) inside the distribution, and
* shipping a wheel that contains something other than the ``g16dump`` package.

The wheel is allowed to be empty while the package is still being written; that
is reported, not failed.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

DIST = Path("dist")

FORBIDDEN_SUFFIXES = (".mat", ".fch", ".chk", ".log", ".zip")
FORBIDDEN_PREFIXES = ("legacy/",)


def relative_members(names: list[str], strip_top_level: bool) -> list[str]:
    """Drop the sdist's ``name-version/`` wrapper directory."""
    if not strip_top_level:
        return names
    out = []
    for name in names:
        head, _, tail = name.partition("/")
        out.append(tail or head)
    return out


def check(names: list[str], label: str) -> list[str]:
    problems = []
    for name in names:
        if not name or name.endswith("/"):
            continue
        if any(name.startswith(p) for p in FORBIDDEN_PREFIXES):
            problems.append(f"{label} contains reference material: {name}")
        if name.endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"{label} contains a Gaussian or archive artifact: {name}")
    return problems


def main() -> int:
    sdists = sorted(DIST.glob("*.tar.gz"))
    wheels = sorted(DIST.glob("*.whl"))
    if not sdists or not wheels:
        print(f"expected an sdist and a wheel in {DIST}/, found "
              f"{[p.name for p in DIST.glob('*')]}", file=sys.stderr)
        return 1

    problems = []

    for sdist in sdists:
        with tarfile.open(sdist) as tar:
            names = relative_members(tar.getnames(), strip_top_level=True)
        problems += check(names, sdist.name)
        print(f"{sdist.name}: {len(names)} members")

    for wheel in wheels:
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
        problems += check(names, wheel.name)
        package_files = [n for n in names if n.startswith("g16dump/")]
        stray = [
            n for n in names
            if not n.startswith("g16dump/") and ".dist-info/" not in n
        ]
        if stray:
            problems.append(f"{wheel.name} contains files outside g16dump/: {stray}")
        if not package_files:
            print(f"{wheel.name}: no g16dump module yet, which is expected "
                  f"before the package lands")
        else:
            print(f"{wheel.name}: {len(package_files)} package files")

    if problems:
        print("\nFAILED: the built distributions carry something they should not",
              file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1

    print("distributions carry only the package and its metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
