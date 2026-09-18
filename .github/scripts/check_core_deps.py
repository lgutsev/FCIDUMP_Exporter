#!/usr/bin/env python3
"""Fail if the core install stops being numpy-only.

Two things are checked, and both are project invariants rather than style:

1. The installed distribution's mandatory requirements are exactly ``numpy``.
   Anything heavier than that belongs in an optional extra, because the core
   path has to run on a cluster login node with nothing else available.

2. The modules the core path must never need are not importable in this
   environment, and the package still imports without them. ``matfile`` is the
   one module allowed to need gauopen, so it is exempt from the import probe.

Run it in an environment built from ``pip install .`` plus test tooling only.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import pkgutil
import sys

DIST = "g16dump"

# Requirements that may appear unconditionally. Everything else must carry an
# `extra == "..."` marker.
ALLOWED_CORE_REQUIREMENTS = {"numpy"}

# Must not be importable in the core environment. pyscf and mokit are
# validation-only; the rest are Gaussian's, and CI never has Gaussian.
FORBIDDEN_MODULES = ("pyscf", "mokit", "gauopen", "QCMatEl", "qcmatrixio")

# Heavy scientific dependencies that would silently arrive as a transitive
# requirement of something added to `dependencies`.
FORBIDDEN_TRANSITIVE = ("scipy", "h5py", "pandas")

# The only module permitted to import gauopen.
GAUOPEN_MODULE = "matfile"


def requirement_name(spec: str) -> str:
    """The bare distribution name from a PEP 508 requirement string."""
    head = spec.split(";", 1)[0].strip()
    for sep in ("[", "(", "=", "<", ">", "!", "~", " "):
        head = head.split(sep, 1)[0]
    return head.strip().lower().replace("_", "-")


def check_requirements() -> list[str]:
    try:
        specs = md.requires(DIST) or []
    except md.PackageNotFoundError:
        return [f"{DIST} is not installed; run `pip install .` before this check"]

    core = sorted({requirement_name(s) for s in specs if "extra ==" not in s})
    expected = sorted(ALLOWED_CORE_REQUIREMENTS)
    if core != expected:
        return [
            "core dependencies drifted",
            f"  expected: {expected}",
            f"  found:    {core}",
            "  Move anything new into an optional extra in pyproject.toml.",
        ]
    print(f"core dependencies are exactly {core}")
    return []


def check_forbidden() -> list[str]:
    problems = []
    for name in FORBIDDEN_MODULES + FORBIDDEN_TRANSITIVE:
        if importlib.util.find_spec(name) is not None:
            problems.append(
                f"{name} is importable here; the core job must prove the package "
                f"works without it"
            )
    if not problems:
        print("no validation-only or Gaussian module is present")
    return problems


def check_package_imports() -> list[str]:
    spec = importlib.util.find_spec("g16dump")
    if spec is None:
        print("g16dump package not present yet; skipping the import probe")
        return []

    problems = []
    try:
        pkg = importlib.import_module("g16dump")
    except Exception as exc:  # noqa: BLE001 - the message is the point
        return [f"import g16dump failed without pyscf/gauopen installed: {exc!r}"]

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name == GAUOPEN_MODULE:
            continue
        try:
            importlib.import_module(f"g16dump.{mod.name}")
        except Exception as exc:  # noqa: BLE001
            problems.append(
                f"g16dump.{mod.name} does not import without pyscf/gauopen: {exc!r}"
            )
    if not problems:
        print("every g16dump module except matfile imports with numpy alone")
    return problems


def main() -> int:
    problems = check_requirements() + check_forbidden() + check_package_imports()
    if problems:
        print("\nFAILED: the core install is no longer numpy-only", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
