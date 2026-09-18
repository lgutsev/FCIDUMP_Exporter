"""Systematic active-space sweeps, and the analysis that makes them mean something.

Given one system and a set of active windows, this generates the manifest and
the Gaussian jobs for the whole series, then collects the results into a table
that answers the question the series was run to answer: *has the active space
converged?*

The convergence criterion that matters is not the absolute energy -- which is
still moving -- but the change from one active space to the next, and, for a
spin-state problem, the change in the *splitting*, which converges differently
and often more slowly than either state alone.

Why this is part of the package rather than a script
----------------------------------------------------
For transition-metal systems the answer depends strongly on where the window is
drawn. Correlation in metalloporphyrins extends well beyond the nominal metal d
orbitals into the ligand framework, and both the magnitude and the state
ordering can change as the window grows. A workflow that makes the sweep easy,
and that reports deviations between consecutive windows rather than a single
number, is what keeps that from being discovered by accident.
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence

from .errors import ValidationError, require
from .manifest import MANIFEST_VERSION, validate_manifest, validate_system


def windows_around(
    reference_orbital: int, sizes: Sequence[int], *, occupied_fraction: float = 0.5
) -> list:
    """Active windows of the given sizes, centred on a reference orbital.

    ``reference_orbital`` is 1-based, normally the HOMO. ``occupied_fraction``
    sets how much of each window sits at or below it -- 0.5 splits evenly, which
    is the usual starting point for a valence active space.

    Returns ``[first, last]`` pairs, 1-based and inclusive, matching the
    manifest's ``window`` field and Gaussian's ``Window=``.
    """
    require(
        reference_orbital >= 1,
        f"reference_orbital is 1-based and must be >= 1, got {reference_orbital}.",
    )
    require(
        0.0 < occupied_fraction < 1.0,
        f"occupied_fraction must lie strictly between 0 and 1, got "
        f"{occupied_fraction}.",
    )

    windows = []
    for size in sizes:
        require(size >= 1, f"window size must be >= 1, got {size}.")
        below = int(round(size * occupied_fraction))
        below = max(1, min(size, below))
        first = reference_orbital - below + 1
        require(
            first >= 1,
            f"a window of {size} orbitals with occupied_fraction "
            f"{occupied_fraction} would start at orbital {first}, below the "
            f"first orbital. Lower occupied_fraction, or use a smaller window.",
        )
        windows.append([first, first + size - 1])
    return windows


def sweep_manifest(
    base_system: dict,
    windows: Sequence[Sequence[int]],
    *,
    name_template: str = "{name}_w{nact:02d}",
) -> dict:
    """One manifest with an entry per active window.

    Every entry is validated, so a window that cannot hold the reference
    determinant is rejected here rather than after a queue wait.
    """
    require(len(windows) > 0, "no windows given; there is nothing to sweep.")

    systems = []
    for window in windows:
        first, last = int(window[0]), int(window[1])
        system = copy.deepcopy(base_system)
        system["window"] = [first, last]
        system["name"] = name_template.format(
            name=base_system["name"], nact=last - first + 1, first=first, last=last
        )
        validate_system(system)
        systems.append(system)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "description": (
            f"Active-space sweep for {base_system['name']}: "
            f"{len(systems)} windows."
        ),
        "sweep": {
            "base_system": base_system["name"],
            "windows": [list(w) for w in windows],
        },
        "systems": systems,
    }
    validate_manifest(manifest)
    return manifest


def spin_state_sweep_manifest(
    base_system: dict,
    windows: Sequence[Sequence[int]],
    multiplicities: Sequence[int],
) -> dict:
    """A sweep over active windows *and* spin states.

    Spin-state splittings converge more slowly with active space than either
    state does alone, so a sweep that fixes the multiplicity cannot answer
    whether the splitting is converged. This runs the cross product.
    """
    require(
        len(multiplicities) >= 2,
        f"a spin-state sweep needs at least two multiplicities, got "
        f"{list(multiplicities)}.",
    )

    systems = []
    for multiplicity in multiplicities:
        for window in windows:
            first, last = int(window[0]), int(window[1])
            system = copy.deepcopy(base_system)
            system["window"] = [first, last]
            system["multiplicity"] = int(multiplicity)
            system["reference"] = (
                system["reference"].replace("RO", "R")
                if multiplicity == 1
                else system["reference"].replace("RHF", "ROHF").replace("RKS", "ROKS")
            )
            system["name"] = (
                f"{base_system['name']}_m{multiplicity}_w{last - first + 1:02d}"
            )
            validate_system(system)
            systems.append(system)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "description": (
            f"Spin-state sweep for {base_system['name']}: "
            f"{len(multiplicities)} multiplicities x {len(windows)} windows."
        ),
        "sweep": {
            "base_system": base_system["name"],
            "windows": [list(w) for w in windows],
            "multiplicities": list(multiplicities),
        },
        "systems": systems,
    }
    validate_manifest(manifest)
    return manifest


# ------------------------------------------------------------------ analysis


def natural_orbital_diagnostics(occupations) -> dict:
    """Summarize natural-orbital occupations of an active space.

    ``n_effectively_unpaired`` is the Head-Gordon index
    ``sum_i min(n_i, 2 - n_i)``, which is 0 for a closed-shell determinant and
    grows with multireference character. ``n_partially_occupied`` counts
    orbitals away from 0 and 2 by more than 0.02, the usual rule of thumb for
    "this orbital is doing something".

    These are the numbers that say whether an active space is well chosen: an
    orbital pinned at 2.00 or 0.00 is dead weight, and one at 1.9 near the edge
    of the window means the window is too small.
    """
    import numpy as np

    occupations = np.asarray(occupations, dtype=float)
    require(
        occupations.ndim == 1,
        f"occupations must be a 1-D array, got shape {occupations.shape}.",
    )

    unpaired = float(np.sum(np.minimum(occupations, 2.0 - occupations)))
    partially = int(np.sum((occupations > 0.02) & (occupations < 1.98)))

    return {
        "n_electrons": float(occupations.sum()),
        "n_effectively_unpaired": unpaired,
        "n_partially_occupied": partially,
        "max_occupation": float(occupations.max()) if occupations.size else 0.0,
        "min_occupation": float(occupations.min()) if occupations.size else 0.0,
        "occupations": [float(x) for x in occupations],
    }


def collect_sweep(results: Sequence[dict]) -> list:
    """Turn raw per-window results into a convergence table.

    Each input entry needs ``nact``, ``nelec_act`` and ``energy``; ``runtime``,
    ``memory_mb``, ``multiplicity`` and ``occupations`` are used when present.

    Adds ``deviation``: the energy change from the next *larger* active space,
    which is the quantity that actually indicates convergence. It is ``None``
    for the largest window, where there is nothing to compare against -- stated
    rather than reported as zero, because zero would read as "converged".
    """
    for entry in results:
        missing = [k for k in ("nact", "nelec_act", "energy") if k not in entry]
        require(
            not missing,
            f"sweep result {entry.get('name', '<unnamed>')!r} is missing "
            f"{missing}. Each result needs at least nact, nelec_act and energy.",
        )

    ordered = sorted(results, key=lambda e: e["nact"])
    table = []
    for index, entry in enumerate(ordered):
        row = dict(entry)
        if index + 1 < len(ordered):
            row["deviation"] = float(ordered[index + 1]["energy"] - entry["energy"])
        else:
            row["deviation"] = None
        if "occupations" in entry and entry["occupations"] is not None:
            row.update(natural_orbital_diagnostics(entry["occupations"]))
        table.append(row)
    return table


def spin_splittings(results: Sequence[dict], *, unit: str = "eV") -> list:
    """Splittings between spin states at each active space.

    Returns one row per ``nact``, containing every multiplicity found there and
    the splitting relative to the lowest state, plus how much that splitting
    moved from the next larger active space. That last column is the one to
    read: a splitting still moving by more than the effect being studied means
    the active space is not converged, however smooth the individual energies
    look.
    """
    factors = {"Ha": 1.0, "eV": 27.211386245988, "kcal": 627.509474063}
    require(
        unit in factors,
        f"unknown unit {unit!r}; expected one of {', '.join(factors)}.",
    )
    factor = factors[unit]

    by_window: dict = {}
    for entry in results:
        require(
            "multiplicity" in entry,
            f"spin splittings need a 'multiplicity' on every result; "
            f"{entry.get('name', '<unnamed>')!r} has none.",
        )
        by_window.setdefault(entry["nact"], []).append(entry)

    rows = []
    for nact in sorted(by_window):
        states = sorted(by_window[nact], key=lambda e: e["energy"])
        ground = states[0]
        rows.append(
            {
                "nact": nact,
                "ground_multiplicity": ground["multiplicity"],
                "unit": unit,
                "splittings": {
                    state["multiplicity"]: (state["energy"] - ground["energy"]) * factor
                    for state in states
                },
            }
        )

    for index, row in enumerate(rows):
        if index + 1 < len(rows):
            later = rows[index + 1]["splittings"]
            row["splitting_change"] = {
                multiplicity: later[multiplicity] - value
                for multiplicity, value in row["splittings"].items()
                if multiplicity in later
            }
        else:
            row["splitting_change"] = None

    return rows


def format_sweep_table(table: Sequence[dict]) -> str:
    """A readable convergence table."""
    header = (
        f"{'nact':>5s} {'nelec':>6s} {'energy / Ha':>16s} {'dE(next) / Ha':>15s} "
        f"{'NEU':>7s} {'runtime / s':>12s}"
    )
    lines = [header, "-" * len(header)]
    for row in table:
        deviation = row.get("deviation")
        deviation_text = "  (largest)" if deviation is None else f"{deviation:15.9f}"
        unpaired = row.get("n_effectively_unpaired")
        unpaired_text = "      -" if unpaired is None else f"{unpaired:7.3f}"
        runtime = row.get("runtime")
        runtime_text = "           -" if runtime is None else f"{runtime:12.1f}"
        lines.append(
            f"{row['nact']:5d} {row['nelec_act']:6d} {row['energy']:16.9f} "
            f"{deviation_text} {unpaired_text} {runtime_text}"
        )
    return "\n".join(lines)
