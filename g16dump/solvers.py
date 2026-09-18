"""Input generation and output parsing for Dice (SHCI) and Block2 (DMRG).

Plain functions returning strings. No plugin architecture, no class hierarchy,
no registry -- adding a solver means adding two functions.

CI never runs the solvers. Generated inputs are compared against golden
fixtures in ``tests/data/golden/``, which is enough to catch a format
regression.

A caveat worth stating plainly
------------------------------
The **input** formats here follow ``legacy/input.dat`` (Dice), which is known to
have been accepted by a real Dice build, and the documented Block2 keyword set.
The **output parsers** are written from the documented output formats and have
*not* been checked against real solver output, because none is available here.
They are deliberately loose -- they look for labelled energies rather than
fixed column positions -- and :func:`parse_dice_output` and
:func:`parse_block2_output` return whatever they can find rather than guessing.
Run one small job and send the output back before relying on them.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from .errors import require
from .hamiltonian import ActiveSpaceHamiltonian
from .write import dice_nocc_block

#: Default variational schedule for Dice: (iteration, epsilon1) pairs.
DEFAULT_DICE_SCHEDULE = ((0, 9e-4), (3, 4e-4))

#: Default Block2 sweep schedule: (start sweep, bond dimension, tolerance, noise).
DEFAULT_BLOCK2_SCHEDULE = (
    (0, 250, 1e-5, 1e-4),
    (4, 500, 1e-6, 1e-5),
    (8, 1000, 1e-7, 0.0),
)


# --------------------------------------------------------------------- Dice


def dice_input(
    ash: ActiveSpaceHamiltonian,
    *,
    fcidump: str = "FCIDUMP",
    schedule: Sequence = DEFAULT_DICE_SCHEDULE,
    maxiter: int = 12,
    epsilon2: float = 1e-8,
    sample_n: int = 200,
    target_error: float = 4e-4,
    do_rdm: bool = True,
    prefix: Optional[str] = None,
) -> str:
    """A Dice ``input.dat`` for this Hamiltonian.

    The ``nocc`` block is generated from the Hamiltonian's own occupation
    counts, so the reference determinant cannot drift out of step with the
    integrals -- the silent failure mode the legacy workflow left open.
    """
    require(
        maxiter > 0, f"maxiter must be positive, got {maxiter}."
    )
    require(
        len(schedule) > 0,
        "the Dice variational schedule is empty; at least one "
        "(iteration, epsilon1) pair is needed.",
    )

    lines = [dice_nocc_block(ash.nocc_a_act, ash.nocc_b_act), ""]
    if fcidump != "FCIDUMP":
        lines.append(f"orbitals {fcidump}")
        lines.append("")

    lines.append("#variational keywords")
    lines.append("schedule")
    for iteration, epsilon1 in schedule:
        lines.append(f"{int(iteration)} {float(epsilon1):g}")
    lines.append("end")
    lines.append(f"maxiter {int(maxiter)}")
    if do_rdm:
        # The 1-RDM is what feeds natural-orbital rotation; ask for it by default.
        lines.append("DoRDM")
    lines.append("")

    lines.append("#perturbative keywords")
    lines.append(f"epsilon2 {float(epsilon2):g}")
    lines.append(f"sampleN {int(sample_n)}")
    lines.append(f"targetError {float(target_error):g}")
    if prefix:
        lines.append(f"prefix {prefix}")
    lines.append("")

    return "\n".join(lines)


def write_dice_input(path: str, ash: ActiveSpaceHamiltonian, **kwargs) -> None:
    with open(path, "w") as handle:
        handle.write(dice_input(ash, **kwargs))


def parse_dice_output(text: str) -> dict:
    """Pull energies and convergence information out of a Dice output.

    Returns a dict with whatever was found: ``variational_energy``,
    ``pt_energy``, ``total_energy``, ``iterations`` (a list of
    ``(iteration, ndets, energy)``), and ``converged``.

    Unverified against real output -- see the module docstring.
    """
    result: dict = {"iterations": [], "converged": None}

    # Variational iterations: "iter  eps1  ndets  energy ..." shaped rows.
    for match in re.finditer(
        r"^\s*(\d+)\s+[\d.eE+-]+\s+(\d+)\s+(-?\d+\.\d+)", text, re.MULTILINE
    ):
        result["iterations"].append(
            (int(match.group(1)), int(match.group(2)), float(match.group(3)))
        )

    patterns = {
        "variational_energy": r"[Vv]ariational\s+[Ee]nergy[^\-\d]*(-?\d+\.\d+)",
        "pt_energy": r"(?:PT|[Pp]erturbation)\s+[Ee]nergy[^\-\d]*(-?\d+\.\d+)",
        "total_energy": r"[Tt]otal\s+[Ee]nergy[^\-\d]*(-?\d+\.\d+)",
    }
    for key, pattern in patterns.items():
        found = re.findall(pattern, text)
        if found:
            result[key] = float(found[-1])

    if result["iterations"] and "variational_energy" not in result:
        result["variational_energy"] = result["iterations"][-1][2]

    if re.search(r"[Cc]onverged", text):
        result["converged"] = True

    return result


# ------------------------------------------------------------------- Block2


def block2_input(
    ash: ActiveSpaceHamiltonian,
    *,
    fcidump: str = "FCIDUMP",
    schedule: Sequence = DEFAULT_BLOCK2_SCHEDULE,
    maxiter: int = 30,
    sweep_tol: float = 1e-7,
    symmetry: str = "c1",
    irrep: int = 1,
    scratch: Optional[str] = None,
) -> str:
    """A Block2 (StackBlock-style) input file for this Hamiltonian.

    ``spin`` in Block2's input is 2*S_z, which is our ``ms2``; taking it from
    the Hamiltonian rather than from the user keeps it consistent with the
    FCIDUMP header.
    """
    require(
        len(schedule) > 0,
        "the Block2 sweep schedule is empty; at least one "
        "(sweep, bond dimension, tolerance, noise) entry is needed.",
    )

    lines = [
        f"sym {symmetry}",
        f"orbitals {fcidump}",
        f"nelec {ash.nelec_act}",
        f"spin {ash.ms2}",
        f"irrep {irrep}",
        "",
        "hf_occ integral",
        "schedule",
    ]
    for sweep, bond_dimension, tolerance, noise in schedule:
        lines.append(
            f"{int(sweep)} {int(bond_dimension)} {float(tolerance):g} {float(noise):g}"
        )
    lines += [
        "end",
        f"maxiter {int(maxiter)}",
        f"sweep_tol {float(sweep_tol):g}",
        "",
        "onepdm",
        "",
    ]
    if scratch:
        lines.append(f"prefix {scratch}")
    return "\n".join(lines) + "\n"


def write_block2_input(path: str, ash: ActiveSpaceHamiltonian, **kwargs) -> None:
    with open(path, "w") as handle:
        handle.write(block2_input(ash, **kwargs))


def parse_block2_output(text: str) -> dict:
    """Pull sweep energies and convergence information out of a Block2 output.

    Returns ``sweeps`` (a list of ``(sweep, bond dimension, energy)``),
    ``energy`` (the final one) and ``converged``.

    Unverified against real output -- see the module docstring.
    """
    result: dict = {"sweeps": [], "converged": None}

    for match in re.finditer(
        r"Sweep\s*=\s*(\d+).*?(?:M|Bond dimension)\s*=\s*(\d+).*?"
        r"(?:E|Energy)\s*=\s*(-?\d+\.\d+)",
        text,
        re.DOTALL | re.IGNORECASE,
    ):
        result["sweeps"].append(
            (int(match.group(1)), int(match.group(2)), float(match.group(3)))
        )

    energies = re.findall(r"DMRG\s+[Ee]nergy\s*=\s*(-?\d+\.\d+)", text)
    if energies:
        result["energy"] = float(energies[-1])
    elif result["sweeps"]:
        result["energy"] = result["sweeps"][-1][2]

    if re.search(r"[Cc]onverged", text):
        result["converged"] = True

    return result


# ------------------------------------------------------- PySCF as a solver


def pyscf_fci_energy(
    ash: ActiveSpaceHamiltonian, *, max_orbitals: int = 14
) -> float:
    """Exact FCI for the active space, when PySCF is available and it is small.

    This is the validation solver: it needs no external executable, so it can be
    used to check that a generated FCIDUMP really describes the Hamiltonian we
    think it does. Refuses spaces large enough to be a mistake rather than
    silently starting an intractable calculation.
    """
    from .errors import MissingDependencyError

    norb = ash.h_eff.shape[0]
    require(
        norb <= max_orbitals,
        f"the active space has {norb} orbitals, above the {max_orbitals}-orbital "
        f"limit for the exact FCI validation solver. Raise max_orbitals "
        f"deliberately if you mean it -- FCI cost grows factorially. Use Dice or "
        f"Block2 for a space this size.",
    )

    try:
        from pyscf import fci
    except ImportError as exc:
        raise MissingDependencyError(
            "the FCI validation solver needs PySCF. Install it with "
            "`pip install 'g16dump[validate]'`."
        ) from exc

    energy = fci.direct_spin1.kernel(
        ash.h_eff,
        ash.eri_act,
        norb,
        (ash.nocc_a_act, ash.nocc_b_act),
        verbose=0,
    )[0]
    return float(energy + ash.e_core)
