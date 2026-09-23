"""Command line interface.

    g16dump extract JOB.mat --ref-type ROHF --fch JOB.fch --out JOB.npz
    g16dump dump    JOB.npz --out FCIDUMP
    g16dump validate JOB.npz
    g16dump rotate  JOB.npz --rotation U.npy --out rotated.npz
    g16dump info    JOB.npz

No path is ever constructed: every input and output is named on the command
line. Every output carries provenance -- the source file, the active window, the
reference type, the basis and method where known, the g16dump commit, the schema
version, and whether the Fock matrices came from Gaussian or were rebuilt with
PySCF. That last distinction changes the physics, so it is recorded rather than
assumed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from typing import Optional

import numpy as np

from . import __version__
from .bundle import Bundle, load, save, validate
from .errors import G16DumpError
from .hamiltonian import (
    active_space_hamiltonian,
    load_hamiltonian,
    save_hamiltonian,
)
from .rotate import check_orthogonal, rotate
from .write import write_dice_nocc, write_from_hamiltonian

KS_WARNING = (
    "Kohn-Sham orbitals: the active-space Hamiltonian is always the HF "
    "Hamiltonian evaluated in the KS orbitals. The stored KS matrix contains "
    "exchange-correlation and is never used as a Fock operator; --rebuild-fock "
    "is required."
)


def _provenance_stamp(command: str, **extra) -> dict:
    """The fields every output carries, whatever produced it."""
    from .matfile import _git_commit

    stamp = {
        "g16dump_version": __version__,
        "g16dump_commit": _git_commit(),
        "command": command,
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv[1:],
    }
    stamp.update(extra)
    return stamp


def _scf_tolerance(args):
    """None to skip, an explicit float, or "auto" (chosen per bundle)."""
    if args.no_scf_check:
        return None
    return "auto" if args.tol_scf is None else args.tol_scf


# ------------------------------------------------------------------ extract


def cmd_extract(args) -> int:
    from .matfile import extract

    if args.ref_type in ("RKS", "ROKS", "UKS") and not args.rebuild_fock:
        print(f"note: {KS_WARNING}", file=sys.stderr)
        print(
            "error: --rebuild-fock is required for a Kohn-Sham reference.",
            file=sys.stderr,
        )
        return 2

    bundle = extract(
        args.matfile,
        ref_type=args.ref_type,
        fch=args.fch,
        rebuild_fock=args.rebuild_fock,
    )
    bundle.provenance.update(
        _provenance_stamp("extract", basis=args.basis, method=args.method)
    )
    save(bundle, args.out)

    print(f"wrote {args.out}")
    _print_bundle_summary(bundle)
    return 0


# --------------------------------------------------------------------- dump


def _load_hamiltonian_or_bundle(path: str, tol_spin: float, tol_scf: Optional[float]):
    """Accept either a bundle or an already-reduced Hamiltonian file."""
    with np.load(path, allow_pickle=False) as data:
        is_hamiltonian = "hamiltonian_schema_version" in data.files

    if is_hamiltonian:
        ash, provenance = load_hamiltonian(path)
        return ash, provenance, None

    bundle = load(path)
    validate(bundle)
    ash = active_space_hamiltonian(bundle, tol_spin=tol_spin, tol_scf=tol_scf)
    return ash, dict(bundle.provenance), bundle


def cmd_dump(args) -> int:
    ash, provenance, bundle = _load_hamiltonian_or_bundle(
        args.bundle, args.tol_spin, _scf_tolerance(args)
    )

    orbsym = None
    if args.orbsym:
        orbsym = [int(x) for x in args.orbsym.replace(",", " ").split()]

    written = write_from_hamiltonian(
        args.out,
        ash,
        orbsym=orbsym,
        isym=args.isym,
        threshold=args.threshold,
    )
    print(f"wrote {args.out}: {written} integral lines")
    print(
        f"  NORB={ash.h_eff.shape[0]} NELEC={ash.nelec_act} MS2={ash.ms2} "
        f"E_core={ash.e_core:.12f}"
    )
    if "spin_error" in ash.diagnostics:
        print(f"  max|h'(alpha) - h'(beta)| = {ash.diagnostics['spin_error']:.3e}")
    if "scf_error" in ash.diagnostics:
        print(
            f"  |E_ref - E_scf|           = {ash.diagnostics['scf_error']:.3e}"
            f"  (tolerance {ash.diagnostics['scf_tolerance']:.0e})"
        )
        if ash.diagnostics["scf_tolerance"] > 1e-8:
            print(
                "  note: Fock rebuilt from a .fch; its 9-digit basis data limits "
                "E_ref to ~1e-7 Ha for transition metals, so the gate is 1e-5."
            )

    if args.dice_input:
        write_dice_nocc(args.dice_input, ash)
        print(f"wrote {args.dice_input}: Dice nocc block for the reference determinant")

    if args.save_hamiltonian:
        stamp = dict(provenance)
        stamp.update(_provenance_stamp("dump", fcidump=args.out))
        stamp["diagnostics"] = ash.diagnostics
        save_hamiltonian(ash, args.save_hamiltonian, stamp)
        print(f"wrote {args.save_hamiltonian}: reduced Hamiltonian")

    return 0


# ----------------------------------------------------------------- validate


def _print_bundle_summary(bundle: Bundle) -> None:
    print(f"  reference     {bundle.ref_type}  (Fock from {bundle.fock_source})")
    print(
        f"  electrons     {bundle.nelec} total, {bundle.nalpha} alpha / "
        f"{bundle.nbeta} beta, multiplicity {bundle.multiplicity}"
    )
    print(
        f"  orbitals      {bundle.nbasis} basis, {bundle.nmo} MO, "
        f"{bundle.ncore} core + {bundle.nact} active + {bundle.nfv} virtual"
    )
    print(
        f"  active space  {bundle.nelec_act} electrons in {bundle.nact} orbitals, "
        f"MS2={bundle.ms2}"
    )
    print(f"  E_nuc         {bundle.e_nuc:.12f}")
    if bundle.e_scf is not None:
        print(f"  E_scf         {bundle.e_scf:.12f}")


def cmd_validate(args) -> int:
    bundle = load(args.bundle)
    residuals = validate(bundle)

    print(f"{args.bundle}: schema version {bundle.schema_version}, valid")
    _print_bundle_summary(bundle)
    print("  residuals:")
    for name, value in sorted(residuals.items()):
        print(f"    {name:26s} {value:.3e}")

    if args.check_hamiltonian:
        ash = active_space_hamiltonian(
            bundle,
            tol_spin=args.tol_spin,
            tol_scf=_scf_tolerance(args),
        )
        print("  active-space Hamiltonian:")
        print(f"    E_ref                      {ash.e_ref:.12f}")
        print(f"    E_act                      {ash.e_act:.12f}")
        print(f"    E_core                     {ash.e_core:.12f}")
        for key in ("spin_error", "scf_error"):
            if key in ash.diagnostics:
                print(f"    {key:26s} {ash.diagnostics[key]:.3e}")

    if bundle.is_ks:
        print(f"\nnote: {KS_WARNING}")
    return 0


# ------------------------------------------------------------------- rotate


def cmd_rotate(args) -> int:
    ash, provenance, _ = _load_hamiltonian_or_bundle(
        args.bundle, args.tol_spin, _scf_tolerance(args)
    )

    u = np.load(args.rotation)
    if u.ndim != 2:
        print(
            f"error: {args.rotation} holds an array of shape {u.shape}; a "
            f"rotation must be a square matrix over the active space.",
            file=sys.stderr,
        )
        return 2
    residual = check_orthogonal(u, args.tol_orthogonal)

    h_rot, eri_rot, e_core = rotate(
        ash.h_eff, ash.eri_act, ash.e_core, u, tol=args.tol_orthogonal
    )

    rotated = type(ash)(
        h_eff=h_rot,
        eri_act=eri_rot,
        e_core=e_core,
        e_ref=ash.e_ref,
        e_act=ash.e_act,
        nelec_act=ash.nelec_act,
        ms2=ash.ms2,
        nocc_a_act=ash.nocc_a_act,
        nocc_b_act=ash.nocc_b_act,
        diagnostics=dict(ash.diagnostics),
    )

    stamp = dict(provenance)
    stamp.update(
        _provenance_stamp(
            "rotate",
            rotation_file=args.rotation,
            rotation_orthogonality_residual=residual,
            source=args.bundle,
        )
    )
    stamp["diagnostics"] = rotated.diagnostics
    stamp["rotation_note"] = (
        "The active-space Hamiltonian was rotated. E_core is unchanged because "
        "the rotation does not touch the frozen core. E_ref and E_act refer to "
        "the ORIGINAL reference determinant: a general rotation mixes occupied "
        "with virtual active orbitals, so the aufbau determinant in the rotated "
        "basis is a different state. Every observable of the Hamiltonian, the "
        "FCI spectrum included, is unchanged."
    )
    save_hamiltonian(rotated, args.out, stamp)

    print(f"wrote {args.out}: rotated Hamiltonian")
    print(f"  max|U.T U - I| = {residual:.3e}")
    print(f"  E_core         = {e_core:.12f} (unchanged)")
    print(f"  dump it with:  g16dump dump {args.out} --out FCIDUMP")
    return 0


# --------------------------------------------------------------------- info


def cmd_info(args) -> int:
    try:
        with np.load(args.bundle, allow_pickle=False) as data:
            is_hamiltonian = "hamiltonian_schema_version" in data.files
    except Exception as exc:
        print(f"error: cannot read {args.bundle}: {exc}", file=sys.stderr)
        return 2

    if is_hamiltonian:
        ash, provenance = load_hamiltonian(args.bundle)
        print(f"{args.bundle}: reduced active-space Hamiltonian")
        print(f"  NORB={ash.h_eff.shape[0]} NELEC={ash.nelec_act} MS2={ash.ms2}")
        print(f"  E_core={ash.e_core:.12f}")
    else:
        bundle = load(args.bundle)
        print(f"{args.bundle}: bundle, schema version {bundle.schema_version}")
        _print_bundle_summary(bundle)
        provenance = bundle.provenance

    print("  provenance:")
    print(json.dumps(provenance, indent=4, default=str))
    return 0


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="g16dump",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"g16dump {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_tolerances(p):
        p.add_argument(
            "--tol-spin",
            type=float,
            default=1e-8,
            help="tolerance on max|h'(alpha) - h'(beta)| in Ha (default 1e-8)",
        )
        p.add_argument(
            "--tol-scf",
            type=float,
            default=None,
            help="tolerance on |E_ref - E_scf| in Ha (default: 1e-8, or 1e-5 "
            "when the Fock was rebuilt from a .fch, whose 9-digit basis data "
            "limits precision)",
        )
        p.add_argument(
            "--no-scf-check",
            action="store_true",
            help="skip the E_ref vs E_scf check (for orbitals that solve no SCF)",
        )

    # --- extract
    p = sub.add_parser("extract", help="Gaussian .mat -> .npz bundle")
    p.add_argument("matfile", help="Gaussian matrix-element file")
    p.add_argument("--out", required=True, help="output .npz bundle")
    p.add_argument(
        "--ref-type",
        required=True,
        choices=["RHF", "ROHF", "RKS", "ROKS", "UHF", "UKS"],
        help="reference type; never inferred, because a .mat does not "
        "distinguish Hartree-Fock from Kohn-Sham orbitals",
    )
    p.add_argument("--fch", help="matching .fch, for the overlap and --rebuild-fock")
    p.add_argument(
        "--rebuild-fock",
        action="store_true",
        help="rebuild F^s = Hcore + J[Pa+Pb] - K[Ps] with PySCF instead of "
        "reading a stored Fock matrix; mandatory for Kohn-Sham orbitals",
    )
    p.add_argument("--basis", help="basis set name, recorded in provenance")
    p.add_argument("--method", help="method name, recorded in provenance")
    p.set_defaults(func=cmd_extract)

    # --- dump
    p = sub.add_parser("dump", help=".npz bundle -> FCIDUMP")
    p.add_argument("bundle", help="input .npz bundle or rotated Hamiltonian")
    p.add_argument("--out", required=True, help="output FCIDUMP path")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="skip integrals with |v| < threshold (default 0, write everything)",
    )
    p.add_argument("--orbsym", help="comma-separated ORBSYM labels (default all 1)")
    p.add_argument("--isym", type=int, default=1, help="ISYM value (default 1)")
    p.add_argument("--dice-input", help="also write the Dice nocc block here")
    p.add_argument(
        "--save-hamiltonian", help="also save the reduced Hamiltonian as .npz"
    )
    _add_tolerances(p)
    p.set_defaults(func=cmd_dump)

    # --- validate
    p = sub.add_parser("validate", help="check a bundle for consistency")
    p.add_argument("bundle", help="input .npz bundle")
    p.add_argument(
        "--check-hamiltonian",
        action="store_true",
        help="also build h' and report E_ref, E_core and the residuals",
    )
    _add_tolerances(p)
    p.set_defaults(func=cmd_validate)

    # --- rotate
    p = sub.add_parser("rotate", help="rotate the active-space Hamiltonian")
    p.add_argument("bundle", help="input .npz bundle or Hamiltonian")
    p.add_argument("--rotation", required=True, help=".npy file holding U")
    p.add_argument("--out", required=True, help="output rotated Hamiltonian .npz")
    p.add_argument(
        "--tol-orthogonal",
        type=float,
        default=1e-10,
        help="tolerance on max|U.T U - I| (default 1e-10)",
    )
    _add_tolerances(p)
    p.set_defaults(func=cmd_rotate)

    # --- info
    p = sub.add_parser("info", help="print a summary and the provenance")
    p.add_argument("bundle", help="input .npz bundle or Hamiltonian")
    p.set_defaults(func=cmd_info)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except G16DumpError as exc:
        # Our own errors already explain the physics; a traceback would only
        # bury the message.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
