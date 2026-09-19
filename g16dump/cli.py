"""``g16dump`` command line: extract, dump, validate, rotate.

No path in this module is hardcoded; everything comes from arguments. Each
subcommand prints what it did and what the result contains, because the usual
failure of a pipeline like this one is not a crash but a plausible number
produced from the wrong input.

``dump`` and ``rotate`` are wired to their modules but those are M3/M4 seams, so
they report that plainly rather than pretending.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .bundle import BundleError, load, save
from .hamiltonian import HamiltonianError, active_hamiltonian

KS_WARNING = (
    "Kohn-Sham orbitals: the stored KS matrix contains exchange-correlation and "
    "is never the HF Fock operator. The active-space Hamiltonian is the HF "
    "Hamiltonian evaluated in the KS orbitals, so the Fock matrices must be "
    "rebuilt through PySCF."
)


def _add_extract(subparsers) -> None:
    parser = subparsers.add_parser(
        "extract",
        help="read a Gaussian .mat into an .npz bundle (needs gauopen)",
        description="Read a Gaussian 16 matrix-element file into an .npz bundle.",
    )
    parser.add_argument("matfile", help="the Gaussian .mat file")
    parser.add_argument("--fch", help="the matching formatted checkpoint file")
    parser.add_argument("--out", required=True, help="path for the .npz bundle")
    parser.add_argument(
        "--reference",
        required=True,
        choices=["RHF", "ROHF", "RKS", "ROKS"],
        help=(
            "the reference type. Required and never inferred: a .mat does not "
            "distinguish these unambiguously, and the wrong choice produces a "
            "plausible, wrong Hamiltonian."
        ),
    )
    parser.add_argument(
        "--window",
        nargs=2,
        type=int,
        metavar=("NFIRST", "NLAST"),
        help=(
            "the 1-based active window from the Gaussian route. Omit to take "
            "the partition the file reports; either way it is cross-checked "
            "against the dimension of the two-electron block."
        ),
    )
    parser.add_argument("--route", help="the Gaussian route line, for provenance")
    parser.add_argument("--basis", help="basis set name, for provenance")
    parser.add_argument("--method", help="method name, for provenance")
    parser.set_defaults(func=_run_extract)


def _run_extract(args) -> int:
    from .matfile import MatFileError, extract

    try:
        bundle = extract(
            args.matfile,
            reference=args.reference,
            window=tuple(args.window) if args.window else None,
            fch=args.fch,
            route=args.route,
            basis=args.basis,
            method=args.method,
        )
        out = save(bundle, args.out)
    except (MatFileError, BundleError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {out}")
    print(bundle.describe())
    if bundle.is_ks:
        print(f"\nnote: {KS_WARNING}")
    elif bundle.fock_source == "none":
        print(
            "\nnote: no Fock matrix was found in the .mat, so the rebuilt-Fock "
            "path is required before this bundle can produce a Hamiltonian."
        )
    return 0


def _add_validate(subparsers) -> None:
    parser = subparsers.add_parser(
        "validate",
        help="check an .npz bundle against the schema and report what it holds",
    )
    parser.add_argument("bundle", help="the .npz bundle")
    parser.add_argument(
        "--hamiltonian",
        action="store_true",
        help=(
            "also build the active-space Hamiltonian, which exercises the "
            "alpha/beta consistency check"
        ),
    )
    parser.add_argument(
        "--provenance", action="store_true", help="print the provenance record"
    )
    parser.set_defaults(func=_run_validate)


def _run_validate(args) -> int:
    try:
        bundle = load(args.bundle)
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{args.bundle}: valid against schema v{bundle.schema_version}")
    print(bundle.describe())
    if args.provenance:
        print("\nprovenance:")
        print(json.dumps(bundle.provenance, indent=2, sort_keys=True))
    if bundle.is_ks:
        print(f"\nnote: {KS_WARNING}")

    if not args.hamiltonian:
        return 0

    try:
        hamiltonian = active_hamiltonian(bundle)
    except HamiltonianError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    print(
        f"\nactive-space Hamiltonian ({hamiltonian.fock_source} Fock):\n"
        f"  h' alpha/beta agreement  {hamiltonian.spin_deviation:.3e}\n"
        f"  E_core                   {hamiltonian.e_core:.10f} Ha\n"
        f"  E_act                    {hamiltonian.e_act:.10f} Ha\n"
        f"  E_ref                    {hamiltonian.e_ref:.10f} Ha"
    )
    if bundle.e_scf is not None:
        difference = abs(hamiltonian.e_ref - bundle.e_scf)
        verdict = "agrees" if difference < 1e-6 else "DISAGREES"
        print(
            f"  E_scf from Gaussian      {bundle.e_scf:.10f} Ha  "
            f"({verdict}, {difference:.3e})"
        )
    return 0


def _add_dump(subparsers) -> None:
    parser = subparsers.add_parser(
        "dump",
        help="write an FCIDUMP from an .npz bundle",
        description=(
            "Write the active-space Hamiltonian of a bundle as a FCIDUMP. The "
            "alpha/beta consistency gate runs first, so a bundle carrying a "
            "Roothaan operator is refused before anything is written."
        ),
    )
    parser.add_argument("bundle", help="the .npz bundle")
    parser.add_argument("--out", required=True, help="path for the FCIDUMP")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help=(
            "omit two-electron integrals smaller than this in magnitude. The "
            "default of 0 writes every symmetry-unique integral; anything else "
            "makes the file stop reproducing the reference energy exactly"
        ),
    )
    parser.add_argument(
        "--isym", type=int, default=1, help="ISYM for the FCIDUMP namelist"
    )
    parser.add_argument(
        "--orbsym",
        help=(
            "comma-separated irrep labels, one per active orbital (1-8). "
            "Defaults to C1, i.e. all ones"
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=None,
        help=(
            "digits after the decimal point in each integral. The default, 16, "
            "is the smallest that reproduces a double exactly"
        ),
    )
    parser.add_argument(
        "--dice-nocc",
        action="store_true",
        help="also print Dice's nocc block for the reference determinant",
    )
    parser.set_defaults(func=_run_dump)


def _run_dump(args) -> int:
    from .write import WriteError, dice_occupation_line, provenance_path, write_fcidump

    options = {}
    if args.precision is not None:
        options["precision"] = args.precision
    if args.orbsym:
        try:
            options["orbsym"] = [int(x) for x in args.orbsym.replace(",", " ").split()]
        except ValueError:
            print(
                f"error: --orbsym must be integers, got {args.orbsym!r}",
                file=sys.stderr,
            )
            return 1

    try:
        bundle = load(args.bundle)
        hamiltonian = active_hamiltonian(bundle)
        out = write_fcidump(
            hamiltonian,
            args.out,
            isym=args.isym,
            threshold=args.threshold,
            provenance=bundle.provenance,
            **options,
        )
    except (BundleError, HamiltonianError, WriteError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {out}")
    print(f"wrote {provenance_path(out)}")
    print(
        f"  NORB   {hamiltonian.nact}\n"
        f"  NELEC  {hamiltonian.nelec_act}\n"
        f"  MS2    {hamiltonian.ms2}\n"
        f"  E_core {hamiltonian.e_core:.10f} Ha\n"
        f"  E_ref  {hamiltonian.e_ref:.10f} Ha"
    )
    if args.threshold > 0:
        print(
            f"\nnote: integrals below {args.threshold:g} were omitted, so this "
            f"file no longer reproduces E_ref exactly."
        )
    if args.dice_nocc:
        print(f"\nDice nocc block:\n{dice_occupation_line(hamiltonian)}", end="")
    return 0


def _add_rotate(subparsers) -> None:
    parser = subparsers.add_parser(
        "rotate", help="rotate a bundle's active space (M4, not implemented)"
    )
    parser.add_argument("bundle", help="the .npz bundle")
    parser.add_argument(
        "--rotation", required=True, help="a .npy file holding the (nact, nact) matrix"
    )
    parser.add_argument("--out", required=True, help="path for the rotated bundle")
    parser.set_defaults(func=_run_rotate)


def _run_rotate(args) -> int:
    import numpy as np

    from .rotate import rotate_active_space

    try:
        bundle = load(args.bundle)
        rotation = np.load(args.rotation)
        rotated = rotate_active_space(bundle, rotation)
        save(rotated, args.out)
    except (BundleError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="g16dump",
        description=(
            "FCIDUMP files for SHCI and DMRG from Gaussian 16 windowed MO "
            "integrals, with correct open-shell frozen-core algebra."
        ),
    )
    parser.add_argument("--version", action="version", version=f"g16dump {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    _add_extract(subparsers)
    _add_dump(subparsers)
    _add_validate(subparsers)
    _add_rotate(subparsers)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
