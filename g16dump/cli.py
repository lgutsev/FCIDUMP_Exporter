"""``g16dump`` command line: inspect, extract, validate, dump, rotate.

No path in this module is hardcoded; everything comes from arguments. Each
subcommand prints what it did and what the result contains, because the usual
failure of a pipeline like this one is not a crash but a plausible number
produced from the wrong input.

Errors say what is scientifically wrong -- that the alpha- and beta-derived
effective Hamiltonians disagree, that the MO coefficients are not orthonormal in
the supplied AO overlap, that a multiplicity is incompatible with the electron
count -- rather than surfacing a raw numpy exception. Anything unanticipated is
caught in :func:`main` and reported with its type, so a genuine bug is still
visible but never arrives as a bare traceback.

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

REFERENCE_HELP = (
    "the reference type. Required and never inferred: a .mat does not "
    "distinguish these unambiguously, and the wrong choice produces a "
    "plausible, wrong Hamiltonian."
)


# ------------------------------------------------------------------ inspect


def _add_inspect(subparsers) -> None:
    parser = subparsers.add_parser(
        "inspect",
        help="report whether extract can read a .mat, and if not, why",
        description=(
            "Check a Gaussian matrix-element file for the blocks g16dump needs, "
            "report which labels they were found under, and cross-check their "
            "dimensions against the header. For an exhaustive dump of "
            "everything a .mat contains, assuming nothing about labels at all, "
            "use scripts/inspect_mat.py instead."
        ),
    )
    parser.add_argument("matfile", help="the Gaussian .mat file")
    parser.add_argument("--json", dest="json_out", help="also write a JSON report here")
    parser.set_defaults(func=_run_inspect)


def _run_inspect(args) -> int:
    from .matfile import MatFileError, survey

    try:
        report = survey(args.matfile)
    except MatFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    header = report["header"]
    print(f"{report['file']}")
    print(
        f"  {header['nao']} AO, {header['nmo']} MO, {header['nelec']} electrons, "
        f"multiplicity {header['multiplicity']}"
    )
    print(
        f"  frozen {header['nfc']} core and {header['nfv']} virtual, so the "
        f"window is MOs {header['window_1based'][0]}-{header['window_1based'][1]}"
    )
    if "eri_nact" in report:
        print(f"  two-electron block spans {report['eri_nact']} orbitals")

    print("\n  blocks:")
    for quantity, entry in report["blocks"].items():
        mark = "  " if entry["label"] else ("!!" if entry["required"] else "--")
        found = entry["label"] or "(absent)"
        elements = f"  {entry['elements']} elements" if entry.get("elements") else ""
        print(f"    {mark} {quantity:18s} {found}{elements}")

    print("\n  scalars:")
    for name, value in report["scalars"].items():
        shown = "(absent)" if value is None else f"{value:.10f}"
        print(f"       {name:18s} {shown}")

    print()
    for note in report["notes"]:
        print(f"  note: {note}")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nJSON report written to {args.json_out}")

    blocked = any("cannot run" in note for note in report["notes"])
    return 1 if blocked else 0


# ------------------------------------------------------------------ extract


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
        help=REFERENCE_HELP,
    )
    parser.add_argument(
        "--window",
        nargs=2,
        type=int,
        metavar=("NFIRST", "NLAST"),
        help=(
            "the 1-based, inclusive active window from the Gaussian route, "
            "stored in the bundle unchanged. Omit to take the partition the "
            "file reports; either way it is cross-checked against the dimension "
            "of the two-electron block."
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
    elif not bundle.has_fock:
        print(
            "\nnote: Gaussian Fock matrix unavailable, so the rebuilt-Fock path "
            "is required before this bundle can produce a Hamiltonian."
        )
    return 0


# ----------------------------------------------------------------- validate


def _add_validate(subparsers) -> None:
    parser = subparsers.add_parser(
        "validate",
        help="check an .npz bundle against the schema and report what it holds",
        description=(
            "Load a bundle, assert every invariant the schema promises, and "
            "print what it contains. With --hamiltonian, also fold the frozen "
            "core, which exercises the alpha/beta consistency check and "
            "compares E_ref against the SCF energy the job recorded."
        ),
    )
    parser.add_argument("bundle", help="the .npz bundle")
    parser.add_argument(
        "--hamiltonian",
        action="store_true",
        help="also build the active-space Hamiltonian and check it",
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
        f"  NORB / NELEC / MS2       {hamiltonian.nact} / "
        f"{hamiltonian.nelec_active} / {hamiltonian.ms2}\n"
        f"  h' alpha/beta agreement  {hamiltonian.spin_deviation:.3e}\n"
        f"  E_core                   {hamiltonian.e_core:.10f} Ha\n"
        f"  E_act                    {hamiltonian.e_act:.10f} Ha\n"
        f"  E_ref                    {hamiltonian.e_ref:.10f} Ha"
    )
    if bundle.escf is not None:
        difference = abs(hamiltonian.e_ref - bundle.escf)
        verdict = "agrees" if difference < 1e-6 else "DISAGREES"
        print(
            f"  escf from the job        {bundle.escf:.10f} Ha  "
            f"({verdict}, {difference:.3e})"
        )
        if verdict == "DISAGREES":
            return 1
    return 0


# --------------------------------------------------------------------- dump


def _add_dump(subparsers) -> None:
    parser = subparsers.add_parser(
        "dump",
        help="write an FCIDUMP from an .npz bundle (M3, not implemented)",
        description=(
            "Fold the frozen core and write the active-space Hamiltonian as a "
            "standard FCIDUMP. The Hamiltonian gates run first, so a bundle "
            "that cannot produce a sound Hamiltonian never reaches the writer."
        ),
    )
    parser.add_argument("bundle", help="the .npz bundle")
    parser.add_argument("--out", required=True, help="path for the FCIDUMP")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="omit two-electron integrals smaller than this in magnitude",
    )
    parser.add_argument(
        "--isym", type=int, default=1, help="ISYM for the FCIDUMP namelist"
    )
    parser.set_defaults(func=_run_dump)


def _run_dump(args) -> int:
    from .write import write_fcidump

    try:
        bundle = load(args.bundle)
        hamiltonian = active_hamiltonian(bundle)
        write_fcidump(
            hamiltonian,
            args.out,
            isym=args.isym,
            threshold=args.threshold,
            provenance=bundle.provenance,
        )
    except (BundleError, HamiltonianError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}")
    return 0


# ------------------------------------------------------------------- rotate


def _add_rotate(subparsers) -> None:
    parser = subparsers.add_parser(
        "rotate",
        help="rotate a bundle's active space (M4, not implemented)",
        description=(
            "Apply a real orthogonal transformation entirely within the active "
            "space and write the rotated bundle. The many-body spectrum is "
            "invariant under this, which makes it a correctness test as much as "
            "a feature."
        ),
    )
    parser.add_argument("bundle", help="the .npz bundle")
    parser.add_argument(
        "--rotation",
        required=True,
        help="a .npy file holding the real orthogonal (nact, nact) matrix",
    )
    parser.add_argument("--out", required=True, help="path for the rotated bundle")
    parser.set_defaults(func=_run_rotate)


def _run_rotate(args) -> int:
    import numpy as np

    from .rotate import rotate_active_space

    try:
        bundle = load(args.bundle)
        rotation = np.load(args.rotation)
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: cannot read the rotation matrix: {exc}", file=sys.stderr)
        return 1

    try:
        save(rotate_active_space(bundle, rotation), args.out)
    except (BundleError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}")
    return 0


# ------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="g16dump",
        description=(
            "FCIDUMP files for SHCI and DMRG from Gaussian 16 windowed MO "
            "integrals, with correct open-shell frozen-core algebra."
        ),
        epilog=(
            "The usual order is: inspect a .mat, extract it to an .npz bundle, "
            "validate the bundle, then dump an FCIDUMP. Kohn-Sham orbitals "
            "always need their Fock matrices rebuilt; the stored KS matrix is "
            "not the HF Fock operator."
        ),
    )
    parser.add_argument("--version", action="version", version=f"g16dump {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    _add_inspect(subparsers)
    _add_extract(subparsers)
    _add_validate(subparsers)
    _add_dump(subparsers)
    _add_rotate(subparsers)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except (BundleError, HamiltonianError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - a bare traceback helps nobody here
        # Unanticipated, so it is reported as such rather than dressed up as a
        # scientific diagnosis. The type is kept: this is a bug to be fixed.
        print(
            f"error: {args.command} failed unexpectedly with "
            f"{type(exc).__name__}: {exc}\n"
            f"This is a bug in g16dump rather than a problem with the input. "
            f"Please report it with the command you ran.",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
