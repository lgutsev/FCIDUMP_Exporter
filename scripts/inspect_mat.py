#!/usr/bin/env python3
"""Probe a Gaussian matrix-element file and report exactly what is inside it.

This script assumes nothing about label names, array packing or object layout.
It discovers and reports.  Its output is the evidence used to decide how
``g16dump/matfile.py`` must read a ``.mat`` file, so run it on the cluster (where
gauopen is installed) and keep the output.

Usage
-----
    python3 inspect_mat.py FILE.mat [--json OUT.json] [--max-expand N]

``--json`` also writes a machine-readable copy of everything printed, which is
the convenient thing to hand back to the developer.

Requires gauopen on ``PYTHONPATH`` (it provides ``QCMatEl`` and ``qcmatrixio``).
Nothing here is imported by the g16dump package itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

# Questions this probe exists to answer.  Keys are searched case-insensitively
# as substrings of the matrix-element labels; we report what matched, never
# assuming a particular spelling exists.
LABEL_PROBES = {
    "fock": "AO Fock matrices (needed for the stored-Fock path)",
    "mo coefficients": "MO coefficients, alpha and possibly beta",
    "2e integrals": "windowed MO two-electron integrals",
    "orbital energies": "orbital energies (diagnostic only)",
    "core hamiltonian": "AO core Hamiltonian",
    "overlap": "AO overlap (used to pin down the C row/column convention)",
    "density": "SCF densities",
    "kinetic": "AO kinetic energy",
    "trans mo coefficients": (
        "the MO coefficients actually used in the transformation, with frozen "
        "core and virtuals removed -- an unambiguous readout of the window "
        "Gaussian applied (gauopen v2 only)"
    ),
}

# Scalar names to try through me.scalar().  Unknown names are expected to fail;
# every failure is reported rather than hidden, so a miss here is information.
SCALAR_PROBES = [
    "ENUCREP",
    "ESCF",
    "escf",
    "SCF ENERGY",
    "ETOTAL",
    "ETOT",
    "ETHERM",
    "VIRIAL RATIO",
    "TOTAL ENERGY",
    "ENERGY",
    "EUHF",
    "ERHF",
    "TE SCF ENERGY",
    # QCMatEl checks this for 1.0 to confirm the job completed, so it turns
    # "the .mat looks truncated" into a one-line diagnosis.
    "JOB STATUS",
]

#: Header fields that answer the route questions directly, with what each value
#: means. ITran is the decisive one: it says whether the transformation ran at
#: all, which no amount of label-hunting can tell you.
HEADER_VERDICTS = {
    "itran": {
        0: "NO MO INTEGRALS WERE STORED. The route did not run a "
           "transformation -- check Output=(MatrixElement,MO2ElectronIntegrals) "
           "and Tran=(Full,Force) before reading anything else here.",
        4: "PARTIAL transformation: only MOs involving at least one occupied "
           "orbital. Not enough for a FCIDUMP; Tran=Full was not in effect.",
        5: "FULL transformation. This is what the method needs.",
    },
    "icgu": {
        "note": "three digits klm; m is 1 for RHF/GHF and 2 for UHF, so it is "
                "the cleanest restricted/unrestricted discriminator",
    },
}


_MISSING = object()


def _safe_getattr(obj, name):
    """getattr that swallows *any* exception, not just AttributeError.

    gauopen objects have properties that raise; a bare getattr(obj, name, None)
    lets that propagate and kills the probe mid-run. Returns _MISSING on failure.
    """
    try:
        return getattr(obj, name)
    except Exception:
        return _MISSING


def _is_callable_attr(obj, name) -> bool:
    value = _safe_getattr(obj, name)
    return value is not _MISSING and callable(value)


def _brief(value, limit: int = 200) -> str:
    """repr() that never floods the terminal."""
    try:
        text = repr(value)
    except Exception as exc:  # pragma: no cover - defensive
        return f"<unreprable: {exc!r}>"
    return text if len(text) <= limit else text[:limit] + "..."


def _triangular_root(length: int) -> int | None:
    """Return n if length == n*(n+1)/2 for a positive integer n, else None."""
    n = int(round(((8 * length + 1) ** 0.5 - 1) / 2))
    for cand in (n - 1, n, n + 1):
        if cand > 0 and cand * (cand + 1) // 2 == length:
            return cand
    return None


def _integer_root(length: int, power: int) -> int | None:
    """Return n if length == n**power for a positive integer n, else None."""
    n = int(round(length ** (1.0 / power)))
    for cand in (n - 1, n, n + 1):
        if cand > 0 and cand**power == length:
            return cand
    return None


def packing_hypotheses(length: int) -> list[str]:
    """Describe every simple packing whose dimension comes out an exact integer.

    This is how we determine storage (square / lower-triangular / 8-fold-packed)
    from the file itself instead of assuming it.  A hypothesis listed here is a
    candidate, not a conclusion -- cross-check against nbasis/nbsuse/nact.
    """
    hypotheses = []

    n = _integer_root(length, 2)
    if n is not None:
        hypotheses.append(f"square n x n with n={n}")

    n = _triangular_root(length)
    if n is not None:
        hypotheses.append(f"lower-triangular packed with n={n}")

    n = _integer_root(length, 4)
    if n is not None:
        hypotheses.append(f"full 4-index n^4 with n={n}")

    # 8-fold packed two-electron integrals: npair = n(n+1)/2 pairs, then the
    # pair index is itself triangular.
    npair = _triangular_root(length)
    if npair is not None:
        n = _triangular_root(npair)
        if n is not None:
            hypotheses.append(f"8-fold packed (pq|rs) with n={n}")

    # 4-fold packed: npair**2.
    npair = _integer_root(length, 2)
    if npair is not None:
        n = _triangular_root(npair)
        if n is not None:
            hypotheses.append(f"4-fold packed (pq|rs) with n={n}")

    return hypotheses or ["no simple packing matches this length"]


def describe_array(arr) -> dict:
    """Shape, dtype, size, packing candidates and a few sample values."""
    info: dict = {}
    for attr in ("shape", "dtype", "size", "ndim", "flags"):
        try:
            value = getattr(arr, attr)
        except Exception:
            continue
        if attr == "flags":
            # C-order vs Fortran-order is exactly the mistake we must not make.
            info["c_contiguous"] = bool(getattr(value, "c_contiguous", False))
            info["f_contiguous"] = bool(getattr(value, "f_contiguous", False))
        else:
            info[attr] = str(value)

    try:
        size = int(arr.size)
        info["packing_candidates"] = packing_hypotheses(size)
    except Exception:
        pass

    try:
        flat = arr.ravel()
        info["first_values"] = [float(v) for v in flat[:6]]
        info["last_values"] = [float(v) for v in flat[-3:]]
        info["abs_max"] = float(abs(flat).max()) if flat.size else 0.0
    except Exception:
        try:
            info["first_values"] = _brief(arr[:6])
        except Exception:
            pass
    return info


def describe_matrix_entry(label: str, obj, max_expand: int) -> dict:
    """Everything we can learn about one matlist entry without assuming its API."""
    entry: dict = {"label": label, "python_type": type(obj).__name__}

    # Any non-callable public attribute the object carries.  This is where
    # nrow/ncol/dimens/asym/type live, whatever they happen to be named.
    attrs: dict = {}
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:
            attrs[name] = f"<raised {type(exc).__name__}: {exc}>"
            continue
        if callable(value):
            continue
        if name == "array":
            continue  # described separately below
        attrs[name] = _brief(value)
    entry["attributes"] = attrs
    entry["methods"] = [
        name
        for name in sorted(dir(obj))
        if not name.startswith("_") and _is_callable_attr(obj, name)
    ]

    try:
        entry["array"] = describe_array(obj.array)
    except Exception as exc:
        entry["array"] = f"<no usable .array: {type(exc).__name__}: {exc}>"

    # expand() is what the legacy code calls; find out what it actually returns.
    # Guard the size so a large window cannot blow up the probe.
    if hasattr(obj, "expand"):
        try:
            size = int(getattr(obj, "array").size)
        except Exception:
            size = 0
        if size and size > max_expand:
            entry["expand"] = (
                f"<skipped: packed size {size} exceeds --max-expand {max_expand}>"
            )
        else:
            try:
                entry["expand"] = describe_array(obj.expand())
            except Exception as exc:
                entry["expand"] = f"<expand() raised {type(exc).__name__}: {exc}>"
    else:
        entry["expand"] = "<no expand() method>"

    return entry


def probe_scalars(me) -> dict:
    """Find the scalar names this file carries, including the SCF energy."""
    result: dict = {}

    # Preferred: whatever container the object exposes.
    for container in ("scalars", "scalarlist", "scalar_names"):
        if hasattr(me, container):
            try:
                result[f"me.{container}"] = _brief(getattr(me, container), 2000)
            except Exception as exc:
                result[f"me.{container}"] = f"<raised {exc!r}>"

    # gauopen keeps the canonical scalar-name table in qcmatrixio.
    try:
        import qcmatrixio

        names = [n for n in dir(qcmatrixio) if "scalar" in n.lower()]
        result["qcmatrixio scalar-ish names"] = names
        for name in names:
            try:
                result[f"qcmatrixio.{name}"] = _brief(getattr(qcmatrixio, name), 2000)
            except Exception:
                pass
    except Exception as exc:
        result["qcmatrixio"] = f"<import failed: {exc!r}>"

    probed: dict = {}
    if hasattr(me, "scalar"):
        for name in SCALAR_PROBES:
            try:
                probed[name] = _brief(me.scalar(name))
            except Exception as exc:
                probed[name] = f"<raised {type(exc).__name__}: {exc}>"
    result["me.scalar(...) probes"] = probed
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matfile", help="path to the Gaussian .mat file")
    parser.add_argument("--json", dest="json_out", help="also write a JSON report here")
    parser.add_argument(
        "--max-expand",
        type=int,
        default=20_000_000,
        help="skip expand() on entries whose packed array exceeds this many elements",
    )
    args = parser.parse_args(argv)

    report: dict = {"matfile": args.matfile}

    try:
        import QCMatEl
    except Exception:
        print("Could not import QCMatEl. Is gauopen on PYTHONPATH?", file=sys.stderr)
        traceback.print_exc()
        return 2

    report["QCMatEl_module_file"] = getattr(QCMatEl, "__file__", "<unknown>")
    try:
        import qcmatrixio

        report["qcmatrixio_module_file"] = getattr(qcmatrixio, "__file__", "<unknown>")
    except Exception as exc:
        report["qcmatrixio_module_file"] = f"<import failed: {exc!r}>"

    print("=" * 78)
    print(f"file: {args.matfile}")
    print(f"QCMatEl:    {report['QCMatEl_module_file']}")
    print(f"qcmatrixio: {report['qcmatrixio_module_file']}")
    print("=" * 78)

    me = QCMatEl.MatEl(file=args.matfile)

    # ---------------------------------------------------------------- header
    print("\n--- MatEl object attributes (non-callable, public) ---")
    header: dict = {}
    for name in sorted(dir(me)):
        if name.startswith("_") or name == "matlist":
            continue
        try:
            value = getattr(me, name)
        except Exception as exc:
            header[name] = f"<raised {type(exc).__name__}: {exc}>"
            continue
        if callable(value):
            continue
        header[name] = _brief(value)
    for name, value in header.items():
        print(f"  {name:28s} = {value}")
    report["matel_attributes"] = header

    report["matel_methods"] = [
        name
        for name in sorted(dir(me))
        if not name.startswith("_") and _is_callable_attr(me, name)
    ]
    print("\n--- MatEl methods ---")
    print("  " + ", ".join(report["matel_methods"]))

    # --------------------------------------------------------------- scalars
    print("\n--- scalars ---")
    scalars = probe_scalars(me)
    report["scalars"] = scalars
    for key, value in scalars.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for name, res in value.items():
                print(f"      {name:16s} -> {res}")
        else:
            print(f"  {key}: {value}")

    # --------------------------------------------------------------- matlist
    keys = sorted(me.matlist.keys())
    report["matlist_keys"] = keys
    print(f"\n--- matlist: {len(keys)} entries ---")
    for key in keys:
        print(f"  {key!r}")

    print("\n--- matlist entries in detail ---")
    entries = []
    for key in keys:
        try:
            entry = describe_matrix_entry(key, me.matlist[key], args.max_expand)
        except Exception as exc:
            entry = {"label": key, "error": f"{type(exc).__name__}: {exc}"}
        entries.append(entry)

        print(f"\n  [{key}]")
        if "error" in entry:
            print(f"    *** PROBE FAILED: {entry['error']}")
            continue
        print(f"    python type: {entry.get('python_type')}")
        arr = entry.get("array")
        if isinstance(arr, dict):
            print(
                f"    array: shape={arr.get('shape')} dtype={arr.get('dtype')} "
                f"size={arr.get('size')} "
                f"C={arr.get('c_contiguous')} F={arr.get('f_contiguous')}"
            )
            for cand in arr.get("packing_candidates", []):
                print(f"      packing candidate: {cand}")
            print(f"      first values: {arr.get('first_values')}")
        else:
            print(f"    array: {arr}")
        exp = entry.get("expand")
        if isinstance(exp, dict):
            print(f"    expand(): shape={exp.get('shape')} size={exp.get('size')}")
        else:
            print(f"    expand(): {exp}")
        attrs = entry.get("attributes", {})
        if attrs:
            print(
                "    attributes: "
                + ", ".join(f"{k}={v}" for k, v in attrs.items() if len(str(v)) < 60)
            )
    report["matlist_entries"] = entries

    # ------------------------------------------------------- targeted answers
    print("\n" + "=" * 78)
    print("QUESTIONS THIS PROBE EXISTS TO ANSWER")
    print("=" * 78)
    answers: dict = {}
    for needle, why in LABEL_PROBES.items():
        matches = [k for k in keys if needle in k.lower()]
        answers[needle] = matches
        status = "PRESENT" if matches else "ABSENT "
        print(f"\n  [{status}] {needle}  ({why})")
        for match in matches:
            print(f"            {match!r}")

    beta_mo = [k for k in keys if "beta" in k.lower() and "mo coefficients" in k.lower()]
    answers["beta_mo_coefficients"] = beta_mo
    print(f"\n  BETA MO COEFFICIENTS present: {bool(beta_mo)}  {beta_mo}")
    print(
        "    (a restricted job writes only ALPHA blocks, so a BETA block here "
        "means the job was unrestricted -- UHF, which this project does not "
        "accept)"
    )

    # ------------------------------------------------- did the transform run?
    # Reported last, because it is the thing to read first: every other answer
    # above is meaningless if no transformation happened.
    print("\n" + "-" * 78)
    itran = _safe_getattr(me, "itran")
    answers["itran"] = None if itran is _MISSING else itran
    if itran is _MISSING:
        print("  ITran: not exposed by this gauopen build")
    else:
        verdict = HEADER_VERDICTS["itran"].get(
            int(itran), f"unrecognised value {itran}"
        )
        print(f"  ITran = {itran}: {verdict}")

    for name in ("nfc", "nfv", "nbasis", "nbsuse", "ne", "multip", "icharg", "icgu"):
        value = _safe_getattr(me, name)
        answers[name] = None if value is _MISSING else value
        if value is not _MISSING:
            print(f"  {name:8s} = {value}")
    print(f"  (icgu: {HEADER_VERDICTS['icgu']['note']})")

    nfc, nfv = answers.get("nfc"), answers.get("nfv")
    nbsuse = answers.get("nbsuse")
    if None not in (nfc, nfv, nbsuse):
        nact = int(nbsuse) - int(nfc) - int(nfv)
        print(
            f"\n  => the window Gaussian applied is {int(nfc) + 1}-"
            f"{int(nbsuse) - int(nfv)} (1-based), {nact} active orbitals, so a "
            f"full two-electron block should expand to {nact}**4 = {nact ** 4} "
            f"elements. Check that against the packing candidates above: if it "
            f"instead matches nbsuse**4, Window= was ignored."
        )
        answers["nact_from_header"] = nact

    report["answers"] = answers
    print(
        "\nNOTE: 'packing candidate' lines are hypotheses derived from array length"
        "\nalone. Cross-check the n they report against nbasis / nbsuse and the"
        "\nactive-window size before writing any reader code.\n"
    )

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"JSON report written to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
