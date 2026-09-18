"""``python -m benchmarks <command>``: validate, sweep, analyze.

Deliberately not a console script in ``pyproject.toml``: the benchmark suite is
repository content rather than part of the installed ``g16dump`` package, and it
is run from a checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import analyze, manifest as mf, sweep as sweep_module


def _validate(argv):
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks validate",
        description="Validate benchmark manifests. With no argument, validates the committed suite.",
    )
    parser.add_argument("paths", nargs="*", help="manifest files or directories")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parent.parent

    manifests = []
    for name in args.paths or [str(mf.systems_dir())]:
        path = Path(name)
        if path.is_dir():
            manifests.extend(mf.load_all(path))
        elif path.is_file():
            manifests.append(mf.load(path))
        else:
            manifests.append(mf.load(mf.systems_dir() / f"{name}.json"))

    failed = 0
    for man in manifests:
        problems = mf.validate(man, repo_root=repo_root)
        status = man.get("status", "?")
        if problems:
            failed += 1
            print(f"FAIL  {man.get('id')}")
            for problem in problems:
                print(f"        {problem}")
        else:
            window = (man.get("active_space") or {}).get("window")
            detail = (f"window {window['nfirst_1based']}-{window['nlast_1based']}"
                      if window else "window from policy")
            print(f"ok    {man.get('id'):<28s} tier {man.get('tier')}  {status:<8s} {detail}")
    print(f"\n{len(manifests) - failed}/{len(manifests)} manifest(s) valid")
    return 1 if failed else 0


COMMANDS = {"validate": _validate, "sweep": sweep_module.main, "analyze": analyze.main}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(f"usage: python -m benchmarks {{{'|'.join(COMMANDS)}}} ...", file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
