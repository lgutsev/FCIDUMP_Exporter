"""The core path must stay NumPy-only.

This is a design constraint, not a preference: PySCF and MOKIT are validation
and fallback dependencies, gauopen is needed only by the reader, and a stray
top-level import of any of them turns a lightweight package into one that
cannot be installed where it is actually used.

Enforced here rather than by convention, because an accidental import is exactly
the kind of thing that survives review and is discovered on a cluster.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: Modules that must import with NumPy alone.
CORE_MODULES = [
    "g16dump",
    "g16dump.errors",
    "g16dump.bundle",
    "g16dump.hamiltonian",
    "g16dump.write",
    "g16dump.rotate",
    "g16dump.cli",
    "g16dump.solvers",
    "g16dump.manifest",
    "g16dump.sweep",
    "g16dump.aoorder",
]

#: Dependencies that must not be pulled in by importing the core.
OPTIONAL_DEPENDENCIES = ["pyscf", "mokit", "QCMatEl", "qcmatrixio", "scipy", "h5py"]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_does_not_import_optional_dependencies(module):
    """Import the module in a fresh interpreter and check what came with it."""
    script = f"""
import sys
import {module}
leaked = [name for name in {OPTIONAL_DEPENDENCIES!r} if name in sys.modules]
print(",".join(leaked))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, (
        f"importing {module} pulled in {leaked}. These are optional "
        f"dependencies and must be imported lazily, inside the function that "
        f"needs them."
    )


def test_matfile_imports_without_gauopen():
    """The reader module must import even where gauopen cannot be installed.

    Only calling ``extract`` should require it, and then with a clear message.
    """
    script = """
import sys
import g16dump.matfile
assert "QCMatEl" not in sys.modules
from g16dump.errors import MissingDependencyError
try:
    g16dump.matfile._import_qcmatel()
except MissingDependencyError as exc:
    assert "gauopen" in str(exc)
    assert "PYTHONPATH" in str(exc)
    print("clean error")
else:
    print("gauopen present")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in ("clean error", "gauopen present")


def test_pyproject_declares_only_numpy_as_a_runtime_dependency():
    text = (REPO / "pyproject.toml").read_text()
    body = text.split("[project.optional-dependencies]")[0]
    dependencies = body.split("dependencies = [")[1].split("]")[0]
    assert "numpy" in dependencies
    for name in ("pyscf", "mokit", "scipy", "h5py"):
        assert name not in dependencies, (
            f"{name} appears as a core runtime dependency; it belongs in the "
            f"[validate] extra."
        )
