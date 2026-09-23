"""The Gaussian -> PySCF bridge: a ``.fch`` read through MOKIT.

The rebuilt-Fock path builds ``J`` and ``K`` with PySCF, on a ``Mole`` that
MOKIT's ``load_mol_from_fch`` reconstructs from the job's formatted checkpoint.
Everything else in a bundle -- the MO coefficients, the overlap, the core
Hamiltonian -- comes from the ``.mat``, in Gaussian's AO basis. The two AO
bases hold the same functions but not in the same order, and not with the same
normalization:

* PySCF sorts an atom's shells by angular momentum and splits Pople ``SP``
  shells, so 6-31G is reordered even though it has no d functions;
* pure shells are ``m = 0, +1, -1, ...`` in Gaussian and ``m = -l..+l`` in PySCF;
* Cartesian shells differ in component order and Gaussian normalizes every
  component to one, PySCF does not.

MOKIT is the compatibility layer that knows all of this; its ``fch2py`` is the
routine MOKIT itself uses to hand Gaussian orbitals to PySCF. This module does
not re-derive the conventions. It asks MOKIT for the transformation, by passing
unit vectors through ``fch2py``, and applies it to the full-precision ``.mat``
coefficients: ``C_pyscf = X @ C_gaussian``. The coefficients printed in the
``.fch`` itself carry nine significant figures and are used only to check that
the ``.fch`` and the ``.mat`` describe the same orbitals.

Before anything is rebuilt, the conversion has to prove itself on quantities
both sides computed independently: the overlap and the core Hamiltonian from
the ``.mat`` must equal PySCF's after transformation, and the converted
orbitals must be orthonormal in PySCF's overlap. Orthonormality alone would not
be enough -- a wrong permutation of equivalent functions can keep it -- which is
why the AO matrices are compared element by element.

Nothing here imports MOKIT or PySCF at module level.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .hamiltonian import HamiltonianError

#: ``max |X.T S_pyscf X - S_mat|``. Gaussian functions are normalized, so this
#: is on a scale of one. The ``.fch`` prints exponents to nine significant
#: figures, which moves the rebuilt overlap by ~1e-10; a wrong order or a wrong
#: ``.fch`` moves it by O(0.1).
AO_OVERLAP_TOL = 1e-6

#: ``max |X.T H_pyscf X - H_mat| / max |H_mat|``. Relative, because the tight
#: core functions of a heavy atom make the core Hamiltonian large.
AO_HCORE_RTOL = 1e-7

#: Orthonormality of the converted orbitals, and their agreement with MOKIT's
#: own transfer of the ``.fch`` orbitals (printed to nine figures there).
ORBITAL_TOL = 1e-6

#: ``max |X C_fch - fch2py(C_fch)|``: that MOKIT's transfer is the linear map
#: this module extracted from it. Exact in principle.
TRANSFER_TOL = 1e-10

MOKIT_INSTALL_HINT = (
    "MOKIT is not on PyPI. Install it from its conda channel,\n"
    "  conda install mokit -c mokit/label/cf -c conda-forge\n"
    "or build it from source (https://gitlab.com/jxzou/mokit), so that both the "
    "Python package 'mokit' and MOKIT's command-line tools (bas_fch2py among "
    "them) are available."
)


class FchError(HamiltonianError):
    """A ``.fch`` cannot be used to rebuild anything for this bundle."""


# ------------------------------------------------------------------ MOKIT


def require_mokit():
    """Import MOKIT's Gaussian module, or explain what is missing and how to fix it."""
    try:
        from mokit.lib import gaussian
    except ImportError as exc:
        raise FchError(
            f"the rebuilt-Fock path reads the .fch through MOKIT, which is not "
            f"importable ({exc}).\n{MOKIT_INSTALL_HINT}"
        ) from exc
    # load_mol_from_fch shells out to bas_fch2py and imports the script it
    # writes. Without the executable that import fails with a meaningless
    # "No module named 'gau1234'", so the absence is reported here instead.
    if shutil.which("bas_fch2py") is None:
        raise FchError(
            "MOKIT's Python package is importable but its bas_fch2py executable "
            "is not on PATH, and load_mol_from_fch needs it.\n" + MOKIT_INSTALL_HINT
        )
    try:
        import pyscf  # noqa: F401 - the Mole MOKIT builds is a PySCF object
    except ImportError as exc:
        raise FchError(
            "the rebuilt-Fock path needs pyscf, which is not installed. Install it "
            "with: pip install 'g16dump[validate]'"
        ) from exc
    return gaussian


def mokit_version() -> str:
    try:
        import mokit

        return str(getattr(mokit, "__version__", "unknown"))
    except ImportError:
        return "not installed"


@contextmanager
def _scratch():
    """Run MOKIT in a private directory: it writes and imports files in the cwd.

    ``load_mol_from_fch`` also puts that directory at the front of ``sys.path``
    and, on its next call, asks whether ``sys.path[0]`` is the cwd -- which fails
    once the directory is gone. The path is therefore restored as well.

    And it names the script it imports ``gau<random 1..10000>``. A second call
    that draws a name already in ``sys.modules`` gets the *earlier* molecule
    back, silently, whatever ``.fch`` it was given. Those modules are dropped
    before and after every call so a name can never be reused.
    """
    previous = os.getcwd()
    saved_path = list(sys.path)
    _forget_mokit_scripts()
    with tempfile.TemporaryDirectory(prefix="g16dump-mokit-") as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(previous)
            sys.path[:] = saved_path
            _forget_mokit_scripts()


_MOKIT_SCRIPT = re.compile(r"^gau\d+$")


def _forget_mokit_scripts() -> None:
    for name in [n for n in sys.modules if _MOKIT_SCRIPT.match(n)]:
        del sys.modules[name]


def mol_from_fch(path):
    """Load a PySCF ``Mole`` from a Gaussian formatted checkpoint, through MOKIT.

    The ``Mole``'s AO basis is PySCF's, not Gaussian's; see
    :func:`ao_transform_from_fch` for the map between them.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FchError(f"{path}: no such .fch file")
    gaussian = require_mokit()
    with _scratch():
        try:
            return gaussian.load_mol_from_fch(str(path))
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise FchError(
                f"MOKIT could not build a molecule from {path} ({type(exc).__name__}: "
                f"{exc}). MOKIT reads a .fch section by section in the order "
                f"Gaussian's formchk writes it, so a hand-edited or truncated file "
                f"fails here; regenerate it with formchk."
            ) from exc


def mokit_mo_coeff(path) -> np.ndarray:
    """MOKIT's own transfer of the ``.fch`` alpha orbitals into PySCF's AO basis."""
    path = Path(path).resolve()
    gaussian = require_mokit()
    with _scratch():
        coeff = np.asarray(gaussian.mo_fch2py(str(path)), dtype=np.float64)
    if coeff.ndim == 3:
        # MOKIT returns (alpha, beta) for an unrestricted .fch.
        if not np.allclose(coeff[0], coeff[1], atol=1e-8):
            raise FchError(
                f"{path} holds different alpha and beta orbitals, i.e. a UHF "
                f"wavefunction, which a spin-restricted FCIDUMP cannot represent."
            )
        coeff = coeff[0]
    return coeff


# ------------------------------------------------------- the .fch, read plainly

_ARRAY = re.compile(r"^(\S.*?)\s+([IRC])\s+N=\s*(\d+)\s*$")
_SCALAR = re.compile(r"^(\S.*?)\s+([IR])\s+(\S+)\s*$")


def read_fch_scalar(path, label: str):
    for line in Path(path).read_text().splitlines():
        match = _SCALAR.match(line)
        if match and match.group(1).strip() == label:
            value = match.group(3)
            return int(value) if match.group(2) == "I" else float(value)
    raise FchError(f"{path}: no scalar {label!r}")


def read_fch_array(path, label: str) -> np.ndarray:
    """One array section of a ``.fch``, exactly as Gaussian printed it."""
    lines = Path(path).read_text().splitlines()
    for index, line in enumerate(lines):
        match = _ARRAY.match(line)
        if match and match.group(1).strip() == label:
            count = int(match.group(3))
            values: list = []
            cursor = index + 1
            while len(values) < count and cursor < len(lines):
                values.extend(lines[cursor].split())
                cursor += 1
            if len(values) != count:
                raise FchError(f"{path}: {label!r} is truncated")
            dtype = np.int64 if match.group(2) == "I" else np.float64
            return np.array([v.replace("D", "E") for v in values], dtype=dtype)
    raise FchError(f"{path}: no array {label!r}")


def _replace_real_array(lines: list, label: str, values: np.ndarray) -> list:
    """``lines`` with one real array section rewritten in formchk's format."""
    out: list = []
    index = 0
    replaced = False
    while index < len(lines):
        match = _ARRAY.match(lines[index])
        if match and match.group(1).strip() == label:
            count = int(match.group(3))
            if count != values.size:
                raise FchError(f"{label!r} holds {count} values, not {values.size}")
            out.append(lines[index])
            index += 1
            seen = 0
            while seen < count:
                seen += len(lines[index].split())
                index += 1
            out.extend(
                "".join(f"{v:16.8E}" for v in values[k:k + 5])
                for k in range(0, values.size, 5)
            )
            replaced = True
            continue
        out.append(lines[index])
        index += 1
    if not replaced:
        raise FchError(f"no array {label!r} to replace")
    return out


def fch_mo_coeff(path) -> np.ndarray:
    """The alpha orbitals as printed in the ``.fch``: Gaussian AO order, column-wise."""
    nbf = read_fch_scalar(path, "Number of basis functions")
    nmo = read_fch_scalar(path, "Number of independent functions")
    return read_fch_array(path, "Alpha MO coefficients").reshape(nmo, nbf).T


def ao_transform_from_fch(path) -> np.ndarray:
    """``X`` with ``C_pyscf = X @ C_gaussian``, taken from MOKIT's ``fch2py``.

    ``fch2py`` acts on each orbital separately and linearly, so passing it unit
    vectors in place of the orbitals returns its matrix column by column. The
    result is MOKIT's transformation exactly, not a re-derivation of it. Unit
    vectors survive the ``.fch`` format's nine figures without rounding.
    """
    path = Path(path).resolve()
    nbf = read_fch_scalar(path, "Number of basis functions")
    nmo = read_fch_scalar(path, "Number of independent functions")
    lines = path.read_text().splitlines()
    labels = ["Alpha MO coefficients"]
    if any(_ARRAY.match(x) and x.startswith("Beta MO coefficients") for x in lines):
        labels.append("Beta MO coefficients")

    gaussian = require_mokit()
    transform = np.zeros((nbf, nbf))
    with _scratch() as tmp:
        probe = tmp / "probe.fch"
        for start in range(0, nbf, nmo):
            width = min(nmo, nbf - start)
            block = np.zeros((nbf, nmo))
            block[start:start + width, :width] = np.eye(width)
            edited = lines
            for label in labels:
                edited = _replace_real_array(edited, label, block.T.ravel())
            probe.write_text("\n".join(edited) + "\n")
            columns = np.asarray(gaussian.mo_fch2py(str(probe)), dtype=np.float64)
            if columns.ndim == 3:
                columns = columns[0]
            if columns.shape[0] != nbf:
                raise FchError(
                    f"MOKIT maps {nbf} Gaussian basis functions onto "
                    f"{columns.shape[0]} PySCF ones. A mixed pure/Cartesian basis "
                    f"(for example 6D 7F with f functions present) is not a "
                    f"one-to-one relabelling; rerun Gaussian with 5D 7F or 6D 10F."
                )
            transform[:, start:start + width] = columns[:, :width]
    return transform


def describe_transform(transform: np.ndarray) -> str:
    """What kind of map MOKIT applied, in words that go into provenance."""
    nonzero = np.abs(transform) > 1e-12
    if np.allclose(transform, np.eye(transform.shape[0]), atol=1e-12):
        return "identity"
    if not (np.all(nonzero.sum(axis=0) == 1) and np.all(nonzero.sum(axis=1) == 1)):
        return "general linear map"
    values = transform[nonzero]
    if np.allclose(np.abs(values), 1.0, atol=1e-12):
        return "signed permutation" if np.any(values < 0) else "permutation"
    return "permutation with per-function rescaling (Cartesian normalization)"


# --------------------------------------------------------------- the bridge


@dataclass
class AOConversion:
    """The Gaussian -> PySCF map for one bundle, and the evidence it is right."""

    fch_file: str
    mol: object
    transform: np.ndarray
    kind: str
    checks: dict = field(default_factory=dict)

    def mo_coeff_pyscf(self, mo_coeff_gaussian: np.ndarray) -> np.ndarray:
        return self.transform @ mo_coeff_gaussian

    def provenance(self) -> dict:
        import pyscf

        return {
            "method": "mokit fch2py applied to the .mat MO coefficients",
            "detail": (
                "the Gaussian->PySCF AO transformation X was read out of MOKIT's "
                "fch2py by transferring unit vectors; C_pyscf = X C_mat, and the "
                "rebuilt F^sigma = Hcore_mat + X^T (J - K)[C_pyscf] X is stored in "
                "the bundle's own (Gaussian) AO basis"
            ),
            "transform": self.kind,
            "fch_file": self.fch_file,
            "mokit_version": mokit_version(),
            "pyscf_version": str(pyscf.__version__),
            "checks": {k: v for k, v in self.checks.items()},
        }


def _max_abs(a) -> float:
    return float(np.max(np.abs(a))) if np.size(a) else 0.0


def _sign_aligned_difference(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """``max |a - s b|`` with a sign ``s`` per column, and how many flipped."""
    signs = np.sign(np.einsum("ij,ij->j", a, b))
    signs[signs == 0] = 1.0
    return _max_abs(a - b * signs), int(np.sum(signs < 0))


def gaussian_to_pyscf(
    bundle,
    fch,
    *,
    overlap_tol: float = AO_OVERLAP_TOL,
    hcore_rtol: float = AO_HCORE_RTOL,
    orbital_tol: float = ORBITAL_TOL,
) -> AOConversion:
    """Build and verify the map from ``bundle``'s AO basis to MOKIT's ``Mole``.

    Raises :class:`FchError`, naming the failed check, when the ``.fch`` does not
    belong to the bundle or the two AO bases cannot be related. Never forces a
    dimension to fit or re-orthonormalizes anything.
    """
    fch = str(Path(fch).resolve())
    mol = mol_from_fch(fch)
    from pyscf import scf  # importable: require_mokit checked

    transform = ao_transform_from_fch(fch)

    if transform.shape != (mol.nao, bundle.nao):
        raise FchError(
            f"{fch} describes {transform.shape[1]} Gaussian basis functions, but "
            f"the bundle has {bundle.nao}. This .fch is not from the job that "
            f"produced the bundle."
        )

    checks: dict = {"nao": int(bundle.nao), "nmo": int(bundle.nmo)}

    # The map MOKIT applies has to be the one extracted from it.
    fch_coeff = fch_mo_coeff(fch)
    mokit_coeff = mokit_mo_coeff(fch)
    checks["transfer_linearity"] = _max_abs(transform @ fch_coeff - mokit_coeff)
    if checks["transfer_linearity"] > TRANSFER_TOL:
        raise FchError(
            f"MOKIT's fch2py is not the linear per-orbital map g16dump extracted "
            f"from it (max deviation {checks['transfer_linearity']:.3e}); this "
            f"MOKIT version behaves differently from the one g16dump was checked "
            f"against."
        )

    overlap_pyscf = np.asarray(mol.intor("int1e_ovlp"))
    hcore_pyscf = np.asarray(scf.hf.get_hcore(mol))
    same_shape = overlap_pyscf.shape == bundle.S.shape
    scale = max(_max_abs(bundle.Hcore_ao), 1.0)

    # Before and after, so the size of what the conversion fixes is on record.
    checks["overlap_before"] = _max_abs(overlap_pyscf - bundle.S) if same_shape else None
    checks["overlap_after"] = _max_abs(transform.T @ overlap_pyscf @ transform - bundle.S)
    checks["hcore_before_rel"] = (
        _max_abs(hcore_pyscf - bundle.Hcore_ao) / scale if same_shape else None
    )
    checks["hcore_after_rel"] = (
        _max_abs(transform.T @ hcore_pyscf @ transform - bundle.Hcore_ao) / scale
    )

    if checks["overlap_after"] > overlap_tol or checks["hcore_after_rel"] > hcore_rtol:
        raise FchError(
            f"the .mat's AO matrices and the ones PySCF builds from {fch} disagree "
            f"even after MOKIT's Gaussian->PySCF transformation: overlap by "
            f"{checks['overlap_after']:.3e} (tolerance {overlap_tol:.0e}), core "
            f"Hamiltonian by {checks['hcore_after_rel']:.3e} relative (tolerance "
            f"{hcore_rtol:.0e}). The .fch is not from the same job as the .mat, "
            f"or used a different basis, geometry or pure/Cartesian setting. "
            f"Nothing can be rebuilt from it for this bundle."
        )

    mo_pyscf = transform @ bundle.C
    checks["orthonormality"] = _max_abs(
        mo_pyscf.T @ overlap_pyscf @ mo_pyscf - np.eye(bundle.nmo)
    )
    checks["orthonormality_unconverted"] = (
        _max_abs(bundle.C.T @ overlap_pyscf @ bundle.C - np.eye(bundle.nmo))
        if same_shape else None
    )
    if checks["orthonormality"] > orbital_tol:
        raise FchError(
            f"the .mat orbitals, transformed into PySCF's AO basis, are not "
            f"orthonormal in PySCF's overlap (max |C^T S C - I| = "
            f"{checks['orthonormality']:.3e}, tolerance {orbital_tol:.0e})."
        )

    if mokit_coeff.shape != mo_pyscf.shape:
        raise FchError(
            f"{fch} carries {mokit_coeff.shape[1]} orbitals over "
            f"{mokit_coeff.shape[0]} functions, the bundle {mo_pyscf.shape[1]} over "
            f"{mo_pyscf.shape[0]}. They are not from the same job."
        )
    difference, flipped = _sign_aligned_difference(mo_pyscf, mokit_coeff)
    checks["mo_coeff_vs_mokit"] = difference
    checks["mo_coeff_sign_flips"] = flipped
    checks["mo_coeff_vs_mokit_unconverted"] = (
        _sign_aligned_difference(bundle.C, mokit_coeff)[0] if same_shape else None
    )
    occ_ours = mo_pyscf[:, : bundle.nalpha]
    occ_mokit = mokit_coeff[:, : bundle.nalpha]
    checks["density_alpha_vs_mokit"] = _max_abs(
        occ_ours @ occ_ours.T - occ_mokit @ occ_mokit.T
    )
    occ_ours = mo_pyscf[:, : bundle.nbeta]
    occ_mokit = mokit_coeff[:, : bundle.nbeta]
    checks["density_beta_vs_mokit"] = _max_abs(
        occ_ours @ occ_ours.T - occ_mokit @ occ_mokit.T
    )
    if difference > orbital_tol:
        raise FchError(
            f"the .mat orbitals and MOKIT's transfer of the .fch orbitals differ by "
            f"{difference:.3e} after allowing for sign (tolerance {orbital_tol:.0e}). "
            f"The .fch holds different orbitals from the .mat -- a different job, "
            f"or orbitals that were rotated or reordered after the .mat was written."
        )

    return AOConversion(
        fch_file=fch,
        mol=mol,
        transform=transform,
        kind=describe_transform(transform),
        checks=checks,
    )


def with_rebuilt_fock_from_fch(bundle, fch, **tolerances):
    """``bundle`` with ``F^alpha``/``F^beta`` rebuilt through PySCF from its ``.fch``.

    Returns the new bundle and the :class:`AOConversion` that justified it.
    """
    from .hamiltonian import with_rebuilt_fock

    conversion = gaussian_to_pyscf(bundle, fch, **tolerances)
    rebuilt = with_rebuilt_fock(
        bundle,
        conversion.mol,
        ao_transform=conversion.transform,
        conversion=conversion.provenance(),
    )
    return rebuilt, conversion


__all__ = [
    "AO_HCORE_RTOL",
    "AO_OVERLAP_TOL",
    "AOConversion",
    "FchError",
    "ORBITAL_TOL",
    "ao_transform_from_fch",
    "describe_transform",
    "fch_mo_coeff",
    "gaussian_to_pyscf",
    "mokit_mo_coeff",
    "mokit_version",
    "mol_from_fch",
    "read_fch_array",
    "read_fch_scalar",
    "require_mokit",
    "with_rebuilt_fock_from_fch",
]
