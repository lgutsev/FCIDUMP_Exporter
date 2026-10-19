"""Read a Gaussian 16 matrix-element file into a :class:`~g16dump.bundle.Bundle`.

The ``.mat`` is read with numpy alone (:class:`MatFile`); gauopen is not needed
anywhere. Everything downstream reads the ``.npz`` this module writes.

The format and the labels in :data:`LABELS` were established on ten real G16
C.01 files (the M0 probe set, LONI job 1072685) and are tested against them in
``tests/test_real_mat.py``. A missing quantity still raises an error that
prints every label the file did carry.

Three things are resolved numerically rather than assumed:

* the MO coefficient convention, by testing ``C.T S C = I`` against both
  orientations (see :func:`orient_mo_coeff`);
* the active window, by cross-checking the frozen-core partition the file
  reports against the dimension of the two-electron block it actually contains;
* that the two-electron integrals are over the stored orbitals at all, against
  the ``TRANS MO COEFFICIENTS`` the transformation used. For ROHF they are not
  (see :func:`_check_transformed_orbitals`).

The reference type is never inferred. ``RHF``, ``ROHF``, ``RKS`` and ``ROKS``
leave overlapping fingerprints in a ``.mat``, and guessing wrong silently
produces a plausible, wrong Hamiltonian, so the caller states it.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from .bundle import Bundle, KS_REFERENCES, REFERENCES, make_provenance

#: Candidate spellings for each quantity, matched case-insensitively as whole
#: labels first and as substrings second. The first spelling of each is the one
#: G16 C.01 writes; the rest are kept as harmless fallbacks. Order is
#: preference order.
LABELS = {
    "overlap": ["OVERLAP"],
    "hcore": ["CORE HAMILTONIAN ALPHA", "CORE HAMILTONIAN"],
    "mo_coeff_alpha": ["ALPHA MO COEFFICIENTS", "ALPHA ORBITAL COEFFICIENTS"],
    "mo_coeff_beta": ["BETA MO COEFFICIENTS", "BETA ORBITAL COEFFICIENTS"],
    "fock_alpha": ["ALPHA FOCK MATRIX", "ALPHA FOCK"],
    "fock_beta": ["BETA FOCK MATRIX", "BETA FOCK"],
    "orbital_energies": ["ALPHA ORBITAL ENERGIES"],
    "orbital_energies_beta": ["BETA ORBITAL ENERGIES"],
    "eri_active": ["AA MO 2E INTEGRALS", "ALPHA-ALPHA MO 2E INTEGRALS"],
}

#: Scalar names to try, in order, for each energy; see :data:`SCALAR_SLOTS`.
#: Only a missing nuclear repulsion is an error.
SCALARS = {
    "enuc": ["ENUCREP"],
    "escf": ["ESCF", "ETOTAL"],
}

#: The blocks ``extract`` cannot do without. The rest are optional: a missing
#: Fock matrix means the rebuilt-Fock path, not a failed extraction.
_REQUIRED_QUANTITIES = ("overlap", "hcore", "mo_coeff_alpha", "eri_active")

#: ``C.T S C - I`` must come in under this for an orientation to be accepted.
ORIENTATION_TOL = 1e-6

#: ``TRANS MO COEFFICIENTS`` must match the active MO coefficients to this.
#: Where they are the same orbitals the file stores identical numbers.
TRANS_TOL = 1e-8


class MatFileError(ValueError):
    """A ``.mat`` file is missing something, or does not mean what we assumed."""


# ------------------------------------------------------------ label lookup


def find_label(matlist_keys: Sequence[str], candidates: Sequence[str]) -> str | None:
    """The first candidate present in ``matlist_keys``, or ``None``.

    Exact matches (case- and whitespace-insensitive) win over substring
    matches, so a file carrying both ``ALPHA FOCK MATRIX`` and
    ``ALPHA FOCK MATRIX (SAVED)`` resolves to the plain one.
    """
    normalised = {" ".join(key.upper().split()): key for key in matlist_keys}
    for candidate in candidates:
        wanted = " ".join(candidate.upper().split())
        if wanted in normalised:
            return normalised[wanted]
    for candidate in candidates:
        wanted = " ".join(candidate.upper().split())
        for norm, original in normalised.items():
            if wanted in norm:
                return original
    return None


def _require_label(matlist_keys, quantity: str, path) -> str:
    label = find_label(matlist_keys, LABELS[quantity])
    if label is None:
        raise MatFileError(
            f"{path}: no matrix-element block for {quantity!r}.\n"
            f"  looked for: {LABELS[quantity]}\n"
            f"  file carries: {sorted(matlist_keys)}\n"
            f"If one of those is the right block under a different name, add its "
            f"spelling to g16dump.matfile.LABELS[{quantity!r}]. Do not work "
            f"around a genuinely absent block: a missing Fock matrix means the "
            f"rebuilt-Fock path, and missing 2e integrals mean the "
            f"transformation did not run."
        )
    return label


def _expand(entry, name: str, path) -> np.ndarray:
    """Whatever the entry holds, as a plain float64 ndarray.

    :class:`MatObject` (like gauopen's objects) unpacks lower triangles and
    two-electron packing through ``expand()``. Objects without it are taken as
    already expanded.
    """
    if hasattr(entry, "expand"):
        try:
            return np.asarray(entry.expand(), dtype=np.float64)
        except Exception as exc:
            raise MatFileError(
                f"{path}: {name}.expand() failed: {type(exc).__name__}: {exc}"
            ) from exc
    try:
        return np.asarray(entry.array, dtype=np.float64)
    except Exception as exc:
        raise MatFileError(
            f"{path}: {name} has neither expand() nor a usable .array"
        ) from exc


def _square(array: np.ndarray, n: int, name: str, path) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.shape == (n, n):
        return array
    if array.size == n * n:
        return array.reshape(n, n)
    if array.size == n * (n + 1) // 2:
        out = np.zeros((n, n))
        rows, cols = np.tril_indices(n)
        out[rows, cols] = array.ravel()
        return out + out.T - np.diag(np.diag(out))
    raise MatFileError(
        f"{path}: {name} has {array.size} elements, which is neither {n}x{n} "
        f"nor a {n}-dimensional packed triangle. The dimension the file reports "
        f"and the block it stores disagree."
    )


# ------------------------------------------------- the coefficient convention


def orient_mo_coeff(
    raw: np.ndarray,
    overlap: np.ndarray,
    nao: int,
    nmo: int,
    tol: float = ORIENTATION_TOL,
) -> np.ndarray:
    """Return the coefficients as ``C[ao, mo]``, decided by ``C.T S C = I``.

    Gaussian's storage order is a documented detail that has changed and that
    differs between quantities; reading it out of the numbers costs one matrix
    product and cannot be wrong in the way an assumption can.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.size != nao * nmo:
        raise MatFileError(
            f"MO coefficients have {raw.size} elements, expected {nao * nmo} "
            f"= nao {nao} x nmo {nmo}"
        )

    if raw.ndim == 2:
        candidates = {"as stored": raw, "transposed": raw.T}
    else:
        candidates = {
            "reshaped (nao, nmo)": raw.reshape(nao, nmo),
            "reshaped (nmo, nao) then transposed": raw.reshape(nmo, nao).T,
        }

    identity = np.eye(nmo)
    scores = {}
    for name, candidate in candidates.items():
        if candidate.shape != (nao, nmo):
            continue
        scores[name] = float(
            np.max(np.abs(candidate.T @ overlap @ candidate - identity))
        )

    if not scores:
        raise MatFileError(
            f"no reshape of the MO coefficients gives shape ({nao}, {nmo})"
        )

    best = min(scores, key=scores.get)
    if scores[best] > tol:
        detail = "\n".join(
            f"    {k}: max |C.T S C - I| = {v:.3e}" for k, v in scores.items()
        )
        raise MatFileError(
            f"MO coefficients are not orthonormal in the supplied AO overlap, "
            f"in either orientation:\n{detail}\n"
            f"  tolerance: {tol:.1e}\n"
            f"Either the overlap and the coefficients come from different AO "
            f"sets (check Int=NoBasisTransform in the route), or the block read "
            f"is not the MO coefficient block."
        )
    return np.ascontiguousarray(candidates[best])


# --------------------------------------------------------------- the reader


def _scalar(me, names: Sequence[str]) -> float | None:
    if not hasattr(me, "scalar"):
        return None
    for name in names:
        try:
            value = me.scalar(name)
        except Exception:
            continue
        if value is not None:
            return float(value)
    return None


def _attr(me, names: Sequence[str], default=None):
    for name in names:
        try:
            value = getattr(me, name)
        except Exception:
            continue
        if value is not None:
            return value
    return default


def read_matel(path) -> MatFile:
    """Open a ``.mat`` with the numpy reader below. Kept as the one seam tests patch."""
    return read_mat(path)


# ------------------------------------------------------------- the format
#
# A Gaussian 16 ``Output=MatrixElement`` file is Fortran sequential
# unformatted: every record is framed by a 4-byte little-endian length before
# and after it. Established on the ten C.01 probe files (LONI job 1072685):
#
# * The first NLab records are the file header, NLab being read from record 0:
#
#     0  "Gaussian matrix elements" (a64), IVers, NLab, Gaussian version (a64)
#     1  title (a64), NAtoms, NBasis, NBsUse, ICharg, Multip, NE, Len12L,
#        Len4L, IOpCl, ICGU
#     2  IAn(NAtoms)        atomic numbers
#     3  IAtTyp(NAtoms)
#     4  AtmChg(NAtoms)     nuclear charges (reals)
#     5  C(3, NAtoms)       coordinates in bohr
#     6  IBfAtm(NBasis), IBfTyp(NBasis)
#     7  AtmWgt(NAtoms)
#     8  NFC, NFV, ITran, IDum9
#     9  the length of record 10
#    10  further integers; the first two are the shell and primitive counts
#
# * Then objects, each a 144-byte header record -- label (a64) and ten
#   integers NI, NR, NTot, LenBuf, N1..N5, ASym -- followed by
#   ceil(NTot / LenBuf) data records. Every data record is padded to its full
#   LenBuf elements; each holds LenBuf*NI integers followed by LenBuf*NR reals.
#   The label "END" closes the file.
#
# * N1..N5 are Fortran dimensions, first index fastest. A negative dimension
#   packs that index with the next one as a lower triangle, row by row:
#   (0,0), (1,0), (1,1), (2,0), ... The patterns that occur are
#
#     (-n, n, [m])          m symmetric (ASym=0) or antisymmetric (ASym=-1)
#                           matrices
#     (-n, -n, -n, n, [m])  two-electron integrals (ij|kl) with 8-fold
#                           symmetry: a triangle of ij-pairs >= kl-pairs
#     (-n, n, -n, n, [m])   two-electron integrals (ij|kl) with only i<->j and
#                           k<->l symmetry: the ij pair index runs fastest
#
# * An object with both NI and NR set is sparse: NTot (index, value) pairs
#   into a dense array of the stated dimensions, 1-based. GAUSSIAN SCALARS is
#   one, 1000 slots long.

#: Text that opens record 0 of every matrix-element file.
FILE_LABEL = "Gaussian matrix elements"

#: The header records this reader knows how to decode. IVers 2 is what G16
#: C.01 writes.
SUPPORTED_VERSIONS = (2,)

#: Slots of GAUSSIAN SCALARS, 1-based as the file stores them, identified by
#: value against the ``.log`` and ``.fch`` of the probe jobs. Slots not listed
#: are carried by :attr:`MatFile.scalars` but not named.
SCALAR_SLOTS = {
    "VIRIAL RATIO": 1,
    "ETOTAL": 32,
    "ENUCREP": 41,
    "ESCF": 43,
    "S**2": 44,
}

_OBJECT_HEADER_BYTES = 144


class MatObject:
    """One labelled object of a ``.mat`` file, decoded but not yet expanded.

    ``array`` is the data exactly as stored (packed triangles stay packed);
    :meth:`expand` unpacks it into the shape the dimensions describe.
    """

    def __init__(self, label, ni, nr, ntot, lenbuf, dims, asym, ints, reals):
        self.label = label
        self.ni, self.nr, self.ntot, self.lenbuf = ni, nr, ntot, lenbuf
        self.dims = tuple(dims)
        self.asym = asym
        self.ints = ints
        self.reals = reals

    def __repr__(self) -> str:
        return (
            f"MatObject({self.label!r}, NI={self.ni}, NR={self.nr}, "
            f"NTot={self.ntot}, dims={self.dims}, ASym={self.asym})"
        )

    @property
    def is_sparse(self) -> bool:
        return self.ni > 0 and self.nr > 0

    @property
    def array(self) -> np.ndarray:
        """The stored values, flat. Reals if the object has any, else integers."""
        if self.is_sparse:
            return self.expand().ravel(order="F")
        return self.reals if self.nr else self.ints

    def _logical_dims(self) -> list[int]:
        dims = list(self.dims)
        while dims and dims[-1] == 0:
            dims.pop()
        return dims

    def expand(self) -> np.ndarray:
        """The object as a dense array, with any packed index pairs unpacked.

        Trailing dimensions of length 1 are dropped, so a single two-electron
        block comes back as ``(n, n, n, n)`` and a single matrix as ``(n, n)``.
        """
        dims = self._logical_dims()
        if self.is_sparse:
            out = _expand_sparse(self, dims)
        else:
            out = _expand_dense(self, dims)
        while out.ndim > 1 and out.shape[-1] == 1:
            out = out[..., 0]
        return out


def _expand_sparse(obj: MatObject, dims: list[int]) -> np.ndarray:
    if obj.ni != 1 or obj.nr != 1 or any(d < 0 for d in dims):
        raise MatFileError(
            f"{obj.label}: sparse object with NI={obj.ni}, NR={obj.nr}, "
            f"dims={obj.dims} is not a layout this reader knows"
        )
    size = int(np.prod(dims))
    index = obj.ints.astype(np.int64)
    if index.size and (index.min() < 1 or index.max() > size):
        raise MatFileError(
            f"{obj.label}: sparse indices run {index.min()}..{index.max()}, "
            f"outside 1..{size}"
        )
    out = np.zeros(size)
    out[index - 1] = obj.reals
    return out.reshape(dims, order="F")


def _tri_pairs(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row/column of each packed lower-triangle element, and the inverse map."""
    rows, cols = np.tril_indices(n)
    pair = np.empty((n, n), dtype=np.int64)
    pair[rows, cols] = np.arange(rows.size)
    pair[cols, rows] = pair[rows, cols]
    return rows, cols, pair


def _expand_dense(obj: MatObject, dims: list[int]) -> np.ndarray:
    data = obj.reals if obj.nr else obj.ints
    if obj.ni > 1 or obj.nr > 1:
        raise MatFileError(
            f"{obj.label}: NI={obj.ni}, NR={obj.nr} per element is not a "
            f"layout this reader knows"
        )
    signs = [d < 0 for d in dims]
    size = [abs(d) for d in dims]

    def _sets(rank: int) -> tuple[list[int], int]:
        rest = size[rank:]
        return rest, int(np.prod(rest)) if rest else 1

    def _check(expected: int, kind: str) -> None:
        if data.size != expected:
            raise MatFileError(
                f"{obj.label}: dims {obj.dims} ({kind}) imply {expected} "
                f"elements, the file holds {data.size}"
            )

    if not any(signs):
        _check(int(np.prod(size)) if size else 0, "dense")
        return data.reshape(size, order="F")

    # One packed matrix, or a stack of them.
    if signs[:2] == [True, False] and not any(signs[2:]) and size[0] == size[1]:
        n = size[0]
        rest, nsets = _sets(2)
        ntri = n * (n + 1) // 2
        _check(ntri * nsets, "packed triangle")
        if obj.asym not in (0, -1):
            raise MatFileError(f"{obj.label}: unknown ASym={obj.asym}")
        rows, cols, _ = _tri_pairs(n)
        packed = data.reshape(ntri, nsets, order="F")
        out = np.zeros((n, n, nsets), dtype=data.dtype)
        # ASym=-1 marks antisymmetric integrals (DIP VEL, R X DEL). Which
        # triangle carries the positive sign is not verified against anything.
        out[cols, rows] = packed if obj.asym == 0 else -packed
        out[rows, cols] = packed
        return out.reshape([n, n] + rest, order="F")

    # Two-electron integrals in chemists' order, (ij|kl) = out[i, j, k, l].
    if len(size) >= 4 and len(set(size[:4])) == 1:
        n = size[0]
        rest, nsets = _sets(4)
        _, _, pair = _tri_pairs(n)
        npair = n * (n + 1) // 2
        flat_pair = pair.ravel()
        if signs[:4] == [True, True, True, False]:
            _check(npair * (npair + 1) // 2 * nsets, "8-fold two-electron")
            prow, pcol, _ = _tri_pairs(npair)
            packed = data.reshape(-1, nsets, order="F")
            square = np.empty((npair, npair, nsets))
            square[prow, pcol] = packed
            square[pcol, prow] = packed
        elif signs[:4] == [True, False, True, False]:
            _check(npair * npair * nsets, "4-fold two-electron")
            square = data.reshape(npair, npair, nsets, order="F")
        else:
            square = None
        if square is not None:
            out = square[flat_pair[:, None], flat_pair[None, :]]
            return out.reshape([n, n, n, n] + rest)

    raise MatFileError(
        f"{obj.label}: dims {obj.dims} use a packing this reader does not know"
    )


class MatFile:
    """A Gaussian ``.mat`` file, read with numpy alone.

    Exposes the header the way gauopen's ``MatEl`` does (``nbasis``,
    ``nbsuse``, ``ne``, ``multip``, ``nfc``, ``nfv``, ``itran``, ``atmchg``,
    ``matlist`` and ``scalar(name)``), so :func:`extract` reads either.
    """

    def __init__(self, path):
        self.path = Path(path)
        records = _records(self.path)
        if not records or not records[0][:64].decode("latin-1").startswith(FILE_LABEL):
            raise MatFileError(
                f"{path}: does not start with {FILE_LABEL!r}; not a Gaussian "
                f"matrix-element file (RawMatrixElement output is text, and is "
                f"not read here)"
            )
        self.ivers, self.nlab = (int(v) for v in np.frombuffer(records[0], "<i8", 2, 64))
        self.gaussian_version = records[0][80:144].decode("latin-1").strip()
        if self.ivers not in SUPPORTED_VERSIONS or self.nlab < 11:
            raise MatFileError(
                f"{path}: matrix-element file version {self.ivers} with "
                f"{self.nlab} header records; this reader knows version "
                f"{SUPPORTED_VERSIONS} with at least 11"
            )

        self.title = records[1][:64].decode("latin-1").strip()
        (
            self.natoms, self.nbasis, self.nbsuse, self.icharg, self.multip,
            self.ne, self.len12l, self.len4l, self.iopcl, self.icgu,
        ) = (int(v) for v in np.frombuffer(records[1], "<i8", 10, 64))
        if (self.len12l, self.len4l) != (8, 8):
            raise MatFileError(
                f"{path}: Len12L={self.len12l}, Len4L={self.len4l}; only 8-byte "
                f"integers are supported"
            )

        def _ints(i):
            return np.frombuffer(records[i], "<i8").copy()

        def _reals(i):
            return np.frombuffer(records[i], "<f8").copy()

        self.ian = _ints(2)
        self.iattyp = _ints(3)
        self.atmchg = _reals(4)
        self.coordinates = _reals(5).reshape(self.natoms, 3)
        bf = _ints(6)
        self.ibfatm, self.ibftyp = bf[: self.nbasis], bf[self.nbasis :]
        self.atmwgt = _reals(7)
        self.nfc, self.nfv, self.itran, self.idum9 = (int(v) for v in _ints(8)[:4])
        self.header_extra = _ints(10)

        self.matlist: dict[str, MatObject] = {}
        cursor = self.nlab
        while cursor < len(records):
            head = records[cursor]
            if len(head) != _OBJECT_HEADER_BYTES:
                raise MatFileError(
                    f"{path}: record {cursor} is {len(head)} bytes where an "
                    f"object header ({_OBJECT_HEADER_BYTES}) was expected"
                )
            label = head[:64].decode("latin-1").strip()
            ni, nr, ntot, lenbuf, *dims, asym = (
                int(v) for v in np.frombuffer(head, "<i8", 10, 64)
            )
            cursor += 1
            if label == "END":
                break
            nrec = -(-ntot // lenbuf) if ntot else 0
            chunk = lenbuf * (ni + nr) * 8
            body = records[cursor : cursor + nrec]
            if len(body) != nrec or any(len(r) != chunk for r in body):
                raise MatFileError(
                    f"{path}: {label} should be followed by {nrec} records of "
                    f"{chunk} bytes; the file has "
                    f"{[len(r) for r in body]}. Truncated or not this format."
                )
            ints, reals = [], []
            for k, rec in enumerate(body):
                count = min(lenbuf, ntot - k * lenbuf)
                words = np.frombuffer(rec, "<i8")
                ints.append(words[: count * ni])
                reals.append(np.frombuffer(rec, "<f8", count * nr, lenbuf * ni * 8))
            self.matlist[label] = MatObject(
                label, ni, nr, ntot, lenbuf, dims, asym,
                np.concatenate(ints) if ni else None,
                np.concatenate(reals) if nr else None,
            )
            cursor += nrec
        else:
            raise MatFileError(f"{path}: no END record; the file is truncated")
        if cursor != len(records):
            raise MatFileError(
                f"{path}: {len(records) - cursor} records after END"
            )

        scalars = self.matlist.get("GAUSSIAN SCALARS")
        self.scalars: dict[int, float] = {}
        if scalars is not None and scalars.is_sparse:
            self.scalars = {
                int(i): float(v) for i, v in zip(scalars.ints, scalars.reals)
            }

    @property
    def nshell(self) -> int:
        return int(self.header_extra[0])

    @property
    def nprimitive(self) -> int:
        return int(self.header_extra[1])

    def scalar(self, name: str) -> float:
        """A named slot of GAUSSIAN SCALARS; ``KeyError`` if unnamed or unset."""
        slot = SCALAR_SLOTS[name.upper()]
        if slot not in self.scalars:
            raise KeyError(f"{self.path}: scalar {name} (slot {slot}) is not set")
        return self.scalars[slot]


def _records(path: Path) -> list[bytes]:
    """Split a Fortran sequential unformatted file into its records."""
    buf = path.read_bytes()
    if path.suffix == ".gz":
        buf = gzip.decompress(buf)
    out, pos = [], 0
    while pos < len(buf):
        if pos + 4 > len(buf):
            raise MatFileError(f"{path}: truncated record marker at byte {pos}")
        (n,) = np.frombuffer(buf, "<i4", 1, pos)
        end = pos + 4 + int(n)
        if n < 0 or end + 4 > len(buf) or np.frombuffer(buf, "<i4", 1, end)[0] != n:
            raise MatFileError(
                f"{path}: broken record marker at byte {pos}. Not a "
                f"little-endian Fortran unformatted file with 4-byte markers, "
                f"or truncated."
            )
        out.append(buf[pos + 4 : end])
        pos = end + 4
    return out


def read_mat(path) -> MatFile:
    """Read a Gaussian ``.mat`` file; numpy only, no gauopen."""
    path = Path(path)
    if not path.is_file():
        raise MatFileError(f"{path}: no such file")
    return MatFile(path)


def _check_transformed_orbitals(
    me, keys, active_coeffs, path, *, refuse: bool = True
) -> float | None:
    """Confirm the two-electron integrals are over the bundle's own orbitals.

    Gaussian writes the orbitals its transformation actually used as
    ``TRANS MO COEFFICIENTS``, one ``(nao, nact)`` slice per spin. For RHF, RKS
    and UHF they are the active columns of the stored MO coefficients. For ROHF
    they are not: Gaussian semicanonicalizes each spin -- diagonalizing F^alpha
    and F^beta within the whole occupied and the whole virtual space, frozen
    orbitals included -- and transforms over those. The resulting active space
    has drifted out of the ROHF one (by about 1e-3 on the probe jobs), so no
    rotation recovers integrals over the ROHF orbitals.

    ``active_coeffs`` holds the active alpha coefficients and, optionally, the
    beta ones; the second slice is compared with the last given. Returns the
    largest deviation found, or ``None`` when the file does not carry the block.
    With ``refuse=False`` a mismatch is returned rather than raised: the caller
    is about to replace the integrals.
    """
    label = find_label(keys, ["TRANS MO COEFFICIENTS"])
    if label is None:
        return None
    trans = np.asarray(_expand(me.matlist[label], label, path), dtype=np.float64)
    if trans.ndim == 2:
        trans = trans[:, :, None]
    active_coeff = active_coeffs[0]
    if trans.shape[:2] != active_coeff.shape:
        raise MatFileError(
            f"{path}: {label} is {trans.shape[0]} x {trans.shape[1]}, but the "
            f"window selects {active_coeff.shape[1]} of the {active_coeff.shape[0]}"
            f"-function MO coefficients. The window and the transformation "
            f"disagree."
        )
    worst = 0.0
    for spin in range(trans.shape[2]):
        used = trans[:, :, spin]
        active_coeff = active_coeffs[min(spin, len(active_coeffs) - 1)]
        signs = np.where(np.sum(used * active_coeff, axis=0) < 0, -1.0, 1.0)
        worst = max(worst, float(np.max(np.abs(used * signs - active_coeff))))
    if worst > TRANS_TOL and refuse:
        raise MatFileError(
            f"{path}: the two-electron integrals are not over the stored "
            f"orbitals. {label} differs from the active MO coefficients by "
            f"{worst:.3e}.\n"
            f"This is what an ROHF job writes: Gaussian semicanonicalizes the "
            f"ROHF orbitals separately for each spin before the windowed "
            f"transformation, so AA, BA and BB are over three different orbital "
            f"sets, none of them the ROHF one, and the frozen core mixes into "
            f"the active space on the way. A spin-restricted FCIDUMP cannot be "
            f"built from them. Build the integrals in PySCF instead: "
            f"extract --integrals pyscf --fch JOB.fch (needs PySCF and MOKIT). "
            f"See gaussian/README.md, 'The ROHF route'."
        )
    return worst


def extract(
    path,
    *,
    reference: str,
    window: tuple | None = None,
    fch: str | None = None,
    route: str | None = None,
    basis: str | None = None,
    method: str | None = None,
    integrals: str = "gaussian",
) -> Bundle:
    """Build a bundle from the Gaussian matrix-element file at ``path``.

    ``reference`` is required and is never guessed; see the module docstring.
    ``window`` is the 1-based ``(NFIRST, NLAST)`` pair from the Gaussian route.
    Omit it to take the frozen-core partition the file reports, which is then
    cross-checked against the dimension of the two-electron block regardless.

    ``integrals="gaussian"`` (the default, numpy only) takes the active
    two-electron integrals from the file and refuses a file whose integrals are
    not over the stored orbitals, which is every ROHF one. ``integrals="pyscf"``
    builds them, and the Fock matrices, with PySCF over the stored orbitals
    instead (:func:`g16dump.fch.with_pyscf_integrals`); it needs ``fch`` and
    PySCF and MOKIT. Everything one-electron still comes from the ``.mat``.
    """
    if integrals not in ("gaussian", "pyscf"):
        raise MatFileError(
            f"integrals={integrals!r}: expected 'gaussian' or 'pyscf'"
        )
    if integrals == "pyscf" and fch is None:
        raise MatFileError(
            "integrals='pyscf' needs the job's .fch: PySCF's molecule and the "
            "Gaussian->PySCF AO map are both read from it (through MOKIT)"
        )
    path = Path(path)
    reference = reference.upper()
    if reference not in REFERENCES:
        raise MatFileError(
            f"reference {reference!r} is not one of {sorted(REFERENCES)}. The "
            f"reference type is not inferred from the file: RHF, ROHF and their "
            f"Kohn-Sham counterparts leave overlapping fingerprints in a .mat, "
            f"and the wrong choice produces a plausible, wrong Hamiltonian."
        )

    me = read_matel(path)
    keys = list(me.matlist.keys())

    nao = int(_attr(me, ["nbasis"], 0))
    nmo = int(_attr(me, ["nbsuse"], nao) or nao)
    nelec = int(_attr(me, ["ne"], 0))
    multiplicity = int(_attr(me, ["multip", "multiplicity"], 1))
    if nao < 1 or nelec < 1:
        raise MatFileError(
            f"{path}: the header reports nbasis={nao}, ne={nelec}. The file is "
            f"truncated or is not a matrix-element file."
        )

    nunpaired = multiplicity - 1
    if (nelec - nunpaired) % 2 or nunpaired > nelec:
        raise MatFileError(
            f"{path}: {nelec} electrons cannot carry multiplicity "
            f"{multiplicity}."
        )
    nalpha = (nelec + nunpaired) // 2
    nbeta = nelec - nalpha

    # UHF is out of scope: a spin-restricted FCIDUMP cannot represent it, and a
    # beta coefficient block that differs from alpha is exactly that case.
    beta_label = find_label(keys, LABELS["mo_coeff_beta"])
    overlap = _square(
        _expand(me.matlist[_require_label(keys, "overlap", path)], "overlap", path),
        nao, "overlap", path,
    )
    hcore = _square(
        _expand(me.matlist[_require_label(keys, "hcore", path)], "hcore", path),
        nao, "core Hamiltonian", path,
    )

    mo_label = _require_label(keys, "mo_coeff_alpha", path)
    mo_coeff = orient_mo_coeff(
        _expand(me.matlist[mo_label], mo_label, path), overlap, nao, nmo
    )
    if beta_label is not None:
        beta_coeff = orient_mo_coeff(
            _expand(me.matlist[beta_label], beta_label, path), overlap, nao, nmo
        )
        if not np.allclose(beta_coeff, mo_coeff, atol=1e-8):
            raise MatFileError(
                f"{path}: the beta MO coefficients differ from the alpha ones, "
                f"so this is a UHF reference. A spin-restricted FCIDUMP cannot "
                f"represent it. Generate UHF natural orbitals in Gaussian and "
                f"send those through instead."
            )

    # ------------------------------------------------------ active window
    eri_label = _require_label(keys, "eri_active", path)
    eri_flat = _expand(me.matlist[eri_label], eri_label, path)
    nact_from_eri = int(round(eri_flat.size ** 0.25))
    if nact_from_eri**4 != eri_flat.size:
        raise MatFileError(
            f"{path}: {eri_label} holds {eri_flat.size} elements, which is not "
            f"n^4 for any integer n. The two-electron block is packed in a way "
            f"this reader does not know; run scripts/inspect_mat.py and read its "
            f"packing candidates."
        )
    eri_active = np.ascontiguousarray(eri_flat.reshape((nact_from_eri,) * 4))

    # The window is stored exactly as the Gaussian route stated it: 1-based and
    # inclusive, so that a bundle records the job that was actually asked for.
    if window is not None:
        active_first, active_last = int(window[0]), int(window[1])
    else:
        nfc = int(_attr(me, ["nfc"], 0) or 0)
        nfv = int(_attr(me, ["nfv"], 0) or 0)
        active_first, active_last = nfc + 1, nmo - nfv

    nact = active_last - active_first + 1
    if nact != nact_from_eri:
        source = "the --window given" if window else "the file's nfc/nfv"
        raise MatFileError(
            f"{path}: active ERI dimension inconsistent with nact. {source} "
            f"implies {nact} active orbitals (MOs {active_first}-{active_last} "
            f"of {nmo}), but {eri_label} holds {nact_from_eri}^4 integrals, i.e. "
            f"{nact_from_eri} orbitals. The Window= keyword did not apply the "
            f"way the route intended. Do not work around this; fix the route."
        )
    trans_deviation = _check_transformed_orbitals(
        me, keys, [mo_coeff[:, active_first - 1 : active_last]], path,
        refuse=integrals == "gaussian",
    )

    # --------------------------------------------------------- Fock matrices
    fock_alpha = fock_beta = None
    fock_source = "none"
    if reference in KS_REFERENCES:
        # Deliberate: the stored KS matrix carries exchange-correlation and is
        # not the HF Fock operator. Storing it here would make it reachable by
        # accident, so a KS bundle carries none at all.
        ks_note = (
            "stored KS matrix not read: the active-space Hamiltonian is the HF "
            "Hamiltonian in the KS orbitals, so the Fock matrices must be "
            "rebuilt (see g16dump.hamiltonian.with_rebuilt_fock)"
        )
    else:
        ks_note = None
        alpha_label = find_label(keys, LABELS["fock_alpha"])
        if alpha_label is not None:
            fock_alpha = _square(
                _expand(me.matlist[alpha_label], alpha_label, path),
                nao, alpha_label, path,
            )
            fock_source = "gaussian"
            beta_fock_label = find_label(keys, LABELS["fock_beta"])
            if beta_fock_label is not None:
                fock_beta = _square(
                    _expand(me.matlist[beta_fock_label], beta_fock_label, path),
                    nao, beta_fock_label, path,
                )

    # ------------------------------------------------- diagnostics and scalars
    def _orbital_energies(quantity):
        label = find_label(keys, LABELS[quantity])
        if label is None:
            return None
        values = np.asarray(
            _expand(me.matlist[label], label, path), dtype=np.float64
        ).ravel()
        return values[:nmo] if values.size >= nmo else None

    enuc = _scalar(me, SCALARS["enuc"])
    if enuc is None:
        raise MatFileError(
            f"{path}: no nuclear repulsion energy (tried {SCALARS['enuc']}). "
            f"Run scripts/inspect_mat.py to see which scalar names this file "
            f"carries."
        )

    atom_charges = _attr(me, ["atmchg", "atomic_numbers", "ian"])
    if atom_charges is not None:
        atom_charges = np.asarray(atom_charges, dtype=np.float64).ravel()
    charge = int(_attr(me, ["icharg", "charge"], 0) or 0)
    if atom_charges is not None and atom_charges.size:
        charge = int(round(float(np.sum(atom_charges)))) - nelec

    provenance = make_provenance(
        source_file=str(path),
        fch_file=str(fch) if fch else None,
        route=route,
        basis=basis,
        method=method,
        window_1based=(active_first, active_last),
        fock_source=fock_source,
        extra={
            key: value
            for key, value in (
                ("ks_note", ks_note),
                ("itran", _attr(me, ["itran"])),
                ("trans_mo_coeff_deviation", trans_deviation),
                ("eri_source", "gaussian"),
            )
            if value is not None
        } or None,
    )

    bundle = Bundle(
        reference_type=reference,
        charge=charge,
        multiplicity=multiplicity,
        nelec=nelec,
        nalpha=nalpha,
        nbeta=nbeta,
        nao=nao,
        nmo=nmo,
        ncore=active_first - 1,
        nact=nact,
        active_first=active_first,
        active_last=active_last,
        enuc=enuc,
        C=mo_coeff,
        S=overlap,
        Hcore_ao=hcore,
        eri_active=eri_active,
        source_program="gaussian16",
        source_file=str(path),
        fock_source=fock_source,
        F_alpha_ao=fock_alpha,
        F_beta_ao=fock_beta,
        orbital_energies=_orbital_energies("orbital_energies"),
        orbital_energies_beta=_orbital_energies("orbital_energies_beta"),
        atom_charges=(
            atom_charges if atom_charges is not None and atom_charges.size else None
        ),
        escf=_scalar(me, SCALARS["escf"]),
        provenance=provenance,
    )
    if integrals == "gaussian":
        return bundle
    # Gaussian's block was read only to confirm the window; it is replaced
    # before the bundle leaves this function, so it is never used.
    from .fch import with_pyscf_integrals

    rebuilt, _ = with_pyscf_integrals(bundle, fch)
    return rebuilt


# ----------------------------------------------------------------- survey


def survey(path) -> dict:
    """Report whether ``extract`` can read this ``.mat``, and if not, why.

    Answers one question: are the blocks g16dump needs present, under what
    names, and are their dimensions consistent with each other? That is the M0
    gate in miniature, and it is what to run before an extraction.

    ``objects`` lists every object in the file with its stored layout, so a
    label this module does not know about is still visible.
    """
    path = Path(path)
    me = read_matel(path)
    keys = list(me.matlist.keys())

    nao = int(_attr(me, ["nbasis"], 0) or 0)
    nmo = int(_attr(me, ["nbsuse"], nao) or nao)
    nfc = int(_attr(me, ["nfc"], 0) or 0)
    nfv = int(_attr(me, ["nfv"], 0) or 0)

    report = {
        "file": str(path),
        "header": {
            "nao": nao,
            "nmo": nmo,
            "nelec": int(_attr(me, ["ne"], 0) or 0),
            "multiplicity": int(_attr(me, ["multip", "multiplicity"], 0) or 0),
            "nfc": nfc,
            "nfv": nfv,
            "window_1based": [nfc + 1, nmo - nfv],
            "itran": _attr(me, ["itran"]),
        },
        "blocks": {},
        "scalars": {},
        "matlist_keys": sorted(keys),
        "objects": {
            key: {"NI": obj.ni, "NR": obj.nr, "NTot": obj.ntot, "dims": list(obj.dims)}
            for key, obj in me.matlist.items()
            if isinstance(obj, MatObject)
        },
        "notes": [],
    }

    for quantity, candidates in LABELS.items():
        label = find_label(keys, candidates)
        entry = {"label": label, "required": quantity in _REQUIRED_QUANTITIES}
        if label is not None:
            try:
                entry["elements"] = int(_expand(me.matlist[label], label, path).size)
            except Exception:
                entry["elements"] = None
        report["blocks"][quantity] = entry

    for name, candidates in SCALARS.items():
        report["scalars"][name] = _scalar(me, candidates)

    missing = [q for q in _REQUIRED_QUANTITIES if report["blocks"][q]["label"] is None]
    if missing:
        report["notes"].append(
            f"extract cannot run: no block found for {', '.join(missing)}. If the "
            f"file carries them under other names, add those spellings to "
            f"g16dump.matfile.LABELS; the report's 'objects' lists them all."
        )
    itran = report["header"]["itran"]
    if itran is not None and itran != 5:
        report["notes"].append(
            f"ITran={itran}, not 5: the full transformation did not run "
            f"(0 means no MO integrals at all, 4 a partial set). Check "
            f"Output=(MatrixElement,MO2ElectronIntegrals) and Tran=(Full,Force)."
        )
    if report["scalars"].get("enuc") is None:
        report["notes"].append(
            "extract cannot run: no nuclear repulsion scalar found."
        )
    if report["blocks"]["fock_alpha"]["label"] is None:
        report["notes"].append(
            "no Fock matrix present, so the rebuilt-Fock path will be required. "
            "That is the normal situation for Kohn-Sham orbitals."
        )
    elif report["blocks"]["fock_beta"]["label"] is None:
        report["notes"].append(
            "an alpha Fock matrix is present but no beta one. That is fine for a "
            "closed shell and is missing information for an open shell."
        )
    if report["blocks"]["mo_coeff_beta"]["label"] is not None:
        report["notes"].append(
            "beta MO coefficients are present (ROHF writes them too, equal to "
            "the alpha ones). If they differ this is a UHF reference, which v1 "
            "cannot represent."
        )
    mo_label = report["blocks"]["mo_coeff_alpha"]["label"]
    if mo_label is not None and report["blocks"]["overlap"]["label"] is not None:
        try:
            overlap = _square(
                _expand(me.matlist[report["blocks"]["overlap"]["label"]], "overlap", path),
                nao, "overlap", path,
            )
            coeffs = [
                orient_mo_coeff(_expand(me.matlist[label], label, path), overlap, nao, nmo)
                for label in (mo_label, report["blocks"]["mo_coeff_beta"]["label"])
                if label is not None
            ]
            if len(coeffs) == 2 and not np.allclose(coeffs[0], coeffs[1], atol=1e-8):
                report["notes"].append(
                    "extract cannot run: the beta MO coefficients differ from the "
                    "alpha ones, so this is a UHF reference."
                )
            report["trans_mo_coeff_deviation"] = _check_transformed_orbitals(
                me, keys, [c[:, nfc : nmo - nfv] for c in coeffs], path
            )
        except MatFileError as exc:
            report["notes"].append(f"extract cannot run: {exc}")

    eri = report["blocks"]["eri_active"]
    if eri["label"] is not None and eri.get("elements"):
        nact_from_eri = int(round(eri["elements"] ** 0.25))
        if nact_from_eri**4 == eri["elements"]:
            report["eri_nact"] = nact_from_eri
            from_header = nmo - nfv - nfc
            if nact_from_eri != from_header:
                report["notes"].append(
                    f"active ERI dimension inconsistent with the header: the "
                    f"integrals hold {nact_from_eri} orbitals but nfc/nfv imply "
                    f"{from_header}. Window= did not apply the way the route "
                    f"intended; pass --window explicitly only if you know which "
                    f"is right."
                )
        else:
            report["notes"].append(
                f"the two-electron block holds {eri['elements']} elements, which "
                f"is not n^4 for any integer n; its packing is not one this "
                f"reader knows."
            )

    if not report["notes"]:
        report["notes"].append("every block extract needs is present and consistent.")
    return report


__all__ = [
    "LABELS",
    "MatFile",
    "MatObject",
    "SCALARS",
    "SCALAR_SLOTS",
    "survey",
    "MatFileError",
    "extract",
    "find_label",
    "orient_mo_coeff",
    "read_mat",
    "read_matel",
]
