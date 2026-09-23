"""Gaussian matrix-element file -> ``.npz`` bundle. The only gauopen consumer.

This is the one module that imports ``QCMatEl``, and it is deliberately the only
one: gauopen needs a compiled component and usually lives only on the cluster.
Run ``g16dump extract`` there, commit or copy the ``.npz``, and every other part
of the package -- and the whole test suite -- works from that.

Status
------
The matrix-element labels used here come from the legacy scripts, which is
evidence that they exist, not that they are the only spelling. The **Fock matrix
labels are unconfirmed**: no legacy script reads a Fock matrix, so this module
searches for them rather than assuming a name, and says exactly what it found
when the search fails. Run ``scripts/inspect_mat.py`` on a real ``.mat`` and
feed the result back before trusting any of the label constants below.

Nothing here guesses. Array layouts are established by measurement (does
``C S C.T = I`` hold?), sizes are asserted explicitly, and ``reshape`` is used
throughout -- never ``ndarray.resize``, which silently pads or truncates.
"""

from __future__ import annotations

import datetime
import subprocess
from typing import Optional

import numpy as np

from .bundle import Bundle, SCHEMA_VERSION, check_orthonormal
from .errors import (
    MissingDependencyError,
    ReferenceTypeError,
    ValidationError,
    require,
)

#: Labels the legacy scripts demonstrably read.
LABEL_CORE_HAMILTONIAN = "CORE HAMILTONIAN ALPHA"
LABEL_KINETIC = "KINETIC ENERGY"
LABEL_ALPHA_MO = "ALPHA MO COEFFICIENTS"
LABEL_BETA_MO = "BETA MO COEFFICIENTS"
LABEL_ALPHA_ENERGIES = "ALPHA ORBITAL ENERGIES"
LABEL_BETA_ENERGIES = "BETA ORBITAL ENERGIES"
LABEL_ERI_AA = "AA MO 2E INTEGRALS"
LABEL_ERI_BA = "BA MO 2E INTEGRALS"
LABEL_ERI_BB = "BB MO 2E INTEGRALS"

#: Candidate overlap labels, searched in order.
OVERLAP_CANDIDATES = ("OVERLAP", "OVERLAP MATRIX", "AO OVERLAP")

#: Substrings identifying a Fock matrix label for each spin. UNCONFIRMED --
#: these are search patterns, not assertions that such a label exists.
FOCK_ALPHA_PATTERNS = ("ALPHA FOCK MATRIX", "FOCK ALPHA", "ALPHA FOCK")
FOCK_BETA_PATTERNS = ("BETA FOCK MATRIX", "FOCK BETA", "BETA FOCK")

#: Scalar names tried for the SCF energy, in order.
SCF_ENERGY_CANDIDATES = ("ESCF", "SCF ENERGY", "ETOTAL", "TOTAL ENERGY")


def _import_qcmatel():
    try:
        import QCMatEl
    except ImportError as exc:
        raise MissingDependencyError(
            "reading a .mat file needs gauopen, which provides QCMatEl and is "
            "not on PYTHONPATH. gauopen is not on PyPI and is not vendored "
            "here; it ships with Gaussian. Build its compiled component and "
            "set PYTHONPATH=/path/to/gauopen. Only `g16dump extract` needs it "
            "-- everything downstream works from the .npz bundle."
        ) from exc
    return QCMatEl


def _find_label(matlist, patterns) -> Optional[str]:
    """First label containing any pattern, case-insensitively."""
    keys = list(matlist.keys())
    for pattern in patterns:
        for key in keys:
            if pattern.upper() in key.upper():
                return key
    return None


def _unpack_square(raw: np.ndarray, n: int, label: str) -> np.ndarray:
    """Return an ``(n, n)`` symmetric matrix from whatever layout ``raw`` is in.

    Handles a 2-D array, a flat ``n*n`` array, and a flat lower-triangular
    ``n(n+1)/2`` packing. The layout is decided by *size*, not by assumption,
    and an unrecognised size is an error rather than a reshape that happens to
    fit.
    """
    raw = np.asarray(raw)
    if raw.ndim == 2:
        require(
            raw.shape == (n, n),
            f"{label} has shape {raw.shape}, expected ({n}, {n}).",
        )
        return np.ascontiguousarray(raw, dtype=float)

    flat = raw.ravel()
    if flat.size == n * n:
        return np.ascontiguousarray(flat.reshape(n, n), dtype=float)

    if flat.size == n * (n + 1) // 2:
        out = np.zeros((n, n))
        rows, cols = np.tril_indices(n)
        out[rows, cols] = flat
        out[cols, rows] = flat
        return out

    raise ValidationError(
        f"{label} has {flat.size} elements, which matches neither a square "
        f"{n}x{n} matrix ({n * n}) nor a lower-triangular packing "
        f"({n * (n + 1) // 2}). Run scripts/inspect_mat.py on this file and "
        f"check what layout it actually uses before going further."
    )


def _unpack_eri(raw: np.ndarray, nact: int, label: str) -> np.ndarray:
    """Return ``(nact,)*4`` chemist's integrals from whatever layout ``raw`` is in."""
    raw = np.asarray(raw)
    if raw.ndim == 4:
        require(
            raw.shape == (nact,) * 4,
            f"{label} has shape {raw.shape}, expected {(nact,) * 4}.",
        )
        return np.ascontiguousarray(raw, dtype=float)

    flat = raw.ravel()
    if flat.size == nact**4:
        return np.ascontiguousarray(flat.reshape((nact,) * 4), dtype=float)

    npair = nact * (nact + 1) // 2
    if flat.size == npair * (npair + 1) // 2:
        out = np.zeros((nact,) * 4)
        rows, cols = np.tril_indices(nact)
        idx = 0
        for p in range(npair):
            i, j = rows[p], cols[p]
            for q in range(p + 1):
                k, l = rows[q], cols[q]
                value = flat[idx]
                idx += 1
                for a, b, c, d in (
                    (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                    (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i),
                ):
                    out[a, b, c, d] = value
        return out

    raise ValidationError(
        f"{label} has {flat.size} elements, which matches neither a full "
        f"{nact}^4 = {nact ** 4} tensor nor an 8-fold packing "
        f"({npair * (npair + 1) // 2}). The active window may not be the size "
        f"we think it is: run scripts/inspect_mat.py and compare."
    )


def _orient_coefficients(
    flat: np.ndarray, nmo: int, nbasis: int, s_ao: np.ndarray, label: str
) -> np.ndarray:
    """Reshape MO coefficients and establish the row/column convention by test.

    Tries ``(nmo, nbasis)`` and ``(nbasis, nmo)`` and keeps whichever satisfies
    ``C S C.T = I``. Reports both residuals on failure. This is measured rather
    than assumed because a transposed ``C`` yields a plausible-looking but
    entirely wrong Hamiltonian.
    """
    flat = np.asarray(flat).ravel()
    require(
        flat.size == nmo * nbasis,
        f"{label} has {flat.size} elements, expected nmo*nbasis = "
        f"{nmo}*{nbasis} = {nmo * nbasis}.",
    )

    mo_major = flat.reshape(nmo, nbasis)
    ao_major = flat.reshape(nbasis, nmo).T

    res_mo = check_orthonormal(mo_major, s_ao, 0.0)
    res_ao = check_orthonormal(ao_major, s_ao, 0.0)

    if res_mo <= 1e-8 and res_mo <= res_ao:
        return np.ascontiguousarray(mo_major)
    if res_ao <= 1e-8:
        return np.ascontiguousarray(ao_major)

    raise ValidationError(
        f"{label} is not orthonormal in either reshape order: "
        f"max|C S C.T - I| is {res_mo:.3e} reshaped as (nmo, nbasis) and "
        f"{res_ao:.3e} reshaped as (nbasis, nmo). Either the overlap does not "
        f"correspond to these coefficients, or the array is stored in a layout "
        f"this reader does not handle. Run scripts/inspect_mat.py."
    )


def _git_commit() -> Optional[str]:
    """The g16dump commit that produced this bundle, if we are in a checkout."""
    try:
        from pathlib import Path

        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def extract(
    matfile: str,
    *,
    ref_type: str,
    fch: Optional[str] = None,
    rebuild_fock: bool = False,
    overlap: Optional[np.ndarray] = None,
) -> Bundle:
    """Read a Gaussian ``.mat`` into a validated :class:`~g16dump.bundle.Bundle`.

    ``ref_type`` is required and never inferred. A matrix-element file does not
    distinguish Hartree-Fock from Kohn-Sham orbitals, and treating KS orbitals
    as HF silently produces a wrong Hamiltonian, so the caller must say.

    ``fch`` supplies the matching formatted checkpoint, needed for the overlap
    when the ``.mat`` does not carry one and for ``rebuild_fock``.
    """
    from .bundle import KS_REFERENCES, REFERENCE_TYPES

    require(
        ref_type in REFERENCE_TYPES,
        f"ref_type must be given explicitly and be one of "
        f"{', '.join(REFERENCE_TYPES)}, got {ref_type!r}. A .mat file does not "
        f"record whether its orbitals are Hartree-Fock or Kohn-Sham, and the "
        f"two need different treatment, so this is never inferred.",
        ReferenceTypeError,
    )
    require(
        ref_type not in ("UHF", "UKS"),
        f"reference type {ref_type} has different alpha and beta orbitals, "
        f"which a spin-restricted FCIDUMP cannot represent. This is out of "
        f"scope for v1. The workaround is to generate UHF natural orbitals "
        f"(UNOs) in Gaussian and pass those through as a restricted reference.",
        ReferenceTypeError,
    )

    qcmatel = _import_qcmatel()
    me = qcmatel.MatEl(file=matfile)
    matlist = me.matlist

    nbasis = int(me.nbasis)
    nmo = int(me.nbsuse)
    ncore = int(me.nfc)
    nfv = int(me.nfv)
    nact = nmo - ncore - nfv
    nelec = int(me.ne)
    multiplicity = int(me.multip)

    require(
        nact > 0,
        f"the active window is empty: nmo={nmo}, nfc={ncore}, nfv={nfv}. Check "
        f"the Window=() setting in the Gaussian route.",
    )

    nbeta = (nelec - (multiplicity - 1)) // 2
    nalpha = nelec - nbeta
    require(
        nalpha - nbeta == multiplicity - 1 and nalpha + nbeta == nelec,
        f"multiplicity {multiplicity} is impossible for {nelec} electrons: the "
        f"parities do not match.",
    )

    # --- one-electron AO matrices ---------------------------------------
    require(
        LABEL_CORE_HAMILTONIAN in matlist,
        f"{LABEL_CORE_HAMILTONIAN!r} is not in this .mat. Labels present: "
        f"{sorted(matlist.keys())}.",
    )
    h_ao = _unpack_square(
        matlist[LABEL_CORE_HAMILTONIAN].expand(), nbasis, LABEL_CORE_HAMILTONIAN
    )

    # --- the .fch, if given: same basis, different AO order -----------------
    # A molecule rebuilt from the .fch produces integrals in PySCF's AO order,
    # while everything in the .mat is in Gaussian's. They differ for every d,
    # f, g... shell, so any quantity crossing between the two goes through the
    # permutation in aoorder.py -- and before trusting it, the .mat core
    # Hamiltonian is compared against PySCF's, reordered. That comparison is
    # what proves the .fch belongs to this .mat and the mapping is right.
    mol = perm = None
    if fch is not None:
        from .aoorder import check_same_basis, gaussian_to_pyscf_permutation
        from .hamiltonian import load_mol_from_fch

        mol = load_mol_from_fch(fch)
        require(
            int(mol.nao) == nbasis,
            f"{fch} describes {mol.nao} basis functions but {matfile} has "
            f"{nbasis}; they are not from the same Gaussian job.",
        )
        perm = gaussian_to_pyscf_permutation(mol)
        from pyscf.scf import hf as _hf

        h_mismatch = check_same_basis(
            LABEL_CORE_HAMILTONIAN, h_ao, _hf.get_hcore(mol), perm
        )

    # --- overlap ----------------------------------------------------------
    s_ao = overlap
    if s_ao is None:
        overlap_label = _find_label(matlist, OVERLAP_CANDIDATES)
        if overlap_label is not None:
            s_ao = _unpack_square(
                matlist[overlap_label].expand(), nbasis, overlap_label
            )
            if mol is not None:
                check_same_basis(
                    overlap_label, s_ao, mol.intor_symmetric("int1e_ovlp"), perm
                )
        elif mol is not None:
            from .aoorder import matrix_to_gaussian

            s_ao = matrix_to_gaussian(mol.intor_symmetric("int1e_ovlp"), perm)
        else:
            raise ValidationError(
                f"no AO overlap matrix in {matfile} (looked for "
                f"{', '.join(OVERLAP_CANDIDATES)}) and no --fch given. The "
                f"overlap is needed to establish the MO coefficient "
                f"convention, which this reader measures rather than assumes. "
                f"Pass --fch JOB.fch, or add the overlap to the Gaussian "
                f"output."
            )
    s_ao = np.asarray(s_ao, dtype=float)

    # --- MO coefficients ---------------------------------------------------
    require(
        LABEL_ALPHA_MO in matlist,
        f"{LABEL_ALPHA_MO!r} is not in this .mat. Labels present: "
        f"{sorted(matlist.keys())}.",
    )
    c_a = _orient_coefficients(
        matlist[LABEL_ALPHA_MO].array, nmo, nbasis, s_ao, LABEL_ALPHA_MO
    )

    c_b = c_a.copy()
    if LABEL_BETA_MO in matlist:
        c_b_raw = _orient_coefficients(
            matlist[LABEL_BETA_MO].array, nmo, nbasis, s_ao, LABEL_BETA_MO
        )
        difference = float(np.max(np.abs(c_a - c_b_raw)))
        require(
            difference <= 1e-10,
            f"this .mat carries separate beta MO coefficients that differ from "
            f"the alpha set by {difference:.3e}. That is a UHF-like reference, "
            f"which a spin-restricted FCIDUMP cannot represent (out of scope "
            f"for v1). Use UHF natural orbitals instead.",
            ReferenceTypeError,
        )
        c_b = c_b_raw

    # --- active-space two-electron integrals --------------------------------
    require(
        LABEL_ERI_AA in matlist,
        f"{LABEL_ERI_AA!r} is not in this .mat, so there are no windowed MO "
        f"two-electron integrals to dump. The Gaussian job probably did not "
        f"perform the transformation -- check Tran= and Window= in the route. "
        f"Labels present: {sorted(matlist.keys())}.",
    )
    eri_act = _unpack_eri(matlist[LABEL_ERI_AA].expand(), nact, LABEL_ERI_AA)

    # For a restricted reference the three spin blocks must coincide.
    for label in (LABEL_ERI_BA, LABEL_ERI_BB):
        if label in matlist:
            other = _unpack_eri(matlist[label].expand(), nact, label)
            difference = float(np.max(np.abs(eri_act - other)))
            require(
                difference <= 1e-10,
                f"the {LABEL_ERI_AA} and {label} blocks differ by "
                f"{difference:.3e}. For a restricted reference they are the "
                f"same integrals over the same spatial orbitals, so this is a "
                f"UHF-like reference, which is out of scope for v1.",
                ReferenceTypeError,
            )

    # --- Fock matrices -------------------------------------------------------
    fock_source = "none"
    f_a_ao = f_b_ao = None

    if not rebuild_fock:
        alpha_label = _find_label(matlist, FOCK_ALPHA_PATTERNS)
        beta_label = _find_label(matlist, FOCK_BETA_PATTERNS)
        if alpha_label is not None:
            f_a_ao = _unpack_square(
                matlist[alpha_label].expand(), nbasis, alpha_label
            )
            if beta_label is not None:
                f_b_ao = _unpack_square(
                    matlist[beta_label].expand(), nbasis, beta_label
                )
            else:
                # No beta block: only defensible for a closed shell, where the
                # two Fock operators coincide.
                require(
                    nalpha == nbeta,
                    f"found {alpha_label!r} but no beta Fock matrix, and this "
                    f"is an open-shell reference (nalpha={nalpha}, "
                    f"nbeta={nbeta}). The alpha and beta Fock operators differ "
                    f"for an open shell, so the beta block cannot be inferred. "
                    f"Use --rebuild-fock --fch JOB.fch instead.",
                )
                f_b_ao = f_a_ao.copy()
            fock_source = "gaussian"

    if f_a_ao is None:
        if fch is None:
            raise ValidationError(
                f"no usable Fock matrix in {matfile} (searched for "
                f"{', '.join(FOCK_ALPHA_PATTERNS)}) and no --fch given for the "
                f"rebuilt-Fock path.\n"
                f"Labels present: {sorted(matlist.keys())}.\n"
                f"Either the Gaussian job did not save a Fock matrix, or it "
                f"uses a label this reader does not know. Run "
                f"scripts/inspect_mat.py on this file and report what it "
                f"finds; meanwhile `--rebuild-fock --fch JOB.fch` reconstructs "
                f"F^s = Hcore + J[P_a + P_b] - K[P_s] from the orbitals with "
                f"one PySCF Fock build."
            )
        from .aoorder import coefficients_to_pyscf, matrix_to_gaussian, matrix_to_pyscf
        from .hamiltonian import rebuild_fock_with_pyscf

        # Build in PySCF's AO order, with the .mat's own core Hamiltonian, then
        # bring the result back to Gaussian order alongside everything else.
        f_a_pyscf, f_b_pyscf = rebuild_fock_with_pyscf(
            mol,
            coefficients_to_pyscf(c_a, perm),
            coefficients_to_pyscf(c_b, perm),
            nalpha,
            nbeta,
            h_ao=matrix_to_pyscf(h_ao, perm),
        )
        f_a_ao = matrix_to_gaussian(f_a_pyscf, perm)
        f_b_ao = matrix_to_gaussian(f_b_pyscf, perm)
        fock_source = "rebuilt-pyscf"

    if ref_type in KS_REFERENCES and fock_source == "gaussian":
        raise ReferenceTypeError(
            f"reference type {ref_type} is Kohn-Sham, but a Fock-like matrix "
            f"was read from the .mat. The stored KS matrix contains the "
            f"exchange-correlation potential and must never be used as f^s. "
            f"Re-run with --rebuild-fock --fch JOB.fch."
        )

    # --- scalars and diagnostics ---------------------------------------------
    e_nuc = float(me.scalar("ENUCREP"))

    e_scf = None
    for name in SCF_ENERGY_CANDIDATES:
        try:
            e_scf = float(me.scalar(name))
            break
        except Exception:
            continue
    if ref_type in KS_REFERENCES:
        # A KS total energy is not the HF reference determinant energy.
        e_scf = None

    mo_energies_a = None
    if LABEL_ALPHA_ENERGIES in matlist:
        raw = np.asarray(matlist[LABEL_ALPHA_ENERGIES].expand()).ravel()
        if raw.size == nmo:
            mo_energies_a = raw

    # Nuclear charges, if the file exposes them under any name we recognise.
    # Without them the molecular charge cannot be determined, and inventing a
    # value would put a wrong number into the bundle's provenance.
    atom_charges = None
    charge = 0
    for attribute in ("ian", "IAn", "atomic_numbers", "atmchg"):
        raw = getattr(me, attribute, None)
        if raw is None:
            continue
        try:
            candidate = np.asarray(raw, dtype=float).ravel()
        except Exception:
            continue
        if candidate.size and np.all(candidate > 0):
            atom_charges = candidate
            charge = int(round(float(candidate.sum()))) - nelec
            break

    provenance = {
        "matfile": matfile,
        "fch": fch,
        "extracted": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "g16dump_commit": _git_commit(),
        "schema_version": SCHEMA_VERSION,
        "ref_type": ref_type,
        "fock_source": fock_source,
        "active_window_1based": [ncore + 1, ncore + nact],
        "nbasis": nbasis,
        "nmo": nmo,
        "labels_present": sorted(matlist.keys()),
    }
    if atom_charges is None:
        provenance["charge_note"] = (
            "no nuclear-charge array found in the .mat, so the molecular "
            "charge could not be determined and is recorded as 0. This field "
            "is metadata only; nothing in the algebra uses it."
        )
    if mol is not None:
        provenance["fch_core_hamiltonian_relative_mismatch"] = h_mismatch
    if fock_source == "rebuilt-pyscf":
        provenance["fock_rebuilt"] = (
            "F^s = Hcore + J[P_a + P_b] - K[P_s], rebuilt with PySCF from the "
            "Gaussian orbitals, with Hcore taken from the .mat."
        )
        # J and K come from a basis read out of the .fch, whose exponents and
        # contraction coefficients carry 9 significant figures. Measured effect
        # on E_ref: ~1e-9 Ha for first-row molecules, ~1-2e-7 Ha for Ni
        # complexes in def2-SVP. Negligible for SHCI/DMRG, but above the 1e-8
        # default of the E_ref gate -- hence this flag, which the gate reads.
        provenance["fock_rebuilt_from_fch"] = True

    return Bundle(
        ref_type=ref_type,
        fock_source=fock_source,
        charge=charge,
        multiplicity=multiplicity,
        nelec=nelec,
        nalpha=nalpha,
        nbeta=nbeta,
        ncore=ncore,
        nact=nact,
        nfv=nfv,
        nmo=nmo,
        nbasis=nbasis,
        e_nuc=e_nuc,
        e_scf=e_scf,
        h_ao=h_ao,
        s_ao=s_ao,
        c_a=c_a,
        c_b=c_b,
        eri_act=eri_act,
        f_a_ao=f_a_ao,
        f_b_ao=f_b_ao,
        mo_energies_a=mo_energies_a,
        atom_charges=atom_charges,
        provenance=provenance,
    )
