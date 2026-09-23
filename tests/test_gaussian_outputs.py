"""The Gaussian -> MOKIT -> PySCF gate on real Gaussian output.

Everything else about the MOKIT bridge is tested on Gaussian-side data generated
from PySCF (``test_mokit_bridge.py``). This module is the part that needs a real
``.mat``/``.fch`` pair from the same Gaussian job, which only a machine with
Gaussian can produce: run ``gaussian/run_probes.sh`` there, unpack the archive
it writes, and point this at the directory::

    G16DUMP_GAUSSIAN_OUTPUTS=probe_results_host_date \\
    PYTHONPATH=/path/to/gauopen pytest -m gaussian tests/test_gaussian_outputs.py -rA

Needs gauopen, MOKIT and PySCF. Without the variable every test here is
skipped, so CI never runs it, and until it has been run on real output the gate
it encodes is **unexecuted**.

For each probe it checks: the ``.mat`` overlap and core Hamiltonian against
PySCF's after MOKIT's transformation, orthonormality of the converted orbitals,
their agreement with MOKIT's transfer of the ``.fch`` orbitals, the alpha/beta
gate on the rebuilt Fock matrices, ``E_ref`` against Gaussian's SCF energy for
Hartree-Fock jobs, and ``h'`` and ``E_core`` against an independent full
transform in the converted orbitals. The measured numbers are printed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.gaussian, pytest.mark.gauopen, pytest.mark.mokit, pytest.mark.pyscf,
]

OUTPUTS = os.environ.get("G16DUMP_GAUSSIAN_OUTPUTS")

#: Probe job -> reference type. Windows come from each file's own header.
PROBES = {
    "probe_h2o_rhf": "RHF",
    "probe_h2o_frozen_virtual": "RHF",
    "probe_ch2_rohf": "ROHF",
    "probe_nh_rohf": "ROHF",
    "probe_h2o_rks": "RKS",
    "probe_ch2_rohf_ccpvdz": "ROHF",
    "probe_h2o_rks_631gs": "RKS",
    "probe_fe_porphine_singlet": "RHF",
    "probe_fe_porphine_triplet": "ROHF",
    "probe_ni_porphine_singlet": "RHF",
    "probe_ni_porphine_triplet": "ROHF",
}

GATE = 1e-8
ORACLE_MAX_NMO = 60


def _pair(name):
    if not OUTPUTS:
        pytest.skip("set G16DUMP_GAUSSIAN_OUTPUTS to a run_probes.sh output directory")
    directory = Path(OUTPUTS)
    mat, fch = directory / f"{name}.mat", directory / f"{name}.fch"
    if not (mat.exists() and fch.exists()):
        pytest.skip(f"{name}: no .mat/.fch pair in {directory}")
    return mat, fch


def _oracle(mol, mo, hcore, e_nuc, start, stop, nalpha, nbeta):
    """Full-transform frozen core in PySCF's basis, from test_oracles.py."""
    from test_oracles import (
        _full_eri,
        _oracle_determinant_energy,
        _oracle_frozen_core,
        _oracle_to_mo,
    )

    hcore_mo = _oracle_to_mo(hcore, mo)
    eri = _full_eri(mol, mo)
    h_eff, e_core = _oracle_frozen_core(hcore_mo, eri, e_nuc, start, start, stop)
    e_ref = _oracle_determinant_energy(hcore_mo, eri, nalpha, nbeta, constant=e_nuc)
    return h_eff, e_core, e_ref


@pytest.mark.parametrize("name", list(PROBES))
def test_real_gaussian_pair_through_mokit(name):
    mat, fch = _pair(name)
    pytest.importorskip("QCMatEl")
    pytest.importorskip("mokit")
    from pyscf import scf

    from g16dump import fch as F
    from g16dump import hamiltonian as H
    from g16dump.matfile import extract

    import mokit
    import pyscf

    bundle = extract(mat, reference=PROBES[name], fch=str(fch))
    conversion = F.gaussian_to_pyscf(bundle, fch)
    rebuilt, _ = F.with_rebuilt_fock_from_fch(bundle, fch)
    result = H.active_hamiltonian(rebuilt)

    mol = conversion.mol
    mo = conversion.mo_coeff_pyscf(bundle.C)
    checks = conversion.checks
    # The full-transform oracle holds nmo**4 doubles; porphine would need ~40 GB.
    oracle = None
    if bundle.nmo <= ORACLE_MAX_NMO:
        oracle = _oracle(
            mol, mo, np.asarray(scf.hf.get_hcore(mol)), bundle.enuc,
            bundle.active_start, bundle.active_stop, bundle.nalpha, bundle.nbeta,
        )

    def _vs_oracle():
        if oracle is None:
            return "skipped (too large for a full transform)"
        return (
            f"max|dh'| {np.max(np.abs(result.h_eff - oracle[0])):.3e}, "
            f"|dE_core| {abs(result.e_core - oracle[1]):.3e}, "
            f"|dE_ref| {abs(result.e_ref - oracle[2]):.3e}"
        )

    print(
        f"\n{name}: {bundle.reference_type}, {bundle.nao} AO, MOKIT "
        f"{mokit.__version__}, PySCF {pyscf.__version__}\n"
        f"  AO map                    {conversion.kind}\n"
        f"  overlap before / after    {checks['overlap_before']} / "
        f"{checks['overlap_after']:.3e}\n"
        f"  Hcore rel before / after  {checks['hcore_before_rel']} / "
        f"{checks['hcore_after_rel']:.3e}\n"
        f"  orthonormality            {checks['orthonormality']:.3e} "
        f"(unconverted {checks['orthonormality_unconverted']})\n"
        f"  .mat vs MOKIT orbitals    {checks['mo_coeff_vs_mokit']:.3e} "
        f"({checks['mo_coeff_sign_flips']} sign flips)\n"
        f"  max|h'a - h'b|            {result.spin_deviation:.3e}\n"
        f"  vs full-transform oracle  {_vs_oracle()}\n"
        f"  E_ref - E_scf             "
        f"{(result.e_ref - bundle.escf) if bundle.escf is not None else 'n/a'}"
    )

    assert result.spin_deviation < GATE
    if oracle is not None:
        assert np.max(np.abs(result.h_eff - oracle[0])) < GATE
        assert abs(result.e_core - oracle[1]) < GATE
        assert abs(result.e_ref - oracle[2]) < GATE
    if not bundle.is_ks and bundle.escf is not None:
        assert abs(result.e_ref - bundle.escf) < 1e-7

    # Where Gaussian's own Fock matrices already pass the gate, the rebuild must
    # reproduce what they give.
    if bundle.has_fock:
        try:
            stored = H.active_hamiltonian(bundle)
        except H.HamiltonianError:
            return
        assert np.max(np.abs(stored.h_eff - result.h_eff)) < GATE
        assert abs(stored.e_core - result.e_core) < GATE
