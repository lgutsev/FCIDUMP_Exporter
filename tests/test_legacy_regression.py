"""The shipped legacy dumps, read back and shown to be wrong.

``legacy/FePorph_1_Window.dat`` and ``legacy/FePorph_3_Window.dat`` are real
FCIDUMP files that the original scripts produced for Fe-porphyrin and that were
fed to Dice. They are committed as evidence, not as a target to reproduce, and
this module is where that evidence is turned into numbers.

The claim the whole package rests on is that the legacy closed-shell algebra is
wrong for an open shell. These two files are the same molecule and the same
CAS(22,21) active space in two spin states, so they can be compared directly,
and the comparison is damning in a specific, quantifiable way:

* The triplet reference lies 0.67 Ha **below** the singlet. The sign alone is
  not evidence: real G16 SCF on a model Fe(II) porphine (LONI job 1115971,
  gaussian/README.md) puts the ROHF triplet 0.057 Ha below the RHF singlet,
  as Hartree-Fock tends to for high spin on a transition metal. The size is:
  twelve times that. The geometry and basis of the legacy jobs are not known,
  so this is suggestive rather than a like-for-like comparison.
* 186 of the 210 one-electron off-diagonal records in each file are exactly
  zero. That is the fingerprint of ``h'`` built from ``diag(orbital energies)``:
  the only off-diagonal content it can have comes from the J/K correction,
  because the Fock off-diagonals were discarded before it started.
* The two files disagree about ``E_core`` by 0.48 Ha, for an active space that
  is supposed to be identical in both.

A note on the reference type: ``legacy/Template.py`` calls ``scf.UHF``, so these
may have been UHF jobs. That ``legacy/FCIDUMP_Write_MOe_3.py`` reads ``AA``,
``BA`` and ``BB MO 2E INTEGRALS`` is no evidence either way: the real G16 probes
show ROHF writes all three blocks too (gaussian/README.md, "M0 outcome"). Either
way the legacy pipeline was building a spin-restricted FCIDUMP from integrals
over more than one orbital set.

Everything here is read with the independent parser in ``test_write.py``, so
these are properties of the files rather than of any code that wrote them. No
Gaussian, no gauopen, no PySCF: the files are in the repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from test_write import _read_fcidump, _reference_energy

LEGACY = Path(__file__).resolve().parent.parent / "legacy"
SINGLET = LEGACY / "FePorph_1_Window.dat"
TRIPLET = LEGACY / "FePorph_3_Window.dat"


@pytest.fixture(scope="module")
def dumps():
    for path in (SINGLET, TRIPLET):
        if not path.exists():
            pytest.skip(f"{path} is missing; legacy/ is reference material")
    return _read_fcidump(SINGLET), _read_fcidump(TRIPLET)


def test_the_two_dumps_describe_the_same_active_space(dumps):
    """Same molecule, same CAS(22,21); only the spin state differs."""
    singlet, triplet = dumps
    assert singlet.norb == triplet.norb == 21
    assert singlet.nelec == triplet.nelec == 22
    assert singlet.ms2 == 0
    assert triplet.ms2 == 2
    assert len(singlet.records) == len(triplet.records)


def test_the_legacy_triplet_sits_far_below_the_legacy_singlet(dumps):
    """The legacy gap, in Hartrees.

    The size is the point, not the sign. Real ROHF and RHF on a model Fe(II)
    porphine put the triplet 0.057 Ha below the singlet; the legacy pair has it
    0.67 Ha (422 kcal/mol) below, twelve times more. Different geometry and
    basis, so suggestive rather than conclusive on its own; the zero
    off-diagonals and the E_core disagreement below are the direct evidence.
    """
    singlet, triplet = dumps
    gap = _reference_energy(triplet) - _reference_energy(singlet)
    assert gap < -0.5, (
        f"the legacy triplet was measured 0.67 Ha below the legacy singlet; "
        f"the gap is now {gap:+.4f} Ha, so the committed files have changed"
    )
    assert gap == pytest.approx(-0.6727, abs=5e-4)


def test_the_legacy_core_energies_disagree(dumps):
    """Same active space, so ``E_core`` should not depend on the spin state."""
    singlet, triplet = dumps
    assert abs(triplet.e_core - singlet.e_core) == pytest.approx(0.48, abs=5e-3)


@pytest.mark.parametrize("which", [0, 1])
def test_most_one_electron_off_diagonals_are_exactly_zero(dumps, which):
    """The fingerprint of ``h'`` built from a diagonal Fock matrix.

    Exactly zero, not small: these are not integrals that happened to cancel,
    they are elements that were never computed. A correct ``h'`` for this system
    has no reason to be this sparse.
    """
    parsed = dumps[which]
    off_diagonal = [r for r in parsed.one_electron if r[1] != r[2]]
    exact_zeros = [r for r in off_diagonal if r[0] == 0.0]
    assert len(off_diagonal) == 210
    assert len(exact_zeros) == 186


def test_a_correct_dump_is_not_this_sparse(fixture_path):
    """The contrast, so the number above means something.

    The same count on a bundle this package produced: an open-shell ``h'`` built
    from the full Fock matrices has essentially no exactly-zero off-diagonals.
    """
    import numpy as np

    from g16dump.bundle import load
    from g16dump.hamiltonian import active_hamiltonian

    h = active_hamiltonian(load(fixture_path("ch2_rohf"))).h_eff
    lower = np.tril_indices(h.shape[0], -1)
    off_diagonal = h[lower]
    fraction_zero = float(np.mean(off_diagonal == 0.0))
    assert fraction_zero < 0.05, (
        f"{fraction_zero:.0%} of this h's off-diagonals are exactly zero, which "
        f"is the legacy fingerprint appearing in code that should not have it"
    )
