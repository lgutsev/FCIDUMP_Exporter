"""FCIDUMP writer: format, symmetry-unique enumeration, and round-tripping.

The round trip is ``internal Hamiltonian -> FCIDUMP -> independent reader ->
compare tensors``, using a parser written from the format description
(``fcidump_reader.py``) and, where PySCF is installed, PySCF's reader as well.
"""

from __future__ import annotations

import numpy as np
import pytest

from fcidump_reader import determinant_energy, read_fcidump
from g16dump.errors import ValidationError
from g16dump.hamiltonian import active_space_hamiltonian
from g16dump.write import (
    count_unique_eri,
    dice_nocc_block,
    format_header,
    iter_unique_eri,
    write_dice_nocc,
    write_fcidump,
    write_from_hamiltonian,
)
from synthetic import make_system

try:
    from pyscf.tools import fcidump as _pyscf_fcidump

    HAVE_PYSCF = True
except ImportError:  # pragma: no cover
    _pyscf_fcidump = None
    HAVE_PYSCF = False


CASES = [
    ("rhf", dict(nalpha=4, nbeta=4, ncore=1, ref_type="RHF", seed=21)),
    ("rohf_triplet", dict(nalpha=5, nbeta=3, ncore=1, ref_type="ROHF", seed=22)),
    ("rohf_doublet", dict(nalpha=4, nbeta=3, ncore=2, ref_type="ROHF", seed=23)),
]


@pytest.fixture(params=[kw for _, kw in CASES], ids=[n for n, _ in CASES])
def built(request):
    system = make_system(nbasis=8, **request.param)
    ash = active_space_hamiltonian(system.bundle, tol_scf=None)
    return system.bundle, ash


# ------------------------------------------------------- enumeration counts


@pytest.mark.parametrize("norb", [1, 2, 3, 5, 8, 13, 21])
def test_unique_enumeration_is_exactly_the_symmetry_unique_set(norb):
    """Every unique (ij|kl) appears exactly once, and nothing else does."""
    rng = np.random.default_rng(norb)
    eri = rng.normal(size=(norb,) * 4)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)

    seen = set()
    total = 0
    for values, i, j, k, l in iter_unique_eri(eri, chunk_rows=97):
        for a, b, c, d in zip(i.tolist(), j.tolist(), k.tolist(), l.tolist()):
            assert a >= b and c >= d, "indices must be in canonical pair order"
            assert (a * (a + 1) // 2 + b) >= (c * (c + 1) // 2 + d)
            key = frozenset(
                [(a, b, c, d), (b, a, c, d), (a, b, d, c), (c, d, a, b)]
            )
            assert key not in seen, f"duplicate entry {(a, b, c, d)}"
            seen.add(key)
        total += len(values)

    assert total == count_unique_eri(norb)


def test_legacy_file_entry_count_is_reproduced():
    """21 orbitals -> 26796 unique entries, as in legacy/FePorph_1_Window.dat."""
    assert count_unique_eri(21) == 26796


def test_chunking_does_not_change_the_result():
    """Chunk size is a memory knob; it must not affect what is written."""
    rng = np.random.default_rng(7)
    norb = 9
    eri = rng.normal(size=(norb,) * 4)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)

    def collect(chunk):
        rows = []
        for values, i, j, k, l in iter_unique_eri(eri, chunk_rows=chunk):
            rows.extend(zip(values.tolist(), i.tolist(), j.tolist(), k.tolist(), l.tolist()))
        return rows

    assert collect(1) == collect(13) == collect(10**9)


# -------------------------------------------------------------- round trip


def test_round_trip_reproduces_the_tensors(built, tmp_path):
    """internal Hamiltonian -> FCIDUMP -> independent reader -> same tensors."""
    bundle, ash = built
    path = tmp_path / "FCIDUMP"
    write_from_hamiltonian(str(path), ash)

    parsed = read_fcidump(str(path))

    assert parsed.norb == bundle.nact
    assert parsed.nelec == bundle.nelec_act
    assert parsed.ms2 == bundle.ms2
    assert parsed.n_two_electron == count_unique_eri(bundle.nact)
    assert parsed.n_one_electron == bundle.nact * (bundle.nact + 1) // 2

    assert np.max(np.abs(parsed.h1 - ash.h_eff)) < 1e-12
    assert np.max(np.abs(parsed.eri - ash.eri_act)) < 1e-12
    assert abs(parsed.e_core - ash.e_core) < 1e-12


def test_round_trip_reproduces_the_reference_energy(built, tmp_path):
    """The determinant energy read back from the file equals E_ref.

    This is the end-to-end statement that matters: the written file describes
    the same reference determinant the SCF did.
    """
    bundle, ash = built
    path = tmp_path / "FCIDUMP"
    write_from_hamiltonian(str(path), ash)

    parsed = read_fcidump(str(path))
    energy = determinant_energy(
        parsed.h1, parsed.eri, parsed.e_core, ash.nocc_a_act, ash.nocc_b_act
    )
    assert abs(energy - ash.e_ref) < 1e-10


@pytest.mark.skipif(not HAVE_PYSCF, reason="pyscf not installed")
def test_round_trip_through_pyscf_reader(built, tmp_path):
    """A second, fully independent reader must see the same Hamiltonian."""
    bundle, ash = built
    path = tmp_path / "FCIDUMP"
    write_from_hamiltonian(str(path), ash)

    data = _pyscf_fcidump.read(str(path))

    assert int(data["NORB"]) == bundle.nact
    assert int(data["NELEC"]) == bundle.nelec_act
    assert int(data["MS2"]) == bundle.ms2
    assert abs(float(data["ECORE"]) - ash.e_core) < 1e-12
    assert np.max(np.abs(np.asarray(data["H1"]) - ash.h_eff)) < 1e-12

    from pyscf import ao2mo

    eri = ao2mo.restore(1, np.asarray(data["H2"]), bundle.nact)
    assert np.max(np.abs(eri - ash.eri_act)) < 1e-12


# ----------------------------------------------------------------- header


def test_header_fields():
    text = format_header(4, 6, 2, orbsym=[1, 2, 3, 4], isym=3)
    assert "NORB=   4" in text
    assert "NELEC=  6" in text
    assert "MS2= 2" in text
    assert "ORBSYM=1,2,3,4," in text
    assert "ISYM=3" in text
    assert text.rstrip().endswith("&END")


def test_orbsym_length_is_checked():
    with pytest.raises(ValidationError, match="ORBSYM has 3 entries but NORB is 4"):
        format_header(4, 4, 0, orbsym=[1, 1, 1])


def test_custom_orbsym_survives_the_round_trip(built, tmp_path):
    bundle, ash = built
    orbsym = [(i % 4) + 1 for i in range(bundle.nact)]
    path = tmp_path / "FCIDUMP"
    write_from_hamiltonian(str(path), ash, orbsym=orbsym, isym=2)

    parsed = read_fcidump(str(path))
    assert parsed.orbsym == orbsym
    assert parsed.isym == 2


# -------------------------------------------------------------- thresholds


def test_threshold_drops_only_small_entries(built, tmp_path):
    bundle, ash = built
    dense = tmp_path / "dense"
    sparse = tmp_path / "sparse"

    n_dense = write_from_hamiltonian(str(dense), ash)
    n_sparse = write_from_hamiltonian(str(sparse), ash, threshold=1e-3)
    assert n_sparse < n_dense

    parsed = read_fcidump(str(sparse))
    kept = np.abs(ash.eri_act) >= 1e-3
    assert np.max(np.abs(parsed.eri[kept] - ash.eri_act[kept])) < 1e-12
    # Dropped entries read back as zero, which is the point of a threshold.
    assert np.max(np.abs(parsed.eri[~kept])) < 1e-3


def test_default_threshold_writes_every_entry(built, tmp_path):
    bundle, ash = built
    path = tmp_path / "FCIDUMP"
    written = write_from_hamiltonian(str(path), ash)
    expected = (
        count_unique_eri(bundle.nact)
        + bundle.nact * (bundle.nact + 1) // 2
        + 1  # the core-energy line
    )
    assert written == expected


# ------------------------------------------------------- invalid arguments


def test_mismatched_eri_shape_is_refused():
    h = np.eye(4)
    eri = np.zeros((3, 3, 3, 3))
    with pytest.raises(ValidationError, match="must span the same"):
        write_fcidump("/dev/null", h, eri, 0.0, 4, 0)


@pytest.mark.parametrize("nelec, ms2", [(4, 3), (4, 6), (2, -3)])
def test_impossible_spin_and_electron_combinations_are_refused(nelec, ms2):
    h = np.eye(3)
    eri = np.zeros((3, 3, 3, 3))
    with pytest.raises(ValidationError, match="impossible for NELEC"):
        write_fcidump("/dev/null", h, eri, 0.0, nelec, ms2)


@pytest.mark.parametrize("nelec, ms2", [(4, 0), (4, 2), (5, 1), (0, 0)])
def test_possible_spin_and_electron_combinations_are_accepted(nelec, ms2, tmp_path):
    h = np.eye(3)
    eri = np.zeros((3, 3, 3, 3))
    write_fcidump(str(tmp_path / "f"), h, eri, 0.0, nelec, ms2)


# ------------------------------------------------------ Dice reference line


def test_dice_nocc_block_matches_the_legacy_layout():
    """Alpha spin-orbitals even, beta odd -- as in legacy/input.dat."""
    text = dice_nocc_block(8, 8)
    assert text.splitlines()[0] == "nocc 16"
    assert text.splitlines()[1] == "0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15"
    assert text.splitlines()[2] == "end"


def test_dice_nocc_block_open_shell():
    text = dice_nocc_block(5, 3)
    assert text.splitlines()[0] == "nocc 8"
    assert text.splitlines()[1] == "0 2 4 6 8 1 3 5"


def test_write_dice_nocc_is_consistent_with_the_dump(built, tmp_path):
    bundle, ash = built
    path = tmp_path / "input.dat"
    write_dice_nocc(str(path), ash)
    lines = path.read_text().splitlines()
    assert lines[0] == f"nocc {bundle.nelec_act}"
    assert len(lines[1].split()) == bundle.nelec_act
