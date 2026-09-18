"""FCIDUMP writer gates.

The round trip is the gate that matters: write a Hamiltonian, read it back with
a parser that shares no code with the writer, and require the tensors to come
back. ``tests/fcidump_oracle.py`` is that parser, written from the format
description; the PySCF-marked tests repeat the round trip through PySCF's own
reader, which is independent of both.

The second gate is physical rather than structural. The reference determinant's
energy, evaluated from the tensors *in the file*, must reproduce the SCF energy
Gaussian reported. A file can round-trip perfectly and still describe the wrong
Hamiltonian; that check is what notices.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from g16dump.bundle import load
from g16dump.hamiltonian import active_hamiltonian
from g16dump.write import (
    DEFAULT_FLOAT_FORMAT,
    WriteError,
    fcidump_header,
    provenance_path_for,
    write_fcidump,
)

from fcidump_oracle import determinant_energy, read_fcidump, spin_occupations

#: One closed-shell and one open-shell derived Hamiltonian, as item 8 requires.
ROUND_TRIP_FIXTURES = ("h2o_rhf", "ch2_rohf")


@pytest.fixture
def hamiltonians(fixture_path):
    """``hamiltonians(name)`` -> ``(bundle, ActiveHamiltonian)`` for a fixture."""

    def _build(name):
        bundle = load(fixture_path(name))
        return bundle, active_hamiltonian(bundle)

    return _build


def _written(hamiltonian, tmp_path, **kwargs):
    path = write_fcidump(hamiltonian, tmp_path / "FCIDUMP", **kwargs)
    return path, read_fcidump(path)


# --------------------------------------------------------------- round trip


@pytest.mark.parametrize("name", ROUND_TRIP_FIXTURES)
def test_round_trip_reproduces_the_tensors(name, hamiltonians, tmp_path):
    _, hamiltonian = hamiltonians(name)
    _, parsed = _written(hamiltonian, tmp_path)

    assert parsed["NORB"] == hamiltonian.nact
    assert parsed["NELEC"] == hamiltonian.nelec_active
    assert parsed["MS2"] == hamiltonian.ms2
    assert np.allclose(parsed["H1"], hamiltonian.h_eff, atol=1e-12, rtol=0)
    assert np.allclose(parsed["H2"], hamiltonian.eri_active, atol=1e-12, rtol=0)
    assert parsed["ECORE"] == pytest.approx(hamiltonian.e_core, abs=1e-12)


@pytest.mark.parametrize("name", ROUND_TRIP_FIXTURES)
def test_default_precision_round_trips_bit_for_bit(name, hamiltonians, tmp_path):
    """Seventeen significant digits, so nothing is lost on the way to a solver.

    Compared over the symmetry-unique entries, which are the ones the file
    carries. ``h_eff`` and the ERIs are symmetric only to rounding, so requiring
    the reconstructed halves to match bit for bit would be a test of the algebra
    upstream rather than of the writing and reading.
    """
    _, hamiltonian = hamiltonians(name)
    _, parsed = _written(hamiltonian, tmp_path)

    rows, cols = np.tril_indices(hamiltonian.nact)
    assert np.array_equal(
        parsed["H1"][rows, cols], np.asarray(hamiltonian.h_eff)[rows, cols]
    )

    ij, kl = np.tril_indices(rows.size)
    pairs = (rows[:, None], cols[:, None], rows[None, :], cols[None, :])
    assert np.array_equal(
        parsed["H2"][pairs][ij, kl], np.asarray(hamiltonian.eri_active)[pairs][ij, kl]
    )
    assert parsed["ECORE"] == float(hamiltonian.e_core)


@pytest.mark.parametrize("name", ROUND_TRIP_FIXTURES)
def test_reference_energy_from_the_file_reproduces_the_scf_energy(
    name, hamiltonians, tmp_path
):
    """The gate that catches a Hamiltonian that is wrong rather than merely garbled."""
    bundle, hamiltonian = hamiltonians(name)
    _, parsed = _written(hamiltonian, tmp_path)

    nalpha, nbeta = spin_occupations(parsed["NELEC"], parsed["MS2"])
    energy = determinant_energy(
        parsed["H1"], parsed["H2"], parsed["ECORE"], nalpha, nbeta
    )

    assert energy == pytest.approx(hamiltonian.e_ref, abs=1e-9)
    assert energy == pytest.approx(bundle.escf, abs=1e-9)


def test_a_lossy_float_format_is_visible_in_the_round_trip(hamiltonians, tmp_path):
    """Guards the precision claim: a shorter format really does lose the tensors."""
    _, hamiltonian = hamiltonians("h2o_rhf")
    _, exact = _written(hamiltonian, tmp_path)
    _, lossy = _written(hamiltonian, tmp_path, float_format="%.6e")

    rows, cols = np.tril_indices(hamiltonian.nact)
    reference = np.asarray(hamiltonian.h_eff)[rows, cols]
    assert np.array_equal(exact["H1"][rows, cols], reference)
    assert not np.array_equal(lossy["H1"][rows, cols], reference)
    assert np.allclose(lossy["H1"], hamiltonian.h_eff, atol=1e-4)


# ------------------------------------------------------ what is on the page


def test_only_symmetry_unique_entries_are_written(hamiltonians, tmp_path):
    _, hamiltonian = hamiltonians("h2o_rhf")
    path, _ = _written(hamiltonian, tmp_path)

    body = [line.split() for line in path.read_text().splitlines()[4:] if line.split()]
    nact = hamiltonian.nact
    npair = nact * (nact + 1) // 2

    two_electron = [fields for fields in body if fields[3] != "0" or fields[4] != "0"]
    one_electron = [
        fields
        for fields in body
        if fields[3] == "0" and fields[4] == "0" and fields[2] != "0"
    ]
    core = [fields for fields in body if fields[1:] == ["0", "0", "0", "0"]]

    assert len(two_electron) == npair * (npair + 1) // 2
    assert len(one_electron) == npair
    assert len(core) == 1

    for _, t, u, v, w in ((f[0], *map(int, f[1:5])) for f in two_electron):
        assert t >= u and v >= w
        assert t * (t - 1) // 2 + u >= v * (v - 1) // 2 + w
    for _, t, u, _v, _w in ((f[0], *map(int, f[1:5])) for f in one_electron):
        assert t >= u


def test_core_energy_is_the_last_line(hamiltonians, tmp_path):
    _, hamiltonian = hamiltonians("ch2_rohf")
    path, _ = _written(hamiltonian, tmp_path)

    fields = path.read_text().splitlines()[-1].split()
    assert fields[1:] == ["0", "0", "0", "0"]
    assert float(fields[0]) == float(hamiltonian.e_core)


def test_header_defaults_to_c1_and_carries_isym(hamiltonians, tmp_path):
    _, hamiltonian = hamiltonians("ch2_rohf")
    _, parsed = _written(hamiltonian, tmp_path)

    assert parsed["ORBSYM"] == [1] * hamiltonian.nact
    assert parsed["ISYM"] == 1


def test_header_carries_the_symmetry_it_is_given(hamiltonians, tmp_path):
    _, hamiltonian = hamiltonians("h2o_rhf")
    orbsym = [1, 1, 3, 1, 4, 2]
    _, parsed = _written(hamiltonian, tmp_path, orbsym=orbsym, isym=3)

    assert parsed["ORBSYM"] == orbsym
    assert parsed["ISYM"] == 3


def test_orbsym_stays_on_one_line(hamiltonians, tmp_path):
    """Readers cap the namelist at a fixed line count; wrapping ORBSYM breaks them."""
    _, hamiltonian = hamiltonians("ch2_rohf")
    path, _ = _written(hamiltonian, tmp_path)

    header = path.read_text().splitlines()[:4]
    assert sum("ORBSYM" in line for line in header) == 1
    assert header[3].strip() == "&END"


def test_the_header_builder_is_usable_on_its_own():
    header = fcidump_header(3, 4, 0, None, 1).splitlines()
    assert header[0].strip().startswith("&FCI")
    assert "NORB=   3" in header[0] and "NELEC= 4" in header[0] and "MS2=0" in header[0]
    assert header[1].strip() == "ORBSYM=1,1,1,"
    assert header[2].strip() == "ISYM=1,"


# ---------------------------------------------------------------- threshold


def test_threshold_omits_small_two_electron_integrals(hamiltonians, tmp_path):
    _, hamiltonian = hamiltonians("h2o_rhf")
    threshold = 1e-3
    _, full = _written(hamiltonian, tmp_path)
    path, sparse = _written(hamiltonian, tmp_path / "cut", threshold=threshold)

    dropped = np.abs(full["H2"]) < threshold
    assert dropped.any(), "the fixture has nothing small enough to test the threshold"
    assert np.all(sparse["H2"][dropped] == 0.0)
    assert np.array_equal(sparse["H2"][~dropped], full["H2"][~dropped])
    full_path, _ = _written(hamiltonian, tmp_path)
    assert len(path.read_text().splitlines()) < len(
        full_path.read_text().splitlines()
    )


def test_threshold_leaves_the_one_electron_block_alone(hamiltonians, tmp_path):
    """A threshold buys file size in the two-electron block and nowhere else."""
    _, hamiltonian = hamiltonians("ch2_rohf")
    _, parsed = _written(hamiltonian, tmp_path, threshold=1.0)

    assert np.allclose(parsed["H1"], hamiltonian.h_eff, atol=1e-12, rtol=0)
    assert parsed["ECORE"] == float(hamiltonian.e_core)
    assert np.all(np.abs(parsed["H2"][np.abs(parsed["H2"]) > 0]) >= 1.0)


def test_zero_threshold_writes_every_unique_integral(hamiltonians, tmp_path):
    _, hamiltonian = hamiltonians("h2o_rhf")
    sparse = replace(
        hamiltonian, eri_active=np.zeros_like(hamiltonian.eri_active)
    )
    path, _ = _written(sparse, tmp_path)

    nact = sparse.nact
    npair = nact * (nact + 1) // 2
    body = [line for line in path.read_text().splitlines()[4:] if line.strip()]
    assert len(body) == npair * (npair + 1) // 2 + npair + 1


# ---------------------------------------------------------------- the paths


def test_the_only_path_is_the_one_passed_in(hamiltonians, tmp_path):
    _, hamiltonian = hamiltonians("h2o_rhf")
    target = tmp_path / "some" / "nested" / "place" / "ni_pap.FCIDUMP"
    written = write_fcidump(hamiltonian, target)

    assert written == target and target.is_file()
    assert list(tmp_path.rglob("FCIDUMP")) == []


def test_a_bare_relative_filename_works(hamiltonians, tmp_path, monkeypatch):
    _, hamiltonian = hamiltonians("h2o_rhf")
    monkeypatch.chdir(tmp_path)
    assert write_fcidump(hamiltonian, "FCIDUMP").is_file()


# --------------------------------------------------------------- provenance


def test_provenance_lands_beside_the_file_not_inside_it(hamiltonians, tmp_path):
    """FCIDUMP has no comment syntax its readers agree on; see write.py."""
    bundle, hamiltonian = hamiltonians("ch2_rohf")
    path, parsed = _written(hamiltonian, tmp_path, provenance=bundle.provenance)

    sidecar = provenance_path_for(path)
    assert sidecar.is_file()
    record = json.loads(sidecar.read_text())
    assert record["fcidump"]["norb"] == hamiltonian.nact
    assert record["fcidump"]["fock_source"] == hamiltonian.fock_source
    assert record["fcidump"]["e_ref"] == pytest.approx(hamiltonian.e_ref)
    for key, value in bundle.provenance.items():
        assert record[key] == value

    assert path.read_text().lstrip().startswith("&FCI")
    assert parsed["NORB"] == hamiltonian.nact


def test_no_provenance_means_no_sidecar(hamiltonians, tmp_path):
    _, hamiltonian = hamiltonians("h2o_rhf")
    path, _ = _written(hamiltonian, tmp_path)
    assert not provenance_path_for(path).exists()


def test_the_threshold_is_recorded_with_the_file(hamiltonians, tmp_path):
    bundle, hamiltonian = hamiltonians("h2o_rhf")
    path, _ = _written(
        hamiltonian, tmp_path, provenance=bundle.provenance, threshold=1e-9
    )
    record = json.loads(provenance_path_for(path).read_text())
    assert record["fcidump"]["eri_threshold"] == 1e-9


# ------------------------------------------------------------- what it refuses


def test_a_non_symmetric_h_eff_is_refused(hamiltonians, tmp_path):
    """Only the lower triangle is written, so the upper one would vanish quietly."""
    _, hamiltonian = hamiltonians("h2o_rhf")
    h_eff = np.array(hamiltonian.h_eff)
    h_eff[0, 1] += 1e-3

    with pytest.raises(WriteError, match="not symmetric"):
        write_fcidump(replace(hamiltonian, h_eff=h_eff), tmp_path / "FCIDUMP")


def test_an_eri_without_its_permutational_symmetry_is_refused(
    hamiltonians, tmp_path
):
    _, hamiltonian = hamiltonians("h2o_rhf")
    eri = np.array(hamiltonian.eri_active)
    eri[0, 1, 2, 3] += 1e-3

    with pytest.raises(WriteError, match="eri_active violates"):
        write_fcidump(replace(hamiltonian, eri_active=eri), tmp_path / "FCIDUMP")


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"h_eff": np.zeros((3, 3))}, "h_eff has shape"),
        ({"eri_active": np.zeros((3, 3, 3, 3))}, "eri_active has shape"),
        ({"nelec_active": 99}, "exceeds"),
        ({"ms2": 3}, "opposite parity"),
        ({"ms2": 99}, "unpaired electrons"),
        ({"e_core": float("nan")}, "e_core is nan"),
    ],
)
def test_an_impossible_hamiltonian_is_refused(
    changes, message, hamiltonians, tmp_path
):
    _, hamiltonian = hamiltonians("h2o_rhf")
    with pytest.raises(WriteError, match=message):
        write_fcidump(replace(hamiltonian, **changes), tmp_path / "FCIDUMP")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"threshold": -1.0}, "is negative"),
        ({"orbsym": [1, 1, 1]}, "labels for 6 active orbitals"),
        ({"orbsym": [0, 1, 1, 1, 1, 1]}, "must be >= 1"),
        ({"isym": "gerade"}, "must be an integer"),
    ],
)
def test_impossible_options_are_refused(kwargs, message, hamiltonians, tmp_path):
    _, hamiltonian = hamiltonians("h2o_rhf")
    with pytest.raises(WriteError, match=message):
        write_fcidump(hamiltonian, tmp_path / "FCIDUMP", **kwargs)


def test_refusing_says_everything_that_is_wrong_at_once(hamiltonians, tmp_path):
    _, hamiltonian = hamiltonians("h2o_rhf")
    with pytest.raises(WriteError) as failure:
        write_fcidump(
            replace(hamiltonian, ms2=3),
            tmp_path / "FCIDUMP",
            threshold=-1.0,
            orbsym=[1, 1],
        )
    message = str(failure.value)
    assert "opposite parity" in message
    assert "is negative" in message
    assert "labels for 6 active orbitals" in message


def test_nothing_is_written_when_the_hamiltonian_is_refused(hamiltonians, tmp_path):
    _, hamiltonian = hamiltonians("h2o_rhf")
    target = tmp_path / "FCIDUMP"
    with pytest.raises(WriteError):
        write_fcidump(replace(hamiltonian, ms2=3), target)
    assert not target.exists()


# ---------------------------------------------------------------- via PySCF


@pytest.mark.pyscf
@pytest.mark.parametrize("name", ROUND_TRIP_FIXTURES)
def test_pyscf_reads_back_what_was_written(name, hamiltonians, tmp_path):
    """The independence gate: PySCF's parser, not ours, and not the writer's."""
    from pyscf import ao2mo
    from pyscf.tools import fcidump

    _, hamiltonian = hamiltonians(name)
    path = write_fcidump(hamiltonian, tmp_path / "FCIDUMP")
    parsed = fcidump.read(str(path), verbose=False)

    assert int(parsed["NORB"]) == hamiltonian.nact
    assert int(parsed["NELEC"]) == hamiltonian.nelec_active
    assert int(parsed["MS2"]) == hamiltonian.ms2
    assert np.allclose(parsed["H1"], hamiltonian.h_eff, atol=1e-12, rtol=0)
    assert np.allclose(
        ao2mo.restore(1, np.asarray(parsed["H2"]), hamiltonian.nact),
        hamiltonian.eri_active,
        atol=1e-12,
        rtol=0,
    )
    assert float(parsed["ECORE"]) == pytest.approx(hamiltonian.e_core, abs=1e-12)


@pytest.mark.pyscf
@pytest.mark.parametrize("name", ROUND_TRIP_FIXTURES)
def test_pyscf_fci_on_the_file_lies_below_the_scf_energy(
    name, hamiltonians, tmp_path
):
    """A sanity gate on the file as a whole: variational, and correlating."""
    from fcidump_oracle import fci_energy_from_fcidump

    bundle, hamiltonian = hamiltonians(name)
    path = write_fcidump(hamiltonian, tmp_path / "FCIDUMP")
    energy = fci_energy_from_fcidump(path)

    assert energy < bundle.escf
    assert energy == pytest.approx(bundle.escf, abs=1.0)


def test_the_default_float_format_is_what_the_module_claims():
    assert DEFAULT_FLOAT_FORMAT % 1.0 == "1.0000000000000000e+00"
    value = -0.1234567890123456789
    assert float(DEFAULT_FLOAT_FORMAT % value) == value
