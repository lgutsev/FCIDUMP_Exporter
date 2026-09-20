"""The FCIDUMP writer, checked by reading its output back.

The reader in this file is written against the *format*, not against
``write.py``. It shares no code with the writer, re-derives the 8-fold scatter
itself, and recomputes the reference determinant's energy from the file alone.
That is the point: a writer tested with its own inverse proves only that the two
agree with each other.

Four claims are being made, in order of how much they matter:

1. The energy of the reference determinant, evaluated from the file, reproduces
   the SCF energy of the job the bundle came from. That number came out of
   PySCF, through a different route, before any of this code ran. It is worth
   knowing exactly what this does and does not establish: the total is blind to
   the two-electron block, because ``E_core`` is defined as ``E_ref - E_act``
   and absorbs any ERI error exactly. The active energy is therefore checked
   separately, and one test deliberately doubles ``eri_act`` to show the total
   really cannot see it.
2. The file carries each symmetry-unique integral exactly once -- no duplicates,
   no omissions. Checked by canonicalising every record's indices under the
   8-fold group and comparing the multiset against the complete orbit set.
3. The numbers survive the trip. At the default precision the round trip is
   bit-exact, which is a property of ``%.16E`` and not an aspiration.
4. A Hamiltonian that cannot be written correctly is refused rather than
   written wrongly.

One test uses ``pyscf.tools.fcidump`` as a genuinely foreign parser and is
marked ``pyscf``; everything else runs on numpy alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np
import pytest

from g16dump.bundle import load
from g16dump.hamiltonian import active_hamiltonian
from g16dump.write import (
    ROUND_TRIP_PRECISION,
    WriteError,
    dice_occupation_line,
    provenance_path,
    write_fcidump,
)

SOUND = ["h2o_rhf", "ch2_rohf", "ch2_rohf_rotated", "nh_rohf"]


# --------------------------------------------------------- independent reader


class ParsedFcidump:
    """A FCIDUMP as a reader sees it: header values and full, scattered tensors."""

    def __init__(self, norb, nelec, ms2, orbsym, isym, h1, eri, e_core, records):
        self.norb, self.nelec, self.ms2 = norb, nelec, ms2
        self.orbsym, self.isym = orbsym, isym
        self.h1, self.eri, self.e_core = h1, eri, e_core
        self.records = records

    @property
    def two_electron(self):
        return [r for r in self.records if r[3] != 0 or r[4] != 0]

    @property
    def one_electron(self):
        return [r for r in self.records if (r[3], r[4]) == (0, 0) and r[1] != 0]


def _read_fcidump(path) -> ParsedFcidump:
    """Parse a FCIDUMP without using anything from ``g16dump``."""
    header_lines, record_lines = [], []
    in_header = True
    with open(path, encoding="ascii") as handle:
        for line in handle:
            if in_header:
                header_lines.append(line)
                if "&END" in line.upper() or line.strip() == "/":
                    in_header = False
                continue
            if line.strip():
                record_lines.append(line)
    assert not in_header, f"{path} never terminated its namelist"

    head = " ".join(header_lines)

    def scalar(name):
        match = re.search(rf"\b{name}\s*=\s*(-?\d+)", head, re.IGNORECASE)
        assert match, f"{name} missing from the header: {head!r}"
        return int(match.group(1))

    norb, nelec, ms2, isym = (scalar(n) for n in ("NORB", "NELEC", "MS2", "ISYM"))
    orbsym_text = re.search(r"ORBSYM\s*=\s*([0-9,\s]+)", head, re.IGNORECASE)
    orbsym = [int(x) for x in orbsym_text.group(1).replace(",", " ").split()]

    h1 = np.zeros((norb, norb))
    eri = np.zeros((norb,) * 4)
    e_core, records = None, []

    for line in record_lines:
        fields = line.split()
        value = float(fields[0])
        i, j, k, cl = (int(x) for x in fields[1:5])
        records.append((value, i, j, k, cl))
        if k != 0 or cl != 0:
            a, b, c, d = i - 1, j - 1, k - 1, cl - 1
            # The 8-fold orbit of a real-orbital (ab|cd), scattered by hand.
            for p, q, r, s in (
                (a, b, c, d), (b, a, c, d), (a, b, d, c), (b, a, d, c),
                (c, d, a, b), (d, c, a, b), (c, d, b, a), (d, c, b, a),
            ):
                eri[p, q, r, s] = value
        elif i != 0:
            h1[i - 1, j - 1] = h1[j - 1, i - 1] = value
        else:
            assert e_core is None, "two scalar records"
            e_core = value

    assert e_core is not None, "no scalar (0 0 0 0) record"
    return ParsedFcidump(norb, nelec, ms2, orbsym, isym, h1, eri, e_core, records)


def _active_energy(parsed: ParsedFcidump) -> float:
    """The reference determinant's energy within the active space, without ``E_core``.

    Separated from :func:`_reference_energy` because the two prove different
    things. This one is the part that actually depends on the two-electron
    block; the sum including ``E_core`` is not, for the reason in that
    function's docstring.
    """
    nalpha = (parsed.nelec + parsed.ms2) // 2
    nbeta = parsed.nelec - nalpha
    h, eri = parsed.h1, parsed.eri

    energy = float(np.sum(np.diag(h)[:nalpha]) + np.sum(np.diag(h)[:nbeta]))
    for nocc in (nalpha, nbeta):
        if nocc:
            block = eri[:nocc, :nocc, :nocc, :nocc]
            energy += 0.5 * float(
                np.einsum("ttuu->", block) - np.einsum("tuut->", block)
            )
    if nalpha and nbeta:
        energy += float(
            np.einsum("ttuu->", eri[:nalpha, :nalpha, :nbeta, :nbeta])
        )
    return energy


def _reference_energy(parsed: ParsedFcidump) -> float:
    """``E_core`` plus the active-space energy, from the file and nothing else.

    A caveat worth stating, because the number is more reassuring than it
    should be: this total is **insensitive to the two-electron block**.
    ``E_core`` is defined as ``E_ref - E_act``, so an error in the ERIs moves
    ``E_act`` and ``E_core`` by equal and opposite amounts and cancels exactly.
    Doubling ``eri_act`` leaves this reproducing the SCF energy to every digit.

    So it checks the record round trip, the header, and the ``E_core + E_act``
    bookkeeping -- real things, and the ones a writer can break. What it does
    not check is whether the ERIs are physically right; :func:`_active_energy`
    catches a writer that corrupts them, and
    ``tests/test_hamiltonian.py::test_against_a_full_ao_to_mo_transform``
    is what validates the tensor itself against an independent transform.
    """
    return parsed.e_core + _active_energy(parsed)


def _canonical(i, j, k, m):
    """The representative of an index quadruple's 8-fold orbit."""
    bra, ket = tuple(sorted((i, j), reverse=True)), tuple(sorted((k, m), reverse=True))
    return max(bra, ket) + min(bra, ket)


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def hamiltonians(fixture_path):
    def _build(name):
        return active_hamiltonian(load(fixture_path(name)))

    return _build


@pytest.fixture
def written(hamiltonians, tmp_path):
    def _write(name, **kwargs):
        hamiltonian = hamiltonians(name)
        path = write_fcidump(hamiltonian, tmp_path / f"{name}.FCIDUMP", **kwargs)
        return hamiltonian, path, _read_fcidump(path)

    return _write


# ------------------------------------------------- the check that comes from outside


@pytest.mark.parametrize("name", SOUND)
def test_reference_energy_read_back_reproduces_the_scf_energy(
    written, fixture_path, name
):
    """The whole pipeline, judged against a number PySCF produced independently."""
    bundle = load(fixture_path(name))
    _, _, parsed = written(name)
    assert bundle.e_scf is not None
    assert _reference_energy(parsed) == pytest.approx(bundle.e_scf, abs=1e-9)


@pytest.mark.parametrize("name", SOUND)
def test_the_active_energy_from_the_file_matches_the_hamiltonian(written, name):
    """The part of the energy that does depend on the two-electron block.

    Needed because the reference-energy test above cannot see the ERIs at all:
    ``E_core`` absorbs any error in them exactly. This one compares the active
    energy alone, so a writer that mangles, drops or misplaces a two-electron
    record changes the number.
    """
    hamiltonian, _, parsed = written(name)
    assert _active_energy(parsed) == pytest.approx(hamiltonian.e_act, abs=1e-9)


def test_the_reference_energy_alone_really_is_blind_to_the_eris(
    hamiltonians, fixture_path, tmp_path
):
    """Guard the guard: proves the caveat in _reference_energy is real.

    If this ever fails -- if the round trip starts noticing a doubled ERI
    tensor -- then the separate active-energy test above is redundant and the
    docstrings that explain why it exists are wrong.
    """
    bundle = load(fixture_path("h2o_rhf"))
    doubled = active_hamiltonian(replace(bundle, eri_act=2.0 * bundle.eri_act))
    parsed = _read_fcidump(
        write_fcidump(doubled, tmp_path / "doubled.FCIDUMP")
    )
    assert _reference_energy(parsed) == pytest.approx(bundle.e_scf, abs=1e-9)
    assert _active_energy(parsed) != pytest.approx(
        active_hamiltonian(bundle).e_act, abs=1e-6
    )


@pytest.mark.parametrize("name", SOUND)
def test_the_scalar_record_is_e_core_and_comes_last(written, name):
    hamiltonian, path, parsed = written(name)
    assert parsed.e_core == pytest.approx(hamiltonian.e_core, abs=0)
    last = path.read_text().rstrip().splitlines()[-1].split()
    assert [int(x) for x in last[1:5]] == [0, 0, 0, 0]


# ---------------------------------------------------------------- the header


@pytest.mark.parametrize("name", SOUND)
def test_the_header_describes_the_active_space(written, name):
    hamiltonian, _, parsed = written(name)
    assert parsed.norb == hamiltonian.nact
    assert parsed.nelec == hamiltonian.nelec_act
    assert parsed.ms2 == hamiltonian.ms2
    assert parsed.orbsym == [1] * hamiltonian.nact
    assert parsed.isym == 1


def test_the_first_line_is_whitespace_separated_for_dice(written):
    """Dice splits this line and indexes tokens; one compact token defeats it."""
    _, path, _ = written("h2o_rhf")
    first = path.read_text().splitlines()[0]
    assert first.startswith("&FCI ")
    assert len(first.split()) >= 7, f"too few tokens for Dice to index: {first!r}"


def test_orbsym_is_one_line_with_a_trailing_comma(written):
    """Readers that match one ``(\\d+),`` per orbital need the final comma."""
    hamiltonian, path, _ = written("h2o_rhf")
    orbsym_line = next(
        line for line in path.read_text().splitlines() if "ORBSYM" in line
    )
    assert orbsym_line.rstrip().endswith(",")
    assert orbsym_line.count(",") == hamiltonian.nact


def test_the_terminator_is_alone_on_its_line(written):
    """Dice only recognises &END when it is the single token on the line."""
    _, path, _ = written("h2o_rhf")
    line = next(
        line for line in path.read_text().splitlines() if "&END" in line.upper()
    )
    assert line.split() == ["&END"]


def test_no_uhf_key_is_written(written):
    """The restricted form is the default; an IUHF line would shift Dice's ORBSYM."""
    _, path, _ = written("ch2_rohf")
    head = path.read_text().split("&END")[0].upper()
    assert "IUHF" not in head and "UHF" not in head


def test_custom_orbsym_and_isym_reach_the_file(hamiltonians, tmp_path):
    hamiltonian = hamiltonians("h2o_rhf")
    orbsym = [1, 2, 3, 4, 1, 2][: hamiltonian.nact]
    path = write_fcidump(
        hamiltonian, tmp_path / "sym.FCIDUMP", orbsym=orbsym, isym=3
    )
    parsed = _read_fcidump(path)
    assert parsed.orbsym == orbsym
    assert parsed.isym == 3


# ------------------------------------------------ one integral per orbit, exactly


@pytest.mark.parametrize("name", SOUND)
def test_every_symmetry_unique_integral_appears_exactly_once(written, name):
    """No duplicates and no omissions, judged against the full orbit set."""
    hamiltonian, _, parsed = written(name)
    nact = hamiltonian.nact

    written_orbits = [_canonical(*r[1:5]) for r in parsed.two_electron]
    assert len(written_orbits) == len(set(written_orbits)), "an orbit was written twice"

    complete = {
        _canonical(i, j, k, m)
        for i in range(1, nact + 1)
        for j in range(1, nact + 1)
        for k in range(1, nact + 1)
        for m in range(1, nact + 1)
    }
    assert set(written_orbits) == complete

    npair = nact * (nact + 1) // 2
    assert len(written_orbits) == npair * (npair + 1) // 2


@pytest.mark.parametrize("name", SOUND)
def test_the_one_electron_block_is_the_lower_triangle(written, name):
    hamiltonian, _, parsed = written(name)
    nact = hamiltonian.nact
    indices = sorted((r[1], r[2]) for r in parsed.one_electron)
    assert indices == sorted(
        (i, j) for i in range(1, nact + 1) for j in range(1, i + 1)
    )


@pytest.mark.parametrize("name", SOUND)
def test_the_record_count_is_exactly_what_the_format_requires(written, name):
    hamiltonian, path, parsed = written(name)
    npair = hamiltonian.nact * (hamiltonian.nact + 1) // 2
    expected = npair * (npair + 1) // 2 + npair + 1
    assert len(parsed.records) == expected
    assert len(path.read_text().splitlines()) == expected + 4  # + the namelist


# --------------------------------------------------------------- the numbers


@pytest.mark.parametrize("name", SOUND)
def test_the_written_triangle_is_bit_exact(written, name):
    """``%.16E`` carries 17 significant digits, which distinguishes every double."""
    hamiltonian, _, parsed = written(name)
    lower = np.tril_indices(hamiltonian.nact)
    np.testing.assert_array_equal(
        parsed.h1[lower], np.asarray(hamiltonian.h_eff)[lower]
    )
    for value, i, j, k, m in parsed.two_electron:
        assert value == hamiltonian.eri_act[i - 1, j - 1, k - 1, m - 1]


@pytest.mark.parametrize("name", SOUND)
def test_the_scattered_tensors_match_within_the_inputs_own_asymmetry(written, name):
    """The file stores one triangle, so the mirror is exact only where the input was.

    ``h'`` comes out of two matrix transforms and is symmetric to about machine
    epsilon, not exactly. Reading it back symmetrises it, so the two differ by
    exactly that residue -- which is the right amount to tolerate, and is
    asserted rather than assumed.
    """
    hamiltonian, _, parsed = written(name)
    h = np.asarray(hamiltonian.h_eff)
    asymmetry = float(np.max(np.abs(h - h.T)))
    assert asymmetry < 1e-12, "the fixture is less symmetric than expected"

    np.testing.assert_allclose(parsed.h1, h, atol=max(asymmetry, 1e-300))
    np.testing.assert_array_equal(parsed.h1, parsed.h1.T)
    np.testing.assert_allclose(parsed.eri, hamiltonian.eri_act, atol=1e-12)


def test_lower_precision_is_lossy_and_says_so_by_being_lossy(hamiltonians, tmp_path):
    """The default is not decoration: fewer digits really do lose the value."""
    hamiltonian = hamiltonians("h2o_rhf")
    exact = _read_fcidump(
        write_fcidump(hamiltonian, tmp_path / "exact.FCIDUMP")
    )
    coarse = _read_fcidump(
        write_fcidump(hamiltonian, tmp_path / "coarse.FCIDUMP", precision=6)
    )
    lower = np.tril_indices(hamiltonian.nact)
    assert np.array_equal(exact.h1[lower], np.asarray(hamiltonian.h_eff)[lower])
    assert not np.array_equal(coarse.h1[lower], np.asarray(hamiltonian.h_eff)[lower])
    assert ROUND_TRIP_PRECISION == 16


# -------------------------------------------------------------- thresholding


@pytest.mark.parametrize("name", SOUND)
def test_a_threshold_drops_exactly_the_integrals_below_it(written, name):
    """An exact, checkable claim: what is present is what is above the threshold."""
    threshold = 1e-6
    hamiltonian, _, parsed = written(name, threshold=threshold)
    for value, *_ in parsed.two_electron:
        assert abs(value) >= threshold

    eri = np.asarray(hamiltonian.eri_act)
    npair = hamiltonian.nact * (hamiltonian.nact + 1) // 2
    pair_t, pair_u = np.tril_indices(hamiltonian.nact)
    kept = 0
    for bra in range(npair):
        values = eri[pair_t[bra], pair_u[bra], pair_t[: bra + 1], pair_u[: bra + 1]]
        kept += int(np.count_nonzero(np.abs(values) >= threshold))
    assert len(parsed.two_electron) == kept


def test_a_threshold_keeps_the_one_electron_block_whole(written):
    """h' is O(nact^2) numbers; thinning it saves nothing worth the risk."""
    hamiltonian, _, parsed = written("ch2_rohf", threshold=1e-3)
    npair = hamiltonian.nact * (hamiltonian.nact + 1) // 2
    assert len(parsed.one_electron) == npair


def test_an_integral_exactly_at_the_threshold_is_kept(hamiltonians, tmp_path):
    """The boundary, which no realistic tensor lands on by accident.

    "Omits integrals smaller than the threshold" makes ``|v| == threshold`` a
    keep. Mutation testing found this: flipping the comparison from ``>=`` to
    ``>`` changed nothing any other test could see, because no ERI in any
    fixture is ever exactly equal to a threshold anyone would pass. Choosing the
    threshold to be one of the values makes the boundary observable.
    """
    hamiltonian = hamiltonians("h2o_rhf")
    eri = np.asarray(hamiltonian.eri_act)

    # A symmetry-unique element, and one large enough not to be the smallest.
    threshold = abs(float(eri[2, 1, 2, 1]))
    assert threshold > 0.0

    parsed = _read_fcidump(
        write_fcidump(hamiltonian, tmp_path / "edge.FCIDUMP", threshold=threshold)
    )
    present = {tuple(r[1:5]) for r in parsed.two_electron}
    assert (3, 2, 3, 2) in present, (
        "an integral whose magnitude equals the threshold exactly was dropped; "
        "the threshold omits what is smaller than it, not what is not larger"
    )
    for value, *_ in parsed.two_electron:
        assert abs(value) >= threshold


def test_the_default_threshold_writes_every_unique_integral(written):
    hamiltonian, _, parsed = written("h2o_rhf")
    npair = hamiltonian.nact * (hamiltonian.nact + 1) // 2
    assert len(parsed.two_electron) == npair * (npair + 1) // 2


def test_a_threshold_can_move_the_reference_energy_by_a_lot(
    hamiltonians, tmp_path, fixture_path
):
    """Why the default is ``0.0`` and not some small, safe-looking number.

    The damage is not proportional to the threshold and it is not smooth. For
    this fixture ``1e-2`` happens to leave ``E_ref`` untouched to 1e-14, because
    every integral the occupied block contributes is larger than that -- and
    ``1e-1`` then costs 0.2 Ha. A threshold that looks harmless on one system is
    no evidence about the next one, so the writer never picks one.
    """
    hamiltonian = hamiltonians("ch2_rohf")
    bundle = load(fixture_path("ch2_rohf"))

    exact = _read_fcidump(write_fcidump(hamiltonian, tmp_path / "exact.FCIDUMP"))
    assert _reference_energy(exact) == pytest.approx(bundle.e_scf, abs=1e-9)

    coarse = _read_fcidump(
        write_fcidump(hamiltonian, tmp_path / "coarse.FCIDUMP", threshold=1e-1)
    )
    assert len(coarse.two_electron) < len(exact.two_electron)
    assert abs(_reference_energy(coarse) - bundle.e_scf) > 0.1


# -------------------------------------------------------------- provenance


def test_provenance_goes_beside_the_file_and_not_into_it(hamiltonians, tmp_path, fixture_path):
    """FCIDUMP has no comment syntax; a comment can truncate or crash a reader."""
    bundle = load(fixture_path("h2o_rhf"))
    hamiltonian = hamiltonians("h2o_rhf")
    path = write_fcidump(
        hamiltonian, tmp_path / "p.FCIDUMP", provenance=bundle.provenance
    )

    text = path.read_text()
    assert "!" not in text and "#" not in text
    for line in text.splitlines()[4:]:
        assert len(line.split()) == 5, f"not an integral record: {line!r}"

    sidecar = provenance_path(path)
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["schema_version"] == bundle.schema_version


def test_no_provenance_means_no_sidecar(hamiltonians, tmp_path):
    path = write_fcidump(hamiltonians("h2o_rhf"), tmp_path / "bare.FCIDUMP")
    assert not provenance_path(path).exists()


def test_rewriting_without_provenance_removes_the_stale_sidecar(
    hamiltonians, tmp_path
):
    """A sidecar describing an earlier dump is worse than no sidecar at all.

    Left in place it sits beside a FCIDUMP it does not describe, and every tool
    that reads the pair -- including this package's own CLI, which reports what
    it wrote -- would report provenance for the wrong file.
    """
    hamiltonian = hamiltonians("h2o_rhf")
    path = tmp_path / "again.FCIDUMP"
    write_fcidump(hamiltonian, path, provenance={"run": "first"})
    assert provenance_path(path).exists()

    write_fcidump(hamiltonian, path)
    assert not provenance_path(path).exists()


def test_a_failed_write_leaves_the_previous_file_intact(hamiltonians, tmp_path):
    """The destination is replaced only once a complete file exists.

    The symmetry check runs inside the record loop, so a write can fail at any
    bra pair. Streaming into the destination would truncate a good FCIDUMP to a
    prefix of itself -- and a truncated FCIDUMP still parses, with every
    integral it does contain correct, so nothing downstream would notice.
    """
    hamiltonian = hamiltonians("h2o_rhf")
    path = tmp_path / "precious.FCIDUMP"
    write_fcidump(hamiltonian, path)
    original = path.read_bytes()

    broken = np.array(hamiltonian.eri_act, copy=True)
    broken[3, 2, 1, 0] += 1e-3
    with pytest.raises(WriteError):
        write_fcidump(replace(hamiltonian, eri_act=broken), path)

    assert path.read_bytes() == original
    assert not path.with_name(path.name + ".partial").exists()


def test_a_failed_first_write_leaves_nothing_behind(hamiltonians, tmp_path):
    hamiltonian = hamiltonians("h2o_rhf")
    path = tmp_path / "never.FCIDUMP"
    broken = np.array(hamiltonian.eri_act, copy=True)
    broken[3, 2, 1, 0] += 1e-3
    with pytest.raises(WriteError):
        write_fcidump(replace(hamiltonian, eri_act=broken), path)
    assert not path.exists()
    assert not path.with_name(path.name + ".partial").exists()


# ------------------------------------------------------- refusing bad input


def test_a_nonsymmetric_h_is_refused(hamiltonians, tmp_path):
    """Only one triangle is written, so the other one would vanish silently."""
    hamiltonian = hamiltonians("h2o_rhf")
    broken = np.array(hamiltonian.h_eff, copy=True)
    broken[0, 1] += 1e-3
    with pytest.raises(WriteError, match="not symmetric"):
        write_fcidump(replace(hamiltonian, h_eff=broken), tmp_path / "x.FCIDUMP")


def test_a_nonsymmetric_eri_is_refused(hamiltonians, tmp_path):
    hamiltonian = hamiltonians("h2o_rhf")
    broken = np.array(hamiltonian.eri_act, copy=True)
    broken[1, 0, 0, 0] += 1e-3
    with pytest.raises(WriteError, match="violates"):
        write_fcidump(replace(hamiltonian, eri_act=broken), tmp_path / "x.FCIDUMP")


def test_the_eri_tolerance_matches_the_schemas(hamiltonians, tmp_path):
    """A writer stricter than bundle.validate would refuse valid bundles."""
    from g16dump.bundle import ERI_SYMMETRY_TOL

    hamiltonian = hamiltonians("h2o_rhf")
    nudged = np.array(hamiltonian.eri_act, copy=True)
    nudged[1, 0, 0, 0] += ERI_SYMMETRY_TOL / 10.0
    write_fcidump(replace(hamiltonian, eri_act=nudged), tmp_path / "ok.FCIDUMP")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"orbsym": [1, 1]}, "ORBSYM must name an irrep"),
        ({"orbsym": [0] * 6}, "outside 1-8"),
        ({"orbsym": [9] * 6}, "outside 1-8"),
        ({"isym": 0}, "isym is 0, outside 1-8"),
        ({"isym": 9}, "isym is 9, outside 1-8"),
        ({"threshold": -1.0}, "non-negative"),
        ({"precision": 0}, "precision must be"),
        ({"precision": 18}, "precision must be"),
    ],
)
def test_bad_arguments_are_refused_by_name(hamiltonians, tmp_path, kwargs, message):
    with pytest.raises(WriteError, match=message):
        write_fcidump(hamiltonians("h2o_rhf"), tmp_path / "x.FCIDUMP", **kwargs)


def test_a_nonfinite_integral_is_refused(hamiltonians, tmp_path):
    hamiltonian = hamiltonians("h2o_rhf")
    broken = np.array(hamiltonian.h_eff, copy=True)
    broken[0, 0] = np.nan
    with pytest.raises(WriteError, match="non-finite"):
        write_fcidump(replace(hamiltonian, h_eff=broken), tmp_path / "x.FCIDUMP")


def test_impossible_electron_counts_are_refused(hamiltonians, tmp_path):
    hamiltonian = hamiltonians("h2o_rhf")
    with pytest.raises(WriteError, match="parity is wrong"):
        write_fcidump(replace(hamiltonian, ms2=1), tmp_path / "x.FCIDUMP")


def test_a_spin_string_that_breaks_the_pauli_bound_is_refused(
    hamiltonians, tmp_path
):
    """The total-electron count is not enough: each spin string has its own bound.

    Six active orbitals hold twelve electrons in total, so NELEC = 10 passes
    that check. With MS2 = 8 those ten are nine alpha and one beta, and nine
    alpha electrons cannot occupy six spatial orbitals whatever the total says.
    """
    hamiltonian = hamiltonians("h2o_rhf")
    assert hamiltonian.nact == 6
    with pytest.raises(WriteError, match="does not fit in 6 spatial orbitals"):
        write_fcidump(
            replace(hamiltonian, nelec_act=10, ms2=8), tmp_path / "pauli.FCIDUMP"
        )


def test_writing_creates_the_parent_directory(hamiltonians, tmp_path):
    path = write_fcidump(
        hamiltonians("h2o_rhf"), tmp_path / "deep" / "er" / "FCIDUMP"
    )
    assert path.is_file()


# ------------------------------------------------------------- Dice's nocc line


def test_the_dice_occupation_line_fills_the_lowest_spin_orbitals():
    """Spatial orbital p is alpha 2p and beta 2p+1, both 0-based."""
    from g16dump.hamiltonian import ActiveHamiltonian

    def fake(nelec, ms2):
        return ActiveHamiltonian(
            h_eff=np.zeros((4, 4)), eri_act=np.zeros((4,) * 4), e_core=0.0,
            e_ref=0.0, e_act=0.0, nact=4, nelec_act=nelec, ms2=ms2,
            spin_deviation=0.0, fock_source="gaussian",
        )

    assert dice_occupation_line(fake(4, 0)).splitlines() == [
        "nocc 4", "0 2 1 3", "end",
    ]
    assert dice_occupation_line(fake(5, 1)).splitlines() == [
        "nocc 5", "0 2 4 1 3", "end",
    ]


@pytest.mark.parametrize("name", SOUND)
def test_the_dice_occupation_line_agrees_with_the_header(written, name):
    hamiltonian, _, parsed = written(name)
    lines = dice_occupation_line(hamiltonian).splitlines()
    assert lines[0] == f"nocc {parsed.nelec}"
    spin_orbitals = [int(x) for x in lines[1].split()]
    assert len(spin_orbitals) == parsed.nelec
    assert len(set(spin_orbitals)) == parsed.nelec
    assert sum(1 for s in spin_orbitals if s % 2 == 0) - sum(
        1 for s in spin_orbitals if s % 2
    ) == parsed.ms2


# --------------------------------------------------- a genuinely foreign parser


@pytest.mark.pyscf
@pytest.mark.parametrize("name", SOUND)
def test_pyscf_reads_the_file_and_agrees(hamiltonians, fixture_path, tmp_path, name):
    """The strongest available check that this is a FCIDUMP and not merely our own.

    PySCF's reader shares nothing with this project. If the namelist spelling,
    the index convention or the 8-fold packing were wrong, this is where it
    shows.
    """
    fcidump = pytest.importorskip("pyscf.tools.fcidump")
    ao2mo = pytest.importorskip("pyscf.ao2mo")

    bundle = load(fixture_path(name))
    hamiltonian = hamiltonians(name)
    path = write_fcidump(hamiltonian, tmp_path / f"{name}.FCIDUMP")

    data = fcidump.read(str(path), verbose=False)
    assert data["NORB"] == hamiltonian.nact
    assert data["NELEC"] == hamiltonian.nelec_act
    assert data["MS2"] == hamiltonian.ms2

    np.testing.assert_allclose(data["H1"], hamiltonian.h_eff, atol=1e-12)
    eri = ao2mo.restore(1, np.asarray(data["H2"]), hamiltonian.nact)
    np.testing.assert_allclose(eri, hamiltonian.eri_act, atol=1e-12)
    assert data["ECORE"] == pytest.approx(hamiltonian.e_core, abs=1e-12)

    parsed = _read_fcidump(path)
    assert _reference_energy(parsed) == pytest.approx(bundle.e_scf, abs=1e-9)
