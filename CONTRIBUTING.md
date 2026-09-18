# Contributing

## The one rule about CI

**CI never has Gaussian and never has gauopen.** It also never has mokit, which
is not installable from PyPI. Everything that runs in CI runs off committed
`.npz` fixtures and PySCF-generated data. A test that needs any of those three
must say so with a marker, or it will fail the build for everyone:

```python
@pytest.mark.gauopen      # needs Gaussian's QCMatEl / qcmatrixio
@pytest.mark.gaussian     # needs a Gaussian binary, or real .mat/.fch output
@pytest.mark.mokit        # needs mokit
@pytest.mark.pyscf        # needs pyscf, the independent oracle
@pytest.mark.slow         # correct, but too slow for the fast feedback loop
```

The markers are declared in `pyproject.toml`. `pyscf` is the only one CI can
satisfy, and it does, in the `oracles` job.

`pytest.importorskip("pyscf")` inside a test works too and needs no marker. Use
the marker when you want the test deselected before collection, and
`importorskip` when a single test function happens to want an oracle.

## What the jobs check

| Job | What it proves |
|---|---|
| `core` (3.9–3.13) | the whole Gaussian-independent suite passes with **numpy and nothing else** installed |
| `oracles` (3.10, 3.12) | the PySCF-backed scientific gates pass |
| `lint` | ruff is clean, and `legacy/` is byte-identical to what was received |
| `build` | the sdist and wheel build, pass `twine check`, and carry no Gaussian output or reference material |

The `core` job is the important one. It installs only `.[test]` and then runs
`.github/scripts/check_core_deps.py`, which fails if:

- anything other than `numpy` appears in `[project] dependencies`;
- `pyscf`, `mokit`, `gauopen`, `QCMatEl`, `qcmatrixio`, `scipy`, `h5py` or
  `pandas` is importable in that environment;
- any module in `g16dump/` other than `matfile` fails to import without them.

That last one is the real invariant: `matfile.py` is the only module allowed to
touch gauopen, and everything downstream reads the `.npz` bundle instead. Keep
imports of the optional libraries lazy — inside the function that needs them,
not at module scope.

## Running it locally

```bash
python -m venv .venv && source .venv/bin/activate

pip install -e ".[test]"                  # what the core job has
pytest -m "not pyscf and not gauopen and not gaussian and not mokit"
python .github/scripts/check_core_deps.py

pip install -e ".[dev]"                   # + pyscf, ruff
pytest -m "not gauopen and not gaussian"
ruff check .
```

`ruff format` is configured (line length 100) but not enforced, so formatting a
file you are already editing is welcome and a repo-wide reformat is not.

## `legacy/`

`legacy/` is regression and reference material: the original scripts exactly as
received, kept so new results can be compared against the old ones. It is never
edited, never reformatted and never linted. `lint` verifies this against the
checksum manifest in `.github/legacy.sha256`. If it ever genuinely has to
change, regenerate the manifest deliberately and say why:

```bash
.github/scripts/check_legacy_untouched.sh --regenerate
```

## Dependencies

`numpy` is the only runtime dependency and that is a design constraint, not an
accident — the extraction path has to run on a cluster login node with nothing
else available. New dependencies go in an optional extra:

- `validate` — `pyscf`, for the oracles and the rebuilt-Fock path;
- `test` — `pytest`;
- `lint` — `ruff`;
- `dev` — all three.

`mokit` and `gauopen` are deliberately absent from every extra because neither
is on PyPI. `gauopen` ships with Gaussian and goes on `PYTHONPATH`; `mokit`
comes from conda-forge or a source build. See the README for both.
