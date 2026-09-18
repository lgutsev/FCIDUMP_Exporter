"""Benchmark manifests, active-space sweeps and sweep analysis for g16dump.

This package is deliberately outside ``g16dump/``: it describes *what to run*
and *how to read what came back*, and it must stay importable on a machine that
has neither Gaussian nor a solver installed. Core dependency is ``numpy`` only,
matching the rest of the repository; ``pyscf`` is used lazily by the optional
FCI validation path in :mod:`benchmarks.analyze`.
"""

from __future__ import annotations

__all__ = ["manifest", "sweep", "analyze"]
