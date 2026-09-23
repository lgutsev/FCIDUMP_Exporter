"""g16dump: FCIDUMP files from Gaussian 16 windowed MO integrals.

The package is a chain of plain modules with a single interchange format in the
middle of it:

    .mat  --matfile.py-->  .npz bundle  --hamiltonian.py-->  h', E_core
                                        --rotate.py-------->  rotated bundle
                                        --write.py--------->  FCIDUMP

Only ``matfile.py`` imports gauopen, and only the rebuilt-Fock path imports
pyscf. Everything else is numpy.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
