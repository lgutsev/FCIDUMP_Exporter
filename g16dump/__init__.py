"""g16dump: FCIDUMP files from Gaussian 16 windowed MO integrals.

Active-space Hamiltonians for SHCI (Dice) and DMRG (Block2), built without ever
transforming integrals over the full MO space. The frozen core is folded
analytically into an effective one-electron Hamiltonian and a scalar, using the
full MO Fock matrices -- one code path for RHF and ROHF.

The pipeline is:

    .mat  --matfile.py-->  .npz bundle  --hamiltonian.py-->  h', E_core
                                        --write.py-------->  FCIDUMP

Only ``matfile.py`` imports gauopen, and only the rebuilt-Fock path imports
PySCF. Everything else, including the whole test suite, is NumPy alone.
"""

from .errors import (  # noqa: F401
    BundleError,
    ConsistencyError,
    G16DumpError,
    MissingDependencyError,
    ReferenceTypeError,
    SchemaError,
    ValidationError,
)

__version__ = "0.0.1.dev0"
