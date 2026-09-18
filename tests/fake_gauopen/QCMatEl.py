"""Mock stand-in for gauopen's QCMatEl, only to smoke-test scripts/inspect_mat.py.

Deliberately awkward: packed lower-triangular 1e blocks, a flat nact**4 2e block,
an attribute that raises, and a scalar() that rejects unknown names -- so the
probe is exercised against the failure modes it must survive on the cluster.
"""
import numpy as np

NAO, NMO, NACT = 7, 7, 6


class _Mat:
    def __init__(self, array, packed=False, n=None, kind="mock"):
        self.array = array
        self.type = kind
        self.nrow = n or array.shape[0]
        self.ncol = n or array.shape[-1]
        self._packed = packed
        self._n = n

    @property
    def exploding(self):
        raise RuntimeError("this attribute always raises, on purpose")

    def expand(self):
        if not self._packed:
            return self.array
        n = self._n
        out = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                out[i, j] = out[j, i] = self.array[idx]
                idx += 1
        return out

    def lenarray(self):
        return self.array.size


def _tri(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n * (n + 1) // 2)


class MatEl:
    def __init__(self, file=None):
        self.filename = file
        self.nbasis, self.nbsuse = NAO, NMO
        self.nfc, self.nfv = 1, 0
        self.ne, self.multip = 10, 1
        self.icgu = 221
        rng = np.random.default_rng(0)
        self.matlist = {
            "CORE HAMILTONIAN ALPHA": _Mat(_tri(NAO, 1), packed=True, n=NAO),
            "KINETIC ENERGY": _Mat(_tri(NAO, 2), packed=True, n=NAO),
            "OVERLAP": _Mat(_tri(NAO, 3), packed=True, n=NAO),
            "ALPHA MO COEFFICIENTS": _Mat(rng.normal(size=NMO * NAO), n=NAO),
            "ALPHA ORBITAL ENERGIES": _Mat(rng.normal(size=NMO), n=NMO),
            "AA MO 2E INTEGRALS": _Mat(rng.normal(size=NACT**4), n=NACT),
        }
        self._scalars = {"ENUCREP": 9.1671, "ESCF": -74.9659}

    def scalar(self, name):
        if name not in self._scalars:
            raise KeyError(f"no scalar named {name!r}")
        return self._scalars[name]
