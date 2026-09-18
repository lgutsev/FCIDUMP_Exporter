"""Errors that say what is physically wrong, not what NumPy tripped over.

Every failure in this package should tell the user which scientific assumption
broke, what was measured, and what to do about it. A bare ``ValueError: shapes
(7,7) and (6,6) not aligned`` is a bug report about us, not about their data.
"""

from __future__ import annotations


class G16DumpError(Exception):
    """Base class for every error this package raises deliberately."""


class BundleError(G16DumpError):
    """The .npz bundle is missing data, malformed, or internally inconsistent."""


class SchemaError(BundleError):
    """The bundle's schema version is absent or unsupported."""


class ValidationError(BundleError):
    """A bundle failed a physical or numerical consistency check."""


class ConsistencyError(G16DumpError):
    """A computed quantity failed a check that proves the algebra is sound.

    The canonical case is ``max|h'(alpha) - h'(beta)|`` exceeding tolerance,
    which means the stored Fock is not the UHF-type operator the derivation
    requires. Never paper over this by averaging.
    """


class ReferenceTypeError(G16DumpError):
    """The reference type is ambiguous or wrong for the requested code path.

    Raised when a Kohn-Sham bundle is fed to the stored-Fock path (the stored KS
    matrix contains exchange-correlation and is not a Fock operator), and when
    the reference type cannot be established from the file and the user has not
    specified it.
    """


class MissingDependencyError(G16DumpError):
    """An optional dependency is needed for this path and is not installed."""


def require(condition: bool, message: str, error=ValidationError) -> None:
    """Raise ``error(message)`` unless ``condition`` holds.

    Used instead of ``assert`` so checks survive ``python -O`` -- these are
    physical validity checks, not debugging aids.
    """
    if not condition:
        raise error(message)
