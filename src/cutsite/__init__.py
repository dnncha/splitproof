__version__ = "0.1.0"

from .errors import (  # noqa: E402
    BackendFailed,
    BackendUnavailable,
    CertificateMismatch,
    ConstraintViolated,
    CutsiteError,
    DataError,
    GiantUnit,
    InfeasibleSplit,
    PolicyError,
    VacuousConstraint,
)
from .sim import TooManyPairs  # noqa: E402
from .policy import Policy  # noqa: E402

__all__ = [
    "__version__",
    "Policy",
    "CutsiteError",
    "PolicyError",
    "DataError",
    "BackendUnavailable",
    "InfeasibleSplit",
    "GiantUnit",
    "CertificateMismatch",
    "ConstraintViolated",
    "VacuousConstraint",
    "BackendFailed",
    "TooManyPairs",
    "audit",
    "make_split",
    "verify",
]


def __getattr__(name: str):
    if name in {"make_split", "audit", "verify", "Outcome"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module 'cutsite' has no attribute {name!r}")
