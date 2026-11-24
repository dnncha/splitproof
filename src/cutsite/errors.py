from __future__ import annotations

from typing import Any, Dict, Optional


class CutsiteError(Exception):
    code = "CUTSITE_ERROR"

    def __init__(self, message: str, *, remedy: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.remedy = remedy
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "remedy": self.remedy,
            "details": self.details,
        }


class PolicyError(CutsiteError):
    code = "POLICY_INVALID"


class DataError(CutsiteError):
    code = "DATA_INVALID"


class BackendUnavailable(CutsiteError):
    code = "BACKEND_UNAVAILABLE"


class InfeasibleSplit(CutsiteError):
    code = "SPLIT_INFEASIBLE"


class GiantUnit(InfeasibleSplit):
    code = "GIANT_UNIT"


class CertificateMismatch(CutsiteError):
    code = "CERT_MISMATCH"


class ConstraintViolated(CutsiteError):
    code = "CONSTRAINT_VIOLATED"
