"""
flux-coop-runtime / fleet_compat.py
Compatibility shim between local error constants and fleet-stdlib.

This module provides a drop-in migration path from the repo-local
ERR_* constants and CooperativeRuntimeError to the canonical fleet-stdlib
ErrorCode enum and FleetError class.

Usage (incremental migration)::

    # Old code:
    from src.runtime import ERR_NO_CAPABLE_AGENT, CooperativeRuntimeError
    raise CooperativeRuntimeError(f"{ERR_NO_CAPABLE_AGENT}: …")

    # New code:
    from src.fleet_compat import ERR_NO_CAPABLE_AGENT, CooperativeRuntimeError
    raise CooperativeRuntimeError(ERR_NO_CAPABLE_AGENT, "…")
    # or, directly:
    from src.fleet_compat import to_fleet_error
    raise to_fleet_error(old_exception)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Re-export fleet-stdlib types
# ---------------------------------------------------------------------------
# fleet-stdlib is expected to be importable (installed as a sibling package
# or via a workspace layout).  Fall back to a lazy-import helper if the
# package is not on sys.path.

try:
    from flux_fleet_stdlib.errors import ErrorCode, FleetError, Severity, fleet_error  # type: ignore[import-untyped]
    from flux_fleet_stdlib.status import Status, status_for_error_code  # type: ignore[import-untyped]
    _STDLIB_AVAILABLE = True
except ImportError:
    _STDLIB_AVAILABLE = False

    # Minimal stubs so the module never hard-crashes during early migration.
    from enum import Enum as _Enum

    class ErrorCode(_Enum):  # type: ignore[no-redef]
        COOP_NO_CAPABLE_AGENT = "COOP_NO_CAPABLE_AGENT"
        COOP_TIMEOUT = "COOP_TIMEOUT"
        COOP_TASK_EXPIRED = "COOP_TASK_EXPIRED"
        COOP_AGENT_REFUSED = "COOP_AGENT_REFUSED"
        COOP_TRANSPORT_FAILURE = "COOP_TRANSPORT_FAILURE"
        TRANSPORT_GIT_ERROR = "TRANSPORT_GIT_ERROR"

    class FleetError(Exception):  # type: ignore[no-redef]
        def __init__(self, code: str, message: str, **kw: Any) -> None:
            self.code = code
            self.message = message
            self.context: Dict[str, Any] = kw.get("context", {})
            super().__init__(f"[{code}] {message}")

    class Severity:  # type: ignore[no-redef]
        ERROR = "ERROR"

    def fleet_error(code: str, message: str, **kw: Any) -> FleetError:  # type: ignore[misc]
        return FleetError(code, message)

    class Status:  # type: ignore[no-redef]
        SUCCESS = "SUCCESS"
        PENDING = "PENDING"
        TIMEOUT = "TIMEOUT"
        ERROR = "ERROR"
        REFUSED = "REFUSED"
        CANCELLED = "CANCELLED"

    def status_for_error_code(code: str) -> "Status":  # type: ignore[misc]
        return Status.ERROR


# ---------------------------------------------------------------------------
# Legacy ERR_* constant mapping
# ---------------------------------------------------------------------------
# These names are kept identical to the old runtime.py constants so that
# existing ``from src.runtime import ERR_*`` statements can be redirected
# to ``from src.fleet_compat import ERR_*`` with zero code changes.

ERR_NO_CAPABLE_AGENT = ErrorCode.COOP_NO_CAPABLE_AGENT.value  # type: ignore[union-attr]
ERR_TIMEOUT = ErrorCode.COOP_TIMEOUT.value  # type: ignore[union-attr]
ERR_TRANSPORT_FAILURE = ErrorCode.TRANSPORT_GIT_ERROR.value  # type: ignore[union-attr]
ERR_TASK_EXPIRED = ErrorCode.COOP_TASK_EXPIRED.value  # type: ignore[union-attr]
ERR_AGENT_REFUSED = ErrorCode.COOP_AGENT_REFUSED.value  # type: ignore[union-attr]

# Full mapping table for programmatic lookups.
LEGACY_CODE_MAP: Dict[str, str] = {
    "NO_CAPABLE_AGENT": ERR_NO_CAPABLE_AGENT,
    "TIMEOUT": ERR_TIMEOUT,
    "TRANSPORT_FAILURE": ERR_TRANSPORT_FAILURE,
    "TASK_EXPIRED": ERR_TASK_EXPIRED,
    "AGENT_REFUSED": ERR_AGENT_REFUSED,
}


# ---------------------------------------------------------------------------
# CooperativeRuntimeError — now backed by FleetError
# ---------------------------------------------------------------------------

class CooperativeRuntimeError(FleetError):
    """Drop-in replacement for the old ``CooperativeRuntimeError``.

    Subclasses :class:`FleetError` so every exception carries the canonical
    ``code`` and structured ``context`` fields expected by fleet consumers.

    Supports two call conventions for backward compatibility:

    1. *Legacy string form* (old code)::

        raise CooperativeRuntimeError("NO_CAPABLE_AGENT: Cannot resolve …")

    2. *Structured form* (new code)::

        raise CooperativeRuntimeError(
            code=ErrorCode.COOP_NO_CAPABLE_AGENT.value,
            message="Cannot resolve target",
            source_repo="flux-coop-runtime",
        )
    """

    def __init__(
        self,
        *args: Any,
        code: Optional[str] = None,
        message: Optional[str] = None,
        severity: str = Severity.ERROR,
        source_repo: str = "flux-coop-runtime",
        source_agent: str = "",
        **kwargs: Any,
    ) -> None:
        # Detect legacy usage: a single positional string argument like
        # "NO_CAPABLE_AGENT: Cannot resolve target"
        if len(args) == 1 and isinstance(args[0], str) and code is None and message is None:
            raw = args[0]
            code, message = _parse_legacy_error_string(raw)
        elif len(args) == 0 and code is not None and message is not None:
            pass  # structured form — nothing to do
        elif len(args) >= 2:
            # Someone passed positional code + message without keywords.
            code = args[0]
            message = args[1]
        else:
            # Fallback: just forward everything as a generic error.
            code = code or "COOP_UNKNOWN_REQUEST"
            message = message or str(args) if args else "unknown cooperative error"

        super().__init__(
            code=code,
            message=message,
            severity=severity,
            source_repo=source_repo,
            source_agent=source_agent,
            context=kwargs.get("context", {}),
        )


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------

def _parse_legacy_error_string(raw: str) -> tuple:
    """Parse ``"ERR_CODE: message"`` into ``(code, message)``.

    Also handles strings where the old constant value (e.g. ``"TIMEOUT"``)
    was embedded directly without a prefix.
    """
    if ": " in raw:
        prefix, _, msg = raw.partition(": ")
        # Check if the prefix matches a known legacy code *value*.
        for old_val, new_code in LEGACY_CODE_MAP.items():
            if prefix == old_val:
                return new_code, msg
        # Check if the prefix matches a known legacy code *name*.
        mapped = LEGACY_CODE_MAP.get(prefix)
        if mapped:
            return mapped, msg
        # Unknown prefix — keep as-is.
        return prefix, msg

    # No separator — treat entire string as the message.
    return "COOP_UNKNOWN_REQUEST", raw


def to_fleet_error(
    exc: Exception,
    *,
    default_code: str = "COOP_UNKNOWN_REQUEST",
    source_repo: str = "flux-coop-runtime",
    source_agent: str = "",
    **extra_context: Any,
) -> FleetError:
    """Wrap any exception in a :class:`FleetError`.

    If *exc* is already a :class:`FleetError` (or subclass), it is returned
    unchanged.  For plain ``CooperativeRuntimeError`` instances from old code,
    the error string is parsed to extract the code.

    Parameters:
        exc:            The exception to wrap.
        default_code:   Fleet error code if parsing fails.
        source_repo:    Repo identifier stamped on the new error.
        source_agent:   Optional agent identifier.
        **extra_context: Additional key-value pairs stored in ``context``.

    Returns:
        A :class:`FleetError` instance.
    """
    if isinstance(exc, FleetError):
        return exc

    raw_message = str(exc)
    code, message = _parse_legacy_error_string(raw_message)

    if code == "COOP_UNKNOWN_REQUEST" and raw_message != default_code:
        # Parsing didn't find a known code — use the default.
        code = default_code
        message = raw_message

    context: Dict[str, Any] = {"original_type": type(exc).__name__}
    context.update(extra_context)

    return FleetError(
        code=code,
        message=message,
        severity=Severity.ERROR,
        source_repo=source_repo,
        source_agent=source_agent,
        context=context,
    )


def map_task_status_to_fleet(status: str) -> Status:
    """Map a local :class:`TaskStatus` value to a fleet :class:`Status`.

    The local ``TaskStatus`` enum uses lowercase values (``"success"``,
    ``"timeout"``, etc.) while fleet :class:`Status` uses uppercase
    (``"SUCCESS"``, ``"TIMEOUT"``).  This helper normalises.
    """
    _MAP = {
        "success": Status.SUCCESS,
        "sent": Status.PENDING,
        "received": Status.PENDING,
        "executing": Status.PENDING,
        "created": Status.PENDING,
        "error": Status.ERROR,
        "timeout": Status.TIMEOUT,
        "refused": Status.REFUSED,
        "cancelled": Status.CANCELLED,
    }
    return _MAP.get(status.lower(), Status.ERROR)


# ---------------------------------------------------------------------------
# Deprecation warnings
# ---------------------------------------------------------------------------

def _warn_legacy_import() -> None:
    """Emit a deprecation warning when code still imports from ``runtime``."""
    warnings.warn(
        "Importing error codes from src.runtime is deprecated.  "
        "Use 'from src.fleet_compat import ERR_NO_CAPABLE_AGENT, CooperativeRuntimeError' instead.",
        DeprecationWarning,
        stacklevel=3,
    )
