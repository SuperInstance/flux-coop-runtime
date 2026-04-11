"""
Core data types for cooperative execution.

Defines CooperativeTask, CooperativeResponse, and related types
used by all cooperative runtime components.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class TaskStatus(str, Enum):
    """Lifecycle states of a cooperative task."""
    CREATED = "created"
    SENT = "sent"
    RECEIVED = "received"
    EXECUTING = "executing"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    REFUSED = "refused"
    CANCELLED = "cancelled"


class RequestType(str, Enum):
    """Types of cooperative requests."""
    EXECUTE_BYTECODE = "execute_bytecode"
    QUERY_KNOWLEDGE = "query_knowledge"
    RUN_TEST = "run_test"
    PING = "ping"
    CUSTOM = "custom"


@dataclass
class CooperativeTask:
    """
    A request from one agent to another for cooperative execution.
    
    Serialized as JSON and written to the target agent's for-fleet/ bottle directory.
    """
    task_id: str
    source_agent: str
    target_agent: str
    request_type: str
    payload: Dict[str, Any]
    context: str = ""
    timeout_ms: int = 30000
    created_at: str = ""
    expires_at: str = ""
    source_repo: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.expires_at:
            expire_seconds = self.timeout_ms / 1000.0
            expire_time = time.time() + expire_seconds
            self.expires_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expire_time))

    def is_expired(self) -> bool:
        """Check if this task has passed its expiration time."""
        return time.time() > time.mktime(time.strptime(self.expires_at, "%Y-%m-%dT%H:%M:%SZ"))

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "version": self.version,
            "task_id": self.task_id,
            "source_agent": self.source_agent,
            "source_repo": self.source_repo,
            "target_agent": self.target_agent,
            "request_type": self.request_type,
            "payload": self.payload,
            "context": self.context,
            "timeout_ms": self.timeout_ms,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'CooperativeTask':
        """Create from JSON dictionary."""
        return cls(
            task_id=data["task_id"],
            source_agent=data["source_agent"],
            target_agent=data.get("target_agent", "any"),
            request_type=data["request_type"],
            payload=data.get("payload", {}),
            context=data.get("context", ""),
            timeout_ms=data.get("timeout_ms", 30000),
            created_at=data.get("created_at", ""),
            expires_at=data.get("expires_at", ""),
            source_repo=data.get("source_repo", ""),
            version=data.get("version", "1.0"),
        )

    @classmethod
    def create(cls, source_agent: str, target_agent: str,
               request_type: str, payload: Dict[str, Any],
               source_repo: str = "", context: str = "",
               timeout_ms: int = 30000) -> 'CooperativeTask':
        """Factory method with auto-generated task_id."""
        short_ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        unique = uuid.uuid4().hex[:6]
        task_id = f"{source_agent}-{short_ts}-{unique}"
        return cls(
            task_id=task_id,
            source_agent=source_agent,
            target_agent=target_agent,
            request_type=request_type,
            payload=payload,
            context=context,
            timeout_ms=timeout_ms,
            source_repo=source_repo,
        )

    def __repr__(self) -> str:
        return (
            f"CooperativeTask({self.task_id}, "
            f"{self.source_agent}→{self.target_agent}, "
            f"{self.request_type})"
        )


@dataclass
class CooperativeResponse:
    """
    A response from a target agent back to the requesting agent.
    
    Written to the source agent's from-fleet/ or for-fleet/ directory.
    """
    task_id: str
    source_agent: str
    target_agent: str
    status: str
    result: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0
    error_code: str = ""
    error_message: str = ""
    vm_info: Optional[Dict[str, Any]] = None
    responded_at: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if not self.responded_at:
            self.responded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        data = {
            "version": self.version,
            "task_id": self.task_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "status": self.status,
            "execution_time_ms": self.execution_time_ms,
            "responded_at": self.responded_at,
        }
        if self.result is not None:
            data["result"] = self.result
        if self.error_code:
            data["error_code"] = self.error_code
        if self.error_message:
            data["error_message"] = self.error_message
        if self.vm_info is not None:
            data["vm_info"] = self.vm_info
        return data

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'CooperativeResponse':
        """Create from JSON dictionary."""
        return cls(
            task_id=data["task_id"],
            source_agent=data.get("source_agent", ""),
            target_agent=data.get("target_agent", ""),
            status=data["status"],
            result=data.get("result"),
            execution_time_ms=data.get("execution_time_ms", 0),
            error_code=data.get("error_code", ""),
            error_message=data.get("error_message", ""),
            vm_info=data.get("vm_info"),
            responded_at=data.get("responded_at", ""),
            version=data.get("version", "1.0"),
        )

    @classmethod
    def success(cls, task_id: str, source_agent: str, target_agent: str,
                result: Dict[str, Any], execution_time_ms: int = 0,
                vm_info: Optional[Dict[str, Any]] = None) -> 'CooperativeResponse':
        """Factory for successful responses."""
        return cls(
            task_id=task_id,
            source_agent=source_agent,
            target_agent=target_agent,
            status="success",
            result=result,
            execution_time_ms=execution_time_ms,
            vm_info=vm_info,
        )

    @classmethod
    def error(cls, task_id: str, source_agent: str, target_agent: str,
              error_code: str, error_message: str) -> 'CooperativeResponse':
        """Factory for error responses."""
        return cls(
            task_id=task_id,
            source_agent=source_agent,
            target_agent=target_agent,
            status="error",
            error_code=error_code,
            error_message=error_message,
        )

    def __repr__(self) -> str:
        return (
            f"CooperativeResponse({self.task_id}, "
            f"status={self.status})"
        )


@dataclass
class TrustRecord:
    """Tracks reliability of an agent for trust scoring."""
    agent_name: str
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    last_seen: str = ""

    @property
    def total(self) -> int:
        return self.successes + self.failures + self.timeouts

    @property
    def score(self) -> float:
        if self.total == 0:
            return 0.5  # Neutral prior
        return self.successes / self.total

    def record(self, status: str) -> None:
        if status == "success":
            self.successes += 1
        elif status == "error":
            self.failures += 1
        elif status == "timeout":
            self.timeouts += 1
        self.last_seen = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "successes": self.successes,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "score": round(self.score, 3),
            "total": self.total,
            "last_seen": self.last_seen,
        }
