"""
Failure Handling and Recovery for Cooperative Tasks.

Provides failure classification, recovery strategy selection, exponential
backoff for retries, and a circuit breaker pattern to prevent cascading
failures when agents are consistently unresponsive.

When a cooperative task fails — due to timeout, transport error, execution
failure, refusal, or expiration — this module determines the appropriate
recovery action and manages retry state to prevent infinite retry loops.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Any, List, Optional


class FailureType(str, Enum):
    """Classification of cooperative task failure modes."""

    TIMEOUT = "timeout"
    """The target agent did not respond within the specified timeout."""

    TRANSPORT = "transport"
    """A transport-layer error occurred (git push/pull failure, network issue)."""

    EXECUTION = "execution"
    """The target agent responded with an error status (bad bytecode, handler crash)."""

    REFUSED = "refused"
    """The target agent explicitly refused the task."""

    EXPIRED = "expired"
    """The task expired before it could be processed."""


class RecoveryStrategy(str, Enum):
    """Actions to take when a cooperative task fails."""

    RETRY = "retry"
    """Retry the task (possibly with a different agent or after backoff)."""

    FALLBACK_LOCAL = "fallback_local"
    """Fall back to local execution instead of cooperative delegation."""

    ESCALATE = "escalate"
    """Escalate to a higher-level handler or human operator."""

    ABORT = "abort"
    """Give up and report the failure."""


class RecoveryError(Exception):
    """Raised when recovery itself fails or is impossible."""
    pass


@dataclass
class FailureRecord:
    """
    A record of a single task failure.

    Attributes:
        failure_type: The classification of the failure.
        error_message: Human-readable error description.
        target_agent: The agent that was targeted when the failure occurred.
        task_id: The ID of the failed task.
        occurred_at: ISO-8601 timestamp.
        attempt_number: Which retry attempt this was (1-based).
    """
    failure_type: FailureType
    error_message: str
    target_agent: str = ""
    task_id: str = ""
    occurred_at: str = ""
    attempt_number: int = 1

    def __post_init__(self):
        if not self.occurred_at:
            self.occurred_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "error_message": self.error_message,
            "target_agent": self.target_agent,
            "task_id": self.task_id,
            "occurred_at": self.occurred_at,
            "attempt_number": self.attempt_number,
        }


@dataclass
class RecoveryAction:
    """
    The outcome of failure analysis: what to do next.

    Attributes:
        strategy: The recovery strategy to apply.
        reason: Human-readable explanation for this choice.
        delay_ms: Delay before taking the action (for backoff).
        alternative_agent: Suggested alternative agent for retry, if any.
    """
    strategy: RecoveryStrategy
    reason: str
    delay_ms: int = 0
    alternative_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "reason": self.reason,
            "delay_ms": self.delay_ms,
            "alternative_agent": self.alternative_agent,
        }


class CircuitBreaker:
    """
    Circuit breaker to prevent repeated failures to the same agent.

    After `failure_threshold` consecutive failures to an agent, the
    circuit opens and further attempts are blocked for `reset_timeout_ms`
    milliseconds before transitioning to half-open (allowing a single
    probe request).

    States:
        CLOSED: Normal operation, requests pass through.
        OPEN:  Failures exceeded threshold, requests are blocked.
        HALF_OPEN: Probing to see if the agent has recovered.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 3, reset_timeout_ms: int = 60000):
        """
        Args:
            failure_threshold: Number of consecutive failures before opening.
            reset_timeout_ms: Milliseconds to wait before transitioning
                             from OPEN to HALF_OPEN.
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout_ms = reset_timeout_ms

        # Per-agent state
        self._consecutive_failures: Dict[str, int] = {}
        self._state: Dict[str, str] = {}  # agent -> state
        self._last_failure_time: Dict[str, float] = {}

    def _get_state(self, agent: str) -> str:
        """Get current circuit state, transitioning from OPEN to HALF_OPEN if due."""
        state = self._state.get(agent, self.CLOSED)
        if state == self.OPEN:
            last_fail = self._last_failure_time.get(agent, 0.0)
            elapsed_ms = (time.time() - last_fail) * 1000
            if elapsed_ms >= self.reset_timeout_ms:
                self._state[agent] = self.HALF_OPEN
                return self.HALF_OPEN
        return state

    def is_available(self, agent: str) -> bool:
        """
        Check whether requests to an agent are currently allowed.

        Returns:
            True if the circuit is CLOSED or HALF_OPEN, False if OPEN.
        """
        state = self._get_state(agent)
        return state in (self.CLOSED, self.HALF_OPEN)

    def record_success(self, agent: str) -> None:
        """Record a successful interaction; reset failure count."""
        self._consecutive_failures[agent] = 0
        self._state[agent] = self.CLOSED

    def record_failure(self, agent: str) -> None:
        """Record a failed interaction; potentially open the circuit."""
        self._consecutive_failures[agent] = self._consecutive_failures.get(agent, 0) + 1
        self._last_failure_time[agent] = time.time()

        if self._consecutive_failures[agent] >= self.failure_threshold:
            self._state[agent] = self.OPEN

    def get_state(self, agent: str) -> str:
        """Get the current circuit state for an agent."""
        return self._get_state(agent)

    def reset(self, agent: str) -> None:
        """Manually reset the circuit breaker for an agent."""
        self._consecutive_failures[agent] = 0
        self._state[agent] = self.CLOSED

    def get_failure_count(self, agent: str) -> int:
        """Get the current consecutive failure count for an agent."""
        return self._consecutive_failures.get(agent, 0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize circuit breaker state to a dictionary."""
        # Include all agents with any recorded failures or explicit state
        all_agents = set(self._state.keys()) | set(self._consecutive_failures.keys())
        return {
            agent: {
                "state": self._get_state(agent),
                "consecutive_failures": self._consecutive_failures.get(agent, 0),
                "last_failure_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(self._last_failure_time.get(agent, 0)),
                ),
            }
            for agent in sorted(all_agents)
        }


class FailureRecovery:
    """
    Central failure handler for cooperative tasks.

    Determines the appropriate recovery strategy when a task fails,
    manages exponential backoff for retries, and integrates with the
    circuit breaker to prevent cascading failures.

    Usage:
        recovery = FailureRecovery()
        action = recovery.analyze(
            failure_type=FailureType.TIMEOUT,
            error_message="Agent did not respond in 30s",
            target_agent="Oracle1",
            attempt_number=2,
        )
        if action.strategy == RecoveryStrategy.RETRY:
            time.sleep(action.delay_ms / 1000.0)
            # retry...
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_ms: int = 1000,
        max_delay_ms: int = 30000,
        backoff_multiplier: float = 2.0,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """
        Args:
            max_retries: Maximum number of retry attempts before giving up.
            base_delay_ms: Initial delay for exponential backoff in ms.
            max_delay_ms: Maximum delay cap for exponential backoff in ms.
            backoff_multiplier: Multiplier for each successive retry delay.
            circuit_breaker: Optional circuit breaker instance. Created
                           internally if not provided.
        """
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.backoff_multiplier = backoff_multiplier
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._failure_history: List[FailureRecord] = []

    def compute_backoff(self, attempt: int) -> int:
        """
        Compute the backoff delay for a given retry attempt.

        Uses exponential backoff: delay = base * multiplier^(attempt - 1),
        capped at max_delay_ms.

        Args:
            attempt: The retry attempt number (1-based).

        Returns:
            Delay in milliseconds before the next retry.
        """
        if attempt < 1:
            attempt = 1
        delay = self.base_delay_ms * (self.backoff_multiplier ** (attempt - 1))
        return min(int(delay), self.max_delay_ms)

    def analyze(
        self,
        failure_type: FailureType,
        error_message: str,
        target_agent: str = "",
        task_id: str = "",
        attempt_number: int = 1,
    ) -> RecoveryAction:
        """
        Analyze a failure and determine the recovery strategy.

        Considers:
        - The type of failure (some are not worth retrying)
        - Current retry count vs. max_retries
        - Circuit breaker state for the target agent
        - The nature of the error message

        Args:
            failure_type: Classification of the failure.
            error_message: Human-readable error description.
            target_agent: The agent that failed.
            task_id: The failed task's ID.
            attempt_number: Current attempt number (1-based).

        Returns:
            RecoveryAction describing what to do next.
        """
        # Record the failure
        record = FailureRecord(
            failure_type=failure_type,
            error_message=error_message,
            target_agent=target_agent,
            task_id=task_id,
            attempt_number=attempt_number,
        )
        self._failure_history.append(record)

        # Record in circuit breaker
        if target_agent:
            self.circuit_breaker.record_failure(target_agent)

        # Exhausted retries → abort or fallback
        if attempt_number >= self.max_retries:
            if failure_type in (FailureType.TIMEOUT, FailureType.TRANSPORT):
                return RecoveryAction(
                    strategy=RecoveryStrategy.FALLBACK_LOCAL,
                    reason=(
                        f"Max retries ({self.max_retries}) exceeded for "
                        f"{failure_type.value}; falling back to local execution"
                    ),
                )
            return RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                reason=(
                    f"Max retries ({self.max_retries}) exceeded for "
                    f"{failure_type.value}: {error_message}"
                ),
            )

        # Circuit breaker is open → don't retry this agent
        if target_agent and not self.circuit_breaker.is_available(target_agent):
            return RecoveryAction(
                strategy=RecoveryStrategy.ESCALATE,
                reason=(
                    f"Circuit breaker is open for '{target_agent}' after "
                    f"{self.circuit_breaker.get_failure_count(target_agent)} "
                    f"consecutive failures"
                ),
            )

        # Refused tasks should not be retried (agent explicitly said no)
        if failure_type == FailureType.REFUSED:
            return RecoveryAction(
                strategy=RecoveryStrategy.ESCALATE,
                reason=f"Agent '{target_agent}' refused the task: {error_message}",
            )

        # Expired tasks should not be retried
        if failure_type == FailureType.EXPIRED:
            return RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                reason=f"Task expired: {error_message}",
            )

        # Execution errors with certain patterns suggest permanent failure
        permanent_hints = ("not recognized", "unsupported", "invalid", "capability")
        if failure_type == FailureType.EXECUTION:
            msg_lower = error_message.lower()
            if any(hint in msg_lower for hint in permanent_hints):
                return RecoveryAction(
                    strategy=RecoveryStrategy.FALLBACK_LOCAL,
                    reason=(
                        f"Execution error appears permanent ({error_message}); "
                        f"falling back to local execution"
                    ),
                )

        # Default: retry with exponential backoff
        delay_ms = self.compute_backoff(attempt_number)
        return RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            reason=(
                f"Retryable {failure_type.value} on attempt {attempt_number}; "
                f"backing off {delay_ms}ms before retry {attempt_number + 1}"
            ),
            delay_ms=delay_ms,
        )

    def get_failure_history(self) -> List[Dict[str, Any]]:
        """Return the full failure history as a list of dicts."""
        return [r.to_dict() for r in self._failure_history]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate failure statistics."""
        type_counts: Dict[str, int] = {}
        agent_counts: Dict[str, int] = {}
        for record in self._failure_history:
            type_counts[record.failure_type.value] = (
                type_counts.get(record.failure_type.value, 0) + 1
            )
            if record.target_agent:
                agent_counts[record.target_agent] = (
                    agent_counts.get(record.target_agent, 0) + 1
                )

        return {
            "total_failures": len(self._failure_history),
            "by_type": type_counts,
            "by_agent": agent_counts,
            "circuit_breaker": self.circuit_breaker.to_dict(),
            "max_retries": self.max_retries,
            "current_backoff_base_ms": self.base_delay_ms,
        }

    def reset(self) -> None:
        """Reset all failure state and circuit breakers."""
        self._failure_history.clear()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.circuit_breaker.failure_threshold,
            reset_timeout_ms=self.circuit_breaker.reset_timeout_ms,
        )
