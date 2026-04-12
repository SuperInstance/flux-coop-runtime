"""
Unit tests for failure/recovery.py.

Covers exponential backoff computation, circuit breaker pattern
(open/close/half-open transitions), recovery strategy selection,
and failure history tracking.
"""

import time
import unittest

from src.failure.recovery import (
    FailureType,
    RecoveryStrategy,
    FailureRecord,
    RecoveryAction,
    CircuitBreaker,
    FailureRecovery,
)


class TestExponentialBackoff(unittest.TestCase):
    """Test exponential backoff delay computation."""

    def test_base_delay_on_first_attempt(self):
        recovery = FailureRecovery(
            max_retries=5,
            base_delay_ms=1000,
            backoff_multiplier=2.0,
        )
        delay = recovery.compute_backoff(1)
        self.assertEqual(delay, 1000)

    def test_doubles_on_each_attempt(self):
        recovery = FailureRecovery(
            max_retries=5,
            base_delay_ms=1000,
            backoff_multiplier=2.0,
        )
        self.assertEqual(recovery.compute_backoff(1), 1000)
        self.assertEqual(recovery.compute_backoff(2), 2000)
        self.assertEqual(recovery.compute_backoff(3), 4000)
        self.assertEqual(recovery.compute_backoff(4), 8000)

    def test_cap_at_max_delay(self):
        recovery = FailureRecovery(
            max_retries=10,
            base_delay_ms=1000,
            max_delay_ms=5000,
            backoff_multiplier=2.0,
        )
        # 1->1000, 2->2000, 3->4000, 4->8000 but capped at 5000
        self.assertEqual(recovery.compute_backoff(4), 5000)
        self.assertEqual(recovery.compute_backoff(10), 5000)

    def test_custom_multiplier(self):
        recovery = FailureRecovery(
            max_retries=5,
            base_delay_ms=500,
            backoff_multiplier=3.0,
        )
        self.assertEqual(recovery.compute_backoff(1), 500)
        self.assertEqual(recovery.compute_backoff(2), 1500)
        self.assertEqual(recovery.compute_backoff(3), 4500)

    def test_clamp_minimum_attempt(self):
        recovery = FailureRecovery(base_delay_ms=1000, backoff_multiplier=2.0)
        # Even attempt 0 should behave like attempt 1
        delay = recovery.compute_backoff(0)
        self.assertEqual(delay, 1000)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker state transitions."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        self.assertTrue(cb.is_available("AgentA"))
        self.assertEqual(cb.get_state("AgentA"), CircuitBreaker.CLOSED)

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("AgentA")
        cb.record_failure("AgentA")
        self.assertTrue(cb.is_available("AgentA"))
        self.assertEqual(cb.get_state("AgentA"), CircuitBreaker.CLOSED)

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("AgentA")
        cb.record_failure("AgentA")
        cb.record_failure("AgentA")
        self.assertFalse(cb.is_available("AgentA"))
        self.assertEqual(cb.get_state("AgentA"), CircuitBreaker.OPEN)

    def test_opens_above_threshold(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("AgentA")
        cb.record_failure("AgentA")
        cb.record_failure("AgentA")
        self.assertFalse(cb.is_available("AgentA"))

    def test_success_resets_circuit(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("AgentA")
        cb.record_failure("AgentA")
        self.assertFalse(cb.is_available("AgentA"))
        cb.record_success("AgentA")
        self.assertTrue(cb.is_available("AgentA"))
        self.assertEqual(cb.get_state("AgentA"), CircuitBreaker.CLOSED)

    def test_manual_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure("AgentA")
        self.assertFalse(cb.is_available("AgentA"))
        cb.reset("AgentA")
        self.assertTrue(cb.is_available("AgentA"))

    def test_independent_per_agent(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("AgentA")
        cb.record_failure("AgentA")
        self.assertFalse(cb.is_available("AgentA"))
        # AgentB should still be available
        self.assertTrue(cb.is_available("AgentB"))

    def test_half_open_after_reset_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout_ms=100)
        cb.record_failure("AgentA")
        cb.record_failure("AgentA")
        self.assertFalse(cb.is_available("AgentA"))

        # Wait for reset timeout
        time.sleep(0.15)
        # Should transition to HALF_OPEN
        self.assertTrue(cb.is_available("AgentA"))
        self.assertEqual(cb.get_state("AgentA"), CircuitBreaker.HALF_OPEN)

    def test_failure_count(self):
        cb = CircuitBreaker(failure_threshold=5)
        self.assertEqual(cb.get_failure_count("AgentA"), 0)
        cb.record_failure("AgentA")
        cb.record_failure("AgentA")
        self.assertEqual(cb.get_failure_count("AgentA"), 2)

    def test_to_dict(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("AgentA")
        d = cb.to_dict()
        self.assertIn("AgentA", d)
        self.assertEqual(d["AgentA"]["consecutive_failures"], 1)
        self.assertEqual(d["AgentA"]["state"], "closed")


class TestRecoveryStrategySelection(unittest.TestCase):
    """Test that FailureRecovery.analyze() selects the correct strategy."""

    def test_timeout_on_first_attempt_retries(self):
        recovery = FailureRecovery(max_retries=3)
        action = recovery.analyze(
            failure_type=FailureType.TIMEOUT,
            error_message="No response in 30s",
            target_agent="Oracle1",
            attempt_number=1,
        )
        self.assertEqual(action.strategy, RecoveryStrategy.RETRY)
        self.assertGreater(action.delay_ms, 0)

    def test_timeout_exhausted_falls_back_to_local(self):
        recovery = FailureRecovery(max_retries=3)
        action = recovery.analyze(
            failure_type=FailureType.TIMEOUT,
            error_message="No response",
            target_agent="Oracle1",
            attempt_number=3,
        )
        self.assertEqual(action.strategy, RecoveryStrategy.FALLBACK_LOCAL)

    def test_transport_exhausted_falls_back_to_local(self):
        recovery = FailureRecovery(max_retries=2)
        action = recovery.analyze(
            failure_type=FailureType.TRANSPORT,
            error_message="git push failed",
            target_agent="Oracle1",
            attempt_number=2,
        )
        self.assertEqual(action.strategy, RecoveryStrategy.FALLBACK_LOCAL)

    def test_refused_always_escalates(self):
        recovery = FailureRecovery(max_retries=5)
        action = recovery.analyze(
            failure_type=FailureType.REFUSED,
            error_message="Agent declined",
            target_agent="Oracle1",
            attempt_number=1,
        )
        self.assertEqual(action.strategy, RecoveryStrategy.ESCALATE)

    def test_expired_always_aborts(self):
        recovery = FailureRecovery(max_retries=5)
        action = recovery.analyze(
            failure_type=FailureType.EXPIRED,
            error_message="Task too old",
            target_agent="Oracle1",
            attempt_number=1,
        )
        self.assertEqual(action.strategy, RecoveryStrategy.ABORT)

    def test_permanent_execution_error_falls_back(self):
        recovery = FailureRecovery(max_retries=5)
        action = recovery.analyze(
            failure_type=FailureType.EXECUTION,
            error_message="Opcode 0x99 not recognized in unified ISA",
            target_agent="Oracle1",
            attempt_number=1,
        )
        self.assertEqual(action.strategy, RecoveryStrategy.FALLBACK_LOCAL)

    def test_transient_execution_error_retries(self):
        recovery = FailureRecovery(max_retries=3)
        action = recovery.analyze(
            failure_type=FailureType.EXECUTION,
            error_message="Temporary internal error",
            target_agent="Oracle1",
            attempt_number=1,
        )
        self.assertEqual(action.strategy, RecoveryStrategy.RETRY)

    def test_circuit_open_blocks_retry(self):
        recovery = FailureRecovery(max_retries=10)
        # Trip the circuit breaker
        for _ in range(3):
            recovery.circuit_breaker.record_failure("Oracle1")
        self.assertFalse(recovery.circuit_breaker.is_available("Oracle1"))

        action = recovery.analyze(
            failure_type=FailureType.TIMEOUT,
            error_message="timeout",
            target_agent="Oracle1",
            attempt_number=1,
        )
        self.assertEqual(action.strategy, RecoveryStrategy.ESCALATE)
        self.assertIn("circuit breaker", action.reason.lower())

    def test_execution_error_exhausted_aborts(self):
        recovery = FailureRecovery(max_retries=2)
        action = recovery.analyze(
            failure_type=FailureType.EXECUTION,
            error_message="handler crashed",
            target_agent="Oracle1",
            attempt_number=2,
        )
        self.assertEqual(action.strategy, RecoveryStrategy.ABORT)


class TestFailureRecoveryStats(unittest.TestCase):
    """Test failure history and statistics tracking."""

    def test_failure_history_records(self):
        recovery = FailureRecovery()
        recovery.analyze(
            failure_type=FailureType.TIMEOUT,
            error_message="first timeout",
            target_agent="A",
            attempt_number=1,
        )
        recovery.analyze(
            failure_type=FailureType.TRANSPORT,
            error_message="push failed",
            target_agent="B",
            attempt_number=1,
        )
        history = recovery.get_failure_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["failure_type"], "timeout")
        self.assertEqual(history[1]["failure_type"], "transport")

    def test_stats_aggregation(self):
        recovery = FailureRecovery()
        recovery.analyze(FailureType.TIMEOUT, "t1", "A", attempt_number=1)
        recovery.analyze(FailureType.TIMEOUT, "t2", "A", attempt_number=2)
        recovery.analyze(FailureType.EXECUTION, "e1", "B", attempt_number=1)

        stats = recovery.get_stats()
        self.assertEqual(stats["total_failures"], 3)
        self.assertEqual(stats["by_type"]["timeout"], 2)
        self.assertEqual(stats["by_type"]["execution"], 1)
        self.assertEqual(stats["by_agent"]["A"], 2)
        self.assertEqual(stats["by_agent"]["B"], 1)

    def test_reset_clears_history(self):
        recovery = FailureRecovery()
        recovery.analyze(FailureType.TIMEOUT, "t", "A", attempt_number=1)
        self.assertEqual(len(recovery.get_failure_history()), 1)
        recovery.reset()
        self.assertEqual(len(recovery.get_failure_history()), 0)


class TestRecoveryAction(unittest.TestCase):
    """Test RecoveryAction dataclass."""

    def test_to_dict(self):
        action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            reason="Backing off before retry",
            delay_ms=2000,
            alternative_agent="BackupAgent",
        )
        d = action.to_dict()
        self.assertEqual(d["strategy"], "retry")
        self.assertEqual(d["delay_ms"], 2000)
        self.assertEqual(d["alternative_agent"], "BackupAgent")

    def test_default_values(self):
        action = RecoveryAction(
            strategy=RecoveryStrategy.ABORT,
            reason="Done",
        )
        self.assertEqual(action.delay_ms, 0)
        self.assertIsNone(action.alternative_agent)


class TestFailureRecord(unittest.TestCase):
    """Test FailureRecord dataclass."""

    def test_auto_timestamp(self):
        record = FailureRecord(
            failure_type=FailureType.TIMEOUT,
            error_message="Timed out",
        )
        self.assertTrue(record.occurred_at)

    def test_to_dict(self):
        record = FailureRecord(
            failure_type=FailureType.TRANSPORT,
            error_message="git push failed",
            target_agent="Oracle1",
            task_id="task-001",
            attempt_number=2,
        )
        d = record.to_dict()
        self.assertEqual(d["failure_type"], "transport")
        self.assertEqual(d["target_agent"], "Oracle1")
        self.assertEqual(d["attempt_number"], 2)


if __name__ == "__main__":
    unittest.main()
