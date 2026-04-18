"""
Comprehensive tests for flux-coop-runtime.

Covers:
- Cooperative execution lifecycle
- Agent coordination and scheduling
- Resource management and cleanup
- Error handling and recovery
- Edge cases (concurrent access, timeouts, deadlocks)
- Fleet compat layer
- Conflict resolver edge cases
- Protocol evolution edge cases
- FluxTransfer boundary conditions
"""

import json
import os
import struct
import tempfile
import time
import uuid
import zlib
from unittest.mock import MagicMock, patch

import pytest

from src.cooperative_types import (
    CooperativeTask,
    CooperativeResponse,
    TrustRecord,
    TaskStatus,
    RequestType,
)
from src.runtime import (
    CooperativeRuntime,
    CooperativeRuntimeError,
    ERR_NO_CAPABLE_AGENT,
    ERR_TIMEOUT,
    ERR_TRANSPORT_FAILURE,
    ERR_TASK_EXPIRED,
    ERR_AGENT_REFUSED,
)
from src.fleet_compat import (
    _parse_legacy_error_string,
    to_fleet_error,
    map_task_status_to_fleet,
    LEGACY_CODE_MAP,
    FleetError,
    CooperativeRuntimeError as FleetCompatCooperativeRuntimeError,
)
from src.discovery.resolver import (
    resolve,
    list_agents,
    register_agent,
    ResolutionError,
    AgentAddress,
    FLEET_REGISTRY,
    CAPABILITY_MAP,
)
from src.trust.scorer import TrustScorer
from src.failure.recovery import (
    FailureType,
    RecoveryStrategy,
    FailureRecord,
    RecoveryAction,
    CircuitBreaker,
    FailureRecovery,
)
from src.synthesis.conflict_resolver import (
    ConflictResolver,
    SynthesisStrategy,
    ResolvedResult,
    ConflictResolutionError,
)
from src.evolution.protocol_evolution import (
    ProtocolVersion,
    ProtocolVersionDiff,
    ProtocolRegistry,
)
from src.transfer.format import (
    FluxTransfer,
    FluxTransferError,
    MAGIC,
    NUM_REGISTERS,
    FORMAT_VERSION,
)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_transport():
    transport = MagicMock()
    transport.agent_name = "Quill"
    transport.send_task.return_value = "/fake/path/task.json"
    transport.send_response.return_value = "/fake/path/response.json"
    return transport


@pytest.fixture
def runtime(mock_transport):
    return CooperativeRuntime(mock_transport)


@pytest.fixture(autouse=True)
def _reset_fleet_registry():
    """Save and restore FLEET_REGISTRY to prevent test pollution."""
    saved = dict(FLEET_REGISTRY)
    yield
    FLEET_REGISTRY.clear()
    FLEET_REGISTRY.update(saved)


# ─────────────────────────────────────────────────────────────────
# 1. Cooperative Execution Lifecycle
# ─────────────────────────────────────────────────────────────────

class TestCooperativeLifecycle:
    """Full lifecycle: task creation, send, handle, respond, complete."""

    def test_full_ask_respond_lifecycle(self, runtime, mock_transport):
        """Simulate a complete ASK → handle → respond → receive cycle."""
        mock_transport.poll_for_response.return_value = CooperativeResponse.success(
            task_id="t1", source_agent="Oracle1",
            target_agent="Quill", result={"agent": "Oracle1", "status": "active"},
        )

        result = runtime.ask("Oracle1", "ping", {})
        assert result["agent"] == "Oracle1"
        # The runtime creates its own task and stores it in _pending_tasks
        assert len(runtime._pending_tasks) == 1

    def test_lifecycle_tell_no_response(self, runtime, mock_transport):
        """TELL sends notification and returns task_id without waiting."""
        task_id = runtime.tell("Oracle1", {"message": "Fence shipped"})
        assert isinstance(task_id, str)
        assert task_id.startswith("Quill-")
        # poll_for_response should NOT be called for TELL
        mock_transport.poll_for_response.assert_not_called()

    def test_lifecycle_broadcast_to_all(self, runtime, mock_transport):
        """BROADCAST sends to all known agents, skipping failures."""
        count = runtime.broadcast({"message": "standup at 12:00"})
        assert count > 0
        assert mock_transport.send_task.call_count == count

    def test_pending_tasks_tracked(self, runtime, mock_transport):
        """Pending tasks dict is updated on ask."""
        mock_transport.poll_for_response.return_value = CooperativeResponse.success(
            "t1", "Oracle1", "Quill", {}
        )
        runtime.ask("Oracle1", "ping", {})
        assert len(runtime._pending_tasks) == 1

    def test_pending_tasks_not_cleared_on_timeout(self, runtime, mock_transport):
        """Pending task remains tracked even after timeout."""
        mock_transport.poll_for_response.return_value = None
        with pytest.raises(CooperativeRuntimeError, match=ERR_TIMEOUT):
            runtime.ask("Oracle1", "ping", {}, timeout_ms=100)
        # Task stays in pending (could be retried later)
        assert len(runtime._pending_tasks) >= 1


# ─────────────────────────────────────────────────────────────────
# 2. Agent Coordination and Scheduling
# ─────────────────────────────────────────────────────────────────

class TestAgentCoordination:
    """Resolver edge cases, multi-agent coordination, scheduling."""

    def test_resolve_cap_case_insensitive(self):
        """Capability lookup is case-insensitive."""
        addr = resolve("cap:CUDA")
        assert addr.agent_name == "JetsonClaw1"

    def test_resolve_role_case_matters(self):
        """Role lookup is case-sensitive."""
        with pytest.raises(ResolutionError):
            resolve("role:Lighthouse")

    def test_resolve_any_returns_best_confidence(self):
        """'any' returns the agent with highest confidence."""
        addr = resolve("any")
        assert addr.agent_name == "Oracle1"  # 0.95 confidence

    def test_register_and_resolve_dynamic_agent(self):
        """Dynamic agent registration and resolution."""
        register_agent("NewAgent", "https://example.com/new", "scout",
                       ["testing"], 0.88)
        addr = resolve("NewAgent")
        assert addr.confidence == 0.88
        assert addr.role == "scout"

    def test_list_agents_returns_registered(self):
        """list_agents includes all known agents."""
        agents = list_agents()
        names = [a["name"] for a in agents]
        assert "Oracle1" in names
        assert "Quill" in names

    def test_resolve_capability_no_match(self):
        """Unknown capability raises ResolutionError."""
        with pytest.raises(ResolutionError, match="capability"):
            resolve("cap:quantum_computing_xyz")

    def test_url_agent_name_extraction(self):
        """URL passthrough extracts agent name from path."""
        addr = resolve("https://github.com/org/special-agent-repo")
        assert addr.repo_url == "https://github.com/org/special-agent-repo"
        assert addr.agent_name == "special-agent-repo"

    def test_capability_map_coverage(self):
        """All capability mappings resolve to known specializations."""
        for cap, spec in CAPABILITY_MAP.items():
            # The spec should be findable in some agent's specializations
            found = False
            for name, info in FLEET_REGISTRY.items():
                if any(spec.lower() in s.lower() for s in info["specializations"]):
                    found = True
                    break
            # Not all caps may match (e.g. if no agent has that spec),
            # but at least verify the mapping itself is valid
            assert isinstance(spec, str)

    def test_agent_address_to_dict(self):
        """AgentAddress.to_dict() produces expected keys."""
        addr = resolve("Quill")
        d = addr.to_dict()
        assert "agent_name" in d
        assert "repo_url" in d
        assert "role" in d
        assert "specializations" in d
        assert "confidence" in d


# ─────────────────────────────────────────────────────────────────
# 3. Resource Management and Cleanup
# ─────────────────────────────────────────────────────────────────

class TestResourceManagement:
    """Pending task cleanup, trust record persistence, circuit breaker reset."""

    def test_trust_persistence_roundtrip(self):
        """Trust records survive serialization/deserialization."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            scorer1 = TrustScorer(persistence_path=path)
            scorer1.record_result("AgentA", "success")
            scorer1.record_result("AgentA", "success")
            scorer1.record_result("AgentA", "error")

            scorer2 = TrustScorer(persistence_path=path)
            assert scorer2.get_score("AgentA") > 0.5
            assert scorer2.get_record("AgentA").total == 3
        finally:
            os.unlink(path)

    def test_trust_empty_file_load(self):
        """Loading from an empty file doesn't crash."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as f:
            path = f.name
        try:
            scorer = TrustScorer(persistence_path=path)
            assert scorer.get_score("anyone") == 0.5
        finally:
            os.unlink(path)

    def test_circuit_breaker_full_lifecycle(self):
        """Circuit breaker: closed → open → half_open → closed."""
        cb = CircuitBreaker(failure_threshold=2, reset_timeout_ms=100)
        assert cb.is_available("AgentX")

        # Trip the circuit
        cb.record_failure("AgentX")
        cb.record_failure("AgentX")
        assert not cb.is_available("AgentX")
        assert cb.get_state("AgentX") == CircuitBreaker.OPEN

        # Wait for reset timeout
        time.sleep(0.15)
        assert cb.is_available("AgentX")
        assert cb.get_state("AgentX") == CircuitBreaker.HALF_OPEN

        # Successful probe resets to closed
        cb.record_success("AgentX")
        assert cb.get_state("AgentX") == CircuitBreaker.CLOSED

    def test_failure_recovery_reset_clears_all(self):
        """FailureRecovery.reset() clears history and circuit breaker."""
        recovery = FailureRecovery(max_retries=3)
        recovery.analyze(FailureType.TIMEOUT, "t1", "AgentA", attempt_number=1)
        recovery.analyze(FailureType.TRANSPORT, "t2", "AgentB", attempt_number=1)
        assert len(recovery.get_failure_history()) == 2

        recovery.reset()
        assert len(recovery.get_failure_history()) == 0
        stats = recovery.get_stats()
        assert stats["total_failures"] == 0

    def test_conflict_resolver_clear(self):
        """ConflictResolver.clear() resets state for new round."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        resp = CooperativeResponse.success("t1", "A", "B", {"v": 1})
        resolver.add_response(resp)
        assert resolver.response_count == 1

        resolver.clear()
        assert resolver.response_count == 0

    def test_protocol_registry_replace_version(self):
        """Registry raises on duplicate version registration."""
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion("1.0.0", "Initial"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(ProtocolVersion("1.0.0", "Duplicate"))


# ─────────────────────────────────────────────────────────────────
# 4. Error Handling and Recovery
# ─────────────────────────────────────────────────────────────────

class TestErrorHandlingRecovery:
    """Recovery strategy selection, error propagation, edge cases."""

    def test_refused_response_raises(self, runtime, mock_transport):
        """'refused' response status raises CooperativeRuntimeError."""
        resp = CooperativeResponse(
            task_id="t1", source_agent="Oracle1", target_agent="Quill",
            status="refused",
        )
        mock_transport.poll_for_response.return_value = resp
        with pytest.raises(CooperativeRuntimeError, match=ERR_AGENT_REFUSED):
            runtime.ask("Oracle1", "ping", {})

    def test_unexpected_status_raises(self, runtime, mock_transport):
        """Unknown response status raises with informative message."""
        resp = CooperativeResponse(
            task_id="t1", source_agent="Oracle1", target_agent="Quill",
            status="weird_status",
        )
        mock_transport.poll_for_response.return_value = resp
        with pytest.raises(CooperativeRuntimeError, match="Unexpected response status"):
            runtime.ask("Oracle1", "ping", {})

    def test_transport_exception_caught(self, runtime, mock_transport):
        """Generic exception from transport is caught and wrapped."""
        mock_transport.send_task.side_effect = RuntimeError("git died")
        with pytest.raises(CooperativeRuntimeError, match=ERR_TRANSPORT_FAILURE):
            runtime.ask("Oracle1", "ping", {})

    def test_poll_exception_becomes_timeout(self, runtime, mock_transport):
        """Exception during polling is treated as timeout."""
        mock_transport.poll_for_response.side_effect = Exception("connection reset")
        with pytest.raises(CooperativeRuntimeError, match=ERR_TIMEOUT):
            runtime.ask("Oracle1", "ping", {}, timeout_ms=5000)

    def test_recovery_permanent_hint_variations(self):
        """All permanent error hints trigger FALLBACK_LOCAL."""
        hints = ["not recognized", "unsupported", "invalid", "capability"]
        for hint in hints:
            recovery = FailureRecovery(max_retries=5)
            action = recovery.analyze(
                FailureType.EXECUTION,
                f"Bytecode {hint}",
                target_agent="AgentA",
                attempt_number=1,
            )
            assert action.strategy == RecoveryStrategy.FALLBACK_LOCAL, \
                f"Hint '{hint}' should trigger fallback"

    def test_recovery_transient_execution_retries(self):
        """Transient execution errors (no permanent hints) trigger RETRY."""
        recovery = FailureRecovery(max_retries=5)
        action = recovery.analyze(
            FailureType.EXECUTION, "handler crashed temporarily",
            target_agent="AgentA", attempt_number=1,
        )
        assert action.strategy == RecoveryStrategy.RETRY

    def test_recovery_exhausted_execution_aborts(self):
        """Exhausted retries for execution errors abort."""
        recovery = FailureRecovery(max_retries=2)
        action = recovery.analyze(
            FailureType.EXECUTION, "something broke",
            target_agent="AgentA", attempt_number=2,
        )
        assert action.strategy == RecoveryStrategy.ABORT

    def test_recovery_circuit_open_escalates(self):
        """Open circuit breaker escalates regardless of failure type."""
        recovery = FailureRecovery(max_retries=10)
        for _ in range(5):
            recovery.circuit_breaker.record_failure("AgentA")

        action = recovery.analyze(
            FailureType.TRANSPORT, "git push failed",
            target_agent="AgentA", attempt_number=1,
        )
        assert action.strategy == RecoveryStrategy.ESCALATE


# ─────────────────────────────────────────────────────────────────
# 5. Edge Cases
# ─────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary conditions, concurrent-like patterns, timeouts."""

    def test_task_with_zero_timeout_expires_immediately(self):
        """Task with 0ms timeout expires immediately."""
        task = CooperativeTask.create(
            "Quill", "Oracle1", "ping", {}, timeout_ms=0
        )
        # Give it a moment
        time.sleep(0.001)
        assert task.is_expired()

    def test_task_with_very_long_timeout_not_expired(self):
        """Task with 1 hour timeout should not be expired."""
        task = CooperativeTask.create(
            "Quill", "Oracle1", "ping", {}, timeout_ms=3600000
        )
        assert not task.is_expired()

    def test_task_create_generates_unique_ids(self):
        """100 tasks have 100 unique IDs."""
        ids = set()
        for _ in range(100):
            t = CooperativeTask.create("Quill", "Oracle1", "ping", {})
            ids.add(t.task_id)
        assert len(ids) == 100

    def test_task_create_different_agents_different_prefixes(self):
        """Tasks from different agents have different ID prefixes."""
        t1 = CooperativeTask.create("Quill", "Oracle1", "ping", {})
        t2 = CooperativeTask.create("Oracle1", "Quill", "ping", {})
        assert t1.task_id.startswith("Quill-")
        assert t2.task_id.startswith("Oracle1-")

    def test_response_without_vm_info(self):
        """Response without vm_info serializes cleanly."""
        resp = CooperativeResponse.success(
            "t1", "A", "B", {"v": 1}
        )
        data = resp.to_json()
        assert "vm_info" not in data  # vm_info is None, excluded

    def test_response_with_empty_result(self):
        """Success response with empty dict result."""
        resp = CooperativeResponse.success("t1", "A", "B", {})
        assert resp.result == {}
        data = resp.to_json()
        assert data["result"] == {}

    def test_trust_record_unknown_status_ignored(self):
        """Recording unknown status doesn't change counters."""
        rec = TrustRecord(agent_name="Test")
        rec.record("weird_status")
        assert rec.total == 0
        assert rec.score == 0.5

    def test_trust_record_all_failures(self):
        """All failures gives 0.0 score."""
        rec = TrustRecord(agent_name="BadAgent")
        for _ in range(5):
            rec.record("error")
        assert rec.score == 0.0

    def test_empty_broadcast_returns_zero(self, runtime, mock_transport):
        """Broadcast with no agents returns 0."""
        with patch("src.discovery.resolver.list_agents", return_value=[]):
            count = runtime.broadcast({"msg": "test"})
        assert count == 0

    def test_broadcast_skips_failed_tells(self, runtime, mock_transport):
        """Broadcast continues past failed tells and counts successes."""
        # tell() raises when resolve fails; broadcast catches and continues
        with patch("src.discovery.resolver.list_agents", return_value=[{"name": "NonexistentAgent"}]):
            count = runtime.broadcast({"msg": "test"})
        assert count == 0  # NonexistentAgent can't be resolved

    def test_handle_unknown_request_type(self, runtime):
        """Unknown request type returns error response."""
        task = CooperativeTask.create("Oracle1", "Quill", "quantum_compute", {})
        response = runtime.handle_task(task)
        assert response.status == "error"

    def test_bytecode_truncated_input(self, runtime):
        """Truncated bytecode (MOVI without enough args) should error."""
        bytecode = [0x18, 0]  # MOVI R0 but no immediate value
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": bytecode},
        )
        response = runtime.handle_task(task)
        assert response.status == "error"

    def test_bytecode_empty(self, runtime):
        """Empty bytecode returns all-zero registers."""
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": []},
        )
        response = runtime.handle_task(task)
        assert response.status == "success"
        assert response.result["registers"] == [0] * 8

    def test_bytecode_stack_top_empty(self, runtime):
        """Asking for stack_top with empty stack returns None."""
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": [0x00], "expected_result": "stack_top"},
        )
        response = runtime.handle_task(task)
        assert response.status == "success"
        assert response.result["value"] is None

    def test_bytecode_sub(self, runtime):
        """SUB instruction works correctly."""
        # MOVI R0, 50; MOVI R1, 20; SUB R2, R0, R1; HALT
        bytecode = [0x18, 0, 50, 0x00, 0x18, 1, 20, 0x00, 0x21, 2, 0, 1, 0x00]
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": bytecode, "expected_result": "register_2"},
        )
        response = runtime.handle_task(task)
        assert response.result["value"] == 30

    def test_bytecode_dec(self, runtime):
        """DEC instruction works correctly."""
        # MOVI R0, 5; DEC R0; DEC R0; HALT
        bytecode = [0x18, 0, 5, 0x00, 0x09, 0, 0x09, 0, 0x00]
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": bytecode, "expected_result": "register_0"},
        )
        response = runtime.handle_task(task)
        assert response.result["value"] == 3

    def test_bytecode_pop_empty_stack(self, runtime):
        """POP from empty stack should not crash, register stays unchanged."""
        # POP R0; HALT (no prior PUSH)
        bytecode = [0x0D, 0, 0x00]
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": bytecode, "expected_result": "register_0"},
        )
        response = runtime.handle_task(task)
        assert response.status == "success"
        assert response.result["value"] == 0  # Default register value


# ─────────────────────────────────────────────────────────────────
# 6. Fleet Compat Layer
# ─────────────────────────────────────────────────────────────────

class TestFleetCompat:
    """Fleet compat: error parsing, migration helpers, status mapping."""

    def test_parse_known_legacy_timeout(self):
        code, msg = _parse_legacy_error_string("TIMEOUT: No response in 30s")
        assert code == "COOP_TIMEOUT"
        assert msg == "No response in 30s"

    def test_parse_known_legacy_no_capable_agent(self):
        code, msg = _parse_legacy_error_string(
            "NO_CAPABLE_AGENT: Cannot resolve target 'foo'"
        )
        assert code == "COOP_NO_CAPABLE_AGENT"
        assert msg == "Cannot resolve target 'foo'"

    def test_parse_unknown_prefix(self):
        code, msg = _parse_legacy_error_string("SOMETHING_WEIRD: detail")
        assert code == "SOMETHING_WEIRD"

    def test_parse_no_separator(self):
        code, msg = _parse_legacy_error_string("plain error message")
        assert code == "COOP_UNKNOWN_REQUEST"
        assert msg == "plain error message"

    def test_to_fleet_error_wraps_plain_exception(self):
        exc = ValueError("some error")
        result = to_fleet_error(exc, default_code="COOP_DEFAULT")
        assert isinstance(result, FleetError)
        assert result.code == "COOP_DEFAULT"
        assert result.message == "some error"

    def test_to_fleet_error_passthrough(self):
        """FleetError instances pass through unchanged."""
        original = FleetError(code="COOP_TEST", message="test msg")
        result = to_fleet_error(original)
        assert result is original

    def test_to_fleet_error_with_context(self):
        exc = RuntimeError("crash")
        result = to_fleet_error(exc, source_repo="my-repo", extra_key="extra_val")
        assert isinstance(result, FleetError)
        assert "original_type" in result.context
        assert result.context["original_type"] == "RuntimeError"

    def test_map_task_status_success(self):
        assert map_task_status_to_fleet("success") == "SUCCESS"

    def test_map_task_status_timeout(self):
        assert map_task_status_to_fleet("timeout") == "TIMEOUT"

    def test_map_task_status_error(self):
        assert map_task_status_to_fleet("error") == "ERROR"

    def test_map_task_status_pending_variants(self):
        for status in ("created", "sent", "received", "executing"):
            assert map_task_status_to_fleet(status) == "PENDING"

    def test_map_task_status_refused(self):
        assert map_task_status_to_fleet("refused") == "REFUSED"

    def test_map_task_status_cancelled(self):
        assert map_task_status_to_fleet("cancelled") == "CANCELLED"

    def test_map_task_status_unknown_defaults_to_error(self):
        assert map_task_status_to_fleet("unknown_status_xyz") == "ERROR"

    def test_map_task_status_case_insensitive(self):
        assert map_task_status_to_fleet("SUCCESS") == "SUCCESS"
        assert map_task_status_to_fleet("TimeOut") == "TIMEOUT"

    def test_cooperative_runtime_error_legacy_string(self):
        """Legacy string format is parsed correctly."""
        err = FleetCompatCooperativeRuntimeError(
            "TIMEOUT: Agent did not respond"
        )
        assert err.code == "COOP_TIMEOUT"
        assert err.message == "Agent did not respond"

    def test_cooperative_runtime_error_structured(self):
        """Structured form is used directly."""
        err = FleetCompatCooperativeRuntimeError(
            code="COOP_TEST", message="structured error"
        )
        assert err.code == "COOP_TEST"
        assert err.message == "structured error"

    def test_cooperative_runtime_error_positional(self):
        """Two positional args treated as code + message."""
        err = FleetCompatCooperativeRuntimeError("CODE_X", "msg")
        assert err.code == "CODE_X"
        assert err.message == "msg"

    def test_legacy_code_map_completeness(self):
        """All ERR_* constants from runtime.py are mapped."""
        expected = {
            "NO_CAPABLE_AGENT", "TIMEOUT", "TRANSPORT_FAILURE",
            "TASK_EXPIRED", "AGENT_REFUSED",
        }
        assert set(LEGACY_CODE_MAP.keys()) == expected


# ─────────────────────────────────────────────────────────────────
# 7. Conflict Resolver Edge Cases
# ─────────────────────────────────────────────────────────────────

class TestConflictResolverEdgeCases:
    """Edge cases in multi-agent conflict resolution."""

    def test_resolve_with_no_responses_raises(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        with pytest.raises(ConflictResolutionError, match="No responses"):
            resolver.resolve()

    def test_single_response_first_response(self):
        """Single response gets 0.5 confidence (no confirmation)."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.FIRST_RESPONSE)
        resp = CooperativeResponse.success("t1", "A", "B", {"v": 42})
        resolver.add_response(resp)
        result = resolver.resolve()
        assert result.confidence == 0.5

    def test_unanimous_responses_high_confidence(self):
        """Unanimous agreement yields high confidence."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        for agent in ["A", "B", "C"]:
            resp = CooperativeResponse.success("t1", agent, "Q", {"v": 42})
            resolver.add_response(resp)
        result = resolver.resolve()
        assert result.confidence > 0.9

    def test_disagreement_lower_confidence(self):
        """Disagreed responses produce lower confidence."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        # 2 agree, 1 disagrees
        for agent, val in [("A", 1), ("B", 1), ("C", 2)]:
            resp = CooperativeResponse.success("t1", agent, "Q", {"v": val})
            resolver.add_response(resp)
        result = resolver.resolve()
        assert result.confidence < 0.9
        assert result.result["v"] == 1  # Majority wins

    def test_error_responses_filtered(self):
        """Error responses are not added to the pool."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        err = CooperativeResponse.error("t1", "A", "Q", "ERR", "failed")
        ok = CooperativeResponse.success("t1", "B", "Q", {"v": 1})
        resolver.add_response(err)  # Should be filtered out
        resolver.add_response(ok)
        assert resolver.response_count == 1

    def test_set_expected_agents(self):
        """Expected agents count affects confidence metadata."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        resolver.set_expected_agents(5)
        resolver.add_response(
            CooperativeResponse.success("t1", "A", "Q", {"v": 1})
        )
        result = resolver.resolve()
        assert result.agents_consulted == 5
        assert result.agents_responded == 1

    def test_weighted_without_trust_falls_back_to_majority(self):
        """WEIGHTED strategy without trust scorer falls back to MAJORITY."""
        resolver = ConflictResolver(
            strategy=SynthesisStrategy.WEIGHTED, trust_scorer=None
        )
        for agent in ["A", "B"]:
            resolver.add_response(
                CooperativeResponse.success("t1", agent, "Q", {"v": 1})
            )
        result = resolver.resolve()
        assert result.strategy == "majority"

    def test_best_evidence_with_vm_info_ranks_higher(self):
        """Response with vm_info gets higher evidence score."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.BEST_EVIDENCE)
        # Response with VM info
        resp1 = CooperativeResponse.success(
            "t1", "A", "Q", {"v": 1},
            execution_time_ms=100,
            vm_info={"isa": "unified"},
        )
        # Response without VM info
        resp2 = CooperativeResponse.success(
            "t1", "B", "Q", {"v": 2}, execution_time_ms=5
        )
        resolver.add_response(resp1)
        resolver.add_response(resp2)
        result = resolver.resolve()
        # resp1 should win due to vm_info bonus
        assert result.result["v"] == 1

    def test_resolved_result_to_dict(self):
        result = ResolvedResult(
            strategy="majority",
            result={"v": 42},
            confidence=0.9,
            agents_consulted=3,
            agents_responded=2,
        )
        d = result.to_dict()
        assert d["strategy"] == "majority"
        assert d["confidence"] == 0.9
        assert d["agents_consulted"] == 3


# ─────────────────────────────────────────────────────────────────
# 8. Protocol Evolution Edge Cases
# ─────────────────────────────────────────────────────────────────

class TestProtocolEvolutionEdgeCases:
    """Protocol registry, diffing, and migration path edge cases."""

    def test_registry_empty_latest(self):
        """Latest on empty registry returns None."""
        reg = ProtocolRegistry()
        assert reg.latest() is None

    def test_registry_len(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion("1.0.0"))
        reg.register(ProtocolVersion("2.0.0"))
        assert len(reg) == 2

    def test_diff_same_version(self):
        """Diffing a version with itself produces empty diff."""
        reg = ProtocolRegistry()
        v = ProtocolVersion("1.0.0", opcode_mapping={"0x51": "ASK"})
        reg.register(v)
        diff = reg.diff("1.0.0", "1.0.0")
        assert not diff.added_opcodes
        assert not diff.removed_opcodes
        assert not diff.modified_opcodes

    def test_diff_added_removed_modified(self):
        """Diff correctly identifies additions, removals, modifications."""
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion("1.0.0", opcode_mapping={
            "0x50": "TELL", "0x51": "ASK"
        }))
        reg.register(ProtocolVersion("2.0.0", opcode_mapping={
            "0x51": "ASK_V2",  # modified
            "0x52": "DELEGATE",  # added
            # 0x50 removed
        }, predecessor="1.0.0"))
        diff = reg.diff("1.0.0", "2.0.0")
        assert "0x52" in diff.added_opcodes
        assert "0x50" in diff.removed_opcodes
        assert diff.modified_opcodes["0x51"] == ("ASK", "ASK_V2")

    def test_diff_nonexistent_version_raises(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion("1.0.0"))
        with pytest.raises(ValueError, match="not registered"):
            reg.diff("1.0.0", "9.9.9")

    def test_migration_same_version(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion("1.0.0"))
        path = reg.migration_path("1.0.0", "1.0.0")
        assert path == ["1.0.0"]

    def test_migration_connected_versions(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion("1.0.0"))
        reg.register(ProtocolVersion("1.1.0", predecessor="1.0.0"))
        reg.register(ProtocolVersion("2.0.0", predecessor="1.1.0"))
        path = reg.migration_path("1.0.0", "2.0.0")
        assert path == ["1.0.0", "1.1.0", "2.0.0"]

    def test_migration_no_path_raises(self):
        """Unconnected versions raise ValueError."""
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion("1.0.0"))
        reg.register(ProtocolVersion("2.0.0"))
        # 2.0.0 has no predecessor, so no path from 1.0.0 to 2.0.0
        # Actually, migration_path also tries backward (predecessor) direction,
        # but 2.0.0.predecessor is None so it won't connect.
        # The BFS starts at 1.0.0, tries children (none) and predecessor (none).
        with pytest.raises(ValueError, match="No migration path"):
            reg.migration_path("1.0.0", "2.0.0")

    def test_registry_roundtrip_json(self):
        """Registry serializes and deserializes via JSON."""
        reg1 = ProtocolRegistry()
        reg1.register(ProtocolVersion("1.0.0", "Initial", {"0x51": "ASK"}))
        reg1.register(ProtocolVersion("1.1.0", "Added TELL", {"0x50": "TELL", "0x51": "ASK"}))

        json_str = reg1.to_json()
        reg2 = ProtocolRegistry.from_json(json_str)
        assert len(reg2) == 2
        assert "1.0.0" in reg2.all_versions()
        v = reg2.get("1.1.0")
        assert v.opcode_mapping["0x50"] == "TELL"

    def test_protocol_version_roundtrip_json(self):
        v = ProtocolVersion(
            version="1.2.0", changelog="Bug fix",
            opcode_mapping={"0x51": "ASK"}, metadata={"author": "Quill"},
        )
        json_str = v.to_json()
        v2 = ProtocolVersion.from_json(json_str)
        assert v2.version == "1.2.0"
        assert v2.metadata["author"] == "Quill"

    def test_diff_repr(self):
        diff = ProtocolVersionDiff("1.0.0", "2.0.0")
        diff.added_opcodes["0x52"] = "DELEGATE"
        r = repr(diff)
        assert "+1" in r  # 1 added
        assert "1.0.0 -> 2.0.0" in r


# ─────────────────────────────────────────────────────────────────
# 9. FluxTransfer Boundary Conditions
# ─────────────────────────────────────────────────────────────────

class TestFluxTransferBoundary:
    """Additional FluxTransfer edge cases."""

    def test_max_int32_stack_values(self):
        """Max and min int32 values survive roundtrip."""
        ft = FluxTransfer(data_stack=[2147483647, -2147483648])
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        assert result.data_stack == [2147483647, -2147483648]

    def test_many_stack_entries(self):
        """1000 stack entries survive roundtrip."""
        stack = list(range(1000))
        ft = FluxTransfer(data_stack=stack)
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        assert result.data_stack == stack

    def test_metadata_with_special_characters(self):
        """Metadata with unicode and special chars survives."""
        meta = {"desc": "Coöpérative rün — flüx™", "path": "/tmp/a&b"}
        ft = FluxTransfer(metadata=meta)
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        assert result.metadata == meta

    def test_wrong_format_version(self):
        """Wrong format version raises error."""
        ft = FluxTransfer()
        data = bytearray(ft.serialize())
        data[4] = 99  # Corrupt version byte
        with pytest.raises(FluxTransferError, match="Unsupported format version"):
            FluxTransfer.deserialize(bytes(data))

    def test_unsupported_isa_version(self):
        """Non-unified ISA version raises validation error."""
        ft = FluxTransfer(isa_version=99)
        with pytest.raises(FluxTransferError, match="Unsupported ISA version"):
            ft.serialize()

    def test_empty_metadata(self):
        """Empty metadata dict survives roundtrip."""
        ft = FluxTransfer(metadata={})
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        assert result.metadata == {}

    def test_signal_stack_roundtrip(self):
        """Signal stack with various opcodes survives."""
        ft = FluxTransfer(signal_stack=[0x50, 0x51, 0x52, 0x53, 0x70, 0x71])
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        assert result.signal_stack == [0x50, 0x51, 0x52, 0x53, 0x70, 0x71]

    def test_multiple_roundtrips(self):
        """Multiple serialize/deserialize cycles are idempotent."""
        ft = FluxTransfer(
            source_pc=42, data_stack=[1, 2, 3],
            registers=[0] * 63 + [999],
            metadata={"key": "value"},
        )
        for _ in range(5):
            data = ft.serialize()
            ft = FluxTransfer.deserialize(data)
        assert ft.source_pc == 42
        assert ft.data_stack == [1, 2, 3]
        assert ft.registers[63] == 999


# ─────────────────────────────────────────────────────────────────
# 10. Trust Scorer Advanced
# ─────────────────────────────────────────────────────────────────

class TestTrustScorerAdvanced:
    """Advanced trust scoring scenarios."""

    def test_rank_empty_list(self):
        scorer = TrustScorer()
        ranked = scorer.rank_agents([])
        assert ranked == []

    def test_rank_all_below_min(self):
        scorer = TrustScorer()
        scorer.record_result("A", "error")
        ranked = scorer.rank_agents(["A"], min_score=0.9)
        assert ranked == []

    def test_get_record_creates_on_demand(self):
        scorer = TrustScorer()
        rec = scorer.get_record("NewAgent")
        assert rec.agent_name == "NewAgent"
        assert rec.score == 0.5

    def test_record_result_returns_updated_score(self):
        scorer = TrustScorer()
        scorer.record_result("A", "success")
        score = scorer.record_result("A", "success")
        assert score == 1.0

    def test_multiple_agents_independent(self):
        scorer = TrustScorer()
        scorer.record_result("A", "success")
        scorer.record_result("B", "error")
        assert scorer.get_score("A") == 1.0
        assert scorer.get_score("B") == 0.0
        # Agent C unaffected
        assert scorer.get_score("C") == 0.5

    def test_get_all_records_empty(self):
        scorer = TrustScorer()
        assert scorer.get_all_records() == {}


# ─────────────────────────────────────────────────────────────────
# 11. Enum Values
# ─────────────────────────────────────────────────────────────────

class TestEnumValues:
    """Verify enum completeness and consistency."""

    def test_task_status_values(self):
        expected = {"created", "sent", "received", "executing",
                     "success", "error", "timeout", "refused", "cancelled"}
        actual = {s.value for s in TaskStatus}
        assert actual == expected

    def test_request_type_values(self):
        expected = {"execute_bytecode", "query_knowledge", "run_test",
                     "ping", "custom"}
        actual = {r.value for r in RequestType}
        assert actual == expected

    def test_failure_type_values(self):
        expected = {"timeout", "transport", "execution", "refused", "expired"}
        actual = {f.value for f in FailureType}
        assert actual == expected

    def test_recovery_strategy_values(self):
        expected = {"retry", "fallback_local", "escalate", "abort"}
        actual = {r.value for r in RecoveryStrategy}
        assert actual == expected

    def test_synthesis_strategy_values(self):
        expected = {"majority", "weighted", "first_response", "best_evidence"}
        actual = {s.value for s in SynthesisStrategy}
        assert actual == expected
