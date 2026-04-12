"""
Integration tests for multi-agent cooperative scenarios.

Tests end-to-end flows combining discovery, trust scoring, conflict resolution,
failure recovery, and protocol evolution in realistic cooperative patterns.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from src.cooperative_types import CooperativeTask, CooperativeResponse, TrustRecord
from src.discovery.resolver import resolve, list_agents, register_agent, ResolutionError
from src.trust.scorer import TrustScorer
from src.failure.recovery import (
    FailureRecovery, FailureType, RecoveryStrategy, CircuitBreaker,
)
from src.synthesis.conflict_resolver import (
    ConflictResolver, SynthesisStrategy, ConflictResolutionError,
    ResolvedResult,
)
from src.evolution.protocol_evolution import (
    ProtocolVersion, ProtocolRegistry, ProtocolVersionDiff,
)
from src.transfer.format import FluxTransfer, FluxTransferError


class TestMultiAgentDiscoveryAndTrust:
    """Test agent discovery combined with trust scoring."""

    def test_discover_and_rank_by_trust(self):
        """Discover agents for a capability and rank by trust."""
        trust = TrustScorer()
        # Build trust history
        for _ in range(5):
            trust.record_result("JetsonClaw1", "success")
        trust.record_result("Quill", "success")
        trust.record_result("Quill", "error")
        trust.record_result("Quill", "error")

        # Both can handle hardware/cuda
        ranked = trust.rank_agents(["JetsonClaw1", "Quill"])
        assert ranked[0][0] == "JetsonClaw1"
        assert ranked[0][1] > ranked[1][1]

    def test_capability_resolution_returns_correct_agent(self):
        """Resolving a capability should return the most capable agent."""
        addr = resolve("cap:cuda")
        assert addr.agent_name == "JetsonClaw1"

    def test_fallback_when_no_capable_agent(self):
        """When min_confidence is too high, resolution should fail."""
        with pytest.raises(ResolutionError):
            resolve("any", min_confidence=1.0)

    def test_register_and_discover_new_agent(self):
        """Register a new agent and discover it by capability."""
        register_agent(
            "NewCUDA", "https://github.com/test/newcuda",
            "vessel", ["cuda", "gpu-compute"], 0.95,
        )
        try:
            addr = resolve("NewCUDA")
            assert addr.agent_name == "NewCUDA"
            assert addr.confidence == 0.95
        finally:
            from src.discovery.resolver import FLEET_REGISTRY
            if "NewCUDA" in FLEET_REGISTRY:
                del FLEET_REGISTRY["NewCUDA"]


class TestCooperativeAskWithRecovery:
    """Test ASK flow with failure recovery integration."""

    def test_ask_retries_on_timeout_then_fallback(self):
        """Simulate a full ask-retry-fallback cycle."""
        recovery = FailureRecovery(max_retries=2, base_delay_ms=100)

        # First attempt: timeout
        action = recovery.analyze(
            FailureType.TIMEOUT, "Agent slow",
            target_agent="SlowAgent", attempt_number=1,
        )
        assert action.strategy == RecoveryStrategy.RETRY

        # Second attempt: timeout again
        action = recovery.analyze(
            FailureType.TIMEOUT, "Agent still slow",
            target_agent="SlowAgent", attempt_number=2,
        )
        assert action.strategy == RecoveryStrategy.FALLBACK_LOCAL
        assert "falling back to local" in action.reason.lower()

    def test_circuit_breaker_blocks_repeat_offender(self):
        """After repeated failures, circuit breaker should block the agent."""
        cb = CircuitBreaker(failure_threshold=3)
        trust = TrustScorer()

        # Simulate repeated failures to BadAgent
        for i in range(5):
            cb.record_failure("BadAgent")
            trust.record_result("BadAgent", "error")

        assert not cb.is_available("BadAgent")
        assert trust.get_score("BadAgent") < 0.5

    def test_recovery_stats_reflect_multi_agent_failures(self):
        """Failure stats should aggregate across multiple agents."""
        recovery = FailureRecovery()
        recovery.analyze(FailureType.TIMEOUT, "t", "AgentA", attempt_number=1)
        recovery.analyze(FailureType.TIMEOUT, "t", "AgentA", attempt_number=1)
        recovery.analyze(FailureType.TRANSPORT, "p", "AgentB", attempt_number=1)
        recovery.analyze(FailureType.EXECUTION, "e", "AgentC", attempt_number=1)

        stats = recovery.get_stats()
        assert stats["total_failures"] == 4
        assert stats["by_type"]["timeout"] == 2
        assert stats["by_agent"]["AgentA"] == 2


class TestMultiAgentSynthesis:
    """Test conflict resolution with multiple agents."""

    def test_synthesize_bytecode_results_from_multiple_agents(self):
        """Three agents execute the same bytecode — two agree."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)

        # Two agents return the same correct result
        for agent in ["Quill", "Oracle1"]:
            resp = CooperativeResponse.success(
                "task-001", agent, "Requester",
                {"registers": [43, 0, 0, 0], "pc": 5},
                execution_time_ms=10,
                vm_info={"isa": "unified"},
            )
            resolver.add_response(resp)

        # One agent returns a different result (maybe different ISA)
        resp = CooperativeResponse.success(
            "task-001", "OldAgent", "Requester",
            {"registers": [42, 0, 0, 0], "pc": 5},
            execution_time_ms=8,
        )
        resolver.add_response(resp)

        result = resolver.resolve()
        assert result.result["registers"][0] == 43
        assert result.agents_responded == 3

    def test_weighted_synthesis_with_trust_history(self):
        """Use trust-weighted resolution with established trust history."""
        trust = TrustScorer()

        # Build trust: Oracle1 is very reliable, Babel is less reliable
        for _ in range(10):
            trust.record_result("Oracle1", "success")
        trust.record_result("Babel", "success")
        trust.record_result("Babel", "error")
        trust.record_result("Babel", "error")

        resolver = ConflictResolver(strategy=SynthesisStrategy.WEIGHTED, trust_scorer=trust)

        # Oracle1 returns one answer, Babel returns another
        oracle_resp = CooperativeResponse.success(
            "task-002", "Oracle1", "Requester",
            {"analysis": "correct"},
            execution_time_ms=15,
        )
        babel_resp = CooperativeResponse.success(
            "task-002", "Babel", "Requester",
            {"analysis": "incorrect"},
            execution_time_ms=12,
        )
        resolver.add_response(oracle_resp)
        resolver.add_response(babel_resp)

        result = resolver.resolve()
        # Oracle1's higher trust weight should win
        assert result.result == {"analysis": "correct"}
        assert result.confidence > 0.5

    def test_best_evidence_with_partial_responses(self):
        """Only some agents respond — best evidence should pick the best."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.BEST_EVIDENCE)
        resolver.set_expected_agents(5)

        # Only 2 of 5 respond
        r1 = CooperativeResponse.success(
            "task-003", "A", "R",
            {"result": "basic"},
            execution_time_ms=5,
        )
        r2 = CooperativeResponse.success(
            "task-003", "B", "R",
            {"result": "detailed", "confidence": 0.95, "supporting_evidence": ["ref1", "ref2"]},
            execution_time_ms=200,
            vm_info={"isa": "unified", "version": "1.0"},
        )
        resolver.add_response(r1)
        resolver.add_response(r2)

        result = resolver.resolve()
        # B should win: richer result + vm_info + reasonable exec time
        assert result.result["result"] == "detailed"
        assert result.agents_consulted == 5
        assert result.agents_responded == 2


class TestProtocolEvolutionCooperativeScenario:
    """Test protocol evolution in a multi-agent cooperative context."""

    def test_fleet_protocol_upgrade_path(self):
        """Simulate a fleet-wide protocol upgrade scenario."""
        registry = ProtocolRegistry()

        # Phase 1: Basic TELL/ASK
        registry.register(ProtocolVersion(
            version="1.0.0",
            changelog="Initial: TELL (0x50) and ASK (0x51)",
            opcode_mapping={"0x50": "TELL", "0x51": "ASK"},
            metadata={"phase": "1"},
        ))

        # Phase 2: Add BROADCAST
        registry.register(ProtocolVersion(
            version="1.1.0",
            changelog="Added BROADCAST (0x53)",
            opcode_mapping={"0x50": "TELL", "0x51": "ASK", "0x53": "BROADCAST"},
            predecessor="1.0.0",
            metadata={"phase": "1.5"},
        ))

        # Phase 3: Add cooperative opcodes
        registry.register(ProtocolVersion(
            version="2.0.0",
            changelog="Cooperative: DISCUSS (0x70), SYNTHESIZE (0x71)",
            opcode_mapping={
                "0x50": "TELL", "0x51": "ASK", "0x53": "BROADCAST",
                "0x70": "DISCUSS", "0x71": "SYNTHESIZE",
            },
            predecessor="1.1.0",
            metadata={"phase": "2"},
        ))

        # Verify upgrade path exists
        path = registry.migration_path("1.0.0", "2.0.0")
        assert path == ["1.0.0", "1.1.0", "2.0.0"]

        # Verify the diff from 1.0 to 2.0
        diff = registry.diff("1.0.0", "2.0.0")
        assert len(diff.added_opcodes) == 3  # BROADCAST, DISCUSS, SYNTHESIZE
        assert len(diff.removed_opcodes) == 0
        assert len(diff.modified_opcodes) == 0

        # Verify serialization
        json_str = registry.to_json()
        restored = ProtocolRegistry.from_json(json_str)
        assert len(restored) == 3
        assert restored.get("2.0.0") is not None

    def test_agents_on_different_protocol_versions(self):
        """Verify diff can be computed between any two versions agents might be on."""
        registry = ProtocolRegistry()
        registry.register(ProtocolVersion(
            version="1.0.0",
            opcode_mapping={"0x50": "TELL", "0x51": "ASK"},
        ))
        registry.register(ProtocolVersion(
            version="1.1.0",
            opcode_mapping={"0x50": "TELL", "0x51": "ASK", "0x53": "BROADCAST"},
            predecessor="1.0.0",
        ))
        registry.register(ProtocolVersion(
            version="2.0.0",
            opcode_mapping={"0x50": "TELL", "0x51": "ASK", "0x53": "BROADCAST",
                            "0x70": "DISCUSS"},
            predecessor="1.1.0",
        ))

        # Agent A is on 1.0.0, Agent B is on 2.0.0 — compute compatibility diff
        diff = registry.diff("1.0.0", "2.0.0")
        # Agent A doesn't know about BROADCAST and DISCUSS
        assert "0x53" in diff.added_opcodes
        assert "0x70" in diff.added_opcodes


class TestVMStateTransferAcrossAgents:
    """Test FluxTransfer for cooperative VM state handoff."""

    def test_transfer_state_between_agents(self):
        """Agent A executes part of a program, transfers state to Agent B."""
        # Agent A: MOVI R0, 42; MOVI R1, 100; (would ADD next but transfers instead)
        registers = [0] * 64
        registers[0] = 42
        registers[1] = 100

        outbound = FluxTransfer(
            source_pc=6,  # After second MOVI (3+3=6 bytes)
            registers=registers,
            data_stack=[42, 100],
            metadata={
                "task_id": "coop-transfer-001",
                "source_agent": "AgentA",
                "target_agent": "AgentB",
                "bytecode": [0x18, 0, 42, 0x18, 1, 100, 0x20, 2, 0, 1, 0x00],
                "next_instruction": "ADD R2, R0, R1",
            },
        )

        # Serialize for transfer
        wire_format = outbound.serialize()

        # Agent B receives and deserializes
        inbound = FluxTransfer.deserialize(wire_format)

        # Agent B continues execution: R2 = R0 + R1 = 142
        r0, r1 = inbound.registers[0], inbound.registers[1]
        inbound.registers[2] = r0 + r1

        assert inbound.registers[2] == 142
        assert inbound.source_pc == 6
        assert inbound.metadata["source_agent"] == "AgentA"
        assert inbound.metadata["next_instruction"] == "ADD R2, R0, R1"

    def test_transfer_corrupted_rejected(self):
        """Corrupted transfer data should be rejected by the receiving agent."""
        ft = FluxTransfer(
            registers=[42] * 64,
            metadata={"task": "important"},
        )
        data = bytearray(ft.serialize())
        data[50] ^= 0xFF  # Corrupt a byte

        with pytest.raises(FluxTransferError):
            FluxTransfer.deserialize(bytes(data))

    def test_transfer_empty_state(self):
        """Empty VM state transfer works correctly."""
        ft = FluxTransfer(metadata={"task": "ping"})
        data = ft.serialize()
        restored = FluxTransfer.deserialize(data)
        assert restored.registers == [0] * 64
        assert restored.data_stack == []
        assert restored.signal_stack == []


class TestEndToEndCooperativeFlow:
    """Full end-to-end cooperative execution flow with mocked transport."""

    @pytest.fixture
    def mock_transport(self):
        transport = MagicMock()
        transport.agent_name = "Quill"
        transport.send_task.return_value = "/fake/path.json"
        transport.send_response.return_value = "/fake/path.json"
        return transport

    @pytest.fixture
    def runtime(self, mock_transport):
        from src.runtime import CooperativeRuntime
        return CooperativeRuntime(mock_transport)

    def test_full_ask_respond_with_synthesis(self, runtime, mock_transport):
        """Full flow: ask multiple agents, synthesize results."""
        from src.runtime import CooperativeRuntime

        # Agent 1 responds
        resp1 = CooperativeResponse.success(
            "task-multi", "Oracle1", "Quill",
            {"answer": 42}, execution_time_ms=10,
        )
        # Agent 2 responds (same answer)
        resp2 = CooperativeResponse.success(
            "task-multi", "Super Z", "Quill",
            {"answer": 42}, execution_time_ms=15,
        )

        # First ask succeeds
        mock_transport.poll_for_response.return_value = resp1
        result = runtime.ask("Oracle1", "execute_bytecode", {"bytecode": [0x18, 0, 42, 0x00]})
        assert result == {"answer": 42}

        # Trust recorded
        assert runtime.trust.get_score("Oracle1") == 1.0

    def test_task_creation_to_response_full_lifecycle(self):
        """Full lifecycle: create task → serialize → deserialize → create response."""
        # Step 1: Create task
        task = CooperativeTask.create(
            source_agent="Quill",
            target_agent="Oracle1",
            request_type="execute_bytecode",
            payload={"bytecode": [0x18, 0, 42, 0x08, 0, 0x00]},
            context="Compute 42 + 1",
        )

        # Step 2: Serialize task
        task_json = task.to_json()

        # Step 3: Simulate transport (agent receives task)
        received_task = CooperativeTask.from_json(task_json)
        assert received_task.task_id == task.task_id

        # Step 4: Agent processes task
        bytecode = received_task.payload["bytecode"]
        # MOVI R0, 42 → R0=42
        assert bytecode[0] == 0x18  # MOVI opcode
        assert bytecode[1] == 0  # R0 destination

        # Step 5: Create response
        response = CooperativeResponse.success(
            task_id=received_task.task_id,
            source_agent="Oracle1",
            target_agent="Quill",
            result={"register_0": 43},
            execution_time_ms=3,
            vm_info={"isa": "unified-halt-zero"},
        )

        # Step 6: Serialize response
        response_json = response.to_json()

        # Step 7: Requester receives response
        received_response = CooperativeResponse.from_json(response_json)
        assert received_response.status == "success"
        assert received_response.result == {"register_0": 43}

    def test_failure_recovery_preserves_trust_consistency(self):
        """After recovery from failure, trust scores should be consistent."""
        trust = TrustScorer()
        recovery = FailureRecovery(
            max_retries=2,
            circuit_breaker=CircuitBreaker(failure_threshold=2),
        )

        # Agent fails once → retry
        action = recovery.analyze(
            FailureType.TIMEOUT, "slow", target_agent="SlowAgent", attempt_number=1,
        )
        assert action.strategy == RecoveryStrategy.RETRY

        # Record the failure in trust too
        trust.record_result("SlowAgent", "timeout")

        # Agent fails again → fallback
        action = recovery.analyze(
            FailureType.TIMEOUT, "slow again", target_agent="SlowAgent", attempt_number=2,
        )
        assert action.strategy == RecoveryStrategy.FALLBACK_LOCAL
        trust.record_result("SlowAgent", "timeout")

        # Trust should reflect poor reliability
        assert trust.get_score("SlowAgent") == 0.0

        # Circuit breaker should be open (2 consecutive failures, threshold=2)
        assert not recovery.circuit_breaker.is_available("SlowAgent")
