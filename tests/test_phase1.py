import pytest
"""Phase 1 integration tests — end-to-end cooperative execution scenarios."""

import json
import time
from src.cooperative_types import CooperativeTask, CooperativeResponse, TrustRecord
from src.discovery.resolver import resolve, ResolutionError
from src.trust.scorer import TrustScorer
from src.transfer.format import FluxTransfer, FluxTransferError


class TestPhase1_Conformance1_BasicAskRespond:
    """Conformance Test 1: Basic Ask/Respond (ping)."""

    def test_create_ping_task(self):
        task = CooperativeTask.create(
            source_agent="Quill",
            target_agent="Oracle1",
            request_type="ping",
            payload={},
            context="Health check",
        )
        assert task.request_type == "ping"
        assert task.source_agent == "Quill"
        assert task.target_agent == "Oracle1"
        assert task.task_id.startswith("Quill-")

    def test_ping_response(self):
        response = CooperativeResponse.success(
            task_id="test-001",
            source_agent="Oracle1",
            target_agent="Quill",
            result={"agent": "Oracle1", "status": "active", "version": "session-1"},
            execution_time_ms=5,
            vm_info={"implementation": "flux-runtime-python", "isa_version": "unified"},
        )
        data = response.to_json()
        assert data["status"] == "success"
        assert data["result"]["agent"] == "Oracle1"

    def test_task_serialization_roundtrip(self):
        task = CooperativeTask.create(
            source_agent="Quill",
            target_agent="Super Z",
            request_type="ping",
            payload={"context": "test"},
        )
        json_data = task.to_json()
        restored = CooperativeTask.from_json(json_data)
        assert restored.task_id == task.task_id
        assert restored.source_agent == "Quill"
        assert restored.request_type == "ping"


class TestPhase1_Conformance2_BytecodeExecution:
    """Conformance Test 2: Bytecode execution request."""

    def test_create_bytecode_task(self):
        # MOVI R0, 42; INC R0; HALT
        bytecode = [0x18, 0, 42, 0x08, 0, 0x00]
        task = CooperativeTask.create(
            source_agent="Quill",
            target_agent="Super Z",
            request_type="execute_bytecode",
            payload={
                "bytecode": bytecode,
                "expected_result": "register_0",
            },
            context="Compute 42 + 1 and return R0",
        )
        assert task.payload["bytecode"] == bytecode
        json_data = task.to_json()
        assert "bytecode" in json_data["payload"]


class TestPhase1_Conformance3_Timeout:
    """Conformance Test 3: Timeout handling."""

    def test_task_expiration(self):
        task = CooperativeTask.create(
            source_agent="Quill",
            target_agent="SlowAgent",
            request_type="execute_bytecode",
            payload={},
            timeout_ms=1,  # 1ms — will expire immediately
        )
        time.sleep(0.01)  # 10ms
        assert task.is_expired()

    def test_task_not_expired(self):
        task = CooperativeTask.create(
            source_agent="Quill",
            target_agent="Oracle1",
            request_type="ping",
            payload={},
            timeout_ms=60000,
        )
        assert not task.is_expired()


class TestPhase1_Conformance4_ErrorHandling:
    """Conformance Test 4: Error response handling."""

    def test_error_response(self):
        response = CooperativeResponse.error(
            task_id="test-004",
            source_agent="Super Z",
            target_agent="Quill",
            error_code="INVALID_BYTECODE",
            error_message="Opcode 0x99 not recognized in unified ISA",
        )
        assert response.status == "error"
        assert response.error_code == "INVALID_BYTECODE"
        json_data = response.to_json()
        restored = CooperativeResponse.from_json(json_data)
        assert restored.error_message == "Opcode 0x99 not recognized in unified ISA"


class TestPhase1_Conformance7_CapabilityDiscovery:
    """Conformance Test 7: Capability-based agent discovery."""

    def test_discover_cuda_agent(self):
        addr = resolve("cap:cuda")
        assert addr.agent_name == "JetsonClaw1"

    def test_discover_protocol_agent(self):
        addr = resolve("cap:protocol")
        assert addr.agent_name == "Quill"

    def test_discover_hardware_agent(self):
        addr = resolve("cap:hardware")
        assert addr.agent_name == "JetsonClaw1"


class TestPhase1_Conformance8_UnknownAgent:
    """Conformance Test 8: Unknown agent handling."""

    def test_unknown_agent_raises(self):
        with pytest.raises(ResolutionError):
            resolve("NonExistentAgent")


class TestPhase1_FluxTransfer:
    """FluxTransfer serialization tests for cooperative state transfer."""

    def test_vm_state_transfer(self):
        """Serialize VM state, transfer, deserialize on other side."""
        registers = [0] * 64
        registers[0] = 42
        registers[1] = 100

        ft = FluxTransfer(
            source_pc=9,
            registers=registers,
            data_stack=[42, 100],
            metadata={
                "task_id": "coop-test-001",
                "request_type": "execute_bytecode",
                "description": "R2 = R0 + R1 = 142",
            },
        )

        # Serialize (what sender does)
        binary = ft.serialize()

        # Deserialize (what receiver does)
        restored = FluxTransfer.deserialize(binary)

        assert restored.source_pc == 9
        assert restored.registers[0] == 42
        assert restored.registers[1] == 100
        assert restored.data_stack == [42, 100]
        assert restored.metadata["description"] == "R2 = R0 + R1 = 142"

    def test_flux_transfer_with_response_metadata(self):
        """Receiver adds execution result to metadata before returning."""
        ft = FluxTransfer(
            source_pc=6,  # After HALT
            registers=[0]*64,
            metadata={"task_id": "test", "request_type": "execute_bytecode"},
        )
        # Receiver executes and modifies registers
        ft.registers[0] = 43  # Result of 42+1
        ft.metadata["execution_result"] = {"register_0": 43}
        ft.metadata["execution_time_ms"] = 2

        binary = ft.serialize()
        restored = FluxTransfer.deserialize(binary)
        assert restored.registers[0] == 43
        assert restored.metadata["execution_result"]["register_0"] == 43


class TestPhase1_TrustScoring:
    """Trust scoring integration tests."""

    def test_trust_after_successful_ping(self):
        scorer = TrustScorer()
        score = scorer.record_result("Oracle1", "success")
        assert score == 1.0

    def test_trust_degrades_after_timeout(self):
        scorer = TrustScorer()
        scorer.record_result("SlowAgent", "success")
        scorer.record_result("SlowAgent", "timeout")
        assert scorer.get_score("SlowAgent") == 0.5

    def test_trust_ranks_agents_for_capability_request(self):
        scorer = TrustScorer()
        # Make Quill most trusted for protocol
        for _ in range(10):
            scorer.record_result("Quill", "success")
        # Super Z less trusted
        for _ in range(3):
            scorer.record_result("Super Z", "success")
        for _ in range(2):
            scorer.record_result("Super Z", "error")

        ranked = scorer.rank_agents(["Quill", "Super Z"])
        assert ranked[0][0] == "Quill"
        assert ranked[0][1] > ranked[1][1]
