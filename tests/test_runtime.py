"""Tests for the cooperative runtime."""

import pytest
from unittest.mock import patch, MagicMock
from src.runtime import (
    CooperativeRuntime, CooperativeRuntimeError,
    ERR_NO_CAPABLE_AGENT, ERR_TIMEOUT, ERR_TRANSPORT_FAILURE,
)
from src.cooperative_types import CooperativeTask, CooperativeResponse
from src.trust.scorer import TrustScorer


@pytest.fixture
def mock_transport():
    """Create a mock GitTransport."""
    transport = MagicMock()
    transport.agent_name = "Quill"
    transport.send_task.return_value = "/fake/path/task.json"
    transport.send_response.return_value = "/fake/path/response.json"
    return transport


@pytest.fixture
def runtime(mock_transport):
    """Create a CooperativeRuntime with mock transport."""
    return CooperativeRuntime(mock_transport)


class TestRuntimeAsk:
    """Test ASK opcode execution."""

    def test_ask_success(self, runtime, mock_transport):
        response = CooperativeResponse.success(
            "test-001", "Oracle1", "Quill", {"value": 42}, execution_time_ms=5
        )
        mock_transport.poll_for_response.return_value = response

        result = runtime.ask("Oracle1", "ping", {})
        assert result == {"value": 42}

    def test_ask_timeout(self, runtime, mock_transport):
        mock_transport.poll_for_response.return_value = None

        with pytest.raises(CooperativeRuntimeError, match=ERR_TIMEOUT):
            runtime.ask("Oracle1", "ping", {}, timeout_ms=1000)

    def test_ask_unknown_target(self, runtime):
        with pytest.raises(CooperativeRuntimeError, match=ERR_NO_CAPABLE_AGENT):
            runtime.ask("Nonexistent", "ping", {})

    def test_ask_transport_failure_with_fallback(self, runtime, mock_transport):
        mock_transport.send_task.side_effect = Exception("push failed")
        result = runtime.ask("Oracle1", "ping", {}, fallback="local")
        assert result["status"] == "fallback_local"
        assert "push failed" in result["reason"]

    def test_ask_records_trust_on_success(self, runtime, mock_transport):
        response = CooperativeResponse.success(
            "test-trust", "Oracle1", "Quill", {"ok": True}
        )
        mock_transport.poll_for_response.return_value = response

        runtime.ask("Oracle1", "ping", {})
        assert runtime.trust.get_score("Oracle1") == 1.0

    def test_ask_records_trust_on_timeout(self, runtime, mock_transport):
        mock_transport.poll_for_response.return_value = None

        try:
            runtime.ask("Oracle1", "ping", {}, timeout_ms=100)
        except CooperativeRuntimeError:
            pass

        assert runtime.trust.get_score("Oracle1") < 0.5

    def test_ask_error_response(self, runtime, mock_transport):
        response = CooperativeResponse.error(
            "test-err", "Oracle1", "Quill", "BAD_OPCODE", "Unknown"
        )
        mock_transport.poll_for_response.return_value = response

        with pytest.raises(CooperativeRuntimeError, match="BAD_OPCODE"):
            runtime.ask("Oracle1", "execute_bytecode", {"bytecode": [0xFF]})


class TestRuntimeTell:
    """Test TELL opcode execution."""

    def test_tell_sends_notification(self, runtime, mock_transport):
        task_id = runtime.tell("Oracle1", {"message": "Hello"})
        assert mock_transport.send_task.called
        assert task_id.startswith("Quill-")

    def test_tell_unknown_target(self, runtime):
        with pytest.raises(CooperativeRuntimeError):
            runtime.tell("Nonexistent", {"message": "Hello"})


class TestRuntimeBroadcast:
    """Test BROADCAST opcode execution."""

    def test_broadcast_sends_to_all(self, runtime, mock_transport):
        count = runtime.broadcast({"message": "Fleet announcement"})
        assert count > 0
        assert mock_transport.send_task.call_count == count

    def test_broadcast_sends_to_multiple(self, runtime, mock_transport):
        count = runtime.broadcast({"message": "test"})
        # Should attempt to send to all known agents
        assert count >= 1


class TestRuntimeIncoming:
    """Test incoming task handling."""

    def test_check_incoming_delegates_to_transport(self, runtime, mock_transport):
        mock_transport.check_for_tasks.return_value = []
        tasks = runtime.check_incoming()
        assert tasks == []
        mock_transport.check_for_tasks.called

    def test_handle_ping_task(self, runtime, mock_transport):
        task = CooperativeTask.create("Oracle1", "Quill", "ping", {})
        response = runtime.handle_task(task)
        assert response.status == "success"
        assert response.result["agent"] == "Quill"

    def test_handle_expired_task(self, runtime):
        task = CooperativeTask.create("Oracle1", "Quill", "ping", {}, timeout_ms=1)
        import time
        time.sleep(0.01)
        response = runtime.handle_task(task)
        assert response.status == "error"
        assert "expired" in response.error_message.lower()

    def test_handle_task_with_custom_handler(self, runtime):
        task = CooperativeTask.create("Oracle1", "Quill", "custom", {"x": 10})
        
        def custom_handler(t):
            return {"result": t.payload["x"] * 2}
        
        response = runtime.handle_task(task, handler=custom_handler)
        assert response.status == "success"
        assert response.result == {"result": 20}

    def test_handle_task_with_failing_handler(self, runtime):
        task = CooperativeTask.create("Oracle1", "Quill", "custom", {})
        
        def failing_handler(t):
            raise ValueError("Something went wrong")
        
        response = runtime.handle_task(task, handler=failing_handler)
        assert response.status == "error"
        assert "Something went wrong" in response.error_message

    def test_send_response(self, runtime, mock_transport):
        response = CooperativeResponse.success("t-1", "Quill", "Oracle1", {"ok": True})
        runtime.send_response(response)
        mock_transport.send_response.called


class TestBytecodeExecution:
    """Test the built-in bytecode executor."""

    def test_execute_mov_inc_halt(self, runtime):
        # MOVI R0, 42; INC R0; HALT
        bytecode = [0x18, 0, 42, 0x00, 0x08, 0, 0x00]
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": bytecode, "expected_result": "register_0"},
        )
        response = runtime.handle_task(task)
        assert response.status == "success"
        assert response.result["value"] == 43

    def test_execute_add(self, runtime):
        # MOVI R0, 10; MOVI R1, 20; ADD R2, R0, R1; HALT
        bytecode = [0x18, 0, 10, 0x00, 0x18, 1, 20, 0x00, 0x20, 2, 0, 1, 0x00]
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": bytecode, "expected_result": "register_2"},
        )
        response = runtime.handle_task(task)
        assert response.result["value"] == 30

    def test_execute_mul(self, runtime):
        # MOVI R0, 7; MOVI R1, 6; MUL R2, R0, R1; HALT
        bytecode = [0x18, 0, 7, 0x00, 0x18, 1, 6, 0x00, 0x22, 2, 0, 1, 0x00]
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": bytecode, "expected_result": "register_2"},
        )
        response = runtime.handle_task(task)
        assert response.result["value"] == 42

    def test_execute_push_pop(self, runtime):
        # MOVI R0, 99; PUSH R0; POP R1; HALT
        bytecode = [0x18, 0, 99, 0x00, 0x0C, 0, 0x0D, 1, 0x00]
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": bytecode, "expected_result": "register_1"},
        )
        response = runtime.handle_task(task)
        assert response.result["value"] == 99

    def test_execute_unknown_opcode(self, runtime):
        bytecode = [0xFE, 0x00]  # Invalid opcode
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": bytecode},
        )
        response = runtime.handle_task(task)
        assert response.status == "error"
        assert "Unknown opcode" in response.error_message

    def test_execute_addi(self, runtime):
        # MOVI R0, 100; ADDI R0, 50; HALT
        bytecode = [0x18, 0, 100, 0x00, 0x19, 0, 50, 0x00, 0x00]
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": bytecode, "expected_result": "register_0"},
        )
        response = runtime.handle_task(task)
        assert response.result["value"] == 150

    def test_execute_subi(self, runtime):
        # MOVI R0, 100; SUBI R0, 30; HALT
        bytecode = [0x18, 0, 100, 0x00, 0x1A, 0, 30, 0x00, 0x00]
        task = CooperativeTask.create(
            "Oracle1", "Quill", "execute_bytecode",
            {"bytecode": bytecode, "expected_result": "register_0"},
        )
        response = runtime.handle_task(task)
        assert response.result["value"] == 70

    def test_vm_info_in_response(self, runtime):
        task = CooperativeTask.create("Oracle1", "Quill", "ping", {})
        response = runtime.handle_task(task)
        assert response.vm_info is not None
        assert response.vm_info["agent"] == "Quill"
        assert response.vm_info["isa_version"] == "unified-halt-zero"


class TestRuntimeStatus:
    """Test runtime status reporting."""

    def test_get_status(self, runtime):
        status = runtime.get_status()
        assert status["agent"] == "Quill"
        assert "trust_scores" in status

    def test_get_trust_report(self, runtime):
        runtime.trust.record_result("Oracle1", "success")
        report = runtime.get_trust_report()
        assert "Oracle1" in report


class TestKnowledgeQuery:
    """Test knowledge query handler."""

    def test_query_knowledge(self, runtime):
        task = CooperativeTask.create("Oracle1", "Quill", "query_knowledge", {})
        response = runtime.handle_task(task)
        assert response.status == "success"
        assert "domains" in response.result
        assert "protocol-design" in response.result["domains"]
