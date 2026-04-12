"""
Unit tests for cooperative_types.py.

Covers CooperativeTask creation/serialization/deserialization,
CooperativeResponse success/error factories, TrustRecord scoring,
task expiration, and JSON round-trip integrity.
"""

import json
import time
import unittest

from src.cooperative_types import (
    CooperativeTask,
    CooperativeResponse,
    TrustRecord,
    TaskStatus,
    RequestType,
)


class TestCooperativeTaskCreation(unittest.TestCase):
    """Test CooperativeTask creation via constructor and factory."""

    def test_create_with_factory(self):
        task = CooperativeTask.create(
            source_agent="Quill",
            target_agent="Oracle1",
            request_type="ping",
            payload={},
            context="Health check",
        )
        self.assertEqual(task.source_agent, "Quill")
        self.assertEqual(task.target_agent, "Oracle1")
        self.assertEqual(task.request_type, "ping")
        self.assertEqual(task.payload, {})
        self.assertTrue(task.task_id.startswith("Quill-"))
        self.assertTrue(task.version, "1.0")
        self.assertTrue(task.created_at)
        self.assertTrue(task.expires_at)

    def test_task_id_contains_timestamp(self):
        task = CooperativeTask.create(
            source_agent="Quill",
            target_agent="Oracle1",
            request_type="ping",
            payload={},
        )
        # Format: Quill-YYYYMMDD-HHMMSS-xxxxxx
        parts = task.task_id.split("-")
        self.assertEqual(parts[0], "Quill")
        self.assertEqual(len(parts[1]), 8)  # YYYYMMDD
        self.assertEqual(len(parts[2]), 6)  # HHMMSS
        self.assertEqual(len(parts[3]), 6)  # uuid hex

    def test_task_id_is_unique(self):
        t1 = CooperativeTask.create("Quill", "Oracle1", "ping", {})
        t2 = CooperativeTask.create("Quill", "Oracle1", "ping", {})
        self.assertNotEqual(t1.task_id, t2.task_id)

    def test_custom_timeout(self):
        task = CooperativeTask.create(
            source_agent="Quill",
            target_agent="Oracle1",
            request_type="execute_bytecode",
            payload={},
            timeout_ms=60000,
        )
        self.assertEqual(task.timeout_ms, 60000)

    def test_default_timeout(self):
        task = CooperativeTask.create("Quill", "Oracle1", "ping", {})
        self.assertEqual(task.timeout_ms, 30000)

    def test_source_repo(self):
        task = CooperativeTask.create(
            source_agent="Quill",
            target_agent="Oracle1",
            request_type="ping",
            payload={},
            source_repo="https://github.com/SuperInstance/quill-vessel",
        )
        self.assertIn("github.com", task.source_repo)


class TestCooperativeTaskSerialization(unittest.TestCase):
    """Test CooperativeTask to_json / from_json round-trip."""

    def test_to_json_keys(self):
        task = CooperativeTask.create("Quill", "Oracle1", "ping", {})
        data = task.to_json()
        expected_keys = {
            "version", "task_id", "source_agent", "source_repo",
            "target_agent", "request_type", "payload", "context",
            "timeout_ms", "created_at", "expires_at",
        }
        self.assertEqual(set(data.keys()), expected_keys)

    def test_from_json_roundtrip(self):
        original = CooperativeTask.create(
            source_agent="Quill",
            target_agent="Super Z",
            request_type="execute_bytecode",
            payload={"bytecode": [0x18, 0, 42, 0x00], "expected_result": "register_0"},
            context="Compute 42+1",
            timeout_ms=15000,
            source_repo="https://github.com/example/repo",
        )
        data = original.to_json()
        restored = CooperativeTask.from_json(data)

        self.assertEqual(restored.task_id, original.task_id)
        self.assertEqual(restored.source_agent, "Quill")
        self.assertEqual(restored.target_agent, "Super Z")
        self.assertEqual(restored.request_type, "execute_bytecode")
        self.assertEqual(restored.payload, original.payload)
        self.assertEqual(restored.context, "Compute 42+1")
        self.assertEqual(restored.timeout_ms, 15000)
        self.assertEqual(restored.source_repo, "https://github.com/example/repo")
        self.assertEqual(restored.version, "1.0")

    def test_from_json_with_missing_optional_fields(self):
        data = {
            "task_id": "minimal-001",
            "source_agent": "Quill",
            "request_type": "ping",
        }
        task = CooperativeTask.from_json(data)
        self.assertEqual(task.target_agent, "any")  # default
        self.assertEqual(task.payload, {})
        self.assertEqual(task.timeout_ms, 30000)  # default
        self.assertEqual(task.version, "1.0")  # default

    def test_json_string_roundtrip(self):
        original = CooperativeTask.create("Quill", "Oracle1", "ping", {})
        json_str = json.dumps(original.to_json())
        restored = CooperativeTask.from_json(json.loads(json_str))
        self.assertEqual(restored.task_id, original.task_id)
        self.assertEqual(restored.source_agent, "Quill")


class TestCooperativeTaskExpiration(unittest.TestCase):
    """Test CooperativeTask.is_expired() behavior."""

    def test_not_expired_after_creation(self):
        task = CooperativeTask.create(
            "Quill", "Oracle1", "ping", {}, timeout_ms=60000
        )
        self.assertFalse(task.is_expired())

    def test_expired_after_short_timeout(self):
        task = CooperativeTask.create(
            "Quill", "Oracle1", "ping", {}, timeout_ms=1
        )
        time.sleep(0.05)  # 50ms
        self.assertTrue(task.is_expired())

    def test_not_expired_with_long_timeout(self):
        task = CooperativeTask.create(
            "Quill", "Oracle1", "ping", {}, timeout_ms=600000
        )
        self.assertFalse(task.is_expired())


class TestCooperativeResponseFactories(unittest.TestCase):
    """Test CooperativeResponse.success() and .error() factories."""

    def test_success_factory(self):
        resp = CooperativeResponse.success(
            task_id="task-001",
            source_agent="Oracle1",
            target_agent="Quill",
            result={"register_0": 43},
            execution_time_ms=12,
        )
        self.assertEqual(resp.status, "success")
        self.assertEqual(resp.task_id, "task-001")
        self.assertEqual(resp.source_agent, "Oracle1")
        self.assertEqual(resp.target_agent, "Quill")
        self.assertEqual(resp.result, {"register_0": 43})
        self.assertEqual(resp.execution_time_ms, 12)
        self.assertTrue(resp.responded_at)

    def test_success_with_vm_info(self):
        resp = CooperativeResponse.success(
            task_id="task-002",
            source_agent="Super Z",
            target_agent="Quill",
            result={"ok": True},
            vm_info={"isa_version": "unified-halt-zero"},
        )
        self.assertEqual(resp.vm_info["isa_version"], "unified-halt-zero")

    def test_error_factory(self):
        resp = CooperativeResponse.error(
            task_id="task-003",
            source_agent="Super Z",
            target_agent="Quill",
            error_code="INVALID_BYTECODE",
            error_message="Opcode 0x99 not recognized",
        )
        self.assertEqual(resp.status, "error")
        self.assertEqual(resp.error_code, "INVALID_BYTECODE")
        self.assertEqual(resp.error_message, "Opcode 0x99 not recognized")
        self.assertIsNone(resp.result)

    def test_response_serialization_success(self):
        resp = CooperativeResponse.success(
            "task-001", "Oracle1", "Quill", {"value": 42}, execution_time_ms=5
        )
        data = resp.to_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["result"]["value"], 42)
        self.assertEqual(data["execution_time_ms"], 5)
        self.assertNotIn("error_code", data)

    def test_response_serialization_error(self):
        resp = CooperativeResponse.error(
            "task-002", "A", "B", "ERR", "Something broke"
        )
        data = resp.to_json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error_code"], "ERR")
        self.assertEqual(data["error_message"], "Something broke")

    def test_response_roundtrip(self):
        original = CooperativeResponse.success(
            "task-rt", "Quill", "Oracle1",
            {"registers": [43, 0, 0, 0, 0, 0, 0, 0]},
            execution_time_ms=8,
            vm_info={"implementation": "flux-coop-runtime-python"},
        )
        data = original.to_json()
        restored = CooperativeResponse.from_json(data)

        self.assertEqual(restored.task_id, "task-rt")
        self.assertEqual(restored.status, "success")
        self.assertEqual(restored.result, original.result)
        self.assertEqual(restored.execution_time_ms, 8)
        self.assertEqual(restored.vm_info["implementation"], "flux-coop-runtime-python")


class TestTrustRecord(unittest.TestCase):
    """Test TrustRecord scoring logic."""

    def test_neutral_prior(self):
        rec = TrustRecord(agent_name="UnknownAgent")
        self.assertEqual(rec.score, 0.5)
        self.assertEqual(rec.total, 0)

    def test_perfect_score(self):
        rec = TrustRecord(agent_name="Reliable")
        rec.record("success")
        rec.record("success")
        self.assertEqual(rec.score, 1.0)
        self.assertEqual(rec.total, 2)

    def test_degraded_score(self):
        rec = TrustRecord(agent_name="Flaky")
        rec.record("success")
        rec.record("error")
        self.assertEqual(rec.score, 0.5)

    def test_poor_score(self):
        rec = TrustRecord(agent_name="Bad")
        for _ in range(7):
            rec.record("error")
        for _ in range(3):
            rec.record("success")
        self.assertAlmostEqual(rec.score, 3.0 / 10.0)

    def test_timeout_counts(self):
        rec = TrustRecord(agent_name="Slow")
        rec.record("timeout")
        rec.record("timeout")
        rec.record("success")
        self.assertAlmostEqual(rec.score, 1.0 / 3.0)
        self.assertEqual(rec.timeouts, 2)

    def test_record_updates_last_seen(self):
        rec = TrustRecord(agent_name="A")
        self.assertEqual(rec.last_seen, "")
        rec.record("success")
        self.assertTrue(rec.last_seen)

    def test_to_dict(self):
        rec = TrustRecord(agent_name="A")
        rec.record("success")
        rec.record("success")
        d = rec.to_dict()
        self.assertEqual(d["agent_name"], "A")
        self.assertEqual(d["successes"], 2)
        self.assertEqual(d["failures"], 0)
        self.assertEqual(d["score"], 1.0)
        self.assertEqual(d["total"], 2)
        self.assertIn("last_seen", d)


class TestTaskRepr(unittest.TestCase):
    """Test __repr__ methods for debugging output."""

    def test_task_repr(self):
        task = CooperativeTask.create("Quill", "Oracle1", "ping", {})
        r = repr(task)
        self.assertIn("CooperativeTask", r)
        self.assertIn("Quill", r)
        self.assertIn("Oracle1", r)
        self.assertIn("ping", r)

    def test_response_repr(self):
        resp = CooperativeResponse.success("t1", "A", "B", {"ok": True})
        r = repr(resp)
        self.assertIn("CooperativeResponse", r)
        self.assertIn("success", r)


if __name__ == "__main__":
    unittest.main()
