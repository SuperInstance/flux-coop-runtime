#!/usr/bin/env python3
"""
End-to-end cooperative execution demonstration.

Shows how two agents can cooperate through the flux-coop-runtime:
1. Agent A asks Agent B to execute bytecode
2. Agent B receives the task, executes it, and responds
3. Agent A receives the result and continues

This demo uses mock transport (no actual git) to illustrate the flow.
"""

from src.runtime import CooperativeRuntime, CooperativeRuntimeError
from src.transport.git_transport import GitTransport
from src.trust.scorer import TrustScorer
from src.cooperative_types import CooperativeTask, CooperativeResponse
from src.discovery.resolver import register_agent
from unittest.mock import MagicMock, patch
import json
import sys

# Register demo agents for discovery
register_agent("AgentA", "https://github.com/demo/a", "vessel", ["testing"], 0.8)
register_agent("AgentB", "https://github.com/demo/b", "vessel", ["execution"], 0.9)
register_agent("SlowAgent", "https://github.com/demo/slow", "vessel", ["testing"], 0.3)


def demo_basic_ping():
    """Demo 1: Basic ping between agents."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Ping (Agent A → Agent B)")
    print("="*60)

    # Agent A's runtime
    transport_a = MagicMock()
    transport_a.agent_name = "AgentA"
    runtime_a = CooperativeRuntime(transport_a)

    # Agent B responds to the ping
    response = CooperativeResponse.success(
        task_id="ping-001",
        source_agent="AgentB",
        target_agent="AgentA",
        result={"agent": "AgentB", "status": "active", "version": "1.0"},
        execution_time_ms=5,
    )
    transport_a.poll_for_response.return_value = response

    # Agent A asks Agent B for a ping
    result = runtime_a.ask("AgentB", "ping", {}, timeout_ms=5000)
    print(f"  Agent A asked Agent B for a ping")
    print(f"  Agent B responded: {result}")
    print(f"  Trust score for Agent B: {runtime_a.trust.get_score('AgentB'):.1f}")
    assert result["agent"] == "AgentB"
    print("  ✓ Basic ping successful")


def demo_bytecode_execution():
    """Demo 2: Cross-agent bytecode execution."""
    print("\n" + "="*60)
    print("DEMO 2: Cross-Agent Bytecode Execution")
    print("="*60)

    # Agent A's runtime
    transport_a = MagicMock()
    transport_a.agent_name = "AgentA"
    runtime_a = CooperativeRuntime(transport_a)

    # Agent B executes bytecode and returns result
    # MOVI R0, 42; INC R0; INC R0; HALT → R0 = 44
    response = CooperativeResponse.success(
        task_id="bytecode-001",
        source_agent="AgentB",
        target_agent="AgentA",
        result={"register_0": 44, "registers": [44, 0, 0, 0, 0, 0, 0, 0], "pc": 7},
        execution_time_ms=12,
        vm_info={"implementation": "flux-runtime-python", "isa_version": "unified"},
    )
    transport_a.poll_for_response.return_value = response

    # Agent A asks Agent B to execute bytecode
    bytecode = [0x18, 0, 42, 0x00, 0x08, 0, 0x08, 0, 0x00]
    result = runtime_a.ask(
        "AgentB",
        "execute_bytecode",
        {"bytecode": bytecode, "expected_result": "register_0"},
        context="Compute 42 + 1 + 1 and return R0",
        timeout_ms=10000,
    )
    print(f"  Agent A sent bytecode: MOVI R0,42; INC R0; INC R0; HALT")
    print(f"  Agent B executed and returned: R0 = {result['register_0']}")
    print(f"  VM used: {response.vm_info['implementation']}")
    print(f"  ISA version: {response.vm_info['isa_version']}")
    assert result["register_0"] == 44
    print("  ✓ Cross-agent bytecode execution successful")


def demo_timeout_with_fallback():
    """Demo 3: Timeout handling with local fallback."""
    print("\n" + "="*60)
    print("DEMO 3: Timeout with Fallback")
    print("="*60)

    transport = MagicMock()
    transport.agent_name = "AgentA"
    runtime = CooperativeRuntime(transport)
    transport.poll_for_response.return_value = None  # No response

    # Agent A asks with fallback
    result = runtime.ask("SlowAgent", "ping", {}, timeout_ms=100, fallback="local")
    print(f"  Agent A asked SlowAgent (no response)")
    print(f"  Timeout after 100ms, fallback: {result}")
    print(f"  Trust score for SlowAgent: {runtime.trust.get_score('SlowAgent'):.1f}")
    assert result["status"] == "fallback_local"
    print("  ✓ Fallback handled gracefully")


def demo_broadcast():
    """Demo 4: Broadcast to all agents."""
    print("\n" + "="*60)
    print("DEMO 4: Fleet Broadcast")
    print("="*60)

    transport = MagicMock()
    transport.agent_name = "Quill"
    runtime = CooperativeRuntime(transport)

    count = runtime.broadcast(
        {"message": "RFC-0001 ISA Canonical Declaration is now CANONICAL"},
        context="Fleet-wide announcement",
    )
    print(f"  Quill broadcast to {count} agents")
    print(f"  Message: RFC-0001 ISA Canonical Declaration is now CANONICAL")
    print("  ✓ Broadcast successful")


def demo_incoming_task_handling():
    """Demo 5: Agent B receives and handles an incoming task."""
    print("\n" + "="*60)
    print("DEMO 5: Incoming Task Handling")
    print("="*60)

    transport = MagicMock()
    transport.agent_name = "AgentB"
    runtime = CooperativeRuntime(transport)

    # Simulate incoming task from Agent A
    task = CooperativeTask.create(
        source_agent="AgentA",
        target_agent="AgentB",
        request_type="execute_bytecode",
        payload={
            "bytecode": [0x18, 0, 10, 0x00, 0x18, 1, 20, 0x00, 0x20, 2, 0, 1, 0x00],
            "expected_result": "register_2",
        },
        context="Compute R0 + R1 = R2",
    )

    # Agent B handles the task using default handler (built-in VM)
    response = runtime.handle_task(task)
    print(f"  Incoming task from {task.source_agent}")
    print(f"  Request type: {task.request_type}")
    print(f"  Execution time: {response.execution_time_ms}ms")
    print(f"  Result: R2 = {response.result['value']}")
    assert response.status == "success"
    assert response.result["value"] == 30
    print("  ✓ Task handled successfully")


def demo_trust_tracking():
    """Demo 6: Trust score tracking across multiple interactions."""
    print("\n" + "="*60)
    print("DEMO 6: Trust Score Tracking")
    print("="*60)

    transport = MagicMock()
    transport.agent_name = "AgentA"
    runtime = CooperativeRuntime(transport)

    # Simulate multiple interactions
    success_resp = CooperativeResponse.success("t-1", "B", "A", {"ok": True})
    error_resp = CooperativeResponse.error("t-2", "B", "A", "ERR", "bad opcode")
    
    # 7 successes, 3 errors
    for _ in range(7):
        transport.poll_for_response.return_value = success_resp
        runtime.ask("AgentB", "ping", {}, timeout_ms=100)
    for _ in range(3):
        transport.poll_for_response.return_value = error_resp
        try:
            runtime.ask("AgentB", "execute_bytecode", {}, timeout_ms=100)
        except CooperativeRuntimeError:
            pass

    score = runtime.trust.get_score("AgentB")
    report = runtime.trust.get_record("AgentB")
    print(f"  Interactions: {report.total} (success={report.successes}, error={report.failures})")
    print(f"  Trust score: {score:.1%}")
    assert score == 0.7
    print("  ✓ Trust tracking working correctly")


def run_all_demos():
    """Run all demonstrations."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  flux-coop-runtime — End-to-End Cooperative Demo         ║")
    print("║  Phase 1: Ask/Respond Cooperative Execution               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_basic_ping()
    demo_bytecode_execution()
    demo_timeout_with_fallback()
    demo_broadcast()
    demo_incoming_task_handling()
    demo_trust_tracking()

    # Use the last runtime from demo 6
    from src.discovery.resolver import FLEET_REGISTRY
    final_runtime = CooperativeRuntime(transport)
    final_runtime.trust = runtime.trust

    print("\n" + "="*60)
    print("ALL DEMOS PASSED — Cooperative Execution is Working!")
    print("="*60)
    print(f"\nRuntime status: {json.dumps(final_runtime.get_status(), indent=2)}")


if __name__ == "__main__":
    run_all_demos()
