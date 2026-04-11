"""
Cooperative Runtime — Main entry point.

Hooks into FLUX VM execution to handle ASK/TELL/BROADCAST opcodes,
translating them into fleet-level cooperative operations via the
transport layer.
"""

import time
from typing import Optional, Dict, Any, Callable
from src.cooperative_types import CooperativeTask, CooperativeResponse, TaskStatus
from src.discovery.resolver import resolve, AgentAddress, ResolutionError
from src.trust.scorer import TrustScorer
from src.transport.git_transport import GitTransport, GitTransportError
from src.transfer.format import FluxTransfer


class CooperativeRuntimeError(Exception):
    """Raised when cooperative runtime operations fail."""
    pass


# Error codes for the fleet
ERR_NO_CAPABLE_AGENT = "NO_CAPABLE_AGENT"
ERR_TIMEOUT = "TIMEOUT"
ERR_TRANSPORT_FAILURE = "TRANSPORT_FAILURE"
ERR_TASK_EXPIRED = "TASK_EXPIRED"
ERR_AGENT_REFUSED = "AGENT_REFUSED"


class CooperativeRuntime:
    """
    Main cooperative runtime that bridges VM-level coordination opcodes
    to fleet-level message passing.
    
    Usage:
        runtime = CooperativeRuntime(transport, trust_sorer)
        
        # ASK another agent for help
        result = runtime.ask(
            target="Oracle1",
            request_type="execute_bytecode",
            payload={"bytecode": [0x18, 0, 42, 0x00]},
            timeout_ms=30000,
        )
        
        # Check for incoming tasks and respond
        for task in runtime.check_incoming():
            response = runtime.handle_task(task)
            if response:
                runtime.send_response(response)
    """

    def __init__(self, transport: GitTransport, trust_scorer: Optional[TrustScorer] = None):
        self.transport = transport
        self.trust = trust_scorer or TrustScorer()
        self._pending_tasks: Dict[str, CooperativeTask] = {}

    def ask(self, target: str, request_type: str, payload: Dict[str, Any],
            context: str = "", timeout_ms: int = 30000,
            fallback: str = "none") -> Any:
        """
        Execute an ASK opcode — send a cooperative request and wait for response.
        
        This is the synchronous cooperative execution primitive. The calling
        VM thread is blocked until a response arrives or timeout expires.
        
        Args:
            target: Agent to ask (name, role:, cap:, any, or URL)
            request_type: Type of request (execute_bytecode, query_knowledge, ping, etc.)
            payload: Request payload (bytecode, query, etc.)
            context: Human-readable description
            timeout_ms: Maximum wait time in milliseconds
            fallback: "none" (raise error) or "local" (return placeholder)
        
        Returns:
            The result from the responding agent.
        
        Raises:
            CooperativeRuntimeError: On timeout, transport failure, or no capable agent.
        """
        # Step 1: Resolve target agent
        try:
            address = resolve(target, min_confidence=0.5)
        except ResolutionError:
            if fallback == "local":
                return {"status": "fallback_local", "reason": f"Cannot resolve target: {target}"}
            raise CooperativeRuntimeError(
                f"{ERR_NO_CAPABLE_AGENT}: Cannot resolve target '{target}'"
            )

        # Step 2: Create cooperative task
        task = CooperativeTask.create(
            source_agent=self.transport.agent_name,
            target_agent=address.agent_name,
            request_type=request_type,
            payload=payload,
            context=context,
            timeout_ms=timeout_ms,
        )

        # Step 3: Send via transport
        try:
            self.transport.send_task(task)
        except (GitTransportError, Exception) as e:
            if fallback == "local":
                return {"status": "fallback_local", "reason": str(e)}
            raise CooperativeRuntimeError(
                f"{ERR_TRANSPORT_FAILURE}: Failed to send task: {e}"
            )

        self._pending_tasks[task.task_id] = task

        # Step 4: Poll for response
        try:
            response = self.transport.poll_for_response(
                task.task_id, timeout_ms=timeout_ms, poll_interval_ms=5000
            )
        except Exception as e:
            response = None

        # Step 5: Handle result
        if response is None:
            # Timeout
            self.trust.record_result(address.agent_name, "timeout")
            if fallback == "local":
                return {"status": "fallback_local", "reason": "timeout"}
            raise CooperativeRuntimeError(
                f"{ERR_TIMEOUT}: No response from {address.agent_name} "
                f"within {timeout_ms}ms"
            )

        # Record trust
        self.trust.record_result(address.agent_name, response.status)

        if response.status == "success":
            return response.result
        elif response.status == "error":
            raise CooperativeRuntimeError(
                f"{response.error_code}: {response.error_message}"
            )
        elif response.status == "refused":
            raise CooperativeRuntimeError(ERR_AGENT_REFUSED)
        else:
            raise CooperativeRuntimeError(f"Unexpected response status: {response.status}")

    def tell(self, target: str, payload: Dict[str, Any], context: str = "") -> str:
        """
        Execute a TELL opcode — one-way notification, no response expected.
        
        Non-blocking. Returns immediately after sending.
        """
        try:
            address = resolve(target, min_confidence=0.3)
        except ResolutionError:
            raise CooperativeRuntimeError(
                f"{ERR_NO_CAPABLE_AGENT}: Cannot resolve target '{target}'"
            )

        task = CooperativeTask.create(
            source_agent=self.transport.agent_name,
            target_agent=address.agent_name,
            request_type="notification",
            payload=payload,
            context=context,
            timeout_ms=0,  # No timeout for notifications
        )

        try:
            self.transport.send_task(task)
        except GitTransportError as e:
            raise CooperativeRuntimeError(f"{ERR_TRANSPORT_FAILURE}: {e}")

        return task.task_id

    def broadcast(self, payload: Dict[str, Any], context: str = "") -> int:
        """
        Execute a BROADCAST opcode — send to all known agents.
        Returns the number of agents notified.
        """
        from src.discovery.resolver import list_agents
        agents = list_agents()
        sent = 0
        for agent_info in agents:
            try:
                self.tell(agent_info["name"], payload, context)
                sent += 1
            except CooperativeRuntimeError:
                continue
        return sent

    def check_incoming(self) -> list:
        """Check for incoming tasks from other agents."""
        return self.transport.check_for_tasks()

    def handle_task(self, task: CooperativeTask,
                    handler: Optional[Callable] = None) -> Optional[CooperativeResponse]:
        """
        Handle an incoming cooperative task.
        
        If a handler function is provided, it will be called with the task
        and should return a result dict or raise an exception.
        
        If no handler is provided, returns a default response based on
        the request type.
        """
        if task.is_expired():
            return CooperativeResponse.error(
                task_id=task.task_id,
                source_agent=self.transport.agent_name,
                target_agent=task.source_agent,
                error_code=ERR_TASK_EXPIRED,
                error_message="Task has expired",
            )

        start = time.time()

        try:
            if handler:
                result = handler(task)
            else:
                result = self._default_handler(task)

            exec_time = int((time.time() - start) * 1000)
            return CooperativeResponse.success(
                task_id=task.task_id,
                source_agent=self.transport.agent_name,
                target_agent=task.source_agent,
                result=result,
                execution_time_ms=exec_time,
                vm_info={
                    "implementation": "flux-coop-runtime-python",
                    "isa_version": "unified-halt-zero",
                    "agent": self.transport.agent_name,
                },
            )
        except Exception as e:
            exec_time = int((time.time() - start) * 1000)
            return CooperativeResponse.error(
                task_id=task.task_id,
                source_agent=self.transport.agent_name,
                target_agent=task.source_agent,
                error_code="HANDLER_ERROR",
                error_message=str(e),
            )

    def send_response(self, response: CooperativeResponse) -> str:
        """Send a response back to the requesting agent."""
        return self.transport.send_response(response)

    def _default_handler(self, task: CooperativeTask) -> Dict[str, Any]:
        """Default task handler when no custom handler is provided."""
        if task.request_type == "ping":
            return {
                "agent": self.transport.agent_name,
                "status": "active",
                "capabilities": self.transport.agent_name + "-capabilities",
            }
        elif task.request_type == "execute_bytecode":
            # Minimal bytecode execution
            bytecode = task.payload.get("bytecode", [])
            return self._execute_bytecode(bytecode, task.payload)
        elif task.request_type == "query_knowledge":
            return {
                "agent": self.transport.agent_name,
                "knowledge_available": True,
                "domains": ["protocol-design", "isa-convergence", "a2a-unification"],
            }
        else:
            raise CooperativeRuntimeError(
                f"Unknown request type: {task.request_type}"
            )

    def _execute_bytecode(self, bytecode: list, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a simple FLUX bytecode program."""
        registers = [0] * 64
        pc = 0
        stack = []

        while pc < len(bytecode):
            op = bytecode[pc]

            if op == 0x00:  # HALT
                break
            elif op == 0x18:  # MOVI rd, imm16
                rd = bytecode[pc + 1]
                imm = bytecode[pc + 2] | (bytecode[pc + 3] << 8)
                registers[rd] = imm
                pc += 4
            elif op == 0x08:  # INC rd
                rd = bytecode[pc + 1]
                registers[rd] += 1
                pc += 2
            elif op == 0x09:  # DEC rd
                rd = bytecode[pc + 1]
                registers[rd] -= 1
                pc += 2
            elif op == 0x20:  # ADD rd, rs1, rs2
                rd = bytecode[pc + 1]
                rs1 = bytecode[pc + 2]
                rs2 = bytecode[pc + 3]
                registers[rd] = registers[rs1] + registers[rs2]
                pc += 4
            elif op == 0x21:  # SUB rd, rs1, rs2
                rd = bytecode[pc + 1]
                rs1 = bytecode[pc + 2]
                rs2 = bytecode[pc + 3]
                registers[rd] = registers[rs1] - registers[rs2]
                pc += 4
            elif op == 0x22:  # MUL rd, rs1, rs2
                rd = bytecode[pc + 1]
                rs1 = bytecode[pc + 2]
                rs2 = bytecode[pc + 3]
                registers[rd] = registers[rs1] * registers[rs2]
                pc += 4
            elif op == 0x0C:  # PUSH rd
                rd = bytecode[pc + 1]
                stack.append(registers[rd])
                pc += 2
            elif op == 0x0D:  # POP rd
                rd = bytecode[pc + 1]
                if stack:
                    registers[rd] = stack.pop()
                pc += 2
            elif op == 0x19:  # ADDI rd, imm16
                rd = bytecode[pc + 1]
                imm = bytecode[pc + 2] | (bytecode[pc + 3] << 8)
                registers[rd] += imm
                pc += 4
            elif op == 0x1A:  # SUBI rd, imm16
                rd = bytecode[pc + 1]
                imm = bytecode[pc + 2] | (bytecode[pc + 3] << 8)
                registers[rd] -= imm
                pc += 4
            else:
                raise CooperativeRuntimeError(
                    f"Unknown opcode: 0x{op:02x} at PC={pc}"
                )

        expected = payload.get("expected_result", "all_registers")
        if expected == "all_registers":
            return {"registers": registers[:8], "stack": stack, "pc": pc}
        elif expected.startswith("register_"):
            idx = int(expected.split("_")[1])
            return {"value": registers[idx]}
        elif expected == "stack_top":
            return {"value": stack[-1] if stack else None}
        else:
            return {"registers": registers[:8], "stack": stack, "pc": pc}

    def get_trust_report(self) -> Dict[str, Any]:
        """Get trust scores for all known agents."""
        return self.trust.get_all_records()

    def get_status(self) -> Dict[str, Any]:
        """Get runtime status."""
        return {
            "agent": self.transport.agent_name,
            "pending_tasks": len(self._pending_tasks),
            "trust_scores": {k: v for k, v in self.trust.get_all_records().items()},
        }
