# flux-coop-runtime Schema

## Repository Structure

```
flux-coop-runtime/
├── README.md                    # Overview and architecture
├── SCHEMA.md                    # This file
├── PHASE-1-SPEC.md              # Phase 1: Ask/Respond specification
├── PHASE-2-SPEC.md              # Phase 2: Delegate/Collect (planned)
├── PHASE-3-SPEC.md              # Phase 3: Co-Iterate (planned)
├── rfc/
│   └── 0002-cooperative-runtime.md  # RFC for this spec
├── src/
│   ├── discovery/               # Agent discovery layer
│   │   ├── resolver.py          # Route tasks to capable agents
│   │   ├── capability_match.py  # Match task requirements to agent skills
│   │   └── tests/
│   ├── transfer/                # State transfer layer
│   │   ├── serializer.py        # Serialize VM state for handoff
│   │   ├── deserializer.py      # Deserialize VM state from agent
│   │   ├── format.py            # FluxTransfer binary format
│   │   └── tests/
│   ├── synthesis/               # Result synthesis layer
│   │   ├── merger.py            # Merge results from multiple agents
│   │   ├── conflict.py          # Resolve conflicting results
│   │   └── tests/
│   ├── trust/                   # Trust scoring layer
│   │   ├── scorer.py            # Score agent reliability
│   │   ├── history.py           # Track agent result history
│   │   └── tests/
│   ├── failure/                 # Failure handling layer
│   │   ├── handler.py           # Timeout, retry, fallback logic
│   │   ├── circuit_breaker.py   # Stop asking unreliable agents
│   │   └── tests/
│   ├── evolution/               # Evolution tracking layer
│   │   ├── observer.py          # Observe cooperative executions
│   │   ├── analyzer.py          # Analyze cooperation patterns
│   │   └── tests/
│   └── runtime.py               # Main cooperative runtime entry point
├── tests/
│   ├── test_phase1.py           # Phase 1 integration tests
│   ├── test_phase2.py           # Phase 2 integration tests
│   └── test_phase3.py           # Phase 3 integration tests
├── message-in-a-bottle/
│   └── for-fleet/
└── examples/
    ├── ask_respond.py           # Phase 1 example
    ├── delegate_collect.py      # Phase 2 example
    └── co_iterate.py            # Phase 3 example
```

## Core Data Types

### CooperativeTask
```python
@dataclass
class CooperativeTask:
    task_id: str           # Unique task identifier
    source_agent: str      # Agent requesting help
    target_agent: str      # Agent being asked (or "any" for discovery)
    opcode: int            # ASK/DELEGATE/BROADCAST
    payload: bytes         # Serialized VM state or bytecode
    query: str             # Human-readable description of what's needed
    timeout_ms: int        # Maximum wait time
    trust_threshold: float # Minimum agent trust score
    fallback: str          # "local" | "none" — what to do on failure
```

### CooperativeResponse
```python
@dataclass
class CooperativeResponse:
    task_id: str
    source_agent: str
    result: bytes          # Serialized result
    status: str            # "success" | "error" | "timeout" | "refused"
    execution_time_ms: int
    agent_signature: str   # For trust scoring
    error_message: str     # If status == "error"
```

### FluxTransfer Format
```python
# Binary format for VM state transfer
struct FluxTransfer {
    magic: u32         # "FXTR" (0x46585452)
    version: u8        # Format version
    isa_version: u8    # ISA compatibility check
    source_pc: u32     # Program counter at transfer point
    data_stack_size: u32
    data_stack: bytes  # Serialized data stack
    signal_stack_size: u32
    signal_stack: bytes
    registers: bytes   # 64 x int32 = 256 bytes
    confidence: bytes  # 64 x int32 = 256 bytes
    metadata_size: u32
    metadata: bytes    # JSON-encoded task context
    checksum: u32      # CRC32
}
```

## Opcode Semantics (Phase 1)

### ASK (0x51)
```
ASK target_agent, {payload, query, timeout}

1. Runtime serializes current VM state as FluxTransfer
2. Discovery layer resolves target_agent to fleet address
3. Transfer layer sends FluxTransfer via message-in-a-bottle
4. Runtime suspends current thread (saves PC, waits)
5. On response: deserialize result, push to data stack
6. On timeout: execute fallback or raise CooperativeError
```

### DELEGATE (0x53)
```
DELEGATE target_agent, {payload, query}

Like ASK but:
- Does NOT suspend (fire-and-forget pattern)
- Result arrives asynchronously via CALLBACK opcode
- Used for parallel task distribution
```

## Commit Convention

```
coop(phase-N): description [component]

Components: discovery, transfer, synthesis, trust, failure, evolution
Phases: 1 (ask/respond), 2 (delegate/collect), 3 (co-iterate)
```
