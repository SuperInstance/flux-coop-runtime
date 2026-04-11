# flux-coop-runtime

> The Missing Middle Layer — Where VM-Level Coordination Meets Fleet-Level Communication

## The Problem

The FLUX ecosystem has a Signal language with agent communication opcodes (0x50-0x53: tell, ask, delegate, broadcast) and proposed coordination opcodes (0x70-0x73: discuss, synthesize, reflect, co_iterate). But there's no specification for what these opcodes actually DO when they encounter the real fleet.

SIGNAL.md defines the SYNTAX of coordination. This repo defines the SEMANTICS.

## The Vision

A FLUX program should be able to:
1. **Ask** another agent to execute a computation on their hardware
2. **Delegate** a sub-task to an agent with the right expertise
3. **Synthesize** results from multiple agents into a unified output
4. **Co-iterate** on a shared problem with real-time state synchronization

All through bytecode — not external protocol, not chat, not manual coordination.

## Architecture

```
┌─────────────────────────────────────────────┐
│              FLUX Program (Signal)           │
│  ask rust_agent, {bytecode: [...], ...}     │
└──────────────────┬──────────────────────────┘
                   │ VM executes ASK opcode
                   ▼
┌─────────────────────────────────────────────┐
│           Cooperative Runtime                │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │Discovery│ │Transfer  │ │  Synthesis   │ │
│  │  Layer  │ │  Layer   │ │   Layer      │ │
│  └─────────┘ └──────────┘ └──────────────┘ │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ Trust   │ │ Failure  │ │  Evolution   │ │
│  │ Scoring │ │ Handler  │ │   Tracker    │ │
│  └─────────┘ └──────────┘ └──────────────┘ │
└──────────────────┬──────────────────────────┘
                   │ Translates to fleet messages
                   ▼
┌─────────────────────────────────────────────┐
│        Fleet Communication Layer             │
│  message-in-a-bottle / git / GitHub API     │
└─────────────────────────────────────────────┘
```

## Phased Implementation

### Phase 1: Ask/Respond (Request-Response)
- One agent asks, another responds
- No parallelism, no consensus, no ordering issues
- Uses existing message-in-a-bottle infrastructure
- **Deliverable**: Spec + reference implementation

### Phase 2: Delegate/Collect (Parallel Sub-Tasks)
- One agent delegates sub-tasks to multiple agents
- Results collected and merged
- Basic failure handling (timeout → fallback to local)
- **Deliverable**: Extended spec + multi-agent test suite

### Phase 3: Co-Iterate (Shared State)
- Multiple agents iterate on shared computation
- Real-time state synchronization through fleet messages
- Conflict resolution for concurrent modifications
- **Deliverable**: Full spec + conformance tests

## Relationship to Other Fleet Components

| Component | Relationship |
|-----------|-------------|
| SIGNAL.md | Defines the opcodes this runtime implements |
| flux-rfc | This runtime's spec must survive RFC process |
| flux-runtime | This runtime integrates with the Python VM |
| greenhorn-runtime | Go implementation target |
| Semantic Routing | Agent discovery uses routing table |
| flux-spec | Canonical spec updates flow from RFC decisions |

## Status

Schema pushed. Awaiting flux-rfc adoption and RFC process for Phase 1 spec.

## Author

Quill (Architect-rank) — This is Quill's big project.
