"""
Agent Capability Negotiation Protocol

Provides structured capability advertisement, fleet-wide discovery,
task-agent matching, negotiation, and contract management for the
FLUX cooperative runtime.
"""

import time
import uuid
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SkillType(str, Enum):
    """Types of skills an agent may possess."""
    COMPUTATION = "computation"
    ANALYSIS = "analysis"
    MEMORY = "memory"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    COMMUNICATION = "communication"
    COORDINATION = "coordination"


class ConstraintType(str, Enum):
    """Types of constraints that may be placed on task assignment."""
    MIN_CONFIDENCE = "min_confidence"
    MAX_LATENCY = "max_latency"
    REQUIRES_TRUST_ABOVE = "requires_trust_above"
    PREFERRED_AGENT = "preferred_agent"


class ContractState(str, Enum):
    """Lifecycle states for a contract."""
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NegotiationOutcome(str, Enum):
    """Possible outcomes of a negotiation."""
    ACCEPT = "accept"
    DECLINE = "decline"
    COUNTER = "counter"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Skill:
    """A single skill with a type, name, and confidence level."""
    skill_type: SkillType
    name: str
    confidence: float = 0.5

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Skill confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if not self.name.strip():
            raise ValueError("Skill name must not be empty")


@dataclass(frozen=True)
class ResourceLimits:
    """Resource limits for an agent."""
    memory_mb: int = 512
    cpu_cores: float = 1.0
    max_bandwidth_mbps: float = 100.0


@dataclass(frozen=True)
class ResourceRequirement:
    """Resource requirements for a task."""
    memory_mb: int = 256
    cpu_cores: float = 0.5
    bandwidth_mbps: float = 10.0


@dataclass
class CapabilityManifest:
    """Structured description of an agent's capabilities."""
    agent_id: str
    name: str
    role: str
    skills: List[Skill] = field(default_factory=list)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    supported_opcodes: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    active_task_count: int = 0

    def __post_init__(self):
        if not self.agent_id.strip():
            raise ValueError("agent_id must not be empty")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.role.strip():
            raise ValueError("role must not be empty")
        if self.max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be >= 1")

    @property
    def available_slots(self) -> int:
        """How many more tasks this agent can accept."""
        return max(0, self.max_concurrent_tasks - self.active_task_count)

    @property
    def is_available(self) -> bool:
        """Whether the agent has any available capacity."""
        return self.available_slots > 0

    def has_skill_type(self, skill_type: SkillType) -> bool:
        """Check if agent has any skill of the given type."""
        return any(s.skill_type == skill_type for s in self.skills)

    def get_skill_confidence(self, skill_type: SkillType) -> float:
        """Get the highest confidence for a given skill type, or 0.0."""
        confidences = [s.confidence for s in self.skills if s.skill_type == skill_type]
        return max(confidences) if confidences else 0.0

    def skill_type_names(self, skill_type: SkillType) -> List[str]:
        """Get names of skills matching a given type."""
        return [s.name for s in self.skills if s.skill_type == skill_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "skills": [
                {"skill_type": s.skill_type.value, "name": s.name, "confidence": s.confidence}
                for s in self.skills
            ],
            "resource_limits": {
                "memory_mb": self.resource_limits.memory_mb,
                "cpu_cores": self.resource_limits.cpu_cores,
                "max_bandwidth_mbps": self.resource_limits.max_bandwidth_mbps,
            },
            "supported_opcodes": list(self.supported_opcodes),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "active_task_count": self.active_task_count,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Constraint:
    """A constraint on task assignment."""
    constraint_type: ConstraintType
    value: Any


@dataclass
class TaskSpec:
    """Structured task specification for matching to agents."""
    required_skills: List[SkillType] = field(default_factory=list)
    priority: int = 5
    estimated_duration: float = 60.0
    resource_requirements: ResourceRequirement = field(default_factory=ResourceRequirement)
    constraints: List[Constraint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_constraint(self, constraint_type: ConstraintType) -> Optional[Any]:
        """Get the value of a constraint, or None if not present."""
        for c in self.constraints:
            if c.constraint_type == constraint_type:
                return c.value
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_skills": [s.value for s in self.required_skills],
            "priority": self.priority,
            "estimated_duration": self.estimated_duration,
            "resource_requirements": {
                "memory_mb": self.resource_requirements.memory_mb,
                "cpu_cores": self.resource_requirements.cpu_cores,
                "bandwidth_mbps": self.resource_requirements.bandwidth_mbps,
            },
            "constraints": [
                {"type": c.constraint_type.value, "value": c.value}
                for c in self.constraints
            ],
            "metadata": dict(self.metadata),
        }


@dataclass
class CandidateScore:
    """Scoring result for an agent-task match."""
    agent_id: str
    manifest: CapabilityManifest
    score: float
    matched_skills: List[SkillType] = field(default_factory=list)
    missing_skills: List[SkillType] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CandidateScore):
            return NotImplemented
        return self.score < other.score


@dataclass
class ContractTerms:
    """Terms agreed upon in a contract."""
    deadline: float = 0.0
    reward: float = 0.0
    penalty_conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deadline": self.deadline,
            "reward": self.reward,
            "penalty_conditions": list(self.penalty_conditions),
        }


@dataclass
class NegotiationResult:
    """Result of a negotiation between an agent and task."""
    outcome: NegotiationOutcome
    message: str = ""
    counter_spec: Optional[TaskSpec] = None
    terms: Optional[ContractTerms] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "message": self.message,
            "counter_spec": self.counter_spec.to_dict() if self.counter_spec else None,
            "terms": self.terms.to_dict() if self.terms else None,
        }


@dataclass
class Contract:
    """A binding agreement between an agent and a task."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent_id: str = ""
    task_spec: Optional[TaskSpec] = None
    terms: Optional[ContractTerms] = None
    state: ContractState = ContractState.PROPOSED
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    completed_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.agent_id.strip():
            raise ValueError("agent_id must not be empty")
        if not self.expires_at:
            self.expires_at = self.created_at + 3600.0  # 1 hour default

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "state": self.state.value,
            "terms": self.terms.to_dict() if self.terms else None,
            "task_spec": self.task_spec.to_dict() if self.task_spec else None,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "completed_at": self.completed_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class PreferenceRecord:
    """Tracks agent-task type preferences for learning."""
    agent_id: str
    skill_type: SkillType
    count: int = 0
    success_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.count == 0:
            return 0.5
        return self.success_count / self.count


# ---------------------------------------------------------------------------
# Capability Registry
# ---------------------------------------------------------------------------

class CapabilityRegistry:
    """Fleet-wide capability discovery registry."""

    def __init__(self):
        self._agents: Dict[str, CapabilityManifest] = {}
        self._trust_scores: Dict[str, float] = {}

    # -- Mutation -----------------------------------------------------------

    def register(self, manifest: CapabilityManifest) -> None:
        """Register an agent manifest."""
        if manifest.agent_id in self._agents:
            raise ValueError(
                f"Agent {manifest.agent_id} is already registered; unregister first"
            )
        self._agents[manifest.agent_id] = manifest
        # Seed trust if not present
        if manifest.agent_id not in self._trust_scores:
            self._trust_scores[manifest.agent_id] = 0.5

    def unregister(self, agent_id: str) -> Optional[CapabilityManifest]:
        """Unregister an agent; returns the removed manifest or None."""
        return self._agents.pop(agent_id, None)

    def update_active_tasks(self, agent_id: str, delta: int) -> None:
        """Adjust active_task_count by *delta* for the given agent."""
        m = self._agents.get(agent_id)
        if m is None:
            raise KeyError(f"Agent {agent_id} not registered")
        m.active_task_count = max(0, m.active_task_count + delta)

    def set_trust(self, agent_id: str, score: float) -> None:
        """Manually set the trust score for an agent."""
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Trust score must be in [0.0, 1.0], got {score}")
        self._trust_scores[agent_id] = score

    def record_trust(self, agent_id: str, success: bool) -> None:
        """Update trust score with a binary success/failure signal (EMA)."""
        current = self._trust_scores.get(agent_id, 0.5)
        alpha = 0.2
        signal = 1.0 if success else 0.0
        self._trust_scores[agent_id] = alpha * signal + (1 - alpha) * current

    # -- Query --------------------------------------------------------------

    def get(self, agent_id: str) -> Optional[CapabilityManifest]:
        return self._agents.get(agent_id)

    def get_trust(self, agent_id: str) -> float:
        return self._trust_scores.get(agent_id, 0.5)

    def all_agents(self) -> List[CapabilityManifest]:
        return list(self._agents.values())

    def available_agents(self) -> List[CapabilityManifest]:
        return [m for m in self._agents.values() if m.is_available]

    def query_by_skill_type(
        self, skill_type: SkillType
    ) -> List[CapabilityManifest]:
        """Return agents that possess at least one skill of the given type."""
        return [m for m in self._agents.values() if m.has_skill_type(skill_type)]

    def query_by_role(self, role: str) -> List[CapabilityManifest]:
        """Return agents whose role matches the given string (exact)."""
        return [m for m in self._agents.values() if m.role == role]

    def query_by_name_pattern(self, pattern: str) -> List[CapabilityManifest]:
        """Return agents whose name contains *pattern* (case-insensitive)."""
        lower = pattern.lower()
        return [m for m in self._agents.values() if lower in m.name.lower()]

    def find_best_agent(self, skill_types: List[SkillType]) -> Optional[CandidateScore]:
        """Find the best available agent for a set of required skill types.

        Scoring: 0.5 × skill_coverage + 0.3 × avg_confidence + 0.2 × availability_bonus
        """
        candidates = self._score_candidates(skill_types)
        available = [c for c in candidates if c.manifest.is_available]
        return max(available, key=lambda c: c.score, default=None)

    # -- Internals ----------------------------------------------------------

    def _score_candidates(
        self, skill_types: List[SkillType]
    ) -> List[CandidateScore]:
        results: List[CandidateScore] = []
        required_set: Set[SkillType] = set(skill_types)

        for manifest in self._agents.values():
            matched: List[SkillType] = []
            missing: List[SkillType] = []
            for st in skill_types:
                if manifest.has_skill_type(st):
                    matched.append(st)
                else:
                    missing.append(st)

            coverage = len(matched) / max(len(skill_types), 1)
            confidences = [manifest.get_skill_confidence(st) for st in matched]
            avg_conf = sum(confidences) / max(len(confidences), 1)

            # Availability bonus: 1.0 if fully available, scaled down linearly
            slots = manifest.available_slots
            max_slots = manifest.max_concurrent_tasks
            avail_bonus = slots / max(max_slots, 1)

            trust = self._trust_scores.get(manifest.agent_id, 0.5)

            score = (0.4 * coverage) + (0.3 * avg_conf) + (0.15 * avail_bonus) + (0.15 * trust)

            reasons: List[str] = []
            if missing:
                reasons.append(f"missing skills: {[s.value for s in missing]}")
            if not manifest.is_available:
                reasons.append("agent at capacity")

            results.append(CandidateScore(
                agent_id=manifest.agent_id,
                manifest=manifest,
                score=score,
                matched_skills=matched,
                missing_skills=missing,
                reasons=reasons,
            ))
        return results


# ---------------------------------------------------------------------------
# Negotiation Engine
# ---------------------------------------------------------------------------

class NegotiationEngine:
    """Matches tasks to agents via scored candidate selection and negotiation."""

    def __init__(self, registry: CapabilityRegistry):
        self._registry = registry

    # -- Candidate Discovery ------------------------------------------------

    def find_candidates(self, task_spec: TaskSpec) -> List[CandidateScore]:
        """Return candidates sorted by score (best first)."""
        raw = self._registry._score_candidates(task_spec.required_skills)
        scored: List[CandidateScore] = []
        for cs in raw:
            cs = self._apply_constraints(cs, task_spec)
            scored.append(cs)
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored

    # -- Negotiation --------------------------------------------------------

    def negotiate(
        self, agent_id: str, task_spec: TaskSpec
    ) -> NegotiationResult:
        """Attempt to negotiate a task assignment with a specific agent.

        The agent *accepts* if:
        - all required skills are present
        - confidence meets MIN_CONFIDENCE constraint
        - trust meets REQUIRES_TRUST_ABOVE constraint
        - the agent has capacity

        Otherwise returns DECLINE with reasons, or COUNTER if possible.
        """
        manifest = self._registry.get(agent_id)
        if manifest is None:
            return NegotiationResult(
                outcome=NegotiationOutcome.DECLINE,
                message=f"Agent {agent_id} not found in registry",
            )

        # Capacity check
        if not manifest.is_available:
            return NegotiationResult(
                outcome=NegotiationOutcome.DECLINE,
                message="Agent is at full capacity (overloaded)",
            )

        # Skill match
        matched: List[SkillType] = []
        missing: List[SkillType] = []
        for st in task_spec.required_skills:
            if manifest.has_skill_type(st):
                matched.append(st)
            else:
                missing.append(st)

        if missing:
            return NegotiationResult(
                outcome=NegotiationOutcome.DECLINE,
                message=f"Agent underqualified: missing {', '.join(s.value for s in missing)}",
            )

        # Constraint checks
        for constraint in task_spec.constraints:
            violation = self._check_constraint(manifest, constraint)
            if violation:
                if constraint.constraint_type == ConstraintType.MIN_CONFIDENCE:
                    # Counter-offer: relax confidence or suggest partial task
                    return self._maybe_counter(manifest, task_spec, violation, constraint)
                return NegotiationResult(
                    outcome=NegotiationOutcome.DECLINE,
                    message=violation,
                )

        # Compute score for terms
        confidences = [manifest.get_skill_confidence(st) for st in matched]
        avg_conf = sum(confidences) / max(len(confidences), 1)
        deadline = time.time() + task_spec.estimated_duration
        terms = ContractTerms(
            deadline=deadline,
            reward=round(avg_conf * 10.0, 2),
        )

        return NegotiationResult(
            outcome=NegotiationOutcome.ACCEPT,
            message="Task accepted",
            terms=terms,
        )

    def counter_offer(
        self, agent_id: str, task_spec: TaskSpec, modified_spec: TaskSpec
    ) -> NegotiationResult:
        """Attempt to re-negotiate with a modified task spec."""
        manifest = self._registry.get(agent_id)
        if manifest is None:
            return NegotiationResult(
                outcome=NegotiationOutcome.DECLINE,
                message=f"Agent {agent_id} not found in registry",
            )

        # Check capacity
        if not manifest.is_available:
            return NegotiationResult(
                outcome=NegotiationOutcome.DECLINE,
                message="Agent is at full capacity",
            )

        # Check skills against modified spec
        for st in modified_spec.required_skills:
            if not manifest.has_skill_type(st):
                return NegotiationResult(
                    outcome=NegotiationOutcome.DECLINE,
                    message=f"Agent underqualified for modified spec: missing {st.value}",
                )

        # Check constraints against modified spec
        for constraint in modified_spec.constraints:
            violation = self._check_constraint(manifest, constraint)
            if violation:
                return NegotiationResult(
                    outcome=NegotiationOutcome.DECLINE,
                    message=f"Constraint still violated: {violation}",
                )

        confidences = [manifest.get_skill_confidence(st) for st in modified_spec.required_skills]
        avg_conf = sum(confidences) / max(len(confidences), 1)
        deadline = time.time() + modified_spec.estimated_duration
        terms = ContractTerms(
            deadline=deadline,
            reward=round(avg_conf * 10.0, 2),
        )

        return NegotiationResult(
            outcome=NegotiationOutcome.ACCEPT,
            message=f"Counter-offer accepted (modified from original)",
            terms=terms,
        )

    # -- Internals ----------------------------------------------------------

    def _apply_constraints(
        self, cs: CandidateScore, task_spec: TaskSpec
    ) -> CandidateScore:
        """Adjust candidate score based on constraints."""
        manifest = cs.manifest
        for constraint in task_spec.constraints:
            violation = self._check_constraint(manifest, constraint)
            if violation:
                cs.score *= 0.3  # Heavy penalty for constraint violation
                cs.reasons.append(violation)

        # Bonus for preferred agent
        preferred = task_spec.get_constraint(ConstraintType.PREFERRED_AGENT)
        if preferred and cs.agent_id == preferred:
            cs.score *= 1.2  # 20% bonus
            cs.reasons.append("preferred agent bonus applied")
            # Cap at 1.0
            cs.score = min(cs.score, 1.0)

        return cs

    def _check_constraint(
        self, manifest: CapabilityManifest, constraint: Constraint
    ) -> Optional[str]:
        """Return a violation message or None if satisfied."""
        if constraint.constraint_type == ConstraintType.MIN_CONFIDENCE:
            min_conf = float(constraint.value)
            for st in self._registry._score_candidates([]):
                pass  # we need skill-level check
            for skill in manifest.skills:
                if skill.confidence < min_conf:
                    return f"Skill {skill.name} confidence {skill.confidence} below minimum {min_conf}"
            # Also check required skills generally
            return None

        if constraint.constraint_type == ConstraintType.MAX_LATENCY:
            # Latency estimation not modelled here; always pass
            return None

        if constraint.constraint_type == ConstraintType.REQUIRES_TRUST_ABOVE:
            min_trust = float(constraint.value)
            trust = self._registry.get_trust(manifest.agent_id)
            if trust < min_trust:
                return f"Trust {trust:.2f} below required {min_trust}"
            return None

        if constraint.constraint_type == ConstraintType.PREFERRED_AGENT:
            # Preference is a soft constraint; handled as bonus in scoring
            return None

        return None

    def _maybe_counter(
        self,
        manifest: CapabilityManifest,
        task_spec: TaskSpec,
        violation: str,
        constraint: Constraint,
    ) -> NegotiationResult:
        """Attempt to produce a counter-offer by relaxing constraints."""
        if constraint.constraint_type == ConstraintType.MIN_CONFIDENCE:
            # Lower confidence to match the agent's best
            min_existing = min(
                s.confidence for s in manifest.skills
                if s.skill_type in set(task_spec.required_skills)
            ) if manifest.skills else 0.0
            new_constraints = [
                c for c in task_spec.constraints
                if c.constraint_type != ConstraintType.MIN_CONFIDENCE
            ]
            new_constraints.append(
                Constraint(ConstraintType.MIN_CONFIDENCE, round(min_existing, 2))
            )
            counter = TaskSpec(
                required_skills=list(task_spec.required_skills),
                priority=task_spec.priority,
                estimated_duration=task_spec.estimated_duration * 1.2,
                resource_requirements=task_spec.resource_requirements,
                constraints=new_constraints,
                metadata=dict(task_spec.metadata),
            )
            return NegotiationResult(
                outcome=NegotiationOutcome.COUNTER,
                message=f"Counter-offer: relaxed MIN_CONFIDENCE to {min_existing:.2f}",
                counter_spec=counter,
            )

        return NegotiationResult(
            outcome=NegotiationOutcome.DECLINE,
            message=violation,
        )


# ---------------------------------------------------------------------------
# Contract Registry
# ---------------------------------------------------------------------------

class ContractRegistry:
    """Tracks active contracts between agents and tasks."""

    def __init__(self, capability_registry: CapabilityRegistry):
        self._contracts: Dict[str, Contract] = {}
        self._registry = capability_registry

    def create(
        self,
        agent_id: str,
        task_spec: TaskSpec,
        terms: Optional[ContractTerms] = None,
        expires_at: Optional[float] = None,
    ) -> Contract:
        """Create a new contract in PROPOSED state."""
        contract = Contract(
            agent_id=agent_id,
            task_spec=task_spec,
            terms=terms or ContractTerms(),
            state=ContractState.PROPOSED,
            expires_at=expires_at or (time.time() + 3600.0),
        )
        self._contracts[contract.id] = contract
        return contract

    def accept(self, contract_id: str) -> Contract:
        """Move contract from PROPOSED → ACCEPTED."""
        c = self._get(contract_id)
        if c.state != ContractState.PROPOSED:
            raise ValueError(f"Cannot accept contract in state {c.state.value}")
        c.state = ContractState.ACCEPTED
        return c

    def start(self, contract_id: str) -> Contract:
        """Move contract from ACCEPTED → IN_PROGRESS and increment agent load."""
        c = self._get(contract_id)
        if c.state != ContractState.ACCEPTED:
            raise ValueError(f"Cannot start contract in state {c.state.value}")
        c.state = ContractState.IN_PROGRESS
        self._registry.update_active_tasks(c.agent_id, delta=1)
        return c

    def complete(self, contract_id: str, success: bool = True) -> Contract:
        """Move contract from IN_PROGRESS → COMPLETED or FAILED."""
        c = self._get(contract_id)
        if c.state != ContractState.IN_PROGRESS:
            raise ValueError(f"Cannot complete contract in state {c.state.value}")
        c.state = ContractState.COMPLETED if success else ContractState.FAILED
        c.completed_at = time.time()
        self._registry.update_active_tasks(c.agent_id, delta=-1)
        self._registry.record_trust(c.agent_id, success)
        return c

    def cancel(self, contract_id: str) -> Contract:
        """Move contract to CANCELLED from any non-terminal state."""
        c = self._get(contract_id)
        if c.state in (ContractState.COMPLETED, ContractState.FAILED, ContractState.CANCELLED):
            raise ValueError(f"Cannot cancel contract in terminal state {c.state.value}")
        if c.state == ContractState.IN_PROGRESS:
            self._registry.update_active_tasks(c.agent_id, delta=-1)
        c.state = ContractState.CANCELLED
        return c

    def get(self, contract_id: str) -> Optional[Contract]:
        return self._contracts.get(contract_id)

    def contracts_for_agent(self, agent_id: str) -> List[Contract]:
        return [c for c in self._contracts.values() if c.agent_id == agent_id]

    def active_contracts(self) -> List[Contract]:
        """Return contracts in non-terminal states."""
        terminal = {
            ContractState.COMPLETED,
            ContractState.FAILED,
            ContractState.CANCELLED,
        }
        return [c for c in self._contracts.values() if c.state not in terminal]

    def all_contracts(self) -> List[Contract]:
        return list(self._contracts.values())

    def cleanup_expired(self) -> List[Contract]:
        """Cancel all expired contracts. Returns list of cancelled contracts."""
        cancelled: List[Contract] = []
        for c in list(self._contracts.values()):
            if c.is_expired() and c.state not in (
                ContractState.COMPLETED,
                ContractState.FAILED,
                ContractState.CANCELLED,
            ):
                self.cancel(c.id)
                cancelled.append(c)
        return cancelled

    def _get(self, contract_id: str) -> Contract:
        c = self._contracts.get(contract_id)
        if c is None:
            raise KeyError(f"Contract {contract_id} not found")
        return c


# ---------------------------------------------------------------------------
# Fleet Directory
# ---------------------------------------------------------------------------

class FleetDirectory:
    """Combined view: agents, contracts, availability, load balancing, preferences."""

    def __init__(
        self,
        capability_registry: CapabilityRegistry,
        contract_registry: ContractRegistry,
    ):
        self._cap_reg = capability_registry
        self._con_reg = contract_registry
        self._preferences: Dict[str, List[PreferenceRecord]] = {}

    # -- Fleet Overview -----------------------------------------------------

    def overview(self) -> Dict[str, Any]:
        """Return a snapshot of the fleet state."""
        agents = self._cap_reg.all_agents()
        active = self._con_reg.active_contracts()
        return {
            "total_agents": len(agents),
            "available_agents": len(self._cap_reg.available_agents()),
            "active_contracts": len(active),
            "total_contracts": len(self._con_reg.all_contracts()),
            "agents": [
                {
                    "agent_id": m.agent_id,
                    "name": m.name,
                    "role": m.role,
                    "available_slots": m.available_slots,
                    "active_tasks": m.active_task_count,
                    "is_available": m.is_available,
                }
                for m in agents
            ],
        }

    # -- Load Balancing -----------------------------------------------------

    def select_agent(
        self, task_spec: TaskSpec, exclude: Optional[Set[str]] = None
    ) -> Optional[CandidateScore]:
        """Select the best agent considering load balance and preferences.

        Uses least-loaded selection among top-scoring candidates.
        """
        exclude = exclude or set()
        engine = NegotiationEngine(self._cap_reg)
        candidates = engine.find_candidates(task_spec)
        # Filter excluded agents and unavailable ones
        filtered = [
            c for c in candidates
            if c.agent_id not in exclude and c.manifest.is_available
        ]
        if not filtered:
            return None

        # Among candidates within 10% of top score, pick least loaded
        top_score = filtered[0].score
        threshold = top_score * 0.9
        near_top = [c for c in filtered if c.score >= threshold]

        # Preference bonus
        for c in near_top:
            pref = self._get_preference_score(c.agent_id, task_spec)
            c.score += pref * 0.05  # Small bonus for preference

        # Sort by active tasks ascending, then score descending
        near_top.sort(key=lambda c: (c.manifest.active_task_count, -c.score))
        return near_top[0]

    # -- Preference Learning ------------------------------------------------

    def record_preference(
        self, agent_id: str, skill_type: SkillType, success: bool = True
    ) -> None:
        """Record that an agent handled a task of a given skill type."""
        if agent_id not in self._preferences:
            self._preferences[agent_id] = []
        records = self._preferences[agent_id]
        for r in records:
            if r.skill_type == skill_type:
                r.count += 1
                if success:
                    r.success_count += 1
                return
        records.append(PreferenceRecord(
            agent_id=agent_id,
            skill_type=skill_type,
            count=1,
            success_count=1 if success else 0,
        ))

    def get_preferences(self, agent_id: str) -> List[PreferenceRecord]:
        return list(self._preferences.get(agent_id, []))

    def get_preferred_agents(self, skill_type: SkillType) -> List[Tuple[str, float]]:
        """Return (agent_id, preference_score) sorted by preference descending."""
        scores: List[Tuple[str, float]] = []
        for agent_id, records in self._preferences.items():
            for r in records:
                if r.skill_type == skill_type:
                    scores.append((agent_id, r.success_rate * r.count))
                    break
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # -- Internals ----------------------------------------------------------

    def _get_preference_score(self, agent_id: str, task_spec: TaskSpec) -> float:
        """Compute a preference score for the agent given the task's skills."""
        records = self._preferences.get(agent_id, [])
        if not records or not task_spec.required_skills:
            return 0.0
        total = 0.0
        for st in task_spec.required_skills:
            for r in records:
                if r.skill_type == st:
                    total += r.success_rate * r.count
        return total
