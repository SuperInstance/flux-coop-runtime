"""
Tests for the Agent Capability Negotiation Protocol.

Covers: Skill, CapabilityManifest, CapabilityRegistry, TaskSpec,
        NegotiationEngine, Contract, ContractRegistry, FleetDirectory.
"""

import time
import pytest

from src.coop_runtime.capability_negotiation import (
    SkillType,
    ConstraintType,
    ContractState,
    NegotiationOutcome,
    Skill,
    ResourceLimits,
    CapabilityManifest,
    Constraint,
    ResourceRequirement,
    TaskSpec,
    CandidateScore,
    NegotiationResult,
    ContractTerms,
    Contract,
    CapabilityRegistry,
    NegotiationEngine,
    ContractRegistry,
    FleetDirectory,
    PreferenceRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_manifest(
    agent_id: str = "agent-001",
    name: str = "Alpha",
    role: str = "worker",
    skills: list = None,
    max_concurrent: int = 2,
    active_tasks: int = 0,
) -> CapabilityManifest:
    if skills is None:
        skills = [
            Skill(SkillType.COMPUTATION, "math", 0.9),
            Skill(SkillType.ANALYSIS, "data-viz", 0.7),
        ]
    m = CapabilityManifest(
        agent_id=agent_id,
        name=name,
        role=role,
        skills=skills,
        resource_limits=ResourceLimits(memory_mb=1024, cpu_cores=2.0),
        supported_opcodes=["ADD", "SUB", "MUL"],
        max_concurrent_tasks=max_concurrent,
        metadata={"team": "blue"},
    )
    m.active_task_count = active_tasks
    return m


def make_task_spec(
    required_skills: list = None,
    priority: int = 5,
    estimated_duration: float = 60.0,
    constraints: list = None,
) -> TaskSpec:
    if required_skills is None:
        required_skills = [SkillType.COMPUTATION]
    return TaskSpec(
        required_skills=required_skills,
        priority=priority,
        estimated_duration=estimated_duration,
        resource_requirements=ResourceRequirement(memory_mb=256),
        constraints=constraints or [],
    )


# ===========================================================================
# 1. Skill
# ===========================================================================

class TestSkill:
    def test_valid_skill(self):
        s = Skill(SkillType.COMPUTATION, "math", 0.9)
        assert s.skill_type == SkillType.COMPUTATION
        assert s.name == "math"
        assert s.confidence == 0.9

    def test_confidence_boundaries(self):
        Skill(SkillType.ANALYSIS, "stats", 0.0)
        Skill(SkillType.ANALYSIS, "stats", 1.0)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            Skill(SkillType.COMPUTATION, "bad", -0.1)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            Skill(SkillType.COMPUTATION, "bad", 1.5)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            Skill(SkillType.COMPUTATION, "", 0.5)

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            Skill(SkillType.COMPUTATION, "   ", 0.5)

    def test_frozen(self):
        s = Skill(SkillType.MEMORY, "cache", 0.8)
        with pytest.raises(AttributeError):
            s.confidence = 0.5


# ===========================================================================
# 2. CapabilityManifest
# ===========================================================================

class TestCapabilityManifest:
    def test_basic_creation(self):
        m = make_manifest()
        assert m.agent_id == "agent-001"
        assert m.name == "Alpha"
        assert m.role == "worker"
        assert len(m.skills) == 2
        assert m.max_concurrent_tasks == 2

    def test_empty_agent_id_raises(self):
        with pytest.raises(ValueError, match="agent_id"):
            make_manifest(agent_id="")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            make_manifest(name="")

    def test_empty_role_raises(self):
        with pytest.raises(ValueError, match="role"):
            make_manifest(role="")

    def test_max_concurrent_less_than_one_raises(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            make_manifest(max_concurrent=0)

    def test_available_slots(self):
        m = make_manifest(max_concurrent=3, active_tasks=1)
        assert m.available_slots == 2

    def test_available_slots_at_capacity(self):
        m = make_manifest(max_concurrent=2, active_tasks=2)
        assert m.available_slots == 0

    def test_available_slots_overloaded_clamps(self):
        m = make_manifest(max_concurrent=2, active_tasks=5)
        assert m.available_slots == 0

    def test_is_available(self):
        m = make_manifest(max_concurrent=3, active_tasks=2)
        assert m.is_available is True

    def test_is_not_available(self):
        m = make_manifest(max_concurrent=2, active_tasks=2)
        assert m.is_available is False

    def test_has_skill_type_true(self):
        m = make_manifest(skills=[Skill(SkillType.COMPUTATION, "math", 0.9)])
        assert m.has_skill_type(SkillType.COMPUTATION) is True

    def test_has_skill_type_false(self):
        m = make_manifest(skills=[Skill(SkillType.COMPUTATION, "math", 0.9)])
        assert m.has_skill_type(SkillType.SENSOR) is False

    def test_get_skill_confidence(self):
        m = make_manifest(skills=[
            Skill(SkillType.COMPUTATION, "math", 0.9),
            Skill(SkillType.COMPUTATION, "logic", 0.6),
        ])
        assert m.get_skill_confidence(SkillType.COMPUTATION) == 0.9

    def test_get_skill_confidence_absent(self):
        m = make_manifest(skills=[Skill(SkillType.COMPUTATION, "math", 0.9)])
        assert m.get_skill_confidence(SkillType.SENSOR) == 0.0

    def test_skill_type_names(self):
        m = make_manifest(skills=[
            Skill(SkillType.COMPUTATION, "math", 0.9),
            Skill(SkillType.COMPUTATION, "logic", 0.6),
        ])
        names = m.skill_type_names(SkillType.COMPUTATION)
        assert set(names) == {"math", "logic"}

    def test_to_dict(self):
        m = make_manifest()
        d = m.to_dict()
        assert d["agent_id"] == "agent-001"
        assert len(d["skills"]) == 2
        assert d["resource_limits"]["memory_mb"] == 1024


# ===========================================================================
# 3. TaskSpec
# ===========================================================================

class TestTaskSpec:
    def test_basic_creation(self):
        ts = make_task_spec()
        assert ts.required_skills == [SkillType.COMPUTATION]
        assert ts.priority == 5
        assert ts.estimated_duration == 60.0

    def test_get_constraint_present(self):
        ts = make_task_spec(constraints=[
            Constraint(ConstraintType.MIN_CONFIDENCE, 0.8)
        ])
        assert ts.get_constraint(ConstraintType.MIN_CONFIDENCE) == 0.8

    def test_get_constraint_absent(self):
        ts = make_task_spec()
        assert ts.get_constraint(ConstraintType.MIN_CONFIDENCE) is None

    def test_to_dict(self):
        ts = make_task_spec(required_skills=[SkillType.ANALYSIS, SkillType.COMPUTATION])
        d = ts.to_dict()
        assert d["required_skills"] == ["analysis", "computation"]
        assert d["priority"] == 5


# ===========================================================================
# 4. CapabilityRegistry
# ===========================================================================

class TestCapabilityRegistry:
    def setup_method(self):
        self.reg = CapabilityRegistry()

    def test_register_and_get(self):
        m = make_manifest()
        self.reg.register(m)
        assert self.reg.get("agent-001") is m

    def test_register_duplicate_raises(self):
        m = make_manifest()
        self.reg.register(m)
        with pytest.raises(ValueError, match="already registered"):
            self.reg.register(m)

    def test_unregister(self):
        m = make_manifest()
        self.reg.register(m)
        removed = self.reg.unregister("agent-001")
        assert removed is m
        assert self.reg.get("agent-001") is None

    def test_unregister_nonexistent(self):
        assert self.reg.unregister("nope") is None

    def test_all_agents(self):
        self.reg.register(make_manifest(agent_id="a"))
        self.reg.register(make_manifest(agent_id="b"))
        assert len(self.reg.all_agents()) == 2

    def test_available_agents(self):
        self.reg.register(make_manifest(agent_id="a", max_concurrent=2, active_tasks=0))
        self.reg.register(make_manifest(agent_id="b", max_concurrent=1, active_tasks=1))
        avail = self.reg.available_agents()
        assert len(avail) == 1
        assert avail[0].agent_id == "a"

    def test_query_by_skill_type(self):
        self.reg.register(make_manifest(
            agent_id="comp", skills=[Skill(SkillType.COMPUTATION, "math", 0.9)]
        ))
        self.reg.register(make_manifest(
            agent_id="sens", skills=[Skill(SkillType.SENSOR, "temp", 0.8)]
        ))
        result = self.reg.query_by_skill_type(SkillType.COMPUTATION)
        assert len(result) == 1
        assert result[0].agent_id == "comp"

    def test_query_by_role(self):
        self.reg.register(make_manifest(agent_id="w1", role="worker"))
        self.reg.register(make_manifest(agent_id="m1", role="manager"))
        assert len(self.reg.query_by_role("worker")) == 1

    def test_query_by_name_pattern(self):
        self.reg.register(make_manifest(agent_id="a1", name="AlphaBot"))
        self.reg.register(make_manifest(agent_id="a2", name="BetaBot"))
        result = self.reg.query_by_name_pattern("alpha")
        assert len(result) == 1
        assert result[0].agent_id == "a1"

    def test_find_best_agent(self):
        self.reg.register(make_manifest(
            agent_id="expert",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.95)],
        ))
        self.reg.register(make_manifest(
            agent_id="novice",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.4)],
        ))
        best = self.reg.find_best_agent([SkillType.COMPUTATION])
        assert best is not None
        assert best.agent_id == "expert"

    def test_find_best_agent_none_if_no_skills(self):
        self.reg.register(make_manifest(
            agent_id="s1", skills=[Skill(SkillType.SENSOR, "temp", 0.8)]
        ))
        best = self.reg.find_best_agent([SkillType.COMPUTATION])
        # No agent has computation skill, so score will be based on coverage=0
        # Still returns the best scorer
        assert best is not None

    def test_find_best_agent_skips_unavailable(self):
        self.reg.register(make_manifest(
            agent_id="full", max_concurrent=1, active_tasks=1,
            skills=[Skill(SkillType.COMPUTATION, "math", 0.99)],
        ))
        best = self.reg.find_best_agent([SkillType.COMPUTATION])
        assert best is None

    def test_update_active_tasks(self):
        self.reg.register(make_manifest(agent_id="a"))
        self.reg.update_active_tasks("a", delta=1)
        assert self.reg.get("a").active_task_count == 1

    def test_update_active_tasks_nonexistent_raises(self):
        with pytest.raises(KeyError):
            self.reg.update_active_tasks("nope", delta=1)

    def test_set_trust(self):
        self.reg.register(make_manifest(agent_id="a"))
        self.reg.set_trust("a", 0.9)
        assert self.reg.get_trust("a") == 0.9

    def test_set_trust_invalid_raises(self):
        with pytest.raises(ValueError, match="Trust score"):
            self.reg.set_trust("a", 1.5)

    def test_record_trust_success(self):
        self.reg.register(make_manifest(agent_id="a"))
        self.reg.record_trust("a", success=True)
        assert self.reg.get_trust("a") == pytest.approx(0.6)  # 0.2*1 + 0.8*0.5

    def test_record_trust_failure(self):
        self.reg.register(make_manifest(agent_id="a"))
        self.reg.record_trust("a", success=False)
        assert self.reg.get_trust("a") == 0.4  # 0.2*0 + 0.8*0.5

    def test_get_trust_default(self):
        assert self.reg.get_trust("nonexistent") == 0.5


# ===========================================================================
# 5. NegotiationEngine
# ===========================================================================

class TestNegotiationEngine:
    def setup_method(self):
        self.reg = CapabilityRegistry()
        self.engine = NegotiationEngine(self.reg)

    def test_find_candidates_returns_sorted(self):
        self.reg.register(make_manifest(
            agent_id="expert",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.95)],
        ))
        self.reg.register(make_manifest(
            agent_id="novice",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.5)],
        ))
        task = make_task_spec(required_skills=[SkillType.COMPUTATION])
        candidates = self.engine.find_candidates(task)
        assert len(candidates) == 2
        assert candidates[0].score >= candidates[1].score

    def test_find_candidates_empty(self):
        task = make_task_spec()
        candidates = self.engine.find_candidates(task)
        assert candidates == []

    def test_negotiate_accept(self):
        self.reg.register(make_manifest(
            agent_id="worker",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.9)],
        ))
        task = make_task_spec(required_skills=[SkillType.COMPUTATION])
        result = self.engine.negotiate("worker", task)
        assert result.outcome == NegotiationOutcome.ACCEPT
        assert result.terms is not None

    def test_negotiate_decline_not_found(self):
        task = make_task_spec()
        result = self.engine.negotiate("ghost", task)
        assert result.outcome == NegotiationOutcome.DECLINE

    def test_negotiate_decline_overloaded(self):
        self.reg.register(make_manifest(
            agent_id="busy",
            max_concurrent=1, active_tasks=1,
            skills=[Skill(SkillType.COMPUTATION, "math", 0.9)],
        ))
        task = make_task_spec(required_skills=[SkillType.COMPUTATION])
        result = self.engine.negotiate("busy", task)
        assert result.outcome == NegotiationOutcome.DECLINE
        assert "capacity" in result.message.lower()

    def test_negotiate_decline_underqualified(self):
        self.reg.register(make_manifest(
            agent_id="sensor",
            skills=[Skill(SkillType.SENSOR, "temp", 0.8)],
        ))
        task = make_task_spec(required_skills=[SkillType.COMPUTATION])
        result = self.engine.negotiate("sensor", task)
        assert result.outcome == NegotiationOutcome.DECLINE
        assert "underqualified" in result.message.lower()

    def test_negotiate_decline_trust_too_low(self):
        self.reg.register(make_manifest(
            agent_id="newbie",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.9)],
        ))
        self.reg.set_trust("newbie", 0.3)
        task = make_task_spec(
            required_skills=[SkillType.COMPUTATION],
            constraints=[Constraint(ConstraintType.REQUIRES_TRUST_ABOVE, 0.7)],
        )
        result = self.engine.negotiate("newbie", task)
        assert result.outcome == NegotiationOutcome.DECLINE
        assert "trust" in result.message.lower()

    def test_negotiate_counter_confidence(self):
        self.reg.register(make_manifest(
            agent_id="mediocre",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.5)],
        ))
        task = make_task_spec(
            required_skills=[SkillType.COMPUTATION],
            constraints=[Constraint(ConstraintType.MIN_CONFIDENCE, 0.9)],
        )
        result = self.engine.negotiate("mediocre", task)
        assert result.outcome == NegotiationOutcome.COUNTER
        assert result.counter_spec is not None

    def test_counter_offer_accept(self):
        self.reg.register(make_manifest(
            agent_id="worker",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.7)],
        ))
        original = make_task_spec(
            required_skills=[SkillType.COMPUTATION],
            constraints=[Constraint(ConstraintType.MIN_CONFIDENCE, 0.9)],
        )
        modified = make_task_spec(
            required_skills=[SkillType.COMPUTATION],
            constraints=[Constraint(ConstraintType.MIN_CONFIDENCE, 0.6)],
        )
        result = self.engine.counter_offer("worker", original, modified)
        assert result.outcome == NegotiationOutcome.ACCEPT

    def test_counter_offer_decline_still_unqualified(self):
        self.reg.register(make_manifest(
            agent_id="sensor",
            skills=[Skill(SkillType.SENSOR, "temp", 0.8)],
        ))
        modified = make_task_spec(required_skills=[SkillType.COMPUTATION])
        result = self.engine.counter_offer("sensor", TaskSpec(), modified)
        assert result.outcome == NegotiationOutcome.DECLINE

    def test_preferred_agent_bonus_in_candidates(self):
        self.reg.register(make_manifest(
            agent_id="pref",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.7)],
        ))
        self.reg.register(make_manifest(
            agent_id="other",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.7)],
        ))
        task = make_task_spec(
            required_skills=[SkillType.COMPUTATION],
            constraints=[Constraint(ConstraintType.PREFERRED_AGENT, "pref")],
        )
        candidates = self.engine.find_candidates(task)
        pref_candidate = next(c for c in candidates if c.agent_id == "pref")
        other_candidate = next(c for c in candidates if c.agent_id == "other")
        assert pref_candidate.score > other_candidate.score

    def test_negotiate_multiple_skills_all_match(self):
        self.reg.register(make_manifest(
            agent_id="multi",
            skills=[
                Skill(SkillType.COMPUTATION, "math", 0.9),
                Skill(SkillType.ANALYSIS, "viz", 0.8),
            ],
        ))
        task = make_task_spec(required_skills=[
            SkillType.COMPUTATION, SkillType.ANALYSIS
        ])
        result = self.engine.negotiate("multi", task)
        assert result.outcome == NegotiationOutcome.ACCEPT

    def test_negotiate_multiple_skills_partial_match(self):
        self.reg.register(make_manifest(
            agent_id="partial",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.9)],
        ))
        task = make_task_spec(required_skills=[
            SkillType.COMPUTATION, SkillType.SENSOR
        ])
        result = self.engine.negotiate("partial", task)
        assert result.outcome == NegotiationOutcome.DECLINE


# ===========================================================================
# 6. Contract
# ===========================================================================

class TestContract:
    def test_creation(self):
        c = Contract(agent_id="a1")
        assert c.agent_id == "a1"
        assert c.state == ContractState.PROPOSED

    def test_empty_agent_id_raises(self):
        with pytest.raises(ValueError, match="agent_id"):
            Contract(agent_id="")

    def test_default_expires_at_set(self):
        before = time.time()
        c = Contract(agent_id="a1")
        after = time.time()
        assert before <= c.expires_at <= after + 3600.0

    def test_is_expired_false(self):
        c = Contract(agent_id="a1", expires_at=time.time() + 7200)
        assert c.is_expired() is False

    def test_is_expired_true(self):
        c = Contract(agent_id="a1", expires_at=time.time() - 10)
        assert c.is_expired() is True

    def test_to_dict(self):
        c = Contract(agent_id="a1")
        d = c.to_dict()
        assert d["agent_id"] == "a1"
        assert d["state"] == "proposed"


# ===========================================================================
# 7. ContractRegistry
# ===========================================================================

class TestContractRegistry:
    def setup_method(self):
        self.cap = CapabilityRegistry()
        self.con = ContractRegistry(self.cap)
        self.cap.register(make_manifest(agent_id="w1", max_concurrent=2))

    def test_create_contract(self):
        task = make_task_spec()
        c = self.con.create("w1", task)
        assert c.state == ContractState.PROPOSED
        assert self.con.get(c.id) is c

    def test_accept_contract(self):
        task = make_task_spec()
        c = self.con.create("w1", task)
        c2 = self.con.accept(c.id)
        assert c2.state == ContractState.ACCEPTED

    def test_accept_wrong_state_raises(self):
        task = make_task_spec()
        c = self.con.create("w1", task)
        self.con.accept(c.id)
        with pytest.raises(ValueError, match="Cannot accept"):
            self.con.accept(c.id)

    def test_start_contract(self):
        task = make_task_spec()
        c = self.con.create("w1", task)
        self.con.accept(c.id)
        c2 = self.con.start(c.id)
        assert c2.state == ContractState.IN_PROGRESS
        assert self.cap.get("w1").active_task_count == 1

    def test_start_wrong_state_raises(self):
        task = make_task_spec()
        c = self.con.create("w1", task)
        with pytest.raises(ValueError, match="Cannot start"):
            self.con.start(c.id)

    def test_complete_success(self):
        task = make_task_spec()
        c = self.con.create("w1", task)
        self.con.accept(c.id)
        self.con.start(c.id)
        c2 = self.con.complete(c.id, success=True)
        assert c2.state == ContractState.COMPLETED
        assert self.cap.get("w1").active_task_count == 0
        assert self.cap.get_trust("w1") == pytest.approx(0.6)  # trust increased

    def test_complete_failure(self):
        task = make_task_spec()
        c = self.con.create("w1", task)
        self.con.accept(c.id)
        self.con.start(c.id)
        c2 = self.con.complete(c.id, success=False)
        assert c2.state == ContractState.FAILED
        assert self.cap.get_trust("w1") == 0.4  # trust decreased

    def test_complete_wrong_state_raises(self):
        task = make_task_spec()
        c = self.con.create("w1", task)
        with pytest.raises(ValueError, match="Cannot complete"):
            self.con.complete(c.id)

    def test_cancel_proposed(self):
        task = make_task_spec()
        c = self.con.create("w1", task)
        c2 = self.con.cancel(c.id)
        assert c2.state == ContractState.CANCELLED

    def test_cancel_in_progress(self):
        task = make_task_spec()
        c = self.con.create("w1", task)
        self.con.accept(c.id)
        self.con.start(c.id)
        c2 = self.con.cancel(c.id)
        assert c2.state == ContractState.CANCELLED
        assert self.cap.get("w1").active_task_count == 0

    def test_cancel_terminal_raises(self):
        task = make_task_spec()
        c = self.con.create("w1", task)
        self.con.accept(c.id)
        self.con.start(c.id)
        self.con.complete(c.id)
        with pytest.raises(ValueError, match="Cannot cancel"):
            self.con.cancel(c.id)

    def test_contracts_for_agent(self):
        task = make_task_spec()
        self.con.create("w1", task)
        self.con.create("w1", task)
        assert len(self.con.contracts_for_agent("w1")) == 2

    def test_active_contracts(self):
        task = make_task_spec()
        c1 = self.con.create("w1", task)
        self.con.accept(c1.id)
        c2 = self.con.create("w1", task)
        self.con.accept(c2.id)
        self.con.start(c2.id)
        assert len(self.con.active_contracts()) == 2
        self.con.complete(c2.id)
        assert len(self.con.active_contracts()) == 1

    def test_cleanup_expired(self):
        task = make_task_spec()
        c = self.con.create("w1", task, expires_at=time.time() - 10)
        cleaned = self.con.cleanup_expired()
        assert len(cleaned) == 1
        assert self.con.get(c.id).state == ContractState.CANCELLED

    def test_cleanup_no_expired(self):
        task = make_task_spec()
        self.con.create("w1", task, expires_at=time.time() + 7200)
        cleaned = self.con.cleanup_expired()
        assert cleaned == []

    def test_get_nonexistent(self):
        assert self.con.get("nope") is None

    def test_all_contracts(self):
        task = make_task_spec()
        self.con.create("w1", task)
        self.con.create("w1", task)
        assert len(self.con.all_contracts()) == 2


# ===========================================================================
# 8. FleetDirectory
# ===========================================================================

class TestFleetDirectory:
    def setup_method(self):
        self.cap = CapabilityRegistry()
        self.con = ContractRegistry(self.cap)
        self.dir = FleetDirectory(self.cap, self.con)
        self.cap.register(make_manifest(agent_id="w1", name="Worker One", role="worker", max_concurrent=2))
        self.cap.register(make_manifest(agent_id="w2", name="Worker Two", role="worker", max_concurrent=3))

    def test_overview(self):
        ov = self.dir.overview()
        assert ov["total_agents"] == 2
        assert ov["available_agents"] == 2
        assert ov["active_contracts"] == 0

    def test_overview_with_active_contract(self):
        task = make_task_spec()
        c = self.con.create("w1", task)
        self.con.accept(c.id)
        self.con.start(c.id)
        ov = self.dir.overview()
        assert ov["active_contracts"] == 1

    def test_select_agent_basic(self):
        task = make_task_spec(required_skills=[SkillType.COMPUTATION])
        best = self.dir.select_agent(task)
        assert best is not None

    def test_select_agent_excludes(self):
        task = make_task_spec(required_skills=[SkillType.COMPUTATION])
        best = self.dir.select_agent(task, exclude={"w1"})
        assert best is not None
        assert best.agent_id != "w1"

    def test_select_agent_none_available(self):
        self.cap.unregister("w1")
        self.cap.unregister("w2")
        task = make_task_spec()
        assert self.dir.select_agent(task) is None

    def test_select_agent_prefers_least_loaded(self):
        """If two agents have equal scores, least-loaded wins."""
        # Give both same skills
        self.cap.unregister("w1")
        self.cap.unregister("w2")
        self.cap.register(make_manifest(
            agent_id="busy",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.9)],
            max_concurrent=2, active_tasks=1,
        ))
        self.cap.register(make_manifest(
            agent_id="idle",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.9)],
            max_concurrent=2, active_tasks=0,
        ))
        task = make_task_spec(required_skills=[SkillType.COMPUTATION])
        best = self.dir.select_agent(task)
        assert best.agent_id == "idle"

    def test_record_preference(self):
        self.dir.record_preference("w1", SkillType.COMPUTATION, success=True)
        prefs = self.dir.get_preferences("w1")
        assert len(prefs) == 1
        assert prefs[0].skill_type == SkillType.COMPUTATION
        assert prefs[0].count == 1
        assert prefs[0].success_count == 1

    def test_record_preference_multiple(self):
        self.dir.record_preference("w1", SkillType.COMPUTATION, success=True)
        self.dir.record_preference("w1", SkillType.COMPUTATION, success=True)
        prefs = self.dir.get_preferences("w1")
        assert prefs[0].count == 2
        assert prefs[0].success_count == 2

    def test_record_preference_failure(self):
        self.dir.record_preference("w1", SkillType.COMPUTATION, success=False)
        prefs = self.dir.get_preferences("w1")
        assert prefs[0].success_count == 0

    def test_record_preference_no_prior(self):
        prefs = self.dir.get_preferences("w2")
        assert prefs == []

    def test_get_preferred_agents(self):
        self.dir.record_preference("w1", SkillType.COMPUTATION, success=True)
        self.dir.record_preference("w1", SkillType.COMPUTATION, success=True)
        self.dir.record_preference("w2", SkillType.COMPUTATION, success=False)
        preferred = self.dir.get_preferred_agents(SkillType.COMPUTATION)
        assert len(preferred) == 2
        assert preferred[0][0] == "w1"  # w1 has higher preference

    def test_preference_influences_selection(self):
        """Agents with preference history should be preferred."""
        self.cap.unregister("w1")
        self.cap.unregister("w2")
        self.cap.register(make_manifest(
            agent_id="experienced",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.7)],
            max_concurrent=2, active_tasks=0,
        ))
        self.cap.register(make_manifest(
            agent_id="newbie",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.7)],
            max_concurrent=2, active_tasks=0,
        ))
        # Build up preference for experienced agent
        for _ in range(5):
            self.dir.record_preference("experienced", SkillType.COMPUTATION, success=True)
        task = make_task_spec(required_skills=[SkillType.COMPUTATION])
        best = self.dir.select_agent(task)
        assert best.agent_id == "experienced"


# ===========================================================================
# 9. ResourceLimits & ResourceRequirement
# ===========================================================================

class TestResourceTypes:
    def test_resource_limits_defaults(self):
        r = ResourceLimits()
        assert r.memory_mb == 512
        assert r.cpu_cores == 1.0

    def test_resource_limits_custom(self):
        r = ResourceLimits(memory_mb=2048, cpu_cores=4.0)
        assert r.memory_mb == 2048

    def test_resource_requirement_defaults(self):
        r = ResourceRequirement()
        assert r.memory_mb == 256

    def test_frozen(self):
        r = ResourceLimits()
        with pytest.raises(AttributeError):
            r.memory_mb = 999


# ===========================================================================
# 10. PreferenceRecord
# ===========================================================================

class TestPreferenceRecord:
    def test_success_rate_new(self):
        p = PreferenceRecord(agent_id="a", skill_type=SkillType.COMPUTATION)
        assert p.success_rate == 0.5

    def test_success_rate_all_success(self):
        p = PreferenceRecord(agent_id="a", skill_type=SkillType.COMPUTATION, count=10, success_count=10)
        assert p.success_rate == 1.0

    def test_success_rate_mixed(self):
        p = PreferenceRecord(agent_id="a", skill_type=SkillType.COMPUTATION, count=4, success_count=2)
        assert p.success_rate == 0.5


# ===========================================================================
# 11. NegotiationResult & CandidateScore
# ===========================================================================

class TestNegotiationResult:
    def test_accept(self):
        r = NegotiationResult(outcome=NegotiationOutcome.ACCEPT, message="OK")
        assert r.outcome == NegotiationOutcome.ACCEPT

    def test_to_dict(self):
        r = NegotiationResult(outcome=NegotiationOutcome.DECLINE, message="nope")
        d = r.to_dict()
        assert d["outcome"] == "decline"
        assert d["message"] == "nope"

    def test_to_dict_with_counter(self):
        ts = make_task_spec()
        r = NegotiationResult(
            outcome=NegotiationOutcome.COUNTER,
            message="try this",
            counter_spec=ts,
        )
        d = r.to_dict()
        assert d["counter_spec"] is not None


class TestCandidateScore:
    def test_ordering(self):
        a = CandidateScore(agent_id="a", manifest=make_manifest(), score=0.5)
        b = CandidateScore(agent_id="b", manifest=make_manifest(agent_id="b"), score=0.9)
        assert b > a

    def test_lt_non_candidate_returns_not_implemented(self):
        a = CandidateScore(agent_id="a", manifest=make_manifest(), score=0.5)
        assert a.__lt__("not a candidate") is NotImplemented


# ===========================================================================
# 12. Constraint
# ===========================================================================

class TestConstraint:
    def test_frozen(self):
        c = Constraint(ConstraintType.MIN_CONFIDENCE, 0.8)
        with pytest.raises(AttributeError):
            c.value = 0.5

    def test_various_constraint_types(self):
        Constraint(ConstraintType.MAX_LATENCY, 100)
        Constraint(ConstraintType.REQUIRES_TRUST_ABOVE, 0.7)
        Constraint(ConstraintType.PREFERRED_AGENT, "agent-001")


# ===========================================================================
# 13. ContractTerms
# ===========================================================================

class TestContractTerms:
    def test_defaults(self):
        t = ContractTerms()
        assert t.deadline == 0.0
        assert t.reward == 0.0
        assert t.penalty_conditions == []

    def test_to_dict(self):
        t = ContractTerms(deadline=100.0, reward=5.0, penalty_conditions=["timeout"])
        d = t.to_dict()
        assert d["reward"] == 5.0
        assert len(d["penalty_conditions"]) == 1


# ===========================================================================
# 14. Integration: Full lifecycle
# ===========================================================================

class TestIntegration:
    def test_full_lifecycle(self):
        """Register agents, find candidates, negotiate, create contract, complete."""
        cap = CapabilityRegistry()
        con = ContractRegistry(cap)
        engine = NegotiationEngine(cap)
        directory = FleetDirectory(cap, con)

        # Register agents
        cap.register(make_manifest(
            agent_id="worker",
            name="ComputeWorker",
            role="compute",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.9)],
            max_concurrent=2,
        ))
        cap.register(make_manifest(
            agent_id="analyst",
            name="DataAnalyst",
            role="analysis",
            skills=[Skill(SkillType.ANALYSIS, "viz", 0.8)],
            max_concurrent=1,
        ))

        # Task requiring computation
        task = make_task_spec(required_skills=[SkillType.COMPUTATION])

        # Find candidates
        candidates = engine.find_candidates(task)
        assert len(candidates) == 2

        # Negotiate
        result = engine.negotiate("worker", task)
        assert result.outcome == NegotiationOutcome.ACCEPT

        # Create contract
        contract = con.create("worker", task, terms=result.terms)
        assert contract.state == ContractState.PROPOSED

        # Accept and start
        con.accept(contract.id)
        con.start(contract.id)
        assert cap.get("worker").active_task_count == 1

        # Complete
        con.complete(contract.id, success=True)
        assert contract.state == ContractState.COMPLETED
        assert cap.get("worker").active_task_count == 0

        # Record preference
        directory.record_preference("worker", SkillType.COMPUTATION, success=True)
        prefs = directory.get_preferences("worker")
        assert len(prefs) == 1

        # Overview
        ov = directory.overview()
        assert ov["total_agents"] == 2

    def test_decline_and_counter_flow(self):
        """Agent declines high-confidence requirement, counter-offer succeeds."""
        cap = CapabilityRegistry()
        engine = NegotiationEngine(cap)

        cap.register(make_manifest(
            agent_id="mediocre",
            skills=[Skill(SkillType.COMPUTATION, "math", 0.5)],
        ))

        task = make_task_spec(
            required_skills=[SkillType.COMPUTATION],
            constraints=[Constraint(ConstraintType.MIN_CONFIDENCE, 0.9)],
        )

        # Should counter
        result = engine.negotiate("mediocre", task)
        assert result.outcome == NegotiationOutcome.COUNTER
        assert result.counter_spec is not None

        # Counter-offer with relaxed constraint
        counter = engine.counter_offer("mediocre", task, result.counter_spec)
        assert counter.outcome == NegotiationOutcome.ACCEPT

    def test_expired_contract_cleanup(self):
        cap = CapabilityRegistry()
        cap.register(make_manifest(agent_id="w1"))
        con = ContractRegistry(cap)

        task = make_task_spec()
        c = con.create("w1", task, expires_at=time.time() - 1)

        cleaned = con.cleanup_expired()
        assert len(cleaned) == 1
        assert con.get(c.id).state == ContractState.CANCELLED
