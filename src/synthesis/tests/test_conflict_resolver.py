"""
Unit tests for synthesis/conflict_resolver.py.

Covers ConflictResolver with all four synthesis strategies (MAJORITY, WEIGHTED,
FIRST_RESPONSE, BEST_EVIDENCE), confidence calculation, edge cases, error
handling, and multi-agent scenarios.
"""

import pytest
from src.synthesis.conflict_resolver import (
    ConflictResolver,
    ConflictResolutionError,
    ResolvedResult,
    SynthesisStrategy,
    _result_key,
    _majority_confidence,
)
from src.cooperative_types import CooperativeResponse
from src.trust.scorer import TrustScorer


def _make_response(agent, task_id, result, exec_time_ms=10, vm_info=None):
    """Helper to create a successful CooperativeResponse."""
    return CooperativeResponse.success(
        task_id=task_id,
        source_agent=agent,
        target_agent="Quill",
        result=result,
        execution_time_ms=exec_time_ms,
        vm_info=vm_info,
    )


def _make_error_response(agent, task_id, error_code, error_msg):
    """Helper to create an error CooperativeResponse."""
    return CooperativeResponse.error(
        task_id=task_id,
        source_agent=agent,
        target_agent="Quill",
        error_code=error_code,
        error_message=error_msg,
    )


class TestResultKey:
    """Test _result_key normalization helper."""

    def test_identical_dicts_produce_same_key(self):
        assert _result_key({"a": 1, "b": 2}) == _result_key({"b": 2, "a": 1})

    def test_different_dicts_produce_different_keys(self):
        assert _result_key({"a": 1}) != _result_key({"a": 2})

    def test_nested_dicts(self):
        k1 = _result_key({"nested": {"x": 1}})
        k2 = _result_key({"nested": {"x": 1}})
        assert k1 == k2

    def test_empty_dict(self):
        key = _result_key({})
        assert isinstance(key, str)


class TestMajorityConfidence:
    """Test _majority_confidence helper function."""

    def test_zero_total(self):
        assert _majority_confidence(0.0, 0) == 0.0

    def test_unanimous_agreement_bonus(self):
        conf = _majority_confidence(1.0, 5)
        # base=1.0, sample_bonus=0.1 (capped), unanimous_bonus=0.15
        # But capped at 1.0
        assert conf == 1.0

    def test_partial_agreement(self):
        conf = _majority_confidence(0.6, 5)
        # base=0.6, sample_bonus=0.1 (5*0.025=0.125, capped at 0.1), unanimous_bonus=0
        assert conf == pytest.approx(0.7, abs=0.01)

    def test_single_responder_no_unanimous_bonus(self):
        conf = _majority_confidence(1.0, 1)
        # base=1.0, sample_bonus=0.025, unanimous_bonus=0.15
        # total=1.175, capped at 1.0
        assert conf == 1.0

    def test_many_responders_sample_bonus_capped(self):
        conf = _majority_confidence(0.5, 100)
        # sample_bonus = min(0.1, 100*0.025) = 0.1
        assert conf == pytest.approx(0.6, abs=0.01)


class TestConflictResolverMajority:
    """Test majority voting strategy."""

    def test_unanimous_agreement(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        r1 = _make_response("A", "t1", {"value": 42})
        r2 = _make_response("B", "t1", {"value": 42})
        r3 = _make_response("C", "t1", {"value": 42})
        resolver.add_response(r1)
        resolver.add_response(r2)
        resolver.add_response(r3)

        result = resolver.resolve()
        assert result.result == {"value": 42}
        assert result.confidence == 1.0  # unanimous
        assert len(result.contributing_responses) == 3
        assert len(result.discarded_responses) == 0
        assert result.agents_responded == 3

    def test_majority_wins(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        r1 = _make_response("A", "t1", {"value": 42})
        r2 = _make_response("B", "t1", {"value": 42})
        r3 = _make_response("C", "t1", {"value": 99})
        resolver.add_response(r1)
        resolver.add_response(r2)
        resolver.add_response(r3)

        result = resolver.resolve()
        assert result.result == {"value": 42}
        assert len(result.contributing_responses) == 2
        assert len(result.discarded_responses) == 1
        assert result.confidence < 1.0  # not unanimous

    def test_single_response(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        r1 = _make_response("A", "t1", {"value": 42})
        resolver.add_response(r1)

        result = resolver.resolve()
        assert result.result == {"value": 42}
        assert len(result.contributing_responses) == 1

    def test_tie_breaking(self):
        """When there's a tie, max() picks one (implementation-defined)."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        r1 = _make_response("A", "t1", {"value": 1})
        r2 = _make_response("B", "t1", {"value": 2})
        resolver.add_response(r1)
        resolver.add_response(r2)

        result = resolver.resolve()
        assert result.result in [{"value": 1}, {"value": 2}]
        assert result.confidence < 1.0

    def test_error_responses_ignored(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        r1 = _make_response("A", "t1", {"value": 42})
        r2 = _make_error_response("B", "t1", "ERR", "Failed")
        resolver.add_response(r1)
        resolver.add_response(r2)

        result = resolver.resolve()
        assert result.result == {"value": 42}
        assert result.agents_responded == 1  # Only the success counts


class TestConflictResolverWeighted:
    """Test trust-weighted voting strategy."""

    def test_trusted_agent_wins(self):
        trust = TrustScorer()
        trust.record_result("A", "success")
        trust.record_result("A", "success")
        trust.record_result("A", "success")
        trust.record_result("B", "error")
        trust.record_result("B", "error")

        resolver = ConflictResolver(strategy=SynthesisStrategy.WEIGHTED, trust_scorer=trust)
        r1 = _make_response("A", "t1", {"value": 42})  # score=1.0
        r2 = _make_response("B", "t1", {"value": 99})  # score=0.0
        resolver.add_response(r1)
        resolver.add_response(r2)

        result = resolver.resolve()
        assert result.result == {"value": 42}
        assert result.confidence == 1.0  # A has all the trust weight

    def test_falls_back_to_majority_without_trust(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.WEIGHTED, trust_scorer=None)
        r1 = _make_response("A", "t1", {"value": 42})
        r2 = _make_response("B", "t1", {"value": 42})
        r3 = _make_response("C", "t1", {"value": 99})
        resolver.add_response(r1)
        resolver.add_response(r2)
        resolver.add_response(r3)

        result = resolver.resolve()
        # Should fall back to majority: 42 wins (2 vs 1)
        assert result.result == {"value": 42}

    def test_equal_trust_weighted_tie(self):
        trust = TrustScorer()
        trust.record_result("A", "success")
        trust.record_result("B", "success")

        resolver = ConflictResolver(strategy=SynthesisStrategy.WEIGHTED, trust_scorer=trust)
        r1 = _make_response("A", "t1", {"value": 1})
        r2 = _make_response("B", "t1", {"value": 2})
        resolver.add_response(r1)
        resolver.add_response(r2)

        result = resolver.resolve()
        # Both agents have score 1.0, tie broken by max()
        assert result.confidence >= 0.5


class TestConflictResolverFirstResponse:
    """Test first-response-wins strategy."""

    def test_first_response_accepted(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.FIRST_RESPONSE)
        r1 = _make_response("A", "t1", {"value": 42})
        r2 = _make_response("B", "t1", {"value": 99})
        resolver.add_response(r1)
        resolver.add_response(r2)

        result = resolver.resolve()
        assert result.result == {"value": 42}
        assert result.contributing_responses[0].source_agent == "A"
        assert len(result.discarded_responses) == 1

    def test_single_response_confidence(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.FIRST_RESPONSE)
        r1 = _make_response("A", "t1", {"value": 42})
        resolver.add_response(r1)

        result = resolver.resolve()
        assert result.confidence == 0.5  # No confirmation with single response

    def test_all_agree_high_confidence(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.FIRST_RESPONSE)
        r1 = _make_response("A", "t1", {"value": 42})
        r2 = _make_response("B", "t1", {"value": 42})
        r3 = _make_response("C", "t1", {"value": 42})
        resolver.add_response(r1)
        resolver.add_response(r2)
        resolver.add_response(r3)

        result = resolver.resolve()
        assert result.confidence == 1.0  # All agree with first

    def test_no_agreement_lower_confidence(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.FIRST_RESPONSE)
        r1 = _make_response("A", "t1", {"value": 42})
        r2 = _make_response("B", "t1", {"value": 99})
        r3 = _make_response("C", "t1", {"value": 100})
        resolver.add_response(r1)
        resolver.add_response(r2)
        resolver.add_response(r3)

        result = resolver.resolve()
        assert result.confidence == pytest.approx(1.0 / 3.0, abs=0.01)


class TestConflictResolverBestEvidence:
    """Test best-evidence selection strategy."""

    def test_prefers_response_with_vm_info(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.BEST_EVIDENCE)
        r1 = _make_response("A", "t1", {"value": 42}, vm_info={"isa": "unified"})
        r2 = _make_response("B", "t1", {"value": 99})  # No vm_info
        resolver.add_response(r1)
        resolver.add_response(r2)

        result = resolver.resolve()
        assert result.result == {"value": 42}

    def test_prefers_richer_result(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.BEST_EVIDENCE)
        r1 = _make_response("A", "t1", {"value": 42, "context": "detailed", "confidence": 0.99})
        r2 = _make_response("B", "t1", {"value": 99})  # Minimal result
        resolver.add_response(r1)
        resolver.add_response(r2)

        result = resolver.resolve()
        assert result.result["value"] == 42

    def test_reasonable_exec_time_bonus(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.BEST_EVIDENCE)
        r1 = _make_response("A", "t1", {"value": 42}, exec_time_ms=100)
        r2 = _make_response("B", "t1", {"value": 99}, exec_time_ms=0)  # No timing
        resolver.add_response(r1)
        resolver.add_response(r2)

        result = resolver.resolve()
        # A should win due to reasonable execution time (0.2 bonus) vs no timing (0.05)
        assert result.result == {"value": 42}

    def test_trust_bonus_applied(self):
        trust = TrustScorer()
        trust.record_result("A", "success")
        trust.record_result("A", "success")
        trust.record_result("A", "success")

        resolver = ConflictResolver(strategy=SynthesisStrategy.BEST_EVIDENCE, trust_scorer=trust)
        # B has slightly richer result but A has trust bonus
        r1 = _make_response("A", "t1", {"value": 42})
        r2 = _make_response("B", "t1", {"value": 99, "extra": "data"})
        resolver.add_response(r1)
        resolver.add_response(r2)

        result = resolver.resolve()
        # A should win due to trust bonus (0.2 * 1.0 = 0.2) making up for smaller result
        assert result.result == {"value": 42}

    def test_confidence_normalized_to_range(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.BEST_EVIDENCE)
        r1 = _make_response("A", "t1", {"value": 42}, vm_info={"isa": "unified"}, exec_time_ms=100)
        resolver.add_response(r1)

        result = resolver.resolve()
        assert 0.0 <= result.confidence <= 1.0


class TestConflictResolverGeneral:
    """Test general resolver behavior across strategies."""

    def test_empty_resolver_raises(self):
        for strategy in SynthesisStrategy:
            resolver = ConflictResolver(strategy=strategy)
            with pytest.raises(ConflictResolutionError, match="No responses"):
                resolver.resolve()

    def test_clear_resets_state(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        resolver.add_response(_make_response("A", "t1", {"value": 1}))
        assert resolver.response_count == 1
        resolver.clear()
        assert resolver.response_count == 0

    def test_set_expected_agents(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        resolver.set_expected_agents(5)
        resolver.add_response(_make_response("A", "t1", {"value": 42}))

        result = resolver.resolve()
        assert result.agents_consulted == 5
        assert result.agents_responded == 1

    def test_resolution_time_recorded(self):
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        resolver.add_response(_make_response("A", "t1", {"value": 42}))

        result = resolver.resolve()
        assert result.resolution_time_ms >= 0

    def test_strategy_in_result(self):
        trust = TrustScorer()
        for strategy in SynthesisStrategy:
            resolver = ConflictResolver(strategy=strategy, trust_scorer=trust)
            resolver.add_response(_make_response("A", "t1", {"v": 1}))
            result = resolver.resolve()
            assert result.strategy == strategy.value

    def test_add_response_ignores_errors(self):
        """Error responses should not be added to the pool."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        err = _make_error_response("A", "t1", "ERR", "Bad")
        resolver.add_response(err)
        assert resolver.response_count == 0

    def test_add_response_ignores_null_result_success(self):
        """A success response with None result should not be added."""
        resp = CooperativeResponse.success("t1", "A", "B", None)
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        resolver.add_response(resp)
        assert resolver.response_count == 0


class TestResolvedResult:
    """Test ResolvedResult dataclass."""

    def test_to_dict(self):
        result = ResolvedResult(
            strategy="majority",
            result={"value": 42},
            confidence=0.95,
            agents_consulted=3,
            agents_responded=3,
        )
        d = result.to_dict()
        assert d["strategy"] == "majority"
        assert d["result"] == {"value": 42}
        assert d["confidence"] == 0.95
        assert d["agents_consulted"] == 3
        assert d["agents_responded"] == 3
        assert "contributing_agents" in d
        assert "discarded_agents" in d

    def test_to_dict_with_contributors(self):
        r1 = _make_response("A", "t1", {"v": 1})
        r2 = _make_response("B", "t1", {"v": 2})
        result = ResolvedResult(
            strategy="majority",
            result={"v": 1},
            confidence=0.8,
            contributing_responses=[r1],
            discarded_responses=[r2],
        )
        d = result.to_dict()
        assert "A" in d["contributing_agents"]
        assert "B" in d["discarded_agents"]

    def test_confidence_rounded(self):
        result = ResolvedResult(strategy="test", result={}, confidence=0.12345678)
        d = result.to_dict()
        assert d["confidence"] == 0.1235


class TestMultiAgentScenarios:
    """Test realistic multi-agent cooperation scenarios."""

    def test_three_agents_two_agree(self):
        """Three agents, two agree on answer — majority wins."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        for agent in ["Quill", "Oracle1"]:
            resolver.add_response(_make_response(agent, "t1", {"result": "A"}))
        resolver.add_response(_make_response("Super Z", "t1", {"result": "B"}))

        result = resolver.resolve()
        assert result.result == {"result": "A"}
        assert result.agents_responded == 3

    def test_five_agents_consensus(self):
        """Five agents all agree — high confidence."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        for agent in ["Quill", "Oracle1", "Super Z", "JetsonClaw1", "Babel"]:
            resolver.add_response(_make_response(agent, "t1", {"answer": 42}))
        resolver.set_expected_agents(5)

        result = resolver.resolve()
        assert result.result == {"answer": 42}
        assert result.confidence == 1.0
        assert result.agents_consulted == 5

    def test_weighted_resolution_trusted_agent_overrules(self):
        """When one highly-trusted agent disagrees, weighted strategy
        should favor the trusted agent."""
        trust = TrustScorer()
        # Make Quill very trusted
        for _ in range(20):
            trust.record_result("Quill", "success")
        # Make others less trusted
        for agent in ["A", "B", "C"]:
            trust.record_result(agent, "success")
            trust.record_result(agent, "error")
            trust.record_result(agent, "error")

        resolver = ConflictResolver(strategy=SynthesisStrategy.WEIGHTED, trust_scorer=trust)
        # Minority but trusted
        resolver.add_response(_make_response("Quill", "t1", {"answer": 42}))
        # Majority but untrusted
        for agent in ["A", "B", "C"]:
            resolver.add_response(_make_response(agent, "t1", {"answer": 99}))

        result = resolver.resolve()
        assert result.result == {"answer": 42}  # Trusted Quill wins

    def test_first_response_with_timeout_simulation(self):
        """Simulate scenario where not all agents respond."""
        resolver = ConflictResolver(strategy=SynthesisStrategy.FIRST_RESPONSE)
        resolver.set_expected_agents(5)
        # Only 3 respond
        for agent in ["Quill", "Oracle1", "Super Z"]:
            resolver.add_response(_make_response(agent, "t1", {"status": "ok"}))

        result = resolver.resolve()
        assert result.agents_consulted == 5
        assert result.agents_responded == 3
