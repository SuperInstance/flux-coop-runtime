"""Tests for trust scoring."""

import os
import tempfile
import pytest
from src.trust.scorer import TrustScorer


class TestTrustRecord:
    """Test TrustRecord basic behavior."""

    def test_new_record_has_neutral_score(self):
        from src.cooperative_types import TrustRecord
        rec = TrustRecord(agent_name="test")
        assert rec.score == 0.5
        assert rec.total == 0

    def test_success_increases_score(self):
        from src.cooperative_types import TrustRecord
        rec = TrustRecord(agent_name="test")
        rec.record("success")
        assert rec.score == 1.0
        assert rec.successes == 1

    def test_failure_decreases_score(self):
        from src.cooperative_types import TrustRecord
        rec = TrustRecord(agent_name="test")
        rec.record("success")
        rec.record("error")
        assert rec.score == 0.5
        assert rec.failures == 1

    def test_timeout_counts(self):
        from src.cooperative_types import TrustRecord
        rec = TrustRecord(agent_name="test")
        rec.record("timeout")
        assert rec.timeouts == 1
        assert rec.score == 0.0

    def test_mixed_results(self):
        from src.cooperative_types import TrustRecord
        rec = TrustRecord(agent_name="test")
        for _ in range(7):
            rec.record("success")
        for _ in range(3):
            rec.record("error")
        assert rec.total == 10
        assert rec.score == 0.7

    def test_to_dict(self):
        from src.cooperative_types import TrustRecord
        rec = TrustRecord(agent_name="test")
        rec.record("success")
        d = rec.to_dict()
        assert d["agent_name"] == "test"
        assert d["score"] == 1.0
        assert "last_seen" in d


class TestTrustScorer:
    """Test TrustScorer manager."""

    def test_get_score_new_agent(self):
        scorer = TrustScorer()
        assert scorer.get_score("new_agent") == 0.5

    def test_record_success(self):
        scorer = TrustScorer()
        score = scorer.record_result("agent_a", "success")
        assert score == 1.0

    def test_record_failure(self):
        scorer = TrustScorer()
        scorer.record_result("agent_a", "success")
        score = scorer.record_result("agent_a", "error")
        assert score == 0.5

    def test_record_unknown_status(self):
        """Unknown status should not crash but shouldn't change score."""
        scorer = TrustScorer()
        score = scorer.record_result("agent_a", "unknown_status")
        # Unknown status doesn't increment anything, stays at neutral
        assert score == 0.5

    def test_rank_agents(self):
        scorer = TrustScorer()
        # Agent A: 100% success
        for _ in range(5):
            scorer.record_result("agent_a", "success")
        # Agent B: 50% success
        scorer.record_result("agent_b", "success")
        scorer.record_result("agent_b", "error")
        # Agent C: 0% success
        scorer.record_result("agent_c", "error")

        ranked = scorer.rank_agents(["agent_a", "agent_b", "agent_c"])
        assert ranked[0][0] == "agent_a"
        assert ranked[0][1] == 1.0
        assert ranked[1][0] == "agent_b"
        assert ranked[2][0] == "agent_c"

    def test_rank_with_min_score(self):
        scorer = TrustScorer()
        scorer.record_result("good", "success")
        scorer.record_result("bad", "error")

        ranked = scorer.rank_agents(["good", "bad"], min_score=0.5)
        assert len(ranked) == 1
        assert ranked[0][0] == "good"

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            scorer = TrustScorer(persistence_path=path)
            scorer.record_result("agent_a", "success")
            scorer.record_result("agent_a", "error")
            scorer.record_result("agent_a", "success")

            # Reload
            scorer2 = TrustScorer(persistence_path=path)
            assert scorer2.get_score("agent_a") == pytest.approx(0.667, abs=0.01)
            assert scorer2.get_record("agent_a").total == 3
        finally:
            os.unlink(path)

    def test_get_all_records(self):
        scorer = TrustScorer()
        scorer.record_result("a", "success")
        scorer.record_result("b", "error")
        records = scorer.get_all_records()
        assert "a" in records
        assert "b" in records
