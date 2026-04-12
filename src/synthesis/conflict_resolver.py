"""
Conflict Resolution for Multi-Agent Synthesis.

When multiple agents respond to a cooperative request with potentially
conflicting answers, this module provides strategies to resolve those
conflicts into a single authoritative result.

Strategies include majority voting, trust-weighted consensus, first-response
wins, and best-evidence selection — each appropriate for different cooperative
scenarios.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

from src.cooperative_types import CooperativeResponse
from src.trust.scorer import TrustScorer


class SynthesisStrategy(str, Enum):
    """Strategy for resolving conflicting agent responses."""

    MAJORITY = "majority"
    """Pick the result that the most agents agree on."""

    WEIGHTED = "weighted"
    """Weight votes by each agent's trust score."""

    FIRST_RESPONSE = "first_response"
    """Accept the first response received, ignore the rest."""

    BEST_EVIDENCE = "best_evidence"
    """Pick the response with the highest self-reported confidence or quality."""


class ConflictResolutionError(Exception):
    """Raised when conflict resolution fails to produce a result."""
    pass


@dataclass
class ResolvedResult:
    """
    The outcome of conflict resolution across multiple agent responses.

    Attributes:
        strategy: The synthesis strategy used.
        result: The resolved result dictionary.
        confidence: Confidence score from 0.0 to 1.0 indicating resolution certainty.
        contributing_responses: List of responses that contributed to the decision.
        discarded_responses: List of responses that were not used.
        agents_consulted: Total number of agents consulted.
        agents_responded: Number of agents that actually responded.
        resolution_time_ms: Time taken to resolve in milliseconds.
    """
    strategy: str
    result: Dict[str, Any]
    confidence: float
    contributing_responses: List[CooperativeResponse] = field(default_factory=list)
    discarded_responses: List[CooperativeResponse] = field(default_factory=list)
    agents_consulted: int = 0
    agents_responded: int = 0
    resolution_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "strategy": self.strategy,
            "result": self.result,
            "confidence": round(self.confidence, 4),
            "contributing_agents": [
                r.source_agent for r in self.contributing_responses
            ],
            "discarded_agents": [
                r.source_agent for r in self.discarded_responses
            ],
            "agents_consulted": self.agents_consulted,
            "agents_responded": self.agents_responded,
            "resolution_time_ms": self.resolution_time_ms,
        }


class ConflictResolver:
    """
    Resolves disagreements when multiple agents provide different answers
    to the same cooperative request.

    Collects CooperativeResponses and applies a SynthesisStrategy to produce
    a single ResolvedResult. Supports timeout-aware resolution so that
    partial results can still be synthesized if not all agents respond.

    Usage:
        resolver = ConflictResolver(strategy=SynthesisStrategy.MAJORITY)
        resolver.add_response(response_from_agent_a)
        resolver.add_response(response_from_agent_b)
        resolved = resolver.resolve()
    """

    def __init__(
        self,
        strategy: SynthesisStrategy = SynthesisStrategy.MAJORITY,
        trust_scorer: Optional[TrustScorer] = None,
        required_responses: int = 1,
        timeout_ms: int = 30000,
    ):
        """
        Args:
            strategy: The synthesis strategy to use.
            trust_scorer: Optional trust scorer for WEIGHTED strategy.
            required_responses: Minimum number of responses before resolving.
                              If fewer arrive, resolution waits until timeout.
            timeout_ms: Maximum time to wait for responses before resolving
                       with whatever is available.
        """
        self.strategy = strategy
        self.trust = trust_scorer
        self.required_responses = required_responses
        self.timeout_ms = timeout_ms
        self._responses: List[CooperativeResponse] = []
        self._expected_agents: int = 0

    def set_expected_agents(self, count: int) -> None:
        """Set how many agents were consulted (for confidence calculation)."""
        self._expected_agents = count

    def add_response(self, response: CooperativeResponse) -> None:
        """Add an agent response to the pool for resolution."""
        if response.status == "success" and response.result is not None:
            self._responses.append(response)

    def clear(self) -> None:
        """Reset the resolver for a new resolution round."""
        self._responses.clear()
        self._expected_agents = 0

    @property
    def response_count(self) -> int:
        """Number of successful responses collected so far."""
        return len(self._responses)

    def resolve(self) -> ResolvedResult:
        """
        Resolve the collected responses into a single result.

        Uses the configured strategy to produce the best answer from
        available responses.

        Returns:
            ResolvedResult with the synthesized answer.

        Raises:
            ConflictResolutionError: If no responses are available.
        """
        start = time.time()

        if not self._responses:
            raise ConflictResolutionError(
                "No responses available for conflict resolution"
            )

        agents_responded = len(self._responses)
        agents_consulted = self._expected_agents or agents_responded

        if self.strategy == SynthesisStrategy.MAJORITY:
            result = self._resolve_majority()
        elif self.strategy == SynthesisStrategy.WEIGHTED:
            result = self._resolve_weighted()
        elif self.strategy == SynthesisStrategy.FIRST_RESPONSE:
            result = self._resolve_first_response()
        elif self.strategy == SynthesisStrategy.BEST_EVIDENCE:
            result = self._resolve_best_evidence()
        else:
            raise ConflictResolutionError(
                f"Unknown strategy: {self.strategy}"
            )

        result.agents_consulted = agents_consulted
        result.agents_responded = agents_responded
        result.resolution_time_ms = int((time.time() - start) * 1000)

        return result

    def _resolve_majority(self) -> ResolvedResult:
        """Pick the result value that appears most frequently."""
        # Normalize results to comparable form using JSON string
        vote_counts: Dict[str, List[CooperativeResponse]] = {}
        for resp in self._responses:
            key = _result_key(resp.result)
            vote_counts.setdefault(key, []).append(resp)

        # Find the majority
        best_key = max(vote_counts, key=lambda k: len(vote_counts[k]))
        winners = vote_counts[best_key]
        losers = [r for r in self._responses if r not in winners]

        total = len(self._responses)
        agreement_ratio = len(winners) / total if total > 0 else 0.0
        confidence = _majority_confidence(agreement_ratio, total)

        return ResolvedResult(
            strategy=SynthesisStrategy.MAJORITY.value,
            result=winners[0].result,
            confidence=confidence,
            contributing_responses=winners,
            discarded_responses=losers,
        )

    def _resolve_weighted(self) -> ResolvedResult:
        """Weight each agent's response by their trust score."""
        if self.trust is None:
            # Fall back to majority if no trust scorer configured
            return self._resolve_majority()

        # Normalize and weight
        vote_weights: Dict[str, float] = {}
        vote_responses: Dict[str, List[CooperativeResponse]] = {}
        total_weight = 0.0

        for resp in self._responses:
            key = _result_key(resp.result)
            weight = self.trust.get_score(resp.source_agent)
            vote_weights[key] = vote_weights.get(key, 0.0) + weight
            vote_responses.setdefault(key, []).append(resp)
            total_weight += weight

        best_key = max(vote_weights, key=lambda k: vote_weights[k])
        winners = vote_responses[best_key]
        losers = [r for r in self._responses if r not in winners]

        dominant_weight = vote_weights[best_key]
        confidence = dominant_weight / total_weight if total_weight > 0 else 0.0

        return ResolvedResult(
            strategy=SynthesisStrategy.WEIGHTED.value,
            result=winners[0].result,
            confidence=confidence,
            contributing_responses=winners,
            discarded_responses=losers,
        )

    def _resolve_first_response(self) -> ResolvedResult:
        """Accept the first response received."""
        winner = self._responses[0]
        losers = self._responses[1:]

        # Confidence degrades with disagreement
        total = len(self._responses)
        if total == 1:
            confidence = 0.5  # Only one response — no confirmation
        else:
            # How many others agree with the first?
            first_key = _result_key(winner.result)
            agree_count = sum(
                1 for r in self._responses
                if _result_key(r.result) == first_key
            )
            confidence = agree_count / total

        return ResolvedResult(
            strategy=SynthesisStrategy.FIRST_RESPONSE.value,
            result=winner.result,
            confidence=confidence,
            contributing_responses=[winner],
            discarded_responses=losers,
        )

    def _resolve_best_evidence(self) -> ResolvedResult:
        """Pick the response with the highest confidence/evidence score."""
        def evidence_score(resp: CooperativeResponse) -> float:
            """Compute an evidence quality score for a response."""
            score = 0.0

            # Execution time: extremely fast or slow may indicate poor quality
            exec_time = resp.execution_time_ms
            if 0 < exec_time < 60000:
                score += 0.2  # Reasonable execution time
            elif exec_time == 0:
                score += 0.05  # No timing info

            # VM info present: indicates a real execution environment
            if resp.vm_info is not None:
                score += 0.3

            # Result richness: more fields suggest a thorough response
            if resp.result:
                score += min(0.3, len(resp.result) * 0.05)

            # Trust score bonus if available
            if self.trust:
                score += self.trust.get_score(resp.source_agent) * 0.2

            return score

        self._responses.sort(key=evidence_score, reverse=True)
        winner = self._responses[0]
        losers = self._responses[1:]

        confidence = evidence_score(winner)
        # Normalize to [0, 1]
        confidence = min(1.0, max(0.0, confidence))

        return ResolvedResult(
            strategy=SynthesisStrategy.BEST_EVIDENCE.value,
            result=winner.result,
            confidence=confidence,
            contributing_responses=[winner],
            discarded_responses=losers,
        )


def _result_key(result: Dict[str, Any]) -> str:
    """
    Normalize a result dict to a comparable string key for voting.
    Uses sorted JSON to ensure equivalent dicts produce identical keys.
    """
    import json
    return json.dumps(result, sort_keys=True)


def _majority_confidence(agreement_ratio: float, total: int) -> float:
    """
    Compute confidence for majority voting.

    Confidence increases with:
    - Higher agreement ratio
    - More total responders
    """
    if total == 0:
        return 0.0

    base = agreement_ratio
    # Bonus for having more responders (up to a cap)
    sample_bonus = min(0.1, total * 0.025)
    # Perfect agreement bonus
    unanimous_bonus = 0.15 if agreement_ratio == 1.0 else 0.0

    return min(1.0, base + sample_bonus + unanimous_bonus)
