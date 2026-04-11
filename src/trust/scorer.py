from src.cooperative_types import TrustRecord
"""
Trust Scoring Layer.

Tracks agent reliability for cooperative task execution.
Phase 1: Simple success/failure/timeout counter.
Phase 3: Sophisticated multi-dimensional trust model.
"""

import json
import os
import time
from typing import Dict, Optional
# TrustRecord imported from cooperative_types


class TrustScorer:
    """Manages trust scores for fleet agents."""

    def __init__(self, persistence_path: Optional[str] = None):
        self._records: Dict[str, TrustRecord] = {}
        self._persistence_path = persistence_path
        if persistence_path and os.path.exists(persistence_path):
            self._load()

    def get_record(self, agent_name: str) -> TrustRecord:
        """Get or create a trust record for an agent."""
        if agent_name not in self._records:
            self._records[agent_name] = TrustRecord(agent_name=agent_name)
        return self._records[agent_name]

    def get_score(self, agent_name: str) -> float:
        """Get trust score for an agent (0.0 to 1.0)."""
        return self.get_record(agent_name).score

    def record_result(self, agent_name: str, status: str) -> float:
        """Record a task result and return updated score."""
        record = self.get_record(agent_name)
        record.record(status)
        if self._persistence_path:
            self._save()
        return record.score

    def rank_agents(self, agents: list, min_score: float = 0.0) -> list:
        """Rank agents by trust score, filtering below minimum."""
        scored = []
        for agent in agents:
            score = self.get_score(agent)
            if score >= min_score:
                scored.append((agent, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_all_records(self) -> Dict[str, Dict]:
        """Get all trust records as dictionaries."""
        return {name: rec.to_dict() for name, rec in self._records.items()}

    def _save(self) -> None:
        """Persist trust records to disk."""
        if not self._persistence_path:
            return
        data = {name: rec.to_dict() for name, rec in self._records.items()}
        os.makedirs(os.path.dirname(self._persistence_path) or ".", exist_ok=True)
        with open(self._persistence_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load trust records from disk."""
        if not self._persistence_path:
            return
        with open(self._persistence_path, 'r') as f:
            content = f.read()
            if not content.strip():
                return
            data = json.loads(content)
        for name, info in data.items():
            rec = self.get_record(name)
            rec.successes = info.get("successes", 0)
            rec.failures = info.get("failures", 0)
            rec.timeouts = info.get("timeouts", 0)
            rec.last_seen = info.get("last_seen", "")
