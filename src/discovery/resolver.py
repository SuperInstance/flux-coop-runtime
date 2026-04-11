"""
Agent Discovery and Resolution Layer.

Resolves agent targets (names, roles, capabilities) to fleet addresses
using the semantic routing table and local configuration.
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class ResolutionError(Exception):
    """Raised when agent resolution fails."""
    pass


@dataclass
class AgentAddress:
    """Resolved fleet address for an agent."""
    agent_name: str
    repo_url: str
    role: str
    specializations: List[str]
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "repo_url": self.repo_url,
            "role": self.role,
            "specializations": self.specializations,
            "confidence": self.confidence,
        }


# Built-in fleet knowledge (sourced from semantic_router.py)
# This is the local cache — updated when agents are discovered
FLEET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Oracle1": {
        "repo": "https://github.com/SuperInstance/oracle1-vessel",
        "role": "lighthouse",
        "specializations": ["vocabulary", "runtime-architecture", "coordination"],
        "confidence": 0.95,
    },
    "JetsonClaw1": {
        "repo": "https://github.com/Lucineer/JetsonClaw1-vessel",
        "role": "vessel",
        "specializations": ["hardware", "c-runtime", "cuda", "rust-crates"],
        "confidence": 0.85,
    },
    "Super Z": {
        "repo": "https://github.com/SuperInstance/superz-vessel",
        "role": "vessel",
        "specializations": [
            "spec-writing", "fleet-auditing", "bytecode-programs",
            "cross-specification-analysis", "a2a-integration",
        ],
        "confidence": 0.90,
    },
    "Babel": {
        "repo": "https://github.com/SuperInstance/babel-vessel",
        "role": "scout",
        "specializations": ["multilingual", "grammatical-analysis"],
        "confidence": 0.80,
    },
    "Quill": {
        "repo": "https://github.com/SuperInstance/superz-vessel",
        "role": "vessel",
        "specializations": [
            "protocol-design", "isa-convergence", "signal-language",
            "a2a-unification", "specification-authoring",
        ],
        "confidence": 0.90,
    },
}

# Capability to specialization mapping
CAPABILITY_MAP: Dict[str, str] = {
    "cuda": "cuda",
    "rust": "rust-crates",
    "hardware": "hardware",
    "gpu": "hardware",
    "c-runtime": "c-runtime",
    "spec-writing": "spec-writing",
    "specification": "spec-writing",
    "auditing": "fleet-auditing",
    "a2a": "a2a-integration",
    "protocol": "protocol-design",
    "isa": "isa-convergence",
    "signal": "signal-language",
    "vocabulary": "vocabulary",
    "multilingual": "multilingual",
}


def resolve(target: str, min_confidence: float = 0.5) -> AgentAddress:
    """
    Resolve an agent target to a fleet address.
    
    Supports multiple target formats:
    - Agent name: "Quill", "Oracle1"
    - Role prefix: "role:lighthouse"
    - Capability prefix: "cap:cuda"
    - Wildcard: "any"
    - URL: "https://github.com/..." (passed through)
    """
    if not target:
        raise ResolutionError("Empty target")

    # Direct name lookup
    if target in FLEET_REGISTRY:
        info = FLEET_REGISTRY[target]
        return AgentAddress(
            agent_name=target,
            repo_url=info["repo"],
            role=info["role"],
            specializations=info["specializations"],
            confidence=info["confidence"],
        )

    # Role-based lookup
    if target.startswith("role:"):
        role = target[5:]
        candidates = [
            (name, info) for name, info in FLEET_REGISTRY.items()
            if info["role"] == role and info["confidence"] >= min_confidence
        ]
        if not candidates:
            raise ResolutionError(
                f"No agent with role '{role}' above confidence {min_confidence}"
            )
        name, info = candidates[0]  # First match
        return AgentAddress(
            agent_name=name,
            repo_url=info["repo"],
            role=info["role"],
            specializations=info["specializations"],
            confidence=info["confidence"],
        )

    # Capability-based lookup
    if target.startswith("cap:"):
        cap = target[4:].lower()
        spec = CAPABILITY_MAP.get(cap, cap)
        candidates = []
        for name, info in FLEET_REGISTRY.items():
            if any(spec in s.lower() for s in info["specializations"]):
                if info["confidence"] >= min_confidence:
                    candidates.append((name, info))
        if not candidates:
            raise ResolutionError(
                f"No agent with capability '{cap}' above confidence {min_confidence}"
            )
        # Sort by confidence descending
        candidates.sort(key=lambda x: x[1]["confidence"], reverse=True)
        name, info = candidates[0]
        return AgentAddress(
            agent_name=name,
            repo_url=info["repo"],
            role=info["role"],
            specializations=info["specializations"],
            confidence=info["confidence"],
        )

    # Wildcard: pick best available
    if target == "any":
        candidates = [
            (name, info) for name, info in FLEET_REGISTRY.items()
            if info["confidence"] >= min_confidence
        ]
        if not candidates:
            raise ResolutionError(
                f"No agents available above confidence {min_confidence}"
            )
        candidates.sort(key=lambda x: x[1]["confidence"], reverse=True)
        name, info = candidates[0]
        return AgentAddress(
            agent_name=name,
            repo_url=info["repo"],
            role=info["role"],
            specializations=info["specializations"],
            confidence=info["confidence"],
        )

    # URL passthrough
    if target.startswith("http://") or target.startswith("https://"):
        return AgentAddress(
            agent_name=target.split("/")[-1],
            repo_url=target,
            role="unknown",
            specializations=[],
            confidence=0.5,
        )

    raise ResolutionError(f"Cannot resolve target: {target}")


def list_agents() -> List[Dict[str, Any]]:
    """List all known agents in the fleet registry."""
    return [
        {"name": name, **info}
        for name, info in FLEET_REGISTRY.items()
    ]


def register_agent(name: str, repo: str, role: str,
                   specializations: List[str], confidence: float = 0.5) -> None:
    """Register or update an agent in the local fleet registry."""
    FLEET_REGISTRY[name] = {
        "repo": repo,
        "role": role,
        "specializations": specializations,
        "confidence": confidence,
    }
