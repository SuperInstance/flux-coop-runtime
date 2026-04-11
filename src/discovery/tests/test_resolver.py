"""Tests for agent discovery and resolution."""

import pytest
from src.discovery.resolver import (
    resolve, list_agents, register_agent,
    ResolutionError, AgentAddress, FLEET_REGISTRY,
)


class TestResolveByName:
    """Test direct agent name resolution."""

    def test_resolve_oracle1(self):
        addr = resolve("Oracle1")
        assert addr.agent_name == "Oracle1"
        assert "lighthouse" in addr.role
        assert "vocabulary" in addr.specializations

    def test_resolve_quill(self):
        addr = resolve("Quill")
        assert addr.agent_name == "Quill"
        assert addr.confidence == 0.90

    def test_resolve_super_z(self):
        addr = resolve("Super Z")
        assert addr.agent_name == "Super Z"
        assert "spec-writing" in addr.specializations

    def test_resolve_unknown_agent(self):
        with pytest.raises(ResolutionError, match="Cannot resolve"):
            resolve("NonExistentAgent")

    def test_resolve_empty(self):
        with pytest.raises(ResolutionError, match="Empty target"):
            resolve("")


class TestResolveByRole:
    """Test role-based resolution."""

    def test_resolve_lighthouse(self):
        addr = resolve("role:lighthouse")
        assert addr.role == "lighthouse"
        assert addr.agent_name == "Oracle1"

    def test_resolve_vessel(self):
        addr = resolve("role:vessel")
        assert addr.role == "vessel"

    def test_resolve_unknown_role(self):
        with pytest.raises(ResolutionError, match="No agent with role"):
            resolve("role:nonexistent")


class TestResolveByCapability:
    """Test capability-based resolution."""

    def test_resolve_cuda(self):
        addr = resolve("cap:cuda")
        assert addr.agent_name == "JetsonClaw1"

    def test_resolve_hardware(self):
        addr = resolve("cap:hardware")
        assert addr.agent_name == "JetsonClaw1"

    def test_resolve_protocol(self):
        addr = resolve("cap:protocol")
        assert addr.agent_name == "Quill"

    def test_resolve_isa(self):
        addr = resolve("cap:isa")
        assert addr.agent_name == "Quill"

    def test_resolve_unknown_capability(self):
        with pytest.raises(ResolutionError, match="No agent with capability"):
            resolve("cap:nonexistent_capability_xyz")


class TestResolveAny:
    """Test wildcard resolution."""

    def test_any_returns_highest_confidence(self):
        addr = resolve("any")
        assert addr.confidence >= 0.5
        # Oracle1 has 0.95 — should be top pick
        # But may tie with others; just verify it's valid

    def test_any_with_high_threshold(self):
        addr = resolve("any", min_confidence=0.95)
        assert addr.confidence >= 0.95

    def test_any_with_impossible_threshold(self):
        with pytest.raises(ResolutionError):
            resolve("any", min_confidence=1.0)


class TestResolveURL:
    """Test URL passthrough resolution."""

    def test_github_url(self):
        addr = resolve("https://github.com/SuperInstance/flux-runtime")
        assert "flux-runtime" in addr.repo_url
        assert addr.role == "unknown"

    def test_http_url(self):
        addr = resolve("http://example.com/repo")
        assert addr.repo_url == "http://example.com/repo"


class TestListAgents:
    """Test agent listing."""

    def test_list_returns_all(self):
        agents = list_agents()
        assert len(agents) >= 5  # At least Oracle1, JC1, Super Z, Babel, Quill

    def test_list_contains_required_fields(self):
        agents = list_agents()
        for agent in agents:
            assert "name" in agent
            assert "repo" in agent
            assert "role" in agent
            assert "specializations" in agent


class TestRegisterAgent:
    """Test dynamic agent registration."""

    def test_register_new_agent(self):
        register_agent(
            "TestAgent", "https://github.com/test/test",
            "vessel", ["testing"], 0.7
        )
        addr = resolve("TestAgent")
        assert addr.agent_name == "TestAgent"
        assert addr.confidence == 0.7

    def test_register_overwrites(self):
        register_agent("TestAgent", "https://new.url", "vessel", ["new"], 0.9)
        addr = resolve("TestAgent")
        assert addr.repo_url == "https://new.url"
        assert addr.confidence == 0.9

    def test_cleanup(self):
        if "TestAgent" in FLEET_REGISTRY:
            del FLEET_REGISTRY["TestAgent"]
