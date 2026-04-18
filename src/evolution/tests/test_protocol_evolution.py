"""
Unit tests for evolution/protocol_evolution.py.

Covers ProtocolVersion serialization/deserialization, ProtocolRegistry
registration, versioning, diffing, migration paths, and edge cases.
"""

import json
import pytest
from src.evolution.protocol_evolution import (
    ProtocolVersion,
    ProtocolVersionDiff,
    ProtocolRegistry,
)


class TestProtocolVersionCreation:
    """Test ProtocolVersion dataclass creation and auto-fields."""

    def test_basic_creation(self):
        v = ProtocolVersion(version="1.0.0", changelog="Initial version")
        assert v.version == "1.0.0"
        assert v.changelog == "Initial version"
        assert v.opcode_mapping == {}
        assert v.created_at  # Auto-populated by __post_init__
        assert v.predecessor is None
        assert v.metadata == {}

    def test_with_opcodes(self):
        v = ProtocolVersion(
            version="1.0.0",
            opcode_mapping={"0x50": "TELL", "0x51": "ASK"},
        )
        assert len(v.opcode_mapping) == 2
        assert v.opcode_mapping["0x51"] == "ASK"

    def test_with_predecessor(self):
        v = ProtocolVersion(
            version="1.1.0",
            predecessor="1.0.0",
        )
        assert v.predecessor == "1.0.0"

    def test_with_metadata(self):
        v = ProtocolVersion(
            version="1.0.0",
            metadata={"author": "Quill", "status": "stable"},
        )
        assert v.metadata["author"] == "Quill"
        assert v.metadata["status"] == "stable"

    def test_auto_created_at(self):
        import time
        before = time.strftime("%Y-%m-%dT%H:%M", time.gmtime())
        v = ProtocolVersion(version="1.0.0")
        after = time.strftime("%Y-%m-%dT%H:%M", time.gmtime())
        # created_at should have today's date
        assert v.created_at.startswith(before[:11])


class TestProtocolVersionSerialization:
    """Test ProtocolVersion to_dict/from_dict/to_json/from_json."""

    def test_to_dict_keys(self):
        v = ProtocolVersion(
            version="1.0.0",
            changelog="Initial",
            opcode_mapping={"0x50": "TELL"},
            predecessor=None,
            metadata={"k": "v"},
        )
        d = v.to_dict()
        assert d["version"] == "1.0.0"
        assert d["changelog"] == "Initial"
        assert d["opcode_mapping"] == {"0x50": "TELL"}
        assert d["predecessor"] is None
        assert d["metadata"] == {"k": "v"}
        assert "created_at" in d

    def test_to_dict_sorts_opcodes(self):
        v = ProtocolVersion(
            version="1.0.0",
            opcode_mapping={"0x52": "DELEGATE", "0x50": "TELL", "0x51": "ASK"},
        )
        d = v.to_dict()
        keys = list(d["opcode_mapping"].keys())
        assert keys == ["0x50", "0x51", "0x52"]

    def test_from_dict_roundtrip(self):
        original = ProtocolVersion(
            version="2.0.0",
            changelog="Major update",
            opcode_mapping={"0x50": "TELL", "0x51": "ASK", "0x53": "BROADCAST"},
            predecessor="1.1.0",
            metadata={"deprecated": ["0xFF"]},
        )
        restored = ProtocolVersion.from_dict(original.to_dict())
        assert restored.version == "2.0.0"
        assert restored.changelog == "Major update"
        assert restored.opcode_mapping == original.opcode_mapping
        assert restored.predecessor == "1.1.0"
        assert restored.metadata == {"deprecated": ["0xFF"]}

    def test_from_dict_with_missing_fields(self):
        data = {"version": "1.0.0"}
        v = ProtocolVersion.from_dict(data)
        assert v.version == "1.0.0"
        assert v.changelog == ""
        assert v.opcode_mapping == {}
        assert v.predecessor is None
        assert v.metadata == {}

    def test_to_json_valid(self):
        v = ProtocolVersion(version="1.0.0", opcode_mapping={"0x50": "TELL"})
        json_str = v.to_json()
        parsed = json.loads(json_str)
        assert parsed["version"] == "1.0.0"
        assert parsed["opcode_mapping"]["0x50"] == "TELL"

    def test_from_json_roundtrip(self):
        original = ProtocolVersion(
            version="1.5.0",
            changelog="Bugfix",
            opcode_mapping={"0x50": "TELL"},
        )
        json_str = original.to_json()
        restored = ProtocolVersion.from_json(json_str)
        assert restored.version == "1.5.0"
        assert restored.changelog == "Bugfix"
        assert restored.opcode_mapping == {"0x50": "TELL"}


class TestProtocolRegistry:
    """Test ProtocolRegistry registration and retrieval."""

    def test_register_and_get(self):
        reg = ProtocolRegistry()
        v = ProtocolVersion(version="1.0.0")
        reg.register(v)
        assert reg.get("1.0.0") is v

    def test_register_duplicate_raises(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(version="1.0.0"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(ProtocolVersion(version="1.0.0"))

    def test_get_missing_returns_none(self):
        reg = ProtocolRegistry()
        assert reg.get("9.9.9") is None

    def test_register_multiple_versions(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(version="1.0.0"))
        reg.register(ProtocolVersion(version="1.1.0"))
        reg.register(ProtocolVersion(version="2.0.0"))
        assert len(reg) == 3

    def test_all_versions_sorted(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(version="2.0.0"))
        reg.register(ProtocolVersion(version="1.0.0"))
        reg.register(ProtocolVersion(version="1.1.0"))
        assert reg.all_versions() == ["1.0.0", "1.1.0", "2.0.0"]

    def test_latest_returns_most_recent(self):
        reg = ProtocolRegistry()
        v1 = ProtocolVersion(version="1.0.0", created_at="2025-01-01T00:00:00Z")
        v2 = ProtocolVersion(version="2.0.0", created_at="2025-01-01T00:01:00Z")
        reg.register(v1)
        reg.register(v2)
        assert reg.latest().version == "2.0.0"

    def test_latest_empty_returns_none(self):
        reg = ProtocolRegistry()
        assert reg.latest() is None

    def test_repr(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(version="1.0.0"))
        reg.register(ProtocolVersion(version="2.0.0"))
        r = repr(reg)
        assert "ProtocolRegistry" in r
        assert "1.0.0" in r
        assert "2.0.0" in r


class TestProtocolVersionDiff:
    """Test diffing between protocol versions."""

    def _make_registry(self) -> ProtocolRegistry:
        """Helper to create a registry with known versions."""
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(
            version="1.0.0",
            opcode_mapping={"0x50": "TELL", "0x51": "ASK"},
            metadata={"phase": "1"},
        ))
        reg.register(ProtocolVersion(
            version="1.1.0",
            opcode_mapping={"0x50": "TELL", "0x51": "ASK", "0x53": "BROADCAST"},
            predecessor="1.0.0",
            metadata={"phase": "1", "new_field": True},
        ))
        reg.register(ProtocolVersion(
            version="2.0.0",
            opcode_mapping={"0x50": "TELL_NEW", "0x51": "ASK", "0x53": "BROADCAST", "0x70": "DISCUSS"},
            predecessor="1.1.0",
            metadata={"phase": "2", "new_field": True},
        ))
        return reg

    def test_added_opcodes(self):
        reg = self._make_registry()
        diff = reg.diff("1.0.0", "1.1.0")
        assert "0x53" in diff.added_opcodes
        assert diff.added_opcodes["0x53"] == "BROADCAST"
        assert len(diff.added_opcodes) == 1

    def test_removed_opcodes(self):
        reg = self._make_registry()
        # Create a version that removes an opcode
        reg.register(ProtocolVersion(
            version="3.0.0",
            opcode_mapping={"0x51": "ASK", "0x53": "BROADCAST"},
            predecessor="2.0.0",
        ))
        diff = reg.diff("2.0.0", "3.0.0")
        assert "0x50" in diff.removed_opcodes
        assert "0x70" in diff.removed_opcodes

    def test_modified_opcodes(self):
        reg = self._make_registry()
        diff = reg.diff("1.0.0", "2.0.0")
        assert "0x50" in diff.modified_opcodes
        assert diff.modified_opcodes["0x50"] == ("TELL", "TELL_NEW")

    def test_metadata_changes(self):
        reg = self._make_registry()
        diff = reg.diff("1.0.0", "1.1.0")
        assert "new_field" in diff.metadata_changes
        assert diff.metadata_changes["new_field"]["from"] is None
        assert diff.metadata_changes["new_field"]["to"] is True

    def test_no_changes_same_version(self):
        reg = self._make_registry()
        diff = reg.diff("1.0.0", "1.0.0")
        assert len(diff.added_opcodes) == 0
        assert len(diff.removed_opcodes) == 0
        assert len(diff.modified_opcodes) == 0

    def test_diff_missing_old_raises(self):
        reg = self._make_registry()
        with pytest.raises(ValueError, match="not registered"):
            reg.diff("9.0.0", "1.0.0")

    def test_diff_missing_new_raises(self):
        reg = self._make_registry()
        with pytest.raises(ValueError, match="not registered"):
            reg.diff("1.0.0", "9.0.0")

    def test_diff_to_dict(self):
        reg = self._make_registry()
        diff = reg.diff("1.0.0", "1.1.0")
        d = diff.to_dict()
        assert d["old_version"] == "1.0.0"
        assert d["new_version"] == "1.1.0"
        assert "added_opcodes" in d
        assert "removed_opcodes" in d
        assert "modified_opcodes" in d
        assert "metadata_changes" in d

    def test_diff_to_json(self):
        reg = self._make_registry()
        diff = reg.diff("1.0.0", "1.1.0")
        json_str = diff.to_json()
        parsed = json.loads(json_str)
        assert parsed["old_version"] == "1.0.0"
        assert parsed["new_version"] == "1.1.0"

    def test_diff_repr(self):
        reg = self._make_registry()
        diff = reg.diff("1.0.0", "1.1.0")
        r = repr(diff)
        assert "ProtocolVersionDiff" in r
        assert "1.0.0" in r
        assert "1.1.0" in r
        assert "+1" in r  # 1 added


class TestMigrationPath:
    """Test migration path generation."""

    def test_same_version_returns_single(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(version="1.0.0"))
        path = reg.migration_path("1.0.0", "1.0.0")
        assert path == ["1.0.0"]

    def test_linear_migration(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(version="1.0.0"))
        reg.register(ProtocolVersion(version="1.1.0", predecessor="1.0.0"))
        reg.register(ProtocolVersion(version="1.2.0", predecessor="1.1.0"))
        path = reg.migration_path("1.0.0", "1.2.0")
        assert path == ["1.0.0", "1.1.0", "1.2.0"]

    def test_backward_migration(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(version="1.0.0"))
        reg.register(ProtocolVersion(version="1.1.0", predecessor="1.0.0"))
        path = reg.migration_path("1.1.0", "1.0.0")
        assert path == ["1.1.0", "1.0.0"]

    def test_missing_from_version_raises(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(version="1.0.0"))
        with pytest.raises(ValueError, match="not registered"):
            reg.migration_path("9.0.0", "1.0.0")

    def test_missing_to_version_raises(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(version="1.0.0"))
        with pytest.raises(ValueError, match="not registered"):
            reg.migration_path("1.0.0", "9.0.0")

    def test_no_path_raises(self):
        """Two disconnected version trees should raise ValueError."""
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(version="1.0.0"))
        reg.register(ProtocolVersion(version="2.0.0"))  # No predecessor link
        with pytest.raises(ValueError, match="No migration path"):
            reg.migration_path("1.0.0", "2.0.0")


class TestRegistrySerialization:
    """Test full registry JSON serialization."""

    def test_to_dict(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(
            version="1.0.0",
            opcode_mapping={"0x50": "TELL"},
        ))
        reg.register(ProtocolVersion(
            version="1.1.0",
            predecessor="1.0.0",
        ))
        d = reg.to_dict()
        assert "versions" in d
        assert "1.0.0" in d["versions"]
        assert "1.1.0" in d["versions"]

    def test_from_dict_roundtrip(self):
        original = ProtocolRegistry()
        original.register(ProtocolVersion(
            version="1.0.0",
            opcode_mapping={"0x50": "TELL", "0x51": "ASK"},
            metadata={"author": "Quill"},
        ))
        original.register(ProtocolVersion(
            version="2.0.0",
            predecessor="1.0.0",
            opcode_mapping={"0x50": "TELL", "0x51": "ASK", "0x53": "BROADCAST"},
        ))

        restored = ProtocolRegistry.from_dict(original.to_dict())
        assert len(restored) == 2
        v1 = restored.get("1.0.0")
        assert v1 is not None
        assert v1.opcode_mapping == {"0x50": "TELL", "0x51": "ASK"}
        v2 = restored.get("2.0.0")
        assert v2.predecessor == "1.0.0"
        assert "0x53" in v2.opcode_mapping

    def test_from_json_roundtrip(self):
        original = ProtocolRegistry()
        original.register(ProtocolVersion(version="1.0.0"))
        original.register(ProtocolVersion(
            version="1.1.0",
            predecessor="1.0.0",
            opcode_mapping={"0x51": "ASK"},
        ))

        json_str = original.to_json()
        restored = ProtocolRegistry.from_json(json_str)
        assert len(restored) == 2
        assert restored.get("1.1.0").opcode_mapping == {"0x51": "ASK"}

    def test_from_dict_empty(self):
        reg = ProtocolRegistry.from_dict({"versions": {}})
        assert len(reg) == 0


class TestProtocolVersionEdgeCases:
    """Edge cases for protocol evolution."""

    def test_version_with_empty_opcode_mapping(self):
        v = ProtocolVersion(version="0.9.0", opcode_mapping={})
        assert len(v.opcode_mapping) == 0
        d = v.to_dict()
        assert d["opcode_mapping"] == {}

    def test_version_with_large_metadata(self):
        meta = {f"key_{i}": f"value_{i}" for i in range(100)}
        v = ProtocolVersion(version="1.0.0", metadata=meta)
        restored = ProtocolVersion.from_dict(v.to_dict())
        assert restored.metadata == meta

    def test_diff_between_identical_versions(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(
            version="1.0.0",
            opcode_mapping={"0x50": "TELL", "0x51": "ASK"},
            metadata={"a": 1},
        ))
        diff = reg.diff("1.0.0", "1.0.0")
        assert len(diff.metadata_changes) == 0
        assert len(diff.added_opcodes) == 0
        assert len(diff.removed_opcodes) == 0
        assert len(diff.modified_opcodes) == 0

    def test_metadata_key_removed(self):
        reg = ProtocolRegistry()
        reg.register(ProtocolVersion(
            version="1.0.0",
            metadata={"a": 1, "b": 2},
        ))
        reg.register(ProtocolVersion(
            version="2.0.0",
            predecessor="1.0.0",
            metadata={"a": 1},
        ))
        diff = reg.diff("1.0.0", "2.0.0")
        assert "b" in diff.metadata_changes
        assert diff.metadata_changes["b"]["from"] == 2
        assert diff.metadata_changes["b"]["to"] is None
