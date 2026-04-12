"""
Protocol Evolution Tracker.

Tracks how cooperative protocols evolve over time, maintaining a versioned
registry of protocol definitions with diffing, migration paths, and
JSON serialization for persistence.

Cooperative protocols change as the fleet matures: new opcodes are added,
payload formats evolve, and compatibility requirements shift. This module
provides the infrastructure to manage that evolution deterministically.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class ProtocolVersion:
    """
    A single version of the cooperative protocol.

    Attributes:
        version: Semantic version string (e.g. "1.0.0").
        changelog: Human-readable description of changes from previous version.
        opcode_mapping: Map of opcode hex values to their semantic names
                       and argument signatures (e.g. {"0x50": "TELL", "0x51": "ASK"}).
        created_at: ISO-8601 timestamp of when this version was registered.
        predecessor: Version string of the immediately preceding version, or None for the root.
        metadata: Arbitrary additional metadata attached to this version.
    """
    version: str
    changelog: str = ""
    opcode_mapping: Dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    predecessor: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "version": self.version,
            "changelog": self.changelog,
            "opcode_mapping": dict(sorted(self.opcode_mapping.items())),
            "created_at": self.created_at,
            "predecessor": self.predecessor,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtocolVersion':
        """Deserialize from a dictionary."""
        return cls(
            version=data["version"],
            changelog=data.get("changelog", ""),
            opcode_mapping=data.get("opcode_mapping", {}),
            created_at=data.get("created_at", ""),
            predecessor=data.get("predecessor"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'ProtocolVersion':
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))


class ProtocolVersionDiff:
    """
    Represents the differences between two protocol versions.

    Produces a structured diff listing added, removed, and modified opcodes
    as well as metadata changes.
    """

    def __init__(self, old_version: str, new_version: str):
        self.old_version = old_version
        self.new_version = new_version
        self.added_opcodes: Dict[str, str] = {}
        self.removed_opcodes: Dict[str, str] = {}
        self.modified_opcodes: Dict[str, Tuple[str, str]] = {}  # opcode -> (old_name, new_name)
        self.metadata_changes: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "added_opcodes": self.added_opcodes,
            "removed_opcodes": self.removed_opcodes,
            "modified_opcodes": {
                k: {"from": v[0], "to": v[1]}
                for k, v in self.modified_opcodes.items()
            },
            "metadata_changes": self.metadata_changes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __repr__(self) -> str:
        n_added = len(self.added_opcodes)
        n_removed = len(self.removed_opcodes)
        n_modified = len(self.modified_opcodes)
        return (
            f"ProtocolVersionDiff({self.old_version} -> {self.new_version}, "
            f"+{n_added} -{n_removed} ~{n_modified} opcodes)"
        )


class ProtocolRegistry:
    """
    Maintains version history of the cooperative protocol.

    Supports registering new versions, diffing between versions,
    generating migration paths, and serializing the full registry to JSON.

    Usage:
        registry = ProtocolRegistry()
        registry.register(ProtocolVersion("1.0.0", opcode_mapping={"0x51": "ASK"}))
        registry.register(ProtocolVersion("1.1.0", opcode_mapping={"0x51": "ASK", "0x50": "TELL"},
                                          predecessor="1.0.0"))
        diff = registry.diff("1.0.0", "1.1.0")
    """

    def __init__(self):
        self._versions: Dict[str, ProtocolVersion] = {}

    def register(self, version: ProtocolVersion) -> None:
        """
        Register a new protocol version.

        Raises:
            ValueError: If a version with the same string already exists.
        """
        if version.version in self._versions:
            raise ValueError(
                f"Protocol version '{version.version}' is already registered"
            )
        self._versions[version.version] = version

    def get(self, version_str: str) -> Optional[ProtocolVersion]:
        """Retrieve a registered protocol version by string, or None."""
        return self._versions.get(version_str)

    def latest(self) -> Optional[ProtocolVersion]:
        """Return the most recently registered version, or None if empty."""
        if not self._versions:
            return None
        return max(self._versions.values(), key=lambda v: v.created_at)

    def all_versions(self) -> List[str]:
        """Return sorted list of all registered version strings."""
        return sorted(self._versions.keys())

    def diff(self, old_version: str, new_version: str) -> ProtocolVersionDiff:
        """
        Compare two protocol versions and produce a structured diff.

        Args:
            old_version: The earlier version string to compare from.
            new_version: The later version string to compare to.

        Returns:
            ProtocolVersionDiff with added/removed/modified opcodes.

        Raises:
            ValueError: If either version is not registered.
        """
        old = self._versions.get(old_version)
        new = self._versions.get(new_version)
        if old is None:
            raise ValueError(f"Version '{old_version}' is not registered")
        if new is None:
            raise ValueError(f"Version '{new_version}' is not registered")

        result = ProtocolVersionDiff(old_version, new_version)

        old_opcodes = old.opcode_mapping
        new_opcodes = new.opcode_mapping

        # Added opcodes: in new but not in old
        for opcode, name in new_opcodes.items():
            if opcode not in old_opcodes:
                result.added_opcodes[opcode] = name

        # Removed opcodes: in old but not in new
        for opcode, name in old_opcodes.items():
            if opcode not in new_opcodes:
                result.removed_opcodes[opcode] = name

        # Modified opcodes: same opcode, different name
        for opcode in old_opcodes:
            if opcode in new_opcodes and old_opcodes[opcode] != new_opcodes[opcode]:
                result.modified_opcodes[opcode] = (
                    old_opcodes[opcode],
                    new_opcodes[opcode],
                )

        # Metadata changes
        for key in set(list(old.metadata.keys()) + list(new.metadata.keys())):
            old_val = old.metadata.get(key)
            new_val = new.metadata.get(key)
            if old_val != new_val:
                result.metadata_changes[key] = {
                    "from": old_val,
                    "to": new_val,
                }

        return result

    def migration_path(self, from_version: str, to_version: str) -> List[str]:
        """
        Generate a migration path from one version to another.

        Follows the predecessor chain to find the shortest sequence of
        versions to traverse.

        Args:
            from_version: Starting version string.
            to_version: Target version string.

        Returns:
            Ordered list of version strings to apply, inclusive of both endpoints.

        Raises:
            ValueError: If either version is not registered, or no path exists.
        """
        if from_version not in self._versions:
            raise ValueError(f"Version '{from_version}' is not registered")
        if to_version not in self._versions:
            raise ValueError(f"Version '{to_version}' is not registered")

        if from_version == to_version:
            return [from_version]

        # Build predecessor graph
        children: Dict[str, List[str]] = {}
        for v_str, v_obj in self._versions.items():
            pred = v_obj.predecessor
            if pred:
                children.setdefault(pred, []).append(v_str)

        # BFS from from_version to to_version
        from collections import deque
        queue: deque = deque()
        queue.append((from_version, [from_version]))
        visited = {from_version}

        while queue:
            current, path = queue.popleft()
            if current == to_version:
                return path

            # Forward: follow children
            for child in children.get(current, []):
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))

            # Also follow predecessor (for backward migration)
            pred = self._versions[current].predecessor
            if pred and pred not in visited and pred in self._versions:
                visited.add(pred)
                queue.append((pred, path + [pred]))

        raise ValueError(
            f"No migration path from '{from_version}' to '{to_version}'"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full registry to a dictionary."""
        return {
            "versions": {
                v_str: v_obj.to_dict()
                for v_str, v_obj in sorted(self._versions.items())
            }
        }

    def to_json(self) -> str:
        """Serialize the full registry to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtocolRegistry':
        """Deserialize a registry from a dictionary."""
        registry = cls()
        for v_str, v_data in data.get("versions", {}).items():
            registry.register(ProtocolVersion.from_dict(v_data))
        return registry

    @classmethod
    def from_json(cls, json_str: str) -> 'ProtocolRegistry':
        """Deserialize a registry from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __len__(self) -> int:
        return len(self._versions)

    def __repr__(self) -> str:
        versions = ", ".join(sorted(self._versions.keys()))
        return f"ProtocolRegistry([{versions}])"
