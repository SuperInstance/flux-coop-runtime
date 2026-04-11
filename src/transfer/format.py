"""
FluxTransfer Binary Format — VM state serialization for cooperative execution.

Serializes a FLUX VM's execution state into a portable binary format
that can be sent to another agent for continued or parallel execution.

Binary Layout:
    [4 bytes] magic:   "FXTR" (0x46585452)
    [1 byte]  version: Format version (currently 1)
    [1 byte]  isa_ver: ISA version (0 = unified HALT=0x00)
    [4 bytes] source_pc: Program counter at transfer point
    [4 bytes] data_stack_size: Number of entries in data stack
    [N*4]    data_stack: Serialized int32 stack entries
    [4 bytes] signal_stack_size: Number of entries in signal stack
    [M*4]    signal_stack: Serialized int32 signal stack entries
    [64*4]   registers: GP registers (64 x int32 = 256 bytes)
    [64*4]   confidence: Confidence registers (64 x int32 = 256 bytes)
    [4 bytes] metadata_size: Length of JSON metadata
    [K]      metadata: JSON-encoded task context
    [4 bytes] checksum: CRC32 of all preceding bytes
"""

import struct
import json
import zlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

MAGIC = b"FXTR"
FORMAT_VERSION = 1
ISA_VERSION_UNIFIED = 0  # HALT = 0x00
NUM_REGISTERS = 64
MAX_METADATA_SIZE = 1048576  # 1MB


class FluxTransferError(Exception):
    """Raised when FluxTransfer serialization/deserialization fails."""
    pass


@dataclass
class FluxTransfer:
    """Portable FLUX VM state container."""
    source_pc: int = 0
    isa_version: int = ISA_VERSION_UNIFIED
    data_stack: List[int] = field(default_factory=list)
    signal_stack: List[int] = field(default_factory=list)
    registers: List[int] = field(default_factory=lambda: [0] * NUM_REGISTERS)
    confidence: List[int] = field(default_factory=lambda: [0] * NUM_REGISTERS)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate transfer data before serialization."""
        if self.isa_version != ISA_VERSION_UNIFIED:
            raise FluxTransferError(
                f"Unsupported ISA version: {self.isa_version}. "
                f"Only unified ISA (HALT=0x00) is supported in Phase 1."
            )
        if len(self.registers) != NUM_REGISTERS:
            raise FluxTransferError(
                f"Expected {NUM_REGISTERS} registers, got {len(self.registers)}"
            )
        if len(self.confidence) != NUM_REGISTERS:
            raise FluxTransferError(
                f"Expected {NUM_REGISTERS} confidence registers, got {len(self.confidence)}"
            )
        for i, val in enumerate(self.data_stack):
            if not (-2147483648 <= val <= 2147483647):
                raise FluxTransferError(
                    f"data_stack[{i}] = {val} out of int32 range"
                )
        meta_json = json.dumps(self.metadata)
        if len(meta_json.encode('utf-8')) > MAX_METADATA_SIZE:
            raise FluxTransferError(
                f"Metadata too large: {len(meta_json)} bytes "
                f"(max {MAX_METADATA_SIZE})"
            )

    def serialize(self) -> bytes:
        """Serialize transfer to binary format."""
        self.validate()
        buf = bytearray()

        # Header
        buf.extend(MAGIC)
        buf.append(FORMAT_VERSION)
        buf.append(self.isa_version)
        buf.extend(struct.pack("<I", self.source_pc))

        # Data stack
        buf.extend(struct.pack("<I", len(self.data_stack)))
        for val in self.data_stack:
            buf.extend(struct.pack("<i", val))

        # Signal stack
        buf.extend(struct.pack("<I", len(self.signal_stack)))
        for val in self.signal_stack:
            buf.extend(struct.pack("<i", val))

        # GP registers (always 64 entries)
        for val in self.registers:
            buf.extend(struct.pack("<i", val))

        # Confidence registers (always 64 entries)
        for val in self.confidence:
            buf.extend(struct.pack("<i", val))

        # Metadata (JSON)
        meta_bytes = json.dumps(self.metadata, separators=(',', ':')).encode('utf-8')
        buf.extend(struct.pack("<I", len(meta_bytes)))
        buf.extend(meta_bytes)

        # Checksum (CRC32 of all preceding bytes)
        checksum = zlib.crc32(bytes(buf)) & 0xFFFFFFFF
        buf.extend(struct.pack("<I", checksum))

        return bytes(buf)

    @classmethod
    def deserialize(cls, data: bytes) -> 'FluxTransfer':
        """Deserialize transfer from binary format."""
        if len(data) < 14:  # Minimum: magic(4) + version(1) + isa(1) + pc(4) + stack_size(4)
            raise FluxTransferError(f"Data too short: {len(data)} bytes")

        offset = 0

        # Magic
        if data[offset:offset+4] != MAGIC:
            raise FluxTransferError(
                f"Invalid magic: {data[offset:offset+4]!r}, expected {MAGIC!r}"
            )
        offset += 4

        # Version
        version = data[offset]
        if version != FORMAT_VERSION:
            raise FluxTransferError(
                f"Unsupported format version: {version}, expected {FORMAT_VERSION}"
            )
        offset += 1

        # ISA version
        isa_version = data[offset]
        offset += 1

        # Source PC
        source_pc = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        # Data stack
        data_stack_size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        data_stack = []
        for _ in range(data_stack_size):
            val = struct.unpack_from("<i", data, offset)[0]
            data_stack.append(val)
            offset += 4

        # Signal stack
        signal_stack_size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        signal_stack = []
        for _ in range(signal_stack_size):
            val = struct.unpack_from("<i", data, offset)[0]
            signal_stack.append(val)
            offset += 4

        # GP registers
        registers = []
        for _ in range(NUM_REGISTERS):
            val = struct.unpack_from("<i", data, offset)[0]
            registers.append(val)
            offset += 4

        # Confidence registers
        confidence = []
        for _ in range(NUM_REGISTERS):
            val = struct.unpack_from("<i", data, offset)[0]
            confidence.append(val)
            offset += 4

        # Metadata
        metadata_size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        metadata = json.loads(data[offset:offset+metadata_size].decode('utf-8'))
        offset += metadata_size

        # Checksum verification
        stored_checksum = struct.unpack_from("<I", data, offset)[0]
        computed_checksum = zlib.crc32(data[:offset]) & 0xFFFFFFFF
        if stored_checksum != computed_checksum:
            raise FluxTransferError(
                f"Checksum mismatch: stored 0x{stored_checksum:08x}, "
                f"computed 0x{computed_checksum:08x}"
            )

        return cls(
            source_pc=source_pc,
            isa_version=isa_version,
            data_stack=data_stack,
            signal_stack=signal_stack,
            registers=registers,
            confidence=confidence,
            metadata=metadata,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "source_pc": self.source_pc,
            "isa_version": self.isa_version,
            "data_stack_size": len(self.data_stack),
            "signal_stack_size": len(self.signal_stack),
            "registers": self.registers[:8],  # First 8 for readability
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"FluxTransfer(pc={self.source_pc}, "
            f"data_stack={len(self.data_stack)} entries, "
            f"registers=[{self.registers[0]}, {self.registers[1]}, ...])"
        )
