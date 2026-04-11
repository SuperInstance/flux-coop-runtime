"""Tests for FluxTransfer binary format."""

import struct
import zlib
import pytest
from src.transfer.format import FluxTransfer, FluxTransferError, MAGIC, NUM_REGISTERS


class TestFluxTransferSerialization:
    """Test serialization and deserialization of FluxTransfer."""

    def test_empty_transfer_roundtrip(self):
        """Empty transfer (no stack, zero registers) serializes and deserializes."""
        ft = FluxTransfer(source_pc=0, metadata={"task": "ping"})
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        assert result.source_pc == 0
        assert result.data_stack == []
        assert result.signal_stack == []
        assert result.registers == [0] * NUM_REGISTERS
        assert result.metadata == {"task": "ping"}

    def test_with_data_stack(self):
        """Data stack values are preserved through roundtrip."""
        ft = FluxTransfer(
            source_pc=42,
            data_stack=[100, 200, -300],
            metadata={"request_type": "execute_bytecode"}
        )
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        assert result.source_pc == 42
        assert result.data_stack == [100, 200, -300]

    def test_with_registers(self):
        """Register values are preserved through roundtrip."""
        regs = [0] * NUM_REGISTERS
        regs[0] = 42
        regs[1] = 1000
        regs[63] = -1
        ft = FluxTransfer(registers=regs)
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        assert result.registers[0] == 42
        assert result.registers[1] == 1000
        assert result.registers[63] == -1

    def test_with_signal_stack(self):
        """Signal stack values are preserved."""
        ft = FluxTransfer(signal_stack=[0x51, 0x00, 0x0C])
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        assert result.signal_stack == [0x51, 0x00, 0x0C]

    def test_with_confidence(self):
        """Confidence registers are preserved."""
        conf = [0] * NUM_REGISTERS
        conf[0] = 95
        conf[5] = 42
        ft = FluxTransfer(confidence=conf)
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        assert result.confidence[0] == 95
        assert result.confidence[5] == 42

    def test_magic_bytes(self):
        """Serialized data starts with FXTR magic."""
        ft = FluxTransfer()
        data = ft.serialize()
        assert data[:4] == MAGIC

    def test_checksum_verification(self):
        """Corrupted data fails checksum verification."""
        ft = FluxTransfer(data_stack=[1, 2, 3])
        data = bytearray(ft.serialize())
        # Flip a byte in the data section (after header)
        data[20] ^= 0xFF
        with pytest.raises(FluxTransferError, match="Checksum mismatch"):
            FluxTransfer.deserialize(bytes(data))

    def test_invalid_magic(self):
        """Wrong magic bytes raise error."""
        ft = FluxTransfer()
        data = bytearray(ft.serialize())
        data[0:4] = b"BAD!"
        with pytest.raises(FluxTransferError, match="Invalid magic"):
            FluxTransfer.deserialize(bytes(data))

    def test_data_too_short(self):
        """Truncated data raises error."""
        with pytest.raises(FluxTransferError, match="too short"):
            FluxTransfer.deserialize(b"FXTR")

    def test_int32_overflow_in_stack(self):
        """Stack values outside int32 range raise error."""
        ft = FluxTransfer(data_stack=[2**31])  # max int32 + 1
        with pytest.raises(FluxTransferError, match="out of int32 range"):
            ft.serialize()

    def test_negative_int32_in_stack(self):
        """Negative values in int32 range work correctly."""
        ft = FluxTransfer(data_stack=[-1, -2147483648])
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        assert result.data_stack == [-1, -2147483648]

    def test_wrong_register_count(self):
        """Wrong number of registers raises error."""
        ft = FluxTransfer(registers=[0, 1, 2])  # Only 3, need 64
        with pytest.raises(FluxTransferError, match="Expected 64 registers"):
            ft.serialize()

    def test_wrong_confidence_count(self):
        """Wrong number of confidence registers raises error."""
        ft = FluxTransfer(confidence=[0])
        with pytest.raises(FluxTransferError, match="Expected 64 confidence"):
            ft.serialize()

    def test_metadata_too_large(self):
        """Metadata exceeding 1MB raises error."""
        big_meta = {"data": "x" * (1048577)}
        ft = FluxTransfer(metadata=big_meta)
        with pytest.raises(FluxTransferError, match="Metadata too large"):
            ft.serialize()

    def test_complex_metadata(self):
        """Complex nested metadata survives roundtrip."""
        meta = {
            "request_type": "execute_bytecode",
            "bytecode": [0x18, 0, 42, 0x08, 0, 0x00],
            "expected_result": "register_0",
            "timeout_ms": 30000,
            "nested": {"key": "value", "numbers": [1, 2, 3]},
        }
        ft = FluxTransfer(metadata=meta)
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        assert result.metadata == meta

    def test_to_dict(self):
        """to_dict produces readable summary."""
        ft = FluxTransfer(source_pc=42, data_stack=[1, 2, 3])
        d = ft.to_dict()
        assert d["source_pc"] == 42
        assert d["data_stack_size"] == 3
        assert len(d["registers"]) == 8  # First 8 only

    def test_repr(self):
        """repr is informative."""
        ft = FluxTransfer(source_pc=42, data_stack=[1, 2])
        r = repr(ft)
        assert "pc=42" in r
        assert "2 entries" in r


class TestFluxTransferFromVMState:
    """Test creating FluxTransfer from actual VM-like state."""

    def test_from_execution_trace(self):
        """Simulate a VM execution and capture state."""
        # Simulate: MOVI R0, 42; MOVI R1, 100; ADD R2, R0, R1; HALT
        registers = [0] * NUM_REGISTERS
        registers[0] = 42
        registers[1] = 100
        registers[2] = 142
        data_stack = [42, 100]
        
        ft = FluxTransfer(
            source_pc=9,  # At HALT instruction
            registers=registers,
            data_stack=data_stack,
            metadata={
                "bytecode": [0x18, 0, 42, 0x18, 1, 100, 0x20, 2, 0, 1, 0x00],
                "description": "R2 = 42 + 100 = 142"
            }
        )
        
        data = ft.serialize()
        result = FluxTransfer.deserialize(data)
        
        assert result.source_pc == 9
        assert result.registers[2] == 142
        assert result.data_stack == [42, 100]
        assert result.metadata["description"] == "R2 = 42 + 100 = 142"
