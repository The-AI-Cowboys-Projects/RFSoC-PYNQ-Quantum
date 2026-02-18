"""Tests for the generic HLS backend (mocked MMIO)."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from pynq_quantum.backends.generic import GenericBackend, Registers
from pynq_quantum.backends.base import PulseInstruction, ReadoutInstruction


class TestGenericBackendInit:
    def test_default_init(self):
        backend = GenericBackend()
        assert backend.name == "generic"
        assert backend.num_channels == 8
        assert not backend._connected

    def test_custom_init(self, mock_mmio):
        backend = GenericBackend(
            base_addr=0x8000_0000, num_channels=4, mmio=mock_mmio
        )
        assert backend._base_addr == 0x8000_0000
        assert backend.num_channels == 4


class TestGenericBackendConnect:
    def test_connect_with_mmio(self, mock_mmio):
        backend = GenericBackend(mmio=mock_mmio)
        backend.connect()
        assert backend._connected

    def test_connect_without_pynq_raises(self):
        backend = GenericBackend()
        with pytest.raises(ImportError, match="pynq"):
            backend.connect()

    def test_disconnect(self, mock_mmio):
        backend = GenericBackend(mmio=mock_mmio)
        backend.connect()
        backend.disconnect()
        assert not backend._connected


class TestGenericBackendOperations:
    @pytest.fixture
    def backend(self, mock_mmio):
        # Make read return done bit (0x01) for status, and some data
        call_count = [0]

        def mock_read(offset):
            if offset == Registers.STATUS:
                return 0x01  # Done
            if offset == Registers.RESULT_COUNT:
                return 3
            if offset == Registers.RESULT_DATA:
                call_count[0] += 1
                return call_count[0] % 4  # Varying bit patterns
            return 0

        mock_mmio.read.side_effect = mock_read
        b = GenericBackend(mmio=mock_mmio)
        b.connect()
        yield b
        b.disconnect()

    def test_configure_qubit(self, backend, mock_mmio):
        backend.configure_qubit(0, frequency=5e9)
        assert 0 in backend._qubit_config
        # Should write channel select and frequency registers
        mock_mmio.write.assert_any_call(Registers.CHANNEL_SEL, 0)

    def test_execute_not_connected_raises(self):
        backend = GenericBackend()
        with pytest.raises(RuntimeError, match="not connected"):
            backend.execute([], [], shots=10)

    def test_execute_basic(self, backend, mock_mmio):
        pulse = PulseInstruction(
            channel=0, frequency=5e9, phase=0.0,
            amplitude=0.5, duration=40e-9, envelope="gaussian",
        )
        readout = ReadoutInstruction(
            channel=0, frequency=7e9, duration=1e-6, qubits=[0],
        )
        result = backend.execute([pulse], [readout], shots=100)
        assert isinstance(result.counts, dict)
        assert result.metadata["backend"] == "generic"

        # Verify trigger was written
        mock_mmio.write.assert_any_call(Registers.TRIGGER, 1)
        mock_mmio.write.assert_any_call(Registers.NUM_SHOTS, 100)

    def test_get_capabilities(self, backend):
        caps = backend.get_capabilities()
        assert caps["register_interface"] == "axi-lite"
        assert caps["simulation"] is False

    def test_execute_timeout(self, mock_mmio):
        mock_mmio.read.return_value = 0x00  # Never done
        backend = GenericBackend(mmio=mock_mmio)
        backend.connect()

        pulse = PulseInstruction(
            channel=0, frequency=5e9, phase=0.0,
            amplitude=0.5, duration=40e-9,
        )
        readout = ReadoutInstruction(
            channel=0, frequency=7e9, duration=1e-6, qubits=[0],
        )
        with pytest.raises(TimeoutError):
            backend._wait_done(timeout_ms=10)

    def test_envelope_mapping(self, backend, mock_mmio):
        """Different envelopes should map to different register values."""
        for envelope, expected_val in [
            ("gaussian", 0), ("square", 1), ("drag", 2), ("flat_top", 3),
        ]:
            pulse = PulseInstruction(
                channel=0, frequency=5e9, phase=0.0,
                amplitude=0.5, duration=40e-9, envelope=envelope,
            )
            backend._program_pulse(pulse)
            mock_mmio.write.assert_any_call(Registers.ENVELOPE, expected_val)
