"""Generic HLS backend — raw AXI-Lite register control for custom FPGA designs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import (
    AbstractBackend,
    ExecutionResult,
    PulseInstruction,
    ReadoutInstruction,
)


# AXI-Lite register offsets for a generic quantum control IP
class Registers:
    CONTROL = 0x00
    STATUS = 0x04
    NUM_SHOTS = 0x08
    CHANNEL_SEL = 0x0C
    FREQ_LO = 0x10
    FREQ_HI = 0x14
    PHASE = 0x18
    AMPLITUDE = 0x1C
    DURATION = 0x20
    ENVELOPE = 0x24
    TRIGGER = 0x28
    RESULT_ADDR = 0x2C
    RESULT_DATA = 0x30
    RESULT_COUNT = 0x34


class GenericBackend(AbstractBackend):
    """Backend for custom HLS-based quantum control IP.

    Communicates with FPGA logic via AXI-Lite memory-mapped registers,
    using either PYNQ's MMIO or a provided register interface.

    Args:
        base_addr: AXI-Lite base address of the quantum control IP.
        num_channels: Number of DAC channels available.
        mmio: Optional MMIO-like object for register access (for testing).
    """

    def __init__(
        self,
        base_addr: int = 0x4000_0000,
        num_channels: int = 8,
        mmio: Any = None,
    ) -> None:
        self._base_addr = base_addr
        self._num_channels_val = num_channels
        self._mmio = mmio
        self._connected = False
        self._qubit_config: dict[int, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "generic"

    @property
    def num_channels(self) -> int:
        return self._num_channels_val

    def connect(self, **kwargs: Any) -> None:
        """Connect to the FPGA IP block.

        If no MMIO object was provided at construction, attempts to
        create one using pynq.MMIO.
        """
        if self._mmio is None:
            try:
                from pynq import MMIO  # type: ignore[import-not-found]

                addr_range = kwargs.get("addr_range", 0x1000)
                self._mmio = MMIO(self._base_addr, addr_range)
            except ImportError:
                raise ImportError(
                    "pynq package required for generic backend hardware access. "
                    "Install with: pip install pynq-quantum[pynq]"
                )
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False
        self._mmio = None

    def configure_qubit(self, qubit_id: int, frequency: float, **params: Any) -> None:
        self._qubit_config[qubit_id] = {"frequency": frequency, **params}
        if self._connected and self._mmio is not None:
            self._write_reg(Registers.CHANNEL_SEL, qubit_id)
            freq_int = int(frequency)
            self._write_reg(Registers.FREQ_LO, freq_int & 0xFFFFFFFF)
            self._write_reg(Registers.FREQ_HI, (freq_int >> 32) & 0xFFFFFFFF)

    def execute(
        self,
        pulses: list[PulseInstruction],
        readouts: list[ReadoutInstruction],
        shots: int = 1000,
    ) -> ExecutionResult:
        if not self._connected:
            raise RuntimeError("Backend not connected. Call connect() first.")

        self._write_reg(Registers.NUM_SHOTS, shots)

        # Program pulse sequence
        for pulse in pulses:
            self._program_pulse(pulse)

        # Program readouts
        for readout in readouts:
            self._program_readout(readout)

        # Trigger execution
        self._write_reg(Registers.TRIGGER, 1)

        # Wait for completion
        self._wait_done()

        # Read results
        counts = self._read_results(readouts, shots)

        return ExecutionResult(
            counts=counts,
            metadata={"backend": "generic", "base_addr": hex(self._base_addr)},
        )

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "max_qubits": self._num_channels_val,
            "supported_gates": ["X", "Y", "Z", "H", "RX", "RY", "RZ", "CNOT", "CZ"],
            "simulation": False,
            "base_addr": hex(self._base_addr),
            "register_interface": "axi-lite",
        }

    # --- Register access helpers ---

    def _write_reg(self, offset: int, value: int) -> None:
        """Write a 32-bit value to an AXI-Lite register."""
        if self._mmio is not None:
            self._mmio.write(offset, value)

    def _read_reg(self, offset: int) -> int:
        """Read a 32-bit value from an AXI-Lite register."""
        if self._mmio is not None:
            return self._mmio.read(offset)
        return 0

    def _program_pulse(self, pulse: PulseInstruction) -> None:
        """Write a pulse instruction to the IP registers."""
        self._write_reg(Registers.CHANNEL_SEL, pulse.channel)
        freq_int = int(pulse.frequency)
        self._write_reg(Registers.FREQ_LO, freq_int & 0xFFFFFFFF)
        self._write_reg(Registers.FREQ_HI, (freq_int >> 32) & 0xFFFFFFFF)
        self._write_reg(Registers.PHASE, int(pulse.phase * 1e6))  # micro-radians
        self._write_reg(Registers.AMPLITUDE, int(pulse.amplitude * 0xFFFF))
        self._write_reg(Registers.DURATION, int(pulse.duration * 1e9))  # nanoseconds

        envelope_map = {"gaussian": 0, "square": 1, "drag": 2, "flat_top": 3}
        self._write_reg(
            Registers.ENVELOPE, envelope_map.get(pulse.envelope, 1)
        )

    def _program_readout(self, readout: ReadoutInstruction) -> None:
        """Write a readout instruction to the IP registers."""
        self._write_reg(Registers.CHANNEL_SEL, readout.channel | 0x8000)
        freq_int = int(readout.frequency)
        self._write_reg(Registers.FREQ_LO, freq_int & 0xFFFFFFFF)
        self._write_reg(Registers.FREQ_HI, (freq_int >> 32) & 0xFFFFFFFF)
        self._write_reg(Registers.DURATION, int(readout.duration * 1e9))

    def _wait_done(self, timeout_ms: int = 5000) -> None:
        """Poll the status register until execution completes."""
        import time

        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            status = self._read_reg(Registers.STATUS)
            if status & 0x01:  # Done bit
                return
            time.sleep(0.001)
        raise TimeoutError("Generic backend execution timed out")

    def _read_results(
        self, readouts: list[ReadoutInstruction], shots: int
    ) -> dict[str, int]:
        """Read measurement results from the result FIFO."""
        measured_qubits = sorted({q for ro in readouts for q in ro.qubits})
        n = len(measured_qubits) if measured_qubits else 1

        result_count = self._read_reg(Registers.RESULT_COUNT)
        counts: dict[str, int] = {}

        for _ in range(min(result_count, shots)):
            val = self._read_reg(Registers.RESULT_DATA)
            bits = ""
            for i in range(n):
                bits += str((val >> i) & 1)
            counts[bits] = counts.get(bits, 0) + 1

        return counts if counts else {"0" * n: shots}
