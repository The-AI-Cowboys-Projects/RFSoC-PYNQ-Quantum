"""Pulse compiler — translates gate operations to pulse instructions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .backends.base import PulseInstruction, ReadoutInstruction
from .gates import GateOp, MeasureOp


@dataclass
class QubitCalibration:
    """Calibration data for a single qubit."""

    frequency: float = 5.0e9  # Hz
    anharmonicity: float = -300e6  # Hz
    pi_amplitude: float = 0.5  # 0.0–1.0
    pi_duration: float = 40e-9  # seconds
    readout_frequency: float = 7.0e9  # Hz
    readout_duration: float = 1e-6  # seconds
    readout_channel: int = 0
    drive_channel: int = 0
    envelope: str = "gaussian"
    envelope_params: dict[str, Any] = field(default_factory=lambda: {"sigma": 10e-9})


class PulseCompiler:
    """Compiles gate operations into pulse instructions.

    Maps abstract gate operations to calibrated pulse waveforms for
    a specific backend and qubit configuration.
    """

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits
        self._calibrations: dict[int, QubitCalibration] = {}
        for i in range(num_qubits):
            cal = QubitCalibration(
                frequency=5.0e9 + i * 100e6,
                drive_channel=i,
                readout_channel=i,
            )
            self._calibrations[i] = cal

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def set_calibration(self, qubit: int, calibration: QubitCalibration) -> None:
        """Set calibration data for a qubit."""
        if qubit < 0 or qubit >= self._num_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self._num_qubits})")
        self._calibrations[qubit] = calibration

    def get_calibration(self, qubit: int) -> QubitCalibration:
        """Get calibration data for a qubit."""
        if qubit not in self._calibrations:
            raise KeyError(f"No calibration for qubit {qubit}")
        return self._calibrations[qubit]

    def compile(
        self, program: list[GateOp | MeasureOp]
    ) -> tuple[list[PulseInstruction], list[ReadoutInstruction]]:
        """Compile a sequence of gate/measure ops into pulse instructions."""
        pulses: list[PulseInstruction] = []
        readouts: list[ReadoutInstruction] = []

        for op in program:
            if isinstance(op, MeasureOp):
                readouts.extend(self._compile_measure(op))
            elif isinstance(op, GateOp):
                pulses.extend(self._compile_gate(op))
            else:
                raise TypeError(f"Unknown operation type: {type(op)}")

        return pulses, readouts

    def _compile_gate(self, op: GateOp) -> list[PulseInstruction]:
        """Compile a single gate operation into pulses."""
        name = op.gate.name
        if name in ("X", "Y", "Z", "H", "S", "T", "I"):
            return self._compile_single_qubit(op)
        if name in ("RX", "RY", "RZ", "P"):
            return self._compile_rotation(op)
        if name in ("CNOT", "CZ"):
            return self._compile_two_qubit(op)
        if name == "SWAP":
            return self._compile_swap(op)
        if name == "Toffoli":
            return self._compile_toffoli(op)
        raise ValueError(f"Unsupported gate: {name}")

    def _compile_single_qubit(self, op: GateOp) -> list[PulseInstruction]:
        """Compile a fixed single-qubit gate."""
        qubit = op.qubits[0]
        cal = self._calibrations[qubit]

        phase_map = {"X": 0.0, "Y": np.pi / 2, "Z": 0.0, "H": 0.0, "I": 0.0}
        amp_map = {"X": 1.0, "Y": 1.0, "Z": 0.0, "H": 1.0, "I": 0.0, "S": 0.0, "T": 0.0}

        name = op.gate.name
        if name in ("Z", "S", "T", "I"):
            # Virtual-Z: zero-duration phase update
            return [
                PulseInstruction(
                    channel=cal.drive_channel,
                    frequency=cal.frequency,
                    phase=_vz_phase(name),
                    amplitude=0.0,
                    duration=0.0,
                    envelope="square",
                )
            ]

        return [
            PulseInstruction(
                channel=cal.drive_channel,
                frequency=cal.frequency,
                phase=phase_map.get(name, 0.0),
                amplitude=amp_map.get(name, 1.0) * cal.pi_amplitude,
                duration=cal.pi_duration,
                envelope=cal.envelope,
                envelope_params=dict(cal.envelope_params),
            )
        ]

    def _compile_rotation(self, op: GateOp) -> list[PulseInstruction]:
        """Compile a parameterized rotation gate."""
        qubit = op.qubits[0]
        cal = self._calibrations[qubit]
        theta = op.gate.params[0]
        frac = abs(theta) / np.pi

        name = op.gate.name
        if name in ("RZ", "P"):
            return [
                PulseInstruction(
                    channel=cal.drive_channel,
                    frequency=cal.frequency,
                    phase=theta,
                    amplitude=0.0,
                    duration=0.0,
                    envelope="square",
                )
            ]

        phase = 0.0 if name == "RX" else np.pi / 2
        return [
            PulseInstruction(
                channel=cal.drive_channel,
                frequency=cal.frequency,
                phase=phase,
                amplitude=frac * cal.pi_amplitude,
                duration=cal.pi_duration,
                envelope=cal.envelope,
                envelope_params=dict(cal.envelope_params),
            )
        ]

    def _compile_two_qubit(self, op: GateOp) -> list[PulseInstruction]:
        """Compile CNOT or CZ using cross-resonance pulses."""
        control, target = op.qubits
        cal_c = self._calibrations[control]
        cal_t = self._calibrations[target]

        cr_duration = cal_c.pi_duration * 4  # Cross-resonance is slower
        pulses = [
            # Cross-resonance drive on target at control frequency
            PulseInstruction(
                channel=cal_t.drive_channel,
                frequency=cal_c.frequency,
                phase=0.0,
                amplitude=cal_c.pi_amplitude * 0.5,
                duration=cr_duration,
                envelope="flat_top",
                envelope_params={"rise_time": 10e-9},
            ),
        ]
        if op.gate.name == "CNOT":
            # Add local rotations for CNOT decomposition
            pulses.append(
                PulseInstruction(
                    channel=cal_t.drive_channel,
                    frequency=cal_t.frequency,
                    phase=np.pi / 2,
                    amplitude=cal_t.pi_amplitude,
                    duration=cal_t.pi_duration,
                    envelope=cal_t.envelope,
                    envelope_params=dict(cal_t.envelope_params),
                )
            )
        return pulses

    def _compile_swap(self, op: GateOp) -> list[PulseInstruction]:
        """Compile SWAP as three CNOTs."""
        q0, q1 = op.qubits
        from .gates import CNOT_GATE
        from .gates import GateOp as GO

        pulses: list[PulseInstruction] = []
        for ctrl, tgt in [(q0, q1), (q1, q0), (q0, q1)]:
            pulses.extend(self._compile_two_qubit(GO(gate=CNOT_GATE, qubits=(ctrl, tgt))))
        return pulses

    def _compile_toffoli(self, op: GateOp) -> list[PulseInstruction]:
        """Compile Toffoli as a sequence of single- and two-qubit gates."""
        q0, q1, q2 = op.qubits
        from .gates import CNOT_GATE, H_GATE, T_GATE
        from .gates import GateOp as GO

        pulses: list[PulseInstruction] = []
        # Simplified Toffoli decomposition
        pulses.extend(self._compile_single_qubit(GO(gate=H_GATE, qubits=(q2,))))
        pulses.extend(self._compile_two_qubit(GO(gate=CNOT_GATE, qubits=(q1, q2))))
        pulses.extend(self._compile_single_qubit(GO(gate=T_GATE, qubits=(q2,))))
        pulses.extend(self._compile_two_qubit(GO(gate=CNOT_GATE, qubits=(q0, q2))))
        pulses.extend(self._compile_single_qubit(GO(gate=T_GATE, qubits=(q2,))))
        pulses.extend(self._compile_two_qubit(GO(gate=CNOT_GATE, qubits=(q1, q2))))
        pulses.extend(self._compile_single_qubit(GO(gate=T_GATE, qubits=(q2,))))
        pulses.extend(self._compile_two_qubit(GO(gate=CNOT_GATE, qubits=(q0, q2))))
        pulses.extend(self._compile_single_qubit(GO(gate=H_GATE, qubits=(q2,))))
        return pulses

    def _compile_measure(self, op: MeasureOp) -> list[ReadoutInstruction]:
        """Compile measurement operations."""
        readouts: list[ReadoutInstruction] = []
        for qubit in op.qubits:
            cal = self._calibrations[qubit]
            readouts.append(
                ReadoutInstruction(
                    channel=cal.readout_channel,
                    frequency=cal.readout_frequency,
                    duration=cal.readout_duration,
                    qubits=[qubit],
                )
            )
        return readouts


def _vz_phase(gate_name: str) -> float:
    """Return virtual-Z phase for phase-only gates."""
    return {"Z": np.pi, "S": np.pi / 2, "T": np.pi / 4, "I": 0.0}[gate_name]
