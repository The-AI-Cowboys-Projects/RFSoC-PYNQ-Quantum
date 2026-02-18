"""Tests for the PulseCompiler."""

import numpy as np
import pytest

from pynq_quantum.compiler import PulseCompiler, QubitCalibration
from pynq_quantum.gates import (
    CNOT_GATE,
    H_GATE,
    S_GATE,
    SWAP_GATE,
    TOFFOLI_GATE,
    X_GATE,
    Z_GATE,
    GateOp,
    MeasureOp,
    rx_gate,
    rz_gate,
)


class TestPulseCompiler:
    def test_init(self):
        pc = PulseCompiler(num_qubits=3)
        assert pc.num_qubits == 3

    def test_default_calibrations(self):
        pc = PulseCompiler(num_qubits=2)
        cal0 = pc.get_calibration(0)
        cal1 = pc.get_calibration(1)
        assert cal0.frequency != cal1.frequency  # Different freqs
        assert cal0.drive_channel == 0
        assert cal1.drive_channel == 1

    def test_set_calibration(self):
        pc = PulseCompiler(num_qubits=2)
        custom = QubitCalibration(frequency=4.5e9, pi_amplitude=0.8)
        pc.set_calibration(0, custom)
        assert pc.get_calibration(0).frequency == 4.5e9

    def test_set_calibration_out_of_range(self):
        pc = PulseCompiler(num_qubits=2)
        with pytest.raises(ValueError, match="out of range"):
            pc.set_calibration(5, QubitCalibration())

    def test_get_calibration_missing(self):
        pc = PulseCompiler(num_qubits=1)
        with pytest.raises(KeyError):
            pc.get_calibration(5)


class TestGateCompilation:
    def setup_method(self):
        self.compiler = PulseCompiler(num_qubits=4)

    def test_compile_x_gate(self):
        program = [GateOp(gate=X_GATE, qubits=(0,))]
        pulses, readouts = self.compiler.compile(program)
        assert len(pulses) == 1
        assert len(readouts) == 0
        assert pulses[0].channel == 0
        assert pulses[0].amplitude > 0

    def test_compile_z_gate_virtual(self):
        """Z gate should compile to virtual-Z (zero duration)."""
        program = [GateOp(gate=Z_GATE, qubits=(0,))]
        pulses, _ = self.compiler.compile(program)
        assert len(pulses) == 1
        assert pulses[0].duration == 0.0
        assert pulses[0].amplitude == 0.0
        assert pulses[0].phase == np.pi

    def test_compile_s_gate_virtual(self):
        program = [GateOp(gate=S_GATE, qubits=(0,))]
        pulses, _ = self.compiler.compile(program)
        assert pulses[0].phase == np.pi / 2

    def test_compile_h_gate(self):
        program = [GateOp(gate=H_GATE, qubits=(1,))]
        pulses, _ = self.compiler.compile(program)
        assert len(pulses) == 1
        assert pulses[0].channel == 1

    def test_compile_rx_rotation(self):
        program = [GateOp(gate=rx_gate(np.pi / 2), qubits=(0,))]
        pulses, _ = self.compiler.compile(program)
        assert len(pulses) == 1
        assert pulses[0].amplitude > 0

    def test_compile_rz_virtual(self):
        program = [GateOp(gate=rz_gate(np.pi / 4), qubits=(0,))]
        pulses, _ = self.compiler.compile(program)
        assert pulses[0].duration == 0.0
        assert pulses[0].phase == np.pi / 4

    def test_compile_cnot(self):
        program = [GateOp(gate=CNOT_GATE, qubits=(0, 1))]
        pulses, _ = self.compiler.compile(program)
        assert len(pulses) >= 2  # Cross-resonance + local rotation

    def test_compile_swap(self):
        program = [GateOp(gate=SWAP_GATE, qubits=(0, 1))]
        pulses, _ = self.compiler.compile(program)
        assert len(pulses) >= 3  # Three CNOTs worth

    def test_compile_toffoli(self):
        program = [GateOp(gate=TOFFOLI_GATE, qubits=(0, 1, 2))]
        pulses, _ = self.compiler.compile(program)
        assert len(pulses) > 0

    def test_compile_measure(self):
        program = [MeasureOp(qubits=(0, 1))]
        pulses, readouts = self.compiler.compile(program)
        assert len(pulses) == 0
        assert len(readouts) == 2
        assert readouts[0].qubits == [0]
        assert readouts[1].qubits == [1]

    def test_compile_full_program(self):
        program = [
            GateOp(gate=H_GATE, qubits=(0,)),
            GateOp(gate=CNOT_GATE, qubits=(0, 1)),
            MeasureOp(qubits=(0, 1)),
        ]
        pulses, readouts = self.compiler.compile(program)
        assert len(pulses) >= 2
        assert len(readouts) == 2

    def test_unknown_gate_raises(self):
        from pynq_quantum.gates import GateDefinition

        weird = GateDefinition(name="WEIRD", num_qubits=1, matrix=np.eye(2, dtype=complex))
        program = [GateOp(gate=weird, qubits=(0,))]
        with pytest.raises(ValueError, match="Unsupported gate"):
            self.compiler.compile(program)

    def test_unknown_op_type_raises(self):
        with pytest.raises(TypeError, match="Unknown operation type"):
            self.compiler.compile(["not_an_op"])  # type: ignore
