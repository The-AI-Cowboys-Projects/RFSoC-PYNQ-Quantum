"""Tests for QubitController."""

import numpy as np
import pytest

from pynq_quantum.controller import QubitController
from pynq_quantum.gates import GateOp, MeasureOp


class TestQubitController:
    def test_init(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=2)
        assert qc.num_qubits == 2
        assert len(qc.program) == 0

    def test_init_invalid_qubits(self, sim_overlay):
        with pytest.raises(ValueError, match="must be >= 1"):
            QubitController(sim_overlay, num_qubits=0)

    def test_qubit_validation(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=2)
        with pytest.raises(ValueError, match="out of range"):
            qc.x(5)

    def test_single_qubit_gates(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=2)
        qc.x(0)
        qc.y(0)
        qc.z(0)
        qc.h(0)
        qc.x90(0)
        qc.rx(0, np.pi)
        qc.ry(0, np.pi)
        qc.rz(0, np.pi)
        assert len(qc.program) == 8

    def test_two_qubit_gates(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=2)
        qc.cnot(0, 1)
        qc.cz(0, 1)
        qc.swap(0, 1)
        assert len(qc.program) == 3

    def test_cnot_same_qubit_raises(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=2)
        with pytest.raises(ValueError, match="different qubits"):
            qc.cnot(0, 0)

    def test_swap_same_qubit_raises(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=2)
        with pytest.raises(ValueError, match="different qubits"):
            qc.swap(0, 0)

    def test_toffoli(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=3)
        qc.toffoli(0, 1, 2)
        assert len(qc.program) == 1

    def test_toffoli_duplicate_qubits_raises(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=3)
        with pytest.raises(ValueError, match="distinct qubits"):
            qc.toffoli(0, 0, 1)

    def test_measure_all(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=2)
        qc.measure()
        assert len(qc.program) == 1
        assert isinstance(qc.program[0], MeasureOp)
        assert qc.program[0].qubits == (0, 1)

    def test_measure_specific(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=3)
        qc.measure([0, 2])
        assert qc.program[0].qubits == (0, 2)

    def test_run_without_measure_raises(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=1)
        qc.x(0)
        with pytest.raises(RuntimeError, match="No measurements"):
            qc.run()

    def test_reset(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=1)
        qc.x(0)
        qc.measure()
        qc.reset()
        assert len(qc.program) == 0

    def test_set_qubit_frequency(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=1)
        qc.set_qubit_frequency(0, 4.5e9)
        cal = qc._compiler.get_calibration(0)
        assert cal.frequency == 4.5e9


class TestQubitControllerExecution:
    def test_x_gate_execution(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=1)
        qc.x(0)
        qc.measure([0])
        result = qc.run(shots=100)
        assert result.counts == {"1": 100}

    def test_bell_state_execution(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=2)
        qc.h(0)
        qc.cnot(0, 1)
        qc.measure([0, 1])
        result = qc.run(shots=10000)
        assert set(result.counts.keys()).issubset({"00", "11"})
        assert result.counts.get("00", 0) > 4000

    def test_identity_returns_zero(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=1)
        qc.measure([0])
        result = qc.run(shots=100)
        assert result.counts == {"0": 100}

    def test_multiple_runs_after_reset(self, sim_overlay):
        qc = QubitController(sim_overlay, num_qubits=1)

        qc.x(0)
        qc.measure([0])
        r1 = qc.run(shots=50)
        assert r1.counts == {"1": 50}

        qc.reset()
        qc.measure([0])
        r2 = qc.run(shots=50)
        assert r2.counts == {"0": 50}
