"""Tests for QuantumCircuit builder."""

import numpy as np
import pytest

from pynq_quantum.circuit import QuantumCircuit
from pynq_quantum.gates import MeasureOp


class TestQuantumCircuit:
    def test_init(self):
        qc = QuantumCircuit(3, name="test")
        assert qc.num_qubits == 3
        assert qc.name == "test"
        assert qc.num_gates == 0
        assert qc.depth == 0

    def test_init_invalid(self):
        with pytest.raises(ValueError):
            QuantumCircuit(0)

    def test_fluent_api(self):
        """Gates return self for chaining."""
        qc = QuantumCircuit(2)
        result = qc.h(0).cnot(0, 1).measure_all()
        assert result is qc

    def test_single_qubit_gates(self):
        qc = QuantumCircuit(1)
        qc.i(0).x(0).y(0).z(0).h(0).s(0).t(0)
        qc.rx(0, np.pi).ry(0, np.pi).rz(0, np.pi).p(0, np.pi)
        assert qc.num_gates == 11

    def test_two_qubit_gates(self):
        qc = QuantumCircuit(2)
        qc.cnot(0, 1).cx(0, 1).cz(0, 1).swap(0, 1)
        assert qc.num_gates == 4

    def test_three_qubit_gates(self):
        qc = QuantumCircuit(3)
        qc.toffoli(0, 1, 2).ccx(0, 1, 2)
        assert qc.num_gates == 2

    def test_measure(self):
        qc = QuantumCircuit(2)
        qc.measure(0).measure(1)
        ops = qc.ops
        measures = [op for op in ops if isinstance(op, MeasureOp)]
        assert len(measures) == 2

    def test_measure_all(self):
        qc = QuantumCircuit(3)
        qc.measure_all()
        ops = qc.ops
        assert isinstance(ops[0], MeasureOp)
        assert ops[0].qubits == (0, 1, 2)

    def test_qubit_validation(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="out of range"):
            qc.x(5)

    def test_repr(self):
        qc = QuantumCircuit(2, name="bell")
        qc.h(0).cnot(0, 1)
        r = repr(qc)
        assert "num_qubits=2" in r
        assert "num_gates=2" in r
        assert "bell" in r


class TestQuantumCircuitExecution:
    def test_run_basic(self, sim_overlay):
        qc = QuantumCircuit(1)
        qc.x(0).measure_all()
        result = qc.run(sim_overlay, shots=100)
        assert result.counts == {"1": 100}

    def test_run_bell_state(self, sim_overlay):
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1).measure_all()
        result = qc.run(sim_overlay, shots=10000)
        assert set(result.counts.keys()).issubset({"00", "11"})

    def test_run_no_measure_raises(self, sim_overlay):
        qc = QuantumCircuit(1)
        qc.x(0)
        with pytest.raises(RuntimeError, match="No measurements"):
            qc.run(sim_overlay)

    def test_ghz_state(self, sim_overlay):
        """GHZ state: H(0), CNOT(0,1), CNOT(1,2)."""
        qc = QuantumCircuit(3)
        qc.h(0).cnot(0, 1).cnot(1, 2).measure_all()
        result = qc.run(sim_overlay, shots=10000)
        assert set(result.counts.keys()).issubset({"000", "111"})

    def test_ops_returns_copy(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        ops = qc.ops
        ops.append(None)  # type: ignore
        assert len(qc.ops) == 1  # Original unchanged
