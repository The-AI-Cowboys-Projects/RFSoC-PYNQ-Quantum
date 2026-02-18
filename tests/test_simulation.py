"""Tests for the simulation backend."""

import numpy as np
import pytest

from pynq_quantum.backends.simulation import SimulationBackend
from pynq_quantum.gates import (
    CNOT_GATE,
    CZ_GATE,
    H_GATE,
    X_GATE,
    Z_GATE,
    GateOp,
    rx_gate,
)


class TestSimulationBackend:
    def test_properties(self, sim_backend):
        assert sim_backend.name == "simulation"
        assert sim_backend.num_channels == 8

    def test_connect_disconnect(self):
        backend = SimulationBackend()
        assert not backend._connected
        backend.connect()
        assert backend._connected
        backend.disconnect()
        assert not backend._connected

    def test_execute_requires_connection(self):
        backend = SimulationBackend()
        with pytest.raises(RuntimeError, match="not connected"):
            backend.execute([], [], shots=10)

    def test_configure_qubit(self, sim_backend):
        sim_backend.configure_qubit(0, 5e9, anharmonicity=-300e6)
        assert sim_backend._qubit_config[0]["frequency"] == 5e9

    def test_capabilities(self, sim_backend):
        caps = sim_backend.get_capabilities()
        assert caps["simulation"] is True
        assert "X" in caps["supported_gates"]
        assert caps["max_qubits"] == 8


class TestStateVectorSimulation:
    def test_initial_state(self, sim_backend):
        """|00⟩ state should give all-zeros measurement."""
        result = sim_backend.execute_circuit(
            num_qubits=2, gate_ops=[], measure_qubits=[0, 1], shots=100
        )
        assert result.counts == {"00": 100}

    def test_x_gate(self, sim_backend):
        """X|0⟩ = |1⟩."""
        ops = [GateOp(gate=X_GATE, qubits=(0,))]
        result = sim_backend.execute_circuit(
            num_qubits=1, gate_ops=ops, measure_qubits=[0], shots=100
        )
        assert result.counts == {"1": 100}

    def test_bell_state(self, sim_backend):
        """H on qubit 0, CNOT(0,1) → Bell state |00⟩+|11⟩."""
        ops = [
            GateOp(gate=H_GATE, qubits=(0,)),
            GateOp(gate=CNOT_GATE, qubits=(0, 1)),
        ]
        result = sim_backend.execute_circuit(
            num_qubits=2, gate_ops=ops, measure_qubits=[0, 1], shots=10000
        )
        # Should only get 00 and 11
        assert set(result.counts.keys()).issubset({"00", "11"})
        # Each should be roughly 50%
        assert result.counts.get("00", 0) > 4000
        assert result.counts.get("11", 0) > 4000

    def test_h_z_h_equals_x(self, sim_backend):
        """HZH|0⟩ = X|0⟩ = |1⟩."""
        ops = [
            GateOp(gate=H_GATE, qubits=(0,)),
            GateOp(gate=Z_GATE, qubits=(0,)),
            GateOp(gate=H_GATE, qubits=(0,)),
        ]
        result = sim_backend.execute_circuit(
            num_qubits=1, gate_ops=ops, measure_qubits=[0], shots=100
        )
        assert result.counts == {"1": 100}

    def test_cz_gate(self, sim_backend):
        """CZ|11⟩ = -|11⟩, but measurement doesn't see global phase."""
        ops = [
            GateOp(gate=X_GATE, qubits=(0,)),
            GateOp(gate=X_GATE, qubits=(1,)),
            GateOp(gate=CZ_GATE, qubits=(0, 1)),
        ]
        result = sim_backend.execute_circuit(
            num_qubits=2, gate_ops=ops, measure_qubits=[0, 1], shots=100
        )
        assert result.counts == {"11": 100}

    def test_rx_pi_flips(self, sim_backend):
        """RX(π)|0⟩ = -i|1⟩, measures as |1⟩."""
        ops = [GateOp(gate=rx_gate(np.pi), qubits=(0,))]
        result = sim_backend.execute_circuit(
            num_qubits=1, gate_ops=ops, measure_qubits=[0], shots=100
        )
        assert result.counts == {"1": 100}

    def test_statevector(self, sim_backend):
        """get_statevector returns correct amplitudes."""
        ops = [GateOp(gate=H_GATE, qubits=(0,))]
        sv = sim_backend.get_statevector(1, ops)
        expected = np.array([1, 1]) / np.sqrt(2)
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_result_metadata(self, sim_backend):
        ops = [GateOp(gate=X_GATE, qubits=(0,))]
        result = sim_backend.execute_circuit(
            num_qubits=1, gate_ops=ops, measure_qubits=[0], shots=10
        )
        assert result.metadata["backend"] == "simulation"
        assert result.raw_data is not None
        assert np.isclose(result.metadata["statevector_norm"], 1.0)

    def test_deterministic_with_seed(self):
        """Same seed → same results."""
        b1 = SimulationBackend(seed=123)
        b1.connect()
        b2 = SimulationBackend(seed=123)
        b2.connect()

        ops = [GateOp(gate=H_GATE, qubits=(0,))]
        r1 = b1.execute_circuit(1, ops, [0], shots=100)
        r2 = b2.execute_circuit(1, ops, [0], shots=100)
        assert r1.counts == r2.counts

    def test_partial_measurement(self, sim_backend):
        """Measure only qubit 0 of a 2-qubit system."""
        ops = [GateOp(gate=X_GATE, qubits=(1,))]  # Flip qubit 1 only
        result = sim_backend.execute_circuit(
            num_qubits=2, gate_ops=ops, measure_qubits=[0], shots=100
        )
        # Qubit 0 untouched → always 0
        assert result.counts == {"0": 100}
