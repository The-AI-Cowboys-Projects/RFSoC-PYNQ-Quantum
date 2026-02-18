"""Tests for standard gate definitions."""

import numpy as np
import pytest

from pynq_quantum.gates import (
    CNOT_GATE,
    CZ_GATE,
    GATE_TABLE,
    H_GATE,
    I_GATE,
    S_GATE,
    SWAP_GATE,
    T_GATE,
    TOFFOLI_GATE,
    X_GATE,
    Y_GATE,
    Z_GATE,
    GateDefinition,
    GateOp,
    MeasureOp,
    phase_gate,
    rx_gate,
    ry_gate,
    rz_gate,
)


class TestGateMatrices:
    """Verify gate matrices are unitary and correct."""

    @pytest.mark.parametrize(
        "gate",
        [
            I_GATE,
            X_GATE,
            Y_GATE,
            Z_GATE,
            H_GATE,
            S_GATE,
            T_GATE,
        ],
    )
    def test_single_qubit_unitary(self, gate):
        """Single-qubit gates should be unitary: U†U = I."""
        product = gate.matrix.conj().T @ gate.matrix
        np.testing.assert_allclose(product, np.eye(2), atol=1e-12)

    @pytest.mark.parametrize("gate", [CNOT_GATE, CZ_GATE, SWAP_GATE])
    def test_two_qubit_unitary(self, gate):
        product = gate.matrix.conj().T @ gate.matrix
        np.testing.assert_allclose(product, np.eye(4), atol=1e-12)

    def test_toffoli_unitary(self):
        product = TOFFOLI_GATE.matrix.conj().T @ TOFFOLI_GATE.matrix
        np.testing.assert_allclose(product, np.eye(8), atol=1e-12)

    def test_x_gate_flips(self):
        """X|0⟩ = |1⟩."""
        state = np.array([1, 0], dtype=complex)
        result = X_GATE.matrix @ state
        np.testing.assert_allclose(result, [0, 1])

    def test_h_gate_superposition(self):
        """H|0⟩ = (|0⟩ + |1⟩)/√2."""
        state = np.array([1, 0], dtype=complex)
        result = H_GATE.matrix @ state
        expected = np.array([1, 1]) / np.sqrt(2)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_cnot_entangles(self):
        """CNOT|10⟩ = |11⟩."""
        state = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
        result = CNOT_GATE.matrix @ state
        np.testing.assert_allclose(result, [0, 0, 0, 1])  # |11⟩

    def test_toffoli_ccx(self):
        """Toffoli|110⟩ = |111⟩."""
        state = np.zeros(8, dtype=complex)
        state[6] = 1  # |110⟩
        result = TOFFOLI_GATE.matrix @ state
        expected = np.zeros(8, dtype=complex)
        expected[7] = 1  # |111⟩
        np.testing.assert_allclose(result, expected)


class TestParameterizedGates:
    def test_rx_pi_equals_x(self):
        rx_pi = rx_gate(np.pi)
        # RX(π) = -iX, so up to global phase
        np.testing.assert_allclose(np.abs(rx_pi.matrix), np.abs(X_GATE.matrix), atol=1e-12)

    def test_ry_pi_equals_y(self):
        ry_pi = ry_gate(np.pi)
        np.testing.assert_allclose(np.abs(ry_pi.matrix), np.abs(Y_GATE.matrix), atol=1e-12)

    def test_rz_pi_equals_z(self):
        rz_pi = rz_gate(np.pi)
        np.testing.assert_allclose(np.abs(rz_pi.matrix), np.abs(Z_GATE.matrix), atol=1e-12)

    def test_rx_zero_is_identity(self):
        rx_0 = rx_gate(0)
        np.testing.assert_allclose(rx_0.matrix, np.eye(2), atol=1e-12)

    def test_phase_gate(self):
        p = phase_gate(np.pi / 2)
        np.testing.assert_allclose(p.matrix, S_GATE.matrix, atol=1e-12)

    def test_rotation_unitary(self):
        for theta in [0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi]:
            for gate_fn in [rx_gate, ry_gate, rz_gate]:
                g = gate_fn(theta)
                product = g.matrix.conj().T @ g.matrix
                np.testing.assert_allclose(product, np.eye(2), atol=1e-12)


class TestGateOp:
    def test_valid_gate_op(self):
        op = GateOp(gate=X_GATE, qubits=(0,))
        assert op.gate.name == "X"
        assert op.qubits == (0,)

    def test_wrong_qubit_count_raises(self):
        with pytest.raises(ValueError, match="expects 1 qubits"):
            GateOp(gate=X_GATE, qubits=(0, 1))

    def test_measure_op(self):
        m = MeasureOp(qubits=(0, 1))
        assert m.qubits == (0, 1)


class TestGateTable:
    def test_all_gates_registered(self):
        expected = {"I", "X", "Y", "Z", "H", "S", "T", "X90", "CNOT", "CZ", "SWAP", "Toffoli"}
        assert set(GATE_TABLE.keys()) == expected

    def test_invalid_matrix_shape_raises(self):
        with pytest.raises(ValueError, match="needs 2x2 matrix"):
            GateDefinition(name="bad", num_qubits=1, matrix=np.eye(4))
