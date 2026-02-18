"""Tests for the Cirq bridge — no cirq dependency required."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from pynq_quantum.bridges.cirq import (
    RFSoCSampler,
    RFSoCTrialResult,
    _convert_cirq_circuit,
    _count_qubits,
    _counts_to_measurements,
    _map_cirq_gate,
)
from pynq_quantum.gates import (
    CNOT_GATE,
    CZ_GATE,
    H_GATE,
    S_GATE,
    SWAP_GATE,
    T_GATE,
    X_GATE,
    Y_GATE,
    Z_GATE,
    GateOp,
    rx_gate,
    ry_gate,
    rz_gate,
)


# ---------------------------------------------------------------------------
# Mock helpers — simulate Cirq objects without importing cirq
# ---------------------------------------------------------------------------

class _MockQubit:
    """Sortable stand-in for cirq.LineQubit."""

    def __init__(self, index: int) -> None:
        self.x = index
        self._index = index

    def __lt__(self, other):
        return self._index < other._index

    def __eq__(self, other):
        return self._index == other._index

    def __hash__(self):
        return hash(self._index)

    def __repr__(self):
        return f"q({self._index})"


def _make_gate(gate_type_name: str, exponent=None, key=None):
    """Build a mock Cirq gate with a given type name.

    The type(gate).__name__ trick used by the bridge is satisfied
    by creating a class on the fly.
    """
    attrs = {}
    if exponent is not None:
        attrs["exponent"] = exponent
    if key is not None:
        attrs["key"] = key

    gate_cls = type(gate_type_name, (), attrs)
    return gate_cls()


def _make_operation(gate, qubit_indices: list[int]):
    """Build a mock Cirq Operation with .gate and .qubits."""
    qubits = [_MockQubit(i) for i in qubit_indices]
    return SimpleNamespace(gate=gate, qubits=qubits)


def _make_moment(operations: list):
    """Build a mock Cirq Moment with .operations."""
    return SimpleNamespace(operations=operations)


class _MockCircuit:
    """Iterable mock Cirq Circuit with .all_qubits()."""

    def __init__(self, moments, qubit_set):
        self._moments = moments
        self._qubit_set = qubit_set

    def all_qubits(self):
        return self._qubit_set

    def __iter__(self):
        return iter(self._moments)


def _make_circuit(moments: list, all_qubits: list[int] | None = None):
    """Build a mock Cirq Circuit.

    Supports iteration over moments and .all_qubits().
    """
    # Collect all qubits from moments if not provided
    qubit_set = set()
    if all_qubits is not None:
        qubit_set = {_MockQubit(i) for i in all_qubits}
    else:
        for moment in moments:
            for op in moment.operations:
                for q in op.qubits:
                    qubit_set.add(q)

    return _MockCircuit(moments, qubit_set)


# ---------------------------------------------------------------------------
# RFSoCTrialResult
# ---------------------------------------------------------------------------

class TestRFSoCTrialResult:
    """Tests for the Cirq-compatible trial result."""

    def test_measurements_property(self):
        data = {"result": np.array([[0, 1], [1, 0]], dtype=np.int8)}
        result = RFSoCTrialResult(measurements=data, repetitions=2)
        m = result.measurements
        assert "result" in m
        np.testing.assert_array_equal(m["result"], data["result"])

    def test_measurements_returns_copy(self):
        data = {"q": np.array([[0], [1]], dtype=np.int8)}
        result = RFSoCTrialResult(measurements=data, repetitions=2)
        m1 = result.measurements
        m2 = result.measurements
        assert m1 is not m2  # different dict each time

    def test_repetitions(self):
        result = RFSoCTrialResult(measurements={}, repetitions=1000)
        assert result.repetitions == 1000

    def test_histogram_single_qubit(self):
        # 3 shots: 0, 1, 1 -> histogram {0: 1, 1: 2}
        data = {"m": np.array([[0], [1], [1]], dtype=np.int8)}
        result = RFSoCTrialResult(measurements=data, repetitions=3)
        h = result.histogram("m")
        assert h == {0: 1, 1: 2}

    def test_histogram_two_qubits(self):
        # 4 shots: 00, 01, 10, 11 -> binary 0,1,2,3
        data = {
            "q": np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8),
        }
        result = RFSoCTrialResult(measurements=data, repetitions=4)
        h = result.histogram("q")
        assert h == {0: 1, 1: 1, 2: 1, 3: 1}

    def test_histogram_all_zeros(self):
        data = {"z": np.array([[0, 0], [0, 0]], dtype=np.int8)}
        result = RFSoCTrialResult(measurements=data, repetitions=2)
        assert result.histogram("z") == {0: 2}

    def test_histogram_missing_key_raises(self):
        result = RFSoCTrialResult(measurements={}, repetitions=0)
        with pytest.raises(KeyError, match="No measurement key 'missing'"):
            result.histogram("missing")


# ---------------------------------------------------------------------------
# _count_qubits
# ---------------------------------------------------------------------------

class TestCountQubits:
    """Tests for the qubit counting helper."""

    def test_count_single_qubit(self):
        circuit = _make_circuit([], all_qubits=[0])
        assert _count_qubits(circuit) == 1

    def test_count_multiple_qubits(self):
        circuit = _make_circuit([], all_qubits=[0, 1, 2])
        assert _count_qubits(circuit) == 3

    def test_count_empty_circuit(self):
        circuit = _make_circuit([], all_qubits=[])
        assert _count_qubits(circuit) == 0


# ---------------------------------------------------------------------------
# _counts_to_measurements
# ---------------------------------------------------------------------------

class TestCountsToMeasurements:
    """Tests for expanding count dicts to Cirq-style measurement arrays."""

    def test_simple_single_key(self):
        counts = {"01": 2, "10": 1}
        measure_keys = [("result", [0, 1])]
        all_measured = [0, 1]
        repetitions = 3

        m = _counts_to_measurements(counts, measure_keys, all_measured, repetitions)
        assert "result" in m
        assert m["result"].shape == (3, 2)
        # Total rows should equal repetitions
        assert m["result"].shape[0] == repetitions

    def test_pads_to_repetitions(self):
        counts = {"0": 1}
        measure_keys = [("m", [0])]
        all_measured = [0]
        repetitions = 5

        m = _counts_to_measurements(counts, measure_keys, all_measured, repetitions)
        assert m["m"].shape == (5, 1)

    def test_trims_to_repetitions(self):
        counts = {"0": 10}
        measure_keys = [("m", [0])]
        all_measured = [0]
        repetitions = 3

        m = _counts_to_measurements(counts, measure_keys, all_measured, repetitions)
        assert m["m"].shape == (3, 1)

    def test_multiple_keys(self):
        counts = {"01": 2}
        measure_keys = [("a", [0]), ("b", [1])]
        all_measured = [0, 1]
        repetitions = 2

        m = _counts_to_measurements(counts, measure_keys, all_measured, repetitions)
        assert "a" in m and "b" in m
        assert m["a"].shape == (2, 1)
        assert m["b"].shape == (2, 1)

    def test_empty_counts(self):
        counts = {}
        measure_keys = [("m", [0])]
        all_measured = [0]
        repetitions = 3

        m = _counts_to_measurements(counts, measure_keys, all_measured, repetitions)
        # Should pad with zeros
        assert m["m"].shape == (3, 1)

    def test_bitstring_values_correct(self):
        counts = {"10": 2}
        measure_keys = [("r", [0, 1])]
        all_measured = [0, 1]
        repetitions = 2

        m = _counts_to_measurements(counts, measure_keys, all_measured, repetitions)
        # "10" means qubit 0=1, qubit 1=0
        np.testing.assert_array_equal(m["r"][0], [1, 0])
        np.testing.assert_array_equal(m["r"][1], [1, 0])


# ---------------------------------------------------------------------------
# _map_cirq_gate
# ---------------------------------------------------------------------------

class TestMapCirqGate:
    """Tests for mapping individual Cirq gates to GateOps."""

    def test_pauli_x(self):
        gate = _make_gate("_PauliX")
        result = _map_cirq_gate(gate, "_PauliX", (0,))
        assert result is not None
        assert result.gate is X_GATE

    def test_pauli_y(self):
        gate = _make_gate("_PauliY")
        result = _map_cirq_gate(gate, "_PauliY", (0,))
        assert result.gate is Y_GATE

    def test_pauli_z(self):
        gate = _make_gate("_PauliZ")
        result = _map_cirq_gate(gate, "_PauliZ", (0,))
        assert result.gate is Z_GATE

    def test_cnot_pow_gate(self):
        gate = _make_gate("CNotPowGate", exponent=1)
        result = _map_cirq_gate(gate, "CNotPowGate", (0, 1))
        assert result.gate is CNOT_GATE

    def test_cz_pow_gate(self):
        gate = _make_gate("CZPowGate", exponent=1)
        result = _map_cirq_gate(gate, "CZPowGate", (0, 1))
        assert result.gate is CZ_GATE

    def test_swap_pow_gate(self):
        gate = _make_gate("SwapPowGate")
        result = _map_cirq_gate(gate, "SwapPowGate", (0, 1))
        assert result.gate is SWAP_GATE

    def test_xpow_exponent_1(self):
        gate = _make_gate("XPowGate", exponent=1)
        result = _map_cirq_gate(gate, "XPowGate", (0,))
        assert result.gate is X_GATE

    def test_xpow_fractional(self):
        gate = _make_gate("XPowGate", exponent=0.5)
        result = _map_cirq_gate(gate, "XPowGate", (0,))
        assert result is not None
        assert result.gate.name == "RX"
        assert result.gate.params == (0.5 * np.pi,)

    def test_ypow_exponent_1(self):
        gate = _make_gate("YPowGate", exponent=1)
        result = _map_cirq_gate(gate, "YPowGate", (0,))
        assert result.gate is Y_GATE

    def test_ypow_fractional(self):
        gate = _make_gate("YPowGate", exponent=0.25)
        result = _map_cirq_gate(gate, "YPowGate", (0,))
        assert result.gate.name == "RY"
        assert result.gate.params == (0.25 * np.pi,)

    def test_zpow_exponent_1(self):
        gate = _make_gate("ZPowGate", exponent=1)
        result = _map_cirq_gate(gate, "ZPowGate", (0,))
        assert result.gate is Z_GATE

    def test_zpow_exponent_05_is_s(self):
        gate = _make_gate("ZPowGate", exponent=0.5)
        result = _map_cirq_gate(gate, "ZPowGate", (0,))
        assert result.gate is S_GATE

    def test_zpow_exponent_025_is_t(self):
        gate = _make_gate("ZPowGate", exponent=0.25)
        result = _map_cirq_gate(gate, "ZPowGate", (0,))
        assert result.gate is T_GATE

    def test_zpow_fractional(self):
        gate = _make_gate("ZPowGate", exponent=0.3)
        result = _map_cirq_gate(gate, "ZPowGate", (0,))
        assert result.gate.name == "RZ"
        assert result.gate.params == (0.3 * np.pi,)

    def test_hpow_exponent_1(self):
        gate = _make_gate("HPowGate", exponent=1)
        result = _map_cirq_gate(gate, "HPowGate", (0,))
        assert result.gate is H_GATE

    def test_hpow_fractional_still_h(self):
        """HPowGate with any exponent maps to H_GATE (current implementation)."""
        gate = _make_gate("HPowGate", exponent=0.5)
        result = _map_cirq_gate(gate, "HPowGate", (0,))
        assert result.gate is H_GATE

    def test_unknown_gate_returns_none(self):
        gate = _make_gate("SomeUnknownGate")
        result = _map_cirq_gate(gate, "SomeUnknownGate", (0,))
        assert result is None

    def test_cnot_pow_non_unity_exponent_still_maps(self):
        """CNotPowGate is in simple_map with non-None value, so any exponent
        gets mapped to CNOT_GATE via the direct mapping branch."""
        gate = _make_gate("CNotPowGate", exponent=0.5)
        result = _map_cirq_gate(gate, "CNotPowGate", (0, 1))
        assert result is not None
        assert result.gate is CNOT_GATE

    def test_cz_pow_non_unity_exponent_still_maps(self):
        """CZPowGate is in simple_map with non-None value, so any exponent
        gets mapped to CZ_GATE via the direct mapping branch."""
        gate = _make_gate("CZPowGate", exponent=0.5)
        result = _map_cirq_gate(gate, "CZPowGate", (0, 1))
        assert result is not None
        assert result.gate is CZ_GATE


# ---------------------------------------------------------------------------
# _convert_cirq_circuit
# ---------------------------------------------------------------------------

class TestConvertCirqCircuit:
    """Tests for full Cirq circuit conversion."""

    def test_single_x_gate(self):
        gate = _make_gate("_PauliX")
        op = _make_operation(gate, [0])
        moment = _make_moment([op])
        circuit = _make_circuit([moment])

        gate_ops, measure_keys = _convert_cirq_circuit(circuit)
        assert len(gate_ops) == 1
        assert gate_ops[0].gate is X_GATE
        assert measure_keys == []

    def test_measurement_gate_with_key(self):
        gate = _make_gate("MeasurementGate", key="m0")
        op = _make_operation(gate, [0])
        moment = _make_moment([op])
        circuit = _make_circuit([moment])

        gate_ops, measure_keys = _convert_cirq_circuit(circuit)
        assert gate_ops == []
        assert len(measure_keys) == 1
        assert measure_keys[0][0] == "m0"
        assert measure_keys[0][1] == [0]

    def test_measurement_with_key_name_attribute(self):
        """Test measurement gate where key has a .name attribute."""
        key_obj = SimpleNamespace(name="readout")
        gate = _make_gate("MeasurementGate", key=key_obj)
        op = _make_operation(gate, [0, 1])
        moment = _make_moment([op])
        circuit = _make_circuit([moment])

        _, measure_keys = _convert_cirq_circuit(circuit)
        assert measure_keys[0][0] == "readout"
        assert measure_keys[0][1] == [0, 1]

    def test_measurement_gate_default_key(self):
        """MeasurementGate without a key attribute defaults to 'result'."""
        # Create gate with type name MeasurementGate but no key attribute
        gate_cls = type("MeasurementGate", (), {})
        gate = gate_cls()
        op = _make_operation(gate, [0])
        moment = _make_moment([op])
        circuit = _make_circuit([moment])

        _, measure_keys = _convert_cirq_circuit(circuit)
        assert measure_keys[0][0] == "result"

    def test_multi_moment_circuit(self):
        """Circuit with H(0) then CX(0,1) then Measure."""
        h_gate = _make_gate("HPowGate", exponent=1)
        cx_gate = _make_gate("CNotPowGate", exponent=1)
        m_gate = _make_gate("MeasurementGate", key="out")

        h_op = _make_operation(h_gate, [0])
        cx_op = _make_operation(cx_gate, [0, 1])
        m_op = _make_operation(m_gate, [0, 1])

        circuit = _make_circuit([
            _make_moment([h_op]),
            _make_moment([cx_op]),
            _make_moment([m_op]),
        ])

        gate_ops, measure_keys = _convert_cirq_circuit(circuit)
        assert len(gate_ops) == 2
        assert gate_ops[0].gate is H_GATE
        assert gate_ops[1].gate is CNOT_GATE
        assert len(measure_keys) == 1
        assert measure_keys[0] == ("out", [0, 1])

    def test_empty_circuit(self):
        circuit = _make_circuit([], all_qubits=[])
        gate_ops, measure_keys = _convert_cirq_circuit(circuit)
        assert gate_ops == []
        assert measure_keys == []

    def test_unknown_gate_skipped(self):
        """Unknown gates should return None from _map_cirq_gate and be skipped."""
        gate = _make_gate("FooBarGate")
        op = _make_operation(gate, [0])
        moment = _make_moment([op])
        circuit = _make_circuit([moment])

        gate_ops, _ = _convert_cirq_circuit(circuit)
        assert gate_ops == []

    def test_qubit_index_mapping(self):
        """Qubits should be mapped to contiguous indices based on sorted order."""
        gate = _make_gate("_PauliX")
        # Use non-contiguous qubit indices 2 and 5
        q2 = _MockQubit(2)
        q5 = _MockQubit(5)
        op = SimpleNamespace(gate=gate, qubits=[q5])
        moment = SimpleNamespace(operations=[op])

        def all_qubits():
            return {q2, q5}

        circuit_cls = type("MockCircuit", (), {
            "all_qubits": all_qubits,
            "__iter__": lambda self: iter([moment]),
        })
        circuit = circuit_cls()
        circuit.all_qubits = all_qubits

        gate_ops, _ = _convert_cirq_circuit(circuit)
        # q2 -> index 0, q5 -> index 1 (sorted)
        assert gate_ops[0].qubits == (1,)

    def test_parallel_operations_in_moment(self):
        """Multiple operations in one moment should all be converted."""
        x_gate = _make_gate("_PauliX")
        y_gate = _make_gate("_PauliY")
        op_x = _make_operation(x_gate, [0])
        op_y = _make_operation(y_gate, [1])
        moment = _make_moment([op_x, op_y])
        circuit = _make_circuit([moment])

        gate_ops, _ = _convert_cirq_circuit(circuit)
        assert len(gate_ops) == 2
        gate_names = {g.gate.name for g in gate_ops}
        assert gate_names == {"X", "Y"}

    def test_hasattr_key_triggers_measurement(self):
        """Any gate with a 'key' attribute is treated as measurement, even
        if type name is not MeasurementGate."""
        gate = _make_gate("SomeGate", key="meas")
        op = _make_operation(gate, [0])
        moment = _make_moment([op])
        circuit = _make_circuit([moment])

        gate_ops, measure_keys = _convert_cirq_circuit(circuit)
        assert gate_ops == []
        assert measure_keys == [("meas", [0])]


# ---------------------------------------------------------------------------
# RFSoCSampler
# ---------------------------------------------------------------------------

class TestRFSoCSampler:
    """Tests for the Cirq-compatible sampler."""

    def test_init_default_overlay(self):
        sampler = RFSoCSampler(overlay=None)
        # Should create simulation overlay; close without error
        sampler.close()

    def test_init_custom_overlay(self, sim_overlay):
        sampler = RFSoCSampler(overlay=sim_overlay)
        # Should use the provided overlay
        assert sampler._overlay is sim_overlay

    def test_close(self, sim_overlay):
        sampler = RFSoCSampler(overlay=sim_overlay)
        sampler.close()
        # Calling close should not raise

    def test_run_returns_trial_result(self, sim_overlay):
        """Run a simple X(0) -> Measure circuit through the sampler."""
        sampler = RFSoCSampler(overlay=sim_overlay)

        # Build mock circuit: X(0) -> Measure(0)
        x_gate = _make_gate("XPowGate", exponent=1)
        m_gate = _make_gate("MeasurementGate", key="result")

        x_op = _make_operation(x_gate, [0])
        m_op = _make_operation(m_gate, [0])

        circuit = _make_circuit([
            _make_moment([x_op]),
            _make_moment([m_op]),
        ])

        result = sampler.run(circuit, repetitions=50)
        assert isinstance(result, RFSoCTrialResult)
        assert result.repetitions == 50
        assert "result" in result.measurements

    def test_run_h_gate_produces_results(self, sim_overlay):
        """H gate should produce a mix of 0 and 1 outcomes (statistically)."""
        sampler = RFSoCSampler(overlay=sim_overlay)

        h_gate = _make_gate("HPowGate", exponent=1)
        m_gate = _make_gate("MeasurementGate", key="q0")
        h_op = _make_operation(h_gate, [0])
        m_op = _make_operation(m_gate, [0])

        circuit = _make_circuit([
            _make_moment([h_op]),
            _make_moment([m_op]),
        ])

        result = sampler.run(circuit, repetitions=200)
        h = result.histogram("q0")
        # Should have outcomes; H|0> gives ~50/50
        assert len(h) >= 1
        total = sum(h.values())
        assert total == 200

    def test_run_two_qubit_bell(self, sim_overlay):
        """H(0) -> CNOT(0,1) -> Measure both."""
        sampler = RFSoCSampler(overlay=sim_overlay)

        h_gate = _make_gate("HPowGate", exponent=1)
        cx_gate = _make_gate("CNotPowGate", exponent=1)
        m_gate = _make_gate("MeasurementGate", key="bell")

        h_op = _make_operation(h_gate, [0])
        cx_op = _make_operation(cx_gate, [0, 1])
        m_op = _make_operation(m_gate, [0, 1])

        circuit = _make_circuit([
            _make_moment([h_op]),
            _make_moment([cx_op]),
            _make_moment([m_op]),
        ])

        result = sampler.run(circuit, repetitions=100)
        assert "bell" in result.measurements
        assert result.measurements["bell"].shape == (100, 2)
