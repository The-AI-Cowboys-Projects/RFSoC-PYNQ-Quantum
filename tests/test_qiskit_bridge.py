"""Tests for the Qiskit bridge — no qiskit dependency required."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from pynq_quantum.bridges.qiskit import (
    _QISKIT_GATE_MAP,
    RFSoCBackend,
    RFSoCJob,
    _convert_qiskit_circuit,
    _RFSoCResult,
    _RFSoCTarget,
)
from pynq_quantum.gates import (
    CNOT_GATE,
    H_GATE,
    S_GATE,
    SWAP_GATE,
    T_GATE,
    X_GATE,
)

# ---------------------------------------------------------------------------
# Mock helpers — simulate Qiskit objects without importing qiskit
# ---------------------------------------------------------------------------


class _MockBitRef:
    """Stands in for qiskit Qubit; identity is by object id."""

    def __init__(self, index: int) -> None:
        self.index = index


class _MockFindBitResult:
    """Returned by QuantumCircuit.find_bit()."""

    def __init__(self, index: int) -> None:
        self.index = index


def _make_instruction(name: str, qubit_indices: list[int], params: list[float] | None = None):
    """Build a mock Qiskit CircuitInstruction.

    Returns an object with:
        .operation.name  — gate name string
        .operation.params — list of gate parameters
        .qubits — list of qubit references
    """
    operation = SimpleNamespace(name=name, params=params or [])
    qubits = [_MockBitRef(i) for i in qubit_indices]
    return SimpleNamespace(operation=operation, qubits=qubits)


def _make_circuit(instructions, num_qubits: int = 2):
    """Build a mock Qiskit QuantumCircuit with .data, .num_qubits, .find_bit()."""
    bit_map = {}
    for instr in instructions:
        for q in instr.qubits:
            bit_map[id(q)] = q.index

    def find_bit(q):
        return _MockFindBitResult(bit_map[id(q)])

    return SimpleNamespace(
        data=instructions,
        num_qubits=num_qubits,
        find_bit=find_bit,
    )


# ---------------------------------------------------------------------------
# _RFSoCResult
# ---------------------------------------------------------------------------


class TestRFSoCResult:
    """Tests for the Qiskit-compatible result wrapper."""

    def test_get_counts_returns_copy(self):
        counts = {"00": 500, "11": 500}
        result = _RFSoCResult(counts, num_qubits=2)
        returned = result.get_counts()
        assert returned == {"00": 500, "11": 500}
        # Must be a copy, not the same dict
        returned["00"] = 999
        assert result.get_counts()["00"] == 500

    def test_get_counts_single_outcome(self):
        result = _RFSoCResult({"0": 1024}, num_qubits=1)
        assert result.get_counts() == {"0": 1024}

    def test_get_memory_expands_counts(self):
        result = _RFSoCResult({"01": 3, "10": 2}, num_qubits=2)
        mem = result.get_memory()
        assert mem.count("01") == 3
        assert mem.count("10") == 2
        assert len(mem) == 5

    def test_get_memory_empty_counts(self):
        result = _RFSoCResult({}, num_qubits=1)
        assert result.get_memory() == []

    def test_get_memory_preserves_bitstring(self):
        result = _RFSoCResult({"110": 1}, num_qubits=3)
        assert result.get_memory() == ["110"]


# ---------------------------------------------------------------------------
# RFSoCJob
# ---------------------------------------------------------------------------


class TestRFSoCJob:
    """Tests for the minimal job wrapper."""

    def _make_job(self):
        backend = MagicMock()
        result = _RFSoCResult({"0": 10}, num_qubits=1)
        return RFSoCJob(backend=backend, job_id="test-123", result=result)

    def test_job_id(self):
        job = self._make_job()
        assert job.job_id() == "test-123"

    def test_result(self):
        job = self._make_job()
        r = job.result()
        assert isinstance(r, _RFSoCResult)
        assert r.get_counts() == {"0": 10}

    def test_status_done(self):
        job = self._make_job()
        assert job.status() == "DONE"


# ---------------------------------------------------------------------------
# _RFSoCTarget
# ---------------------------------------------------------------------------


class TestRFSoCTarget:
    """Tests for the minimal transpiler target."""

    def test_num_qubits(self):
        target = _RFSoCTarget(4)
        assert target.num_qubits == 4

    def test_operation_names(self):
        target = _RFSoCTarget(8)
        expected = {"x", "y", "z", "h", "s", "t", "cx", "cz", "rx", "ry", "rz"}
        assert target.operation_names == expected

    def test_default_num_qubits(self):
        target = _RFSoCTarget(1)
        assert target.num_qubits == 1


# ---------------------------------------------------------------------------
# RFSoCBackend init and properties
# ---------------------------------------------------------------------------


class TestRFSoCBackendInit:
    """Tests for backend construction and properties."""

    def test_default_overlay(self, sim_overlay):
        backend = RFSoCBackend(overlay=sim_overlay, num_qubits=8)
        assert backend.num_qubits == 8
        assert "simulation" in backend.name

    def test_custom_num_qubits(self, sim_overlay):
        backend = RFSoCBackend(overlay=sim_overlay, num_qubits=4)
        assert backend.num_qubits == 4

    def test_name_includes_backend(self, sim_overlay):
        backend = RFSoCBackend(overlay=sim_overlay)
        assert backend.name.startswith("rfsoc_")

    def test_target_type(self, sim_overlay):
        backend = RFSoCBackend(overlay=sim_overlay)
        t = backend.target
        assert isinstance(t, _RFSoCTarget)
        assert t.num_qubits == backend.num_qubits

    def test_none_overlay_creates_simulation(self):
        """Passing overlay=None should auto-create a simulation overlay."""
        backend = RFSoCBackend(overlay=None, num_qubits=4)
        assert backend.num_qubits == 4
        assert "simulation" in backend.name
        backend.close()

    def test_close(self, sim_overlay):
        backend = RFSoCBackend(overlay=sim_overlay)
        # close() delegates to overlay.close(); should not raise
        backend.close()


# ---------------------------------------------------------------------------
# _convert_qiskit_circuit
# ---------------------------------------------------------------------------


class TestConvertQiskitCircuit:
    """Tests for Qiskit circuit to pynq-quantum conversion."""

    def test_single_x_gate(self):
        instr = _make_instruction("x", [0])
        qc = _make_circuit([instr], num_qubits=1)
        gate_ops, measure_qubits = _convert_qiskit_circuit(qc)
        assert len(gate_ops) == 1
        assert gate_ops[0].gate is X_GATE
        assert gate_ops[0].qubits == (0,)
        assert measure_qubits == []

    def test_h_gate(self):
        instr = _make_instruction("h", [0])
        qc = _make_circuit([instr], num_qubits=1)
        gate_ops, _ = _convert_qiskit_circuit(qc)
        assert gate_ops[0].gate is H_GATE

    def test_all_simple_gates(self):
        """Every gate in _QISKIT_GATE_MAP should convert correctly."""
        for name, expected_gate in _QISKIT_GATE_MAP.items():
            n_qubits = expected_gate.num_qubits
            qubit_indices = list(range(n_qubits))
            instr = _make_instruction(name, qubit_indices)
            qc = _make_circuit([instr], num_qubits=max(qubit_indices) + 1)
            gate_ops, _ = _convert_qiskit_circuit(qc)
            assert len(gate_ops) == 1, f"Failed for gate {name}"
            assert gate_ops[0].gate is expected_gate, f"Wrong gate for {name}"

    def test_cx_two_qubit(self):
        instr = _make_instruction("cx", [0, 1])
        qc = _make_circuit([instr], num_qubits=2)
        gate_ops, _ = _convert_qiskit_circuit(qc)
        assert gate_ops[0].gate is CNOT_GATE
        assert gate_ops[0].qubits == (0, 1)

    def test_rx_gate(self):
        theta = np.pi / 4
        instr = _make_instruction("rx", [0], params=[theta])
        qc = _make_circuit([instr], num_qubits=1)
        gate_ops, _ = _convert_qiskit_circuit(qc)
        assert len(gate_ops) == 1
        assert gate_ops[0].gate.name == "RX"
        assert gate_ops[0].gate.params == (theta,)

    def test_ry_gate(self):
        theta = np.pi / 3
        instr = _make_instruction("ry", [0], params=[theta])
        qc = _make_circuit([instr], num_qubits=1)
        gate_ops, _ = _convert_qiskit_circuit(qc)
        assert gate_ops[0].gate.name == "RY"
        assert gate_ops[0].gate.params == (theta,)

    def test_rz_gate(self):
        theta = np.pi / 6
        instr = _make_instruction("rz", [0], params=[theta])
        qc = _make_circuit([instr], num_qubits=1)
        gate_ops, _ = _convert_qiskit_circuit(qc)
        assert gate_ops[0].gate.name == "RZ"
        assert gate_ops[0].gate.params == (theta,)

    def test_measure_collects_qubits(self):
        h_instr = _make_instruction("h", [0])
        m0 = _make_instruction("measure", [0])
        m1 = _make_instruction("measure", [1])
        qc = _make_circuit([h_instr, m0, m1], num_qubits=2)
        gate_ops, measure_qubits = _convert_qiskit_circuit(qc)
        assert len(gate_ops) == 1  # only H
        assert measure_qubits == [0, 1]

    def test_measure_deduplicates(self):
        m0a = _make_instruction("measure", [0])
        m0b = _make_instruction("measure", [0])
        qc = _make_circuit([m0a, m0b], num_qubits=1)
        _, measure_qubits = _convert_qiskit_circuit(qc)
        assert measure_qubits == [0]  # sorted set

    def test_barrier_ignored(self):
        barrier = _make_instruction("barrier", [0, 1])
        x_instr = _make_instruction("x", [0])
        qc = _make_circuit([x_instr, barrier], num_qubits=2)
        gate_ops, _ = _convert_qiskit_circuit(qc)
        assert len(gate_ops) == 1
        assert gate_ops[0].gate is X_GATE

    def test_unsupported_gate_raises(self):
        instr = _make_instruction("ccx", [0, 1, 2])
        qc = _make_circuit([instr], num_qubits=3)
        with pytest.raises(ValueError, match="Unsupported Qiskit gate: ccx"):
            _convert_qiskit_circuit(qc)

    def test_mixed_circuit(self):
        """H(0) -> CX(0,1) -> measure(0) -> measure(1)."""
        instrs = [
            _make_instruction("h", [0]),
            _make_instruction("cx", [0, 1]),
            _make_instruction("measure", [0]),
            _make_instruction("measure", [1]),
        ]
        qc = _make_circuit(instrs, num_qubits=2)
        gate_ops, measure_qubits = _convert_qiskit_circuit(qc)
        assert len(gate_ops) == 2
        assert gate_ops[0].gate is H_GATE
        assert gate_ops[1].gate is CNOT_GATE
        assert measure_qubits == [0, 1]

    def test_empty_circuit(self):
        qc = _make_circuit([], num_qubits=1)
        gate_ops, measure_qubits = _convert_qiskit_circuit(qc)
        assert gate_ops == []
        assert measure_qubits == []

    def test_case_insensitive(self):
        """Gate names are lowercased, so 'X' should map correctly."""
        instr = _make_instruction("X", [0])
        qc = _make_circuit([instr], num_qubits=1)
        gate_ops, _ = _convert_qiskit_circuit(qc)
        assert gate_ops[0].gate is X_GATE

    def test_swap_gate(self):
        instr = _make_instruction("swap", [0, 1])
        qc = _make_circuit([instr], num_qubits=2)
        gate_ops, _ = _convert_qiskit_circuit(qc)
        assert gate_ops[0].gate is SWAP_GATE

    def test_s_gate(self):
        instr = _make_instruction("s", [0])
        qc = _make_circuit([instr], num_qubits=1)
        gate_ops, _ = _convert_qiskit_circuit(qc)
        assert gate_ops[0].gate is S_GATE

    def test_t_gate(self):
        instr = _make_instruction("t", [0])
        qc = _make_circuit([instr], num_qubits=1)
        gate_ops, _ = _convert_qiskit_circuit(qc)
        assert gate_ops[0].gate is T_GATE


# ---------------------------------------------------------------------------
# RFSoCBackend.run() integration with simulation backend
# ---------------------------------------------------------------------------


class TestRFSoCBackendRun:
    """Integration tests using the simulation backend (no real qiskit)."""

    def test_run_single_circuit(self, sim_overlay):
        backend = RFSoCBackend(overlay=sim_overlay, num_qubits=2)
        instrs = [
            _make_instruction("h", [0]),
            _make_instruction("measure", [0]),
            _make_instruction("measure", [1]),
        ]
        qc = _make_circuit(instrs, num_qubits=2)

        job = backend.run(qc, shots=100)
        assert isinstance(job, RFSoCJob)
        assert job.status() == "DONE"
        counts = job.result().get_counts()
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total == 100

    def test_run_accepts_list(self, sim_overlay):
        backend = RFSoCBackend(overlay=sim_overlay, num_qubits=1)
        instr = _make_instruction("x", [0])
        measure = _make_instruction("measure", [0])
        qc = _make_circuit([instr, measure], num_qubits=1)

        job = backend.run([qc], shots=50)
        counts = job.result().get_counts()
        assert sum(counts.values()) == 50

    def test_run_job_id_unique(self, sim_overlay):
        backend = RFSoCBackend(overlay=sim_overlay, num_qubits=1)
        instr = _make_instruction("x", [0])
        measure = _make_instruction("measure", [0])
        qc = _make_circuit([instr, measure], num_qubits=1)

        job1 = backend.run(qc, shots=10)
        job2 = backend.run(qc, shots=10)
        assert job1.job_id() != job2.job_id()

    def test_run_x_produces_1(self, sim_overlay):
        """X gate on |0> should produce |1> deterministically."""
        backend = RFSoCBackend(overlay=sim_overlay, num_qubits=1)
        instrs = [
            _make_instruction("x", [0]),
            _make_instruction("measure", [0]),
        ]
        qc = _make_circuit(instrs, num_qubits=1)
        job = backend.run(qc, shots=100)
        counts = job.result().get_counts()
        # Should be all "1"
        assert counts.get("1", 0) == 100

    def test_run_memory(self, sim_overlay):
        backend = RFSoCBackend(overlay=sim_overlay, num_qubits=1)
        instrs = [
            _make_instruction("x", [0]),
            _make_instruction("measure", [0]),
        ]
        qc = _make_circuit(instrs, num_qubits=1)
        job = backend.run(qc, shots=20)
        memory = job.result().get_memory()
        assert len(memory) == 20
        assert all(shot == "1" for shot in memory)
