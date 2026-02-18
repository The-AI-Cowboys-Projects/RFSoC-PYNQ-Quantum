"""Qiskit BackendV2 provider for RFSoC hardware."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import numpy as np

from ..backends.base import ExecutionResult
from ..backends.simulation import SimulationBackend
from ..compiler import PulseCompiler
from ..gates import (
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
    MeasureOp,
    rx_gate,
    ry_gate,
    rz_gate,
)
from ..overlay import QuantumOverlay


class RFSoCJob:
    """Minimal job object returned by RFSoCBackend.run()."""

    def __init__(
        self,
        backend: RFSoCBackend,
        job_id: str,
        result: _RFSoCResult,
    ) -> None:
        self._backend = backend
        self._job_id = job_id
        self._result = result

    def job_id(self) -> str:
        return self._job_id

    def result(self) -> _RFSoCResult:
        return self._result

    def status(self) -> str:
        return "DONE"


class _RFSoCResult:
    """Qiskit-compatible result wrapper."""

    def __init__(self, counts: dict[str, int], num_qubits: int) -> None:
        self._counts = counts
        self._num_qubits = num_qubits

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)

    def get_memory(self) -> list[str]:
        memory = []
        for bits, count in self._counts.items():
            memory.extend([bits] * count)
        return memory


class RFSoCBackend:
    """Qiskit-compatible backend for running circuits on RFSoC hardware.

    Implements enough of the Qiskit BackendV2 interface to be usable
    with `backend.run(circuit)`. Uses the pynq-quantum simulation
    backend by default.

    Args:
        overlay: Optional QuantumOverlay. If None, creates one with simulation.
        num_qubits: Maximum number of qubits (default 8).
    """

    def __init__(
        self,
        overlay: QuantumOverlay | None = None,
        num_qubits: int = 8,
    ) -> None:
        if overlay is None:
            overlay = QuantumOverlay(backend="simulation", num_qubits=num_qubits)
        self._overlay = overlay
        self._num_qubits = num_qubits

    @property
    def name(self) -> str:
        return f"rfsoc_{self._overlay.backend_name}"

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def target(self) -> Any:
        """Return a target-like object for Qiskit transpiler."""
        return _RFSoCTarget(self._num_qubits)

    def run(
        self,
        circuits: Any,
        shots: int = 1000,
        **kwargs: Any,
    ) -> RFSoCJob:
        """Execute a Qiskit QuantumCircuit (or list) on this backend.

        Args:
            circuits: A Qiskit QuantumCircuit or list of circuits.
            shots: Number of measurement samples.

        Returns:
            An RFSoCJob with results.
        """
        if not isinstance(circuits, list):
            circuits = [circuits]

        # Process the first circuit (batch not yet supported)
        qc = circuits[0]
        gate_ops, measure_qubits = _convert_qiskit_circuit(qc)

        backend = self._overlay.backend
        if isinstance(backend, SimulationBackend):
            exec_result = backend.execute_circuit(
                qc.num_qubits, gate_ops, measure_qubits, shots
            )
        else:
            compiler = PulseCompiler(qc.num_qubits)
            ops: list[GateOp | MeasureOp] = list(gate_ops)
            if measure_qubits:
                ops.append(MeasureOp(qubits=tuple(measure_qubits)))
            pulses, readouts = compiler.compile(ops)
            exec_result = backend.execute(pulses, readouts, shots)

        job_id = str(uuid.uuid4())
        result = _RFSoCResult(exec_result.counts, qc.num_qubits)
        return RFSoCJob(self, job_id, result)

    def close(self) -> None:
        self._overlay.close()


class _RFSoCTarget:
    """Minimal target for Qiskit transpiler compatibility."""

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self.operation_names = {"x", "y", "z", "h", "s", "t", "cx", "cz", "rx", "ry", "rz"}


# --- Qiskit circuit conversion ---

_QISKIT_GATE_MAP: dict[str, Any] = {
    "x": X_GATE,
    "y": Y_GATE,
    "z": Z_GATE,
    "h": H_GATE,
    "s": S_GATE,
    "t": T_GATE,
    "cx": CNOT_GATE,
    "cz": CZ_GATE,
    "swap": SWAP_GATE,
}


def _convert_qiskit_circuit(
    qc: Any,
) -> tuple[list[GateOp], list[int]]:
    """Convert a Qiskit QuantumCircuit to pynq-quantum gate ops."""
    gate_ops: list[GateOp] = []
    measure_qubits: list[int] = []

    for instruction in qc.data:
        op = instruction.operation
        qubits = tuple(qc.find_bit(q).index for q in instruction.qubits)
        name = op.name.lower()

        if name == "measure":
            measure_qubits.extend(qubits)
            continue

        if name == "barrier":
            continue

        if name in _QISKIT_GATE_MAP:
            gate_ops.append(GateOp(gate=_QISKIT_GATE_MAP[name], qubits=qubits))
        elif name == "rx":
            theta = float(op.params[0])
            gate_ops.append(GateOp(gate=rx_gate(theta), qubits=qubits))
        elif name == "ry":
            theta = float(op.params[0])
            gate_ops.append(GateOp(gate=ry_gate(theta), qubits=qubits))
        elif name == "rz":
            theta = float(op.params[0])
            gate_ops.append(GateOp(gate=rz_gate(theta), qubits=qubits))
        else:
            raise ValueError(f"Unsupported Qiskit gate: {name}")

    return gate_ops, sorted(set(measure_qubits))
