"""Cirq Sampler implementation for RFSoC hardware."""

from __future__ import annotations

from typing import Any

import numpy as np

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


class RFSoCSampler:
    """Cirq-compatible sampler for running circuits on RFSoC hardware.

    Implements a sampling interface compatible with Cirq's Sampler.
    Uses the pynq-quantum simulation backend by default.

    Usage::

        sampler = RFSoCSampler()
        result = sampler.run(cirq_circuit, repetitions=1000)
        print(result.measurements)

    Args:
        overlay: Optional QuantumOverlay. If None, creates one with simulation.
    """

    def __init__(self, overlay: QuantumOverlay | None = None) -> None:
        if overlay is None:
            overlay = QuantumOverlay(backend="simulation")
        self._overlay = overlay

    def run(
        self,
        circuit: Any,
        repetitions: int = 1000,
        **kwargs: Any,
    ) -> RFSoCTrialResult:
        """Sample from a Cirq circuit.

        Args:
            circuit: A cirq.Circuit object.
            repetitions: Number of samples (shots).

        Returns:
            An RFSoCTrialResult with measurement arrays.
        """
        gate_ops, measure_keys = _convert_cirq_circuit(circuit)
        num_qubits = _count_qubits(circuit)
        all_measured = sorted({q for _, qubits in measure_keys for q in qubits})

        backend = self._overlay.backend
        if isinstance(backend, SimulationBackend):
            exec_result = backend.execute_circuit(num_qubits, gate_ops, all_measured, repetitions)
        else:
            compiler = PulseCompiler(num_qubits)
            ops: list[GateOp | MeasureOp] = list(gate_ops)
            if all_measured:
                ops.append(MeasureOp(qubits=tuple(all_measured)))
            pulses, readouts = compiler.compile(ops)
            exec_result = backend.execute(pulses, readouts, repetitions)

        measurements = _counts_to_measurements(
            exec_result.counts, measure_keys, all_measured, repetitions
        )
        return RFSoCTrialResult(measurements=measurements, repetitions=repetitions)

    def close(self) -> None:
        self._overlay.close()


class RFSoCTrialResult:
    """Cirq-compatible trial result."""

    def __init__(
        self,
        measurements: dict[str, np.ndarray],
        repetitions: int,
    ) -> None:
        self._measurements = measurements
        self._repetitions = repetitions

    @property
    def measurements(self) -> dict[str, np.ndarray]:
        """Measurement results keyed by measurement key.

        Each value is a 2D numpy array of shape (repetitions, num_qubits_in_key).
        """
        return dict(self._measurements)

    @property
    def repetitions(self) -> int:
        return self._repetitions

    def histogram(self, key: str) -> dict[int, int]:
        """Get a histogram of measurement results for a key.

        Returns dict mapping integer values to counts.
        """
        if key not in self._measurements:
            raise KeyError(f"No measurement key '{key}'")
        data = self._measurements[key]
        counts: dict[int, int] = {}
        for row in data:
            val = int("".join(str(b) for b in row), 2)
            counts[val] = counts.get(val, 0) + 1
        return counts


# --- Cirq circuit conversion ---


def _convert_cirq_circuit(
    circuit: Any,
) -> tuple[list[GateOp], list[tuple[str, list[int]]]]:
    """Convert a Cirq circuit to pynq-quantum gate ops.

    Returns:
        Tuple of (gate_ops, measure_keys) where measure_keys is a list of
        (key_name, [qubit_indices]).
    """
    gate_ops: list[GateOp] = []
    measure_keys: list[tuple[str, list[int]]] = []

    # Build qubit-to-index mapping
    all_qubits = sorted(circuit.all_qubits())
    qubit_map = {q: i for i, q in enumerate(all_qubits)}

    for moment in circuit:
        for op in moment.operations:
            qubits = tuple(qubit_map[q] for q in op.qubits)
            gate = op.gate

            gate_type = type(gate).__name__

            # Check for measurement
            if hasattr(gate, "key") or gate_type == "MeasurementGate":
                key = getattr(gate, "key", "result")
                if hasattr(key, "name"):
                    key = key.name
                measure_keys.append((str(key), list(qubits)))
                continue

            # Map Cirq gates to pynq-quantum gates
            mapped = _map_cirq_gate(gate, gate_type, qubits)
            if mapped is not None:
                gate_ops.append(mapped)

    return gate_ops, measure_keys


def _map_cirq_gate(gate: Any, gate_type: str, qubits: tuple[int, ...]) -> GateOp | None:
    """Map a single Cirq gate to a GateOp."""
    simple_map: dict[str, Any] = {
        "XPowGate": None,  # Handled specially
        "YPowGate": None,
        "ZPowGate": None,
        "HPowGate": None,
        "_PauliX": X_GATE,
        "_PauliY": Y_GATE,
        "_PauliZ": Z_GATE,
        "CNotPowGate": CNOT_GATE,
        "CZPowGate": CZ_GATE,
        "SwapPowGate": SWAP_GATE,
    }

    # Direct gate mapping
    if gate_type in simple_map and simple_map[gate_type] is not None:
        return GateOp(gate=simple_map[gate_type], qubits=qubits)

    # Power gates (Cirq uses exponents)
    exponent = getattr(gate, "exponent", None) if hasattr(gate, "exponent") else None

    if gate_type == "XPowGate":
        if exponent == 1:
            return GateOp(gate=X_GATE, qubits=qubits)
        theta = float(exponent) * np.pi if exponent is not None else np.pi
        return GateOp(gate=rx_gate(theta), qubits=qubits)

    if gate_type == "YPowGate":
        if exponent == 1:
            return GateOp(gate=Y_GATE, qubits=qubits)
        theta = float(exponent) * np.pi if exponent is not None else np.pi
        return GateOp(gate=ry_gate(theta), qubits=qubits)

    if gate_type == "ZPowGate":
        if exponent == 1:
            return GateOp(gate=Z_GATE, qubits=qubits)
        if exponent == 0.5:
            return GateOp(gate=S_GATE, qubits=qubits)
        if exponent == 0.25:
            return GateOp(gate=T_GATE, qubits=qubits)
        theta = float(exponent) * np.pi if exponent is not None else np.pi
        return GateOp(gate=rz_gate(theta), qubits=qubits)

    if gate_type == "HPowGate":
        if exponent == 1:
            return GateOp(gate=H_GATE, qubits=qubits)
        return GateOp(gate=H_GATE, qubits=qubits)

    # CNOT/CZ with exponent != 1 — approximate
    if gate_type == "CNotPowGate" and exponent == 1:
        return GateOp(gate=CNOT_GATE, qubits=qubits)
    if gate_type == "CZPowGate" and exponent == 1:
        return GateOp(gate=CZ_GATE, qubits=qubits)

    return None


def _count_qubits(circuit: Any) -> int:
    """Count qubits in a Cirq circuit."""
    return len(circuit.all_qubits())


def _counts_to_measurements(
    counts: dict[str, int],
    measure_keys: list[tuple[str, list[int]]],
    all_measured: list[int],
    repetitions: int,
) -> dict[str, np.ndarray]:
    """Convert count dict to Cirq-style measurement arrays."""
    # Expand counts to individual shots
    shots: list[str] = []
    for bitstring, count in counts.items():
        shots.extend([bitstring] * count)

    # Pad or trim to match repetitions
    while len(shots) < repetitions:
        shots.append(shots[-1] if shots else "0" * len(all_measured))
    shots = shots[:repetitions]

    qubit_to_pos = {q: i for i, q in enumerate(all_measured)}

    measurements: dict[str, np.ndarray] = {}
    for key, qubits in measure_keys:
        data = np.zeros((repetitions, len(qubits)), dtype=np.int8)
        for shot_idx, bitstring in enumerate(shots):
            for col, q in enumerate(qubits):
                pos = qubit_to_pos.get(q, 0)
                if pos < len(bitstring):
                    data[shot_idx, col] = int(bitstring[pos])
        measurements[key] = data

    return measurements
