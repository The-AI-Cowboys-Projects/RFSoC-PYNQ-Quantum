"""NumPy state-vector simulation backend.

Provides a full quantum simulation using dense matrix multiplication.
No hardware required — default backend for development and CI.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import (
    AbstractBackend,
    ExecutionResult,
    PulseInstruction,
    ReadoutInstruction,
)


class SimulationBackend(AbstractBackend):
    """State-vector quantum simulator using NumPy."""

    def __init__(self, num_qubits: int = 8, seed: int | None = None) -> None:
        self._num_qubits = num_qubits
        self._connected = False
        self._qubit_config: dict[int, dict[str, Any]] = {}
        self._rng = np.random.default_rng(seed)

    # --- AbstractBackend interface ---

    @property
    def name(self) -> str:
        return "simulation"

    @property
    def num_channels(self) -> int:
        return self._num_qubits

    def connect(self, **kwargs: Any) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def configure_qubit(self, qubit_id: int, frequency: float, **params: Any) -> None:
        self._qubit_config[qubit_id] = {"frequency": frequency, **params}

    def execute(
        self,
        pulses: list[PulseInstruction],
        readouts: list[ReadoutInstruction],
        shots: int = 1000,
    ) -> ExecutionResult:
        if not self._connected:
            raise RuntimeError("Backend not connected. Call connect() first.")
        # This method handles raw pulse instructions — for the simulation
        # backend we sample uniformly since we can't reverse-engineer gates
        # from pulses. The simulation is driven by execute_circuit() instead.
        measured_qubits = sorted({q for ro in readouts for q in ro.qubits})
        n = len(measured_qubits) if measured_qubits else 1
        counts: dict[str, int] = {}
        for _ in range(shots):
            bits = "".join(str(self._rng.integers(0, 2)) for _ in range(n))
            counts[bits] = counts.get(bits, 0) + 1
        return ExecutionResult(counts=counts, metadata={"backend": "simulation"})

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "max_qubits": self._num_qubits,
            "supported_gates": [
                "I",
                "X",
                "Y",
                "Z",
                "H",
                "S",
                "T",
                "RX",
                "RY",
                "RZ",
                "P",
                "CNOT",
                "CZ",
                "SWAP",
                "Toffoli",
            ],
            "simulation": True,
            "max_shots": 1_000_000,
        }

    # --- Simulation-specific API ---

    def execute_circuit(
        self,
        num_qubits: int,
        gate_ops: list[Any],
        measure_qubits: list[int],
        shots: int = 1000,
    ) -> ExecutionResult:
        """Run a circuit defined by gate operations (statevector sim).

        Args:
            num_qubits: Total number of qubits.
            gate_ops: List of GateOp objects.
            measure_qubits: Which qubits to measure.
            shots: Number of measurement samples.

        Returns:
            ExecutionResult with counts.
        """
        state = _initial_state(num_qubits)

        for op in gate_ops:
            matrix = op.gate.matrix
            qubits = op.qubits
            state = _apply_gate(state, matrix, qubits, num_qubits)

        probs = _measurement_probs(state, measure_qubits, num_qubits)
        counts = _sample_counts(probs, measure_qubits, shots, self._rng)

        return ExecutionResult(
            counts=counts,
            raw_data=np.abs(state) ** 2,
            metadata={
                "backend": "simulation",
                "num_qubits": num_qubits,
                "statevector_norm": float(np.linalg.norm(state)),
            },
        )

    def get_statevector(
        self,
        num_qubits: int,
        gate_ops: list[Any],
    ) -> np.ndarray:
        """Return the full statevector after applying gate operations."""
        state = _initial_state(num_qubits)
        for op in gate_ops:
            state = _apply_gate(state, op.gate.matrix, op.qubits, num_qubits)
        return state


def _initial_state(num_qubits: int) -> np.ndarray:
    """Create |00...0⟩ state."""
    state = np.zeros(2**num_qubits, dtype=complex)
    state[0] = 1.0
    return state


def _apply_gate(
    state: np.ndarray,
    matrix: np.ndarray,
    qubits: tuple[int, ...],
    num_qubits: int,
) -> np.ndarray:
    """Apply a gate matrix to specified qubits in the statevector."""
    n = num_qubits
    gate_n = len(qubits)

    if gate_n == 1:
        return _apply_single_qubit_gate(state, matrix, qubits[0], n)
    elif gate_n == 2:
        return _apply_two_qubit_gate(state, matrix, qubits, n)
    else:
        return _apply_multi_qubit_gate(state, matrix, qubits, n)


def _apply_single_qubit_gate(
    state: np.ndarray, matrix: np.ndarray, qubit: int, num_qubits: int
) -> np.ndarray:
    """Efficient single-qubit gate application."""
    n = num_qubits
    new_state = np.zeros_like(state)
    step = 1 << qubit

    for i in range(2**n):
        if i & step:
            continue  # Skip — handled by partner
        j = i | step
        new_state[i] += matrix[0, 0] * state[i] + matrix[0, 1] * state[j]
        new_state[j] += matrix[1, 0] * state[i] + matrix[1, 1] * state[j]

    return new_state


def _apply_two_qubit_gate(
    state: np.ndarray,
    matrix: np.ndarray,
    qubits: tuple[int, ...],
    num_qubits: int,
) -> np.ndarray:
    """Efficient two-qubit gate application."""
    q0, q1 = qubits
    n = num_qubits
    new_state = np.zeros_like(state)

    for i in range(2**n):
        b0 = (i >> q0) & 1
        b1 = (i >> q1) & 1
        row = b0 * 2 + b1

        for cb0 in range(2):
            for cb1 in range(2):
                col = cb0 * 2 + cb1
                j = i
                # Set qubit q0 to cb0
                if cb0:
                    j |= 1 << q0
                else:
                    j &= ~(1 << q0)
                # Set qubit q1 to cb1
                if cb1:
                    j |= 1 << q1
                else:
                    j &= ~(1 << q1)
                new_state[i] += matrix[row, col] * state[j]

    return new_state


def _apply_multi_qubit_gate(
    state: np.ndarray,
    matrix: np.ndarray,
    qubits: tuple[int, ...],
    num_qubits: int,
) -> np.ndarray:
    """General multi-qubit gate application."""
    n = num_qubits
    gate_n = len(qubits)
    new_state = np.zeros_like(state)

    for i in range(2**n):
        # Extract qubit values for gate qubits
        row = 0
        for k, q in enumerate(qubits):
            row |= ((i >> q) & 1) << k

        for col in range(2**gate_n):
            j = i
            for k, q in enumerate(qubits):
                bit = (col >> k) & 1
                if bit:
                    j |= 1 << q
                else:
                    j &= ~(1 << q)
            new_state[i] += matrix[row, col] * state[j]

    return new_state


def _measurement_probs(
    state: np.ndarray, measure_qubits: list[int], num_qubits: int
) -> dict[str, float]:
    """Calculate measurement probabilities for specified qubits."""
    probs: dict[str, float] = {}
    n = num_qubits

    for i in range(2**n):
        p = abs(state[i]) ** 2
        if p < 1e-15:
            continue
        bits = ""
        for q in measure_qubits:
            bits += str((i >> q) & 1)
        probs[bits] = probs.get(bits, 0.0) + p

    return probs


def _sample_counts(
    probs: dict[str, float],
    measure_qubits: list[int],
    shots: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Sample measurement counts from probability distribution."""
    if not probs:
        n = len(measure_qubits)
        return {"0" * n: shots}

    labels = list(probs.keys())
    weights = np.array([probs[label] for label in labels])
    # Normalize
    weights = weights / weights.sum()

    indices = rng.choice(len(labels), size=shots, p=weights)
    counts: dict[str, int] = {}
    for idx in indices:
        label = labels[idx]
        counts[label] = counts.get(label, 0) + 1
    return counts
