"""QuantumCircuit — Qiskit/Cirq-style circuit builder."""

from __future__ import annotations

from .backends.base import ExecutionResult
from .backends.simulation import SimulationBackend
from .compiler import PulseCompiler
from .gates import (
    CNOT_GATE,
    CZ_GATE,
    H_GATE,
    S_GATE,
    SWAP_GATE,
    T_GATE,
    TOFFOLI_GATE,
    X_GATE,
    Y_GATE,
    Z_GATE,
    GateOp,
    MeasureOp,
    phase_gate,
    rx_gate,
    ry_gate,
    rz_gate,
)
from .overlay import QuantumOverlay


class QuantumCircuit:
    """Declarative quantum circuit builder.

    Provides a fluent API for building circuits that can be executed
    on any backend via an overlay, or converted to Qiskit/Cirq.

    Args:
        num_qubits: Number of qubits in the circuit.
        name: Optional circuit name.
    """

    def __init__(self, num_qubits: int, name: str = "circuit") -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self._num_qubits = num_qubits
        self._name = name
        self._ops: list[GateOp | MeasureOp] = []

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def name(self) -> str:
        return self._name

    @property
    def ops(self) -> list[GateOp | MeasureOp]:
        """Operations in the circuit (read-only copy)."""
        return list(self._ops)

    @property
    def depth(self) -> int:
        """Circuit depth (number of gate layers)."""
        if not self._ops:
            return 0
        # Simple depth: count gate ops
        return sum(1 for op in self._ops if isinstance(op, GateOp))

    @property
    def num_gates(self) -> int:
        """Total number of gate operations."""
        return sum(1 for op in self._ops if isinstance(op, GateOp))

    # --- Single-qubit gates ---

    def i(self, qubit: int) -> QuantumCircuit:
        """Identity gate."""
        from .gates import I_GATE

        self._validate_qubit(qubit)
        self._ops.append(GateOp(gate=I_GATE, qubits=(qubit,)))
        return self

    def x(self, qubit: int) -> QuantumCircuit:
        """Pauli-X gate."""
        self._validate_qubit(qubit)
        self._ops.append(GateOp(gate=X_GATE, qubits=(qubit,)))
        return self

    def y(self, qubit: int) -> QuantumCircuit:
        """Pauli-Y gate."""
        self._validate_qubit(qubit)
        self._ops.append(GateOp(gate=Y_GATE, qubits=(qubit,)))
        return self

    def z(self, qubit: int) -> QuantumCircuit:
        """Pauli-Z gate."""
        self._validate_qubit(qubit)
        self._ops.append(GateOp(gate=Z_GATE, qubits=(qubit,)))
        return self

    def h(self, qubit: int) -> QuantumCircuit:
        """Hadamard gate."""
        self._validate_qubit(qubit)
        self._ops.append(GateOp(gate=H_GATE, qubits=(qubit,)))
        return self

    def s(self, qubit: int) -> QuantumCircuit:
        """S (phase pi/2) gate."""
        self._validate_qubit(qubit)
        self._ops.append(GateOp(gate=S_GATE, qubits=(qubit,)))
        return self

    def t(self, qubit: int) -> QuantumCircuit:
        """T (phase pi/4) gate."""
        self._validate_qubit(qubit)
        self._ops.append(GateOp(gate=T_GATE, qubits=(qubit,)))
        return self

    def rx(self, qubit: int, theta: float) -> QuantumCircuit:
        """Rotation around X axis."""
        self._validate_qubit(qubit)
        self._ops.append(GateOp(gate=rx_gate(theta), qubits=(qubit,)))
        return self

    def ry(self, qubit: int, theta: float) -> QuantumCircuit:
        """Rotation around Y axis."""
        self._validate_qubit(qubit)
        self._ops.append(GateOp(gate=ry_gate(theta), qubits=(qubit,)))
        return self

    def rz(self, qubit: int, theta: float) -> QuantumCircuit:
        """Rotation around Z axis."""
        self._validate_qubit(qubit)
        self._ops.append(GateOp(gate=rz_gate(theta), qubits=(qubit,)))
        return self

    def p(self, qubit: int, theta: float) -> QuantumCircuit:
        """Phase gate P(theta)."""
        self._validate_qubit(qubit)
        self._ops.append(GateOp(gate=phase_gate(theta), qubits=(qubit,)))
        return self

    # --- Two-qubit gates ---

    def cnot(self, control: int, target: int) -> QuantumCircuit:
        """CNOT (CX) gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self._ops.append(GateOp(gate=CNOT_GATE, qubits=(control, target)))
        return self

    def cx(self, control: int, target: int) -> QuantumCircuit:
        """Alias for cnot."""
        return self.cnot(control, target)

    def cz(self, control: int, target: int) -> QuantumCircuit:
        """Controlled-Z gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self._ops.append(GateOp(gate=CZ_GATE, qubits=(control, target)))
        return self

    def swap(self, qubit1: int, qubit2: int) -> QuantumCircuit:
        """SWAP gate."""
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self._ops.append(GateOp(gate=SWAP_GATE, qubits=(qubit1, qubit2)))
        return self

    # --- Three-qubit gates ---

    def toffoli(self, c1: int, c2: int, target: int) -> QuantumCircuit:
        """Toffoli (CCX) gate."""
        for q in (c1, c2, target):
            self._validate_qubit(q)
        self._ops.append(GateOp(gate=TOFFOLI_GATE, qubits=(c1, c2, target)))
        return self

    def ccx(self, c1: int, c2: int, target: int) -> QuantumCircuit:
        """Alias for toffoli."""
        return self.toffoli(c1, c2, target)

    # --- Measurement ---

    def measure(self, qubit: int) -> QuantumCircuit:
        """Measure a single qubit."""
        self._validate_qubit(qubit)
        self._ops.append(MeasureOp(qubits=(qubit,)))
        return self

    def measure_all(self) -> QuantumCircuit:
        """Measure all qubits."""
        self._ops.append(MeasureOp(qubits=tuple(range(self._num_qubits))))
        return self

    # --- Execution ---

    def run(self, overlay: QuantumOverlay, shots: int = 1000) -> ExecutionResult:
        """Execute this circuit on the given overlay.

        Args:
            overlay: QuantumOverlay with an active backend.
            shots: Number of measurement samples.

        Returns:
            ExecutionResult with counts.
        """
        gate_ops = [op for op in self._ops if isinstance(op, GateOp)]
        measure_qubits: list[int] = []
        for op in self._ops:
            if isinstance(op, MeasureOp):
                measure_qubits.extend(op.qubits)
        measure_qubits = sorted(set(measure_qubits))

        if not measure_qubits:
            raise RuntimeError("No measurements in circuit. Call measure() first.")

        backend = overlay.backend

        if isinstance(backend, SimulationBackend):
            return backend.execute_circuit(self._num_qubits, gate_ops, measure_qubits, shots)

        compiler = PulseCompiler(self._num_qubits)
        pulses, readouts = compiler.compile(self._ops)
        return backend.execute(pulses, readouts, shots)

    # --- Utility ---

    def _validate_qubit(self, qubit: int) -> None:
        if qubit < 0 or qubit >= self._num_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self._num_qubits})")

    def __repr__(self) -> str:
        return (
            f"QuantumCircuit(num_qubits={self._num_qubits}, "
            f"num_gates={self.num_gates}, name='{self._name}')"
        )
