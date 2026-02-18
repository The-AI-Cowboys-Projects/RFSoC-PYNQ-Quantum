"""QubitController — RFC API for gate accumulation and execution."""

from __future__ import annotations

from .backends.base import ExecutionResult
from .backends.simulation import SimulationBackend
from .compiler import PulseCompiler
from .gates import (
    CNOT_GATE,
    CZ_GATE,
    H_GATE,
    SWAP_GATE,
    TOFFOLI_GATE,
    X_GATE,
    Y_GATE,
    Z_GATE,
    GateOp,
    MeasureOp,
    rx_gate,
    ry_gate,
    rz_gate,
)
from .overlay import QuantumOverlay


class QubitController:
    """High-level quantum control interface per RFC #57.

    Accumulates gate operations and compiles them to pulse instructions
    for the backend when run() is called.

    Args:
        overlay: QuantumOverlay with an active backend.
        num_qubits: Number of qubits to control.
    """

    def __init__(self, overlay: QuantumOverlay, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self._overlay = overlay
        self._num_qubits = num_qubits
        self._compiler = PulseCompiler(num_qubits)
        self._program: list[GateOp | MeasureOp] = []
        self._measured = False

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def program(self) -> list[GateOp | MeasureOp]:
        """Current gate program (read-only copy)."""
        return list(self._program)

    def set_qubit_frequency(self, qubit: int, freq: float) -> None:
        """Set the drive frequency for a qubit."""
        self._validate_qubit(qubit)
        cal = self._compiler.get_calibration(qubit)
        cal.frequency = freq
        self._overlay.backend.configure_qubit(qubit, freq)

    # --- Single-qubit gates ---

    def x(self, qubit: int) -> None:
        """Pauli-X (NOT) gate."""
        self._validate_qubit(qubit)
        self._program.append(GateOp(gate=X_GATE, qubits=(qubit,)))

    def x90(self, qubit: int) -> None:
        """X90 (pi/2 rotation around X)."""
        self._validate_qubit(qubit)
        self._program.append(GateOp(gate=rx_gate(3.141592653589793 / 2), qubits=(qubit,)))

    def y(self, qubit: int) -> None:
        """Pauli-Y gate."""
        self._validate_qubit(qubit)
        self._program.append(GateOp(gate=Y_GATE, qubits=(qubit,)))

    def z(self, qubit: int) -> None:
        """Pauli-Z gate."""
        self._validate_qubit(qubit)
        self._program.append(GateOp(gate=Z_GATE, qubits=(qubit,)))

    def h(self, qubit: int) -> None:
        """Hadamard gate."""
        self._validate_qubit(qubit)
        self._program.append(GateOp(gate=H_GATE, qubits=(qubit,)))

    def rx(self, qubit: int, theta: float) -> None:
        """Rotation around X axis."""
        self._validate_qubit(qubit)
        self._program.append(GateOp(gate=rx_gate(theta), qubits=(qubit,)))

    def ry(self, qubit: int, theta: float) -> None:
        """Rotation around Y axis."""
        self._validate_qubit(qubit)
        self._program.append(GateOp(gate=ry_gate(theta), qubits=(qubit,)))

    def rz(self, qubit: int, theta: float) -> None:
        """Rotation around Z axis."""
        self._validate_qubit(qubit)
        self._program.append(GateOp(gate=rz_gate(theta), qubits=(qubit,)))

    # --- Two-qubit gates ---

    def cnot(self, control: int, target: int) -> None:
        """CNOT (controlled-X) gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target must be different qubits")
        self._program.append(GateOp(gate=CNOT_GATE, qubits=(control, target)))

    def cz(self, control: int, target: int) -> None:
        """Controlled-Z gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target must be different qubits")
        self._program.append(GateOp(gate=CZ_GATE, qubits=(control, target)))

    def swap(self, qubit1: int, qubit2: int) -> None:
        """SWAP gate."""
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        if qubit1 == qubit2:
            raise ValueError("SWAP requires two different qubits")
        self._program.append(GateOp(gate=SWAP_GATE, qubits=(qubit1, qubit2)))

    # --- Three-qubit gates ---

    def toffoli(self, control1: int, control2: int, target: int) -> None:
        """Toffoli (CCX) gate."""
        for q in (control1, control2, target):
            self._validate_qubit(q)
        if len({control1, control2, target}) != 3:
            raise ValueError("Toffoli requires three distinct qubits")
        self._program.append(GateOp(gate=TOFFOLI_GATE, qubits=(control1, control2, target)))

    # --- Measurement + execution ---

    def measure(self, qubits: list[int] | None = None) -> None:
        """Add measurement operations."""
        if qubits is None:
            qubits = list(range(self._num_qubits))
        for q in qubits:
            self._validate_qubit(q)
        self._program.append(MeasureOp(qubits=tuple(qubits)))
        self._measured = True

    def run(self, shots: int = 1000) -> ExecutionResult:
        """Compile and execute the accumulated program.

        If the backend is the simulation backend, uses the efficient
        statevector simulation path. Otherwise, compiles gates to
        pulse instructions and sends to hardware.
        """
        if not self._measured:
            raise RuntimeError("No measurements in program. Call measure() first.")

        # Collect gate ops and measure qubits
        gate_ops = [op for op in self._program if isinstance(op, GateOp)]
        measure_qubits: list[int] = []
        for op in self._program:
            if isinstance(op, MeasureOp):
                measure_qubits.extend(op.qubits)
        measure_qubits = sorted(set(measure_qubits))

        backend = self._overlay.backend

        # Fast path for simulation
        if isinstance(backend, SimulationBackend):
            return backend.execute_circuit(self._num_qubits, gate_ops, measure_qubits, shots)

        # Hardware path: compile to pulses
        pulses, readouts = self._compiler.compile(self._program)
        return backend.execute(pulses, readouts, shots)

    def reset(self) -> None:
        """Clear the accumulated program."""
        self._program.clear()
        self._measured = False

    def _validate_qubit(self, qubit: int) -> None:
        if qubit < 0 or qubit >= self._num_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self._num_qubits})")
