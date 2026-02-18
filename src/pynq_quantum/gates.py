"""Standard quantum gate definitions with unitary matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GateDefinition:
    """Immutable definition of a quantum gate."""

    name: str
    num_qubits: int
    matrix: np.ndarray
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        expected = 2**self.num_qubits
        if self.matrix.shape != (expected, expected):
            raise ValueError(
                f"Gate '{self.name}' needs {expected}x{expected} matrix, "
                f"got {self.matrix.shape}"
            )


@dataclass
class GateOp:
    """A gate operation applied to specific qubits."""

    gate: GateDefinition
    qubits: tuple[int, ...]
    params: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.qubits) != self.gate.num_qubits:
            raise ValueError(
                f"Gate '{self.gate.name}' expects {self.gate.num_qubits} qubits, "
                f"got {len(self.qubits)}"
            )


@dataclass
class MeasureOp:
    """Measurement on specified qubits."""

    qubits: tuple[int, ...]


# --- Single-qubit gates ---

I_GATE = GateDefinition(
    name="I", num_qubits=1, matrix=np.eye(2, dtype=complex)
)

X_GATE = GateDefinition(
    name="X",
    num_qubits=1,
    matrix=np.array([[0, 1], [1, 0]], dtype=complex),
)

Y_GATE = GateDefinition(
    name="Y",
    num_qubits=1,
    matrix=np.array([[0, -1j], [1j, 0]], dtype=complex),
)

Z_GATE = GateDefinition(
    name="Z",
    num_qubits=1,
    matrix=np.array([[1, 0], [0, -1]], dtype=complex),
)

H_GATE = GateDefinition(
    name="H",
    num_qubits=1,
    matrix=np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
)

S_GATE = GateDefinition(
    name="S",
    num_qubits=1,
    matrix=np.array([[1, 0], [0, 1j]], dtype=complex),
)

T_GATE = GateDefinition(
    name="T",
    num_qubits=1,
    matrix=np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
)


def rx_gate(theta: float) -> GateDefinition:
    """Rotation around X axis by angle theta."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return GateDefinition(
        name="RX",
        num_qubits=1,
        matrix=np.array([[c, -1j * s], [-1j * s, c]], dtype=complex),
        params=(theta,),
    )


def ry_gate(theta: float) -> GateDefinition:
    """Rotation around Y axis by angle theta."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return GateDefinition(
        name="RY",
        num_qubits=1,
        matrix=np.array([[c, -s], [s, c]], dtype=complex),
        params=(theta,),
    )


def rz_gate(theta: float) -> GateDefinition:
    """Rotation around Z axis by angle theta."""
    return GateDefinition(
        name="RZ",
        num_qubits=1,
        matrix=np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
            dtype=complex,
        ),
        params=(theta,),
    )


def phase_gate(theta: float) -> GateDefinition:
    """Phase gate P(theta) = diag(1, e^{i*theta})."""
    return GateDefinition(
        name="P",
        num_qubits=1,
        matrix=np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex),
        params=(theta,),
    )


# X90 — pi/2 rotation around X
X90_GATE = rx_gate(np.pi / 2)

# --- Two-qubit gates ---

CNOT_GATE = GateDefinition(
    name="CNOT",
    num_qubits=2,
    matrix=np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    ),
)

CZ_GATE = GateDefinition(
    name="CZ",
    num_qubits=2,
    matrix=np.diag([1, 1, 1, -1]).astype(complex),
)

SWAP_GATE = GateDefinition(
    name="SWAP",
    num_qubits=2,
    matrix=np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=complex,
    ),
)

# --- Three-qubit gates ---

TOFFOLI_GATE = GateDefinition(
    name="Toffoli",
    num_qubits=3,
    matrix=np.eye(8, dtype=complex),
)
# Swap |110⟩ ↔ |111⟩
TOFFOLI_GATE.matrix[6, 6] = 0
TOFFOLI_GATE.matrix[7, 7] = 0
TOFFOLI_GATE.matrix[6, 7] = 1
TOFFOLI_GATE.matrix[7, 6] = 1

# Lookup table for gate names → definitions (parameterless gates only)
GATE_TABLE: dict[str, GateDefinition] = {
    "I": I_GATE,
    "X": X_GATE,
    "Y": Y_GATE,
    "Z": Z_GATE,
    "H": H_GATE,
    "S": S_GATE,
    "T": T_GATE,
    "X90": X90_GATE,
    "CNOT": CNOT_GATE,
    "CZ": CZ_GATE,
    "SWAP": SWAP_GATE,
    "Toffoli": TOFFOLI_GATE,
}
