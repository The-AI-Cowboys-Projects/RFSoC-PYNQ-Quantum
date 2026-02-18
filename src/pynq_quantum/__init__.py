"""pynq-quantum — Unified Python API for multi-backend quantum control on Xilinx RFSoC."""

from __future__ import annotations

from .backends.base import ExecutionResult, PulseInstruction, ReadoutInstruction
from .circuit import QuantumCircuit
from .compiler import PulseCompiler, QubitCalibration
from .controller import QubitController
from .gates import (
    CNOT_GATE,
    CZ_GATE,
    GATE_TABLE,
    H_GATE,
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
    rx_gate,
    ry_gate,
    rz_gate,
)
from .overlay import QuantumOverlay

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "QuantumOverlay",
    "QubitController",
    "QuantumCircuit",
    "PulseCompiler",
    "QubitCalibration",
    # Data types
    "ExecutionResult",
    "PulseInstruction",
    "ReadoutInstruction",
    "GateDefinition",
    "GateOp",
    "MeasureOp",
    # Gate constants
    "X_GATE",
    "Y_GATE",
    "Z_GATE",
    "H_GATE",
    "S_GATE",
    "T_GATE",
    "CNOT_GATE",
    "CZ_GATE",
    "SWAP_GATE",
    "TOFFOLI_GATE",
    "GATE_TABLE",
    # Gate constructors
    "rx_gate",
    "ry_gate",
    "rz_gate",
]
