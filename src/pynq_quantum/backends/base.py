"""Abstract backend interface and shared data types for quantum control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PulseInstruction:
    """A pulse to emit on a DAC channel."""

    channel: int
    frequency: float  # Hz
    phase: float  # radians
    amplitude: float  # 0.0–1.0
    duration: float  # seconds
    envelope: str = "gaussian"  # 'gaussian', 'square', 'drag', 'flat_top'
    envelope_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadoutInstruction:
    """A readout acquisition on an ADC channel."""

    channel: int
    frequency: float  # Hz
    duration: float  # seconds
    qubits: list[int] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Result returned from a backend execution."""

    counts: dict[str, int]  # e.g. {'00': 512, '11': 488}
    raw_data: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AbstractBackend(ABC):
    """Base class all quantum control backends must implement."""

    @abstractmethod
    def connect(self, **kwargs: Any) -> None:
        """Establish connection to hardware or simulator."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release hardware resources."""

    @abstractmethod
    def configure_qubit(self, qubit_id: int, frequency: float, **params: Any) -> None:
        """Set qubit parameters (frequency, anharmonicity, etc.)."""

    @abstractmethod
    def execute(
        self,
        pulses: list[PulseInstruction],
        readouts: list[ReadoutInstruction],
        shots: int = 1000,
    ) -> ExecutionResult:
        """Execute a pulse program and return measurement results."""

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Return backend capabilities (max qubits, gates, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""

    @property
    @abstractmethod
    def num_channels(self) -> int:
        """Number of available DAC channels."""
