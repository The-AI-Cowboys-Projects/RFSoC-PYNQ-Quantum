# Custom Backend Guide

## Overview

`pynq-quantum` uses a pluggable backend system. You can implement your own backend by subclassing `AbstractBackend` and registering it with the backend registry.

## Implementing a Backend

### Step 1: Subclass AbstractBackend

```python
from pynq_quantum.backends.base import (
    AbstractBackend,
    ExecutionResult,
    PulseInstruction,
    ReadoutInstruction,
)

class MyBackend(AbstractBackend):
    def __init__(self, **kwargs):
        self._connected = False

    @property
    def name(self) -> str:
        return "my_backend"

    @property
    def num_channels(self) -> int:
        return 4

    def connect(self, **kwargs) -> None:
        # Initialize hardware connection
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def configure_qubit(self, qubit_id: int, frequency: float, **params) -> None:
        # Store qubit configuration
        pass

    def execute(
        self,
        pulses: list[PulseInstruction],
        readouts: list[ReadoutInstruction],
        shots: int = 1000,
    ) -> ExecutionResult:
        # Execute the pulse program on hardware
        # Return measurement results
        counts = {"0": shots}  # Placeholder
        return ExecutionResult(counts=counts)

    def get_capabilities(self) -> dict:
        return {
            "max_qubits": self.num_channels,
            "simulation": False,
        }
```

### Step 2: Register the Backend

```python
from pynq_quantum.backends import register_backend
register_backend("my_backend", MyBackend)
```

### Step 3: Use It

```python
from pynq_quantum import QuantumOverlay

overlay = QuantumOverlay(backend="my_backend")
```

## Data Types

### PulseInstruction

Represents a DAC pulse to emit:

| Field | Type | Description |
|-------|------|-------------|
| `channel` | `int` | DAC channel index |
| `frequency` | `float` | Carrier frequency (Hz) |
| `phase` | `float` | Carrier phase (radians) |
| `amplitude` | `float` | Pulse amplitude (0.0–1.0) |
| `duration` | `float` | Pulse duration (seconds) |
| `envelope` | `str` | Envelope shape: `gaussian`, `square`, `drag`, `flat_top` |
| `envelope_params` | `dict` | Shape-specific parameters (e.g. `sigma`) |

### ReadoutInstruction

Represents an ADC acquisition:

| Field | Type | Description |
|-------|------|-------------|
| `channel` | `int` | ADC channel index |
| `frequency` | `float` | Readout frequency (Hz) |
| `duration` | `float` | Acquisition window (seconds) |
| `qubits` | `list[int]` | Which qubits this readout measures |

### ExecutionResult

Returned from `execute()`:

| Field | Type | Description |
|-------|------|-------------|
| `counts` | `dict[str, int]` | Bitstring histogram, e.g. `{'00': 512, '11': 488}` |
| `raw_data` | `np.ndarray \| None` | Optional raw IQ data |
| `metadata` | `dict` | Backend-specific metadata |

## Built-in Backends

| Backend | Module | Hardware | Description |
|---------|--------|----------|-------------|
| `simulation` | `backends/simulation.py` | None | NumPy statevector sim |
| `qick` | `backends/qick.py` | QICK-compatible RFSoC | Wraps QickSoc + AveragerProgram |
| `qubic` | `backends/qubic.py` | QubiC-compatible RFSoC | Wraps QubiC compiler + JSON config |
| `generic` | `backends/generic.py` | Any RFSoC with AXI-Lite IP | Raw register control |

## Tips

- Use lazy imports for hardware dependencies to avoid `ImportError` when users don't have your hardware library installed
- Support dependency injection (pass hardware objects via constructor) for testability
- Return meaningful `get_capabilities()` so the compiler can optimize for your hardware
- Handle the `connect()` → `execute()` → `disconnect()` lifecycle properly
