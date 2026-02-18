# Architecture Overview

## Package Design

`pynq-quantum` follows a layered architecture that separates user-facing APIs from hardware-specific implementations:

```
┌──────────────────────────────────────────────────┐
│                User Application                   │
├────────────────────┬─────────────────────────────┤
│   QuantumCircuit   │      QubitController         │
│   (declarative)    │      (imperative / RFC API)  │
├────────────────────┴─────────────────────────────┤
│                PulseCompiler                       │
│   gates → calibrated pulse instructions            │
├──────────────────────────────────────────────────┤
│              QuantumOverlay                        │
│   board detection + backend lifecycle              │
├────────┬─────────┬──────────┬────────────────────┤
│  QICK  │  QubiC  │ Generic  │    Simulation      │
│Backend │Backend  │ Backend  │    Backend          │
├────────┴─────────┴──────────┴────────────────────┤
│          RFSoC Hardware / NumPy                    │
└──────────────────────────────────────────────────┘
```

## Core Components

### QuantumOverlay (`overlay.py`)

Entry point. Handles:
- Board auto-detection (`/proc/device-tree/model`, `PYNQ_BOARD` env, PYNQ library)
- Backend instantiation from the registry
- Optional bitstream loading via `pynq.Overlay`
- Context manager for resource cleanup

### QubitController (`controller.py`)

Imperative API matching RFC #57. Users accumulate gate operations, then call `run()` to execute:

```python
qc = QubitController(overlay, num_qubits=2)
qc.h(0)
qc.cnot(0, 1)
qc.measure([0, 1])
result = qc.run(shots=1000)
```

For the simulation backend, `run()` uses the efficient statevector path. For hardware backends, it compiles gates to pulse instructions via `PulseCompiler`.

### QuantumCircuit (`circuit.py`)

Declarative circuit builder with fluent API:

```python
circuit = QuantumCircuit(2)
circuit.h(0).cnot(0, 1).measure_all()
result = circuit.run(overlay, shots=1000)
```

### PulseCompiler (`compiler.py`)

Translates `GateOp` / `MeasureOp` sequences into `PulseInstruction` and `ReadoutInstruction` objects. Per-qubit calibration data (frequency, amplitude, duration) is stored in `QubitCalibration` dataclasses.

Key design decisions:
- **Virtual-Z gates**: Z, S, T are zero-duration phase updates (no physical pulse)
- **Cross-resonance**: CNOT uses cross-resonance drive at control frequency on target channel
- **SWAP decomposition**: Three CNOTs
- **Toffoli decomposition**: Standard H/CNOT/T sequence

### Gate Library (`gates.py`)

Immutable `GateDefinition` objects with unitary matrices. All gates are verified unitary in tests. Parameterized gates (`rx_gate`, `ry_gate`, `rz_gate`, `phase_gate`) return new definitions for each angle.

## Backend System

### AbstractBackend (`backends/base.py`)

ABC with five required methods:
- `connect()` / `disconnect()` — lifecycle
- `configure_qubit()` — set qubit parameters
- `execute()` — run pulse program, return `ExecutionResult`
- `get_capabilities()` — report backend features

Three dataclasses define the pulse/readout/result contract: `PulseInstruction`, `ReadoutInstruction`, `ExecutionResult`.

### Backend Registry (`backends/__init__.py`)

Lazy-loading registry. Backends are loaded on first use to avoid import errors when optional dependencies are missing:

```python
from pynq_quantum.backends import get_backend, list_backends
SimBackend = get_backend("simulation")  # Loads on demand
```

### Simulation Backend (`backends/simulation.py`)

Full statevector simulation using NumPy matrix multiplication. Supports all gates. Provides two execution paths:
1. `execute()` — raw pulse instructions (random sampling, for API compatibility)
2. `execute_circuit()` — gate operations (proper quantum simulation)

### Hardware Backends

- **QICK** (`backends/qick.py`) — Wraps `qick.QickSoc` + `AveragerProgram`
- **QubiC** (`backends/qubic.py`) — Wraps QubiC compiler + JSON config
- **Generic** (`backends/generic.py`) — Raw AXI-Lite register control

All hardware backends use lazy imports and support dependency injection for testing.

## Multi-Board Clustering

### QuantumCluster (`cluster.py`)

Connects to multiple RFSoC boards and provides:
- Circuit distribution with configurable partitioning
- Clock synchronization
- Per-board local measurement

### QuantumCollective (`collective.py`)

MPI-style collective operations (allreduce, broadcast, gather, scatter, barrier). Uses `pyaccl` if available, falls back to software emulation.

## Framework Bridges

### Qiskit Bridge (`bridges/qiskit.py`)

`RFSoCBackend` — Qiskit BackendV2 compatible. Converts Qiskit `QuantumCircuit` to pynq-quantum gate ops, then executes on the selected backend.

### Cirq Bridge (`bridges/cirq.py`)

`RFSoCSampler` — Cirq Sampler compatible. Converts Cirq circuits to pynq-quantum gate ops. Returns `RFSoCTrialResult` with numpy measurement arrays.

## Dependency Strategy

- **numpy** is the only hard dependency
- All hardware libraries (pynq, qick, qubic, pyaccl) are optional extras
- Framework bridges (qiskit, cirq) are optional extras
- The simulation backend runs with just numpy installed
