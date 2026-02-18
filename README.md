# RFSoC-PYNQ-Quantum

**Unified Python API for multi-backend quantum control on Xilinx RFSoC.**

Implements [RFC #57](https://github.com/Xilinx/RFSoC-PYNQ/issues/57) — a `pynq.quantum` package that unifies fragmented quantum control ecosystems (QICK, QubiC, custom HLS) under a standard Python API.

## Features

- **Multi-backend support** — QICK, QubiC, generic HLS, and simulation backends behind one API
- **NumPy simulation** — Full state-vector simulator for development and CI (no hardware needed)
- **Standard gate library** — X, Y, Z, H, CNOT, CZ, RX, RY, RZ, S, T, SWAP, Toffoli
- **Pulse compiler** — Translates gate operations to calibrated pulse instructions
- **Multi-board clustering** — Distribute circuits across multiple RFSoC boards
- **Qiskit/Cirq bridges** — Run Qiskit or Cirq circuits on RFSoC hardware
- **Pip-installable** — `numpy` is the only hard dependency

## Install

```bash
# Core (simulation only — no hardware deps)
pip install pynq-quantum

# With specific backends
pip install pynq-quantum[qick]
pip install pynq-quantum[qubic]
pip install pynq-quantum[qiskit]
pip install pynq-quantum[cirq]

# Everything
pip install pynq-quantum[all]

# Development
pip install -e ".[dev]"
```

## Quickstart

```python
from pynq_quantum import QuantumOverlay, QubitController

# Create overlay with simulation backend (default)
overlay = QuantumOverlay(backend='simulation')

# Create controller for 2 qubits
qc = QubitController(overlay, num_qubits=2)

# Build a Bell state
qc.h(0)
qc.cnot(0, 1)
qc.measure([0, 1])

# Run 1000 shots
result = qc.run(shots=1000)
print(result.counts)  # {'00': ~500, '11': ~500}
```

### Using QuantumCircuit Builder

```python
from pynq_quantum import QuantumOverlay, QuantumCircuit

overlay = QuantumOverlay(backend='simulation')

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cnot(0, 1)
circuit.measure_all()

result = circuit.run(overlay, shots=1000)
print(result.counts)
```

### Qiskit Bridge

```python
from pynq_quantum.bridges.qiskit import RFSoCBackend
from qiskit import QuantumCircuit, transpile

backend = RFSoCBackend()  # Uses simulation by default
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

job = backend.run(transpile(qc, backend), shots=1000)
result = job.result()
print(result.get_counts())
```

## Architecture

```
┌────────────────────────────────────────┐
│          User Application              │
├────────────────────────────────────────┤
│  QuantumCircuit  │  QubitController    │
├──────────────────┴─────────────────────┤
│            PulseCompiler               │
├────────────────────────────────────────┤
│          QuantumOverlay                │
├──────┬──────┬──────┬───────────────────┤
│ QICK │QubiC │ HLS  │  Simulation      │
│ Back │ Back │ Back │  Backend          │
│ end  │ end  │ end  │  (NumPy)         │
├──────┴──────┴──────┴───────────────────┤
│      RFSoC Hardware / Simulator        │
└────────────────────────────────────────┘
```

## Supported Boards

| Board | Backend | Status |
|-------|---------|--------|
| ZCU111 | QICK | Supported |
| ZCU216 | QICK | Supported |
| RFSoC 4x2 | QICK / QubiC | Supported |
| RFSoC 2x2 | QICK / QubiC | Supported |
| Custom HLS | Generic | Supported |
| No hardware | Simulation | Default |

## Development

```bash
git clone https://github.com/The-AI-Cowboys-Projects/RFSoC-PYNQ-Quantum.git
cd RFSoC-PYNQ-Quantum
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=pynq_quantum

# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Type check
mypy src/
```

## License

BSD-3-Clause — see [LICENSE](LICENSE).
