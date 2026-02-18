# API Reference

## Core Classes

### QuantumOverlay

```python
class QuantumOverlay(backend='simulation', bitstream=None, **kwargs)
```

Entry point for quantum control. Handles board detection and backend lifecycle.

**Parameters:**
- `backend` (`str`) — Backend name: `'simulation'`, `'qick'`, `'qubic'`, `'generic'`, or `'auto'`
- `bitstream` (`str | None`) — Path to `.bit` file for PYNQ overlay loading
- `**kwargs` — Passed to the backend constructor

**Properties:**
- `backend` → `AbstractBackend` — Active backend instance
- `backend_name` → `str` — Name of the active backend

**Methods:**
- `detect_board()` → `str | None` — Detect RFSoC board model
- `close()` — Disconnect backend and free resources

**Context Manager:** `with QuantumOverlay(...) as overlay: ...`

---

### QubitController

```python
class QubitController(overlay, num_qubits)
```

Imperative quantum control API (RFC #57).

**Parameters:**
- `overlay` (`QuantumOverlay`) — Active overlay with backend
- `num_qubits` (`int`) — Number of qubits to control

**Properties:**
- `num_qubits` → `int`
- `program` → `list[GateOp | MeasureOp]` — Current program (copy)

**Configuration:**
- `set_qubit_frequency(qubit, freq)` — Set drive frequency (Hz)

**Single-Qubit Gates:**
- `x(qubit)` — Pauli-X (NOT)
- `y(qubit)` — Pauli-Y
- `z(qubit)` — Pauli-Z
- `h(qubit)` — Hadamard
- `x90(qubit)` — X90 (π/2 rotation around X)
- `rx(qubit, theta)` — Rotation around X
- `ry(qubit, theta)` — Rotation around Y
- `rz(qubit, theta)` — Rotation around Z

**Two-Qubit Gates:**
- `cnot(control, target)` — Controlled-NOT
- `cz(control, target)` — Controlled-Z
- `swap(qubit1, qubit2)` — SWAP

**Three-Qubit Gates:**
- `toffoli(control1, control2, target)` — Toffoli (CCX)

**Execution:**
- `measure(qubits=None)` — Add measurement (all qubits if None)
- `run(shots=1000)` → `ExecutionResult` — Compile and execute
- `reset()` — Clear accumulated program

---

### QuantumCircuit

```python
class QuantumCircuit(num_qubits, name='circuit')
```

Declarative circuit builder with fluent API.

**Properties:**
- `num_qubits` → `int`
- `name` → `str`
- `ops` → `list[GateOp | MeasureOp]` — Operations (copy)
- `depth` → `int` — Circuit depth
- `num_gates` → `int` — Gate count

**Gates:** Same as QubitController, but return `self` for chaining. Additional aliases:
- `i(qubit)` — Identity
- `s(qubit)` — S gate (phase π/2)
- `t(qubit)` — T gate (phase π/4)
- `p(qubit, theta)` — Phase gate P(θ)
- `cx(control, target)` — Alias for `cnot`
- `ccx(c1, c2, target)` — Alias for `toffoli`

**Measurement:**
- `measure(qubit)` — Measure single qubit
- `measure_all()` — Measure all qubits

**Execution:**
- `run(overlay, shots=1000)` → `ExecutionResult`

---

### PulseCompiler

```python
class PulseCompiler(num_qubits)
```

Compiles gate operations to pulse instructions.

**Methods:**
- `set_calibration(qubit, calibration)` — Set qubit calibration
- `get_calibration(qubit)` → `QubitCalibration`
- `compile(program)` → `tuple[list[PulseInstruction], list[ReadoutInstruction]]`

---

### QuantumCluster

```python
class QuantumCluster(hosts, backend='qick', **kwargs)
```

Multi-board orchestration.

**Properties:**
- `num_boards` → `int`
- `total_qubits` → `int`
- `boards` → `list[BoardInfo]`

**Methods:**
- `connect()` — Connect to all boards
- `disconnect()` — Disconnect all
- `sync_clocks()` — Synchronize board clocks
- `distribute_circuit(circuit, partition=None)` → `list[QuantumCircuit]`
- `run(circuit, shots=1000, partition=None)` → `list[ExecutionResult]`
- `local_measure(board_index, qubits)` → `np.ndarray`

**Context Manager:** `with QuantumCluster(...) as cluster: ...`

---

### QuantumCollective

```python
class QuantumCollective(cluster)
```

ACCL-Q collective operations.

**Properties:**
- `rank` → `int`
- `size` → `int`

**Methods:**
- `init_accl(**kwargs)` — Initialize communication layer
- `barrier()` — Block until all boards synchronize
- `broadcast(data, root=0)` → `np.ndarray`
- `allreduce(data, op=ReduceOp.SUM)` → `np.ndarray`
- `gather(data, root=0)` → `np.ndarray | None`
- `scatter(data, root=0)` → `np.ndarray`
- `merge_counts(local_counts)` → `dict[str, int]`

---

## Data Types

### ExecutionResult

```python
@dataclass
class ExecutionResult:
    counts: dict[str, int]       # {'00': 512, '11': 488}
    raw_data: np.ndarray | None  # Optional raw measurement data
    metadata: dict[str, Any]     # Backend-specific metadata
```

### PulseInstruction

```python
@dataclass
class PulseInstruction:
    channel: int          # DAC channel
    frequency: float      # Hz
    phase: float          # radians
    amplitude: float      # 0.0–1.0
    duration: float       # seconds
    envelope: str         # 'gaussian', 'square', 'drag', 'flat_top'
    envelope_params: dict
```

### GateDefinition

```python
@dataclass(frozen=True)
class GateDefinition:
    name: str
    num_qubits: int
    matrix: np.ndarray    # Unitary matrix
    params: tuple[float, ...]
```

### GateOp / MeasureOp

```python
@dataclass
class GateOp:
    gate: GateDefinition
    qubits: tuple[int, ...]

@dataclass
class MeasureOp:
    qubits: tuple[int, ...]
```

---

## Gate Constants

| Name | Qubits | Import |
|------|--------|--------|
| `I_GATE` | 1 | `from pynq_quantum.gates import I_GATE` |
| `X_GATE` | 1 | `from pynq_quantum import X_GATE` |
| `Y_GATE` | 1 | `from pynq_quantum import Y_GATE` |
| `Z_GATE` | 1 | `from pynq_quantum import Z_GATE` |
| `H_GATE` | 1 | `from pynq_quantum import H_GATE` |
| `S_GATE` | 1 | `from pynq_quantum import S_GATE` |
| `T_GATE` | 1 | `from pynq_quantum import T_GATE` |
| `CNOT_GATE` | 2 | `from pynq_quantum import CNOT_GATE` |
| `CZ_GATE` | 2 | `from pynq_quantum import CZ_GATE` |
| `SWAP_GATE` | 2 | `from pynq_quantum import SWAP_GATE` |
| `TOFFOLI_GATE` | 3 | `from pynq_quantum import TOFFOLI_GATE` |

**Parameterized Gate Constructors:**
- `rx_gate(theta)` → `GateDefinition`
- `ry_gate(theta)` → `GateDefinition`
- `rz_gate(theta)` → `GateDefinition`
- `phase_gate(theta)` → `GateDefinition`

---

## Bridges

### RFSoCBackend (Qiskit)

```python
from pynq_quantum.bridges.qiskit import RFSoCBackend

backend = RFSoCBackend(overlay=None, num_qubits=8)
job = backend.run(qiskit_circuit, shots=1000)
result = job.result()
counts = result.get_counts()
```

### RFSoCSampler (Cirq)

```python
from pynq_quantum.bridges.cirq import RFSoCSampler

sampler = RFSoCSampler(overlay=None)
result = sampler.run(cirq_circuit, repetitions=1000)
measurements = result.measurements  # dict[str, np.ndarray]
histogram = result.histogram("key")  # dict[int, int]
```
