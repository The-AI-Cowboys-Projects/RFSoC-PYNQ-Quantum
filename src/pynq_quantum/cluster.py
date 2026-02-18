"""QuantumCluster — multi-board orchestration for distributed quantum control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .backends.base import ExecutionResult
from .circuit import QuantumCircuit
from .overlay import QuantumOverlay


@dataclass
class BoardInfo:
    """Information about a connected board in the cluster."""

    host: str
    overlay: QuantumOverlay | None = None
    num_qubits: int = 0
    connected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class QuantumCluster:
    """Orchestrates quantum control across multiple RFSoC boards.

    Connects to multiple boards and provides unified circuit distribution,
    clock synchronization, and result aggregation.

    Args:
        hosts: List of board hostnames/IPs.
        backend: Backend type for each board.
        **kwargs: Passed to each QuantumOverlay constructor.
    """

    def __init__(
        self,
        hosts: list[str],
        backend: str = "qick",
        **kwargs: Any,
    ) -> None:
        self._backend_name = backend
        self._kwargs = kwargs
        self._boards: list[BoardInfo] = [BoardInfo(host=h) for h in hosts]
        self._synced = False

    @property
    def num_boards(self) -> int:
        return len(self._boards)

    @property
    def total_qubits(self) -> int:
        return sum(b.num_qubits for b in self._boards if b.connected)

    @property
    def boards(self) -> list[BoardInfo]:
        return list(self._boards)

    def connect(self) -> None:
        """Connect to all boards in the cluster."""
        for board in self._boards:
            try:
                overlay = QuantumOverlay(
                    backend=self._backend_name, **self._kwargs
                )
                caps = overlay.backend.get_capabilities()
                board.overlay = overlay
                board.num_qubits = caps.get("max_qubits", 0)
                board.connected = True
            except Exception as e:
                board.metadata["error"] = str(e)
                board.connected = False

    def disconnect(self) -> None:
        """Disconnect all boards."""
        for board in self._boards:
            if board.overlay is not None:
                board.overlay.close()
                board.overlay = None
            board.connected = False
        self._synced = False

    def sync_clocks(self) -> None:
        """Synchronize clocks across all connected boards.

        On real hardware this would use the SYSREF signal or
        a shared reference clock. In software we record timestamps
        and flag the cluster as synchronized.
        """
        import time

        for board in self._boards:
            if board.connected:
                board.metadata["clock_sync_time"] = time.time()
        self._synced = True

    def distribute_circuit(
        self,
        circuit: QuantumCircuit,
        partition: list[list[int]] | None = None,
    ) -> list[QuantumCircuit]:
        """Partition a circuit across boards.

        Args:
            circuit: The full circuit to distribute.
            partition: Optional explicit qubit-to-board mapping.
                       partition[i] = list of qubit indices for board i.
                       If None, qubits are split evenly.

        Returns:
            List of sub-circuits, one per board.
        """
        connected_boards = [b for b in self._boards if b.connected]
        if not connected_boards:
            raise RuntimeError("No connected boards in cluster")

        n_boards = len(connected_boards)

        if partition is None:
            # Even split
            nq = circuit.num_qubits
            chunk = max(1, nq // n_boards)
            partition = []
            for i in range(n_boards):
                start = i * chunk
                end = start + chunk if i < n_boards - 1 else nq
                partition.append(list(range(start, end)))

        sub_circuits: list[QuantumCircuit] = []
        for i, qubit_set in enumerate(partition):
            sub = QuantumCircuit(
                num_qubits=len(qubit_set),
                name=f"{circuit.name}_board{i}",
            )
            qubit_map = {old: new for new, old in enumerate(qubit_set)}

            for op in circuit.ops:
                from .gates import GateOp, MeasureOp

                if isinstance(op, GateOp):
                    if all(q in qubit_map for q in op.qubits):
                        mapped = tuple(qubit_map[q] for q in op.qubits)
                        sub._ops.append(GateOp(gate=op.gate, qubits=mapped))
                elif isinstance(op, MeasureOp):
                    mapped_qubits = tuple(
                        qubit_map[q] for q in op.qubits if q in qubit_map
                    )
                    if mapped_qubits:
                        sub._ops.append(MeasureOp(qubits=mapped_qubits))

            sub_circuits.append(sub)

        return sub_circuits

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1000,
        partition: list[list[int]] | None = None,
    ) -> list[ExecutionResult]:
        """Distribute and run a circuit across the cluster.

        Returns one ExecutionResult per board.
        """
        sub_circuits = self.distribute_circuit(circuit, partition)
        results: list[ExecutionResult] = []

        connected = [b for b in self._boards if b.connected]
        for board, sub_circuit in zip(connected, sub_circuits):
            if board.overlay is not None:
                result = sub_circuit.run(board.overlay, shots=shots)
                results.append(result)

        return results

    def local_measure(self, board_index: int, qubits: list[int]) -> np.ndarray:
        """Measure specific qubits on a single board.

        Returns raw IQ data as numpy array (board-specific).
        """
        if board_index >= len(self._boards):
            raise IndexError(f"Board index {board_index} out of range")
        board = self._boards[board_index]
        if not board.connected or board.overlay is None:
            raise RuntimeError(f"Board {board_index} ({board.host}) not connected")

        # Create a minimal measure circuit
        sub = QuantumCircuit(num_qubits=max(qubits) + 1)
        for q in qubits:
            sub.measure(q)
        result = sub.run(board.overlay, shots=1)
        return result.raw_data if result.raw_data is not None else np.array([])

    def __enter__(self) -> QuantumCluster:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()
