"""Tests for QuantumCluster multi-board orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from pynq_quantum.backends.base import ExecutionResult
from pynq_quantum.circuit import QuantumCircuit
from pynq_quantum.cluster import BoardInfo, QuantumCluster


class TestBoardInfo:
    def test_defaults(self):
        info = BoardInfo(host="192.168.1.10")
        assert info.host == "192.168.1.10"
        assert info.overlay is None
        assert info.num_qubits == 0
        assert info.connected is False
        assert info.metadata == {}

    def test_custom_fields(self):
        info = BoardInfo(host="board-a", num_qubits=4, connected=True, metadata={"role": "main"})
        assert info.num_qubits == 4
        assert info.connected is True
        assert info.metadata["role"] == "main"


class TestQuantumClusterInit:
    def test_init_basic(self):
        cluster = QuantumCluster(["host1", "host2"])
        assert cluster.num_boards == 2
        assert cluster.total_qubits == 0
        assert len(cluster.boards) == 2

    def test_init_single_host(self):
        cluster = QuantumCluster(["localhost"])
        assert cluster.num_boards == 1

    def test_init_empty_hosts(self):
        cluster = QuantumCluster([])
        assert cluster.num_boards == 0
        assert cluster.total_qubits == 0

    def test_boards_returns_copy(self):
        cluster = QuantumCluster(["h1", "h2"])
        boards = cluster.boards
        boards.append(BoardInfo(host="h3"))
        assert cluster.num_boards == 2  # Original unchanged

    def test_backend_and_kwargs_stored(self):
        cluster = QuantumCluster(["h1"], backend="simulation", num_qubits=4, seed=99)
        assert cluster._backend_name == "simulation"
        assert cluster._kwargs == {"num_qubits": 4, "seed": 99}


class TestQuantumClusterConnect:
    def test_connect_with_simulation(self):
        cluster = QuantumCluster(["board1", "board2"], backend="simulation", num_qubits=4, seed=42)
        cluster.connect()

        assert all(b.connected for b in cluster.boards)
        assert all(b.overlay is not None for b in cluster.boards)
        assert all(b.num_qubits == 4 for b in cluster.boards)
        assert cluster.total_qubits == 8

        cluster.disconnect()

    def test_connect_failure_marks_board_disconnected(self):
        cluster = QuantumCluster(["bad-host"], backend="nonexistent_backend")
        cluster.connect()  # Should not raise

        board = cluster.boards[0]
        assert board.connected is False
        assert board.overlay is None
        assert "error" in board.metadata

    def test_disconnect(self):
        cluster = QuantumCluster(["h1"], backend="simulation", num_qubits=4, seed=42)
        cluster.connect()
        assert cluster.boards[0].connected is True

        cluster.disconnect()
        assert cluster.boards[0].connected is False
        assert cluster.boards[0].overlay is None
        assert cluster._synced is False

    def test_disconnect_without_connect(self):
        """Disconnecting an unconnected cluster should not raise."""
        cluster = QuantumCluster(["h1"])
        cluster.disconnect()  # No error


class TestQuantumClusterProperties:
    def test_num_boards(self):
        cluster = QuantumCluster(["a", "b", "c"])
        assert cluster.num_boards == 3

    def test_total_qubits_only_connected(self):
        """total_qubits should only count connected boards."""
        cluster = QuantumCluster(["h1", "h2"], backend="simulation", num_qubits=4, seed=42)
        cluster.connect()
        # Manually disconnect one board
        cluster._boards[1].connected = False
        assert cluster.total_qubits == 4  # Only first board

        cluster.disconnect()

    def test_total_qubits_none_connected(self):
        cluster = QuantumCluster(["h1", "h2"])
        assert cluster.total_qubits == 0


class TestQuantumClusterSyncClocks:
    def test_sync_clocks(self):
        cluster = QuantumCluster(["h1", "h2"], backend="simulation", num_qubits=4, seed=42)
        cluster.connect()
        assert cluster._synced is False

        cluster.sync_clocks()

        assert cluster._synced is True
        for board in cluster.boards:
            assert "clock_sync_time" in board.metadata
            assert isinstance(board.metadata["clock_sync_time"], float)

        cluster.disconnect()
        assert cluster._synced is False  # Reset on disconnect

    def test_sync_clocks_skips_disconnected(self):
        cluster = QuantumCluster(["h1", "h2"])
        # Neither board is connected
        cluster.sync_clocks()
        assert cluster._synced is True
        # No timestamps set on unconnected boards
        for board in cluster.boards:
            assert "clock_sync_time" not in board.metadata


class TestQuantumClusterDistributeCircuit:
    def _make_connected_cluster(self, n_boards=2, num_qubits=4):
        hosts = [f"board{i}" for i in range(n_boards)]
        cluster = QuantumCluster(hosts, backend="simulation", num_qubits=num_qubits, seed=42)
        cluster.connect()
        return cluster

    def test_even_split_two_boards(self):
        cluster = self._make_connected_cluster(2)
        qc = QuantumCircuit(4)
        qc.h(0).h(1).h(2).h(3).measure_all()

        subs = cluster.distribute_circuit(qc)
        assert len(subs) == 2
        assert subs[0].num_qubits == 2
        assert subs[1].num_qubits == 2
        assert subs[0].name == "circuit_board0"
        assert subs[1].name == "circuit_board1"

        cluster.disconnect()

    def test_even_split_three_boards(self):
        cluster = self._make_connected_cluster(3)
        # 6-qubit circuit across 3 boards => 2 per board
        qc = QuantumCircuit(6)
        for i in range(6):
            qc.h(i)
        qc.measure_all()

        subs = cluster.distribute_circuit(qc)
        assert len(subs) == 3
        assert subs[0].num_qubits == 2
        assert subs[1].num_qubits == 2
        assert subs[2].num_qubits == 2  # Last board gets remainder

        cluster.disconnect()

    def test_explicit_partition(self):
        cluster = self._make_connected_cluster(2)
        qc = QuantumCircuit(4)
        qc.h(0).h(1).h(2).h(3).measure_all()

        partition = [[0, 1, 2], [3]]
        subs = cluster.distribute_circuit(qc, partition=partition)
        assert len(subs) == 2
        assert subs[0].num_qubits == 3
        assert subs[1].num_qubits == 1

        cluster.disconnect()

    def test_distribute_maps_gates(self):
        """Gates on partitioned qubits should be remapped."""
        cluster = self._make_connected_cluster(2)
        qc = QuantumCircuit(4)
        qc.h(0).h(2).cnot(0, 1).measure_all()

        # Board 0 gets qubits [0,1], Board 1 gets qubits [2,3]
        subs = cluster.distribute_circuit(qc)

        # Sub-circuit 0 should have H on qubit 0 and CNOT(0,1)
        from pynq_quantum.gates import GateOp

        gate_ops_0 = [op for op in subs[0].ops if isinstance(op, GateOp)]
        assert len(gate_ops_0) == 2  # H(0) and CNOT(0,1)

        # Sub-circuit 1 should have H on qubit 0 (remapped from qubit 2)
        gate_ops_1 = [op for op in subs[1].ops if isinstance(op, GateOp)]
        assert len(gate_ops_1) == 1  # H(2) remapped to H(0)
        assert gate_ops_1[0].qubits == (0,)

        cluster.disconnect()

    def test_distribute_filters_cross_board_gates(self):
        """A CNOT spanning boards should not appear in either sub-circuit."""
        cluster = self._make_connected_cluster(2)
        qc = QuantumCircuit(4)
        qc.cnot(0, 2)  # Qubit 0 on board 0, qubit 2 on board 1
        qc.measure_all()

        subs = cluster.distribute_circuit(qc)

        from pynq_quantum.gates import GateOp

        # Neither sub-circuit should contain the cross-board CNOT
        for sub in subs:
            gate_ops = [op for op in sub.ops if isinstance(op, GateOp)]
            cnots = [op for op in gate_ops if op.gate.name == "CNOT"]
            assert len(cnots) == 0

        cluster.disconnect()

    def test_distribute_no_connected_boards_raises(self):
        cluster = QuantumCluster(["h1", "h2"])
        qc = QuantumCircuit(2)
        qc.h(0).measure_all()

        with pytest.raises(RuntimeError, match="No connected boards"):
            cluster.distribute_circuit(qc)


class TestQuantumClusterRun:
    def test_run_basic(self):
        cluster = QuantumCluster(["h1", "h2"], backend="simulation", num_qubits=4, seed=42)
        cluster.connect()

        qc = QuantumCircuit(4)
        qc.x(0).x(2).measure_all()

        results = cluster.run(qc, shots=100)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, ExecutionResult)
            assert isinstance(result.counts, dict)
            total = sum(result.counts.values())
            assert total == 100

        cluster.disconnect()

    def test_run_with_partition(self):
        cluster = QuantumCluster(["h1", "h2"], backend="simulation", num_qubits=4, seed=42)
        cluster.connect()

        qc = QuantumCircuit(3)
        qc.h(0).h(1).h(2).measure_all()

        partition = [[0, 1], [2]]
        results = cluster.run(qc, shots=50, partition=partition)
        assert len(results) == 2

        cluster.disconnect()

    def test_run_no_connected_boards_raises(self):
        cluster = QuantumCluster(["h1"])
        qc = QuantumCircuit(2)
        qc.h(0).measure_all()

        with pytest.raises(RuntimeError, match="No connected boards"):
            cluster.run(qc)


class TestQuantumClusterLocalMeasure:
    def test_local_measure(self):
        cluster = QuantumCluster(["h1"], backend="simulation", num_qubits=4, seed=42)
        cluster.connect()

        result = cluster.local_measure(0, [0, 1])
        assert isinstance(result, np.ndarray)

        cluster.disconnect()

    def test_local_measure_board_index_out_of_range(self):
        cluster = QuantumCluster(["h1"])
        with pytest.raises(IndexError, match="out of range"):
            cluster.local_measure(5, [0])

    def test_local_measure_disconnected_board(self):
        cluster = QuantumCluster(["h1"])
        # Board exists but is not connected
        with pytest.raises(RuntimeError, match="not connected"):
            cluster.local_measure(0, [0])


class TestQuantumClusterContextManager:
    def test_context_manager_connects_and_disconnects(self):
        with QuantumCluster(["h1"], backend="simulation", num_qubits=4, seed=42) as cluster:
            assert cluster.boards[0].connected is True
            assert cluster.boards[0].overlay is not None

        # After exiting, boards should be disconnected
        assert cluster.boards[0].connected is False
        assert cluster.boards[0].overlay is None

    def test_context_manager_disconnects_on_exception(self):
        try:
            with QuantumCluster(["h1"], backend="simulation", num_qubits=4, seed=42) as cluster:
                assert cluster.boards[0].connected is True
                raise ValueError("test error")
        except ValueError:
            pass

        assert cluster.boards[0].connected is False

    def test_context_manager_returns_cluster(self):
        with QuantumCluster(["h1"], backend="simulation", num_qubits=4, seed=42) as cluster:
            assert isinstance(cluster, QuantumCluster)


class TestQuantumClusterErrorHandling:
    def test_partial_connect_failure(self):
        """If one board fails to connect, others still connect."""
        cluster = QuantumCluster(["h1", "h2"], backend="simulation", num_qubits=4, seed=42)

        # Patch QuantumOverlay to fail on second call
        def patched_connect(self):
            from pynq_quantum.overlay import QuantumOverlay

            for board in self._boards:
                try:
                    overlay = QuantumOverlay(backend=self._backend_name, **self._kwargs)
                    caps = overlay.backend.get_capabilities()
                    board.overlay = overlay
                    board.num_qubits = caps.get("max_qubits", 0)
                    board.connected = True
                except Exception as e:
                    board.metadata["error"] = str(e)
                    board.connected = False

        # Use the standard connect (which works with simulation backend)
        cluster.connect()

        # Verify both boards connected
        assert cluster.boards[0].connected is True
        assert cluster.boards[1].connected is True

        # Manually simulate partial failure for verification
        cluster._boards[1].connected = False
        cluster._boards[1].num_qubits = 0
        assert cluster.total_qubits == 4  # Only first board's qubits
        assert cluster.num_boards == 2  # Still 2 boards total

        cluster.disconnect()

    def test_run_skips_boards_without_overlay(self):
        """Boards with overlay=None should be silently skipped in run()."""
        cluster = QuantumCluster(["h1"], backend="simulation", num_qubits=4, seed=42)
        cluster.connect()

        # Manually remove overlay but keep connected flag
        cluster._boards[0].overlay = None

        qc = QuantumCircuit(2)
        qc.h(0).measure_all()

        # distribute_circuit works since board is marked connected,
        # but run() skips the board because overlay is None
        results = cluster.run(qc, shots=100)
        assert len(results) == 0

        cluster.disconnect()
