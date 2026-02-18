"""Tests for QuantumCollective and _SoftwareACCL communication primitives."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from pynq_quantum.collective import QuantumCollective, ReduceOp, _SoftwareACCL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_cluster(num_boards: int = 3) -> MagicMock:
    """Create a mock QuantumCluster with a configurable board count."""
    cluster = MagicMock()
    cluster.num_boards = num_boards
    return cluster


# ---------------------------------------------------------------------------
# ReduceOp enum
# ---------------------------------------------------------------------------


class TestReduceOp:
    def test_values(self):
        assert ReduceOp.SUM.value == "sum"
        assert ReduceOp.MEAN.value == "mean"
        assert ReduceOp.MAX.value == "max"
        assert ReduceOp.MIN.value == "min"

    def test_members(self):
        assert len(ReduceOp) == 4


# ---------------------------------------------------------------------------
# _SoftwareACCL (emulation layer)
# ---------------------------------------------------------------------------


class TestSoftwareACCL:
    def test_init(self):
        cluster = _mock_cluster(4)
        accl = _SoftwareACCL(cluster)
        assert accl._cluster is cluster
        assert accl._data_store == []
        assert accl._count_store == []

    def test_barrier_is_noop(self):
        accl = _SoftwareACCL(_mock_cluster())
        accl.barrier()  # Should not raise

    def test_broadcast_returns_copy(self):
        accl = _SoftwareACCL(_mock_cluster())
        data = np.array([1.0, 2.0, 3.0])
        result = accl.broadcast(data, root=0)
        np.testing.assert_array_equal(result, data)
        assert result is not data  # Must be a copy

    def test_broadcast_nonzero_root(self):
        accl = _SoftwareACCL(_mock_cluster())
        data = np.array([10, 20])
        result = accl.broadcast(data, root=2)
        np.testing.assert_array_equal(result, data)

    def test_allreduce_returns_copy(self):
        accl = _SoftwareACCL(_mock_cluster())
        data = np.array([5.0, 6.0])
        result = accl.allreduce(data, op=ReduceOp.SUM)
        np.testing.assert_array_equal(result, data)
        assert result is not data

    def test_allreduce_all_ops(self):
        """All ReduceOp values accepted without error in emulation."""
        accl = _SoftwareACCL(_mock_cluster())
        data = np.array([1.0, 2.0, 3.0])
        for op in ReduceOp:
            result = accl.allreduce(data, op=op)
            np.testing.assert_array_equal(result, data)

    def test_gather_returns_copy(self):
        accl = _SoftwareACCL(_mock_cluster())
        data = np.array([7.0, 8.0])
        result = accl.gather(data, root=0)
        np.testing.assert_array_equal(result, data)
        assert result is not data

    def test_scatter_splits_data(self):
        accl = _SoftwareACCL(_mock_cluster(num_boards=3))
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = accl.scatter(data, root=0)
        # 6 / 3 = chunk_size 2 => first 2 elements
        np.testing.assert_array_equal(result, np.array([1.0, 2.0]))

    def test_scatter_none_data(self):
        accl = _SoftwareACCL(_mock_cluster())
        result = accl.scatter(None, root=0)
        assert len(result) == 0

    def test_scatter_with_one_board(self):
        accl = _SoftwareACCL(_mock_cluster(num_boards=1))
        data = np.array([10.0, 20.0, 30.0])
        result = accl.scatter(data, root=0)
        # chunk_size = 3 // 1 = 3, returns full array
        np.testing.assert_array_equal(result, data)

    def test_allgather_counts(self):
        accl = _SoftwareACCL(_mock_cluster())
        counts = {"00": 50, "11": 50}
        result = accl.allgather_counts(counts)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == counts


# ---------------------------------------------------------------------------
# QuantumCollective
# ---------------------------------------------------------------------------


class TestQuantumCollectiveInit:
    def test_init(self):
        cluster = _mock_cluster(4)
        coll = QuantumCollective(cluster)
        assert coll.rank == 0
        assert coll.size == 4
        assert coll._accl is None
        assert coll._cluster is cluster

    def test_init_single_board(self):
        cluster = _mock_cluster(1)
        coll = QuantumCollective(cluster)
        assert coll.size == 1

    def test_rank_default(self):
        coll = QuantumCollective(_mock_cluster())
        assert coll.rank == 0


class TestQuantumCollectiveInitACCL:
    def test_init_accl_uses_software_fallback(self):
        """Without pyaccl installed, should fall back to _SoftwareACCL."""
        cluster = _mock_cluster(3)
        coll = QuantumCollective(cluster)
        coll.init_accl()
        assert isinstance(coll._accl, _SoftwareACCL)

    def test_init_accl_twice(self):
        """Calling init_accl twice replaces the previous instance."""
        cluster = _mock_cluster(2)
        coll = QuantumCollective(cluster)
        coll.init_accl()
        first = coll._accl
        coll.init_accl()
        second = coll._accl
        assert first is not second


class TestQuantumCollectiveBarrier:
    def test_barrier(self):
        cluster = _mock_cluster()
        coll = QuantumCollective(cluster)
        coll.init_accl()
        coll.barrier()  # Should not raise

    def test_barrier_without_init_raises(self):
        coll = QuantumCollective(_mock_cluster())
        with pytest.raises(RuntimeError, match="Call init_accl"):
            coll.barrier()


class TestQuantumCollectiveBroadcast:
    def test_broadcast(self):
        cluster = _mock_cluster(3)
        coll = QuantumCollective(cluster)
        coll.init_accl()

        data = np.array([1.0, 2.0, 3.0])
        result = coll.broadcast(data, root=0)
        np.testing.assert_array_equal(result, data)

    def test_broadcast_nonzero_root(self):
        cluster = _mock_cluster(4)
        coll = QuantumCollective(cluster)
        coll.init_accl()

        data = np.array([42.0])
        result = coll.broadcast(data, root=2)
        np.testing.assert_array_equal(result, data)

    def test_broadcast_without_init_raises(self):
        coll = QuantumCollective(_mock_cluster())
        with pytest.raises(RuntimeError, match="Call init_accl"):
            coll.broadcast(np.array([1.0]))


class TestQuantumCollectiveAllreduce:
    def test_allreduce_sum(self):
        coll = QuantumCollective(_mock_cluster())
        coll.init_accl()
        data = np.array([1.0, 2.0, 3.0])
        result = coll.allreduce(data, op=ReduceOp.SUM)
        np.testing.assert_array_equal(result, data)

    def test_allreduce_mean(self):
        coll = QuantumCollective(_mock_cluster())
        coll.init_accl()
        data = np.array([10.0, 20.0])
        result = coll.allreduce(data, op=ReduceOp.MEAN)
        np.testing.assert_array_equal(result, data)

    def test_allreduce_max(self):
        coll = QuantumCollective(_mock_cluster())
        coll.init_accl()
        data = np.array([3.0, 1.0, 2.0])
        result = coll.allreduce(data, op=ReduceOp.MAX)
        np.testing.assert_array_equal(result, data)

    def test_allreduce_min(self):
        coll = QuantumCollective(_mock_cluster())
        coll.init_accl()
        data = np.array([7.0, 8.0])
        result = coll.allreduce(data, op=ReduceOp.MIN)
        np.testing.assert_array_equal(result, data)

    def test_allreduce_default_op_is_sum(self):
        coll = QuantumCollective(_mock_cluster())
        coll.init_accl()
        data = np.array([5.0])
        result = coll.allreduce(data)
        np.testing.assert_array_equal(result, data)

    def test_allreduce_without_init_raises(self):
        coll = QuantumCollective(_mock_cluster())
        with pytest.raises(RuntimeError, match="Call init_accl"):
            coll.allreduce(np.array([1.0]))


class TestQuantumCollectiveGather:
    def test_gather(self):
        coll = QuantumCollective(_mock_cluster())
        coll.init_accl()
        data = np.array([1.0, 2.0])
        result = coll.gather(data, root=0)
        np.testing.assert_array_equal(result, data)

    def test_gather_nonzero_root(self):
        coll = QuantumCollective(_mock_cluster(4))
        coll.init_accl()
        data = np.array([9.0])
        result = coll.gather(data, root=3)
        np.testing.assert_array_equal(result, data)

    def test_gather_without_init_raises(self):
        coll = QuantumCollective(_mock_cluster())
        with pytest.raises(RuntimeError, match="Call init_accl"):
            coll.gather(np.array([1.0]))


class TestQuantumCollectiveScatter:
    def test_scatter(self):
        coll = QuantumCollective(_mock_cluster(3))
        coll.init_accl()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = coll.scatter(data, root=0)
        # _SoftwareACCL: chunk_size = 6 // 3 = 2 => first 2 elements
        np.testing.assert_array_equal(result, np.array([1.0, 2.0]))

    def test_scatter_none_data(self):
        coll = QuantumCollective(_mock_cluster(2))
        coll.init_accl()
        result = coll.scatter(None, root=0)
        assert len(result) == 0

    def test_scatter_without_init_raises(self):
        coll = QuantumCollective(_mock_cluster())
        with pytest.raises(RuntimeError, match="Call init_accl"):
            coll.scatter(np.array([1.0]))


class TestQuantumCollectiveMergeCounts:
    def test_merge_counts_single_source(self):
        """With software emulation, allgather_counts returns a single dict."""
        coll = QuantumCollective(_mock_cluster(2))
        coll.init_accl()

        counts = {"00": 250, "01": 130, "10": 120, "11": 500}
        merged = coll.merge_counts(counts)
        assert merged == counts

    def test_merge_counts_empty(self):
        coll = QuantumCollective(_mock_cluster())
        coll.init_accl()
        merged = coll.merge_counts({})
        assert merged == {}

    def test_merge_counts_accumulates(self):
        """Verify the merge logic sums overlapping bitstrings."""
        coll = QuantumCollective(_mock_cluster())
        coll.init_accl()

        # Patch allgather_counts to simulate multiple boards
        coll._accl.allgather_counts = lambda local: [
            {"00": 100, "11": 200},
            {"00": 50, "10": 75},
        ]

        merged = coll.merge_counts({"00": 100, "11": 200})
        assert merged["00"] == 150  # 100 + 50
        assert merged["11"] == 200
        assert merged["10"] == 75

    def test_merge_counts_without_init_raises(self):
        coll = QuantumCollective(_mock_cluster())
        with pytest.raises(RuntimeError, match="Call init_accl"):
            coll.merge_counts({"0": 10})


class TestQuantumCollectiveOperationsWithoutInit:
    """All collective operations must raise RuntimeError before init_accl()."""

    def setup_method(self):
        self.coll = QuantumCollective(_mock_cluster())

    def test_barrier_raises(self):
        with pytest.raises(RuntimeError, match="Call init_accl"):
            self.coll.barrier()

    def test_broadcast_raises(self):
        with pytest.raises(RuntimeError, match="Call init_accl"):
            self.coll.broadcast(np.array([1.0]))

    def test_allreduce_raises(self):
        with pytest.raises(RuntimeError, match="Call init_accl"):
            self.coll.allreduce(np.array([1.0]))

    def test_gather_raises(self):
        with pytest.raises(RuntimeError, match="Call init_accl"):
            self.coll.gather(np.array([1.0]))

    def test_scatter_raises(self):
        with pytest.raises(RuntimeError, match="Call init_accl"):
            self.coll.scatter(np.array([1.0]))

    def test_merge_counts_raises(self):
        with pytest.raises(RuntimeError, match="Call init_accl"):
            self.coll.merge_counts({"0": 1})
