"""Collective operations for multi-board quantum clusters via ACCL-Q."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class ReduceOp(Enum):
    """Reduction operations for allreduce."""

    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


class QuantumCollective:
    """ACCL-Q collective operations for distributed quantum computing.

    Provides MPI-style collective communication primitives adapted
    for quantum measurement data exchange between RFSoC boards.

    Args:
        cluster: A connected QuantumCluster instance.
    """

    def __init__(self, cluster: Any) -> None:
        self._cluster = cluster
        self._accl: Any = None
        self._rank = 0
        self._size = cluster.num_boards

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return int(self._size)

    def init_accl(self, **kwargs: Any) -> None:
        """Initialize ACCL communication layer.

        Attempts to use pyaccl if available, otherwise falls back
        to a software emulation layer using the cluster's overlays.
        """
        try:
            import pyaccl  # type: ignore[import-not-found]

            self._accl = pyaccl.ACCL(**kwargs)
        except ImportError:
            # Software emulation — no real ACCL hardware
            self._accl = _SoftwareACCL(self._cluster)

    def barrier(self) -> None:
        """Block until all boards reach this point."""
        if self._accl is None:
            raise RuntimeError("Call init_accl() first")
        self._accl.barrier()

    def broadcast(self, data: np.ndarray, root: int = 0) -> np.ndarray:
        """Broadcast data from root board to all boards.

        Args:
            data: Array to broadcast (only used on root).
            root: Index of the source board.

        Returns:
            The broadcast data on all boards.
        """
        if self._accl is None:
            raise RuntimeError("Call init_accl() first")
        result: np.ndarray = self._accl.broadcast(data, root=root)
        return result

    def allreduce(
        self,
        data: np.ndarray,
        op: ReduceOp = ReduceOp.SUM,
    ) -> np.ndarray:
        """Reduce data across all boards and distribute the result.

        Args:
            data: Local data array from this board.
            op: Reduction operation (SUM, MEAN, MAX, MIN).

        Returns:
            Reduced result array (identical on all boards).
        """
        if self._accl is None:
            raise RuntimeError("Call init_accl() first")
        result: np.ndarray = self._accl.allreduce(data, op=op)
        return result

    def gather(self, data: np.ndarray, root: int = 0) -> np.ndarray | None:
        """Gather data from all boards to root.

        Args:
            data: Local data array.
            root: Destination board index.

        Returns:
            Concatenated array on root, None on other boards.
        """
        if self._accl is None:
            raise RuntimeError("Call init_accl() first")
        result: np.ndarray | None = self._accl.gather(data, root=root)
        return result

    def scatter(self, data: np.ndarray | None, root: int = 0) -> np.ndarray:
        """Scatter chunks of data from root to all boards.

        Args:
            data: Full array on root (split evenly across boards).
            root: Source board index.

        Returns:
            Local chunk on each board.
        """
        if self._accl is None:
            raise RuntimeError("Call init_accl() first")
        result: np.ndarray = self._accl.scatter(data, root=root)
        return result

    def merge_counts(self, local_counts: dict[str, int]) -> dict[str, int]:
        """Merge measurement counts from all boards.

        Sums counts with matching bitstrings across all boards.
        """
        if self._accl is None:
            raise RuntimeError("Call init_accl() first")

        all_counts = self._accl.allgather_counts(local_counts)
        merged: dict[str, int] = {}
        for counts in all_counts:
            for bits, count in counts.items():
                merged[bits] = merged.get(bits, 0) + count
        return merged


class _SoftwareACCL:
    """Software emulation of ACCL for development/testing."""

    def __init__(self, cluster: Any) -> None:
        self._cluster = cluster
        self._data_store: list[np.ndarray] = []
        self._count_store: list[dict[str, int]] = []

    def barrier(self) -> None:
        pass  # No-op in single-process emulation

    def broadcast(self, data: np.ndarray, root: int = 0) -> np.ndarray:
        return np.array(data, copy=True)

    def allreduce(self, data: np.ndarray, op: ReduceOp = ReduceOp.SUM) -> np.ndarray:
        # In emulation, we only have local data
        return np.array(data, copy=True)

    def gather(self, data: np.ndarray, root: int = 0) -> np.ndarray:
        return np.array(data, copy=True)

    def scatter(self, data: np.ndarray | None, root: int = 0) -> np.ndarray:
        if data is None:
            return np.array([])
        n: int = self._cluster.num_boards
        chunk_size = len(data) // max(n, 1)
        return np.array(data[:chunk_size], copy=True)

    def allgather_counts(self, local_counts: dict[str, int]) -> list[dict[str, int]]:
        return [local_counts]
