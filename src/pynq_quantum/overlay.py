"""QuantumOverlay — board detection and backend loading."""

from __future__ import annotations

import os
from typing import Any

from .backends import get_backend
from .backends.base import AbstractBackend

# Known RFSoC board identifiers (from /proc/device-tree or PYNQ)
BOARD_MAP = {
    "ZCU111": "qick",
    "ZCU216": "qick",
    "RFSoC4x2": "qick",
    "RFSoC2x2": "qick",
    "HTG-ZRF8": "qick",
}


class QuantumOverlay:
    """Entry point for quantum control on RFSoC.

    Handles board auto-detection, backend loading, and optional
    bitstream management via PYNQ.

    Args:
        backend: Backend name ('simulation', 'qick', 'qubic', 'generic')
                 or 'auto' for hardware auto-detection.
        bitstream: Path to .bit file (loaded via pynq.Overlay if provided).
        **kwargs: Passed to the backend constructor.
    """

    def __init__(
        self,
        backend: str = "simulation",
        bitstream: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._bitstream = bitstream
        self._pynq_overlay: Any = None
        self._kwargs = kwargs

        if backend == "auto":
            detected = self.detect_board()
            if detected and detected in BOARD_MAP:
                backend = BOARD_MAP[detected]
            else:
                backend = "simulation"

        backend_cls = get_backend(backend)
        self._backend: AbstractBackend = backend_cls(**kwargs)
        self._backend.connect()

        if bitstream:
            self._load_bitstream(bitstream)

    @property
    def backend(self) -> AbstractBackend:
        """The active backend instance."""
        return self._backend

    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._backend.name

    def detect_board(self) -> str | None:
        """Detect the RFSoC board model.

        Checks (in order):
        1. PYNQ_BOARD environment variable
        2. /proc/device-tree/model (Linux device tree)
        3. pynq.Device if pynq is importable

        Returns:
            Board name string or None if not on RFSoC hardware.
        """
        # 1. Environment variable
        board = os.environ.get("PYNQ_BOARD")
        if board:
            return board

        # 2. Device tree
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().strip()
            for board_name in BOARD_MAP:
                if board_name.lower() in model.lower():
                    return board_name
        except (FileNotFoundError, PermissionError):
            pass

        # 3. PYNQ library
        try:
            from pynq import Device  # type: ignore[import-not-found]

            devices = Device.devices
            if devices:
                return devices[0].name
        except (ImportError, Exception):
            pass

        return None

    def _load_bitstream(self, path: str) -> None:
        """Load a bitstream file using PYNQ."""
        try:
            from pynq import Overlay  # type: ignore[import-not-found]

            self._pynq_overlay = Overlay(path)
        except ImportError:
            raise ImportError(
                "pynq package required for bitstream loading. "
                "Install with: pip install pynq-quantum[pynq]"
            )

    def close(self) -> None:
        """Disconnect backend and free resources."""
        self._backend.disconnect()
        if self._pynq_overlay is not None:
            try:
                self._pynq_overlay.free()
            except Exception:
                pass

    def __enter__(self) -> QuantumOverlay:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
