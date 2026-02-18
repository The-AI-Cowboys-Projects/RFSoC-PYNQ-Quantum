"""Shared test fixtures for pynq-quantum."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pynq_quantum.backends.simulation import SimulationBackend
from pynq_quantum.overlay import QuantumOverlay


@pytest.fixture
def sim_backend():
    """A connected simulation backend with a fixed seed."""
    backend = SimulationBackend(num_qubits=8, seed=42)
    backend.connect()
    yield backend
    backend.disconnect()


@pytest.fixture
def sim_overlay():
    """A QuantumOverlay using the simulation backend."""
    overlay = QuantumOverlay(backend="simulation", num_qubits=8, seed=42)
    yield overlay
    overlay.close()


@pytest.fixture
def mock_mmio():
    """A mock MMIO object for the generic backend."""
    mmio = MagicMock()
    mmio.read.return_value = 0x01  # Done bit set
    return mmio
