"""Tests for the QICK backend (mocked — no hardware required)."""

from __future__ import annotations

import numpy as np
import pytest

from pynq_quantum.backends.qick import QICKBackend


class TestQICKBackendInit:
    def test_default_init(self):
        backend = QICKBackend()
        assert backend.name == "qick"
        assert backend.num_channels == 8
        assert not backend._connected

    def test_injected_soc(self):
        """Dependency injection path for testing."""
        fake_soc = {"gens": [1, 2, 3, 4], "readouts": [1, 2], "fs": 384.0}
        backend = QICKBackend(soc=fake_soc)
        assert backend._connected
        assert backend.num_channels == 4

    def test_connect_without_qick_raises(self):
        backend = QICKBackend()
        with pytest.raises(ImportError, match="qick"):
            backend.connect()


class TestQICKBackendOperations:
    @pytest.fixture
    def backend(self):
        fake_soc = {"gens": list(range(8)), "readouts": [0, 1], "fs": 384.0}
        b = QICKBackend(soc=fake_soc)
        yield b
        b.disconnect()

    def test_configure_qubit(self, backend):
        backend.configure_qubit(0, frequency=5e9, dac_channel=0)
        assert 0 in backend._qubit_config
        assert backend._qubit_config[0]["frequency"] == 5e9

    def test_disconnect(self, backend):
        assert backend._connected
        backend.disconnect()
        assert not backend._connected
        assert backend.num_channels == 8  # Falls back to default

    def test_execute_not_connected_raises(self):
        backend = QICKBackend()
        with pytest.raises(RuntimeError, match="not connected"):
            backend.execute([], [], shots=10)

    def test_get_capabilities(self, backend):
        caps = backend.get_capabilities()
        assert caps["backend"] == "qick"
        assert caps["simulation"] is False
        assert caps["num_dac_channels"] == 8
        assert caps["num_adc_channels"] == 2

    def test_iq_to_counts_3d(self):
        """Test IQ data thresholding with 3D array (readouts, quadratures, shots)."""
        iq = np.array(
            [
                [[0.5, -0.3, 0.1, -0.8], [0.1, 0.2, 0.3, 0.4]],
                [[-0.1, 0.2, 0.3, -0.4], [0.5, 0.6, 0.7, 0.8]],
            ]
        )
        counts, raw = QICKBackend._iq_to_counts(iq, [], 4)
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 4
        assert raw is not None

    def test_iq_to_counts_2d(self):
        iq = np.array([[0.5, -0.3, 0.1], [-0.1, 0.2, 0.3]])
        counts, raw = QICKBackend._iq_to_counts(iq, [], 3)
        assert sum(counts.values()) == 3

    def test_iq_to_counts_1d(self):
        iq = np.array([0.5, -0.3, 0.1, -0.8])
        counts, raw = QICKBackend._iq_to_counts(iq, [], 4)
        assert sum(counts.values()) == 4

    def test_iq_to_counts_empty(self):
        iq = np.array([])
        counts, raw = QICKBackend._iq_to_counts(iq, [], 10)
        assert counts == {"0": 10}
