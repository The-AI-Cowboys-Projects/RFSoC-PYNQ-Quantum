"""Tests for the QubiC backend (mocked — no hardware required)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from pynq_quantum.backends.base import PulseInstruction, ReadoutInstruction
from pynq_quantum.backends.qubic import QubiCBackend


@pytest.fixture
def config_file(tmp_path):
    """Create a temporary QubiC config file."""
    config = {
        "channels": {
            "0": {"type": "drive", "frequency": 5e9},
            "1": {"type": "drive", "frequency": 5.1e9},
            "2": {"type": "readout", "frequency": 7e9},
        },
        "compiler": {},
        "max_shots": 100000,
        "sample_rate": 1e9,
    }
    path = tmp_path / "qubic_config.json"
    path.write_text(json.dumps(config))
    return str(path)


@pytest.fixture
def mock_compiler():
    compiler = MagicMock()
    compiler.compile.return_value = [{"instruction": "mock"}]
    return compiler


class TestQubiCBackendInit:
    def test_default_init(self):
        backend = QubiCBackend()
        assert backend.name == "qubic"
        assert backend.num_channels == 8  # Default when no config loaded
        assert not backend._connected

    def test_custom_config_path(self):
        backend = QubiCBackend(config_path="/some/path.json")
        assert backend._config_path == "/some/path.json"


class TestQubiCBackendConnect:
    def test_connect_with_injected_compiler(self, config_file, mock_compiler):
        backend = QubiCBackend(config_path=config_file, compiler=mock_compiler)
        backend.connect()
        assert backend._connected
        assert backend.num_channels == 3
        backend.disconnect()

    def test_connect_missing_config_raises(self):
        backend = QubiCBackend(config_path="/nonexistent/config.json")
        with pytest.raises(FileNotFoundError):
            backend.connect()

    def test_connect_without_qubic_raises(self, config_file):
        backend = QubiCBackend(config_path=config_file)
        with pytest.raises(ImportError, match="qubic"):
            backend.connect()

    def test_disconnect(self, config_file, mock_compiler):
        backend = QubiCBackend(config_path=config_file, compiler=mock_compiler)
        backend.connect()
        backend.disconnect()
        assert not backend._connected
        assert backend._config == {}


class TestQubiCBackendOperations:
    @pytest.fixture
    def backend(self, config_file, mock_compiler):
        b = QubiCBackend(config_path=config_file, compiler=mock_compiler)
        b.connect()
        yield b
        b.disconnect()

    def test_configure_qubit(self, backend):
        backend.configure_qubit(0, frequency=4.8e9, anharmonicity=-300e6)
        assert 0 in backend._qubit_config
        assert backend._qubit_config[0]["frequency"] == 4.8e9
        assert backend._channel_map[0]["frequency"] == 4.8e9

    def test_get_capabilities(self, backend):
        caps = backend.get_capabilities()
        assert caps["backend"] == "qubic"
        assert caps["num_channels"] == 3
        assert caps["simulation"] is False
        assert caps["max_shots"] == 100000

    def test_execute_not_connected_raises(self):
        backend = QubiCBackend()
        with pytest.raises(RuntimeError, match="not connected"):
            backend.execute([], [], shots=10)


class TestQubiCSequenceBuilding:
    def test_build_sequence(self):
        pulses = [
            PulseInstruction(
                channel=0,
                frequency=5e9,
                phase=0.0,
                amplitude=0.5,
                duration=40e-9,
                envelope="gaussian",
                envelope_params={"sigma": 10e-9},
            ),
        ]
        readouts = [
            ReadoutInstruction(channel=2, frequency=7e9, duration=1e-6, qubits=[0]),
        ]
        seq = QubiCBackend._build_sequence(pulses, readouts)
        assert len(seq) == 2
        assert seq[0]["type"] == "pulse"
        assert seq[0]["channel"] == 0
        assert seq[1]["type"] == "readout"
        assert seq[1]["qubits"] == [0]


class TestQubiCResultParsing:
    def test_parse_counts_dict(self):
        raw = {"counts": {"00": 500, "11": 500}}
        counts = QubiCBackend._parse_results(raw, [])
        assert counts == {"00": 500, "11": 500}

    def test_parse_2d_array(self):
        raw = np.array([[0.5, -0.3], [-0.1, 0.2], [0.3, 0.3]])
        counts = QubiCBackend._parse_results(raw, [])
        assert sum(counts.values()) == 3

    def test_parse_1d_array(self):
        raw = np.array([0.5, -0.3, 0.1, -0.8])
        counts = QubiCBackend._parse_results(raw, [])
        assert sum(counts.values()) == 4

    def test_parse_unknown_format(self):
        counts = QubiCBackend._parse_results("unexpected", [])
        assert counts == {"0": 1}

    def test_extract_raw_data_array(self):
        arr = np.array([1, 2, 3])
        result = QubiCBackend._extract_raw_data(arr)
        np.testing.assert_array_equal(result, arr)

    def test_extract_raw_data_dict(self):
        result = QubiCBackend._extract_raw_data({"raw": [1, 2, 3]})
        assert result is not None
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_extract_raw_data_none(self):
        assert QubiCBackend._extract_raw_data({}) is None

    def test_channel_map_from_list(self):
        config = {"channels": [{"freq": 5e9}, {"freq": 5.1e9}]}
        cmap = QubiCBackend._build_channel_map(config)
        assert 0 in cmap
        assert 1 in cmap
        assert cmap[0]["freq"] == 5e9
