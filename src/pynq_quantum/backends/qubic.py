"""QubiC backend for RFSoC quantum control.

Wraps the QubiC compiler and JSON configuration system to drive
RFSoC hardware through the QubiC toolchain.  QubiC is an optional
dependency -- the module imports it lazily so the rest of the
framework can load without it installed.

Typical usage::

    backend = QubiCBackend(config_path="my_rfsoc.json")
    backend.connect()
    backend.configure_qubit(0, frequency=5.0e9, anharmonicity=-300e6)
    result = backend.execute(pulses, readouts, shots=2000)
    backend.disconnect()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import (
    AbstractBackend,
    ExecutionResult,
    PulseInstruction,
    ReadoutInstruction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of the optional ``qubic`` package
# ---------------------------------------------------------------------------
_qubic: Any | None = None
_QUBIC_IMPORT_ERROR: str | None = None


def _ensure_qubic() -> Any:
    """Return the ``qubic`` module, raising a clear error if unavailable."""
    global _qubic, _QUBIC_IMPORT_ERROR  # noqa: PLW0603

    if _qubic is not None:
        return _qubic

    try:
        import qubic  # type: ignore[import-untyped]

        _qubic = qubic
        return _qubic
    except ImportError as exc:
        _QUBIC_IMPORT_ERROR = str(exc)
        raise ImportError(
            "The 'qubic' package is required for QubiCBackend but is not "
            "installed.  Install it with:  pip install qubic\n"
            f"Original error: {exc}"
        ) from exc


def _is_qubic_available() -> bool:
    """Check whether the ``qubic`` package can be imported."""
    try:
        _ensure_qubic()
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# QubiC backend
# ---------------------------------------------------------------------------


class QubiCBackend(AbstractBackend):
    """Quantum control backend using the QubiC compiler and JSON config.

    Parameters
    ----------
    config_path:
        Filesystem path to a QubiC JSON configuration file that describes
        the channel map, DAC/ADC wiring, and default pulse parameters.
    compiler:
        Optional pre-built QubiC compiler instance.  When *None* (the
        default), a compiler is created from the configuration during
        :meth:`connect`.  Passing a compiler is useful for dependency
        injection in tests.
    """

    def __init__(
        self,
        config_path: str = "qubic_config.json",
        compiler: Any | None = None,
    ) -> None:
        self._config_path: str = config_path
        self._config: dict[str, Any] = {}
        self._compiler: Any | None = compiler
        self._connected: bool = False
        self._qubit_config: dict[int, dict[str, Any]] = {}
        self._channel_map: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        """Human-readable backend name."""
        return "qubic"

    @property
    def num_channels(self) -> int:
        """Number of available DAC channels from the loaded config."""
        if self._config:
            channels = self._config.get("channels", {})
            if isinstance(channels, dict):
                return len(channels) or self._config.get("num_channels", 8)
            if isinstance(channels, list):
                return len(channels) or self._config.get("num_channels", 8)
        return 8

    # ------------------------------------------------------------------
    # AbstractBackend interface
    # ------------------------------------------------------------------

    def connect(self, **kwargs: Any) -> None:
        """Load the QubiC JSON config and initialise the compiler.

        The configuration file is expected to contain at minimum:

        * ``channels`` -- a mapping of channel indices to DAC/ADC settings.
        * ``compiler`` (optional) -- compiler-specific tunables.

        Any extra *kwargs* are forwarded to the QubiC compiler constructor.

        Raises
        ------
        FileNotFoundError
            If *config_path* does not point to an existing file.
        ImportError
            If the ``qubic`` package is not installed.
        """
        config_file = Path(self._config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"QubiC configuration file not found: {config_file.resolve()}")

        with open(config_file, "r", encoding="utf-8") as fh:
            self._config = json.load(fh)

        logger.info("Loaded QubiC config from %s", config_file)

        # Build the internal channel map from the config.
        self._channel_map = self._build_channel_map(self._config)

        # Create the compiler (unless one was injected via the constructor).
        if self._compiler is None:
            qubic = _ensure_qubic()
            compiler_cfg = {
                **self._config.get("compiler", {}),
                **kwargs,
            }
            self._compiler = qubic.Compiler(self._config, **compiler_cfg)
            logger.info("QubiC compiler initialised")

        self._connected = True

    def disconnect(self) -> None:
        """Release the QubiC compiler and clear loaded state."""
        if self._compiler is not None:
            # If the compiler exposes a cleanup hook, call it.
            cleanup = getattr(self._compiler, "close", None)
            if callable(cleanup):
                cleanup()
            self._compiler = None

        self._config = {}
        self._channel_map = {}
        self._connected = False
        logger.info("QubiC backend disconnected")

    def configure_qubit(self, qubit_id: int, frequency: float, **params: Any) -> None:
        """Store qubit parameters and update the QubiC channel map.

        Parameters
        ----------
        qubit_id:
            Logical qubit index.
        frequency:
            Qubit drive frequency in Hz.
        **params:
            Additional parameters (``anharmonicity``, ``t1``, ``t2``, etc.)
            forwarded to the channel map entry.
        """
        self._qubit_config[qubit_id] = {"frequency": frequency, **params}

        # Mirror into the channel map so the compiler sees the new freq.
        if qubit_id in self._channel_map:
            self._channel_map[qubit_id]["frequency"] = frequency
            self._channel_map[qubit_id].update(params)
        else:
            self._channel_map[qubit_id] = {"frequency": frequency, **params}

        # Push updated channel map into the live compiler if available.
        if self._compiler is not None:
            updater = getattr(self._compiler, "update_channel_map", None)
            if callable(updater):
                updater(self._channel_map)
            logger.debug(
                "Updated QubiC channel map for qubit %d (%.3f GHz)",
                qubit_id,
                frequency / 1e9,
            )

    def execute(
        self,
        pulses: list[PulseInstruction],
        readouts: list[ReadoutInstruction],
        shots: int = 1000,
    ) -> ExecutionResult:
        """Compile and execute a pulse schedule on the QubiC hardware.

        The method performs four stages:

        1. **Convert** -- translate :class:`PulseInstruction` and
           :class:`ReadoutInstruction` objects into the QubiC sequence
           format (list of dicts).
        2. **Compile** -- invoke the QubiC compiler to produce low-level
           FPGA instructions.
        3. **Upload & Run** -- send the compiled program to the RFSoC and
           acquire *shots* measurements.
        4. **Parse** -- convert raw ADC data into bitstring counts.

        Raises
        ------
        RuntimeError
            If the backend is not connected.
        """
        if not self._connected:
            raise RuntimeError("QubiC backend is not connected. Call connect() first.")

        qubic = _ensure_qubic()

        # 1. Build QubiC sequence ------------------------------------------
        sequence = self._build_sequence(pulses, readouts)

        # 2. Compile -------------------------------------------------------
        compiled = self._compiler.compile(sequence)
        logger.debug("QubiC compilation produced %d instructions", len(compiled))

        # 3. Upload & run --------------------------------------------------
        raw_results = qubic.run(compiled, shots=shots)

        # 4. Parse results -------------------------------------------------
        counts = self._parse_results(raw_results, readouts)
        raw_data = self._extract_raw_data(raw_results)

        return ExecutionResult(
            counts=counts,
            raw_data=raw_data,
            metadata={
                "backend": self.name,
                "shots": shots,
                "num_pulses": len(pulses),
                "num_readouts": len(readouts),
                "config_path": self._config_path,
            },
        )

    def get_capabilities(self) -> dict[str, Any]:
        """Return channel and configuration information.

        The returned dict always contains ``num_channels`` and
        ``channels``; when a configuration file has been loaded it also
        includes any extra metadata present in the config.
        """
        capabilities: dict[str, Any] = {
            "backend": self.name,
            "num_channels": self.num_channels,
            "channels": dict(self._channel_map),
            "simulation": False,
        }

        # Forward hardware-level caps from the config if present.
        for key in ("max_shots", "sample_rate", "dac_resolution", "adc_resolution"):
            if key in self._config:
                capabilities[key] = self._config[key]

        return capabilities

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_channel_map(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
        """Normalise the ``channels`` section of a QubiC config.

        Accepts both dict-keyed (``{"0": {...}}``) and list-based
        (``[{...}, ...]``) channel definitions and returns a dict
        mapping integer channel indices to their settings.
        """
        raw = config.get("channels", {})
        channel_map: dict[int, dict[str, Any]] = {}

        if isinstance(raw, dict):
            for key, value in raw.items():
                channel_map[int(key)] = dict(value) if isinstance(value, dict) else {}
        elif isinstance(raw, list):
            for idx, entry in enumerate(raw):
                channel_map[idx] = dict(entry) if isinstance(entry, dict) else {}

        return channel_map

    @staticmethod
    def _build_sequence(
        pulses: list[PulseInstruction],
        readouts: list[ReadoutInstruction],
    ) -> list[dict[str, Any]]:
        """Convert framework pulse/readout objects to QubiC sequence dicts.

        Each pulse becomes a dict with keys expected by the QubiC
        compiler: ``type``, ``channel``, ``freq``, ``phase``,
        ``amplitude``, ``duration``, ``envelope``, and optional
        ``envelope_params``.

        Readout instructions become entries with ``type: "readout"``.
        """
        sequence: list[dict[str, Any]] = []

        for pulse in pulses:
            entry: dict[str, Any] = {
                "type": "pulse",
                "channel": pulse.channel,
                "freq": pulse.frequency,
                "phase": pulse.phase,
                "amplitude": pulse.amplitude,
                "duration": pulse.duration,
                "envelope": pulse.envelope,
            }
            if pulse.envelope_params:
                entry["envelope_params"] = dict(pulse.envelope_params)
            sequence.append(entry)

        for readout in readouts:
            sequence.append(
                {
                    "type": "readout",
                    "channel": readout.channel,
                    "freq": readout.frequency,
                    "duration": readout.duration,
                    "qubits": list(readout.qubits),
                }
            )

        return sequence

    @staticmethod
    def _parse_results(
        raw_results: Any,
        readouts: list[ReadoutInstruction],
    ) -> dict[str, int]:
        """Convert raw QubiC acquisition data to bitstring counts.

        The QubiC runner may return results in several shapes:

        * A ``dict`` with a ``counts`` key -- used directly.
        * A ``dict`` with per-channel arrays -- thresholded into bits.
        * A ``numpy.ndarray`` of shape ``(shots, n_readouts)`` --
          thresholded per column.
        * Anything else -- returns a fallback ``{"0": total_shots}``
          count.
        """
        # Already has counts
        if isinstance(raw_results, dict) and "counts" in raw_results:
            return {str(k): int(v) for k, v in raw_results["counts"].items()}

        # Per-channel dict of arrays
        if isinstance(raw_results, dict):
            return _threshold_channel_dict(raw_results, readouts)

        # 2-D numpy array (shots x readout_channels)
        if isinstance(raw_results, np.ndarray) and raw_results.ndim == 2:
            return _threshold_array(raw_results)

        # 1-D numpy array (single readout channel)
        if isinstance(raw_results, np.ndarray) and raw_results.ndim == 1:
            return _threshold_array(raw_results.reshape(-1, 1))

        logger.warning(
            "Unrecognised QubiC result format (%s); returning fallback counts",
            type(raw_results).__name__,
        )
        return {"0": 1}

    @staticmethod
    def _extract_raw_data(raw_results: Any) -> np.ndarray | None:
        """Pull a raw numpy array out of QubiC results when available."""
        if isinstance(raw_results, np.ndarray):
            return raw_results
        if isinstance(raw_results, dict):
            if "raw" in raw_results:
                arr = raw_results["raw"]
                return np.asarray(arr) if not isinstance(arr, np.ndarray) else arr
            if "iq_data" in raw_results:
                arr = raw_results["iq_data"]
                return np.asarray(arr) if not isinstance(arr, np.ndarray) else arr
        return None


# ---------------------------------------------------------------------------
# Module-level result parsing helpers
# ---------------------------------------------------------------------------


def _threshold_channel_dict(
    data: dict[str, Any],
    readouts: list[ReadoutInstruction],
) -> dict[str, int]:
    """Threshold per-channel arrays into bitstring counts."""
    # Determine which channels carry readout data.
    ro_channels = sorted({ro.channel for ro in readouts}) if readouts else []
    if not ro_channels:
        ro_channels = sorted(
            k
            for k, v in data.items()
            if isinstance(v, (list, np.ndarray)) and k != "raw" and k != "iq_data"
        )

    if not ro_channels:
        return {"0": 1}

    # Stack arrays column-wise: (shots, n_readouts)
    arrays = []
    for ch in ro_channels:
        arr = data.get(ch) if not isinstance(ch, int) else data.get(str(ch), data.get(ch))
        if arr is None:
            continue
        arrays.append(np.asarray(arr).ravel())

    if not arrays:
        return {"0": 1}

    # Ensure equal length (pad shorter with 0)
    max_len = max(len(a) for a in arrays)
    padded = [np.pad(a, (0, max_len - len(a))) if len(a) < max_len else a for a in arrays]
    matrix = np.column_stack(padded)
    return _threshold_array(matrix)


def _threshold_array(matrix: np.ndarray) -> dict[str, int]:
    """Threshold a (shots, n_readouts) array into bitstring counts.

    Values above zero are mapped to ``'1'``; at or below zero to ``'0'``.
    """
    bits = (matrix > 0).astype(int)
    counts: dict[str, int] = {}
    for row in bits:
        key = "".join(str(b) for b in row)
        counts[key] = counts.get(key, 0) + 1
    return counts
