"""QICK backend for RFSoC quantum control.

Wraps the QICK library (https://github.com/openquantumhardware/qick) to drive
DAC/ADC channels on Xilinx RFSoC FPGAs.  The ``qick`` package is an optional
dependency -- if it is not installed the module can still be imported, but
:meth:`connect` will raise a clear error directing the user to install it.

Typical usage::

    backend = QICKBackend()
    backend.connect()
    backend.configure_qubit(0, frequency=5.0e9, dac_channel=0, adc_channel=0)
    result = backend.execute(pulses, readouts, shots=1024)
    backend.disconnect()
"""

from __future__ import annotations

import logging
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
# Lazy import of the qick library.  We defer the import so that this module
# can be loaded on machines where qick is not installed (CI, laptops, etc.).
# ---------------------------------------------------------------------------
_qick = None  # module reference, populated by _ensure_qick()
_QICK_IMPORT_ERROR: str | None = None

try:
    import qick as _qick  # type: ignore[import-untyped,no-redef]
except ImportError as exc:
    _QICK_IMPORT_ERROR = (
        f"The 'qick' package is not installed ({exc}). "
        "Install it with:  pip install qick  "
        "or see https://github.com/openquantumhardware/qick for details."
    )


def _ensure_qick() -> Any:
    """Return the ``qick`` module or raise a helpful :class:`ImportError`."""
    if _qick is None:
        raise ImportError(_QICK_IMPORT_ERROR)
    return _qick


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------

_DEFAULT_SIGMA_FRAC = 4  # Gaussian sigma = duration / _DEFAULT_SIGMA_FRAC


def _cycles_from_seconds(duration_s: float, fs_mhz: float) -> int:
    """Convert a duration in seconds to the nearest number of fabric cycles.

    Parameters
    ----------
    duration_s:
        Pulse/readout duration in seconds.
    fs_mhz:
        Fabric clock frequency in MHz (``soccfg['fs']``).

    Returns
    -------
    int
        Number of 16-sample fabric clock cycles (rounded to nearest).
    """
    samples = duration_s * fs_mhz * 1e6  # seconds -> samples
    cycles = int(round(samples / 16))  # 16 samples per fabric clk
    return max(cycles, 1)


def _freq_hz_to_reg(freq_hz: float, fs_mhz: float, b_freq: int = 32) -> int:
    """Convert a frequency in Hz to a QICK frequency register value.

    Parameters
    ----------
    freq_hz:
        Target frequency in Hz.
    fs_mhz:
        DAC/ADC sampling frequency in MHz.
    b_freq:
        Bit-width of the frequency register (default 32).
    """
    return int(round(freq_hz / (fs_mhz * 1e6) * (2**b_freq)))


# ---------------------------------------------------------------------------
# QICKBackend
# ---------------------------------------------------------------------------


class QICKBackend(AbstractBackend):
    """Hardware backend using the QICK firmware on Xilinx RFSoC boards.

    Parameters
    ----------
    soc:
        An existing :class:`qick.QickSoc` instance for dependency injection
        (e.g. in tests).  When *None* (the default), a new ``QickSoc`` is
        created inside :meth:`connect`.
    """

    def __init__(self, soc: Any | None = None) -> None:
        self._soc: Any | None = soc
        self._soccfg: Any | None = None
        self._connected: bool = False
        self._qubit_config: dict[int, dict[str, Any]] = {}

        # If a pre-built soc was injected, treat as already connected.
        if soc is not None:
            self._soccfg = soc
            self._connected = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D401
        """Human-readable backend name."""
        return "qick"

    @property
    def num_channels(self) -> int:
        """Number of available DAC channels reported by the firmware.

        Falls back to 8 when no SoC is connected.
        """
        if self._soccfg is not None:
            try:
                # soccfg exposes DAC channel count via ``['gens']`` list.
                return len(self._soccfg["gens"])
            except (TypeError, KeyError):
                pass
        return 8

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, **kwargs: Any) -> None:
        """Open a connection to the RFSoC hardware.

        If no ``soc`` was injected via the constructor a new
        :class:`qick.QickSoc` is created.  Extra *kwargs* are forwarded to
        the ``QickSoc`` constructor (e.g. ``bitfile``).

        Raises
        ------
        ImportError
            If the ``qick`` package is not installed.
        RuntimeError
            If the SoC cannot be initialised.
        """
        if self._connected:
            logger.info("QICK backend already connected.")
            return

        qick_mod = _ensure_qick()

        try:
            soc = qick_mod.QickSoc(**kwargs)  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise QickSoc: {exc}"
            ) from exc

        self._soc = soc
        self._soccfg = soc
        self._connected = True
        logger.info("QICK backend connected (%d DAC channels).", self.num_channels)

    def disconnect(self) -> None:
        """Release the SoC resources and mark the backend as disconnected."""
        if self._soc is not None:
            # QickSoc does not expose a formal close() -- release our refs.
            logger.info("Disconnecting QICK backend.")
        self._soc = None
        self._soccfg = None
        self._connected = False
        self._qubit_config.clear()

    # ------------------------------------------------------------------
    # Qubit configuration
    # ------------------------------------------------------------------

    def configure_qubit(
        self,
        qubit_id: int,
        frequency: float,
        **params: Any,
    ) -> None:
        """Register a qubit's drive parameters.

        Parameters
        ----------
        qubit_id:
            Logical qubit index (0-based).
        frequency:
            Qubit drive frequency in Hz.
        **params:
            Optional keys understood by the backend:

            * ``dac_channel`` (int) -- DAC generator channel for qubit drive.
            * ``adc_channel`` (int) -- ADC readout channel.
            * ``anharmonicity`` (float) -- anharmonicity in Hz (for DRAG).
            * ``readout_frequency`` (float) -- readout resonator frequency in Hz.
            * ``gain`` (float) -- default pulse gain 0.0--1.0.
        """
        self._qubit_config[qubit_id] = {"frequency": frequency, **params}
        logger.debug(
            "Qubit %d configured: freq=%.6g Hz, extras=%s",
            qubit_id,
            frequency,
            params,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        pulses: list[PulseInstruction],
        readouts: list[ReadoutInstruction],
        shots: int = 1000,
    ) -> ExecutionResult:
        """Compile and run a pulse schedule on the RFSoC hardware.

        Internally builds a :class:`_QICKProgram` (an
        :class:`~qick.AveragerProgram` subclass), loads pulses and readout
        windows, then calls ``acquire()``.

        Parameters
        ----------
        pulses:
            Drive pulses to emit.
        readouts:
            Readout acquisitions to perform.
        shots:
            Number of repetitions (averages).

        Returns
        -------
        ExecutionResult
            Measurement counts and optional raw IQ data.

        Raises
        ------
        RuntimeError
            If the backend is not connected.
        """
        if not self._connected or self._soc is None:
            raise RuntimeError("Backend not connected. Call connect() first.")

        qick_mod = _ensure_qick()

        # Build the program configuration dict expected by AveragerProgram.
        prog_cfg = self._build_program_config(pulses, readouts, shots)

        # Instantiate the inner program class.
        prog = _QICKProgram(
            soccfg=self._soccfg,
            cfg=prog_cfg,
            qick_mod=qick_mod,
            pulses=pulses,
            readouts=readouts,
        )

        # Run acquisition.  acquire() returns arrays of shape
        # (num_readouts, shots) for I and Q.
        try:
            iq_data = prog.acquire(  # type: ignore[union-attr]
                self._soc,
                readouts_per_experiment=len(readouts) or 1,
                load_pulses=True,
            )
        except Exception as exc:
            raise RuntimeError(f"QICK acquire failed: {exc}") from exc

        # Convert IQ data to measurement counts via thresholding.
        counts, raw = self._iq_to_counts(iq_data, readouts, shots)

        return ExecutionResult(
            counts=counts,
            raw_data=raw,
            metadata={
                "backend": "qick",
                "shots": shots,
                "num_pulses": len(pulses),
                "num_readouts": len(readouts),
            },
        )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def get_capabilities(self) -> dict[str, Any]:
        """Return hardware capabilities derived from the SoC configuration."""
        caps: dict[str, Any] = {
            "backend": "qick",
            "simulation": False,
            "num_dac_channels": self.num_channels,
        }

        if self._soccfg is not None:
            try:
                caps["num_adc_channels"] = len(self._soccfg["readouts"])
            except (TypeError, KeyError):
                pass
            try:
                caps["fabric_clk_mhz"] = float(self._soccfg["fs"])
            except (TypeError, KeyError):
                pass

        caps["supported_envelopes"] = [
            "gaussian",
            "square",
            "drag",
            "flat_top",
        ]
        caps["configured_qubits"] = list(self._qubit_config.keys())
        return caps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_program_config(
        self,
        pulses: list[PulseInstruction],
        readouts: list[ReadoutInstruction],
        shots: int,
    ) -> dict[str, Any]:
        """Create the ``cfg`` dict consumed by :class:`_QICKProgram`."""
        fs_mhz = 384.0  # default fabric clock; overridden if soccfg exists
        if self._soccfg is not None:
            try:
                fs_mhz = float(self._soccfg["fs"])
            except (TypeError, KeyError):
                pass

        pulse_cfgs = []
        for p in pulses:
            length = _cycles_from_seconds(p.duration, fs_mhz)
            sigma = max(length // _DEFAULT_SIGMA_FRAC, 1)
            if "sigma" in p.envelope_params:
                sigma = int(p.envelope_params["sigma"])
            pulse_cfgs.append(
                {
                    "channel": p.channel,
                    "freq_reg": _freq_hz_to_reg(p.frequency, fs_mhz),
                    "phase_deg": float(np.degrees(p.phase)),
                    "gain": int(p.amplitude * 32767),  # 15-bit signed DAC
                    "length": length,
                    "sigma": sigma,
                    "envelope": p.envelope,
                    "envelope_params": p.envelope_params,
                }
            )

        readout_cfgs = []
        for ro in readouts:
            readout_cfgs.append(
                {
                    "channel": ro.channel,
                    "freq_reg": _freq_hz_to_reg(ro.frequency, fs_mhz),
                    "length": _cycles_from_seconds(ro.duration, fs_mhz),
                    "qubits": ro.qubits,
                }
            )

        return {
            "reps": shots,
            "pulses": pulse_cfgs,
            "readouts": readout_cfgs,
            "fs_mhz": fs_mhz,
        }

    @staticmethod
    def _iq_to_counts(
        iq_data: Any,
        readouts: list[ReadoutInstruction],
        shots: int,
    ) -> tuple[dict[str, int], np.ndarray | None]:
        """Threshold IQ data into binary counts.

        A simple threshold is applied: positive I values map to ``|1>``,
        negative to ``|0>``.  For multi-qubit readouts the per-readout
        bits are concatenated.

        Returns
        -------
        counts:
            Bitstring histogram, e.g. ``{'00': 512, '11': 488}``.
        raw:
            The raw IQ numpy array (or *None* if unavailable).
        """
        try:
            iq_arr = np.asarray(iq_data)
        except Exception:
            # Fallback when data is not array-like.
            return {"0": shots}, None

        if iq_arr.ndim == 0 or iq_arr.size == 0:
            return {"0": shots}, None

        # iq_arr shape is typically (num_readouts, 2, shots) or
        # (num_readouts, shots) for just I.  Normalise to (n_ro, shots).
        if iq_arr.ndim == 3:
            # Take I quadrature (index 0).
            i_data = iq_arr[:, 0, :]
        elif iq_arr.ndim == 2:
            i_data = iq_arr
        elif iq_arr.ndim == 1:
            i_data = iq_arr.reshape(1, -1)
        else:
            return {"0": shots}, iq_arr

        n_ro = i_data.shape[0]
        n_shots = i_data.shape[1]

        # Threshold: I >= 0 -> "1", I < 0 -> "0"
        bits = (i_data >= 0).astype(int)  # shape (n_ro, n_shots)

        counts: dict[str, int] = {}
        for s in range(n_shots):
            bitstring = "".join(str(bits[ro, s]) for ro in range(n_ro))
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts, iq_arr


# ---------------------------------------------------------------------------
# Inner AveragerProgram subclass
# ---------------------------------------------------------------------------


class _QICKProgram:
    """Thin wrapper that builds a QICK ``AveragerProgram`` at runtime.

    We do *not* inherit from ``AveragerProgram`` at class-definition time
    because the ``qick`` module may not be installed.  Instead we dynamically
    create the real program instance inside :meth:`acquire`.
    """

    def __init__(
        self,
        soccfg: Any,
        cfg: dict[str, Any],
        qick_mod: Any,
        pulses: list[PulseInstruction],
        readouts: list[ReadoutInstruction],
    ) -> None:
        self._soccfg = soccfg
        self._cfg = cfg
        self._qick_mod = qick_mod
        self._pulses = pulses
        self._readouts = readouts

    # ------------------------------------------------------------------
    # Dynamic program construction
    # ------------------------------------------------------------------

    def _make_program(self) -> Any:
        """Build and return a concrete ``AveragerProgram`` instance."""
        qick_mod = self._qick_mod
        soccfg = self._soccfg
        cfg = self._cfg

        class _Program(qick_mod.AveragerProgram):  # type: ignore[misc,name-defined]
            """Dynamically generated AveragerProgram for a single schedule."""

            def initialize(self) -> None:  # noqa: D401 -- QICK API name
                """Called once before the acquisition loop."""
                pcfg = self.cfg

                # Declare generator channels.
                declared_gens: set[int] = set()
                for p in pcfg["pulses"]:
                    ch = p["channel"]
                    if ch in declared_gens:
                        continue
                    self.declare_gen(ch=ch, nqz=1)
                    declared_gens.add(ch)

                # Declare readout channels.
                declared_ros: set[int] = set()
                for ro in pcfg["readouts"]:
                    ch = ro["channel"]
                    if ch in declared_ros:
                        continue
                    self.declare_readout(
                        ch=ch,
                        freq=ro["freq_reg"],
                        length=ro["length"],
                    )
                    declared_ros.add(ch)

                # Configure pulses.
                for p in pcfg["pulses"]:
                    style = _envelope_to_style(p["envelope"])
                    kwargs: dict[str, Any] = {
                        "ch": p["channel"],
                        "freq": p["freq_reg"],
                        "phase": self.deg2reg(p["phase_deg"], gen_ch=p["channel"]),
                        "gain": p["gain"],
                        "length": p["length"],
                        "style": style,
                    }
                    if style == "arb":
                        sigma = p["sigma"]
                        idata = _build_envelope(p["envelope"], sigma, p["envelope_params"])
                        self.add_pulse(ch=p["channel"], name=f"pulse_{p['channel']}", idata=idata)
                        kwargs["waveform"] = f"pulse_{p['channel']}"
                    self.set_pulse_registers(**kwargs)

                self.synci(200)  # sync all channels

            def body(self) -> None:
                """Executed on every shot."""
                pcfg = self.cfg
                # Fire all pulses.
                fired: set[int] = set()
                for p in pcfg["pulses"]:
                    ch = p["channel"]
                    if ch not in fired:
                        self.pulse(ch=ch)
                        fired.add(ch)

                # Trigger readouts.
                for ro in pcfg["readouts"]:
                    self.trigger(
                        adcs=[ro["channel"]],
                        adc_trig_offset=ro["length"],
                    )

                # Wait for readout to finish before next rep.
                max_len = max(
                    (ro["length"] for ro in pcfg["readouts"]),
                    default=10,
                )
                self.waiti(0, max_len + 100)

        return _Program(soccfg, cfg)

    def acquire(self, soc: Any, **kwargs: Any) -> Any:
        """Build the hardware program and run ``acquire()`` on the SoC.

        Parameters
        ----------
        soc:
            The ``QickSoc`` instance.
        **kwargs:
            Forwarded to ``AveragerProgram.acquire()``.

        Returns
        -------
        numpy.ndarray
            Raw IQ data from the acquisition.
        """
        prog = self._make_program()
        return prog.acquire(soc, **kwargs)


# ---------------------------------------------------------------------------
# Envelope utilities
# ---------------------------------------------------------------------------


def _envelope_to_style(envelope: str) -> str:
    """Map a :class:`PulseInstruction` envelope name to a QICK pulse style.

    QICK supports ``'const'`` (square), ``'arb'`` (arbitrary waveform),
    and ``'flat_top'``.
    """
    mapping = {
        "square": "const",
        "gaussian": "arb",
        "drag": "arb",
        "flat_top": "flat_top",
    }
    return mapping.get(envelope, "arb")


def _build_envelope(
    envelope: str,
    sigma: int,
    params: dict[str, Any],
) -> np.ndarray:
    """Generate an envelope waveform array for an arbitrary pulse.

    Parameters
    ----------
    envelope:
        One of ``'gaussian'``, ``'drag'``.
    sigma:
        Width parameter in fabric clock cycles.
    params:
        Extra parameters (e.g. ``delta`` for DRAG).

    Returns
    -------
    numpy.ndarray
        Integer waveform samples scaled to 15-bit signed range.
    """
    length = sigma * _DEFAULT_SIGMA_FRAC
    t = np.linspace(-length / 2, length / 2, length, endpoint=False)

    if envelope == "gaussian":
        waveform = np.exp(-0.5 * (t / sigma) ** 2)
    elif envelope == "drag":
        gauss = np.exp(-0.5 * (t / sigma) ** 2)
        delta = params.get("delta", params.get("anharmonicity", -200e6))
        # DRAG correction: derivative scaled by 1/delta.
        deriv = -(t / sigma**2) * gauss
        drag_scale = params.get("drag_scale", 1.0)
        waveform = gauss + 1j * drag_scale * deriv / delta
        # Return real part scaled; imaginary part would go to Q channel.
        waveform = np.real(waveform)
    else:
        # Default to Gaussian.
        waveform = np.exp(-0.5 * (t / sigma) ** 2)

    # Normalise to [-1, 1] then scale to 15-bit signed int range.
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    return np.round(waveform * 32767).astype(np.int16)
