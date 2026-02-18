"""Backend registry and auto-detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import AbstractBackend

_REGISTRY: dict[str, type[AbstractBackend]] = {}


def register_backend(name: str, cls: type[AbstractBackend]) -> None:
    """Register a backend class by name."""
    _REGISTRY[name] = cls


def get_backend(name: str) -> type[AbstractBackend]:
    """Retrieve a registered backend class by name."""
    if name not in _REGISTRY:
        _load_builtin(name)
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_backends() -> list[str]:
    """List all registered backend names."""
    _load_all_builtins()
    return sorted(_REGISTRY.keys())


def _load_builtin(name: str) -> None:
    """Lazy-load a built-in backend by name."""
    loaders = {
        "simulation": _load_simulation,
        "qick": _load_qick,
        "qubic": _load_qubic,
        "generic": _load_generic,
    }
    if name in loaders:
        loaders[name]()


def _load_all_builtins() -> None:
    for name in ("simulation", "qick", "qubic", "generic"):
        try:
            _load_builtin(name)
        except ImportError:
            pass


def _load_simulation() -> None:
    from .simulation import SimulationBackend

    register_backend("simulation", SimulationBackend)


def _load_qick() -> None:
    from .qick import QICKBackend

    register_backend("qick", QICKBackend)


def _load_qubic() -> None:
    from .qubic import QubiCBackend

    register_backend("qubic", QubiCBackend)


def _load_generic() -> None:
    from .generic import GenericBackend

    register_backend("generic", GenericBackend)
