"""Operational runtime entry points, loaded after package configuration."""

from importlib import import_module

__all__ = [
    "OperationalNamelist",
    "read_checkpoint",
    "read_checkpoint_with_runtime_state",
    "run_forecast_operational",
    "write_checkpoint",
]


def __getattr__(name):
    if name in {"OperationalNamelist", "run_forecast_operational"}:
        module = import_module(".operational_mode", __name__)
    elif name in {"read_checkpoint", "read_checkpoint_with_runtime_state", "write_checkpoint"}:
        module = import_module(".checkpoint", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value
