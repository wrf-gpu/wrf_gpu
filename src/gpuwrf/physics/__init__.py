"""Column physics kernels for M5 and later coupling work."""

__all__ = ["ThompsonColumnState", "step_thompson_column"]


def __getattr__(name: str):
    if name in __all__:
        from .thompson_column import ThompsonColumnState, step_thompson_column

        return {"ThompsonColumnState": ThompsonColumnState, "step_thompson_column": step_thompson_column}[name]
    raise AttributeError(name)
