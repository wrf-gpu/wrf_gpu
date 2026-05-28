"""Reduced M4 dycore public API; reexports the two timestep entry points."""

from .step import run, step

__all__ = ["run", "step"]
