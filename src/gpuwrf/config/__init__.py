"""Configuration helpers for gpuwrf.

Currently exposes :mod:`gpuwrf.config.paths`, the single source of truth for
environment-overridable filesystem locations so that nothing in the runnable
path hardcodes a private workstation directory (``<DATA_ROOT>`` / ``<USER_HOME>``).
"""

from gpuwrf.config import paths

__all__ = ["paths"]
