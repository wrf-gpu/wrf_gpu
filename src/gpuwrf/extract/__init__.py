"""Opt-in host-side in-run extract (ALISIOS EXTRACT_SPEC V1).

Nothing in this package is imported by the traced step program (it is outside
the AOT source-fingerprint closure). It consumes the already host-materialized
history payload of each frame and writes a compact NetCDF4 extract.
"""
