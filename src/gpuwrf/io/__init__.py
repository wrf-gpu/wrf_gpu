"""Shared Gen2 and validation I/O for M6 infrastructure."""

from gpuwrf.io.gen2_accessor import Gen2GridSpec, Gen2Run, LazyNetCDFArray
from gpuwrf.io.validation import domain_mask, lead_time_slice, load_gen2_var, regrid, unit_convert
from gpuwrf.io.wrfout_writer import (
    DOWNSTREAM_CRITICAL_VARIABLES,
    MINIMUM_WRFOUT_VARIABLES,
    WRFOUT_VARIABLE_SPECS,
    write_wrfout_netcdf,
)

__all__ = [
    "DOWNSTREAM_CRITICAL_VARIABLES",
    "Gen2GridSpec",
    "Gen2Run",
    "LazyNetCDFArray",
    "MINIMUM_WRFOUT_VARIABLES",
    "WRFOUT_VARIABLE_SPECS",
    "domain_mask",
    "lead_time_slice",
    "load_gen2_var",
    "regrid",
    "unit_convert",
    "write_wrfout_netcdf",
]
