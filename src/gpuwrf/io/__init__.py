"""Shared Gen2 and validation I/O for M6 infrastructure."""

from gpuwrf.io.auxhist_stream import (
    DEFAULT_SURFACE_AUXHIST_VARIABLES,
    AuxhistStreamConfig,
    auxhist_output_boundaries,
    auxhist_substeps_per_hour,
    coerce_auxhist_streams,
)
from gpuwrf.io.gen2_accessor import Gen2GridSpec, Gen2Run, LazyNetCDFArray
from gpuwrf.io.validation import domain_mask, lead_time_slice, load_gen2_var, regrid, unit_convert
from gpuwrf.io.wrfout_writer import (
    DOWNSTREAM_CRITICAL_VARIABLES,
    MINIMUM_WRFOUT_VARIABLES,
    OPERATIONAL_WRFOUT_VARIABLES,
    WRFOUT_VARIABLE_SPECS,
    write_prepared_wrfout,
    write_wrfout_netcdf,
)
from gpuwrf.io.wrfrst_netcdf import (
    WRF_STANDARD_RESTART_VARIABLES,
    inspect_wrfrst_schema,
    read_wrfrst_carry,
    read_wrfrst_state,
    read_wrfrst_stochastic_seeds,
    write_wrfrst_carry,
    write_wrfrst_state,
)

__all__ = [
    "DEFAULT_SURFACE_AUXHIST_VARIABLES",
    "DOWNSTREAM_CRITICAL_VARIABLES",
    "AuxhistStreamConfig",
    "Gen2GridSpec",
    "Gen2Run",
    "LazyNetCDFArray",
    "MINIMUM_WRFOUT_VARIABLES",
    "OPERATIONAL_WRFOUT_VARIABLES",
    "WRFOUT_VARIABLE_SPECS",
    "WRF_STANDARD_RESTART_VARIABLES",
    "auxhist_output_boundaries",
    "auxhist_substeps_per_hour",
    "coerce_auxhist_streams",
    "domain_mask",
    "inspect_wrfrst_schema",
    "lead_time_slice",
    "load_gen2_var",
    "read_wrfrst_carry",
    "read_wrfrst_state",
    "read_wrfrst_stochastic_seeds",
    "regrid",
    "unit_convert",
    "write_prepared_wrfout",
    "write_wrfout_netcdf",
    "write_wrfrst_carry",
    "write_wrfrst_state",
]
