"""Shared I/O exports, loaded on demand so header-only sizing stays backend-dark."""
from importlib import import_module

_EXPORTS = {
    'DEFAULT_SURFACE_AUXHIST_VARIABLES': ('gpuwrf.io.auxhist_stream', 'DEFAULT_SURFACE_AUXHIST_VARIABLES'),
    'AuxhistStreamConfig': ('gpuwrf.io.auxhist_stream', 'AuxhistStreamConfig'),
    'auxhist_output_boundaries': ('gpuwrf.io.auxhist_stream', 'auxhist_output_boundaries'),
    'auxhist_substeps_per_hour': ('gpuwrf.io.auxhist_stream', 'auxhist_substeps_per_hour'),
    'coerce_auxhist_streams': ('gpuwrf.io.auxhist_stream', 'coerce_auxhist_streams'),
    'Gen2GridSpec': ('gpuwrf.io.gen2_accessor', 'Gen2GridSpec'),
    'Gen2Run': ('gpuwrf.io.gen2_accessor', 'Gen2Run'),
    'LazyNetCDFArray': ('gpuwrf.io.gen2_accessor', 'LazyNetCDFArray'),
    'domain_mask': ('gpuwrf.io.validation', 'domain_mask'),
    'lead_time_slice': ('gpuwrf.io.validation', 'lead_time_slice'),
    'load_gen2_var': ('gpuwrf.io.validation', 'load_gen2_var'),
    'regrid': ('gpuwrf.io.validation', 'regrid'),
    'unit_convert': ('gpuwrf.io.validation', 'unit_convert'),
    'DOWNSTREAM_CRITICAL_VARIABLES': ('gpuwrf.io.wrfout_writer', 'DOWNSTREAM_CRITICAL_VARIABLES'),
    'FULL_WRFOUT_VARIABLES': ('gpuwrf.io.wrfout_writer', 'FULL_WRFOUT_VARIABLES'),
    'MINIMUM_WRFOUT_VARIABLES': ('gpuwrf.io.wrfout_writer', 'MINIMUM_WRFOUT_VARIABLES'),
    'OPERATIONAL_WRFOUT_VARIABLES': ('gpuwrf.io.wrfout_writer', 'OPERATIONAL_WRFOUT_VARIABLES'),
    'WrfoutDomainAuthority': ('gpuwrf.io.wrfout_writer', 'WrfoutDomainAuthority'),
    'WRFOUT_VARIABLE_SPECS': ('gpuwrf.io.wrfout_writer', 'WRFOUT_VARIABLE_SPECS'),
    'bind_wrfout_domain_authority': ('gpuwrf.io.wrfout_writer', 'bind_wrfout_domain_authority'),
    'write_prepared_wrfout': ('gpuwrf.io.wrfout_writer', 'write_prepared_wrfout'),
    'write_wrfout_netcdf': ('gpuwrf.io.wrfout_writer', 'write_wrfout_netcdf'),
    'WRF_STANDARD_RESTART_VARIABLES': ('gpuwrf.io.wrfrst_netcdf', 'WRF_STANDARD_RESTART_VARIABLES'),
    'inspect_wrfrst_schema': ('gpuwrf.io.wrfrst_netcdf', 'inspect_wrfrst_schema'),
    'read_wrfrst_carry': ('gpuwrf.io.wrfrst_netcdf', 'read_wrfrst_carry'),
    'read_wrfrst_state': ('gpuwrf.io.wrfrst_netcdf', 'read_wrfrst_state'),
    'read_wrfrst_stochastic_seeds': ('gpuwrf.io.wrfrst_netcdf', 'read_wrfrst_stochastic_seeds'),
    'write_wrfrst_carry': ('gpuwrf.io.wrfrst_netcdf', 'write_wrfrst_carry'),
    'write_wrfrst_state': ('gpuwrf.io.wrfrst_netcdf', 'write_wrfrst_state'),
}

__all__ = ['DEFAULT_SURFACE_AUXHIST_VARIABLES', 'DOWNSTREAM_CRITICAL_VARIABLES', 'AuxhistStreamConfig', 'Gen2GridSpec', 'Gen2Run', 'FULL_WRFOUT_VARIABLES', 'LazyNetCDFArray', 'MINIMUM_WRFOUT_VARIABLES', 'OPERATIONAL_WRFOUT_VARIABLES', 'WrfoutDomainAuthority', 'WRFOUT_VARIABLE_SPECS', 'bind_wrfout_domain_authority', 'WRF_STANDARD_RESTART_VARIABLES', 'auxhist_output_boundaries', 'auxhist_substeps_per_hour', 'coerce_auxhist_streams', 'domain_mask', 'inspect_wrfrst_schema', 'lead_time_slice', 'load_gen2_var', 'read_wrfrst_carry', 'read_wrfrst_state', 'read_wrfrst_stochastic_seeds', 'regrid', 'unit_convert', 'write_prepared_wrfout', 'write_wrfout_netcdf', 'write_wrfrst_carry', 'write_wrfrst_state']


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = target
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
