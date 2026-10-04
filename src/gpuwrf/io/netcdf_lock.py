"""One process-wide lock for netCDF-C/HDF5 access, including async history."""
from contextlib import contextmanager
import threading

import netCDF4

# Preserve the same lock if a diagnostic reloads this module.
if "NETCDF_LOCK" not in globals():
    NETCDF_LOCK = threading.RLock()


@contextmanager
def Dataset(*args, **kwargs):
    """Serialize the entire open/read/write/close lifetime of a Dataset."""
    with NETCDF_LOCK:
        with netCDF4.Dataset(*args, **kwargs) as dataset:
            yield dataset
