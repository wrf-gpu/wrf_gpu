"""Independent REAL4 caller adapter for the source-literal scalar oracle."""
import math
from types import FunctionType
import numpy as np
import v0234_wrf_scalar_boundary_oracle as source


def rate(records, lead, cadence):
    records=np.asarray(records,np.float32)
    if records.shape[0]<2:return np.zeros_like(records[0])
    lower=min(max(int(math.ceil(float(lead)/float(cadence)))-1,0),records.shape[0]-2)
    # bdy_interp1: REAL subtraction, explicit REAL*8 rdt multiplication, REAL assignment.
    delta=records[lower+1]-records[lower]
    return (delta.astype(np.float64)*(1.0/float(cadence))).astype(np.float32)


def value(records, lead, cadence):
    records=np.asarray(records,np.float32)
    lower=min(max(int(math.ceil(float(lead)/float(cadence)))-1,0),max(records.shape[0]-2,0))
    dtbc=np.float32(min(max(float(lead)-lower*float(cadence),0.),float(cadence)))
    return records[lower]+dtbc*rate(records,lead,cadence)


_globals=dict(source.__dict__,boundary_value=value,boundary_rate=rate)
_real4_body=FunctionType(source.scalar_boundary_tendency.__code__,_globals)


def tendency(field,mu,c1,c2,records,**kwargs):
    return _real4_body(*(np.asarray(v,np.float32) for v in (field,mu,c1,c2,records)),**kwargs)
