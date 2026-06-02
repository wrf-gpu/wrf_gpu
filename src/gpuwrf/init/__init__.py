"""Native initialization (v0.3.0 metgrid-equiv ingest, v0.4.0 real-equiv init).

v0.3.0 builds a *metgrid-equivalent artifact* from raw AIFS GRIB forcing + static
geog, gated against the real WPS ``met_em.*`` oracle. The FROZEN schema of that
artifact lives in :mod:`gpuwrf.init.metgrid_schema` and is the interface every
v0.3.0 lane (S1 forcing decode, S2 static geog, S3 interp, S4 parity, S5
integration) and the v0.4.0 native-real consumer build against.
"""

from gpuwrf.init.metgrid_schema import (
    METGRID_SCHEMA_VERSION,
    MetEmArtifact,
    MetgridFieldSpec,
    MetgridProjection,
    Stagger,
    metem_field_specs,
    metgrid_levels_spec,
)

__all__ = [
    "METGRID_SCHEMA_VERSION",
    "MetEmArtifact",
    "MetgridFieldSpec",
    "MetgridProjection",
    "Stagger",
    "metem_field_specs",
    "metgrid_levels_spec",
]
