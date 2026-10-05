"""Resident WRF history sums. Radiation fluxes are held, sums advance every DT."""
from __future__ import annotations

import os
from collections.abc import Mapping

import jax
import jax.numpy as jnp

RADIATION_SOURCES = {
    "ACSWDNB": "swdown", "ACSWUPB": "swup", "ACSWDNT": "sw_toa_down", "ACSWUPT": "sw_toa_up",
    "ACLWDNB": "glw", "ACLWUPB": "glw_up", "ACLWDNT": "lw_toa_down", "ACLWUPT": "lw_toa_up",
    "ACSWDNBC": "sw_clear_sfc_down", "ACSWUPBC": "sw_clear_sfc_up",
    "ACSWDNTC": "sw_clear_toa_down", "ACSWUPTC": "sw_clear_toa_up",
    "ACLWDNBC": "lw_clear_sfc_down", "ACLWUPBC": "lw_clear_sfc_up",
    "ACLWDNTC": "lw_clear_toa_down", "ACLWUPTC": "lw_clear_toa_up",
}
SURFACE_SOURCES = {"ACHFX": "hfx", "ACLHF": "lh", "ACGRDFLX": "grdflx"}
SNOW_ACCUMULATORS = ("ACSNOM",)
LAND_FLUX_FIELDS = (
    "SAV", "SAG", "FSA", "FIRA", "FVEG", "ECAN", "ETRAN", "EDIR", "GRDFLX", "TRAD",
    "CANHS", "SOILENERGY", "SNOWENERGY", "CHV", "CHB", "TR", "EVB", "EVC", "EVG",
    "IRB", "IRC", "IRG", "SHB", "SHC", "SHG", "GHB", "GHV",
    "CHLEAF", "CHUC", "CHV2", "CHB2", "RSSUN", "RSSHA", "APAR", "PSN",
    "PAH", "PAHV", "PAHG", "PAHB", "FORCTLSM", "FORCQLSM", "FORCPLSM", "FORCZLSM", "FORCWLSM",
    "T2V", "T2B", "Q2V", "Q2B", "RUNSF", "RUNSB", "SNOWC",
    "TGV", "TGB", "SNOM_INCREMENT", "QSNOWXY", "QRAINXY",
)


ENERGY_ACCUMULATORS = (*RADIATION_SOURCES, *SURFACE_SOURCES, *SNOW_ACCUMULATORS)


@jax.tree_util.register_pytree_node_class
class PackedFields(Mapping):
    """Same-shape named fields stacked in ONE array: one carry leaf per family.

    Every executable parameter costs host work per execute (PjRt/XLA buffer
    handling, ~14 us/param measured with the 74 history fields, b-diff BD73),
    so each history family rides the hot carry as a single [k, ny, nx] leaf.
    Read access by name returns the stacked slice (values unchanged).
    """

    __slots__ = ("names", "data")

    def __init__(self, names, data):
        self.names = tuple(names)
        self.data = data

    @classmethod
    def pack(cls, fields: Mapping) -> "PackedFields":
        names = tuple(fields)
        return cls(names, jnp.stack([jnp.asarray(fields[name]) for name in names]))

    def __getitem__(self, name):
        try:
            return self.data[self.names.index(name)]
        except ValueError:
            raise KeyError(name) from None

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)

    def tree_flatten(self):
        return (self.data,), self.names

    @classmethod
    def tree_unflatten(cls, names, children):
        return cls(names, children[0])


def as_packed(fields, names):
    """Restart upgrade: a legacy per-name dict becomes the packed family (canonical order).

    Instantaneous land history added after the checkpoint (QSNOWXY/QRAINXY) is seeded zero like
    seed_history: every Noah step rewrites it before the next history frame. Accumulators are never invented.
    """
    if fields is None or (isinstance(fields, PackedFields) and fields.names == tuple(names)):
        return fields
    like = jnp.asarray(fields[next(iter(fields))])
    return PackedFields.pack({name: fields[name] if name in fields else jnp.zeros_like(like)
                              for name in names if name in fields or name in LAND_FLUX_FIELDS})


def full_history_enabled():
    from gpuwrf.config.history_output import full_wrfout_variables_enabled

    return full_wrfout_variables_enabled()


def seed_history(state):
    shape = jnp.shape(state.t_skin)
    return (PackedFields(LAND_FLUX_FIELDS, jnp.zeros((len(LAND_FLUX_FIELDS), *shape), dtype=jnp.float32)),
            PackedFields(ENERGY_ACCUMULATORS, jnp.zeros((len(ENERGY_ACCUMULATORS), *shape), dtype=jnp.float32)))


def accumulate_energy(old, radiation, surface, dt):
    """WRF radiation_driver:3308-3368 and surface_driver:3432-3434.

    Energy sums are J m-2, including signed sensible/ground fluxes. With WRF's
    default bucket_J=-1 there is no reset and no counter increment. Every call
    consumes this step's held fluxes, including non-radiation steps.
    """
    result = dict(old)
    for name, attr in RADIATION_SOURCES.items():
        value = None if radiation is None else getattr(radiation, attr, None)
        if value is not None:
            result[name] = old[name] + jnp.asarray(value, old[name].dtype) * jnp.asarray(dt, old[name].dtype)
    for name, attr in SURFACE_SOURCES.items():
        result[name] = old[name] + jnp.asarray(surface[attr], old[name].dtype) * jnp.asarray(dt, old[name].dtype)
    # Noah-MP driver:1242 — melt rate*DT plus the three already-integrated
    # phase-change/snow-collapse ponding terms. This sum is mm, not J/m2.
    if "ACSNOM" in old:
        result["ACSNOM"] = old["ACSNOM"] + jnp.asarray(surface["land_history"]["SNOM_INCREMENT"], old["ACSNOM"].dtype)
    return result
