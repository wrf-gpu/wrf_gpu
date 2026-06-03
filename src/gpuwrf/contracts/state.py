"""Device-resident prognostic state and tendency contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import jax
from jax import config
import jax.numpy as jnp

from .grid import GridSpec
from .precision import DEFAULT_DTYPES


config.update("jax_enable_x64", True)


def _gpu_device() -> jax.Device:
    """Centralizes the mandatory GPU check used by constructors and self-test."""

    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("State.zeros requires a GPU device; no JAX GPU backend is visible")
    return devices[0]


def _zeros(shape: tuple[int, ...], field: str, device: jax.Device) -> jax.Array:
    """Allocates one frozen state/tendency field on the selected GPU during init only."""

    return jax.device_put(jnp.zeros(shape, dtype=DEFAULT_DTYPES.dtype_for(field)), device)


def _state_field_shapes(grid: GridSpec) -> dict[str, tuple[int, ...]]:
    """Returns the frozen SoA field-shape contract for the coupled M6 state."""

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    mass_3d = (nz, ny, nx)
    surface_2d = (ny, nx)
    boundary_side = max(nx + 1, ny + 1)
    boundary_width = 5
    boundary_mass = (1, 4, boundary_width, nz, boundary_side)
    boundary_face = (1, 4, boundary_width, nz + 1, boundary_side)
    boundary_surface = (1, 4, boundary_width, 1, boundary_side)
    return {
        "u": (nz, ny, nx + 1),
        "v": (nz, ny + 1, nx),
        "w": (nz + 1, ny, nx),
        "theta": mass_3d,
        "qv": mass_3d,
        "p": mass_3d,
        "p_total": mass_3d,
        "p_perturbation": mass_3d,
        "ph": (nz + 1, ny, nx),
        "ph_total": (nz + 1, ny, nx),
        "ph_perturbation": (nz + 1, ny, nx),
        "mu": surface_2d,
        "mu_total": surface_2d,
        "mu_perturbation": surface_2d,
        "qc": mass_3d,
        "qr": mass_3d,
        "qi": mass_3d,
        "qs": mass_3d,
        "qg": mass_3d,
        "Ni": mass_3d,
        "Nr": mass_3d,
        "Ns": mass_3d,
        "Ng": mass_3d,
        "qke": mass_3d,
        "ustar": surface_2d,
        "theta_flux": surface_2d,
        "qv_flux": surface_2d,
        "tau_u": surface_2d,
        "tau_v": surface_2d,
        "rhosfc": surface_2d,
        "fltv": surface_2d,
        "t_skin": surface_2d,
        "soil_moisture": surface_2d,
        "xland": surface_2d,
        "lakemask": surface_2d,
        "mavail": surface_2d,
        "roughness_m": surface_2d,
        "lu_index": surface_2d,
        "rain_acc": surface_2d,
        "snow_acc": surface_2d,
        "graupel_acc": surface_2d,
        "ice_acc": surface_2d,
        "u_bdy": boundary_mass,
        "v_bdy": boundary_mass,
        "theta_bdy": boundary_mass,
        "qv_bdy": boundary_mass,
        "ph_bdy": boundary_face,
        "mu_bdy": boundary_surface,
        "w_bdy": boundary_face,
        "p_bdy": boundary_mass,
        "pb_bdy": boundary_mass,
        "phb_bdy": boundary_face,
        "mub_bdy": boundary_surface,
        # v0.6.0 S0 additive physics leaves (append-only): WDM6 Nc/Nn number
        # concentrations on mass points; cumulus rainc_acc on surface points.
        "Nc": mass_3d,
        "Nn": mass_3d,
        "rainc_acc": surface_2d,
    }


def _leaf_nbytes(leaves: Iterable[jax.Array]) -> int:
    """Computes persistent byte totals from pytree leaves for the spacetime budget."""

    return int(sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in leaves))


@jax.tree_util.register_pytree_node_class
class BaseState:
    """Read-only WRF base-state fields separated from prognostic State.

    These fields may vary over terrain but do not belong in the high-frequency
    prognostic timestep carry. They are explicit so c2 pressure/geopotential
    helpers do not infer static WRF quantities from reduced c1 state.
    """

    __slots__ = ("pb", "phb", "mub", "t0", "theta_base")

    def __init__(
        self,
        pb: jax.Array,
        phb: jax.Array,
        mub: jax.Array,
        t0: jax.Array,
        theta_base: jax.Array,
    ) -> None:
        self.pb = pb
        self.phb = phb
        self.mub = mub
        self.t0 = t0
        self.theta_base = theta_base

    @classmethod
    def zeros(cls, grid: GridSpec) -> "BaseState":
        """Allocates base-state placeholders once on the first visible GPU."""

        device = _gpu_device()
        nz, ny, nx = grid.nz, grid.ny, grid.nx
        return cls(
            _zeros((nz, ny, nx), "p", device),
            _zeros((nz + 1, ny, nx), "ph", device),
            _zeros((ny, nx), "mu", device),
            _zeros((nz, ny, nx), "theta", device),
            _zeros((nz, ny, nx), "theta", device),
        )

    @property
    def mu_base(self) -> jax.Array:
        """Compatibility alias for the WRF base dry-column mass ``mub``."""

        return self.mub

    def replace(self, **updates) -> "BaseState":
        """Returns an updated base-state pytree with explicit field names."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def bytes(self) -> int:
        """Reports persistent base-state bytes for proof-object generation."""

        leaves, _ = jax.tree_util.tree_flatten(self)
        return _leaf_nbytes(leaves)

    def tree_flatten(self):
        """Presents base-state arrays as JAX leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds BaseState after JAX transformations."""

        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class BoundaryState:
    """Time-interpolated lateral forcing separated from prognostic State.

    The current M6 State still carries legacy six-leaf boundary arrays for
    compatibility. c2 code should use this object so future boundary replay can
    add pressure, base-pressure, and vertical-velocity forcing without widening
    the prognostic timestep state.
    """

    __slots__ = ("u", "v", "w", "theta", "qv", "p", "pb", "ph", "phb", "mu", "mub")

    def __init__(
        self,
        u: jax.Array,
        v: jax.Array,
        w: jax.Array,
        theta: jax.Array,
        qv: jax.Array,
        p: jax.Array,
        pb: jax.Array,
        ph: jax.Array,
        phb: jax.Array,
        mu: jax.Array,
        mub: jax.Array,
    ) -> None:
        self.u = u
        self.v = v
        self.w = w
        self.theta = theta
        self.qv = qv
        self.p = p
        self.pb = pb
        self.ph = ph
        self.phb = phb
        self.mu = mu
        self.mub = mub

    @classmethod
    def zeros(cls, grid: GridSpec, *, n_times: int = 1) -> "BoundaryState":
        """Allocates a complete boundary forcing schema once on the first GPU."""

        device = _gpu_device()
        nz = grid.nz
        boundary_side = max(grid.nx + 1, grid.ny + 1)
        shape_mass = (n_times, 4, nz, boundary_side)
        shape_face = (n_times, 4, nz + 1, boundary_side)
        shape_mu = (n_times, 4, 1, boundary_side)
        return cls(
            _zeros(shape_mass, "u", device),
            _zeros(shape_mass, "v", device),
            _zeros(shape_face, "w", device),
            _zeros(shape_mass, "theta", device),
            _zeros(shape_mass, "qv", device),
            _zeros(shape_mass, "p", device),
            _zeros(shape_mass, "p", device),
            _zeros(shape_face, "ph", device),
            _zeros(shape_face, "ph", device),
            _zeros(shape_mu, "mu", device),
            _zeros(shape_mu, "mu", device),
        )

    @classmethod
    def from_legacy_state(cls, state: "State") -> "BoundaryState":
        """Builds the new boundary object from the current six legacy leaves."""

        return cls(
            u=state.u_bdy,
            v=state.v_bdy,
            w=getattr(
                state,
                "w_bdy",
                jnp.zeros((state.u_bdy.shape[0], 4, state.ph_bdy.shape[-2], state.u_bdy.shape[-1]), dtype=state.w.dtype),
            ),
            theta=state.theta_bdy,
            qv=state.qv_bdy,
            p=getattr(state, "p_bdy", jnp.zeros_like(state.theta_bdy, dtype=state.p_perturbation.dtype)),
            pb=getattr(state, "pb_bdy", jnp.zeros_like(state.theta_bdy, dtype=state.p_total.dtype)),
            ph=state.ph_bdy,
            phb=getattr(state, "phb_bdy", jnp.zeros_like(state.ph_bdy, dtype=state.ph_total.dtype)),
            mu=state.mu_bdy,
            mub=getattr(state, "mub_bdy", jnp.zeros_like(state.mu_bdy, dtype=state.mu_total.dtype)),
        )

    def replace(self, **updates) -> "BoundaryState":
        """Returns an updated boundary pytree with explicit field names."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def bytes(self) -> int:
        """Reports persistent boundary bytes for proof-object generation."""

        leaves, _ = jax.tree_util.tree_flatten(self)
        return _leaf_nbytes(leaves)

    def tree_flatten(self):
        """Presents boundary arrays as JAX leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds BoundaryState after JAX transformations."""

        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class Tendencies:
    """Pytree of preallocated tendency buffers matching every prognostic state field."""

    __slots__ = ("u", "v", "w", "theta", "qv", "p", "ph", "mu")

    def __init__(self, u, v, w, theta, qv, p, ph, mu) -> None:
        self.u = u
        self.v = v
        self.w = w
        self.theta = theta
        self.qv = qv
        self.p = p
        self.ph = ph
        self.mu = mu

    @classmethod
    def zeros(cls, grid: GridSpec) -> "Tendencies":
        """Allocates all tendency buffers once; reused by the timestep scan carry."""

        device = _gpu_device()
        nz, ny, nx = grid.nz, grid.ny, grid.nx
        return cls(
            _zeros((nz, ny, nx + 1), "u", device),
            _zeros((nz, ny + 1, nx), "v", device),
            _zeros((nz + 1, ny, nx), "w", device),
            _zeros((nz, ny, nx), "theta", device),
            _zeros((nz, ny, nx), "qv", device),
            _zeros((nz, ny, nx), "p", device),
            _zeros((nz + 1, ny, nx), "ph", device),
            _zeros((ny, nx), "mu", device),
        )

    def replace(self, **updates) -> "Tendencies":
        """Returns an updated pytree with explicit field names; mirrors State.replace."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def bytes(self) -> int:
        """Reports persistent tendency-buffer bytes for proof-object generation."""

        leaves, _ = jax.tree_util.tree_flatten(self)
        return _leaf_nbytes(leaves)

    def tree_flatten(self):
        """Presents tendency arrays as JAX scan carry leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds Tendencies after JAX transformations."""

        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class State:
    """Pytree of GPU-resident WRF-shaped prognostic and coupling fields.

    Units and staggering:
    - `u`, `v`, `w`: m s^-1 on Arakawa C-grid faces.
    - `theta`: K, `qv/qc/qr/qi/qs/qg`: kg kg^-1 on mass points.
    - `p`/`p_total`: Pa total pressure on mass points; `p_perturbation`
      is WRF perturbation pressure. It is diagnostically refreshed inside
      the dycore after acoustic `ph/theta/mu` changes and is used by c2 PGF
      terms.
    - `ph`/`ph_total`: m2 s^-2 total geopotential on vertical faces;
      `ph_perturbation` is WRF perturbation geopotential advanced by the
      nonhydrostatic acoustic dycore.
    - `mu`/`mu_total`: Pa column dry mass on mass points; `mu_perturbation`
      is the perturbation dry-column mass relative to `BaseState.mub`.
    - `Ni/Nr/Ns/Ng`: m^-3 number concentrations on mass points.
    - `qke`: m2 s^-2 MYNN turbulent kinetic energy on mass points.
    - `ustar`: m s^-1, `theta_flux`: K m s^-1, `qv_flux`: kg kg^-1 m s^-1,
      `tau_u/tau_v`: m2 s^-2, `rhosfc`: kg m^-3, `fltv`: K m s^-1,
      `t_skin`: K, `soil_moisture`: m3 m^-3 on surface mass points.
    - `xland`, `lakemask`, `mavail`: prescribed land/water and moisture
      availability fields from Gen2 `wrfinput_d02`; `roughness_m`: prescribed
      or derived surface roughness length in m; `lu_index`: int32 WRF
      land-use category on mass points from `wrfinput_d02`.
    - `rain_acc/snow_acc/graupel_acc/ice_acc`: mm accumulated precipitation
      on surface mass points.
    - `*_bdy`: time-varying lateral forcing as
      `(time, side=W/E/S/N, bdy_width, z-like, padded side index)`.
      Legacy four-dimensional leaves without `bdy_width` are still accepted
      by the boundary application adapter.
    """

    __slots__ = (
        "u",
        "v",
        "w",
        "theta",
        "qv",
        "p",
        "ph",
        "mu",
        "p_total",
        "p_perturbation",
        "ph_total",
        "ph_perturbation",
        "mu_total",
        "mu_perturbation",
        "qc",
        "qr",
        "qi",
        "qs",
        "qg",
        "Ni",
        "Nr",
        "Ns",
        "Ng",
        "qke",
        "ustar",
        "theta_flux",
        "qv_flux",
        "tau_u",
        "tau_v",
        "rhosfc",
        "fltv",
        "t_skin",
        "soil_moisture",
        "xland",
        "lakemask",
        "mavail",
        "roughness_m",
        "rain_acc",
        "snow_acc",
        "graupel_acc",
        "ice_acc",
        "u_bdy",
        "v_bdy",
        "theta_bdy",
        "qv_bdy",
        "ph_bdy",
        "mu_bdy",
        "w_bdy",
        "p_bdy",
        "pb_bdy",
        "phb_bdy",
        "mub_bdy",
        "lu_index",
        # --- v0.6.0 S0 additive physics leaves (append-only; manager patch) ---
        # WDM6 cloud-droplet (Nc) + CCN (Nn) number concentrations (m^-3) and the
        # cumulus precipitation accumulator (rainc_acc, mm -> RAINC). Appended AT
        # THE END so the prefix of every existing leaf keeps its pytree position;
        # ``__init__`` gives them ``None`` defaults (-> zeros) so a pre-v0.6.0
        # ``cls(*children)`` flatten with the old leaf count still reconstructs.
        "Nc",
        "Nn",
        "rainc_acc",
    )

    def __init__(
        self,
        u: jax.Array,
        v: jax.Array,
        w: jax.Array,
        theta: jax.Array,
        qv: jax.Array,
        p: jax.Array,
        ph: jax.Array,
        mu: jax.Array,
        p_total: jax.Array,
        p_perturbation: jax.Array,
        ph_total: jax.Array,
        ph_perturbation: jax.Array,
        mu_total: jax.Array,
        mu_perturbation: jax.Array,
        qc: jax.Array,
        qr: jax.Array,
        qi: jax.Array,
        qs: jax.Array,
        qg: jax.Array,
        Ni: jax.Array,
        Nr: jax.Array,
        Ns: jax.Array,
        Ng: jax.Array,
        qke: jax.Array,
        ustar: jax.Array,
        theta_flux: jax.Array,
        qv_flux: jax.Array,
        tau_u: jax.Array,
        tau_v: jax.Array,
        rhosfc: jax.Array,
        fltv: jax.Array,
        t_skin: jax.Array,
        soil_moisture: jax.Array,
        xland: jax.Array,
        lakemask: jax.Array,
        mavail: jax.Array,
        roughness_m: jax.Array,
        rain_acc: jax.Array,
        snow_acc: jax.Array,
        graupel_acc: jax.Array,
        ice_acc: jax.Array,
        u_bdy: jax.Array,
        v_bdy: jax.Array,
        theta_bdy: jax.Array,
        qv_bdy: jax.Array,
        ph_bdy: jax.Array,
        mu_bdy: jax.Array,
        w_bdy: jax.Array | None = None,
        p_bdy: jax.Array | None = None,
        pb_bdy: jax.Array | None = None,
        phb_bdy: jax.Array | None = None,
        mub_bdy: jax.Array | None = None,
        lu_index: jax.Array | None = None,
        # --- v0.6.0 S0 additive physics leaves (append-only; manager patch) ---
        Nc: jax.Array | None = None,
        Nn: jax.Array | None = None,
        rainc_acc: jax.Array | None = None,
    ) -> None:
        self.u = u
        self.v = v
        self.w = w
        self.theta = theta
        self.qv = qv
        self.p = p
        self.ph = ph
        self.mu = mu
        self.p_total = p_total
        self.p_perturbation = p_perturbation
        self.ph_total = ph_total
        self.ph_perturbation = ph_perturbation
        self.mu_total = mu_total
        self.mu_perturbation = mu_perturbation
        self.qc = qc
        self.qr = qr
        self.qi = qi
        self.qs = qs
        self.qg = qg
        self.Ni = Ni
        self.Nr = Nr
        self.Ns = Ns
        self.Ng = Ng
        self.qke = qke
        self.ustar = ustar
        self.theta_flux = theta_flux
        self.qv_flux = qv_flux
        self.tau_u = tau_u
        self.tau_v = tau_v
        self.rhosfc = rhosfc
        self.fltv = fltv
        self.t_skin = t_skin
        self.soil_moisture = soil_moisture
        self.xland = xland
        self.lakemask = lakemask
        self.mavail = mavail
        self.roughness_m = roughness_m
        self.rain_acc = rain_acc
        self.snow_acc = snow_acc
        self.graupel_acc = graupel_acc
        self.ice_acc = ice_acc
        self.u_bdy = u_bdy
        self.v_bdy = v_bdy
        self.w_bdy = jnp.zeros_like(ph_bdy, dtype=w.dtype) if w_bdy is None else w_bdy
        self.theta_bdy = theta_bdy
        self.qv_bdy = qv_bdy
        self.p_bdy = jnp.zeros_like(theta_bdy, dtype=p_perturbation.dtype) if p_bdy is None else p_bdy
        self.pb_bdy = jnp.zeros_like(theta_bdy, dtype=p_total.dtype) if pb_bdy is None else pb_bdy
        self.ph_bdy = ph_bdy
        self.phb_bdy = jnp.zeros_like(ph_bdy, dtype=ph_total.dtype) if phb_bdy is None else phb_bdy
        self.mu_bdy = mu_bdy
        self.mub_bdy = jnp.zeros_like(mu_bdy, dtype=mu_total.dtype) if mub_bdy is None else mub_bdy
        self.lu_index = (
            jnp.zeros_like(xland, dtype=jnp.int32)
            if lu_index is None
            else jnp.asarray(lu_index, dtype=jnp.int32)
        )
        # v0.6.0 additive physics leaves. ``None`` -> zeros so existing call sites
        # and pre-v0.6.0 pytree flattens (old leaf count) still construct: Nc/Nn are
        # 3-D mass-point number concentrations (templated on Ni); rainc_acc is the
        # 2-D cumulus precip accumulator (templated on rain_acc). The cast keeps the
        # ADR-007 precision matrix (Nc/Nn FP32-gated, rainc_acc FP64) consistent.
        self.Nc = (
            jnp.zeros_like(qc, dtype=DEFAULT_DTYPES.dtype_for("Nc"))
            if Nc is None
            else jnp.asarray(Nc, dtype=DEFAULT_DTYPES.dtype_for("Nc"))
        )
        self.Nn = (
            jnp.zeros_like(qc, dtype=DEFAULT_DTYPES.dtype_for("Nn"))
            if Nn is None
            else jnp.asarray(Nn, dtype=DEFAULT_DTYPES.dtype_for("Nn"))
        )
        self.rainc_acc = (
            jnp.zeros_like(rain_acc, dtype=DEFAULT_DTYPES.dtype_for("rainc_acc"))
            if rainc_acc is None
            else jnp.asarray(rainc_acc, dtype=DEFAULT_DTYPES.dtype_for("rainc_acc"))
        )

    @classmethod
    def zeros(cls, grid: GridSpec) -> "State":
        """Allocates the full M6 SoA state once on the first visible GPU."""

        device = _gpu_device()
        return cls(**{field: _zeros(shape, field, device) for field, shape in _state_field_shapes(grid).items()})

    @classmethod
    def from_init(cls, grid: GridSpec, ic: Path) -> "State":
        """Keeps the future IC-loading call shape while M3 only supports zero init."""

        del ic
        return cls.zeros(grid)

    def replace(self, *, _cast: bool = True, **updates) -> "State":
        """Returns an updated pytree with explicit field names for JAX functional steps.

        ``p_total`` is authoritative; ``p_perturbation`` is a delta the caller
        maintains explicitly against the current ``BaseState.pb``. Updating
        ``p_perturbation`` alone does not auto-recompute ``p_total`` because
        ``pb`` is not visible to ``State.replace``.

        By default each updated value is cast to the *current* field dtype so
        ordinary functional updates cannot silently change precision.  Pass
        ``_cast=False`` to let the supplied dtype stand -- this is how
        ``_enforce_operational_precision(force_fp64=True)`` actually upcasts a
        mixed-precision real-case state to fp64 (otherwise the canonicalisation
        below pins fields back to their loaded dtype and force_fp64 is a no-op;
        Sprint U P0-1 / GPT firm-rule confirm-close).
        """

        values = {name: getattr(self, name) for name in self.__slots__}
        for name, value in updates.items():
            current = values[name]
            if _cast and hasattr(current, "dtype") and hasattr(value, "astype"):
                value = value.astype(current.dtype)
            values[name] = value

        def sync_total_legacy_perturbation(total: str, legacy: str, perturbation: str) -> None:
            """Keeps transitional legacy aliases aligned with explicit c2 totals."""

            total_changed = total in updates
            legacy_changed = legacy in updates
            perturbation_changed = perturbation in updates
            old_total = getattr(self, total)
            if total_changed:
                values[legacy] = values[total]
            elif legacy_changed:
                values[total] = values[legacy]
            if (total_changed or legacy_changed) and not perturbation_changed:
                values[perturbation] = values[perturbation] + (values[total] - old_total)

        sync_total_legacy_perturbation("p_total", "p", "p_perturbation")
        sync_total_legacy_perturbation("ph_total", "ph", "ph_perturbation")
        sync_total_legacy_perturbation("mu_total", "mu", "mu_perturbation")
        return type(self)(**values)

    def bytes(self) -> int:
        """Reports persistent state bytes for the spacetime budget."""

        leaves, _ = jax.tree_util.tree_flatten(self)
        return _leaf_nbytes(leaves)

    def tree_flatten(self):
        """Presents state arrays as JAX scan carry leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds State after JAX transformations."""

        del aux
        return cls(*children)


def _self_test() -> None:
    """Runs a small allocation check for the sprint validation command."""

    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    leaves, _ = jax.tree_util.tree_flatten((state, tendencies))
    platforms = {leaf.devices().pop().platform for leaf in leaves}
    if platforms != {"gpu"}:
        raise RuntimeError(f"expected all leaves on gpu, got {sorted(platforms)}")
    print(f"ok state_bytes={state.bytes()} tendency_bytes={tendencies.bytes()} device=gpu")


def main(argv: list[str] | None = None) -> int:
    """CLI exists only to serve the contract's explicit self-test command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
