"""CPU-mode peak-memory probe for the ML-training-data target geometry.

GOAL
----
Measure whether the principal's ideal training config fits the 32 GiB RTX 5090
and find the largest 1 km grid that fits.  Target geometry: 3-domain nest
(d01 9 km, d02 3 km, d03 1 km), 50 levels, full operational physics
(Thompson MP, MYNN PBL+sfclay, RRTMG SW+LW, Noah-MP, GWD).

WHY CPU MEASURES THE GPU FOOTPRINT
----------------------------------
The fp64 JAX arrays have the SAME shape/dtype (and hence byte footprint)
whether allocated on the CPU host or on GPU HBM.  A CPU-mode allocation of the
resident timestep carry therefore measures the persistent VRAM the GPU run
needs.  CPU ONLY (taskset -c 8-15) so we never touch the GPU running the TOST
marathon.

TWO PATHS (this probe runs BOTH and reports both)
-------------------------------------------------
Path (a) -- DIRECT CPU peak-RSS measurement of the PERSISTENT timestep carry
  (the full operational ``State`` pytree + ``OperationalCarry`` scratch family),
  built from the frozen field-shape + precision contracts and committed on the
  CPU JAX backend.  Reported as a measured-array-bytes total and a /proc RSS
  delta.  This is the true resident footprint (the part that lives for the
  whole run).

Path (b) -- ANALYTIC sum of the dominant PER-STEP TRANSIENT working set
  (the RRTMG SW + LW g-point radiation intermediate, which the v0.12.0 nested
  OOM post-mortem identified as the single largest recurring XLA temporary),
  ANCHORED to the GPU-MEASURED proof artifacts in this directory:
    proofs/v013/optics_taumol_chunk.json  (ncol=24576, nlev=48, GPU RTX 5090)
  plus the RK3/acoustic save family and advection flux scratch (analytic).

The reported PEAK = persistent carry (a) + dominant transient (b), because XLA
holds the resident carry while it materialises the radiation transient.

Run:
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
    taskset -c 8-15 python proofs/v013/target_1km_vram_probe.py
"""

from __future__ import annotations

import gc
import json
import os
import resource
import sys
import tracemalloc
from dataclasses import dataclass

# CPU only, fp64, before importing jax.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gpuwrf.contracts.precision import DEFAULT_DTYPES  # noqa: E402
from gpuwrf.contracts.state import _state_field_shapes  # noqa: E402

GIB = 1024.0 ** 3
NZ = 50  # the target geometry uses 50 vertical levels


# ---------------------------------------------------------------------------
# Shared shape/byte helpers (driven by the FROZEN field-shape contract).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Grid:
    nz: int
    ny: int
    nx: int


def _dtype_bytes(dt) -> int:
    return jnp.dtype(dt).itemsize


def _state_persistent_bytes(grid: _Grid) -> int:
    """Sum the frozen State SoA field bytes using the real precision matrix.

    Uses ``_state_field_shapes`` (the SAME contract ``State.zeros`` uses) and the
    ADR-007 precision matrix so per-field fp64/fp32/int32 dtypes are exact.
    """

    class _GS:  # minimal duck-typed GridSpec for _state_field_shapes
        nz = grid.nz
        ny = grid.ny
        nx = grid.nx

    total = 0
    for field, shape in _state_field_shapes(_GS).items():
        n = 1
        for d in shape:
            n *= int(d)
        total += n * _dtype_bytes(DEFAULT_DTYPES.dtype_for(field))
    return total


def _operational_carry_extra_bytes(grid: _Grid) -> int:
    """Bytes of the OperationalCarry scratch family ON TOP of State.

    Fields (operational_state.OperationalCarry), all fp64:
      t_2ave (nz,ny,nx), ww (nz+1,ny,nx), mudf (ny,nx), muave (ny,nx),
      muts (ny,nx), ph_tend (nz+1,ny,nx), u_save (nz,ny,nx+1),
      v_save (nz,ny+1,nx), w_save (nz+1,ny,nx), t_save (nz,ny,nx),
      ph_save (nz+1,ny,nx), mu_save (ny,nx), ww_save (nz+1,ny,nx),
      rthraten (nz,ny,nx).
    Noah-MP land carry + KF cumulus carry add a few more 2D/3D fields when
    active (small relative to the 3D save family); accounted separately.
    """

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    shapes = [
        (nz, ny, nx),       # t_2ave
        (nz + 1, ny, nx),   # ww
        (ny, nx),           # mudf
        (ny, nx),           # muave
        (ny, nx),           # muts
        (nz + 1, ny, nx),   # ph_tend
        (nz, ny, nx + 1),   # u_save
        (nz, ny + 1, nx),   # v_save
        (nz + 1, ny, nx),   # w_save
        (nz, ny, nx),       # t_save
        (nz + 1, ny, nx),   # ph_save
        (ny, nx),           # mu_save
        (nz + 1, ny, nx),   # ww_save
        (nz, ny, nx),       # rthraten
    ]
    b = 0
    for s in shapes:
        n = 1
        for d in s:
            n *= int(d)
        b += n * 8  # all fp64
    return b


# ---------------------------------------------------------------------------
# PATH (a): direct CPU peak-RSS of the resident carry (State + scratch).
# ---------------------------------------------------------------------------

def _alloc_resident_carry(grid: _Grid):
    """Allocate the full resident timestep carry on the CPU JAX backend.

    Mirrors State.zeros (per-field dtype from the precision matrix) WITHOUT the
    GPU-device requirement, then appends the OperationalCarry scratch family.
    Returns a list of jax arrays held alive on device (CPU).
    """

    class _GS:
        nz = grid.nz
        ny = grid.ny
        nx = grid.nx

    arrays = []
    for field, shape in _state_field_shapes(_GS).items():
        dt = DEFAULT_DTYPES.dtype_for(field)
        arrays.append(jax.device_put(jnp.zeros(shape, dtype=dt)))
    # OperationalCarry scratch family (all fp64).
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    for s in [
        (nz, ny, nx), (nz + 1, ny, nx), (ny, nx), (ny, nx), (ny, nx),
        (nz + 1, ny, nx), (nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx),
        (nz, ny, nx), (nz + 1, ny, nx), (ny, nx), (nz + 1, ny, nx), (nz, ny, nx),
    ]:
        arrays.append(jax.device_put(jnp.zeros(s, dtype=jnp.float64)))
    # force materialization
    for a in arrays:
        a.block_until_ready()
    return arrays


def measure_path_a(grid: _Grid) -> dict:
    """Measure the resident-carry peak via tracemalloc + RSS delta."""

    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux: KiB -> bytes
    tracemalloc.start()
    arrays = _alloc_resident_carry(grid)
    _cur, _peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    total_bytes = sum(int(a.size) * int(a.dtype.itemsize) for a in arrays)
    del arrays
    gc.collect()
    return {
        "measured_array_bytes": total_bytes,
        "measured_array_gib": round(total_bytes / GIB, 4),
        "rss_delta_gib": round(max(0, rss_after - rss_before) / GIB, 4),
    }


# ---------------------------------------------------------------------------
# PATH (b): analytic transient working set, anchored to GPU-measured proofs.
# ---------------------------------------------------------------------------

# GPU-MEASURED ANCHOR (proofs/v013/optics_taumol_chunk.json, RTX 5090 fp64):
#   ncol = 24576, nlev = 48.
# Radiation runs on the FULL ncol = ny*nx columns at once (band-tiled, NOT
# column-tiled -- coupling/physics_couplers.py reshapes (nz,ny,nx)->(ncol,nz)).
# Peak in-use (incremental allocator high-water) at this anchor:
#   SW upfront  16729.67 MiB   SW chunked  1906.42 MiB
#   LW upfront  17853.85 MiB   LW chunked 10068.45 MiB
# These peaks scale ~linearly in ncol and ~linearly in nlev (per-g-point
# temporaries are (ncol, nlev, band, 16) fp64).  We scale from the anchor.
_ANCHOR_NCOL = 24576
_ANCHOR_NLEV = 48
_ANCHOR_MIB = {
    ("sw", "upfront"): 16729.67,
    ("sw", "chunked"): 1906.42,
    ("lw", "upfront"): 17853.85,
    ("lw", "chunked"): 10068.45,
}
_MIB = 1024.0 ** 2


def _rad_transient_gib(ncol: int, nlev: int, kind: str, chunked: bool) -> float:
    """Scale the GPU-measured radiation peak to (ncol, nlev) by linear ncol*nlev.

    The dominant g-point temporary is (ncol, nlev(+1), nbands, 16) fp64, so the
    peak is ~proportional to ncol*nlev.  We anchor on the measured RTX 5090 peak
    and scale; this is conservative-realistic (the measured peak includes the
    incremental allocator overhead and the resident inputs).
    """

    base = _ANCHOR_MIB[(kind, "chunked" if chunked else "upfront")]
    scaled_mib = base * (ncol / _ANCHOR_NCOL) * (nlev / _ANCHOR_NLEV)
    return scaled_mib * _MIB / GIB


def _advection_rk3_scratch_gib(grid: _Grid) -> float:
    """Analytic estimate of the RK3 + acoustic + advection flux scratch peak.

    Beyond the OperationalCarry *save* family (already in the resident carry),
    the dynamics step materialises transient flux/tendency buffers.  A faithful
    WRF RK3 step holds, concurrently, on the order of ~15-25 full 3D mass-grid
    fp64 working arrays (3 flux components per advected scalar, the divergence,
    the PGF terms, the acoustic small-step tendencies).  We bound this at ~20
    equivalent 3D mass arrays.  This is far smaller than the radiation transient
    but is included so the dynamics-only peak (radiation off-cadence step) is
    not under-counted.
    """

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    one_3d = nz * ny * nx * 8
    return 20.0 * one_3d / GIB


def measure_path_b(grid: _Grid, chunked: bool) -> dict:
    ncol = grid.ny * grid.nx
    nlev = grid.nz
    sw = _rad_transient_gib(ncol, nlev, "sw", chunked)
    lw = _rad_transient_gib(ncol, nlev, "lw", chunked)
    # WRF runs SW and LW drivers sequentially; XLA frees the SW transient before
    # the LW one (and vice-versa), so the per-step radiation peak is the MAX of
    # the two, not the sum.  (Confirmed by the v0.12.0 OOM post-mortem: the 9.24
    # GiB OOM was a SINGLE recurring transient, not SW+LW summed.)
    rad_peak = max(sw, lw)
    dyn = _advection_rk3_scratch_gib(grid)
    return {
        "ncol": ncol,
        "nlev": nlev,
        "sw_transient_gib": round(sw, 3),
        "lw_transient_gib": round(lw, 3),
        "rad_peak_gib": round(rad_peak, 3),
        "dyn_scratch_gib": round(dyn, 3),
        "transient_total_gib": round(rad_peak + dyn, 3),
    }


# ---------------------------------------------------------------------------
# Domain table + driver
# ---------------------------------------------------------------------------

# 1 km single-domain shapes to sweep (ny, nx).  The principal's target is
# 641x321; we bracket it.
ONE_KM_SHAPES = [
    (250, 400),   # 400x250
    (280, 560),   # 560x280
    (321, 641),   # 641x321  <- the principal's target d03
    (360, 720),   # 720x360
]

# Live-nest companions held resident at the same time as d03 (their own State +
# carry stay on device for the whole nested run; only the ACTIVE domain's step
# transient is live at a time, so the nest adds resident-carry bytes + the
# largest companion's transient is NOT concurrent with d03's transient -- but
# the d02/d01 RESIDENT carry IS concurrent with d03).
D01_9KM = (60, 90)     # ~90x60
D02_3KM = (161, 301)   # ~301x161


def _carry_resident_gib(grid: _Grid) -> float:
    return (_state_persistent_bytes(grid) + _operational_carry_extra_bytes(grid)) / GIB


def main() -> int:
    backend = jax.default_backend()
    print(f"# jax backend = {backend}  x64 = {jax.config.read('jax_enable_x64')}")
    if backend != "cpu":
        print("WARNING: not on CPU backend; aborting to avoid touching the GPU.")
        return 2

    report = {
        "probe": "target 1km nest peak-VRAM (CPU-mode, fp64)",
        "nz": NZ,
        "anchor": {
            "source": "proofs/v013/optics_taumol_chunk.json (RTX 5090, fp64, GPU-measured)",
            "ncol": _ANCHOR_NCOL,
            "nlev": _ANCHOR_NLEV,
            "peaks_mib": {f"{k[0]}_{k[1]}": v for k, v in _ANCHOR_MIB.items()},
        },
        "resident_carry": {},
        "shapes": [],
    }

    # Resident carry for the nest companions (path a + analytic cross-check).
    for name, (ny, nx) in [("d01_9km", D01_9KM), ("d02_3km", D02_3KM)]:
        g = _Grid(NZ, ny, nx)
        a = measure_path_a(g)
        analytic = round(_carry_resident_gib(g), 4)
        report["resident_carry"][name] = {
            "shape": f"{nx}x{ny}x{NZ}",
            "analytic_resident_gib": analytic,
            "measured_array_gib": a["measured_array_gib"],
            "rss_delta_gib": a["rss_delta_gib"],
        }
        print(f"[carry] {name} {nx}x{ny}: analytic={analytic} GiB  "
              f"measured_array={a['measured_array_gib']} GiB  rss_delta={a['rss_delta_gib']} GiB")

    nest_companion_resident = (
        report["resident_carry"]["d01_9km"]["analytic_resident_gib"]
        + report["resident_carry"]["d02_3km"]["analytic_resident_gib"]
    )

    for ny, nx in ONE_KM_SHAPES:
        g = _Grid(NZ, ny, nx)
        label = f"{nx}x{ny}x{NZ}"
        a = measure_path_a(g)
        resident = round(_carry_resident_gib(g), 4)
        b_chunk = measure_path_b(g, chunked=True)
        b_upfront = measure_path_b(g, chunked=False)

        # d03-replay: only d03 resident + d03 step transient.
        replay_chunk = round(resident + b_chunk["transient_total_gib"], 3)
        replay_upfront = round(resident + b_upfront["transient_total_gib"], 3)
        # live-nest: d01+d02+d03 resident all concurrent + the active step
        # transient (max over domains; d03 transient is largest since it has the
        # most columns of the three -- d02 3km ~301x161=48461 cols < d03).
        live_chunk = round(nest_companion_resident + resident + b_chunk["transient_total_gib"], 3)
        live_upfront = round(nest_companion_resident + resident + b_upfront["transient_total_gib"], 3)

        row = {
            "shape": label,
            "ny": ny,
            "nx": nx,
            "resident_carry_gib": resident,
            "measured_array_gib": a["measured_array_gib"],
            "rss_delta_gib": a["rss_delta_gib"],
            "transient_chunked": b_chunk,
            "transient_upfront": b_upfront,
            "d03_replay_peak_gib_chunked": replay_chunk,
            "d03_replay_peak_gib_upfront": replay_upfront,
            "live_nest_peak_gib_chunked": live_chunk,
            "live_nest_peak_gib_upfront": live_upfront,
            "fits_32gib_replay_chunked": replay_chunk < 29.0,
            "fits_32gib_live_chunked": live_chunk < 29.0,
        }
        report["shapes"].append(row)
        print(f"\n[{label}] ncol={ny*nx}")
        print(f"  resident carry (1 dom)      = {resident} GiB "
              f"(measured arrays {a['measured_array_gib']} GiB, rss_delta {a['rss_delta_gib']} GiB)")
        print(f"  rad transient chunk=1  SW/LW = {b_chunk['sw_transient_gib']}/{b_chunk['lw_transient_gib']} GiB "
              f"-> peak {b_chunk['rad_peak_gib']} GiB")
        print(f"  rad transient chunk=14 SW/LW = {b_upfront['sw_transient_gib']}/{b_upfront['lw_transient_gib']} GiB "
              f"-> peak {b_upfront['rad_peak_gib']} GiB")
        print(f"  d03-REPLAY peak  chunk=1 / chunk=14 = {replay_chunk} / {replay_upfront} GiB")
        print(f"  LIVE-NEST  peak  chunk=1 / chunk=14 = {live_chunk} / {live_upfront} GiB "
              f"(+ d01+d02 resident {round(nest_companion_resident,3)} GiB)")

    # Largest-fitting 1km grid (binding = LW chunked radiation transient).
    budget = 29.0  # usable of 32 GiB total (~3 GiB held by desktop/driver).
    report["usable_budget_gib"] = budget
    report["nest_companion_resident_gib"] = round(nest_companion_resident, 3)

    def _replay_peak(ncol: int, chunked: bool) -> float:
        g = _Grid(NZ, max(1, int(round((ncol / 2.0) ** 0.5))), max(1, int(round((2.0 * ncol) ** 0.5))))
        resident = _carry_resident_gib(_Grid(NZ, 1, ncol))  # ncol-linear resident
        b = measure_path_b(_Grid(NZ, 1, ncol), chunked=chunked)
        return resident + b["transient_total_gib"]

    def _max_ncol(chunked: bool, add_nest: float) -> int:
        lo, hi = 1, 5_000_000
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if _replay_peak(mid, chunked) + add_nest <= budget:
                lo = mid
            else:
                hi = mid
        return lo

    def _grid_label(ncol: int) -> str:
        nx = int(round((2.0 * ncol) ** 0.5))
        ny = int(round((ncol / 2.0) ** 0.5))
        return f"~{nx}x{ny} (2:1), ~{int(ncol**0.5)}x{int(ncol**0.5)} (square); ncol={ncol}"

    report["largest_fitting_1km"] = {
        "replay_chunked": {"max_ncol": _max_ncol(True, 0.0), "grid": _grid_label(_max_ncol(True, 0.0))},
        "replay_upfront": {"max_ncol": _max_ncol(False, 0.0), "grid": _grid_label(_max_ncol(False, 0.0))},
        "live_nest_chunked": {
            "max_ncol": _max_ncol(True, nest_companion_resident),
            "grid": _grid_label(_max_ncol(True, nest_companion_resident)),
        },
    }
    print("\n# Largest-fitting 1km grid (binding = LW chunked radiation transient):")
    for k, v in report["largest_fitting_1km"].items():
        print(f"  {k}: {v['grid']}")

    out_path = os.path.join(os.path.dirname(__file__), "target_1km_vram_probe.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n# wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
