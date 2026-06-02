"""P0-5b proof: CPU save/load bit-fidelity of the full-carry wrfrst-equivalent.

GPU-FREE. Builds a fully-populated ``OperationalCarry`` on the CPU (the prognostic
``State`` from the frozen field-shape contract, the WRF small-step scratch + held
``rthraten`` via ``initial_operational_carry``, a fully-populated Noah-MP land carry,
and the held ``noahmp_rad`` forcing), writes a restart with
``gpuwrf.io.restart.write_restart``, reloads it with ``read_restart``, and asserts
the reloaded carry is BIT-IDENTICAL to the in-memory carry across EVERY leaf
(State, scratch, land, rad), plus that the step index / metadata round-trip and
that schema-drift reads fail closed.

Run (from repo root):
    PYTHONPATH=src JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 taskset -c 0-3 \
        python proofs/p0_5/restart_roundtrip_proof.py

Emits proofs/p0_5/restart_roundtrip.json.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.noahmp_state import NoahMPLandState
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.io import restart as restart_mod
from gpuwrf.io.restart import (
    FORMAT,
    FORMAT_VERSION,
    read_restart,
    read_restart_metadata,
    write_restart,
)
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry


def _patterned_state(grid: GridSpec) -> State:
    """Distinct, deterministic, non-zero values in every State leaf (so a dropped
    or swapped field cannot accidentally compare equal)."""

    def pattern(shape, dtype, offset):
        values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
        values = values + float(offset) / 997.0 + 0.5
        return jnp.asarray(values, dtype=dtype)

    return State(
        **{
            field: pattern(shape, DEFAULT_DTYPES.dtype_for(field), index)
            for index, (field, shape) in enumerate(_state_field_shapes(grid).items(), start=1)
        }
    )


def _patterned_land(grid: GridSpec) -> NoahMPLandState:
    ny, nx = grid.ny, grid.nx

    def s2(off):
        return jnp.asarray(np.arange(ny * nx, dtype=np.float64).reshape(ny, nx) + off)

    def soil(off):
        return jnp.asarray(np.arange(4 * ny * nx, dtype=np.float64).reshape(4, ny, nx) + off)

    def snow(off):
        return jnp.asarray(np.arange(3 * ny * nx, dtype=np.float64).reshape(3, ny, nx) + off)

    return NoahMPLandState(
        tslb=soil(285.0), smois=soil(0.3), sh2o=soil(0.28), smcwtd=s2(0.31),
        isnow=jnp.asarray(np.zeros((ny, nx), dtype=np.int32)),
        tsno=snow(270.0), snice=snow(0.0), snliq=snow(0.0),
        zsnso=jnp.asarray(np.full((7, ny, nx), -0.1)),
        snowh=s2(0.0), sneqv=s2(0.0), sneqvo=s2(0.0), tauss=s2(0.0), albold=s2(0.2),
        tv=s2(288.0), tg=s2(287.0), tah=s2(288.5), eah=s2(1000.0),
        canliq=s2(0.01), canice=s2(0.0), fwet=s2(0.0), lai=s2(2.0), sai=s2(0.5),
        cm=s2(0.012), ch=s2(0.011), t_skin=s2(287.2), qsfc=s2(0.008), znt=s2(0.1),
        emiss=s2(0.98), albedo=s2(0.2), sfcrunoff=s2(0.001), udrunoff=s2(0.0005),
    )


def _leaf_equal(a, b) -> bool:
    a_h, b_h = np.asarray(a), np.asarray(b)
    return bool(a_h.shape == b_h.shape and a_h.dtype == b_h.dtype and np.array_equal(a_h, b_h))


def _carry_leaf_report(reference, restored) -> dict:
    mismatches: list[str] = []
    leaf_count = 0
    for field in State.__slots__:
        leaf_count += 1
        if not _leaf_equal(getattr(reference.state, field), getattr(restored.state, field)):
            mismatches.append(f"state.{field}")
    for field in restart_mod._CARRY_SCRATCH_FIELDS:
        leaf_count += 1
        if not _leaf_equal(getattr(reference, field), getattr(restored, field)):
            mismatches.append(f"scratch.{field}")
    if reference.noahmp_land is not None:
        for field in NoahMPLandState.__slots__:
            leaf_count += 1
            if not _leaf_equal(getattr(reference.noahmp_land, field), getattr(restored.noahmp_land, field)):
                mismatches.append(f"noahmp_land.{field}")
    if reference.noahmp_rad is not None:
        for i in range(len(reference.noahmp_rad)):
            leaf_count += 1
            if not _leaf_equal(reference.noahmp_rad[i], restored.noahmp_rad[i]):
                mismatches.append(f"noahmp_rad[{i}]")
    return {"leaf_count": leaf_count, "mismatches": mismatches}


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    grid = GridSpec.canary_3km_template()
    namelist = OperationalNamelist(
        grid=grid, tendencies=None, metrics=grid.metrics, dt_s=10.0, acoustic_substeps=10
    )

    results: dict = {
        "artifact_type": "operational_restart_roundtrip",
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_free": True,
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz},
        "cases": {},
    }

    # --- Case 1: FULL carry (Noah-MP land + held radiation) ---
    state = _patterned_state(grid)
    land = _patterned_land(grid)
    rad = (
        jnp.asarray(np.full((grid.ny, grid.nx), 412.0)),
        jnp.asarray(np.full((grid.ny, grid.nx), 305.0)),
        jnp.asarray(np.full((grid.ny, grid.nx), 0.42)),
    )
    carry = initial_operational_carry(state, noahmp_land=land, noahmp_rad=rad)
    path = out_dir / "restart_full.wrfrst"
    write_restart(carry, namelist, grid, 137, path, extra_metadata={"case": "full_carry"})
    meta = read_restart_metadata(path)
    restored, _, _, step = read_restart(path)
    report = _carry_leaf_report(carry, restored)
    results["cases"]["full_carry"] = {
        "metadata": {k: v for k, v in meta.items() if k != "metadata"},
        "step_roundtrip": step == 137,
        "leaf_count": report["leaf_count"],
        "mismatches": report["mismatches"],
        "bit_identical": not report["mismatches"] and step == 137,
    }

    # --- Case 2: land-less carry (Noah-MP off) ---
    carry2 = initial_operational_carry(_patterned_state(grid))
    path2 = out_dir / "restart_landless.wrfrst"
    write_restart(carry2, namelist, grid, 5, path2, extra_metadata={"case": "landless"})
    restored2, _, _, step2 = read_restart(path2)
    report2 = _carry_leaf_report(carry2, restored2)
    results["cases"]["landless_carry"] = {
        "has_noahmp_land": read_restart_metadata(path2)["has_noahmp_land"],
        "step_roundtrip": step2 == 5,
        "leaf_count": report2["leaf_count"],
        "mismatches": report2["mismatches"],
        "bit_identical": not report2["mismatches"] and step2 == 5,
    }

    # --- Case 3: schema drift must FAIL CLOSED (corrupt the recorded field order) ---
    import pickle

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["carry"]["state_field_order"] = list(payload["carry"]["state_field_order"])[:-1] + ["BOGUS"]
    bad = out_dir / "restart_bad_schema.wrfrst"
    with bad.open("wb") as handle:
        pickle.dump(payload, handle)
    failed_closed = False
    try:
        read_restart(bad)
    except ValueError:
        failed_closed = True
    results["cases"]["schema_drift_fails_closed"] = {"raised_valueerror": failed_closed}
    bad.unlink(missing_ok=True)

    all_pass = (
        results["cases"]["full_carry"]["bit_identical"]
        and results["cases"]["landless_carry"]["bit_identical"]
        and results["cases"]["schema_drift_fails_closed"]["raised_valueerror"]
    )
    results["status"] = "PASS" if all_pass else "FAIL"
    results["artifact_paths"] = [str(path.name), str(path2.name)]

    out_path = out_dir / "restart_roundtrip.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
