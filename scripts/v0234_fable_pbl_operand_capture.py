#!/usr/bin/env python3
"""Phase 2 of the Fable PBL solve discriminator: GPU-side operand capture.

Re-executes the sealed CPU reference producer path (same sealed capture state,
same grid inputs, hash-identical port sources) with one additional read-only
hook on ``mynn_pbl._apply_mean_tendencies`` that records the exact operand
arrays feeding the U/V momentum tridiagonal coefficient construction:

    dz, rho, u, v (column state), dfm, s_aw, s_awu, s_awv, ustar, wind,
    rhosfc, and the derived kdz (post mass-flux stability floor), dtz,
    rhoinv, drag.

Authority gates:
  1. All 8 sealed ``gpu_source_sha256`` files hash-identical in this worktree.
  2. The re-run reproduces the sealed 28-array bundle's mix_/solve_/tendency
     arrays BIT-EXACTLY (bitpayload SHA equality against the sealed npz).
  3. The port coefficient formula re-evaluated offline from the captured
     operands reproduces the hooked solver inputs (reported max-ulp).

CPU backend only; no GPU discovery; no WRF/MPI. Output: operand npz + proof
JSON in the Fable sprint folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-kimi-pbl-solve-discriminator")
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-fable-pbl-solve-discriminator"
AUTHORITY_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_0b18530a1dc9cac2")
OUTPUT_AUTHORITY_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084")
CAPTURE_ROOT = OUTPUT_AUTHORITY_ROOT / "capture/authentic-ac6712170cbe5084"
REFERENCE_ARCHIVE = OUTPUT_AUTHORITY_ROOT / "cpu-reference/pbl-sp2-reference-28.npz"
REFERENCE_ARCHIVE_SHA256 = "a6416b7245d26f23f0df398dd6a3a926a1749cba2069dc3d0ea39c98bc3d2566"
INPUT_DIR = AUTHORITY_ROOT / "capture-inputs"

SOURCE_SHA256 = {
    "src/gpuwrf/contracts/state.py": "f959da39d8957ef72f165c2e4fad98a03e8235a58379e5a2743f9a6da93af9f8",
    "src/gpuwrf/coupling/physics_couplers.py": "cf569c4780cecca17e144c63386e3c456a78db8db48253aa75d0720441cc24bd",
    "src/gpuwrf/io/gen2_accessor.py": "3f552e3b3552ee16de6770170881d52330e91a3b9a3960fb174c2e9dc568c898",
    "src/gpuwrf/physics/mynn_constants.py": "0172e9424a126e2bde760066be7ba3f483240b53889b834c1f6441a7ff058aee",
    "src/gpuwrf/physics/mynn_pbl.py": "e4778816aeeba28acabdee49043eeac2693ab54b218b6c221b684c89281a1434",
    "src/gpuwrf/physics/noahmp_coupler.py": "cd857efa18731d796fd4a60047eb833fae94588e3150d2973cbd8100e34ced76",
    "src/gpuwrf/physics/tridiagonal_solver.py": "1f141e6581c41daf3a720ecbedfbe557b72d4a0c84bf61a7d3b3a7d449da193f",
    "src/gpuwrf/runtime/operational_mode.py": "68efe76b9d1f91860e9a49a6e6c9ab9e573986dd11a47677fa161badb3a79282",
}

ALLOWED_CPUS = {13, 14, 15, 29, 30, 31}


class CaptureError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bitpayload_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operand-output", required=True)
    parser.add_argument("--proof-output", required=True)
    args = parser.parse_args()
    operand_out = Path(args.operand_output)
    proof_out = Path(args.proof_output)
    if operand_out.exists() or proof_out.exists():
        print("REFUSE: output exists", file=sys.stderr)
        return 74

    if not set(os.sched_getaffinity(0)) <= ALLOWED_CPUS:
        raise CaptureError(f"affinity outside allowed set: {sorted(os.sched_getaffinity(0))}")

    proof: dict = {"schema": "wrfgpu2-v0234-fable-pbl-operand-capture-v1"}

    source_hashes = {}
    for rel, expected in SOURCE_SHA256.items():
        actual = sha256_file(REPO / rel)
        source_hashes[rel] = {"expected": expected, "actual": actual, "match": actual == expected}
    proof["source_authority"] = source_hashes
    mismatched = [rel for rel, item in source_hashes.items() if not item["match"]]
    if mismatched:
        raise CaptureError(f"source authority mismatch: {mismatched}")

    if sha256_file(REFERENCE_ARCHIVE) != REFERENCE_ARCHIVE_SHA256:
        raise CaptureError("sealed reference archive drift")

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(REPO / "src"))
    import jax
    import jax.numpy as jnp
    from gpuwrf.contracts.state import State
    from gpuwrf.coupling import physics_couplers as couplers
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.physics import mynn_pbl as mynn

    if jax.default_backend() != "cpu" or any(d.platform != "cpu" for d in jax.devices()):
        raise CaptureError("non-CPU backend")
    proof["backend"] = "cpu"
    proof["gpu_actions"] = 0

    manifest = json.loads((CAPTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "CAPTURE_COMPLETE_HOLD_5_OF_28":
        raise CaptureError("capture terminal drift")
    values = []
    for item in manifest.get("state_leaves", []):
        values.append(
            None if item["kind"] == "none"
            else jax.device_put(np.load(CAPTURE_ROOT / item["file"], allow_pickle=False))
        )
    if len(values) != len(State.__slots__):
        raise CaptureError("capture state schema drift")
    state = State.tree_unflatten(None, tuple(values))
    grid = Gen2Run(INPUT_DIR).grid("d03").as_grid_spec()

    initialized = couplers._mynn_state_with_first_call_qke(state, grid, True)
    column = couplers._mynn_column_from_state(initialized, grid)
    surface = couplers._surface_fluxes_from_state(initialized)
    ny, nx = column.theta.shape[:2]
    column_b = couplers._flatten_columns_to_batch(column, ny, nx)
    surface_b = couplers._flatten_columns_to_batch(surface, ny, nx)

    solves = []
    recorded: dict = {}
    original_solve = mynn._solve_tridiagonal
    original_amt = mynn._apply_mean_tendencies

    def solve_hook(a, b, c, d):
        x = original_solve(a, b, c, d)
        solves.append((a, b, c, d, x))
        return x

    def amt_hook(state_c, turb, dt, flux, wind, rhosfc, mf=None):
        if recorded:
            raise CaptureError("apply_mean_tendencies called twice")
        kdz = mynn._rho_interfaces(state_c, turb["dfm"])
        kdz = mynn._apply_s_aw_stability_floor(kdz, mf["s_aw"])
        recorded.update({
            "dz": state_c.dz, "rho": state_c.rho, "u": state_c.u, "v": state_c.v,
            "dfm": turb["dfm"], "kdz_floored": kdz,
            "s_aw": mf["s_aw"], "s_awu": mf["s_awu"], "s_awv": mf["s_awv"],
            "ustar": flux.ustar, "wind": wind, "rhosfc": rhosfc,
            "dt": dt,
        })
        return original_amt(state_c, turb, dt, flux, wind, rhosfc, mf=mf)

    mynn._solve_tridiagonal = solve_hook
    mynn._apply_mean_tendencies = amt_hook
    try:
        output, _pblh = mynn._step_mynn_pbl_impl_with_pblh(
            column_b, 6.0, False, surface_b, True, 1000.0
        )
    finally:
        mynn._solve_tridiagonal = original_solve
        mynn._apply_mean_tendencies = original_amt
    if len(solves) != 6:
        raise CaptureError(f"tridiagonal call order drift: {len(solves)}")
    if not recorded:
        raise CaptureError("operand hook never fired")

    def to_mass(value):
        return np.ascontiguousarray(np.asarray(couplers._from_columns(value)))

    def col3(value):
        return to_mass(jnp.asarray(value).reshape((ny, nx, -1)))

    # --- gate 2: bit-exact reproduction of the sealed bundle ---------------
    sealed = np.load(REFERENCE_ARCHIVE, allow_pickle=False)
    repro = {}
    mine: dict[str, np.ndarray] = {}
    for component, index in (("u", 2), ("v", 3)):
        for name, value in zip(("a", "b", "c", "d", "x"), solves[index]):
            mine[f"solve_{component}_{name}"] = col3(value)
    mine["sp2_rublten"] = (col3(output.u) - np.asarray(couplers._u_mass(initialized))) / 6.0
    mine["sp2_rvblten"] = (col3(output.v) - np.asarray(couplers._v_mass(initialized))) / 6.0
    all_bit_exact = True
    for name, value in mine.items():
        sealed_arr = np.asarray(sealed[name])
        equal = bool(np.array_equal(value.view(np.uint64), sealed_arr.view(np.uint64)))
        repro[name] = {
            "bit_exact": equal,
            "max_abs_delta": float(np.max(np.abs(value - sealed_arr))),
        }
        all_bit_exact = all_bit_exact and equal
    proof["sealed_bundle_reproduction"] = repro
    proof["sealed_bundle_reproduction_all_bit_exact"] = all_bit_exact

    # --- operands to mass grid ---------------------------------------------
    operands: dict[str, np.ndarray] = {}
    for name in ("dz", "rho", "u", "v", "dfm"):
        operands[name] = col3(recorded[name])
    operands["kdz_floored"] = col3(recorded["kdz_floored"])  # nz+1 interfaces
    for name in ("s_aw", "s_awu", "s_awv"):
        operands[name] = col3(recorded[name])  # nz+1 interfaces
    for name in ("ustar", "wind", "rhosfc"):
        value = np.asarray(recorded[name]).reshape((ny, nx))
        operands[name] = np.ascontiguousarray(value)
    proof["dt"] = float(recorded["dt"])

    # --- gate 3: offline port-formula rebuild vs hooked coefficients -------
    dz = operands["dz"]
    dtz = 6.0 / dz
    rhoinv = 1.0 / np.maximum(operands["rho"], 1.0e-4)
    kdz = operands["kdz_floored"]
    drag = operands["rhosfc"] * operands["ustar"] * operands["ustar"] / operands["wind"]
    rebuild_report = {}
    for component in ("u", "v"):
        s_aw = operands["s_aw"]
        s_awx = operands[f"s_aw{component}"]
        x_state = operands[component]
        nz = x_state.shape[0]
        half0 = 0.5 * dtz[0] * rhoinv[0]
        a = np.empty_like(x_state)
        b = np.empty_like(x_state)
        c = np.empty_like(x_state)
        d = np.empty_like(x_state)
        a[0] = -dtz[0] * kdz[0] * rhoinv[0]
        b[0] = 1.0 + dtz[0] * (kdz[1] + kdz[0] + drag) * rhoinv[0] - half0 * s_aw[1]
        c[0] = -dtz[0] * kdz[1] * rhoinv[0] - half0 * s_aw[1]
        d[0] = x_state[0] + 0.0 - dtz[0] * rhoinv[0] * s_awx[1]
        sl = slice(1, nz - 1)
        half_i = 0.5 * dtz[sl] * rhoinv[sl]
        a[sl] = -dtz[sl] * kdz[1:nz - 1] * rhoinv[sl] + half_i * s_aw[1:nz - 1]
        b[sl] = (1.0 + dtz[sl] * (kdz[1:nz - 1] + kdz[2:nz]) * rhoinv[sl]
                 + half_i * (s_aw[1:nz - 1] - s_aw[2:nz]))
        c[sl] = -dtz[sl] * kdz[2:nz] * rhoinv[sl] - half_i * s_aw[2:nz]
        d[sl] = x_state[sl] + dtz[sl] * rhoinv[sl] * (s_awx[1:nz - 1] - s_awx[2:nz])
        a[-1], b[-1], c[-1], d[-1] = 0.0, 1.0, 0.0, x_state[-1]
        for coeff, arr in (("a", a), ("b", b), ("c", c), ("d", d)):
            hooked = mine[f"solve_{component}_{coeff}"]
            rebuild_report[f"{component}_{coeff}"] = {
                "max_abs_delta": float(np.max(np.abs(arr - hooked))),
                "bit_exact": bool(np.array_equal(arr.view(np.uint64), hooked.view(np.uint64))),
            }
    proof["port_formula_rebuild_vs_hooked"] = rebuild_report

    np.savez(operand_out, **operands)
    proof["operand_archive"] = {
        "path": str(operand_out),
        "sha256": sha256_file(operand_out),
        "arrays": {name: {"shape": list(arr.shape),
                          "dtype": str(arr.dtype),
                          "bitpayload_sha256": bitpayload_sha(arr)}
                   for name, arr in operands.items()},
    }
    canonical = json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    proof["canonical_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    proof_out.write_text(json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "sealed_bundle_all_bit_exact": all_bit_exact,
        "rebuild_max": max(item["max_abs_delta"] for item in rebuild_report.values()),
        "proof": str(proof_out),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
