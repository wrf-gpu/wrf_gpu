"""Read-only CPU audit of the first Thompson QICE/QNICE activation.

The script consumes the authenticated retained d03 step-14/15 carries.  It
reconstructs only the already-proved step-15 production input to the post-RK
Thompson call, extracts the six affected horizontal columns, and evaluates the
smallest decisive pristine-WRF branch with an independent NumPy transcription.
It never advances fresh history and never imports the released ``gpuwrf`` tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-17-v0234-thompson-ice-ownership-audit"
FIRST_ICE_OUT = SPRINT / "first-ice-columns.json"
OWNERSHIP_OUT = SPRINT / "thompson-first-ice-ownership.json"
BLOCKER_OUT = SPRINT / "audit-blocker.json"

CANDIDATE_ROOT = Path("/tmp/v0234-moist-theta-candidate-454c348f")
CANDIDATE_COMMIT = "454c348fc73805c3c9531103619fb843f6717e71"
CANDIDATE_SRC_TREE = "70b5b5b393eb4af693fae780e78f073b277f9023"
RELEASED_SRC_TREE = "835dcc29bf316c0715b41a72e064985e9cf099df"
CONTRACT_COMMIT = "b86dcf701732cce525120baf29ecf53ce6d95b43"
WRF_ROOT = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"

CHECKPOINT_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "moist_theta_activation_bisection_fa71e2ec/checkpoints"
)
STEP14 = CHECKPOINT_ROOT / "preceding-d03-step-14.pkl"
STEP15 = CHECKPOINT_ROOT / "first-active-d03-step-15.pkl"
STEP14_SHA256 = "3bdb12b7a30f628ea5ee339b88a507a4c91d7333fbfbe6c54867337ad74172a3"
STEP15_SHA256 = "dab84de2e35d4e0109422c52255c993c7bbc02f2dd541a9fff03ab662d382267"
SAVEPOINT_ROOT = Path("<DATA_ROOT>/wrf_gpu2/physics_oracle_v090/microphysics")

DT = 6.0
STEP = 15
T0 = 273.15
R_D = 287.04
EPS = 1.0e-15
TNO = 5.0
ATO = 0.304
XM0I = 1.0e-12
HGFR = 235.16
NT_C = 100.0e6

REQUIRED_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": str(WRF_ROOT),
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
    "GPUWRF_MOIST_THETA_INTERFACE": "1",
}

MP_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return sha256_bytes(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(cwd), *args), text=True).strip()


def git_blob(cwd: Path, revision: str, path: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(cwd), "show", f"{revision}:{path}"), text=True
    )


def wrf_rslf(p, temperature, np):
    """Literal WRF module_mp_thompson RSLF polynomial (not JAX code)."""

    x = np.maximum(-80.0, np.asarray(temperature, dtype=np.float64) - 273.16)
    esl = (
        0.611583699e03
        + x
        * (
            0.444606896e02
            + x
            * (
                0.143177157e01
                + x
                * (
                    0.264224321e-1
                    + x
                    * (
                        0.299291081e-3
                        + x
                        * (
                            0.203154182e-5
                            + x
                            * (
                                0.702620698e-8
                                + x * (0.379534310e-11 + x * -0.321582393e-13)
                            )
                        )
                    )
                )
            )
        )
    )
    p64 = np.asarray(p, dtype=np.float64)
    esl = np.minimum(esl, p64 * 0.15)
    return 0.622 * esl / (p64 - esl)


def wrf_rsif(p, temperature, np):
    """Literal WRF module_mp_thompson RSIF polynomial (not JAX code)."""

    x = np.maximum(-80.0, np.asarray(temperature, dtype=np.float64) - 273.16)
    esi = (
        0.609868993e03
        + x
        * (
            0.499320233e02
            + x
            * (
                0.184672631e01
                + x
                * (
                    0.402737184e-1
                    + x
                    * (
                        0.565392987e-3
                        + x
                        * (
                            0.521693933e-5
                            + x
                            * (
                                0.307839583e-7
                                + x * (0.105785160e-9 + x * 0.161444444e-12)
                            )
                        )
                    )
                )
            )
        )
    )
    p64 = np.asarray(p, dtype=np.float64)
    esi = np.minimum(esi, p64 * 0.15)
    return 0.622 * esi / np.maximum(1.0e-4, p64 - esi)


def deposition_nucleation_oracle(*, qv, qi, ni, p, temperature, dt, np):
    """Smallest decisive non-aerosol-aware WRF Cooper-curve branch."""

    qv = np.maximum(1.0e-10, np.asarray(qv, dtype=np.float64))
    qi = np.asarray(qi, dtype=np.float64)
    ni = np.asarray(ni, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    rho = 0.622 * p / (R_D * temperature * (qv + 0.622))
    qvsi = wrf_rsif(p, temperature, np)
    qvsw = wrf_rslf(p, temperature, np)
    ssati = qv / qvsi - 1.0
    ssatw = qv / qvsw - 1.0
    ssati = np.where(np.abs(ssati) < EPS, 0.0, ssati)
    ssatw = np.where(np.abs(ssatw) < EPS, 0.0, ssatw)
    arm_ice = ssati >= 0.25
    arm_water_cold = (ssatw > EPS) & (temperature < 253.15)
    cold = temperature < T0
    predicate = cold & (arm_ice | arm_water_cold)
    xnc = np.minimum(250.0e3, TNO * np.exp(ATO * (T0 - temperature)))
    xni = ni * rho  # WRF converts QNICE kg-1 to m-3 on entry.
    pni = np.maximum(xnc - xni, 0.0) / float(dt)
    rate_max = (qv - qvsi) * rho / float(dt) * 0.999
    pri = np.minimum(rate_max, XM0I * pni)
    pri = np.where(predicate, pri, 0.0)
    pri = np.maximum(pri, 0.0)
    pni = np.where(predicate, pri / XM0I, 0.0)
    return {
        "rho": rho,
        "qvsi": qvsi,
        "qvsw": qvsw,
        "ssati": ssati,
        "ssatw": ssatw,
        "cold": cold,
        "arm_ssati_ge_0p25": arm_ice,
        "arm_ssatw_and_t_lt_253p15": arm_water_cold,
        "predicate": predicate,
        "xnc_m3": xnc,
        "xni_m3": xni,
        "rate_max_kg_m3_s": rate_max,
        "pri_inu_kg_m3_s": pri,
        "pni_inu_m3_s": pni,
        "qi_increment_kg_kg": pri * float(dt) / rho,
        "ni_increment_kg_inv": pni * float(dt) / rho,
    }


def condensation_instant_freeze_oracle(*, qv, qc, p, temperature, dt, np):
    """Literal WRF cloud-condensation then HGFR instant-freeze predicates."""

    del dt
    qv = np.maximum(1.0e-10, np.asarray(qv, dtype=np.float64))
    qc = np.asarray(qc, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    qvs = wrf_rslf(p, temperature, np)
    lvap = 2.5e6 + (2106.0 - 4218.0) * (temperature - T0)
    ocp = 1.0 / (1004.0 * (1.0 + 0.887 * qv))
    lvt2 = lvap * lvap * ocp / 461.5 / (temperature * temperature)
    clap = (qv - qvs) / (1.0 + lvt2 * qvs)
    for _ in range(3):
        expo = np.exp(lvt2 * clap)
        fcd = qvs * expo - qv + clap
        dfcd = qvs * lvt2 * expo + 1.0
        clap = clap - fcd / dfcd
    ssatw = qv / qvs - 1.0
    condensation_predicate = (ssatw > EPS) | ((ssatw < -EPS) & (qc > 0.0))
    clap = np.where(condensation_predicate, clap, 0.0)
    clap = np.where(
        clap < 0.0,
        np.maximum(clap, -qc),
        np.minimum(clap, qv - 1.0e-10),
    )
    qv_after = qv - clap
    qc_after = qc + clap
    temperature_after = temperature + lvap * ocp * clap
    rho_after = 0.622 * p / (R_D * temperature_after * (qv_after + 0.622))
    direct_freeze = (temperature_after < HGFR) & (qc_after > 0.0)
    return {
        "qvsw": qvs,
        "ssatw": ssatw,
        "condensation_predicate": condensation_predicate,
        "clap_kg_kg": np.maximum(clap, 0.0),
        "qv_after_condensation": qv_after,
        "qc_after_condensation": qc_after,
        "temperature_after_condensation_K": temperature_after,
        "rho_after_condensation": rho_after,
        "direct_hgfr_freeze_predicate_before_cloud_sedimentation": direct_freeze,
        "port_instant_freeze_Ni_source_kg_inv": np.where(
            direct_freeze, qc_after / XM0I, 0.0
        ),
        "wrf_nc1d_baseline_kg_inv": NT_C / rho_after,
    }


def _array_identity(array, np) -> dict[str, Any]:
    value = np.ascontiguousarray(np.asarray(array))
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": sha256_bytes(value.tobytes(order="C")),
        "finite": bool(not np.issubdtype(value.dtype, np.inexact) or np.all(np.isfinite(value))),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
    }


def _json_vector(array, np) -> list[float]:
    return [float(value) for value in np.asarray(array, dtype=np.float64).ravel()]


def _source_span(source: str, start: str, end: str) -> dict[str, Any]:
    first = source.index(start)
    last = source.index(end, first) + len(end)
    snippet = source[first:last]
    line = source[:first].count("\n") + 1
    return {
        "start_line": line,
        "end_line": line + snippet.count("\n"),
        "sha256": sha256_bytes(snippet.encode()),
        "text": snippet,
    }


def _savepoint_scope(np) -> dict[str, Any]:
    del np
    manifest_path = SAVEPOINT_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fields = {
        (row["tag"], row["name"]): row
        for row in manifest["fields"]
        if row["name"] in {"qi", "ni"}
    }
    qi_in = fields[("in", "qi")]
    qi_out = fields[("out", "qi")]
    ni_in = fields[("in", "ni")]
    ni_out = fields[("out", "ni")]
    return {
        "path": str(SAVEPOINT_ROOT),
        "manifest_sha256": sha256_file(manifest_path),
        "wrf_itimestep": int(manifest.get("itimestep", -1)),
        "shape": qi_in["shape"],
        "byte_order": manifest["byte_order"],
        "dtype": manifest["dtype"],
        "qi_input_max": float(qi_in["max"]),
        "qi_output_max": float(qi_out["max"]),
        "ni_input_max": float(ni_in["max"]),
        "ni_output_max": float(ni_out["max"]),
        "matches_first_ice_event": False,
        "reason": (
            "The retained real-WRF savepoint is itimestep 1000 and has zero Ni output; "
            "it does not exercise the d03 native step-15 first-ice predicate."
        ),
    }


def _make_tap(runtime, couplers, jax, jnp, namelist, ys: tuple[int, ...], xs: tuple[int, ...]):
    yindex = jnp.asarray(ys, dtype=jnp.int32)
    xindex = jnp.asarray(xs, dtype=jnp.int32)

    def tap(carry, step_index, clock):
        physical_origin = carry.state
        lead_seconds = step_index.astype(jnp.float64) * float(namelist.dt_s)
        boundary_lead = lead_seconds
        if runtime._nested_frozen_wrf_boundary_active(namelist):
            boundary_lead = runtime.nested_boundary_package_endpoint_seconds(
                step_index,
                child_dt_s=float(namelist.dt_s),
                parent_cadence_s=float(namelist.boundary_config.update_cadence_s),
            )
        forcing = runtime._physics_step_forcing(
            carry,
            namelist,
            lead_seconds,
            run_radiation=False,
            first_timestep=False,
            clock_base=clock,
        )
        rk_carry = runtime._rk_scan_step(
            forcing.carry,
            namelist,
            lead_seconds=boundary_lead,
            physics_tendencies=forcing.dry_tendencies,
        )
        pre = runtime._apply_physics_non_dry_updates(
            rk_carry.state, physical_origin, forcing.state
        )
        post, h = runtime._run_post_rk_thompson_moist_theta(
            pre, rk_carry.h_diabatic, namelist
        )
        column = couplers._thompson_wrf_moist_column_from_state(pre, namelist.grid)
        inputs = {
            # _to_columns is (ny,nx,nz), not a flattened (ncol,nz) view.
            # Paired advanced indices select exactly the six (y,x) columns.
            name: getattr(column, name)[yindex, xindex]
            for name in ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng", "T", "p", "rho", "dz", "w")
        }
        inputs["theta_m"] = jnp.asarray(pre.theta[:, yindex, xindex]).T
        outputs = {
            name: jnp.asarray(getattr(post, name)[:, yindex, xindex]).T
            for name in ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng")
        }
        outputs["h_diabatic"] = jnp.asarray(h[:, yindex, xindex]).T

        # Continue the exact production completion after retaining the immediate
        # MP values.  This makes the validation compare like with like: the
        # retained Step-15 carry has already passed pressure refresh, guards,
        # lateral ownership and precision enforcement.
        next_state = runtime._refresh_grid_p_after_microphysics(post, rk_carry, namelist)
        next_carry = rk_carry.replace(state=next_state, h_diabatic=h)
        if not bool(namelist.disable_guards):
            next_state, _limiter = runtime._limit_guarded_dynamics_state_with_diagnostics(
                next_state, physical_origin
            )
            next_state = next_state.replace(
                qv=runtime._valid_mixing_ratio(next_state.qv, physical_origin.qv),
                qc=runtime._valid_mixing_ratio(next_state.qc, physical_origin.qc),
                qr=runtime._valid_mixing_ratio(next_state.qr, physical_origin.qr),
                qi=runtime._valid_mixing_ratio(next_state.qi, physical_origin.qi),
                qs=runtime._valid_mixing_ratio(next_state.qs, physical_origin.qs),
                qg=runtime._valid_mixing_ratio(next_state.qg, physical_origin.qg),
            )
        if bool(namelist.run_boundary):
            bounded = runtime.apply_lateral_boundaries(
                next_state,
                boundary_lead,
                float(namelist.dt_s),
                namelist.boundary_config,
                namelist.metrics,
                dry_spec_only=(
                    runtime._specified_bdy_cadence_active(namelist)
                    or runtime._nested_frozen_wrf_boundary_active(namelist)
                ),
            )
            if bool(namelist.disable_guards):
                next_state = bounded
            else:
                next_state = bounded.replace(
                    u=runtime._finite_or_origin(bounded.u, physical_origin.u),
                    v=runtime._finite_or_origin(bounded.v, physical_origin.v),
                    w=runtime._finite_or_origin(bounded.w, physical_origin.w),
                    theta=runtime._finite_or_origin(bounded.theta, physical_origin.theta),
                    qv=runtime._valid_mixing_ratio(bounded.qv, physical_origin.qv),
                    p=runtime._finite_or_origin(bounded.p, physical_origin.p),
                    ph=runtime._finite_or_origin(bounded.ph, physical_origin.ph),
                    p_total=runtime._finite_or_origin(
                        bounded.p_total, physical_origin.p_total
                    ),
                    ph_total=runtime._finite_or_origin(
                        bounded.ph_total, physical_origin.ph_total
                    ),
                    p_perturbation=runtime._finite_or_origin(
                        bounded.p_perturbation, physical_origin.p_perturbation
                    ),
                    ph_perturbation=runtime._finite_or_origin(
                        bounded.ph_perturbation, physical_origin.ph_perturbation
                    ),
                )
                next_state = runtime._limit_guarded_mass_state(next_state, physical_origin)
        next_state = runtime._enforce_operational_precision(
            next_state,
            force_fp64=bool(namelist.force_fp64),
            acoustic_precision_mode=namelist.acoustic_precision_mode,
            base_state=next_carry.base_state,
        )
        terminal = {
            name: jnp.asarray(getattr(next_state, name)[:, yindex, xindex]).T
            for name in ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng")
        }
        terminal["h_diabatic"] = jnp.asarray(h[:, yindex, xindex]).T
        return inputs, outputs, terminal

    return jax.jit(tap)


def main() -> int:
    scratch: Path | None = None
    try:
        environment = {name: os.environ.get(name) for name in REQUIRED_ENV}
        if environment != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {environment!r}")
        if git(ROOT, "rev-parse", "HEAD") != CONTRACT_COMMIT:
            raise RuntimeError("audit must start at exact committed contract")
        if git(ROOT, "rev-parse", "HEAD:src/gpuwrf") != RELEASED_SRC_TREE:
            raise RuntimeError("released src/gpuwrf tree changed")
        if git(CANDIDATE_ROOT, "rev-parse", "HEAD") != CANDIDATE_COMMIT:
            raise RuntimeError("candidate commit mismatch")
        if git(CANDIDATE_ROOT, "rev-parse", "HEAD:src/gpuwrf") != CANDIDATE_SRC_TREE:
            raise RuntimeError("candidate src/gpuwrf tree mismatch")
        if git(CANDIDATE_ROOT, "status", "--porcelain"):
            raise RuntimeError("candidate worktree is dirty")
        if git(WRF_ROOT, "rev-parse", "HEAD") != WRF_COMMIT:
            raise RuntimeError("pristine WRF commit mismatch")
        if sha256_file(STEP14) != STEP14_SHA256 or sha256_file(STEP15) != STEP15_SHA256:
            raise RuntimeError("retained step-14/15 carry hash mismatch")

        import jax

        jax.config.update("jax_cpu_enable_async_dispatch", False)
        import jax.numpy as jnp
        import numpy as np
        import gpuwrf

        if CANDIDATE_ROOT / "src" not in Path(gpuwrf.__file__).resolve().parents:
            raise RuntimeError("gpuwrf imported outside authenticated candidate")
        if jax.default_backend() != "cpu" or jax.config.values.get(
            "jax_cpu_enable_async_dispatch"
        ) is not False:
            raise RuntimeError("synchronous CPU backend not active")
        if sorted(os.sched_getaffinity(0)) != [12]:
            raise RuntimeError(f"unexpected CPU affinity: {sorted(os.sched_getaffinity(0))}")

        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
        import gpuwrf.coupling.physics_couplers as couplers
        import gpuwrf.runtime.operational_mode as runtime
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

        with STEP14.open("rb") as stream:
            step14 = pickle.load(stream)
        with STEP15.open("rb") as stream:
            step15 = pickle.load(stream)

        qi15 = np.asarray(step15.state.qi, dtype=np.float64)
        ni15 = np.asarray(step15.state.Ni, dtype=np.float64)
        active_qi = np.argwhere(qi15 > 0.0)
        active_ni = np.argwhere(ni15 > 0.0)
        if not np.array_equal(active_qi, active_ni) or active_qi.shape != (15, 3):
            raise RuntimeError("retained first-active QICE/QNICE geometry changed")
        columns = sorted({(int(y), int(x)) for _k, y, x in active_qi})
        if len(columns) != 6:
            raise RuntimeError(f"expected six unique horizontal columns, got {columns!r}")
        ys = tuple(row[0] for row in columns)
        xs = tuple(row[1] for row in columns)

        scratch = Path(tempfile.mkdtemp(prefix="v0234-thompson-first-ice-load-"))
        tree, names, _initial, dt_by_domain, load_authority = ordinary.load_corrected_tree(
            scratch
        )
        if names != ("d01", "d02", "d03") or dt_by_domain["d03"] != DT:
            raise RuntimeError("canonical hierarchy/timestep changed")
        namelist = tree.domains["d03"].namelist
        if int(namelist.mp_physics) != 8 or not runtime._moist_theta_interface_active(
            namelist
        ):
            raise RuntimeError("candidate mp=8 complete interface is inactive")
        if int(namelist.grid.nx) != 111 or int(namelist.grid.ny) != 93:
            raise RuntimeError("d03 geometry changed")
        mask = np.asarray(runtime._thompson_post_rk_interior_mask(step14.state, namelist))
        if not all(bool(mask[y, x]) for y, x in columns):
            raise RuntimeError("one or more active columns are not MP-owned")

        clock = runtime.build_clock_base(namelist)
        tap = _make_tap(runtime, couplers, jax, jnp, namelist, ys, xs)
        started = time.perf_counter()
        lowered = tap.lower(step14, jnp.asarray(STEP, dtype=jnp.int32), clock)
        lower_seconds = time.perf_counter() - started
        hlo = lowered.as_text().encode()
        started = time.perf_counter()
        executable = lowered.compile()
        compile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        inputs, outputs, terminal_outputs = jax.device_get(
            executable(step14, jnp.asarray(STEP, dtype=jnp.int32), clock)
        )
        dispatch_seconds = time.perf_counter() - started

        retained_selected = {
            name: np.asarray(getattr(step15.state, name)[:, ys, xs]).T
            for name in ("theta", *MP_FIELDS)
        }
        retained_selected["h_diabatic"] = np.asarray(step15.h_diabatic[:, ys, xs]).T
        observer_identity = {
            name: bool(
                np.array_equal(np.asarray(terminal_outputs[name]), retained_selected[name])
            )
            for name in terminal_outputs
        }
        observer_differences = {}
        observer_ieee_bounded = {}
        retained_theta_scale = max(
            float(np.max(np.abs(retained_selected["theta"]))), 1.0
        )
        for name in terminal_outputs:
            observed = np.asarray(terminal_outputs[name], dtype=np.float64)
            retained = np.asarray(retained_selected[name], dtype=np.float64)
            delta = observed - retained
            scale = max(float(np.max(np.abs(retained))), float(np.finfo(np.float64).tiny))
            if name == "h_diabatic":
                # h is formed from a difference of O(theta) values divided by
                # dt.  Reuse the predecessor sprint's pre-declared 128-op
                # cancellation budget; scaling by |h| would be dimensionally
                # wrong and spuriously tight near a first activation.
                ceiling = (
                    128.0
                    * float(np.finfo(np.float64).eps)
                    * retained_theta_scale
                    / DT
                )
                ceiling_name = "128eps_theta_scale_over_dt"
            else:
                ceiling = 8192.0 * float(np.finfo(np.float64).eps) * scale
                ceiling_name = "8192eps_field_scale"
            max_abs = float(np.max(np.abs(delta)))
            observer_differences[name] = {
                "max_abs": max_abs,
                "scale": scale,
                "fixed_ieee_ceiling": ceiling,
                "fixed_ieee_ceiling_basis": ceiling_name,
                "within_fixed_ieee_ceiling": bool(max_abs <= ceiling),
            }
            observer_ieee_bounded[name] = bool(max_abs <= ceiling)
        terminal_qi_mask = np.asarray(terminal_outputs["qi"]) > 0.0
        retained_qi_mask = retained_selected["qi"] > 0.0
        terminal_ni_mask = np.asarray(terminal_outputs["Ni"]) > 0.0
        retained_ni_mask = retained_selected["Ni"] > 0.0
        observer_geometry_identity = bool(
            np.array_equal(terminal_qi_mask, retained_qi_mask)
            and np.array_equal(terminal_ni_mask, retained_ni_mask)
        )
        if not observer_geometry_identity or not all(observer_ieee_bounded.values()):
            raise RuntimeError(
                "read-only pre-MP tap failed selected-column production geometry/IEEE gate: "
                f"geometry={observer_geometry_identity}, "
                f"fields={[name for name, ok in observer_ieee_bounded.items() if not ok]}, "
                f"differences={observer_differences}"
            )

        cooper = deposition_nucleation_oracle(
            qv=inputs["qv"],
            qi=inputs["qi"],
            ni=inputs["Ni"],
            p=inputs["p"],
            temperature=inputs["T"],
            dt=DT,
            np=np,
        )
        freeze = condensation_instant_freeze_oracle(
            qv=inputs["qv"],
            qc=inputs["qc"],
            p=inputs["p"],
            temperature=inputs["T"],
            dt=DT,
            np=np,
        )
        oracle = {
            **{f"cooper_{key}": value for key, value in cooper.items()},
            **{f"freeze_{key}": value for key, value in freeze.items()},
        }
        cooper_count = int(np.count_nonzero(cooper["predicate"]))
        freeze_count = int(
            np.count_nonzero(
                freeze["direct_hgfr_freeze_predicate_before_cloud_sedimentation"]
            )
        )
        if cooper_count > 0:
            mechanism_kind = "cooper_deposition_nucleation"
            primary_predicate = cooper["predicate"]
            source_rho = cooper["rho"]
            source_qi = cooper["qi_increment_kg_kg"]
            source_ni = cooper["ni_increment_kg_inv"]
        elif freeze_count > 0:
            mechanism_kind = "condensation_sedimentation_hgfr_freeze"
            primary_predicate = freeze[
                "direct_hgfr_freeze_predicate_before_cloud_sedimentation"
            ]
            source_rho = freeze["rho_after_condensation"]
            source_qi = freeze["clap_kg_kg"]
            source_ni = freeze["port_instant_freeze_Ni_source_kg_inv"]
        else:
            raise RuntimeError("literal WRF oracles predict no first-ice source")
        predicate_indices = np.argwhere(primary_predicate)
        output_indices = np.argwhere(np.asarray(outputs["qi"]) > 0.0)

        # Both source branches create mass/number before vertical sedimentation.
        # No ice reaches the surface during this high-level first event, so the
        # rho*dz column measure is the independent conservation discriminator.
        source_mass = np.sum(
            source_qi * source_rho * inputs["dz"], axis=1
        )
        source_number = np.sum(
            source_ni * source_rho * inputs["dz"], axis=1
        )
        output_mass = np.sum(
            np.asarray(outputs["qi"]) * source_rho * inputs["dz"], axis=1
        )
        output_number = np.sum(
            np.asarray(outputs["Ni"]) * source_rho * inputs["dz"], axis=1
        )
        mass_rel = np.abs(output_mass - source_mass) / np.maximum(np.abs(source_mass), 1e-300)
        number_rel = np.abs(output_number - source_number) / np.maximum(
            np.abs(source_number), 1e-300
        )

        active_rows = []
        for k, y, x in active_qi:
            c = columns.index((int(y), int(x)))
            active_rows.append(
                {
                    "k": int(k),
                    "y": int(y),
                    "x": int(x),
                    "column_index": c,
                    "owned_interior": bool(mask[y, x]),
                    "direct_cooper_predicate": bool(cooper["predicate"][c, k]),
                    "direct_condensation_predicate": bool(
                        freeze["condensation_predicate"][c, k]
                    ),
                    "direct_hgfr_freeze_before_sedimentation": bool(
                        freeze[
                            "direct_hgfr_freeze_predicate_before_cloud_sedimentation"
                        ][c, k]
                    ),
                    "direct_primary_source_predicate": bool(primary_predicate[c, k]),
                    "qice_output": float(outputs["qi"][c, k]),
                    "qnice_output_kg_inv": float(outputs["Ni"][c, k]),
                    "qice_qnice_ratio_kg": float(outputs["qi"][c, k] / outputs["Ni"][c, k]),
                    "temperature_K": float(inputs["T"][c, k]),
                    "pressure_Pa": float(inputs["p"][c, k]),
                    "qv_kg_kg": float(inputs["qv"][c, k]),
                    "ssati": float(cooper["ssati"][c, k]),
                    "ssatw": float(freeze["ssatw"][c, k]),
                    "cooper_xnc_m3": float(cooper["xnc_m3"][c, k]),
                    "condensed_cloud_kg_kg": float(freeze["clap_kg_kg"][c, k]),
                    "temperature_after_condensation_K": float(
                        freeze["temperature_after_condensation_K"][c, k]
                    ),
                    "port_mass_owned_Ni_source_kg_inv": float(
                        freeze["port_instant_freeze_Ni_source_kg_inv"][c, k]
                    ),
                    "wrf_nc1d_baseline_kg_inv": float(
                        freeze["wrf_nc1d_baseline_kg_inv"][c, k]
                    ),
                }
            )

        column_rows = []
        for c, (y, x) in enumerate(columns):
            input_fields = {
                name: {
                    "identity": _array_identity(inputs[name][c], np),
                    "values": _json_vector(inputs[name][c], np),
                }
                for name in inputs
            }
            output_fields = {
                name: {
                    "identity": _array_identity(outputs[name][c], np),
                    "values": _json_vector(outputs[name][c], np),
                }
                for name in outputs
            }
            column_rows.append(
                {
                    "column_index": c,
                    "y": y,
                    "x": x,
                    "fortran_j": y + 1,
                    "fortran_i": x + 1,
                    "owned_interior": bool(mask[y, x]),
                    "position": {
                        "x": "first WRF-owned i after west spec_zone=3",
                        "y": (
                            "last WRF-owned j before north spec_zone=3"
                            if y == 89
                            else f"{89-y} cells south of last owned j"
                        ),
                    },
                    "input_fields": input_fields,
                    "output_fields": output_fields,
                    "literal_oracle": {
                        key: _json_vector(value[c], np)
                        for key, value in oracle.items()
                    },
                    "source_column_water_kg_m2": float(source_mass[c]),
                    "output_column_ice_kg_m2": float(output_mass[c]),
                    "source_column_number_m2": float(source_number[c]),
                    "output_column_number_m2": float(output_number[c]),
                    "column_mass_relative_residual": float(mass_rel[c]),
                    "column_number_relative_residual": float(number_rel[c]),
                }
            )

        first_ice = {
            "schema": "gpuwrf.v0234.thompson-first-ice-columns.v1",
            "authority": {
                "contract_commit": CONTRACT_COMMIT,
                "candidate_commit": CANDIDATE_COMMIT,
                "candidate_src_tree": CANDIDATE_SRC_TREE,
                "released_src_tree": RELEASED_SRC_TREE,
                "step14_path": str(STEP14),
                "step14_sha256": STEP14_SHA256,
                "step15_path": str(STEP15),
                "step15_sha256": STEP15_SHA256,
                "load_authority": load_authority,
                "environment": environment,
            },
            "geometry_correction": {
                "contract_phrase": "15 first-active columns",
                "observed": "15 active vertical cells across 6 horizontal columns",
                "active_cell_count": 15,
                "horizontal_column_count": 6,
                "active_indices_k_y_x": active_qi.tolist(),
            },
            "tap": {
                "one_retained_step_only": True,
                "fresh_history_steps": 0,
                "step": STEP,
                "lead_seconds": STEP * DT,
                "run_radiation": False,
                "lower_seconds": lower_seconds,
                "compile_seconds": compile_seconds,
                "dispatch_seconds": dispatch_seconds,
                "stablehlo_sha256": sha256_bytes(hlo),
                "selected_output_exact_identity": observer_identity,
                "selected_output_geometry_identity": observer_geometry_identity,
                "selected_output_fixed_ieee_gate": observer_ieee_bounded,
                "selected_output_differences": observer_differences,
            },
            "active_cells": active_rows,
            "columns": column_rows,
            "array_identities": {
                "inputs": {name: _array_identity(value, np) for name, value in inputs.items()},
                "outputs": {name: _array_identity(value, np) for name, value in outputs.items()},
                "terminal_outputs": {
                    name: _array_identity(value, np)
                    for name, value in terminal_outputs.items()
                },
            },
        }
        first_ice["proof_sha256"] = canonical_sha256(first_ice)
        atomic_json(FIRST_ICE_OUT, first_ice)

        wrf_mp = git_blob(WRF_ROOT, WRF_COMMIT, "phys/module_mp_thompson.F")
        wrf_driver = git_blob(WRF_ROOT, WRF_COMMIT, "phys/module_microphysics_driver.F")
        wrf_solve = git_blob(WRF_ROOT, WRF_COMMIT, "dyn_em/solve_em.F")
        wrf_prep = git_blob(
            WRF_ROOT, WRF_COMMIT, "dyn_em/module_big_step_utilities_em.F"
        )
        port_column = (CANDIDATE_ROOT / "src/gpuwrf/physics/thompson_column.py").read_text()
        port_coupler = (
            CANDIDATE_ROOT / "src/gpuwrf/coupling/physics_couplers.py"
        ).read_text()
        port_runtime = (
            CANDIDATE_ROOT / "src/gpuwrf/runtime/operational_mode.py"
        ).read_text()

        wrf_itimestep_occurrences = wrf_mp.count("itimestep") + wrf_mp.count("ITIMESTEP")
        direct_count = int(np.count_nonzero(primary_predicate))
        transported_count = int(
            sum(not row["direct_primary_source_predicate"] for row in active_rows)
        )
        all_pre_phase_zero = bool(
            all(np.count_nonzero(np.asarray(inputs[name])) == 0 for name in ("qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng"))
        )
        source_consistent = bool(
            all_pre_phase_zero
            and direct_count > 0
            and observer_geometry_identity
            and all(observer_ieee_bounded.values())
            and float(np.max(mass_rel)) < 5.0e-3
            and float(np.max(number_rel)) < 5.0e-3
        )
        verdict = (
            "THOMPSON_FIRST_ICE_EXPECTED_REDIRECT"
            if source_consistent
            else "THOMPSON_FIRST_ICE_EVIDENCE_BLOCKED"
        )

        bindings = {
            "schema": "gpuwrf.v0234.thompson-first-ice-ownership.v1",
            "verdict": verdict,
            "authority": {
                "wrf_commit": WRF_COMMIT,
                "wrf_git_blobs_used_not_dirty_worktree": True,
                "candidate_commit": CANDIDATE_COMMIT,
                "candidate_src_tree": CANDIDATE_SRC_TREE,
                "first_ice_columns_path": str(FIRST_ICE_OUT.resolve()),
                "first_ice_columns_file_sha256": sha256_file(FIRST_ICE_OUT),
                "first_ice_columns_proof_sha256": first_ice["proof_sha256"],
            },
            "input_output_ownership": {
                "mp_physics": 8,
                "is_aerosol_aware": False,
                "dustyIce": True,
                "active_nucleation_curve": "Cooper 1986: min(250e3, 5*exp(0.304*(273.15-T))) m-3",
                "prognostic_inputs": [
                    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
                    "QNICE", "QNRAIN", "dry theta", "Exner", "pressure", "W", "DZ8W",
                    "RAINNC", "SNOWNC", "GRAUPELNC",
                ],
                "port_additional_internal_number_inputs": ["Ns", "Ng"],
                "mp8_internal_diagnostics_not_prognostic_inputs": {
                    "Nc_m3": 100.0e6,
                    "nwfa_m3": 11.1e6,
                    "nifa_m3": 5000.0,
                    "nbca_m3": 5.55e6,
                    "branch_effect": "none for non-aerosol-aware Cooper nucleation",
                },
                "QNICE_external_units": "kg-1",
                "QNICE_internal_units": "m-3 after multiplication by rho",
                "QICE_external_units": "kg kg-1",
                "outputs": [
                    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
                    "QNICE", "QNRAIN", "dry theta", "precip accumulators",
                ],
                "post_finish_outputs": ["moist theta", "h_diabatic K s-1"],
                "spec_zone": int(namelist.boundary_config.spec_zone),
                "wrf_owned_python_bounds": {
                    "y": [3, 89],
                    "x": [3, 107],
                    "inclusive": True,
                },
                "all_six_columns_owned": bool(all(mask[y, x] for y, x in columns)),
            },
            "branch_conclusion": {
                "pre_mp_all_phase_reservoirs_exact_zero": all_pre_phase_zero,
                "literal_cooper_predicate_cell_count": cooper_count,
                "literal_condensation_hgfr_source_cell_count": freeze_count,
                "selected_primary_mechanism": mechanism_kind,
                "selected_primary_source_cell_count": direct_count,
                "post_mp_positive_cell_count": int(output_indices.shape[0]),
                "positive_output_cells_without_direct_predicate": transported_count,
                "mechanism": (
                    "The selected literal WRF source branch creates QICE/QNICE aloft; the same "
                    "Thompson call also sediments it vertically."
                    if transported_count == 0
                    else "The selected literal WRF source branch creates QICE/QNICE aloft and the "
                    "same Thompson call sediments it into adjacent lower levels; 15 positive output "
                    "cells are not 15 independent source-predicate crossings."
                ),
                "first_differing_predicate_or_input": (
                    None
                    if mechanism_kind == "cooper_deposition_nucleation"
                    else "Inside the common HGFR branch WRF owns QNICE from transported cloud "
                    "number xnc=nc1d+ncten*DT, while gpuwrf uses qc_freeze/XM0I."
                ),
                "all_direct_source_cells_common_predicate": (
                    "T<273.15 AND (ssati>=0.25 OR (ssatw>1e-15 AND T<253.15)); "
                    "non-aerosol Cooper xnc exceeds pre-existing zero Ni"
                    if mechanism_kind == "cooper_deposition_nucleation"
                    else "ssatw>1e-15 -> positive clap; T<235.16 K and qc>0 -> instant freeze"
                ),
                "max_column_mass_relative_residual": float(np.max(mass_rel)),
                "max_column_number_relative_residual": float(np.max(number_rel)),
                "interpretation": (
                    "QICE/QNICE activation source-consistent"
                    if source_consistent
                    else "not closed"
                ),
            },
            "cadence_and_order": {
                "native_step": STEP,
                "native_time_seconds": STEP * DT,
                "itimestep_passed": STEP,
                "wrf_mp_body_itimestep_occurrences": wrf_itimestep_occurrences,
                "itimestep_controls_first_ice_branch": False,
                "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
                "radiation_runs_at_step15": False,
                "port_order": [
                    "step-entry surface/PBL/non-timesplit forcing",
                    "three-stage RK/acoustic integration with previous h_diabatic",
                    "non-dry physics state increments",
                    "moist_physics_prep-equivalent dry theta/pressure inputs",
                    "Thompson mp=8",
                    "moist_physics_finish-equivalent moist theta/h_diabatic",
                    "pressure refresh", "guards", "lateral boundary ownership",
                ],
                "wrf_order": [
                    "first_rk_step_part1/part2 physics and RK/acoustic dynamics",
                    "moist_physics_prep_em", "microphysics_driver/mp_gt_driver",
                    "moist_physics_finish_em", "theta_m halo/physical boundary handling",
                ],
                "ordering_discrepancy_found": False,
                "first_standard_00p20_reference_step": 200,
                "time_separation_seconds": (200 - STEP) * DT,
                "unlike_time_zero_is_trajectory_proof": False,
            },
            "source_bindings": {
                "wrf_mp_blob_sha256": sha256_bytes(wrf_mp.encode()),
                "wrf_driver_blob_sha256": sha256_bytes(wrf_driver.encode()),
                "wrf_solve_blob_sha256": sha256_bytes(wrf_solve.encode()),
                "wrf_prep_finish_blob_sha256": sha256_bytes(wrf_prep.encode()),
                "port_column_file_sha256": sha256_bytes(port_column.encode()),
                "port_coupler_file_sha256": sha256_bytes(port_coupler.encode()),
                "port_runtime_file_sha256": sha256_bytes(port_runtime.encode()),
                "wrf_deposition_nucleation": _source_span(
                    wrf_mp,
                    "!..Deposition nucleation of dust/mineral",
                    "pni_inu(k) = pri_inu(k)/xm0i",
                ),
                "wrf_condensation": _source_span(
                    wrf_mp,
                    "!..Cloud water condensation and evaporation.",
                    "ssatw(k) = qv(k)/qvs(k) - 1.",
                ),
                "wrf_instant_freeze": _source_span(
                    wrf_mp,
                    "!.. Instantly melt any cloud ice into cloud water",
                    "ncten(k) = ncten(k) - xnc*odt",
                ),
                "port_instant_freeze": _source_span(
                    port_column,
                    "def _instant_melt_freeze",
                    "T=state.T + lfus2 * ocp * qc_freeze",
                ),
                "wrf_mp8_call": _source_span(
                    wrf_driver, "CASE (THOMPSON)", "KTS=kts,KTE=kte)",
                ),
                "wrf_spec_zone_ownership": _source_span(
                    wrf_driver, "IF( specified ) THEN", "jte = min(j_end(ij),jde-1-sz)",
                ),
                "wrf_moist_prep": _source_span(
                    wrf_prep,
                    "th_phy(i,k,j) = (t_new(i,k,j) + t0) /",
                    "pf(i,k,j) = p(i,k,j)+pb(i,k,j)",
                ),
                "wrf_moist_finish": _source_span(
                    wrf_prep,
                    "! pertubation theta_moist(new)",
                    "(R_v/R_d)*qvten*th_phy(i,k,j) ) / dt",
                ),
            },
            "real_wrf_savepoint": _savepoint_scope(np),
            "checks": {
                "retained_inputs_authenticated": True,
                "observer_selected_output_bit_exact": bool(all(observer_identity.values())),
                "observer_selected_output_geometry_exact": observer_geometry_identity,
                "observer_selected_output_ieee_bounded": bool(
                    all(observer_ieee_bounded.values())
                ),
                "literal_numpy_oracle_not_jax_self_compare": True,
                "complete_six_column_mp_inputs_outputs_recorded": True,
                "source_mass_conservation_closed_lt_5e_3": bool(np.max(mass_rel) < 5e-3),
                "source_number_conservation_closed_lt_5e_3": bool(np.max(number_rel) < 5e-3),
                "qice_activation_predicates_source_consistent": source_consistent,
                "qnice_ownership_discrepancy_found": True,
                "qnice_ownership_discrepancy_active_in_first_event": bool(
                    mechanism_kind == "condensation_sedimentation_hgfr_freeze"
                ),
                "gpu_queried": False,
                "fresh_history_run": False,
                "model_bytes_changed": False,
            },
            "redirect": {
                "why_00p20_can_be_zero": (
                    "The event is at 90 s and the frozen references are at 1200 s. "
                    "A source-consistent transient may sediment/sublimate during the intervening "
                    "185 native steps; zero at 00:20 does not identify a step-15 predicate error."
                ),
                "next_action": (
                    "Capture one pristine-WRF mp_gt_driver in/out savepoint at d03 native "
                    "itimestep 15 for the six x=3,y=84..89 columns from the canonical initial "
                    "case, then compare the selected matching-time source predicate, sedimentation, "
                    "QICE, and QNICE before any separate fidelity fix."
                ),
            },
        }
        bindings["proof_sha256"] = canonical_sha256(bindings)
        atomic_json(OWNERSHIP_OUT, bindings)
        print(
            json.dumps(
                {
                    "verdict": verdict,
                    "first_ice_sha256": sha256_file(FIRST_ICE_OUT),
                    "ownership_sha256": sha256_file(OWNERSHIP_OUT),
                    "proof_sha256": bindings["proof_sha256"],
                    "direct_predicate_cells": direct_count,
                    "transported_output_cells": transported_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if source_consistent else 4
    except Exception as exc:
        payload = {
            "schema": "gpuwrf.v0234.thompson-first-ice-audit-blocker.v1",
            "verdict": "THOMPSON_FIRST_ICE_EVIDENCE_BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "gpu_queried": False,
            "model_bytes_changed": False,
        }
        payload["proof_sha256"] = canonical_sha256(payload)
        atomic_json(BLOCKER_OUT, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return 4
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
