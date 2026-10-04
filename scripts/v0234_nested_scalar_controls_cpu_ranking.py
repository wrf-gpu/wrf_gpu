"""Offline complete-carry ranking for unresolved nested scalar controls.

This is deliberately CPU-only.  It evaluates three mutually exclusive static
configurations from the authenticated d03 Step0 carry:

* released nested configuration: h_sca_adv_order=2, moist/scalar_adv_opt=0/0;
* only the pristine Registry h_sca default: 5, 0/0;
* only the explicit operational moist/scalar controls: 2, 1/1.

The result is a ranking proof, not forecast evidence and not GPU authority.
"""

from __future__ import annotations

import hashlib
import gc
import json
import os
import pickle
import shutil
import tempfile
import traceback
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-scalar-controls-cpu-ranking-proof.json"
FAILURE = SPRINT / "nested-scalar-controls-cpu-ranking-blocker.json"
NAMELIST_CACHE = Path("/tmp/v0234-scalar-controls-d03-namelist-dc3fecdd.pkl")
STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_scalar_sixth_order_18d97595_full18h_discriminator1/"
    "failure/last-healthy-d03-step-0.pkl"
)
STEP0_SHA256 = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
NAMELIST = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/config/namelist.input"
)
NAMELIST_SHA256 = "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838"
REGISTRY = Path("<USER_HOME>/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON")
REGISTRY_SHA256 = "6f3ee02175b76487c5c6c046ff2fb4c5d41980b86c0f06eb2e09e34dacc9623a"
REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rms(np, value) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _ring(np, ny: int, nx: int, distance: int):
    yy, xx = np.indices((ny, nx))
    return np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx)) == distance


def _delta_summary(np, parent, candidate) -> dict[str, Any]:
    rows = []
    for field in sorted(set(dir(parent.state)) & set(dir(candidate.state))):
        if field.startswith("_"):
            continue
        left = getattr(parent.state, field)
        right = getattr(candidate.state, field)
        if left is None or right is None or not hasattr(left, "shape"):
            continue
        a = np.asarray(left)
        b = np.asarray(right)
        if a.shape != b.shape or not np.issubdtype(a.dtype, np.floating):
            continue
        delta = b.astype(np.float64) - a.astype(np.float64)
        changed = int(np.count_nonzero(delta))
        if not changed:
            continue
        row = {
            "field": field,
            "changed_values": changed,
            "rms": _rms(np, delta),
            "max_abs": float(np.max(np.abs(delta))),
        }
        if delta.ndim >= 2 and delta.shape[-2:] == candidate.state.theta.shape[-2:]:
            ring1 = _ring(np, delta.shape[-2], delta.shape[-1], 1)
            row["ring1_rms"] = _rms(np, delta[..., ring1])
            row["east_ring1_rms"] = _rms(np, delta[..., 1:-1, -2])
            if delta.ndim == 3:
                level_rms = np.sqrt(np.mean(delta[:, ring1] ** 2, axis=1))
                order = np.argsort(level_rms)[::-1][:10]
                row["ranked_ring1_levels"] = [
                    {"k": int(k), "rms": float(level_rms[k])} for k in order
                ]
        rows.append(row)

    theta0 = np.asarray(parent.state.theta, dtype=np.float64)
    qv0 = np.asarray(parent.state.qv, dtype=np.float64)
    theta1 = np.asarray(candidate.state.theta, dtype=np.float64)
    qv1 = np.asarray(candidate.state.qv, dtype=np.float64)
    rvovrd = 461.6 / 287.0
    t0 = theta0 / (1.0 + rvovrd * np.maximum(qv0, 0.0)) - 300.0
    t1 = theta1 / (1.0 + rvovrd * np.maximum(qv1, 0.0)) - 300.0
    t_delta = t1 - t0
    ring1 = _ring(np, t_delta.shape[-2], t_delta.shape[-1], 1)
    thm_only = theta1 / (1.0 + rvovrd * np.maximum(qv0, 0.0)) - 300.0 - t0
    qv_only = theta0 / (1.0 + rvovrd * np.maximum(qv1, 0.0)) - 300.0 - t0
    return {
        "changed_fields": [row["field"] for row in rows],
        "fields": rows,
        "diagnostic_T": {
            "full_rms_K": _rms(np, t_delta),
            "ring1_rms_K": _rms(np, t_delta[:, ring1]),
            "east_ring1_rms_K": _rms(np, t_delta[:, 1:-1, -2]),
            "THM_fixed_qv_ring1_rms_K": _rms(np, thm_only[:, ring1]),
            "qv_fixed_THM_ring1_rms_K": _rms(np, qv_only[:, ring1]),
        },
    }


def _load_d03_namelist_slim(nested, d02_replay, Gen2Run, input_dir: Path):
    """Reproduce d03's live-nest namelist without retaining all domain bundles."""

    print("SCALAR_CONTROLS_RANK stage=slim-run", flush=True)
    run = Gen2Run(input_dir)
    names = ("d01", "d02", "d03")
    dt_by_domain = nested._dt_by_domain(run, names)
    print("SCALAR_CONTROLS_RANK stage=load-d01", flush=True)
    root = d02_replay.build_replay_case(input_dir, domain="d01", standalone=True)
    print("SCALAR_CONTROLS_RANK stage=load-d02", flush=True)
    parent = d02_replay.build_replay_case(
        input_dir,
        domain="d02",
        load_lateral_boundaries=False,
        live_nest_parent=root,
    )
    del root
    gc.collect()
    print("SCALAR_CONTROLS_RANK stage=load-d03", flush=True)
    case = d02_replay.build_replay_case(
        input_dir,
        domain="d03",
        load_lateral_boundaries=False,
        live_nest_parent=parent,
    )
    del parent
    gc.collect()

    run_start = nested._coerce_run_start(str(case.metadata["run_start_label"]))
    print("SCALAR_CONTROLS_RANK stage=d03-radiation", flush=True)
    radiation_static = None
    try:
        radiation_static, _ = nested.load_radiation_static(
            case.run, "d03", grid=case.grid, metrics=case.metrics
        )
    except Exception:  # noqa: BLE001 - exact production best-effort rule.
        radiation_static = None

    print("SCALAR_CONTROLS_RANK stage=d03-gwd", flush=True)
    gwd_opt = nested._domain_gwd_opt(run, "d03")
    gwdo_statics = None
    if gwd_opt == 1 and os.environ.get("GPUWRF_GWD_NESTED", "1") == "0":
        gwd_opt = 0
    if gwd_opt == 1:
        try:
            gwdo_statics, _ = nested.load_gwdo_statics(
                case.run, "d03", grid=case.grid, metrics=case.metrics
            )
        except Exception:  # noqa: BLE001 - exact production best-effort rule.
            gwdo_statics = None
        if gwdo_statics is None:
            gwd_opt = 0

    print("SCALAR_CONTROLS_RANK stage=d03-namelist", flush=True)
    namelist = nested._make_namelist(
        grid=case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=dt_by_domain["d03"],
        parent_dt_s=dt_by_domain["d02"],
        run_start=run_start,
        radiation_static=radiation_static,
        cu_physics=nested._domain_cu_physics(run, "d03"),
        gwd_opt=gwd_opt,
        gwdo_statics=gwdo_statics,
        diff_opt=nested._domain_int(run, "dynamics", "diff_opt", "d03", 0),
        km_opt=nested._domain_int(run, "dynamics", "km_opt", "d03", 0),
        h_sca_adv_order=nested._domain_int(
            run, "dynamics", "h_sca_adv_order", "d03", 5
        ),
    )
    print("SCALAR_CONTROLS_RANK stage=d03-noahmp", flush=True)
    sf_surface_physics = nested._domain_sf_surface_physics(run, "d03")
    if sf_surface_physics == 4:
        _land, noahmp_static, _meta = nested.build_noahmp_land_state(input_dir, "d03")
        energy, rad, nroot = nested.build_noahmp_params(noahmp_static)
        julian, yearlen = nested._wrf_julian_yearlen(run_start)
        namelist = dataclass_replace(
            namelist,
            use_noahmp=True,
            sf_surface_physics=4,
            noahmp_static=noahmp_static,
            noahmp_energy_params=energy,
            noahmp_rad_params=rad,
            noahmp_nroot=nroot,
            noahmp_julian=julian,
            noahmp_yearlen=yearlen,
        )
    authority = {
        "mode": "sequential_live_parent_reconstitution_without_retained_parent_bundles",
        "domain_order": list(names),
        "run_start": run_start.isoformat(),
        "dt_by_domain": dt_by_domain,
        "d03_live_nest_base_init": case.metadata.get("live_nest_base_init", {}),
        "radiation_static_present": radiation_static is not None,
        "gwd_opt": int(gwd_opt),
        "sf_surface_physics": int(sf_surface_physics),
    }
    del case
    gc.collect()
    return namelist, authority


def main() -> int:
    actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual_env!r}")
    for path, expected in (
        (STEP0, STEP0_SHA256),
        (NAMELIST, NAMELIST_SHA256),
        (REGISTRY, REGISTRY_SHA256),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"authenticated input mismatch: {path}")

    import jax
    import jax.numpy as jnp
    import numpy as np

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
    import gpuwrf.contracts.state as state_contract

    state_contract._gpu_device = lambda: jax.devices("cpu")[0]

    import gpuwrf.runtime.operational_mode as runtime
    import gpuwrf.integration.d02_replay as d02_replay
    import gpuwrf.integration.nested_pipeline as nested
    from gpuwrf.io.gen2_accessor import Gen2Run
    from scripts import v0234_corrected_ni_ordinary_bisection as ordinary
    from scripts.v0234_rk1_frozen_theta_diffusion_cpu_ab import _manifest, _run_arm

    scratch = Path(tempfile.mkdtemp(prefix="v0234-scalar-controls-ranking-"))
    try:
        if os.environ.get("GPUWRF_SCALAR_CONTROLS_REUSE_NAMELIST") == "1":
            expected_cache = os.environ.get("GPUWRF_SCALAR_CONTROLS_NAMELIST_SHA256")
            if not expected_cache or _sha256(NAMELIST_CACHE) != expected_cache:
                raise RuntimeError("authenticated d03 namelist cache mismatch")
            with NAMELIST_CACHE.open("rb") as stream:
                base, load_authority = pickle.load(stream)
            load_authority = {
                **load_authority,
                "transient_cache": {
                    "path": str(NAMELIST_CACHE),
                    "sha256": expected_cache,
                    "authenticated_before_read": True,
                },
            }
        else:
            base, load_authority = _load_d03_namelist_slim(
                nested, d02_replay, Gen2Run, ordinary.INPUT_DIR
            )
            if NAMELIST_CACHE.exists():
                raise RuntimeError(f"refusing stale transient cache: {NAMELIST_CACHE}")
            temporary_cache = NAMELIST_CACHE.with_suffix(".pkl.tmp")
            with temporary_cache.open("xb") as stream:
                pickle.dump((base, load_authority), stream, protocol=5)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_cache, NAMELIST_CACHE)
            print(
                "SCALAR_CONTROLS_RANK namelist_cache_sha256="
                + _sha256(NAMELIST_CACHE),
                flush=True,
            )
        if load_authority["dt_by_domain"] != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
            raise RuntimeError(
                f"timestep hierarchy changed: {load_authority['dt_by_domain']!r}"
            )
        if os.environ.get("GPUWRF_SCALAR_CONTROLS_LOAD_ONLY") == "1":
            print("SCALAR_CONTROLS_RANK load_only=green", flush=True)
            return 0
        run = Gen2Run(ordinary.INPUT_DIR)
        loader_controls = {
            name: {
                "h_sca_adv_order": nested._domain_int(
                    run, "dynamics", "h_sca_adv_order", name, 5
                ),
                # These two values intentionally expose the current loader gap:
                # _make_namelist has no corresponding arguments, so its static
                # OperationalNamelist defaults remain active despite opts 1/1 in
                # the source namelist.
                "moist_adv_opt": int(base.moist_adv_opt),
                "scalar_adv_opt": int(base.scalar_adv_opt),
            }
            for name in ("d01", "d02", "d03")
        }
        with STEP0.open("rb") as stream:
            carry = pickle.load(stream)
        clock = runtime.build_clock_base(base)
        configs = {
            "released_h2_opts00": dataclass_replace(
                base, h_sca_adv_order=2, moist_adv_opt=0, scalar_adv_opt=0
            ),
            "h_sca_only_h5_opts00": dataclass_replace(
                base, h_sca_adv_order=5, moist_adv_opt=0, scalar_adv_opt=0
            ),
            "scalar_controls_only_h2_opts11": dataclass_replace(
                base, h_sca_adv_order=2, moist_adv_opt=1, scalar_adv_opt=1
            ),
        }
        arms: dict[str, Any] = {}
        audits: dict[str, Any] = {}
        manifests: dict[str, Any] = {}
        for name, namelist in configs.items():
            print(f"SCALAR_CONTROLS_RANK {name} lower/compile/dispatch", flush=True)
            result, audit = _run_arm(runtime, jax, jnp, carry, namelist, clock)
            arms[name] = result
            audits[name] = audit
            manifests[name] = _manifest(jax, np, result)

        parent = arms["released_h2_opts00"]
        comparisons = {
            name: _delta_summary(np, parent, arms[name])
            for name in ("h_sca_only_h5_opts00", "scalar_controls_only_h2_opts11")
        }
        checks = {
            "authenticated_inputs": True,
            "actual_loader_all_h5_after_worktree_patch": all(
                row["h_sca_adv_order"] == 5 for row in loader_controls.values()
            ),
            "actual_loader_still_drops_explicit_opts11": all(
                row["moist_adv_opt"] == 0 and row["scalar_adv_opt"] == 0
                for row in loader_controls.values()
            ),
            "all_arms_interface_106_identity": all(
                row["interface_identity"] for row in audits.values()
            ),
            "all_arms_callback_free": all(
                not row["forbidden_targets"] for row in audits.values()
            ),
            "all_arms_finite": all(
                all(leaf["finite"] for leaf in manifest["leaves"])
                for manifest in manifests.values()
            ),
            "both_discrepancies_material": all(
                manifests[name]["sha256"] != manifests["released_h2_opts00"]["sha256"]
                for name in comparisons
            ),
        }
        proof = {
            "schema": "gpuwrf.v0234.nested-scalar-controls-cpu-ranking.v1",
            "environment": actual_env,
            "inputs": {
                "step0": {"path": str(STEP0), "sha256": STEP0_SHA256, "leaf_count": 106},
                "canonical_namelist": {"path": str(NAMELIST), "sha256": NAMELIST_SHA256},
                "pristine_registry": {"path": str(REGISTRY), "sha256": REGISTRY_SHA256},
            },
            "load_authority": load_authority,
            "actual_loader_controls": loader_controls,
            "arms": {
                name: {
                    "configuration": {
                        "h_sca_adv_order": int(config.h_sca_adv_order),
                        "moist_adv_opt": int(config.moist_adv_opt),
                        "scalar_adv_opt": int(config.scalar_adv_opt),
                    },
                    "audit": audits[name],
                    "manifest_sha256": manifests[name]["sha256"],
                }
                for name, config in configs.items()
            },
            "comparisons_vs_released": comparisons,
            "checks": checks,
            "causal_limit": "One authenticated complete CPU step ranks static source discrepancies; only the bounded full-history Step200 GPU gate can establish trajectory improvement.",
            "verdict": (
                "NESTED_SCALAR_CONTROLS_CPU_RANKING_GREEN"
                if all(checks.values())
                else "NESTED_SCALAR_CONTROLS_CPU_RANKING_RED"
            ),
        }
        proof["proof_sha256"] = _canonical(proof)
        temporary = OUT.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, OUT)
        print(
            json.dumps(
                {
                    "verdict": proof["verdict"],
                    "proof_sha256": proof["proof_sha256"],
                    "loader_controls": loader_controls,
                    "comparisons": {
                        name: {
                            "changed_fields": row["changed_fields"],
                            "diagnostic_T": row["diagnostic_T"],
                        }
                        for name, row in comparisons.items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if all(checks.values()) else 3
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - retain exact offline blocker.
        payload = {
            "schema": "gpuwrf.v0234.nested-scalar-controls-cpu-ranking-blocker.v1",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "gpu_commands": 0,
            "model_arms_started": 0,
        }
        payload["proof_sha256"] = _canonical(payload)
        temporary = FAILURE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, FAILURE)
        print(json.dumps(payload, sort_keys=True), flush=True)
        raise
