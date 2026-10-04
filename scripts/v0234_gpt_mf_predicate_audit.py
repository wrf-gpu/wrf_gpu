#!/usr/bin/env python3
"""Seal the v0234 MYNN mass-flux predicate audit and residual bound.

This is a CPU-only discriminator over the sprint's one-shot production capture
and the retained pristine-WRF SP2 dump.  It calls the MYNN EDMF column kernel
directly (never the production adapter), proves that the kernel reconstruction
is bit-identical to the capture, and substitutes every authenticated surface
operand available in the WRF dump.  The result distinguishes a surface seam
from an unobserved internal plume/predicate seam without tuning a threshold.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.mynn_pbl import (
    MynnPBLColumnState,
    _edmf_arrays_from_state,
)
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-mf-seam-closure"
CAPTURE = Path(
    "/tmp/v0234_gpt_mf_seam_capture/endpoint-v2/"
    "single-authority-capture.npz"
)
CAPTURE_MANIFEST = CAPTURE.parent / "manifest.json"
CAPTURE_PROOF = SPRINT / "combined-capture-proof.json"
CHANNEL_GATE = SPRINT / "channel-decomposed-sp2-gate.json"
COMPARATOR = REPO / "scripts/v0234_gpt_operand_attribution.py"
READER = REPO / "scripts/v0234_gpt_adversarial_audit.py"
WRF_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_mynn_sp2_d03_horizon55_gpt_fresh01/"
    "evidence-dumps-fresh-d03-runtime"
)
WRF_SOURCE = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_mynnedmf.F")
PORT_EDMF = REPO / "src/gpuwrf/physics/mynn_edmf.py"
PORT_PBL = REPO / "src/gpuwrf/physics/mynn_pbl.py"

EXPECTED = {
    CAPTURE: "882bd72120011dbe76fa82d52379bf29a5b42f0e0a7fceb4811bd9a2a4e68e11",
    CAPTURE_MANIFEST: "db2b3949af18cc0ebf0c762235509fd3548e218375a31c3cb35a875fb8365b8a",
    CAPTURE_PROOF: "e2b905573884f2839c3858a75494b50870874c1a60f4d1bacdff2bd4128ca11b",
    CHANNEL_GATE: "e73bbcf104575676ea3482838577e87a1914954249919fbe167e642a10168c25",
    COMPARATOR: "247a1dca01b7aeb35af1adcfd3eb1dc459472c04275edc7f5b2cb3f380df3244",
    READER: "49bc7b9761e2da1e807000bbbc101d05039a585b8b00f4b31c3b404b66d79d2a",
    WRF_SOURCE: "6e4a7d5b35ce46f01591f2c1d58e545380d546e654b4a59ee1bcf99cfbce2d72",
    PORT_EDMF: "2aeeffae999369c9c334f2db05595ac6044a2d9feb8c78bda549b0f89fa34ff0",
    PORT_PBL: "bad38f5bd6d7cb82e57f6c24016152f00688559251acb48dbab232cdb9b82b68",
}
EXPECTED_WRF_TREE = "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b"
ALLOWED_CPUS = {13, 14, 15, 29, 30, 31}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
FIELDS = ("s_aw", "s_awu", "s_awv")


class AuditFailure(RuntimeError):
    """Fail-closed evidence, resource, source, or arithmetic failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Mapping[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key != "canonical_payload_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise AuditFailure(f"OUTPUT_NOT_FRESH:{path}")
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise AuditFailure(f"TEMP_NOT_FRESH:{temporary}")
    with temporary.open("xb") as stream:
        stream.write(
            (
                json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            ).encode()
        )
        stream.flush()
    os.link(temporary, path)
    temporary.unlink()


def git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    if result.returncode:
        stderr = result.stderr.decode() if binary else result.stderr
        raise AuditFailure(f"GIT:{' '.join(args)}:{stderr.strip()}")
    return result.stdout if binary else result.stdout.strip()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditFailure(f"JSON_OBJECT:{path}")
    return value


def import_comparator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "v0234_mf_predicate_frozen_comparator", COMPARATOR
    )
    if spec is None or spec.loader is None:
        raise AuditFailure("COMPARATOR_IMPORT_SPEC")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def line_number(text: str, token: str) -> int:
    for number, line in enumerate(text.splitlines(), start=1):
        if token in line:
            return number
    raise AuditFailure(f"SOURCE_TOKEN:{token}")


def authority_gate(approved_head: str) -> dict[str, Any]:
    if Path("/tmp/PREEMPT_CPU").exists():
        raise AuditFailure("PREEMPT_CPU")
    if git("rev-parse", "HEAD") != approved_head:
        raise AuditFailure("APPROVED_HEAD_DRIFT")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise AuditFailure("TRACKED_WORKTREE_NOT_CLEAN")
    if set(os.sched_getaffinity(0)) != ALLOWED_CPUS:
        raise AuditFailure(f"CPUSET:{sorted(os.sched_getaffinity(0))}")
    observed_env = {key: os.environ.get(key) for key in THREAD_ENV}
    if observed_env != THREAD_ENV:
        raise AuditFailure(f"THREAD_ENV:{observed_env}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AuditFailure("CUDA_VISIBLE_DEVICES")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise AuditFailure("JAX_PLATFORMS")
    if jax.default_backend() != "cpu" or any(
        device.platform != "cpu" for device in jax.devices()
    ):
        raise AuditFailure("JAX_BACKEND")
    for path, expected in EXPECTED.items():
        observed = sha256_file(path)
        if observed != expected:
            raise AuditFailure(f"HASH:{path}:{observed}")
    relative = Path(__file__).resolve().relative_to(REPO).as_posix()
    disk = Path(__file__).resolve().read_bytes()
    if disk != git("show", f"HEAD:{relative}", binary=True):
        raise AuditFailure("AUDIT_SCRIPT_NOT_HEAD")
    return {
        "approved_head": approved_head,
        "branch": git("branch", "--show-current"),
        "cpuset": sorted(ALLOWED_CPUS),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "thread_environment": observed_env,
        "cuda_visible_devices": "",
        "jax_platforms": "cpu",
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "gpu_actions": 0,
        "wrf_or_mpi_executions": 0,
        "production_adapter_invocations": 0,
        "direct_edmf_kernel_invocations": 6,
        "script": {
            "path": relative,
            "sha256": hashlib.sha256(disk).hexdigest(),
            "git_blob": git("rev-parse", f"HEAD:{relative}"),
        },
        "sealed_hashes": {str(path): digest for path, digest in EXPECTED.items()},
    }


def validate_authorities(comparator: Any) -> dict[str, Any]:
    manifest = load_json(CAPTURE_MANIFEST)
    capture_proof = load_json(CAPTURE_PROOF)
    channel = load_json(CHANNEL_GATE)
    if not (
        canonical(manifest) == manifest.get("canonical_payload_sha256")
        and manifest.get("passed") is True
        and manifest.get("single_adapter_invocation") is True
        and manifest.get("archive", {}).get("sha256") == EXPECTED[CAPTURE]
    ):
        raise AuditFailure("CAPTURE_MANIFEST_AUTHORITY")
    if not (
        canonical(capture_proof) == capture_proof.get("canonical_payload_sha256")
        and capture_proof.get("passed") is True
        and capture_proof.get("authority", {}).get("adapter_invocations") == 1
        and capture_proof.get("authority", {}).get("gpu_actions") == 0
        and capture_proof.get("authority", {}).get("wrf_or_mpi_executions") == 0
    ):
        raise AuditFailure("CAPTURE_PROOF_AUTHORITY")
    activity = channel.get("channel_decomposed_gate", {}).get(
        "activation_mask_primary", {}
    )
    if not (
        canonical(channel) == channel.get("canonical_payload_sha256")
        and channel.get("passed") is True
        and all(activity.get(field, {}).get("final_port_only_columns") == 67
                for field in FIELDS)
        and all(activity.get(field, {}).get("final_wrf_only_columns") == 0
                for field in FIELDS)
    ):
        raise AuditFailure("CHANNEL_GATE_AUTHORITY")

    tree = {
        path.relative_to(WRF_ROOT).as_posix(): sha256_file(path)
        for path in sorted(WRF_ROOT.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    if any(path.is_symlink() for path in WRF_ROOT.rglob("*")):
        raise AuditFailure("WRF_TREE_SYMLINK")
    tree_sha = hashlib.sha256(
        json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if len(tree) != 462 or tree_sha != EXPECTED_WRF_TREE:
        raise AuditFailure(f"WRF_TREE:{len(tree)}:{tree_sha}")

    reader, reader_record = comparator.import_dump_reader()
    return {
        "manifest": manifest,
        "capture_proof": capture_proof,
        "channel_gate": channel,
        "reader": reader,
        "reader_record": reader_record,
        "wrf_tree": {"file_count": len(tree), "sha256": tree_sha},
    }


def load_capture(manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = {
        *(f"surface_terms_state_{name}" for name in (
            "u", "v", "w", "theta", "qv", "tke", "p", "rho", "dz",
            "qc", "qi", "qs", "qsq",
        )),
        *(f"surface_terms_flux_{name}" for name in (
            "ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc",
            "fltv", "xland", "t_skin",
        )),
        "surface_terms_wind",
        "turbulence_pblh",
        *(f"mass_flux_{name}" for name in FIELDS),
    }
    declared = manifest["archive"]["arrays"]
    order = manifest["archive"]["array_order"]
    values: dict[str, np.ndarray] = {}
    with np.load(CAPTURE, allow_pickle=False) as archive:
        if archive.files != order or set(archive.files) != set(declared):
            raise AuditFailure("CAPTURE_INVENTORY")
        if not required.issubset(archive.files):
            raise AuditFailure(f"CAPTURE_REQUIRED:{sorted(required-set(archive.files))}")
        for name in required:
            value = np.asarray(archive[name])
            observed = {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "nbytes": value.nbytes,
                "logical_c_bitpayload_sha256": hashlib.sha256(
                    np.ascontiguousarray(value).tobytes()
                ).hexdigest(),
            }
            if observed != declared[name]:
                raise AuditFailure(f"CAPTURE_ARRAY:{name}:{observed}")
            if value.dtype.hasobject or not np.isfinite(value).all():
                raise AuditFailure(f"CAPTURE_FINITE:{name}")
            values[name] = np.ascontiguousarray(value)
    return values


def source_audit() -> dict[str, Any]:
    wrf = WRF_SOURCE.read_text(encoding="utf-8")
    port = PORT_EDMF.read_text(encoding="utf-8")
    pbl = PORT_PBL.read_text(encoding="utf-8")
    wrf_tokens = (
        "fltv=flt + flqv*p608*th_sfc",
        "if ( fltv2 > 0.002 .AND. (maxwidth > minwidth) .AND. superadiabatic) then",
        "IF (k==kts+1 .AND. Wn == zero) THEN",
        "NUP2=0",
        "IF (nup2 > 0) THEN",
    )
    port_tokens = (
        "def _wrf_superadiabatic_gate",
        "active = (fltv2 > 0.002) & (maxwidth > minwidth) & superad",
        "def _wrf_first_level_plume_survival",
        "return jnp.all(first_level_w > 0.0)",
        "active = active & first_level_survives",
    )
    pbl_tokens = (
        "ts = jnp.broadcast_to(ts, fltv.shape) / exner[..., 0]",
        "pblh=pblh, ts=ts",
    )
    return {
        "wrf": {
            "path": str(WRF_SOURCE),
            "sha256": EXPECTED[WRF_SOURCE],
            "tokens": {token: line_number(wrf, token) for token in wrf_tokens},
        },
        "port": {
            "edmf_path": str(PORT_EDMF.relative_to(REPO)),
            "edmf_sha256": EXPECTED[PORT_EDMF],
            "edmf_tokens": {token: line_number(port, token) for token in port_tokens},
            "pbl_path": str(PORT_PBL.relative_to(REPO)),
            "pbl_sha256": EXPECTED[PORT_PBL],
            "pbl_tokens": {token: line_number(pbl, token) for token in pbl_tokens},
        },
        "first_source_authorized_divergence": (
            "At WRF's first integrated plume level, any Wn==0 writes the shared "
            "NUP2=0 and the later NUP2 guard suppresses all s_aw* fluxes. The "
            "historical port instead retained the other surviving plumes."
        ),
    }


def rms(delta: np.ndarray) -> float:
    value = np.asarray(delta, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value)))


def delta_metrics(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(left, dtype=np.float64)[mask] - np.asarray(
        right, dtype=np.float64
    )[mask]
    if selected.size == 0:
        raise AuditFailure("EMPTY_METRIC_MASK")
    return {
        "count": int(selected.size),
        "rms": rms(selected),
        "max_abs": float(np.max(np.abs(selected))),
        "median_abs": float(np.median(np.abs(selected))),
    }


def replace_flux(base: SurfaceFluxes, **updates: object) -> SurfaceFluxes:
    values = base._asdict()
    values.update({key: jnp.asarray(value) for key, value in updates.items()})
    return SurfaceFluxes(**values)


def evaluate(
    capture: Mapping[str, np.ndarray], reader: Any, comparator: Any
) -> dict[str, Any]:
    ny, nx = capture["surface_terms_flux_ustar"].shape
    nz = capture["surface_terms_state_u"].shape[0]
    columns = ny * nx

    def profile(name: str) -> jax.Array:
        value = capture[name]
        return jnp.asarray(np.transpose(value, (1, 2, 0)).reshape(columns, nz))

    def plane(name: str) -> jax.Array:
        return jnp.asarray(capture[name].reshape(columns))

    zeros = jnp.zeros((columns, nz), dtype=jnp.float64)
    state = MynnPBLColumnState(
        u=profile("surface_terms_state_u"),
        v=profile("surface_terms_state_v"),
        w=profile("surface_terms_state_w"),
        theta=profile("surface_terms_state_theta"),
        qv=profile("surface_terms_state_qv"),
        tke=profile("surface_terms_state_tke"),
        p=profile("surface_terms_state_p"),
        rho=profile("surface_terms_state_rho"),
        dz=profile("surface_terms_state_dz"),
        km=zeros,
        kh=zeros,
        el=zeros,
        qc=profile("surface_terms_state_qc"),
        qi=profile("surface_terms_state_qi"),
        qs=profile("surface_terms_state_qs"),
        qsq=profile("surface_terms_state_qsq"),
    )
    port_flux = SurfaceFluxes(
        ustar=plane("surface_terms_flux_ustar"),
        theta_flux=plane("surface_terms_flux_theta_flux"),
        qv_flux=plane("surface_terms_flux_qv_flux"),
        tau_u=plane("surface_terms_flux_tau_u"),
        tau_v=plane("surface_terms_flux_tau_v"),
        rhosfc=plane("surface_terms_flux_rhosfc"),
        fltv=plane("surface_terms_flux_fltv"),
        xland=plane("surface_terms_flux_xland"),
        t_skin=plane("surface_terms_flux_t_skin"),
    )
    pblh = plane("turbulence_pblh")

    wrf_outer = {
        tag: comparator.load_outer2(reader, tag).astype(np.float64).reshape(columns)
        for tag in ("ust", "hfx", "qfx", "wspd", "tsk")
    }
    lower = reader.columns("bc_lower_operands").astype(np.float64)
    wrf_fields = {
        field: reader.columns(f"bc_{field}") for field in FIELDS
    }
    wrf_masks = {
        field: np.any(value != 0.0, axis=0).reshape(columns)
        for field, value in wrf_fields.items()
    }
    wrf_rhosfc = lower[10].reshape(columns)
    wrf_qv0 = lower[2].reshape(columns)
    exner0 = (np.asarray(state.p[:, 0]) / 100000.0) ** (287.0 / (3.5 * 287.0))
    wrf_theta_flux = wrf_outer["hfx"] / (
        wrf_rhosfc * (1004.5 * (1.0 + 0.84 * np.maximum(wrf_qv0, 1.0e-8)))
    )
    wrf_qv_flux = wrf_outer["qfx"] / wrf_rhosfc
    wrf_th_sfc = wrf_outer["tsk"] / exner0
    wrf_fltv = wrf_theta_flux + 0.608 * wrf_qv_flux * wrf_th_sfc

    variants = {
        "captured_port_surface": port_flux,
        "wrf_ust_only": replace_flux(port_flux, ustar=wrf_outer["ust"]),
        "wrf_heat_moisture_only": replace_flux(
            port_flux,
            theta_flux=wrf_theta_flux,
            qv_flux=wrf_qv_flux,
            fltv=wrf_fltv,
        ),
        "wrf_rhosfc_only": replace_flux(port_flux, rhosfc=wrf_rhosfc),
        "wrf_tsk_only": replace_flux(port_flux, t_skin=wrf_outer["tsk"]),
        "all_available_wrf_surface": replace_flux(
            port_flux,
            ustar=wrf_outer["ust"],
            theta_flux=wrf_theta_flux,
            qv_flux=wrf_qv_flux,
            rhosfc=wrf_rhosfc,
            fltv=wrf_fltv,
            t_skin=wrf_outer["tsk"],
        ),
    }

    @jax.jit
    def arm(
        column_state: MynnPBLColumnState,
        flux: SurfaceFluxes,
        height: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        result = _edmf_arrays_from_state(
            column_state, flux, flux.fltv, height, dt=6.0, dx=1000.0
        )
        return result["s_aw"], result["s_awu"], result["s_awv"]

    captured_fields = {
        field: np.transpose(capture[f"mass_flux_{field}"], (1, 2, 0)).reshape(
            columns, nz + 1
        )
        for field in FIELDS
    }
    variant_results: dict[str, Any] = {}
    variant_arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, flux in variants.items():
        outputs = arm(state, flux, pblh)
        arrays = {field: np.asarray(value) for field, value in zip(FIELDS, outputs)}
        variant_arrays[name] = arrays
        fields: dict[str, Any] = {}
        for field in FIELDS:
            mask = np.any(arrays[field] != 0.0, axis=1)
            wrf_mask = wrf_masks[field]
            fields[field] = {
                "active_columns": int(np.sum(mask)),
                "both_columns": int(np.sum(mask & wrf_mask)),
                "port_only_columns": int(np.sum(mask & ~wrf_mask)),
                "wrf_only_columns": int(np.sum(wrf_mask & ~mask)),
                "mask_disagreement_vs_capture": int(
                    np.sum(mask != np.any(captured_fields[field] != 0.0, axis=1))
                ),
                "max_abs_vs_capture": float(
                    np.max(np.abs(arrays[field] - captured_fields[field]))
                ),
            }
        variant_results[name] = fields

    reconstruction = variant_results["captured_port_surface"]
    if not all(
        reconstruction[field]["mask_disagreement_vs_capture"] == 0
        and reconstruction[field]["max_abs_vs_capture"] == 0.0
        for field in FIELDS
    ):
        raise AuditFailure(f"DIRECT_RECONSTRUCTION:{reconstruction}")
    final = variant_results["captured_port_surface"]
    substituted = variant_results["all_available_wrf_surface"]
    if not all(
        final[field]["port_only_columns"] == 67
        and final[field]["wrf_only_columns"] == 0
        and substituted[field]["port_only_columns"] == 66
        and substituted[field]["wrf_only_columns"] == 0
        for field in FIELDS
    ):
        raise AuditFailure(f"OBSERVATIONAL_BOUND:{final}:{substituted}")

    port_mask = np.any(captured_fields["s_aw"] != 0.0, axis=1)
    wrf_mask = wrf_masks["s_aw"]
    residual = port_mask & ~wrf_mask
    both = port_mask & wrf_mask
    port_rhosfc = np.asarray(port_flux.rhosfc)
    port_qv0 = np.asarray(state.qv[:, 0])
    port_hfx = (
        np.asarray(port_flux.theta_flux)
        * port_rhosfc
        * (1004.5 * (1.0 + 0.84 * np.maximum(port_qv0, 1.0e-8)))
    )
    port_qfx = np.asarray(port_flux.qv_flux) * port_rhosfc
    surface_deltas = {}
    for name, port_value, wrf_value in (
        ("ustar_m_s-1", np.asarray(port_flux.ustar), wrf_outer["ust"]),
        ("wind_m_s-1", capture["surface_terms_wind"].reshape(columns), wrf_outer["wspd"]),
        ("hfx_W_m-2", port_hfx, wrf_outer["hfx"]),
        ("qfx_kg_m-2_s-1", port_qfx, wrf_outer["qfx"]),
        ("rhosfc_kg_m-3", port_rhosfc, wrf_rhosfc),
        ("tsk_K", np.asarray(port_flux.t_skin), wrf_outer["tsk"]),
    ):
        surface_deltas[name] = {
            "on_67_port_only_columns": delta_metrics(port_value, wrf_value, residual),
            "on_3692_both_active_columns": delta_metrics(port_value, wrf_value, both),
        }

    amplitudes = {}
    for field in FIELDS:
        per_column = np.max(np.abs(captured_fields[field][residual]), axis=1)
        amplitudes[field] = {
            "columns": int(per_column.size),
            "minimum_column_max_abs": float(np.min(per_column)),
            "median_column_max_abs": float(np.median(per_column)),
            "maximum_column_max_abs": float(np.max(per_column)),
            "rms_all_values": rms(captured_fields[field][residual]),
        }

    # Reconstruct the pre-plume predicates with the same source expressions.
    theta = np.asarray(state.theta)
    qv = np.asarray(state.qv)
    w = np.asarray(state.w)
    dz = np.asarray(state.dz)
    skin_theta = np.asarray(port_flux.t_skin) / exner0
    xland = np.asarray(port_flux.xland)
    zw = np.concatenate(
        (np.zeros((columns, 1)), np.cumsum(dz, axis=1)), axis=1
    )
    mass_height = zw[:, :-1] + 0.5 * dz
    wpbl = np.where(w < 0.0, 2.0 * w, w)
    maxw = np.max(
        np.where(mass_height <= np.asarray(pblh)[:, None] + 500.0,
                 np.abs(wpbl), 0.0),
        axis=1,
    )
    psig_w = np.minimum(1.0, np.maximum(0.0, 1.0 - np.maximum(0.0, maxw - 1.0)))
    fltv = np.asarray(port_flux.fltv)
    fltv2 = np.where((psig_w == 0.0) & (fltv > 0.0), -fltv, fltv)
    thv0 = theta[:, 0] * (1.0 + 0.608 * qv[:, 0])
    tvs = skin_theta * (1.0 + 0.608 * qv[:, 0])
    hux = np.where(xland >= 1.5, -0.001, -0.003)
    superadiabatic = (thv0 - tvs) / (0.5 * dz[:, 0]) < hux
    maxwidth_dx = min(1000.0 * 1.2, 1000.0)
    maxwidth_pbl = np.minimum(1.1 * np.asarray(pblh), 1000.0)
    maxwidth_cloud = np.minimum(
        1000.0, np.maximum(np.where(xland >= 1.5, 0.9, 0.5) * 9000.0, 400.0)
    )
    maxwidth_flux = np.maximum(
        np.minimum(
            np.where(
                xland >= 1.5,
                1000.0 * (0.6 * np.tanh((fltv - 0.007) / 0.02) + 0.5),
                1000.0 * (0.6 * np.tanh((fltv - 0.040) / 0.04) + 0.5),
            ),
            1000.0,
        ),
        0.0,
    )
    maxwidth = np.minimum(
        np.minimum(maxwidth_dx, maxwidth_pbl),
        np.minimum(maxwidth_cloud, maxwidth_flux),
    )
    base_active = (fltv2 > 0.002) & (maxwidth > 300.0) & superadiabatic
    predicate_counts = {
        "domain_columns": columns,
        "psig_w_positive": int(np.sum(psig_w > 0.0)),
        "fltv2_gt_0p002": int(np.sum(fltv2 > 0.002)),
        "superadiabatic_surface": int(np.sum(superadiabatic)),
        "maxwidth_gt_minwidth": int(np.sum(maxwidth > 300.0)),
        "pre_plume_all": int(np.sum(base_active)),
        "rejected_by_first_level_plume_survival": int(np.sum(base_active & ~port_mask)),
        "post_plume_final": int(np.sum(port_mask)),
        "post_plume_not_pre_plume": int(np.sum(port_mask & ~base_active)),
    }
    if predicate_counts != {
        "domain_columns": 10323,
        "psig_w_positive": 10204,
        "fltv2_gt_0p002": 8310,
        "superadiabatic_surface": 8287,
        "maxwidth_gt_minwidth": 8220,
        "pre_plume_all": 8213,
        "rejected_by_first_level_plume_survival": 4454,
        "post_plume_final": 3759,
        "post_plume_not_pre_plume": 0,
    }:
        raise AuditFailure(f"PREDICATE_COUNTS:{predicate_counts}")

    return {
        "direct_kernel_reconstruction": {
            "bit_identical_for_all_s_aw_fields": True,
            "fields": reconstruction,
        },
        "ordered_predicate_counts": predicate_counts,
        "surface_substitution": {
            "method": (
                "Retain the captured atmospheric profiles, pblh, and exner; "
                "replace every available authenticated WRF DMP surface operand: "
                "ust, HFX-derived flt, QFX-derived flqv, WRF fltv expression, "
                "rhosfc, and TSK. wspd is audited but is not a DMP_mf input."
            ),
            "variants": variant_results,
            "available_surface_deltas": surface_deltas,
            "port_only_columns_removed_by_all_available_wrf_surface": 1,
            "remaining_port_only_columns": 66,
            "wrf_only_columns_introduced": 0,
        },
        "frozen_67_column_bound": {
            "captured_port_only_columns": 67,
            "captured_wrf_only_columns": 0,
            "amplitude_is_not_a_roundoff_floor": True,
            "amplitudes": amplitudes,
            "interpretation": (
                "The direct production-input reconstruction is bit exact. All "
                "retained authenticated WRF surface operands explain only one "
                "of 67 columns. The dump does not retain WRF DMP pblh, full "
                "thermodynamic/plume state, cloud-base index, or per-plume Wn, "
                "so no later first divergence is source-authorized from this "
                "authority. The 67-column capture result is frozen; no threshold "
                "or mask tuning is permitted."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        authority = authority_gate(args.approved_head)
        comparator = import_comparator()
        upstream = validate_authorities(comparator)
        capture = load_capture(upstream["manifest"])
        source = source_audit()
        evaluation = evaluate(capture, upstream["reader"], comparator)
        proof = {
            "schema": "wrfgpu2-v0234-gpt-mynn-mf-predicate-audit-v1",
            "verdict": "FIRST_DIVERGENCE_FIXED_RESIDUAL_FROZEN_AT_67",
            "generated_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "passed": True,
            "acceptance_semantics": (
                "Pass means the source-authorized 4,526-column seam is fixed, "
                "the direct capture reconstruction is exact, no WRF-only mask "
                "regression exists, and the remaining 67 columns are sealed as "
                "an explicit observational bound. It does not claim exact mask "
                "parity or that the residual is numerical noise."
            ),
            "execution": authority,
            "input_evidence": {
                "capture_archive": {
                    "path": str(CAPTURE),
                    "sha256": EXPECTED[CAPTURE],
                },
                "capture_manifest_sha256": EXPECTED[CAPTURE_MANIFEST],
                "capture_proof_sha256": EXPECTED[CAPTURE_PROOF],
                "channel_gate_sha256": EXPECTED[CHANNEL_GATE],
                "wrf_dump_tree": upstream["wrf_tree"],
                "reader": upstream["reader_record"],
            },
            "source_localization": source,
            **evaluation,
        }
        atomic_json(output, proof)
        print(json.dumps({
            "output": str(output),
            "verdict": proof["verdict"],
            "canonical_payload_sha256": canonical(proof),
        }, sort_keys=True))
        return 0
    except (AuditFailure, OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL:{error}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
