#!/usr/bin/env python3
"""V0234 final-holistic late-window corner-drain discriminator.

Pre-declared bounded experiment for the recurring late (step ~9314) d03
blow-up.  The original CPU arms are retained byte-for-byte in scientific
meaning.  The post-Fable continuation additionally admits exactly one Arm-S
CUDA route; its only experimental change is backend/resource selection.

Evidence base (authenticated in the sprint amendment before execution):

* The retained ordinary-bisection first-bad transition 9313->9314 detonates at
  the four ring-1 corner cells (mass drain mu' ~ -11k..-12.3k Pa at 9313,
  overflow to ~1e181..1e204 / scratch NaN L1-radius-9 diamonds at 9314).
* All three 15:00 frames (Stage-Omega, Retry20, CPU-WRF) have clean corners,
  so the drain fires entirely inside the untested late window.

Arms (each arm is one separate process invocation):

* ``--arm S``  (primary): released Stage-Omega tree, input = the Stage-Omega
  replay's own authenticated d03 step-9000 carry, sequential single-step
  dispatches for steps 9001..9300 with per-step corner/health monitoring.
* ``--arm O``  (sensitivity control): old bisection-era tree (detached
  worktree at commit effe1f43), input = authenticated step-9313 carry,
  up to 3 dispatches; must reproduce the retained corner detonation on CPU.
* ``--arm X``  (cross-check): released Stage-Omega tree, input = the same
  step-9313 carry, up to 3 dispatches.

Pre-declared outcome semantics (see amendment JSON): Arm O red validates CPU
sensitivity; Arm S corner-drain growth or nonfinite = mechanism survives the
released tree (localization RED); Arm S bounded corners through 9300 =
dispatch-lane late window green (localization narrowed; live-parent forcing
remains for the full-tree gate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


LINEAGE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a"
)
STAGE_OMEGA_9000 = (
    LINEAGE
    / "nested_stage_omega_transport_470e6111_full18h_toolingrepair2/checkpoints/"
    "authenticated-d03-step-9000.pkl"
)
STAGE_OMEGA_9000_SHA256 = (
    "1b1521090539cc38a5660712ff1fe57a4251c3b790f7d619a176a8f32037d055"
)
BISECTION_9313 = (
    LINEAGE / "ordinary_bisection_effe1f43/checkpoints/"
    "last-finite-input-to-first-bad-d03-step-9313.pkl"
)
BISECTION_9313_SHA256 = (
    "f82a25c35e6bd738cd248d759c291d825ca0cfc4e08753dc5082741a78b23fc7"
)
# Filled from the pre-run authentication step and enforced here so a swapped
# file cannot silently change the experiment.
EXPECTED_INPUT_SHA256_ENV = "GPUWRF_FHX2_EXPECTED_INPUT_SHA256"

RELEASED_SRC_TREE = "835dcc29bf316c0715b41a72e064985e9cf099df"
OLD_CANDIDATE_COMMIT = "effe1f43bec389be6fb30b5ff61e41fcad64f7bf"
FABLE_TERMINAL_COMMIT = "84b311f97306e2ffca90fb9353daf7fa22ca9cc4"
APPROVED_SHA_ENV = "GPUWRF_POST_FABLE_APPROVED_SHA"
CONTRACT_SHA_ENV = "GPUWRF_POST_FABLE_CONTRACT_SHA256"
CONTRACT_PATH = Path(
    ".agent/sprints/2026-07-17-v0234-post-fable-corner-window/CONTRACT.md"
)
CONTRACT_SHA256 = "3906ea43edfd3f4f0ff53a56fa926d2b740939dd612e0c03a77796f7b9fbe582"

WRF_ROOT = (
    LINEAGE.parent
    / "gpu_validation_retry20_relative_rmse_3ee02c19/authority/wrf_root"
)

LOCK_ROOT = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-gpu-lock-v2")
LOCK_COMMIT = "8152309aff1e85e1052d44d549a5a5409e710bdd"
LOCK_WRAPPER_SHA256 = (
    "c75b3a4eda17e94df921986e1c15fcc71e182e01d51b6fe877e4c52533077e1a"
)
LOCK_VERIFIER_SHA256 = (
    "ec911f565ee9d2c58dee81d8d1e17411ed677be500a57e73ded5f793e2c4187d"
)

CPU_PREFIX_ROOT = LINEAGE / "final_holistic_fable5_xhigh2_8d99421f"
CPU_PREFIX_LOGS = {
    "first42": {
        "path": CPU_PREFIX_ROOT / "armS_stalled_superseded/armS_first42.log",
        "sha256": "208a424d94b15ee49f3266574f5998c046e81ecf3e692c42f0a04329a05f1d11",
        "last_step": 9042,
    },
    "first75": {
        "path": CPU_PREFIX_ROOT / "armS_stall2_superseded/armS_at9075.log",
        "sha256": "2a3a4308a8d41469beaf0466f6d43e9cf1b3d59b60aea1913ba0bb6e85360db0",
        "last_step": 9075,
    },
    "first16": {
        "path": CPU_PREFIX_ROOT / "armS_stall3_superseded/armS_at9016.log",
        "sha256": "0e3c01e250f4e8849ce7dc2dbf97fce903615cc279f1c196d8ebae0ca1553f99",
        "last_step": 9016,
    },
}
CPU_PREFIX_REFERENCES = {
    9016: {
        "corner_mu_maxabs": 1406.6845861449613,
        "mu_min": 1087.8628786805145,
        "w_maxabs": 5.833913024292236,
    },
    9042: {
        "corner_mu_maxabs": 1405.5705885280174,
        "mu_min": 1086.2702055597574,
        "w_maxabs": 5.832544572503565,
    },
    9075: {
        "corner_mu_maxabs": 1404.0304771861643,
        "mu_min": 1083.8869481374068,
        "w_maxabs": 5.830854439135598,
    },
}
# Declared before the GPU result.  These are deliberately wider than the
# variation across the retained CPU prefixes while remaining tight enough to
# reject a different structure/trajectory before the late window is read.
CPU_PREFIX_MAX_ABS_DELTA = {
    "corner_mu_maxabs": 5.0,
    "mu_min": 5.0,
    "w_maxabs": 0.05,
}

# Physically-motivated experiment stop bound: |mu'| at any monitored corner
# beyond this is >30x the largest clean-lineage magnitude (~1.5e3) and far
# below fp64 overflow; crossing it is recorded as DRAIN_RED and stops the arm.
CORNER_MU_STOP = 5.0e4

RING1_CORNERS = ((1, 1), (1, 109), (91, 1), (91, 109))

CPU_CORE_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
}
CUDA_CORE_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "JAX_PLATFORMS": "cuda",
    "JAX_ENABLE_X64": "true",
    "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false",
    "JAX_ENABLE_COMPILATION_CACHE": "false",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "GPUWRF_ALLOCATOR": "cuda_async",
    "GPUWRF_ADVANCE_CHUNK_LOOP": "fori",
    "GPUWRF_FINITE_CHECK": "1",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
}
ARM_ENV = {
    "S": {"GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1"},
    "S2": {"GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1"},
    "X": {"GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1"},
    "O": {},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(path: Path) -> tuple[str, dict[str, Any]]:
    before = path.stat()
    digest = _sha256(path)
    after = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest, {
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
        "mode": oct(after.st_mode & 0o777),
    }


def _parse_prefix_rows(path: Path) -> dict[int, dict[str, float | int]]:
    rows: dict[int, dict[str, float | int]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ARM_S step="):
            continue
        fields = dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
        step = int(fields["step"])
        rows[step] = {
            "nonfinite": int(fields["nonfinite"]),
            "corner_mu_maxabs": float(fields["corner_mu_maxabs"]),
            "mu_min": float(fields["mu_min"]),
            "w_maxabs": float(fields["w_maxabs"]),
        }
    return rows


def _authenticate_cpu_prefixes() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    parsed: dict[str, dict[int, dict[str, float | int]]] = {}
    for name, declared in CPU_PREFIX_LOGS.items():
        path = Path(declared["path"])
        digest, identity = _stable_sha256(path)
        if digest != declared["sha256"]:
            raise RuntimeError(f"retained CPU prefix hash changed: {path}")
        rows = _parse_prefix_rows(path)
        if not rows or max(rows) != declared["last_step"]:
            raise RuntimeError(f"retained CPU prefix endpoint changed: {path}")
        if any(row["nonfinite"] != 0 for row in rows.values()):
            raise RuntimeError(f"retained CPU green prefix contains a nonfinite row: {path}")
        parsed[name] = rows
        evidence[name] = {**identity, "green_rows": len(rows), "last_step": max(rows)}

    expected_sources = {9016: ("first16", "first42", "first75"),
                        9042: ("first42", "first75"),
                        9075: ("first75",)}
    for step, sources in expected_sources.items():
        expected = CPU_PREFIX_REFERENCES[step]
        for source in sources:
            actual = parsed[source].get(step)
            if actual is None or actual["nonfinite"] != 0:
                raise RuntimeError(f"CPU prefix checkpoint {step} missing in {source}")
            for metric, value in expected.items():
                if actual[metric] != value:
                    raise RuntimeError(
                        f"CPU prefix checkpoint {step} changed for {metric}: "
                        f"{actual[metric]} != {value}"
                    )
    return {
        "logs": evidence,
        "references": CPU_PREFIX_REFERENCES,
        "max_abs_delta": CPU_PREFIX_MAX_ABS_DELTA,
    }


def _compare_cpu_prefix(step: int, stats: dict[str, Any]) -> dict[str, Any] | None:
    reference = CPU_PREFIX_REFERENCES.get(step)
    if reference is None:
        return None
    deltas: dict[str, float | None] = {}
    violations: list[str] = []
    if stats.get("nonfinite_total") != 0 or stats.get("corner_nonfinite") != 0:
        violations.append("nonfinite_structure")
    for metric, expected in reference.items():
        actual = stats.get(metric)
        delta = None if actual is None else abs(float(actual) - expected)
        deltas[metric] = delta
        if delta is None or delta > CPU_PREFIX_MAX_ABS_DELTA[metric]:
            violations.append(metric)
    return {
        "step": step,
        "reference": reference,
        "max_abs_delta": CPU_PREFIX_MAX_ABS_DELTA,
        "observed_abs_delta": deltas,
        "passed": not violations,
        "violations": violations,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("S", "S2", "O", "X"), required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None,
                        help="dispatch count; defaults: S=300, S2=105, O=3, X=3")
    parser.add_argument("--input-path", type=Path, default=None,
                        help="S2 only: the arm-S final carry (final-d03-step-9300.pkl)")
    args = parser.parse_args()
    arm = args.arm
    backend = args.backend
    run_dir = args.run_dir.resolve()
    if backend == "cuda":
        if arm != "S":
            raise RuntimeError("the post-Fable CUDA route admits Arm S only")
        if run_dir.exists():
            raise RuntimeError(f"CUDA Arm-S run directory must not exist: {run_dir}")
        run_dir.mkdir(parents=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    defaults = {"S": 300, "S2": 105, "O": 3, "X": 3}
    steps = args.steps if args.steps is not None else defaults[arm]
    if backend == "cuda" and steps != 405:
        raise RuntimeError("the admitted CUDA Arm-S window is exactly 405 steps (9001..9405)")

    core_env = CUDA_CORE_ENV if backend == "cuda" else CPU_CORE_ENV
    for name, want in {**core_env, **ARM_ENV[arm]}.items():
        actual = os.environ.get(name)
        if actual != want:
            raise RuntimeError(f"env {name}={actual!r} expected {want!r}")
    if arm == "O" and os.environ.get("GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE"):
        raise RuntimeError("arm O must run without the frozen-bundle flag")
    wrf_root = os.environ.get("GPUWRF_WRF_ROOT")
    if not wrf_root or Path(wrf_root).resolve() != WRF_ROOT.resolve():
        raise RuntimeError(f"GPUWRF_WRF_ROOT must equal {WRF_ROOT}")
    expected_input_sha = os.environ.get(EXPECTED_INPUT_SHA256_ENV)
    if not expected_input_sha or len(expected_input_sha) != 64:
        raise RuntimeError(f"{EXPECTED_INPUT_SHA256_ENV} must carry the declared input hash")

    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    head_src_tree = subprocess.run(
        ("git", "rev-parse", "HEAD:src/gpuwrf"), cwd=repo_root,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty_model = subprocess.run(
        ("git", "diff", "--name-only", "--", "src/gpuwrf"), cwd=repo_root,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty_model:
        raise RuntimeError(f"model bytes dirty in {repo_root}: {dirty_model}")
    if arm in ("S", "S2", "X") and head_src_tree != RELEASED_SRC_TREE:
        raise RuntimeError(f"arm {arm} must run on the released tree, got {head_src_tree}")
    head_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repo_root,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    contract_authority = None
    if backend == "cuda":
        approved = os.environ.get(APPROVED_SHA_ENV)
        if approved != head_commit:
            raise RuntimeError(f"{APPROVED_SHA_ENV} must equal committed HEAD")
        dirty_any = subprocess.run(
            ("git", "status", "--porcelain"), cwd=repo_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if dirty_any:
            raise RuntimeError(f"CUDA Arm S requires a clean committed worktree: {dirty_any}")
        contract_digest, contract_authority = _stable_sha256(repo_root / CONTRACT_PATH)
        if (
            contract_digest != CONTRACT_SHA256
            or os.environ.get(CONTRACT_SHA_ENV) != CONTRACT_SHA256
        ):
            raise RuntimeError("post-Fable contract authority changed")
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", FABLE_TERMINAL_COMMIT, head_commit),
            cwd=repo_root, check=True,
        )
        affinity = sorted(os.sched_getaffinity(0))
        if affinity != [12, 13, 14, 15]:
            raise RuntimeError(f"CUDA Arm S requires CPU affinity 12-15, got {affinity}")
    if arm == "O" and head_commit != OLD_CANDIDATE_COMMIT:
        raise RuntimeError(f"arm O must run at {OLD_CANDIDATE_COMMIT}, got {head_commit}")

    if arm == "S":
        input_path, start_step = STAGE_OMEGA_9000, 9001
    elif arm == "S2":
        if args.input_path is None:
            raise RuntimeError("arm S2 requires --input-path")
        input_path = args.input_path.resolve()
        if input_path.name != "final-d03-step-9300.pkl":
            raise RuntimeError(f"arm S2 input must be the arm-S 9300 final carry: {input_path}")
        start_step = 9301
    else:
        input_path, start_step = BISECTION_9313, 9314
    input_sha, input_authority = _stable_sha256(input_path)
    if input_sha != expected_input_sha:
        raise RuntimeError(f"input hash mismatch: {input_sha} != {expected_input_sha}")
    if arm == "S" and input_sha != STAGE_OMEGA_9000_SHA256:
        raise RuntimeError("arm S input must be the authenticated Stage-Omega step-9000 carry")
    if arm in ("O", "X") and input_sha != BISECTION_9313_SHA256:
        raise RuntimeError("arm O/X input must be the authenticated 9313 carry")

    cpu_prefix_authority = _authenticate_cpu_prefixes() if backend == "cuda" else None

    import jax

    jax.config.update("jax_cpu_enable_async_dispatch", False)
    import numpy as np

    devices = []
    if backend == "cpu":
        devices = jax.devices()
        if jax.default_backend() != "cpu":
            raise RuntimeError(f"backend {jax.default_backend()} is not cpu")

    import gpuwrf
    imported_root = Path(gpuwrf.__file__).resolve()
    if repo_root / "src" not in imported_root.parents:
        raise RuntimeError(f"gpuwrf imported from {imported_root}, not {repo_root}")

    import gpuwrf.contracts.state as state_contract
    if backend == "cpu":
        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
    import gpuwrf.runtime.operational_mode as runtime
    sys.path.insert(0, str(repo_root))
    from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

    lock_authority = None
    if backend == "cuda":
        if ordinary.LOCK_ROOT != LOCK_ROOT:
            raise RuntimeError("ordinary lock authority root changed")
        lock_authority = ordinary.assert_lock_authority()
        if (
            lock_authority["commit"] != LOCK_COMMIT
            or lock_authority["wrapper_sha256"] != LOCK_WRAPPER_SHA256
            or lock_authority["verifier_sha256"] != LOCK_VERIFIER_SHA256
            or lock_authority["intent"] != "production-preemptible"
        ):
            raise RuntimeError("canonical lock-v2 authority mismatch")
        preempt_present = [
            str(path) for path in ordinary.PREEMPT_PATHS
            if path.exists() or path.is_symlink()
        ]
        if preempt_present:
            raise RuntimeError(f"production preemption sentinel present: {preempt_present}")
        # This is the first CUDA availability query.  It occurs only after the
        # live lock-v2 lease, intent, FD, cgroup, source commit, and verifier
        # bytes have all authenticated above.
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise RuntimeError(f"CUDA Arm S requires exactly one visible GPU: {devices!r}")
        if jax.default_backend() != "gpu":
            raise RuntimeError(f"backend {jax.default_backend()} is not gpu")

    loader_scratch = Path(tempfile.mkdtemp(prefix=f"v0234-fhx2-load-{arm}-"))
    tree, names, _initial, dt_by_domain, load_authority = ordinary.load_corrected_tree(
        loader_scratch
    )
    if names != ("d01", "d02", "d03") or dt_by_domain["d03"] != 6.0:
        raise RuntimeError("canonical hierarchy changed")
    namelist = tree.domains["d03"].namelist
    clock = runtime.build_clock_base(namelist)
    cadence = int(namelist.radiation_cadence_steps)
    if cadence != 300:
        raise RuntimeError(f"radiation cadence changed: {cadence}")

    with input_path.open("rb") as stream:
        carry = pickle.load(stream)

    leaves = jax.tree_util.tree_leaves(carry)
    if len(leaves) != 106:
        raise RuntimeError(f"carry leaf count {len(leaves)} != 106")

    def corner_stats(c) -> dict[str, Any]:
        mu = np.asarray(c.state.mu_perturbation)
        muts = np.asarray(c.muts)
        w = np.asarray(c.state.w)
        u = np.asarray(c.state.u)
        stats: dict[str, Any] = {}
        nonfinite = 0
        for leaf in jax.tree_util.tree_leaves(c):
            arr = np.asarray(leaf)
            if np.issubdtype(arr.dtype, np.floating):
                nonfinite += int((~np.isfinite(arr)).sum())
        stats["nonfinite_total"] = nonfinite
        stats["corner_mu"] = {
            f"y{y}x{x}": float(mu[y, x]) if np.isfinite(mu[y, x]) else None
            for (y, x) in RING1_CORNERS
        }
        finite_mu = mu[np.isfinite(mu)]
        stats["mu_min"] = float(finite_mu.min()) if finite_mu.size else None
        stats["mu_max"] = float(finite_mu.max()) if finite_mu.size else None
        finite_muts = muts[np.isfinite(muts)]
        stats["muts_min"] = float(finite_muts.min()) if finite_muts.size else None
        fw = np.abs(w[np.isfinite(w)])
        stats["w_maxabs"] = float(fw.max()) if fw.size else None
        fu = np.abs(u[np.isfinite(u)])
        stats["u_maxabs"] = float(fu.max()) if fu.size else None
        corner_vals = [v for v in stats["corner_mu"].values() if v is not None]
        stats["corner_mu_maxabs"] = max(abs(v) for v in corner_vals) if corner_vals else None
        stats["corner_nonfinite"] = sum(1 for v in stats["corner_mu"].values() if v is None)
        return stats

    seed_stats = corner_stats(carry)
    if arm == "S" and seed_stats["nonfinite_total"] != 0:
        raise RuntimeError("arm S input carry is not finite")

    if backend == "cuda":
        carry = jax.device_put(carry, devices[0])

    started = time.perf_counter()
    lowered = runtime._advance_chunk_fori.lower(
        carry, namelist, start_step, clock, n_steps=1, cadence=cadence
    )
    lower_seconds = time.perf_counter() - started
    input_tree = jax.tree_util.tree_structure(carry)
    output_tree = jax.tree_util.tree_structure(lowered.out_info)
    if input_tree != output_tree:
        raise RuntimeError("carry interface identity failed")
    started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started

    metric_compile_seconds = None
    metric_executable = None
    if backend == "cuda":
        import jax.numpy as jnp

        def device_corner_stats(c):
            floating_leaves = [
                leaf for leaf in jax.tree_util.tree_leaves(c)
                if jnp.issubdtype(leaf.dtype, jnp.floating)
            ]
            nonfinite_total = sum(
                (jnp.count_nonzero(~jnp.isfinite(leaf)) for leaf in floating_leaves),
                start=jnp.asarray(0, dtype=jnp.int64),
            )
            mu = c.state.mu_perturbation
            muts = c.muts
            w = c.state.w
            u = c.state.u
            corners = jnp.stack([mu[y, x] for y, x in RING1_CORNERS])

            def finite_min(value):
                finite = jnp.isfinite(value)
                return jnp.min(jnp.where(finite, value, jnp.inf)), jnp.count_nonzero(finite)

            def finite_maxabs(value):
                finite = jnp.isfinite(value)
                return (
                    jnp.max(jnp.where(finite, jnp.abs(value), -jnp.inf)),
                    jnp.count_nonzero(finite),
                )

            mu_min, mu_finite = finite_min(mu)
            mu_max = jnp.max(jnp.where(jnp.isfinite(mu), mu, -jnp.inf))
            muts_min, muts_finite = finite_min(muts)
            w_maxabs, w_finite = finite_maxabs(w)
            u_maxabs, u_finite = finite_maxabs(u)
            corner_finite = jnp.isfinite(corners)
            corner_mu_maxabs = jnp.max(
                jnp.where(corner_finite, jnp.abs(corners), -jnp.inf)
            )
            return {
                "nonfinite_total": nonfinite_total,
                "corners": corners,
                "corner_nonfinite": jnp.count_nonzero(~corner_finite),
                "corner_mu_maxabs": corner_mu_maxabs,
                "mu_min": mu_min,
                "mu_max": mu_max,
                "mu_finite": mu_finite,
                "muts_min": muts_min,
                "muts_finite": muts_finite,
                "w_maxabs": w_maxabs,
                "w_finite": w_finite,
                "u_maxabs": u_maxabs,
                "u_finite": u_finite,
            }

        def host_scalar_stats(c) -> dict[str, Any]:
            raw = jax.device_get(metric_executable(c))
            corner_values = np.asarray(raw["corners"])

            def optional_float(name: str, count_name: str) -> float | None:
                return float(raw[name]) if int(raw[count_name]) else None

            return {
                "nonfinite_total": int(raw["nonfinite_total"]),
                "corner_mu": {
                    f"y{y}x{x}": (
                        float(corner_values[index])
                        if np.isfinite(corner_values[index]) else None
                    )
                    for index, (y, x) in enumerate(RING1_CORNERS)
                },
                "mu_min": optional_float("mu_min", "mu_finite"),
                "mu_max": optional_float("mu_max", "mu_finite"),
                "muts_min": optional_float("muts_min", "muts_finite"),
                "w_maxabs": optional_float("w_maxabs", "w_finite"),
                "u_maxabs": optional_float("u_maxabs", "u_finite"),
                "corner_mu_maxabs": (
                    float(raw["corner_mu_maxabs"])
                    if int(raw["corner_nonfinite"]) < len(RING1_CORNERS) else None
                ),
                "corner_nonfinite": int(raw["corner_nonfinite"]),
            }

        started = time.perf_counter()
        metric_executable = jax.jit(device_corner_stats).lower(carry).compile()
        metric_compile_seconds = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    verdict = "WINDOW_COMPLETED_BOUNDED"
    first_red_step = None
    first_red_reason = None
    terminal_host_carry = None
    for index in range(steps):
        step = start_step + index
        if backend == "cuda":
            preempt_present = [
                str(path) for path in ordinary.PREEMPT_PATHS
                if path.exists() or path.is_symlink()
            ]
            if preempt_present:
                _atomic_json(
                    run_dir / "preempt-observed.json",
                    {"before_step": step, "paths": preempt_present, "rows": rows},
                )
                raise RuntimeError(
                    f"production preemption observed before step {step}: {preempt_present}"
                )
        t0 = time.perf_counter()
        carry = executable(carry, namelist, step, clock, n_steps=1, cadence=cadence)
        if backend == "cuda":
            stats = host_scalar_stats(carry)
        else:
            carry = jax.device_get(carry)
            stats = corner_stats(carry)
        dispatch_seconds = time.perf_counter() - t0
        prefix_comparison = _compare_cpu_prefix(step, stats) if backend == "cuda" else None
        row = {
            "step": step,
            "dispatch_seconds": dispatch_seconds,
            **stats,
            "cpu_prefix_comparison": prefix_comparison,
        }
        rows.append(row)
        print(
            f"ARM_{arm} step={step} nonfinite={stats['nonfinite_total']} "
            f"corner_mu_maxabs={stats['corner_mu_maxabs']} "
            f"mu_min={stats['mu_min']} w_maxabs={stats['w_maxabs']}",
            flush=True,
        )
        red = (
            stats["nonfinite_total"] > 0
            or stats["corner_nonfinite"] > 0
            or (
                stats["corner_mu_maxabs"] is not None
                and stats["corner_mu_maxabs"] > CORNER_MU_STOP
            )
            or (prefix_comparison is not None and not prefix_comparison["passed"])
        )
        if red:
            if prefix_comparison is not None and not prefix_comparison["passed"]:
                verdict = "CPU_PREFIX_MISMATCH"
                first_red_reason = "retained_cpu_prefix_envelope"
            else:
                verdict = "DRAIN_RED"
                first_red_reason = (
                    "nonfinite" if stats["nonfinite_total"] > 0
                    else "corner_mu_stop"
                )
            first_red_step = step
            terminal_host_carry = jax.device_get(carry)
            with (run_dir / f"first-red-d03-step-{step}.pkl").open("xb") as stream:
                pickle.dump(terminal_host_carry, stream, protocol=5)
            break
        if index % 25 == 24:
            _atomic_json(run_dir / "progress.json", {"rows": rows})

    if backend == "cuda" and verdict == "WINDOW_COMPLETED_BOUNDED":
        checkpoint_rows = {
            row["step"]: row["cpu_prefix_comparison"]
            for row in rows if row["cpu_prefix_comparison"] is not None
        }
        if set(checkpoint_rows) != set(CPU_PREFIX_REFERENCES) or not all(
            item["passed"] for item in checkpoint_rows.values()
        ):
            raise RuntimeError("CUDA Arm S completed without all three CPU-prefix gates")
    if terminal_host_carry is None:
        terminal_host_carry = jax.device_get(carry)

    final_path = run_dir / f"final-d03-step-{rows[-1]['step'] if rows else start_step}.pkl"
    if not final_path.exists():
        with final_path.open("xb") as stream:
            pickle.dump(terminal_host_carry, stream, protocol=5)

    final_input_sha, final_input_authority = _stable_sha256(input_path)
    if final_input_sha != input_sha or final_input_authority != input_authority:
        raise RuntimeError("authenticated input identity changed during Arm S")

    payload: dict[str, Any] = {
        "schema": "gpuwrf.v0234.final-holistic-late-window-corner-discriminator.v1",
        "arm": arm,
        "verdict": verdict,
        "first_red_step": first_red_step,
        "first_red_reason": first_red_reason,
        "repo_root": str(repo_root),
        "head_commit": head_commit,
        "head_src_tree": head_src_tree,
        "contract_authority": contract_authority,
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "input_authority": input_authority,
        "start_step": start_step,
        "requested_steps": steps,
        "completed_steps": len(rows),
        "seed_stats": seed_stats,
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "metric_compile_seconds": metric_compile_seconds,
        "cadence": cadence,
        "requested_backend": backend,
        "backend": jax.default_backend(),
        "device_authority": {
            "jax_version": jax.__version__,
            "jaxlib_version": getattr(jax.lib, "__version__", "unknown"),
            "devices": [str(device) for device in devices],
            "authenticated_before_lower": True,
        },
        "lock_v2": lock_authority,
        "cpu_prefix_authority": cpu_prefix_authority,
        "hot_loop_full_carry_transfers": 0 if backend == "cuda" else len(rows),
        "terminal_full_carry_transfers": 1 if backend == "cuda" else 0,
        "model_bytes_changed": False,
        "environment": {name: os.environ.get(name) for name in
                        sorted({**core_env, **ARM_ENV[arm], "GPUWRF_WRF_ROOT": "",
                                APPROVED_SHA_ENV: "", CONTRACT_SHA_ENV: ""})},
        "load_authority": load_authority,
        "rows": rows,
        "final_carry": {"path": str(final_path), "sha256": _sha256(final_path)},
    }
    payload["proof_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "proof_sha256"},
                   sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    _atomic_json(run_dir / f"arm-{arm}-proof.json", payload)
    print(f"ARM_{arm}_DONE verdict={verdict} first_red_step={first_red_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
